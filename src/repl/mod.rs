//! The interactive REPL loop.
//!
//! `main.rs` keeps only process wiring (config, key, sandbox); the loop
//! itself lives here, seamed over injected input/output — and, for the
//! `/model <provider>:<model>` switch, an injected key resolver and provider
//! builder, plus an injected persistence sink for session autosave — so its
//! whole surface (banner, prompt, dispatch, error reporting, the fail-soft
//! missing-key and failed-save paths) carries hermetic coverage.
//! Line classification stays in [`crate::command`].

mod completion;
mod curation;
mod editor;
mod history;
mod raw_live;
mod sigint_live;
mod stdin_live;

pub use completion::Completion;
pub use history::History;
pub use raw_live::{TtyEditor, stdin_stdout_are_ttys};
pub use sigint_live::install_sigint_cancel;

use crate::agent::{Agent, AgentError};
use crate::command::{Command, ModelSpec};
use crate::config::{AgentProfile, ConfirmMode};
use crate::provider::{ProviderFactory, ProviderKind};
use crate::session::Session;
use crate::session_store::SessionStore;
use crate::tools::subagent::profile_is_executor;
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

/// The input prompt, shared by [`prompt`] and the editor's in-place line
/// redraws so the two renderings can never drift apart.
pub(crate) const PROMPT: &str = "> ";

/// Fold one SIGINT into the shared cancellation flag: raise it, and report
/// whether a cancellation was *already* pending — the second Ctrl-C of a turn
/// that isn't stopping, which the live handler ([`install_sigint_cancel`])
/// answers by force-exiting the process. The decision lives here, in covered
/// code, so the `_live.rs` handler stays pure wiring.
fn request_cancel(flag: &AtomicBool) -> bool {
    flag.swap(true, Ordering::Relaxed)
}

/// The REPL's input seam: one line per call, `Ok(0)` at EOF — the semantics
/// of [`BufRead::read_line`]. `complete` supplies Tab candidates for an
/// in-progress line; only the interactive editor ([`TtyEditor`]) consults
/// it — plain buffered sources (piped input, tests) have no Tab to press.
///
/// The seam is deliberately narrower than `BufRead` because the production
/// source cannot be a held [`std::io::StdinLock`]: the shell confirmation
/// gate reads the same stdin mid-`Agent::run`, and stdin's lock is not
/// reentrant — holding it across the loop would deadlock the first
/// confirmation prompt. A private `BufReader<Stdin>` is equally wrong: its
/// read-ahead would swallow buffered lines the gate needs (piped input
/// arrives all at once). `Stdin`'s inherent `read_line` — a transient lock
/// over the shared, process-wide buffer — has neither problem; that impl
/// lives in [`stdin_live`].
pub trait LineSource {
    /// Read the next line into `buf`, returning the bytes read (0 = EOF).
    fn read_line(
        &mut self,
        buf: &mut String,
        complete: &mut dyn FnMut(&str) -> Completion,
    ) -> std::io::Result<usize>;
}

/// Any buffered reader is a line source — the hermetic test seam (`&[u8]`)
/// and any future scripted input. Whole lines arrive at once, so there is
/// no in-progress buffer to complete and the completer goes unused.
impl<R: Read> LineSource for BufReader<R> {
    fn read_line(
        &mut self,
        buf: &mut String,
        _complete: &mut dyn FnMut(&str) -> Completion,
    ) -> std::io::Result<usize> {
        BufRead::read_line(self, buf)
    }
}

/// Run the interactive loop until `/quit` (or `/exit`), EOF, or an input
/// read error. Regular output — banner, prompt, command feedback, the
/// agent's streamed turns — goes to `out`; failed turns and failed provider
/// switches are reported on `err`, mirroring the process's stdout/stderr
/// split.
///
/// `providers` is the provider-switch seam: a `/model <provider>:<model>` line
/// looks up the target's API key by environment-variable name and constructs
/// the replacement provider through it — and `/model <provider>:` plus Tab
/// completion reuse the same seam to list a *non-active* provider's models.
/// Production passes a [`ProviderFactory`] over [`crate::load_env_var`] and
/// [`crate::provider::build`] (which threads the shared cancellation flag into
/// each provider it constructs); tests inject one over stubs, keeping the
/// switch — including its fail-soft missing-key branch — hermetic. Key
/// resolution stays lazy: a switch reads the target's key only when the line
/// is entered, never eagerly.
///
/// `persist` is the session-persistence sink, injected the same way: it
/// receives a fresh conversation snapshot after every `Command::Prompt` turn
/// — succeeded, failed, or cancelled, since the agent's post-run history is
/// always a clean, valid sequence between turns — and after `Command::Clear`, so a
/// restart never reloads history the operator cleared. Production wires the
/// closure [`persist_sink`] builds — the configured `session_file` when set,
/// else a per-session auto-named file in the store, else a no-op (see that
/// factory's doc for the selection) — while tests inject a recording or
/// failing stub. See [`autosave`] for the fail-soft policy.
///
/// `profiles` is the configured subagent profile table (`config.agents`),
/// consulted only by the `/agents` report — a read-only view the model-facing
/// `task` description does not surface. `main` clones it once more for this
/// borrow, since the table itself is moved into the `task` tool.
///
/// `store` is the named session store backing `/save`, `/load`, and
/// `/sessions` — `None` when there is no `$HOME` to root it under, which those
/// three commands report as unavailable while every other command keeps
/// working. It is a peer of `persist`, not a replacement: `/load` restores
/// into the agent and then autosaves the restored state through `persist` —
/// into whichever target the sink selected, exactly as a turn would; the
/// loaded save itself is never touched.
#[allow(clippy::too_many_arguments)]
pub fn run_repl<R: LineSource, W: Write, E: Write>(
    mut input: R,
    mut out: W,
    mut err: E,
    agent: &mut Agent,
    providers: &ProviderFactory,
    profiles: &[AgentProfile],
    store: Option<&SessionStore>,
    mut persist: impl FnMut(&Session) -> Result<(), String>,
) {
    // Print registered tools for operator visibility.
    let names = agent.tool_names();
    if names.is_empty() {
        let _ = writeln!(out, "omega (no tools registered)");
    } else {
        let _ = writeln!(out, "omega [tools: {}]", names.join(", "));
    }
    // No "> " substring in this line — tests count prompts by that marker.
    let _ = writeln!(
        out,
        "(type /model to view or switch the model (provider:model switches provider too, provider: lists its models), /effort to view or set the reasoning effort, /agents to list configured subagent profiles, /clear to reset the conversation, /quit or /exit to exit)"
    );
    let _ = writeln!(out);
    prompt(&mut out);

    // Listings for providers other than the active one, fetched at most once
    // per session through the switch seam (the active provider's listing
    // lives in the agent's own cache). Two entries at most today — a linear
    // scan, not a map.
    let mut other_models: Vec<(ProviderKind, Vec<String>)> = Vec::new();

    loop {
        let mut line = String::new();
        // The completer binds the session for exactly one read: Tab
        // candidates come from the live state — the active provider seeds
        // bare tokens, and any provider's ids complete its `<name>:` prefix.
        // Candidates draw from the curated catalog: non-chat ids and
        // alias-shadowed snapshots never complete (typed in full they still
        // switch — completion offers, `/model` accepts).
        let read = input.read_line(&mut line, &mut |text| {
            let kind = agent.provider_kind();
            completion::complete(text, kind, &mut |queried| {
                curation::curate(models_for(queried, agent, &mut other_models, providers)).shown
            })
        });
        match read {
            Ok(0) | Err(_) => break, // EOF or error
            Ok(_) => {}
        }

        match Command::parse(&line) {
            Command::Empty => {}
            Command::Quit => break,
            Command::Clear => {
                agent.clear();
                let _ = writeln!(out, "(conversation cleared)");
                autosave(agent, &mut persist, &mut err);
            }
            Command::Agents => {
                // A pure report of the configured profiles — no conversation
                // state touched, so no autosave. Each line names the safety
                // flags the model-facing `task` description omits.
                if profiles.is_empty() {
                    let _ = writeln!(out, "(no agent profiles configured)");
                } else {
                    for profile in profiles {
                        let _ = writeln!(out, "{}", agent_profile_line(profile));
                    }
                }
            }
            Command::Save(None) => {
                let _ = writeln!(out, "(usage: /save <name>)");
            }
            Command::Load(None) => {
                let _ = writeln!(out, "(usage: /load <name>)");
            }
            Command::Save(Some(name)) => match store {
                None => report_no_store(&mut err),
                Some(store) => match store.save(&name, &agent.session()) {
                    Ok(()) => {
                        let _ = writeln!(out, "(saved session '{name}')");
                    }
                    // Fail soft, like a failed autosave: a bad name or a disk
                    // error warns and the session keeps running.
                    Err(e) => {
                        let _ = writeln!(err, "error: {e}");
                    }
                },
            },
            Command::Load(Some(name)) => match store {
                None => report_no_store(&mut err),
                Some(store) => match store.load(&name) {
                    Ok(session) => {
                        // The restore replaces the whole live conversation;
                        // say so, then persist the restored state through the
                        // sink — the session_file when configured, else this
                        // session's auto-save file — exactly as a turn would,
                        // so a crash right after a load does not resurrect the
                        // discarded conversation. The loaded save is untouched.
                        agent.restore(session);
                        let _ = writeln!(
                            out,
                            "(loaded session '{name}', replacing the current conversation)"
                        );
                        autosave(agent, &mut persist, &mut err);
                    }
                    // Fail soft: report the failure and list what *is* saved,
                    // so a typo'd name lands the operator on the real choices.
                    Err(e) => {
                        let _ = writeln!(err, "error: {e}");
                        list_sessions(store, &mut out);
                    }
                },
            },
            Command::Sessions => match store {
                None => report_no_store(&mut err),
                Some(store) => list_sessions(store, &mut out),
            },
            Command::SessionsDelete(None) => {
                let _ = writeln!(out, "(usage: /sessions delete <name>)");
            }
            Command::SessionsDelete(Some(name)) => match store {
                None => report_no_store(&mut err),
                Some(store) => match store.delete(&name) {
                    Ok(()) => {
                        let _ = writeln!(out, "(deleted session '{name}')");
                    }
                    // Fail soft, mirroring `/load`: report the failure and list
                    // what *is* saved, so a typo lands the operator on the real
                    // names.
                    Err(e) => {
                        let _ = writeln!(err, "error: {e}");
                        list_sessions(store, &mut out);
                    }
                },
            },
            Command::Model(None) => {
                // The effort rides on the report only when set, so the default
                // view is unchanged for the common no-effort session.
                match agent.effort() {
                    Some(effort) => {
                        let _ = writeln!(
                            out,
                            "(provider: {}, model: {}, effort: {})",
                            agent.provider_kind(),
                            agent.model(),
                            effort
                        );
                    }
                    None => {
                        let _ = writeln!(
                            out,
                            "(provider: {}, model: {})",
                            agent.provider_kind(),
                            agent.model()
                        );
                    }
                }
            }
            Command::Effort(None) => match agent.effort() {
                Some(effort) => {
                    let _ = writeln!(out, "(effort: {effort})");
                }
                None => {
                    let _ = writeln!(out, "(no effort set — using the model default)");
                }
            },
            Command::Effort(Some(effort)) => {
                agent.set_effort(Some(effort));
                let _ = writeln!(out, "(effort set to {})", agent.effort().unwrap());
            }
            Command::Confirm(None) => {
                let _ = writeln!(out, "(confirm mode: {})", agent.confirm_policy().mode());
            }
            Command::Confirm(Some(mode)) => {
                // A judge switch is validated against the profile table here,
                // at the prompt — a typo must fail loudly now, not as a
                // fail-closed deny on the first dangerous call of an
                // unattended run. Fail soft like a bad /model switch: report
                // and keep the current mode.
                if let ConfirmMode::Judge(profile) = &mode
                    && !agent.confirm_policy().has_profile(profile)
                {
                    let _ = writeln!(
                        err,
                        "error: unknown judge profile '{profile}' (mode unchanged)"
                    );
                } else {
                    agent.confirm_policy().set_mode(mode.clone());
                    let _ = writeln!(out, "(confirm mode: {mode})");
                    // Leaving `ask` removes the human from the loop — say
                    // exactly what now stands between the model and a
                    // dangerous call, on the warning channel.
                    match mode {
                        ConfirmMode::Ask => {}
                        ConfirmMode::Allow => {
                            let _ = writeln!(
                                err,
                                "warning: dangerous tool calls are now auto-approved without a \
                                 prompt; validation, guardrails, and the sandbox still apply"
                            );
                        }
                        ConfirmMode::Judge(profile) => {
                            let _ = writeln!(
                                err,
                                "warning: dangerous tool calls are now adjudicated by profile \
                                 '{profile}' with no human prompt; a denial or malformed verdict \
                                 fails closed"
                            );
                        }
                    }
                }
            }
            Command::Model(Some(ModelSpec {
                provider: None,
                model,
            })) => {
                warn_if_unlisted(
                    agent.provider_kind(),
                    &model,
                    agent,
                    &other_models,
                    &mut err,
                );
                agent.set_model(model);
                let _ = writeln!(out, "(model set to {})", agent.model());
            }
            Command::Model(Some(ModelSpec {
                provider: Some(kind),
                model,
            })) if kind == agent.provider_kind() => {
                warn_if_unlisted(kind, &model, agent, &other_models, &mut err);
                agent.set_model(model);
                let _ = writeln!(out, "(model set to {})", agent.model());
            }
            Command::Models(kind) => {
                // The active provider needs no key — it is already built.
                // Another provider without one reports the same fail-soft
                // error a switch does, rather than silently caching an empty
                // listing that would read as "this vendor has no models".
                let key_env = kind.api_key_env();
                if kind != agent.provider_kind() && providers.resolve_key(key_env).is_none() {
                    let _ = writeln!(
                        err,
                        "error: {key_env} not found in environment or .env file"
                    );
                } else {
                    // The same curated catalog Tab offers, so the two
                    // surfaces can never disagree; the hidden count keeps
                    // the listing honest about the ids it withheld.
                    let curation::Curated { shown, hidden } =
                        curation::curate(models_for(kind, agent, &mut other_models, providers));
                    if shown.is_empty() && hidden == 0 {
                        let _ = writeln!(out, "({kind}: no models listed)");
                    } else {
                        let _ = writeln!(out, "({kind} models — /model {kind}:<id> switches)");
                        for id in &shown {
                            let _ = writeln!(out, "  {id}");
                        }
                        if hidden > 0 {
                            let _ = writeln!(
                                out,
                                "  (+{hidden} hidden: dated snapshots and non-chat models — any id typed in full still switches)"
                            );
                        }
                    }
                }
            }
            Command::Model(Some(ModelSpec {
                provider: Some(kind),
                model,
            })) => {
                let key_env = kind.api_key_env();
                match providers.resolve_key(key_env) {
                    Some(key) => {
                        // Checked before the switch: the target's listing may
                        // sit in `other_models` while it is still non-active,
                        // and `set_provider` drops the agent-side cache.
                        warn_if_unlisted(kind, &model, agent, &other_models, &mut err);
                        // Read before the switch: `set_provider` clears the
                        // effort, so the notice must know whether
                        // there was one to clear — a no-effort switch stays
                        // byte-identical to today's output.
                        let had_effort = agent.effort().is_some();
                        agent.set_provider(kind, providers.build(kind, key), model);
                        let cleared = if had_effort { "; effort cleared" } else { "" };
                        let _ = writeln!(
                            out,
                            "(provider set to {}, model set to {}{})",
                            agent.provider_kind(),
                            agent.model(),
                            cleared
                        );
                    }
                    // Fail soft: a switch without a key reports and keeps the
                    // current provider — it must never kill the session.
                    None => {
                        let _ = writeln!(
                            err,
                            "error: {key_env} not found in environment or .env file (provider unchanged)"
                        );
                    }
                }
            }
            Command::Prompt(text) => {
                let _ = writeln!(out);
                match agent.run(&text, &mut out) {
                    Ok(_) => {}
                    // Cancellation is a deliberate act, not a failure: a
                    // short notice on `out` — no error dump — and the loop
                    // re-prompts. The agent already rolled the turn back
                    // (or kept it, under the side-effect carve-out).
                    Err(AgentError::Cancelled) => {
                        let _ = writeln!(out, "(turn cancelled)");
                    }
                    Err(e) => {
                        let _ = writeln!(err, "error: {e}");
                    }
                }
                let _ = writeln!(out);
                autosave(agent, &mut persist, &mut err);
            }
        }
        prompt(&mut out);
    }
}

/// Build the persistence sink [`run_repl`] fires after every turn, `/clear`,
/// and `/load` — the one place the three autosave targets are selected. Lives
/// beside the loop, in covered code, because `main.rs` (coverage-excluded) must
/// only call it:
///
/// - `session_file` set → the pre-existing fixed-path autosave, byte-identical
///   (the project override wins; no auto-save file is written even when a store
///   is available).
/// - `session_file` unset **and** a `store` present → every session autosaves
///   into its own `auto-<UTC-timestamp>` file under the store, named lazily on
///   the first write (see [`auto_session_base`]) and reused for the rest of the
///   session, so within a session the file mirrors the live conversation while
///   a new session always opens a new file.
/// - neither (no `session_file`, no `$HOME` for a store) → the no-op sink,
///   today's no-persistence behavior.
///
/// The store is **borrowed**, not owned — [`SessionStore`] is not `Clone`, and
/// `main.rs` passes the same `store` on into [`run_repl`] — so the returned
/// closure is tied to its lifetime.
pub fn persist_sink(
    session_file: Option<String>,
    store: Option<&SessionStore>,
) -> impl FnMut(&Session) -> Result<(), String> + '_ {
    // The lazily-chosen auto-save name, held for the session's lifetime so
    // every write after the first rolls the same file in place.
    let mut auto_name: Option<String> = None;
    move |session: &Session| match (&session_file, store) {
        // The project override wins outright — no auto file, today's behavior.
        (Some(path), _) => session.save(path),
        (None, Some(store)) => persist_auto(store, &mut auto_name, SystemTime::now(), session),
        // No override and nowhere to store — persistence off.
        (None, None) => Ok(()),
    }
}

/// Save to the name already owned by this session, or atomically allocate the
/// first one. The name is cached only after the initial commit succeeds, so a
/// fail-soft autosave retries allocation rather than claiming an unwritten
/// target for the rest of the process.
fn persist_auto(
    store: &SessionStore,
    auto_name: &mut Option<String>,
    now: SystemTime,
    session: &Session,
) -> Result<(), String> {
    if let Some(name) = auto_name {
        return store.save(name, session);
    }
    let name = store.save_auto(&auto_session_base(now), session)?;
    *auto_name = Some(name);
    Ok(())
}

/// The unsuffixed auto-save base for this session. [`SessionStore::save_auto`]
/// owns collision handling because uniqueness must be decided atomically at
/// the filesystem, not inferred from a stale listing.
fn auto_session_base(now: SystemTime) -> String {
    format!("auto-{}", format_utc_timestamp(now))
}

/// Format `now` as `YYYYMMDD-HHMMSS` in UTC. The date is a hand-rolled
/// days-to-civil conversion (Howard Hinnant's algorithm) rather than a
/// dependency — the project builds this kind of thing from scratch, and the
/// std library exposes no calendar. A time before the Unix epoch — only
/// reachable from a caller-supplied `SystemTime`, never from `now()` — clamps
/// to the epoch rather than underflowing.
fn format_utc_timestamp(now: SystemTime) -> String {
    let secs = now
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (hour, minute, second) = (rem / 3_600, (rem % 3_600) / 60, rem % 60);

    // civil_from_days: shift the epoch to 0000-03-01 so leap days fall at the
    // end of the 400-year era, then invert the day-of-era arithmetic. All
    // operands stay non-negative (days ≥ 0 after the epoch clamp), so no
    // signed-era correction is needed.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!("{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}")
}

/// Push the agent's conversation snapshot through the persistence sink.
///
/// **Save fails soft:** a failed autosave — disk
/// full, permissions — prints a warning to `err` and the session keeps
/// running. A best-effort write must never kill an interactive turn,
/// mirroring the loop's best-effort terminal writes and the fail-soft
/// provider-switch precedent. The counterpart policy — **load fails fast**,
/// exactly like `Config::load` — lives at the startup wiring in `main.rs`,
/// the only place a session file is read. `dyn` callbacks for the same
/// reason as [`models_for`]: one instantiation shared by every `run_repl`
/// instantiation, so all branches carry coverage collectively.
fn autosave(
    agent: &Agent,
    persist: &mut dyn FnMut(&Session) -> Result<(), String>,
    err: &mut dyn Write,
) {
    if let Err(e) = persist(&agent.session()) {
        let _ = writeln!(err, "warning: session not saved: {e}");
    }
}

/// Report, on `err`, that the named session store is unavailable — the shared
/// answer of `/save`, `/load`, and `/sessions` when there was no `$HOME` to
/// root the store under. `dyn Write` for the same one-instantiation reason as
/// [`autosave`].
fn report_no_store(err: &mut dyn Write) {
    let _ = writeln!(
        err,
        "error: session store unavailable (no home directory to store sessions under)"
    );
}

/// List the store's saved sessions on `out`, newest first, or say there are
/// none. Each row carries the name, a relative age, and a size through the
/// shared [`SessionEntry::describe`](crate::session_store::SessionEntry::describe)
/// formatter (metadata,
/// never file content). Shared by `/sessions` and the fail-soft `/load` and
/// `/sessions delete` fallbacks, so every listing renders the same way. `dyn
/// Write` for the same reason as [`autosave`].
fn list_sessions(store: &SessionStore, out: &mut dyn Write) {
    let saves = store.list();
    if saves.is_empty() {
        let _ = writeln!(out, "(no saved sessions)");
    } else {
        let _ = writeln!(out, "(saved sessions, newest first:)");
        let now = SystemTime::now();
        for entry in &saves {
            let _ = writeln!(out, "  {}", entry.describe(now));
        }
    }
}

/// The model ids for `kind`, fetched at most once per session per provider.
/// The active provider answers from the agent's own lazy cache; any other
/// provider is built fresh from its resolved key through the switch seam,
/// listed once, and held in `cache` — empty on a missing key or a failed
/// fetch, the same fail-soft, no-retry contract as the agent's cache, so a
/// flaky network cannot tax every Tab press. Taking the concrete
/// [`ProviderFactory`] (not a generic closure) keeps this one instantiation
/// shared by every `run_repl` instantiation, so all of its branches carry
/// coverage collectively.
fn models_for(
    kind: ProviderKind,
    agent: &mut Agent,
    cache: &mut Vec<(ProviderKind, Vec<String>)>,
    providers: &ProviderFactory,
) -> Vec<String> {
    if kind == agent.provider_kind() {
        return agent.list_models_cached().to_vec();
    }
    if let Some((_, ids)) = cache.iter().find(|(cached, _)| *cached == kind) {
        return ids.clone();
    }
    let ids = providers
        .resolve_key(kind.api_key_env())
        .map(|key| providers.build(kind, key).list_models().unwrap_or_default())
        .unwrap_or_default();
    cache.push((kind, ids.clone()));
    ids
}

/// Warn (softly, one line on `err`) when a switched-to id is absent from the
/// target provider's *already cached* catalog. Purely advisory: ids stay
/// unvalidated (the standing no-registry decision — fine-tune and alias ids
/// must always work) and the caller switches anyway; request-time fail-fast
/// remains the authority. No fetch is ever triggered here — an uncached
/// catalog stays silent, and so does the *empty* cache a failed fetch leaves
/// behind (both mean "catalog unknown"; warning on every id because the
/// network was down once would be noise, not signal). The check reads the
/// raw, uncurated listing: curation hides real, switchable ids — dated
/// snapshots, non-chat models — that must not draw a warning. `dyn` for the
/// same one-instantiation reason as [`models_for`].
fn warn_if_unlisted(
    kind: ProviderKind,
    model: &str,
    agent: &Agent,
    cache: &[(ProviderKind, Vec<String>)],
    err: &mut dyn Write,
) {
    let listed = if kind == agent.provider_kind() {
        agent.cached_models()
    } else {
        cache
            .iter()
            .find(|(cached, _)| *cached == kind)
            .map(|(_, ids)| ids.as_slice())
    };
    if listed.is_some_and(|ids| !ids.is_empty() && !ids.iter().any(|id| id == model)) {
        let _ = writeln!(
            err,
            "warning: {model} is not in {kind}'s listed models (switching anyway)"
        );
    }
}

/// One line of the `/agents` report: a profile's name and its safety-relevant
/// configuration, dense enough to scan the table at a glance. The `executor`
/// marker comes from the same [`profile_is_executor`] classifier the `task`
/// tool uses, so the operator's view matches how a delegation is actually
/// treated — a read-only child fans out; an executor forces serial execution
/// and prompts before each mutating call. `effort`, the `confirm` override,
/// and the `max_turns` override ride the line only when set, and the tool
/// allowlist collapses to `read-only default` when the profile omits `tools`.
fn agent_profile_line(profile: &AgentProfile) -> String {
    let mut line = format!(
        "({} — provider: {}, model: {}",
        profile.name, profile.provider, profile.model
    );
    if let Some(effort) = &profile.effort {
        line.push_str(&format!(", effort: {effort}"));
    }
    match &profile.tools {
        Some(tools) => line.push_str(&format!(", tools: {}", tools.join(", "))),
        None => line.push_str(", read-only default"),
    }
    if profile_is_executor(profile) {
        line.push_str(", executor");
    }
    if let Some(confirm) = &profile.confirm {
        line.push_str(&format!(", confirm: {confirm}"));
    }
    if let Some(max_turns) = profile.max_turns {
        line.push_str(&format!(", max_turns: {max_turns}"));
    }
    line.push(')');
    line
}

/// Write the input prompt and flush it, so it shows before the read blocks.
/// Terminal writes are best-effort throughout the loop — a failed write is
/// deliberately ignored rather than crashing an interactive session.
fn prompt(out: &mut impl Write) {
    let _ = write!(out, "{PROMPT}");
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentConfig;
    use crate::provider::Provider;
    use crate::testing::{ErrProvider, MockProvider, TestTool};
    use crate::tools::default_tools;
    use crate::tools::sandbox::Sandbox;
    use crate::turn::{Block, Role, StopReason, StreamDelta, TurnMessage, Usage};
    use std::sync::Arc;

    /// A canned stream that replies with `text` and ends the turn.
    fn text_stream(text: &str) -> Vec<StreamDelta> {
        vec![
            StreamDelta::MessageStart {
                usage: Usage::default(),
            },
            StreamDelta::TextStart {
                index: 0,
                text: text.to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage::default(),
            },
        ]
    }

    fn agent_with(provider: Box<dyn Provider>) -> Agent {
        Agent::new(
            provider,
            AgentConfig {
                provider_kind: ProviderKind::Anthropic,
                model: crate::TEST_MODEL.to_string(),
                max_tokens: 64,
                system: None,
                context_token_limit: 100_000,
                effort: None,
                max_turns: crate::agent::MAX_TURNS,
            },
            vec![],
        )
    }

    // Provider-switch callbacks shared across tests. Named fns rather than
    // per-test closures: under the scoped-100 gate a closure only some tests
    // pass — and never execute — would be a permanently-dead line.

    /// Resolves OpenAI's key (tagged with the variable name, so a test can
    /// assert the builder received it) and misses Anthropic's. One resolver
    /// serving both outcomes keeps the switch's success *and* fail-soft
    /// branches reachable through the single shared [`repl`] instantiation —
    /// the gate scores a generic fn by its best-covered instantiation, so
    /// every `run_repl` line must be reachable without changing the callback
    /// types.
    fn test_resolver(env: &str) -> Option<String> {
        (env == "OPENAI_API_KEY").then(|| format!("key-for-{env}"))
    }

    /// A switch target with no canned streams — for tests that assert the
    /// report lines and never run a post-switch turn.
    fn build_mock(_kind: ProviderKind, _key: String) -> Box<dyn Provider> {
        Box::new(MockProvider::new(vec![]))
    }

    /// The no-op persistence sink — persistence-off behavior. Named, like
    /// [`test_resolver`]: it executes in every test that prompts or clears,
    /// so it carries no dead line.
    fn ignore_session(_: &Session) -> Result<(), String> {
        Ok(())
    }

    /// Drive the loop over scripted input, returning (out, err) as strings.
    /// Key lookups go through [`test_resolver`] and build an empty
    /// `MockProvider`; tests that need different switch behavior use
    /// [`repl_with`]; tests that observe persistence use [`repl_persisting`].
    fn repl(input: &[u8], agent: &mut Agent) -> (String, String) {
        repl_with(input, agent, test_resolver, build_mock)
    }

    /// [`repl`] with the provider-switch seam made explicit.
    fn repl_with(
        input: &[u8],
        agent: &mut Agent,
        resolve_key: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
        build_provider: impl Fn(ProviderKind, String) -> Box<dyn Provider> + Send + Sync + 'static,
    ) -> (String, String) {
        let providers = ProviderFactory::from_fns(build_provider, resolve_key);
        let mut out = Vec::new();
        let mut err = Vec::new();
        run_repl(
            BufReader::new(input),
            &mut out,
            &mut err,
            agent,
            &providers,
            &[],
            None,
            ignore_session,
        );
        (
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    /// [`repl`] with a configured subagent profile table, for the `/agents`
    /// report. The switch seam and persistence stay the standard stubs.
    fn repl_agents(input: &[u8], agent: &mut Agent, profiles: &[AgentProfile]) -> (String, String) {
        let providers = ProviderFactory::from_fns(build_mock, test_resolver);
        let mut out = Vec::new();
        let mut err = Vec::new();
        run_repl(
            BufReader::new(input),
            &mut out,
            &mut err,
            agent,
            &providers,
            profiles,
            None,
            ignore_session,
        );
        (
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    /// [`repl`] with the persistence sink made explicit.
    fn repl_persisting(
        input: &[u8],
        agent: &mut Agent,
        persist: impl FnMut(&Session) -> Result<(), String>,
    ) -> (String, String) {
        let providers = ProviderFactory::from_fns(build_mock, test_resolver);
        let mut out = Vec::new();
        let mut err = Vec::new();
        run_repl(
            BufReader::new(input),
            &mut out,
            &mut err,
            agent,
            &providers,
            &[],
            None,
            persist,
        );
        (
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    /// A `SessionStore` rooted under fresh temp dirs. The returned temp dirs
    /// must outlive the store — they own the directory the store writes into.
    fn temp_store() -> (tempfile::TempDir, tempfile::TempDir, SessionStore) {
        let home = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        let store = SessionStore::new(home.path(), project.path());
        (home, project, store)
    }

    /// [`repl`] with a real session store wired in. Persistence stays the no-op
    /// [`ignore_session`], so these run in the same `run_repl` instantiation as
    /// every other store-free test.
    fn repl_with_store(input: &[u8], agent: &mut Agent, store: &SessionStore) -> (String, String) {
        let providers = ProviderFactory::from_fns(build_mock, test_resolver);
        let mut out = Vec::new();
        let mut err = Vec::new();
        run_repl(
            BufReader::new(input),
            &mut out,
            &mut err,
            agent,
            &providers,
            &[],
            Some(store),
            ignore_session,
        );
        (
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    /// One user-text message, for crafting the sessions these tests save and
    /// restore.
    fn user_msg(text: &str) -> TurnMessage {
        TurnMessage {
            role: Role::User,
            content: vec![Block::Text(text.to_string())],
        }
    }

    /// A session carrying `messages` and nothing else set.
    fn session_of(messages: Vec<TurnMessage>) -> Session {
        Session {
            version: crate::session::SESSION_VERSION,
            messages,
            compacted_summary: None,
            last_input_tokens: 0,
        }
    }

    #[test]
    fn eof_exits_after_banner_and_prompt() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl(b"", &mut agent);
        assert!(out.starts_with("omega (no tools registered)\n"));
        assert!(out.contains(
            "(type /model to view or switch the model (provider:model switches provider too, provider: lists its models), /effort to view or set the reasoning effort, /agents to list configured subagent profiles, /clear to reset the conversation, /quit or /exit to exit)\n"
        ));
        assert!(out.ends_with("> "));
        assert!(err.is_empty());
    }

    #[test]
    fn banner_lists_registered_tools() {
        // The real tool set (minus the key-gated web_search), in registration
        // order — the banner is the operator's view of what the agent can do.
        let dir = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![])),
            AgentConfig {
                provider_kind: ProviderKind::Anthropic,
                model: crate::TEST_MODEL.to_string(),
                max_tokens: 64,
                system: None,
                context_token_limit: 100_000,
                effort: None,
                max_turns: crate::agent::MAX_TURNS,
            },
            default_tools(sandbox, None, Arc::new(AtomicBool::new(false))),
        );
        let (out, _) = repl(b"", &mut agent);
        assert!(out.starts_with(
            "omega [tools: web_fetch, read_file, write_file, edit_file, list_directory, search_files, shell]\n"
        ));
    }

    #[test]
    fn completer_serves_session_state_to_the_line_source() {
        // The closure run_repl hands each read binds the live agent: provider
        // names, the active provider's gating, and 6b's cached ids must all
        // flow through it. Buffered sources never press Tab, so a probing
        // source queries the completer the way the interactive editor would.
        struct CompletionProbe {
            query: &'static str,
            // A shared handle, like MockProvider's logs: the probe is moved
            // into run_repl, so the test observes through the clone.
            seen: std::rc::Rc<std::cell::RefCell<Option<Completion>>>,
        }
        impl LineSource for CompletionProbe {
            fn read_line(
                &mut self,
                _buf: &mut String,
                complete: &mut dyn FnMut(&str) -> Completion,
            ) -> std::io::Result<usize> {
                *self.seen.borrow_mut() = Some(complete(self.query));
                Ok(0) // one probe, then EOF
            }
        }
        let mut agent = agent_with(Box::new(
            MockProvider::new(vec![]).with_models(&["claude-a", "claude-b"]),
        ));
        let seen = std::rc::Rc::new(std::cell::RefCell::new(None));
        let providers = ProviderFactory::from_fns(build_mock, test_resolver);
        run_repl(
            CompletionProbe {
                query: "/model ",
                seen: std::rc::Rc::clone(&seen),
            },
            &mut Vec::new(),
            &mut std::io::sink(),
            &mut agent,
            &providers,
            &[],
            None,
            ignore_session,
        );
        let completion = seen.borrow_mut().take().unwrap();
        assert_eq!(completion.start, 7);
        assert_eq!(
            completion.candidates,
            ["anthropic:", "openai:", "claude-a", "claude-b"]
        );
    }

    #[test]
    fn read_error_exits_cleanly() {
        struct FailingReader;
        impl Read for FailingReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("terminal vanished"))
            }
        }
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let mut out = Vec::new();
        let providers = ProviderFactory::from_fns(build_mock, test_resolver);
        run_repl(
            BufReader::new(FailingReader),
            &mut out,
            &mut std::io::sink(),
            &mut agent,
            &providers,
            &[],
            None,
            ignore_session,
        );
        // The loop stops at the failed read, after the banner and one prompt.
        assert!(String::from_utf8(out).unwrap().ends_with("> "));
    }

    #[test]
    fn blank_lines_reprompt_without_side_effects() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl(b"\n   \n", &mut agent);
        // The initial prompt plus one re-prompt per blank line.
        assert_eq!(out.matches("> ").count(), 3);
        assert!(err.is_empty());
    }

    #[test]
    fn quit_stops_the_loop_before_later_input() {
        // The zero-stream MockProvider panics if the agent ever runs, so
        // reaching the line after /quit would fail the test.
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl(b"/quit\nnever sent\n", &mut agent);
        assert_eq!(out.matches("> ").count(), 1);
    }

    #[test]
    fn exit_is_an_alias_for_quit() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl(b"/exit\nnever sent\n", &mut agent);
        assert_eq!(out.matches("> ").count(), 1);
    }

    #[test]
    fn clear_reports_the_reset() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl(b"/clear\n", &mut agent);
        assert!(out.contains("(conversation cleared)\n"));
    }

    // ── /agents report ──

    /// A profile with the read-only default toolset and no optional fields.
    fn profile_for(name: &str, provider: ProviderKind, model: &str) -> AgentProfile {
        AgentProfile {
            name: name.to_string(),
            provider,
            model: model.to_string(),
            effort: None,
            system: None,
            tools: None,
            confirm: None,
            max_turns: None,
        }
    }

    #[test]
    fn agents_reports_no_profiles_configured() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl_agents(b"/agents\n", &mut agent, &[]);
        assert!(
            out.contains("(no agent profiles configured)\n"),
            "got: {out}"
        );
        assert!(err.is_empty());
    }

    #[test]
    fn agents_reports_a_full_profile() {
        // Every optional field set, an explicit (read-only) allowlist rather
        // than the default, and a confirm override — the dense line carries
        // them all, in order, and the read-only allowlist draws no executor
        // marker.
        let mut p = profile_for("reviewer", ProviderKind::Openai, "gpt-x");
        p.effort = Some("high".to_string());
        p.tools = Some(vec!["read_file".to_string(), "web_fetch".to_string()]);
        p.confirm = Some(ConfirmMode::Allow);
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl_agents(b"/agents\n", &mut agent, &[p]);
        assert!(
            out.contains(
                "(reviewer — provider: openai, model: gpt-x, effort: high, tools: read_file, web_fetch, confirm: allow)\n"
            ),
            "got: {out}"
        );
        assert!(!out.contains("executor"), "got: {out}");
    }

    #[test]
    fn agents_reports_a_minimal_profile_as_read_only_default() {
        // No effort, no allowlist, no confirm override: the read-only default
        // and no executor marker.
        let p = profile_for("reader", ProviderKind::Anthropic, "m");
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl_agents(b"/agents\n", &mut agent, &[p]);
        assert!(
            out.contains("(reader — provider: anthropic, model: m, read-only default)\n"),
            "got: {out}"
        );
        assert!(!out.contains("executor"), "got: {out}");
    }

    #[test]
    fn agents_marks_an_executor_profile() {
        // A mutating tool makes the profile an executor — the marker comes
        // from the same classifier the `task` tool applies.
        let mut p = profile_for("editor", ProviderKind::Anthropic, "m");
        p.tools = Some(vec!["edit_file".to_string()]);
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl_agents(b"/agents\n", &mut agent, &[p]);
        assert!(
            out.contains("(editor — provider: anthropic, model: m, tools: edit_file, executor)\n"),
            "got: {out}"
        );
    }

    #[test]
    fn agents_shows_a_confirm_pinned_executor() {
        // The confirm override renders with the ConfirmMode spelling, after
        // the executor marker.
        let mut p = profile_for("guarded", ProviderKind::Anthropic, "m");
        p.tools = Some(vec!["shell".to_string()]);
        p.confirm = Some(ConfirmMode::Judge("guarded".to_string()));
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl_agents(b"/agents\n", &mut agent, &[p]);
        assert!(
            out.contains(
                "(guarded — provider: anthropic, model: m, tools: shell, executor, confirm: judge guarded)\n"
            ),
            "got: {out}"
        );
    }

    #[test]
    fn agents_shows_a_max_turns_override() {
        // The per-profile max_turns override rides the line only when set,
        // trailing the other conditional fields.
        let mut p = profile_for("surveyor", ProviderKind::Anthropic, "m");
        p.max_turns = Some(40);
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl_agents(b"/agents\n", &mut agent, &[p]);
        assert!(
            out.contains(
                "(surveyor — provider: anthropic, model: m, read-only default, max_turns: 40)\n"
            ),
            "got: {out}"
        );
    }

    #[test]
    fn agents_lists_every_profile_one_per_line() {
        let a = profile_for("a", ProviderKind::Anthropic, "m1");
        let b = profile_for("b", ProviderKind::Openai, "m2");
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl_agents(b"/agents\n", &mut agent, &[a, b]);
        assert!(
            out.contains("(a — provider: anthropic, model: m1, read-only default)\n"),
            "got: {out}"
        );
        assert!(
            out.contains("(b — provider: openai, model: m2, read-only default)\n"),
            "got: {out}"
        );
    }

    #[test]
    fn bare_model_reports_provider_and_model() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl(b"/model\n", &mut agent);
        assert!(out.contains(&format!(
            "(provider: anthropic, model: {})\n",
            crate::TEST_MODEL
        )));
    }

    #[test]
    fn bare_confirm_reports_the_default_ask() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl(b"/confirm\n", &mut agent);
        assert!(out.contains("(confirm mode: ask)\n"), "got: {out}");
    }

    #[test]
    fn confirm_allow_switches_with_a_warning() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl(b"/confirm allow\n/confirm\n", &mut agent);
        assert!(out.contains("(confirm mode: allow)\n"), "got: {out}");
        // The warning names what now stands between the model and a
        // dangerous call, on the warning channel.
        assert!(
            err.contains("warning: dangerous tool calls are now auto-approved"),
            "got: {err}"
        );
        // The switch outlives the loop.
        assert_eq!(agent.confirm_policy().mode(), ConfirmMode::Allow);
    }

    #[test]
    fn confirm_ask_switches_back_without_a_warning() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl(b"/confirm allow\n/confirm ask\n", &mut agent);
        assert!(out.contains("(confirm mode: ask)\n"), "got: {out}");
        // One warning (the allow switch) — returning to ask adds none.
        assert_eq!(err.matches("warning:").count(), 1, "got: {err}");
        assert_eq!(agent.confirm_policy().mode(), ConfirmMode::Ask);
    }

    #[test]
    fn confirm_judge_with_unknown_profile_fails_soft() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (_, err) = repl(b"/confirm judge ghost\n", &mut agent);
        assert!(
            err.contains("error: unknown judge profile 'ghost' (mode unchanged)"),
            "got: {err}"
        );
        assert_eq!(agent.confirm_policy().mode(), ConfirmMode::Ask);
    }

    #[test]
    fn confirm_judge_with_a_known_profile_switches_with_a_warning() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        // The session policy carries the profile table the switch validates
        // against — installed here as main would from config.json.
        agent.set_confirm_policy(crate::agent::ConfirmPolicy::new(
            ConfirmMode::Ask,
            crate::testing::no_prompt,
            ProviderFactory::from_fns(build_mock, test_resolver),
            vec![crate::config::AgentProfile {
                name: "sentinel".to_string(),
                provider: ProviderKind::Anthropic,
                model: "judge-model".to_string(),
                effort: None,
                system: None,
                tools: None,
                confirm: None,
                max_turns: None,
            }],
            crate::tools::sandbox::Sandbox::unbounded(),
        ));
        let (out, err) = repl(b"/confirm judge sentinel\n", &mut agent);
        assert!(
            out.contains("(confirm mode: judge sentinel)\n"),
            "got: {out}"
        );
        assert!(
            err.contains("warning: dangerous tool calls are now adjudicated by profile 'sentinel'"),
            "got: {err}"
        );
        assert_eq!(
            agent.confirm_policy().mode(),
            ConfirmMode::Judge("sentinel".to_string())
        );
    }

    #[test]
    fn bare_effort_reports_none_when_unset() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl(b"/effort\n", &mut agent);
        assert!(out.contains("(no effort set — using the model default)\n"));
    }

    #[test]
    fn effort_with_value_sets_and_bare_effort_reports_it() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl(b"/effort high\n/effort\n", &mut agent);
        assert!(out.contains("(effort set to high)\n"));
        // The report reads the swap back, and it outlives the loop.
        assert!(out.contains("(effort: high)\n"));
        assert_eq!(agent.effort(), Some("high"));
    }

    #[test]
    fn bare_model_report_includes_effort_when_set() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl(b"/effort xhigh\n/model\n", &mut agent);
        assert!(out.contains(&format!(
            "(provider: anthropic, model: {}, effort: xhigh)\n",
            crate::TEST_MODEL
        )));
    }

    #[test]
    fn provider_switch_clears_a_set_effort_and_says_so() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl(b"/effort high\n/model openai:gpt-4o\n/effort\n", &mut agent);
        assert!(out.contains("(provider set to openai, model set to gpt-4o; effort cleared)\n"));
        // The clear is real: the effort is gone after the switch.
        assert!(out.contains("(no effort set — using the model default)\n"));
        assert!(agent.effort().is_none());
        assert!(err.is_empty());
    }

    #[test]
    fn model_with_id_switches_and_reports() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl(b"/model gpt-4o\n/model\n", &mut agent);
        assert!(out.contains("(model set to gpt-4o)\n"));
        // The bare report reflects the swap, and it outlives the loop.
        assert!(out.contains("(provider: anthropic, model: gpt-4o)\n"));
        assert_eq!(agent.model(), "gpt-4o");
    }

    #[test]
    fn model_with_provider_prefix_switches_provider_and_model() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl(b"/model openai:gpt-4o\n/model\n", &mut agent);
        assert!(out.contains("(provider set to openai, model set to gpt-4o)\n"));
        // The bare report reads the swap back from the agent — the single
        // source of provider identity — and the swap outlives the loop.
        assert!(out.contains("(provider: openai, model: gpt-4o)\n"));
        assert_eq!(agent.provider_kind(), ProviderKind::Openai);
        assert_eq!(agent.model(), "gpt-4o");
        assert!(err.is_empty());
    }

    #[test]
    fn provider_switch_builds_with_the_resolved_key_and_routes_turns_to_it() {
        // The builder receives the kind and the key the resolver produced,
        // and the provider it returns answers subsequent prompts.
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl_with(
            b"/model openai:gpt-4o\nhi\n",
            &mut agent,
            test_resolver,
            |kind, key| {
                assert_eq!(kind, ProviderKind::Openai);
                assert_eq!(key, "key-for-OPENAI_API_KEY");
                Box::new(MockProvider::new(vec![text_stream("routed to openai")]))
            },
        );
        // The canned reply proves the built provider took over the turn (the
        // original provider has no streams and would panic if still active).
        assert!(out.contains("routed to openai\n"));
        assert!(err.is_empty());
    }

    #[test]
    fn models_listing_for_the_active_provider_needs_no_key() {
        // `/model anthropic:` under an Anthropic session answers from the
        // agent's own cache. The shared resolver has no Anthropic key, so
        // reaching the listing proves the active provider skips resolution.
        let mut agent = agent_with(Box::new(
            MockProvider::new(vec![]).with_models(&["claude-a", "claude-b"]),
        ));
        let (out, err) = repl(b"/model anthropic:\n", &mut agent);
        assert!(out.contains(
            "(anthropic models — /model anthropic:<id> switches)\n  claude-a\n  claude-b\n"
        ));
        assert!(err.is_empty());
        // A listing is a view, not a switch.
        assert_eq!(agent.provider_kind(), ProviderKind::Anthropic);
    }

    #[test]
    fn models_listing_for_another_provider_builds_it_once_with_the_key() {
        // `/model openai:` under an Anthropic session goes through the
        // switch seam: resolve the key, build the provider, list. The second
        // request answers from the session cache — the builder fires once.
        // Shared `Arc<AtomicUsize>`, not a borrowed `Cell`: the seam stores its
        // builder in a `Send + Sync + 'static` `Arc`, so an observer captured
        // into it must be `'static` and thread-safe too.
        let builds = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl_with(
            b"/model openai:\n/model openai:\n",
            &mut agent,
            test_resolver,
            {
                let builds = Arc::clone(&builds);
                move |kind, key| {
                    builds.fetch_add(1, Ordering::Relaxed);
                    assert_eq!(kind, ProviderKind::Openai);
                    assert_eq!(key, "key-for-OPENAI_API_KEY");
                    Box::new(MockProvider::new(vec![]).with_models(&["gpt-a", "gpt-b"]))
                }
            },
        );
        assert_eq!(
            out.matches("(openai models — /model openai:<id> switches)\n  gpt-a\n  gpt-b\n")
                .count(),
            2
        );
        assert_eq!(builds.load(Ordering::Relaxed), 1);
        assert!(err.is_empty());
        // Listing another provider does not switch to it.
        assert_eq!(agent.provider_kind(), ProviderKind::Anthropic);
    }

    #[test]
    fn models_listing_without_key_fails_soft() {
        // A missing key reports on stderr instead of caching an empty
        // listing, and the loop survives to serve the next line. Reached
        // through the shared resolver — switch to OpenAI first, then ask for
        // Anthropic's listing, whose key the resolver lacks — so the branch
        // is covered in the same `run_repl` instantiation as every other.
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl(
            b"/model openai:gpt-4o\n/model anthropic:\n/model\n",
            &mut agent,
        );
        assert_eq!(
            err,
            "error: ANTHROPIC_API_KEY not found in environment or .env file\n"
        );
        assert!(out.contains("(provider: openai, model: gpt-4o)\n"));
    }

    #[test]
    fn models_listing_empty_is_reported_not_blank() {
        // A provider whose listing comes back empty (offline, or genuinely
        // none) says so, rather than printing a heading over nothing.
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl(b"/model openai:\n", &mut agent);
        assert!(out.contains("(openai: no models listed)\n"));
    }

    #[test]
    fn models_listing_curates_and_reports_the_hidden_count() {
        // Non-chat ids and alias-shadowed snapshots drop out of the printed
        // listing, and the trailing note owns up to how many.
        let mut agent = agent_with(Box::new(MockProvider::new(vec![]).with_models(&[
            "claude-x",
            "claude-x-2026-06-01",
            "whisper-1",
        ])));
        let (out, err) = repl(b"/model anthropic:\n", &mut agent);
        assert!(out.contains(
            "(anthropic models — /model anthropic:<id> switches)\n  claude-x\n  \
             (+2 hidden: dated snapshots and non-chat models — any id typed in full still switches)\n"
        ));
        assert!(!out.contains("whisper"));
        assert!(err.is_empty());
    }

    #[test]
    fn models_listing_with_everything_hidden_still_prints_the_count() {
        // A catalog of nothing but junk is not "no models listed" — the
        // heading and the count explain where the ids went.
        let mut agent = agent_with(Box::new(
            MockProvider::new(vec![]).with_models(&["whisper-1"]),
        ));
        let (out, _) = repl(b"/model anthropic:\n", &mut agent);
        assert!(out.contains(
            "(anthropic models — /model anthropic:<id> switches)\n  \
             (+1 hidden: dated snapshots and non-chat models — any id typed in full still switches)\n"
        ));
    }

    #[test]
    fn completer_fetches_another_providers_ids_through_the_seam() {
        // Tab on `openai:g` under an Anthropic session: the completer routes
        // the queried kind through models_for, which builds the non-active
        // provider and serves its ids.
        struct CompletionProbe {
            query: &'static str,
            seen: std::rc::Rc<std::cell::RefCell<Option<Completion>>>,
        }
        impl LineSource for CompletionProbe {
            fn read_line(
                &mut self,
                _buf: &mut String,
                complete: &mut dyn FnMut(&str) -> Completion,
            ) -> std::io::Result<usize> {
                *self.seen.borrow_mut() = Some(complete(self.query));
                Ok(0)
            }
        }
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let seen = std::rc::Rc::new(std::cell::RefCell::new(None));
        let providers = ProviderFactory::from_fns(
            |_, _| Box::new(MockProvider::new(vec![]).with_models(&["gpt-4o", "o4-mini"])),
            test_resolver,
        );
        run_repl(
            CompletionProbe {
                query: "/model openai:g",
                seen: std::rc::Rc::clone(&seen),
            },
            &mut Vec::new(),
            &mut std::io::sink(),
            &mut agent,
            &providers,
            &[],
            None,
            ignore_session,
        );
        let completion = seen.borrow_mut().take().unwrap();
        assert_eq!(completion.start, 14);
        assert_eq!(completion.candidates, ["gpt-4o"]);
    }

    #[test]
    fn completer_offers_only_the_curated_catalog() {
        // Tab candidates come from the same curated pool the printed listing
        // shows: the snapshot collapses onto its listed alias and the
        // non-chat id never surfaces.
        struct CompletionProbe {
            seen: std::rc::Rc<std::cell::RefCell<Option<Completion>>>,
        }
        impl LineSource for CompletionProbe {
            fn read_line(
                &mut self,
                _buf: &mut String,
                complete: &mut dyn FnMut(&str) -> Completion,
            ) -> std::io::Result<usize> {
                *self.seen.borrow_mut() = Some(complete("/model anthropic:"));
                Ok(0)
            }
        }
        let mut agent = agent_with(Box::new(MockProvider::new(vec![]).with_models(&[
            "claude-x",
            "claude-x-2026-06-01",
            "whisper-1",
        ])));
        let seen = std::rc::Rc::new(std::cell::RefCell::new(None));
        let providers = ProviderFactory::from_fns(build_mock, test_resolver);
        run_repl(
            CompletionProbe {
                seen: std::rc::Rc::clone(&seen),
            },
            &mut Vec::new(),
            &mut std::io::sink(),
            &mut agent,
            &providers,
            &[],
            None,
            ignore_session,
        );
        let completion = seen.borrow_mut().take().unwrap();
        assert_eq!(completion.candidates, ["claude-x"]);
    }

    #[test]
    fn completer_without_key_offers_nothing_for_the_other_provider() {
        // The same probe with no resolvable key: completion is fail-soft —
        // empty candidates, never an error mid-keystroke.
        struct CompletionProbe {
            seen: std::rc::Rc<std::cell::RefCell<Option<Completion>>>,
        }
        impl LineSource for CompletionProbe {
            fn read_line(
                &mut self,
                _buf: &mut String,
                complete: &mut dyn FnMut(&str) -> Completion,
            ) -> std::io::Result<usize> {
                *self.seen.borrow_mut() = Some(complete("/model openai:"));
                Ok(0)
            }
        }
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let seen = std::rc::Rc::new(std::cell::RefCell::new(None));
        let providers = ProviderFactory::from_fns(build_mock, |_| None);
        run_repl(
            CompletionProbe {
                seen: std::rc::Rc::clone(&seen),
            },
            &mut Vec::new(),
            &mut std::io::sink(),
            &mut agent,
            &providers,
            &[],
            None,
            ignore_session,
        );
        let completion = seen.borrow_mut().take().unwrap();
        assert!(completion.candidates.is_empty());
    }

    #[test]
    fn same_provider_prefix_switches_without_a_resolvable_key_and_keeps_cache() {
        // The shared resolver has no Anthropic key. The already-running
        // provider does not need one to change its model, and the two listings
        // around the switch must share its existing catalog cache.
        let provider = MockProvider::new(vec![]).with_models(&["claude-opus-4"]);
        let calls = provider.list_models_calls();
        let mut agent = agent_with(Box::new(provider));
        let (out, err) = repl(
            b"/model anthropic:\n/model anthropic:claude-opus-4\n/model anthropic:\n/model\n",
            &mut agent,
        );
        assert!(err.is_empty());
        assert!(out.contains("(model set to claude-opus-4)\n"));
        assert!(out.contains("(provider: anthropic, model: claude-opus-4)\n"));
        assert_eq!(calls.get(), 1);
        assert_eq!(agent.provider_kind(), ProviderKind::Anthropic);
        assert_eq!(agent.model(), "claude-opus-4");
    }

    #[test]
    fn different_provider_switch_without_key_still_fails_soft() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl_with(
            b"/model openai:gpt-4o\n/model\n",
            &mut agent,
            |_| None,
            build_mock,
        );
        assert_eq!(
            err,
            "error: OPENAI_API_KEY not found in environment or .env file (provider unchanged)\n"
        );
        assert!(out.contains(&format!(
            "(provider: anthropic, model: {})\n",
            crate::TEST_MODEL
        )));
        assert_eq!(agent.provider_kind(), ProviderKind::Anthropic);
        assert_eq!(agent.model(), crate::TEST_MODEL);
    }

    #[test]
    fn same_provider_prefix_skips_resolution_and_rebuild_and_preserves_effort() {
        // The later cross-provider switch covers both callbacks exactly once.
        // A needless resolve or build for the first switch would make either
        // count two, while the intervening report pins effort preservation.
        let resolves = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let builds = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let resolver_count = Arc::clone(&resolves);
        let builder_count = Arc::clone(&builds);
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl_with(
            b"/effort high\n/model anthropic:another-model\n/effort\n/model openai:gpt-4o\n",
            &mut agent,
            move |env| {
                resolver_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(format!("key-for-{env}"))
            },
            move |_, _| {
                builder_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Box::new(MockProvider::new(vec![]))
            },
        );
        assert!(out.contains("(model set to another-model)\n"));
        assert!(out.contains("(effort: high)\n"));
        assert!(out.contains("(provider set to openai, model set to gpt-4o; effort cleared)\n"));
        assert_eq!(resolves.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(builds.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(agent.provider_kind(), ProviderKind::Openai);
        assert_eq!(agent.model(), "gpt-4o");
        assert!(agent.effort().is_none());
        assert!(err.is_empty());
    }

    // ── switch-time catalog warning (5c) ──

    #[test]
    fn unlisted_id_warns_from_the_cached_catalog_and_switches_anyway() {
        // The listing primes the active provider's cache; the switch to an
        // absent id then draws the soft warning — and still goes through,
        // because ids stay unvalidated.
        let mut agent = agent_with(Box::new(
            MockProvider::new(vec![]).with_models(&["claude-a", "claude-b"]),
        ));
        let (out, err) = repl(b"/model anthropic:\n/model bogus\n", &mut agent);
        assert_eq!(
            err,
            "warning: bogus is not in anthropic's listed models (switching anyway)\n"
        );
        assert!(out.contains("(model set to bogus)\n"));
        assert_eq!(agent.model(), "bogus");
    }

    #[test]
    fn listed_id_switches_without_warning() {
        let mut agent = agent_with(Box::new(
            MockProvider::new(vec![]).with_models(&["claude-a", "claude-b"]),
        ));
        let (out, err) = repl(b"/model anthropic:\n/model claude-b\n", &mut agent);
        assert!(err.is_empty());
        assert!(out.contains("(model set to claude-b)\n"));
    }

    #[test]
    fn uncached_catalog_stays_silent_and_triggers_no_fetch() {
        // No listing or completion has primed the cache, so the warning
        // check has no opinion — and, pinned by the call counter, it must
        // not reach for one: the check never fetches.
        let provider = MockProvider::new(vec![]).with_models(&["claude-a"]);
        let calls = provider.list_models_calls();
        let mut agent = agent_with(Box::new(provider));
        let (out, err) = repl(b"/model bogus\n", &mut agent);
        assert!(err.is_empty());
        assert_eq!(calls.get(), 0);
        assert!(out.contains("(model set to bogus)\n"));
    }

    #[test]
    fn empty_cached_catalog_reads_as_unknown_and_stays_silent() {
        // A failed fetch caches as empty (the fail-soft contract). Treating
        // that as a real catalog would warn on every id because the network
        // was down once — so empty means "unknown", not "nothing exists".
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl(b"/model anthropic:\n/model bogus\n", &mut agent);
        assert!(out.contains("(anthropic: no models listed)\n")); // cached empty
        assert!(err.is_empty());
    }

    #[test]
    fn curation_hidden_ids_do_not_warn() {
        // The check reads the raw catalog: a dated snapshot and a non-chat
        // id are hidden from the listing but really listed — switching to
        // either must stay quiet.
        let mut agent = agent_with(Box::new(MockProvider::new(vec![]).with_models(&[
            "claude-x",
            "claude-x-2026-06-01",
            "whisper-1",
        ])));
        let (_, err) = repl(
            b"/model anthropic:\n/model claude-x-2026-06-01\n/model whisper-1\n",
            &mut agent,
        );
        assert!(err.is_empty());
    }

    #[test]
    fn unlisted_id_warns_for_the_provider_prefixed_form_too() {
        // `/model openai:` primes the non-active provider's session cache;
        // the prefixed switch checks it before `set_provider` drops the
        // agent-side one — warning on stderr, switch still through.
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl_with(
            b"/model openai:\n/model openai:bogus\n",
            &mut agent,
            test_resolver,
            |_, _| Box::new(MockProvider::new(vec![]).with_models(&["gpt-a"])),
        );
        assert_eq!(
            err,
            "warning: bogus is not in openai's listed models (switching anyway)\n"
        );
        assert!(out.contains("(provider set to openai, model set to bogus)\n"));
        assert_eq!(agent.provider_kind(), ProviderKind::Openai);
        assert_eq!(agent.model(), "bogus");
    }

    #[test]
    fn listed_id_in_the_prefixed_form_switches_without_warning() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl_with(
            b"/model openai:\n/model openai:gpt-a\n",
            &mut agent,
            test_resolver,
            |_, _| Box::new(MockProvider::new(vec![]).with_models(&["gpt-a"])),
        );
        assert!(err.is_empty());
        assert!(out.contains("(provider set to openai, model set to gpt-a)\n"));
    }

    #[test]
    fn prompt_streams_the_reply_to_out() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![text_stream("hello!")])));
        let (out, err) = repl(b"hi\n", &mut agent);
        assert!(out.contains("hello!\n"));
        assert!(err.is_empty());
        // The loop re-prompts after the turn completes.
        assert_eq!(out.matches("> ").count(), 2);
    }

    #[test]
    fn agent_error_goes_to_stderr_and_the_loop_continues() {
        let mut agent = agent_with(Box::new(ErrProvider));
        let (out, err) = repl(b"hi\n", &mut agent);
        assert_eq!(err, "error: IO error: boom\n");
        // The failed turn does not kill the session — it re-prompts.
        assert_eq!(out.matches("> ").count(), 2);
    }

    // ── turn cancellation ──

    #[test]
    fn request_cancel_raises_the_flag_and_reports_a_pending_one() {
        let flag = AtomicBool::new(false);
        // The first Ctrl-C: no cancellation pending — just raise the flag.
        assert!(!request_cancel(&flag));
        assert!(flag.load(Ordering::Relaxed));
        // The second, with one still pending: the force-exit signal.
        assert!(request_cancel(&flag));
    }

    /// An agent whose one tool call raises its own cancellation flag mid-run
    /// — the REPL-level stand-in for a Ctrl-C landing during a turn.
    fn cancelling_agent() -> Agent {
        let flag = Arc::new(AtomicBool::new(false));
        let handle = Arc::clone(&flag);
        let tool = TestTool::new("halt", "").with_run(move |_| {
            handle.store(true, Ordering::Relaxed);
            Ok("ok".to_string())
        });
        let stream = vec![
            StreamDelta::MessageStart {
                usage: Usage::default(),
            },
            StreamDelta::ToolUseStart {
                index: 0,
                id: "t1".to_string(),
                name: "halt".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: Usage::default(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![stream])),
            AgentConfig {
                provider_kind: ProviderKind::Anthropic,
                model: crate::TEST_MODEL.to_string(),
                max_tokens: 64,
                system: None,
                context_token_limit: 100_000,
                effort: None,
                max_turns: crate::agent::MAX_TURNS,
            },
            vec![Box::new(tool)],
        );
        agent.set_cancel_flag(flag);
        agent
    }

    #[test]
    fn cancelled_turn_prints_a_quiet_notice_and_reprompts() {
        // Not an error dump: the notice goes to `out`, stderr stays empty,
        // and the loop survives to prompt again.
        let mut agent = cancelling_agent();
        let (out, err) = repl(b"hi\n", &mut agent);
        assert!(out.contains("(turn cancelled)\n"));
        assert!(err.is_empty());
        assert_eq!(out.matches("> ").count(), 2);
    }

    #[test]
    fn cancelled_turn_autosaves_the_rolled_back_state() {
        // A cancelled turn persists exactly like a failed one: the agent
        // rolled back to the last clean state, and that state is what must
        // survive a crash that follows.
        let mut agent = cancelling_agent();
        let (saved, sink) = recording_sink();
        let (out, err) = repl_persisting(b"hi\n", &mut agent, sink);
        assert!(out.contains("(turn cancelled)\n"));
        assert!(err.is_empty());
        let saved = saved.borrow();
        assert_eq!(saved.len(), 1);
        assert!(saved[0].messages.is_empty()); // rolled back, not half a turn
    }

    // ── session autosave ──

    /// The snapshots a [`recording_sink`] collects, observed through a shared
    /// handle like [`MockProvider::send_log`].
    type SavedSessions = std::rc::Rc<std::cell::RefCell<Vec<Session>>>;

    /// A recording sink and the snapshots it collects.
    fn recording_sink() -> (SavedSessions, impl FnMut(&Session) -> Result<(), String>) {
        let saved = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let handle = std::rc::Rc::clone(&saved);
        (saved, move |session: &Session| {
            handle.borrow_mut().push(session.clone());
            Ok(())
        })
    }

    #[test]
    fn autosave_fires_after_a_successful_turn() {
        let mut agent = agent_with(Box::new(MockProvider::new(vec![text_stream("hello!")])));
        let (saved, sink) = recording_sink();
        let (_, err) = repl_persisting(b"hi\n", &mut agent, sink);
        assert!(err.is_empty());
        // One snapshot, carrying the completed exchange.
        let saved = saved.borrow();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].messages.len(), 2);
    }

    #[test]
    fn autosave_fires_after_a_failed_turn() {
        // A failed turn still persists: the agent rolled its history back to
        // the last clean state, and that state is what must survive a crash
        // that follows.
        let mut agent = agent_with(Box::new(ErrProvider));
        let (saved, sink) = recording_sink();
        let (_, err) = repl_persisting(b"hi\n", &mut agent, sink);
        assert!(err.contains("error: IO error: boom\n"));
        let saved = saved.borrow();
        assert_eq!(saved.len(), 1);
        assert!(saved[0].messages.is_empty()); // rolled back, not half a turn
    }

    #[test]
    fn autosave_fires_after_clear_with_emptied_state() {
        // /clear persists too — otherwise a restart would reload the very
        // history the operator just cleared.
        let mut agent = agent_with(Box::new(MockProvider::new(vec![text_stream("hello!")])));
        let (saved, sink) = recording_sink();
        repl_persisting(b"hi\n/clear\n", &mut agent, sink);
        let saved = saved.borrow();
        assert_eq!(saved.len(), 2);
        assert_eq!(saved[0].messages.len(), 2);
        assert!(saved[1].messages.is_empty());
        assert!(saved[1].compacted_summary.is_none());
        assert_eq!(saved[1].last_input_tokens, 0);
    }

    #[test]
    fn autosave_skips_non_conversation_commands() {
        // /model reports and switches touch no conversation state — writing
        // the same session back on every keystroke-level command would be
        // pointless disk churn.
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (saved, sink) = recording_sink();
        repl_persisting(b"/model\n/model gpt-4o\n", &mut agent, sink);
        assert!(saved.borrow().is_empty());
    }

    #[test]
    fn failed_autosave_warns_and_the_loop_survives() {
        // Save fails soft (the 2c policy): a sink failure — disk full,
        // permissions — warns on stderr and the session keeps running; the
        // next turn still reaches the agent.
        let mut agent = agent_with(Box::new(MockProvider::new(vec![
            text_stream("first"),
            text_stream("second"),
        ])));
        let (out, err) =
            repl_persisting(b"hi\nagain\n", &mut agent, |_| Err("disk full".to_string()));
        assert_eq!(
            err.matches("warning: session not saved: disk full\n")
                .count(),
            2
        );
        assert!(out.contains("first\n"));
        assert!(out.contains("second\n"));
    }

    #[test]
    fn autosave_writes_a_loadable_file_and_clear_empties_it() {
        // End to end through a real file: the production-shaped sink
        // (`agent.session().save(path)`) leaves a file Session::load reads
        // back, and /clear rewrites it empty.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");
        let path = path.to_str().unwrap();

        let mut agent = agent_with(Box::new(MockProvider::new(vec![text_stream("hello!")])));
        repl_persisting(b"hi\n", &mut agent, |session| session.save(path));
        let restored = Session::load(path).unwrap();
        assert_eq!(restored.messages.len(), 2);

        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        agent.restore(restored);
        repl_persisting(b"/clear\n", &mut agent, |session| session.save(path));
        assert!(Session::load(path).unwrap().messages.is_empty());
    }

    // ── default autosave: sink selection & auto-naming ──

    /// A `SystemTime` `secs` after the Unix epoch — the fixtures for the
    /// hand-rolled UTC calendar conversion.
    fn utc(secs: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs)
    }

    #[test]
    fn timestamp_formats_known_utc_instants() {
        // Epoch, a plain date, a time-of-day, a leap day, both month-length
        // boundaries, and a year-end that crosses the era arithmetic — the
        // known answers the from-scratch civil conversion must reproduce.
        assert_eq!(
            format_utc_timestamp(SystemTime::UNIX_EPOCH),
            "19700101-000000"
        );
        assert_eq!(format_utc_timestamp(utc(1_609_459_200)), "20210101-000000");
        assert_eq!(format_utc_timestamp(utc(1_609_462_861)), "20210101-010101");
        assert_eq!(format_utc_timestamp(utc(1_582_934_400)), "20200229-000000"); // leap day
        assert_eq!(format_utc_timestamp(utc(1_614_556_800)), "20210301-000000"); // after Feb 28
        assert_eq!(format_utc_timestamp(utc(1_619_740_800)), "20210430-000000"); // 30-day month end
        assert_eq!(format_utc_timestamp(utc(1_640_995_199)), "20211231-235959"); // year end
    }

    #[test]
    fn timestamp_clamps_a_pre_epoch_instant_to_the_epoch() {
        // Only reachable from a caller-supplied time, never from `now()`: a
        // pre-epoch instant clamps rather than underflowing the day math.
        let before = SystemTime::UNIX_EPOCH - std::time::Duration::from_secs(1);
        assert_eq!(format_utc_timestamp(before), "19700101-000000");
    }

    #[test]
    fn auto_base_is_the_timestamp() {
        assert_eq!(
            auto_session_base(SystemTime::UNIX_EPOCH),
            "auto-19700101-000000"
        );
    }

    #[test]
    fn sink_writes_the_session_file_and_leaves_the_store_untouched() {
        // The set+present cell: the project override wins and no auto-save
        // file is written into the store.
        let (_home, _project, store) = temp_store();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixed.json");
        let path = path.to_str().unwrap();
        let mut sink = persist_sink(Some(path.to_string()), Some(&store));
        sink(&session_of(vec![user_msg("hi")])).unwrap();
        assert_eq!(Session::load(path).unwrap().messages.len(), 1);
        assert!(store.list().is_empty());
    }

    #[test]
    fn sink_writes_the_session_file_without_a_store() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixed.json");
        let path = path.to_str().unwrap();
        let mut sink = persist_sink(Some(path.to_string()), None);
        sink(&session_of(vec![user_msg("hi")])).unwrap();
        assert_eq!(Session::load(path).unwrap().messages.len(), 1);
    }

    #[test]
    fn sink_is_a_noop_without_a_session_file_or_store() {
        // No `$HOME`, no override — today's no-persistence behavior.
        let mut sink = persist_sink(None, None);
        assert!(sink(&session_of(vec![user_msg("hi")])).is_ok());
    }

    #[test]
    fn sink_autosaves_into_the_store_by_default() {
        let (_home, _project, store) = temp_store();
        let mut sink = persist_sink(None, Some(&store));
        sink(&session_of(vec![user_msg("hi")])).unwrap();
        let saves = store.list();
        assert_eq!(saves.len(), 1);
        let name = &saves[0].name;
        assert!(name.starts_with("auto-"), "got: {name}");
    }

    #[test]
    fn sink_rolls_one_file_across_a_sessions_turns() {
        // No file before the first write; then one file, named once and rolled
        // in place, holding the latest content.
        let (_home, _project, store) = temp_store();
        let mut sink = persist_sink(None, Some(&store));
        assert!(store.list().is_empty());
        sink(&session_of(vec![user_msg("one")])).unwrap();
        sink(&session_of(vec![user_msg("one"), user_msg("two")])).unwrap();
        let saves = store.list();
        assert_eq!(saves.len(), 1);
        assert_eq!(store.load(&saves[0].name).unwrap().messages.len(), 2);
    }

    #[test]
    fn failed_first_auto_save_retries_allocation_without_caching_a_name() {
        let (home, _project, store) = temp_store();
        let blocker = home.path().join(".omega-system");
        std::fs::write(&blocker, "not a directory").unwrap();
        let mut name = None;
        let first = session_of(vec![user_msg("failed")]);

        let err = persist_auto(&store, &mut name, SystemTime::UNIX_EPOCH, &first).unwrap_err();

        assert!(
            err.contains("cannot create session directory"),
            "got: {err}"
        );
        assert!(name.is_none());
        std::fs::remove_file(blocker).unwrap();
        let occupied = session_of(vec![user_msg("occupied")]);
        store.save("auto-19700101-000000", &occupied).unwrap();
        let retried = session_of(vec![user_msg("retried")]);
        persist_auto(&store, &mut name, SystemTime::UNIX_EPOCH, &retried).unwrap();

        assert_eq!(name.as_deref(), Some("auto-19700101-000000-2"));
        assert_eq!(store.load("auto-19700101-000000").unwrap(), occupied);
        assert_eq!(store.load("auto-19700101-000000-2").unwrap(), retried);
    }

    #[test]
    fn a_new_session_opens_a_second_file_leaving_the_first() {
        // Two sinks (two sessions) against one store: two distinct files, the
        // first never overwritten — the accumulating-history invariant. When
        // both open in the same second the collision check appends a suffix.
        let (_home, _project, store) = temp_store();
        let mut first = persist_sink(None, Some(&store));
        first(&session_of(vec![user_msg("first")])).unwrap();
        let mut second = persist_sink(None, Some(&store));
        second(&session_of(vec![user_msg("second"), user_msg("more")])).unwrap();
        let saves = store.list();
        assert_eq!(saves.len(), 2);
        let names: Vec<&String> = saves.iter().map(|e| &e.name).collect();
        assert_ne!(names[0], names[1]);
        // The single-message session is still on disk under its own name.
        assert!(
            saves
                .iter()
                .any(|e| store.load(&e.name).unwrap().messages == vec![user_msg("first")])
        );
    }

    #[test]
    fn a_turn_autosaves_into_the_store_end_to_end() {
        // Through the loop: a completed turn lands in the store with no config,
        // and bare `--resume` restores it.
        use crate::cli::{CliArgs, Resume, select_initial_session};
        let (_home, _project, store) = temp_store();
        let mut agent = agent_with(Box::new(MockProvider::new(vec![text_stream("hello!")])));
        let (_, err) = repl_persisting(b"hi\n", &mut agent, persist_sink(None, Some(&store)));
        assert!(err.is_empty(), "got: {err}");
        assert_eq!(store.list().len(), 1);
        let restored = select_initial_session(
            &CliArgs {
                resume: Some(Resume::Last),
            },
            Some(&store),
            None,
            false,
            &mut std::io::empty(),
            &mut std::io::sink(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(restored.messages.len(), 2);
    }

    #[test]
    fn load_autosaves_the_restored_state_into_the_current_auto_file() {
        // `/load snap` mirrors the restored state into *this* session's auto
        // file while the loaded save stays untouched — forking stays free.
        let (_home, _project, store) = temp_store();
        store
            .save("snap", &session_of(vec![user_msg("restored")]))
            .unwrap();
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let providers = ProviderFactory::from_fns(build_mock, test_resolver);
        let mut out = Vec::new();
        let mut err = Vec::new();
        run_repl(
            BufReader::new(&b"/load snap\n"[..]),
            &mut out,
            &mut err,
            &mut agent,
            &providers,
            &[],
            Some(&store),
            persist_sink(None, Some(&store)),
        );
        let saves = store.list();
        let auto = saves.iter().find(|e| e.name.starts_with("auto-")).unwrap();
        assert_eq!(
            store.load(&auto.name).unwrap().messages,
            vec![user_msg("restored")]
        );
        // The source save is present and unchanged.
        assert_eq!(
            store.load("snap").unwrap().messages,
            vec![user_msg("restored")]
        );
    }

    #[test]
    fn clear_rolls_the_current_auto_file_to_an_empty_session() {
        // `/clear` mirrors the emptied conversation into the current file; the
        // restart it protects against reloads nothing.
        use crate::cli::{CliArgs, Resume, select_initial_session};
        let (_home, _project, store) = temp_store();
        let mut agent = agent_with(Box::new(MockProvider::new(vec![text_stream("hello!")])));
        let providers = ProviderFactory::from_fns(build_mock, test_resolver);
        let mut out = Vec::new();
        let mut err = Vec::new();
        run_repl(
            BufReader::new(&b"hi\n/clear\n"[..]),
            &mut out,
            &mut err,
            &mut agent,
            &providers,
            &[],
            Some(&store),
            persist_sink(None, Some(&store)),
        );
        let saves = store.list();
        assert_eq!(saves.len(), 1);
        assert!(store.load(&saves[0].name).unwrap().messages.is_empty());
        let restored = select_initial_session(
            &CliArgs {
                resume: Some(Resume::Last),
            },
            Some(&store),
            None,
            false,
            &mut std::io::empty(),
            &mut std::io::sink(),
        )
        .unwrap()
        .unwrap();
        assert!(restored.messages.is_empty());
    }

    #[test]
    fn a_failed_store_autosave_warns_and_the_loop_survives() {
        // A regular file where the store directory must be makes every store
        // write fail; the factory surfaces the error, autosave warns, and the
        // next turn still reaches the agent.
        let home = tempfile::tempdir().unwrap();
        std::fs::write(home.path().join(".omega-system"), "x").unwrap();
        let project = tempfile::tempdir().unwrap();
        let store = SessionStore::new(home.path(), project.path());
        let mut agent = agent_with(Box::new(MockProvider::new(vec![
            text_stream("first"),
            text_stream("second"),
        ])));
        let (out, err) =
            repl_persisting(b"hi\nagain\n", &mut agent, persist_sink(None, Some(&store)));
        assert!(err.contains("warning: session not saved:"), "got: {err}");
        assert!(
            out.contains("first") && out.contains("second"),
            "got: {out}"
        );
    }

    // ── named session store (/save, /load, /sessions) ──

    #[test]
    fn save_then_sessions_lists_the_save() {
        let (_home, _project, store) = temp_store();
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl_with_store(b"/save mysnap\n/sessions\n", &mut agent, &store);
        assert!(out.contains("(saved session 'mysnap')\n"), "got: {out}");
        // The row leads with the name, then the metadata columns (age, size).
        assert!(
            out.contains("(saved sessions, newest first:)\n  mysnap"),
            "got: {out}"
        );
        assert!(out.contains(" B\n") || out.contains(" KB\n"), "got: {out}");
        assert!(err.is_empty(), "got: {err}");
    }

    #[test]
    fn sessions_reports_none_when_empty() {
        let (_home, _project, store) = temp_store();
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl_with_store(b"/sessions\n", &mut agent, &store);
        assert!(out.contains("(no saved sessions)\n"), "got: {out}");
    }

    #[test]
    fn bare_save_and_load_report_usage() {
        let (_home, _project, store) = temp_store();
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl_with_store(b"/save\n/load\n", &mut agent, &store);
        assert!(out.contains("(usage: /save <name>)\n"), "got: {out}");
        assert!(out.contains("(usage: /load <name>)\n"), "got: {out}");
    }

    #[test]
    fn save_invalid_name_fails_soft() {
        let (_home, _project, store) = temp_store();
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (_, err) = repl_with_store(b"/save ../evil\n", &mut agent, &store);
        assert!(err.contains("error: invalid session name"), "got: {err}");
    }

    #[test]
    fn load_replaces_the_live_conversation() {
        // A saved snapshot restores over whatever the agent currently holds,
        // and the arm says so.
        let (_home, _project, store) = temp_store();
        let mut snap = session_of(vec![user_msg("restored")]);
        snap.compacted_summary = Some("older summary".to_string());
        snap.last_input_tokens = 9;
        store.save("snap", &snap).unwrap();

        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        agent.restore(session_of(vec![user_msg("live")]));
        let (out, err) = repl_with_store(b"/load snap\n", &mut agent, &store);
        assert!(
            out.contains("(loaded session 'snap', replacing the current conversation)\n"),
            "got: {out}"
        );
        assert_eq!(agent.session(), snap);
        assert!(err.is_empty(), "got: {err}");
    }

    #[test]
    fn load_autosaves_the_restored_state() {
        // After a load, the restored conversation is persisted to the fixed
        // session_file through the autosave sink, exactly as a turn would —
        // so a crash right after a load cannot resurrect the discarded one.
        let (_home, _project, store) = temp_store();
        let snap = session_of(vec![user_msg("restored")]);
        store.save("snap", &snap).unwrap();

        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (saved, sink) = recording_sink();
        let providers = ProviderFactory::from_fns(build_mock, test_resolver);
        let mut out = Vec::new();
        let mut err = Vec::new();
        run_repl(
            BufReader::new(&b"/load snap\n"[..]),
            &mut out,
            &mut err,
            &mut agent,
            &providers,
            &[],
            Some(&store),
            sink,
        );
        let saved = saved.borrow();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0], snap);
    }

    #[test]
    fn load_unknown_name_fails_soft_and_lists_available() {
        // A missing name reports softly and drops the operator onto the real
        // choices rather than the raw file-not-found.
        let (_home, _project, store) = temp_store();
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl_with_store(b"/save real\n/load ghost\n", &mut agent, &store);
        assert!(
            err.contains("error: no saved session named 'ghost'\n"),
            "got: {err}"
        );
        assert!(
            out.contains("(saved sessions, newest first:)\n  real"),
            "got: {out}"
        );
    }

    #[test]
    fn sessions_delete_removes_a_save() {
        let (_home, _project, store) = temp_store();
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) = repl_with_store(
            b"/save doomed\n/sessions delete doomed\n/sessions\n",
            &mut agent,
            &store,
        );
        assert!(out.contains("(deleted session 'doomed')\n"), "got: {out}");
        // Gone from the subsequent listing.
        assert!(out.contains("(no saved sessions)\n"), "got: {out}");
        assert!(err.is_empty(), "got: {err}");
    }

    #[test]
    fn sessions_delete_missing_fails_soft_and_lists_available() {
        // A missing name reports softly and lists the real saves, mirroring
        // `/load`.
        let (_home, _project, store) = temp_store();
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, err) =
            repl_with_store(b"/save real\n/sessions delete ghost\n", &mut agent, &store);
        assert!(
            err.contains("error: no saved session named 'ghost'\n"),
            "got: {err}"
        );
        assert!(
            out.contains("(saved sessions, newest first:)\n  real"),
            "got: {out}"
        );
    }

    #[test]
    fn bare_sessions_delete_reports_usage() {
        let (_home, _project, store) = temp_store();
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (out, _) = repl_with_store(b"/sessions delete\n", &mut agent, &store);
        assert!(
            out.contains("(usage: /sessions delete <name>)\n"),
            "got: {out}"
        );
    }

    #[test]
    fn sessions_delete_without_a_store_reports_unavailable() {
        // No `$HOME` → no store: a named delete reports unavailable on the
        // warning channel, like `/save` / `/load` / `/sessions`.
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (_, err) = repl(b"/sessions delete foo\n", &mut agent);
        assert!(
            err.contains(
                "error: session store unavailable (no home directory to store sessions under)"
            ),
            "got: {err}"
        );
    }

    #[test]
    fn save_load_sessions_report_unavailable_without_a_store() {
        // No `$HOME` → no store: all three commands report unavailable on the
        // warning channel, and the loop keeps running. `repl` passes `None`.
        let mut agent = agent_with(Box::new(MockProvider::new(vec![])));
        let (_, err) = repl(b"/save foo\n/load foo\n/sessions\n", &mut agent);
        assert_eq!(
            err.matches(
                "error: session store unavailable (no home directory to store sessions under)"
            )
            .count(),
            3,
            "got: {err}"
        );
    }
}
