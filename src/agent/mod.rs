use crate::provider::{ApiError, Provider, ProviderKind};
use crate::tools::ToolDef;
use crate::turn::{Block, Role, StopReason, TurnMessage, Usage};
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

mod budget;
mod concurrency;
mod confirm;
mod confirm_live;
mod context;
mod execution;
#[cfg(test)]
mod fixtures;
mod stream;
#[cfg(test)]
mod tests_live;

pub use budget::BudgetLedger;
pub use concurrency::Concurrency;
pub use confirm::ConfirmPolicy;
pub use confirm_live::interactive_confirm;

/// The default tool-loop round budget, applied when a config sets no
/// `max_turns`. Public so [`crate::config`] can source its default from the one
/// definition, keeping the constant and the config default from drifting.
pub const MAX_TURNS: u32 = 10;

/// Prefix stamped on a subagent's salvaged reply when it ran out of tool
/// rounds. It tells the orchestrator the findings are partial — the child
/// answered from what it had gathered, not from a completed investigation.
const SALVAGE_MARKER: &str = "[turn limit reached — partial findings]";

/// The wrap-up instruction coalesced into the trailing tool-result message
/// when a nested child hits the turn wall. It steers the one extra round-trip
/// toward a text summary; the tools stay defined on the wire (the contract
/// requires it once history carries `tool_use`), so the instruction, not a
/// stripped tool list, is what asks the model to stop calling them.
const WRAP_UP_INSTRUCTION: &str = "\
You have reached the maximum number of tool-use rounds and cannot call any more \
tools. Do not attempt another tool call. Summarize your findings so far as \
plain text: what you have established, any partial conclusions, and what remains \
unresolved.";

/// Join a message's `Text` blocks into one string, ignoring every other block
/// kind. The end-of-turn reply, the wrap-up reply, and the salvage fallback
/// all reduce a block list to its text the same way.
fn text_of(blocks: &[Block]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            Block::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

// ── Error type ──

#[derive(Debug)]
pub enum AgentError {
    Api(ApiError),
    /// The tool loop ran its full round budget without the model ending the
    /// turn. Carries the resolved limit (the config's `max_turns`, defaulting
    /// to [`MAX_TURNS`]) so the message reports the number actually enforced,
    /// not a hardcoded constant that a raised limit would contradict.
    TurnLimitExceeded(u32),
    /// The operator cancelled the turn (Ctrl-C). A deliberate act, not a
    /// failure: the REPL renders it as a short quiet notice, never an error
    /// dump. The turn's history rolls back exactly like a failed turn's,
    /// including the side-effect carve-out.
    Cancelled,
    /// The confirmation circuit breaker tripped: [`CONFIRM_BREAKER_LIMIT`]
    /// consecutive *automated* denials (judge or policy floor — never a
    /// human's "no") in one turn. Under `allow`/`judge` nobody is at the
    /// terminal to notice a model re-trying denied variants forever, so the
    /// loop stops itself instead of burning provider calls against a wall.
    ConfirmBreaker,
    /// Non-reducible prompt content exceeds the conservative request budget.
    ContextLimit(u32),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::Api(e) => write!(f, "{e}"),
            AgentError::TurnLimitExceeded(limit) => {
                write!(f, "agent exceeded maximum of {limit} turns")
            }
            AgentError::Cancelled => write!(f, "turn cancelled"),
            AgentError::ContextLimit(limit) => write!(
                f,
                "request cannot fit context_token_limit {limit}; use a shorter prompt, smaller read_file ranges, or /clear"
            ),
            AgentError::ConfirmBreaker => write!(
                f,
                "confirmation circuit breaker: {CONFIRM_BREAKER_LIMIT} consecutive automated \
                 denials — stopping the turn"
            ),
        }
    }
}

impl From<ApiError> for AgentError {
    fn from(e: ApiError) -> Self {
        AgentError::Api(e)
    }
}

// ── Agent config ──

pub struct AgentConfig {
    /// Which provider the agent's `Box<dyn Provider>` talks to. Kept beside
    /// `model` — the pair changes together on a `/model <provider>:<model>`
    /// switch — and held by the agent, not the REPL, so the agent is the
    /// single source of provider identity and the bare-`/model` report can
    /// never read a stale copy.
    pub provider_kind: ProviderKind,
    pub model: String,
    pub max_tokens: u32,
    pub system: Option<String>,
    /// The context-window size the compaction guard defends (see
    /// [`crate::config::Config::context_token_limit`]).
    pub context_token_limit: u32,
    /// The tool-loop round budget for this agent — [`Config::max_turns`]
    /// resolved (a delegated child folds in its profile override before this).
    /// `run_loop` runs at most this many provider round-trips, then stops with
    /// [`AgentError::TurnLimitExceeded`] carrying this value.
    ///
    /// [`Config::max_turns`]: crate::config::Config::max_turns
    pub max_turns: u32,
    /// Optional reasoning-effort level applied to conversation turns. Opaque
    /// and validated provider-side, like `model` (see
    /// [`crate::turn::TurnRequest::effort`]). `None` — the default — sends no
    /// effort field, so the request stays byte-identical to today's.
    pub effort: Option<String>,
}

// ── Agent ──

/// Which end of the shared cancel flag's lifecycle a `run` owns. The
/// top-level turn *owns* it — resets it at the start, consumes it at the end;
/// a nested child turn (the `task` tool) merely *observes* it, so a Ctrl-C
/// that stopped the child still reaches the parent's owning run. One run path
/// serves both; this is the only thing that differs between them.
enum CancelRole {
    Owner,
    Observer,
}

/// Consecutive automated confirmation denials (judge, policy floor) that end
/// the turn. Three is the number shipping agents converge on for their own
/// breakers; a human denial resets nothing here because it never increments —
/// only unattended denials loop.
const CONFIRM_BREAKER_LIMIT: u32 = 3;

/// Thousands-grouped decimal for the usage meta line: `12345` → `"12,345"`.
/// Token counts and context limits run five to seven digits, where ungrouped
/// numbers stop being readable at a glance.
fn group_thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Core of the confirmation gate, seamed over an injected reader/writer pair
/// so the approve/deny logic carries hermetic coverage; the real stdin/stderr
/// wiring is the thin [`confirm_live::interactive_confirm`] wrapper.
/// Fail-closed: an unreadable input (EOF, closed pipe, read error) denies — a
/// headless embedding that inherits the interactive default gets denial, not
/// silent approval. The gate being permanent means "always gated", not
/// "always prompts".
fn confirm_with(input: &mut dyn BufRead, err: &mut dyn Write, summary: &str) -> bool {
    let displayed = if summary.contains('\n') {
        write!(err, "{summary}\nAllow this change? [y/N] ")
    } else {
        write!(err, "Allow {summary}? [y/N] ")
    };
    if displayed.and_then(|()| err.flush()).is_err() {
        return false;
    }

    let mut line = String::new();
    if input.read_line(&mut line).is_err() {
        return false;
    }
    matches!(line.trim(), "y" | "Y" | "yes" | "YES")
}

pub struct Agent {
    provider: Box<dyn Provider>,
    config: AgentConfig,
    tools: Vec<Box<dyn ToolDef>>,
    messages: Vec<TurnMessage>,
    /// Process-wide tool-call budget by cost tier. Owned by `Arc` inside, so
    /// sharing it (the `task` tool, via [`Agent::set_budget_ledger`])
    /// clones the handle rather than the counters — every clone draws against
    /// the same ceiling. The top-level agent gets its own fresh ledger from
    /// [`Agent::new`].
    budget: BudgetLedger,
    /// Process-wide fan-out concurrency cap. Owned by `Arc` inside, so sharing
    /// it (the `task` tool, via [`Agent::set_concurrency`]) clones the handle
    /// rather than the count — every clone contends for the same permits, so a
    /// parent's fan-out and its children's fan-outs draw from one ceiling. The
    /// top-level agent gets its own fresh pool from [`Agent::new`].
    concurrency: Concurrency,
    /// The confirmation policy behind the dangerous-tool gate:
    /// `ask` prompts, `allow` waves the human prompt, `judge` adjudicates.
    /// Shared by clone with the `task` tool, so children inherit it and a
    /// `/confirm` switch reaches both.
    confirm: ConfirmPolicy,
    /// Consecutive automated confirmation denials this turn — the circuit
    /// breaker's count. Reset at turn start and on any approval; a human
    /// denial also resets it (someone is present and answering).
    auto_denials: u32,
    /// Whether a side-effecting tool ran during the current `run`. Set in
    /// `execute_tools`, reset at the top of `run`. When true, a failed turn's
    /// history is kept verbatim rather than rolled back, so the model's record
    /// never diverges from a mutation that already touched disk.
    side_effected_this_turn: bool,
    /// The last measured prompt size: `input_tokens` plus both cache counters
    /// from the most recent completed response. The only token signal is
    /// post-response, so this is the context-compaction guard's input — one
    /// turn stale by construction. Bare `input_tokens` would undercount a
    /// cached conversation, hence the sum. Zero until a turn completes.
    last_input_tokens: u32,
    /// The rolling summary of compacted-away history. `build_request` folds it
    /// into the system prompt — the turn model has no system-summary role, and
    /// a synthetic `User`/`Assistant` message would break the alternating-role
    /// sequence — so the message vector stays clean. Cumulative: each
    /// compaction folds the previous summary into the next one.
    compacted_summary: Option<String>,
    /// The provider's model ids, fetched lazily by [`Agent::list_models_cached`]
    /// and held for the session. `None` until the first request; `set_provider`
    /// resets it (the list is provider state), `clear` does not (it is not
    /// conversation state).
    models_cache: Option<Vec<String>>,
    /// Whether meta lines are dimmed with ANSI faint. Wired from main's TTY
    /// check: styling belongs on an interactive terminal, never in a piped
    /// transcript (or a test buffer). Defaults to plain.
    styled: bool,
    /// The turn-cancellation flag, checked at `run`'s natural seams: before
    /// each provider request in the tool loop, between stream events, and
    /// between tool executions. Injected via [`Agent::set_cancel_flag`] so
    /// the hermetic core never touches signals — production shares one flag
    /// between the SIGINT handler (`repl::install_sigint_cancel`) and the
    /// shell tool's deadline loop; tests flip their own. The default (a
    /// private flag nothing else holds) means an un-wired agent never
    /// cancels.
    cancel: Arc<AtomicBool>,
}

impl Agent {
    pub fn new(
        provider: Box<dyn Provider>,
        config: AgentConfig,
        tools: Vec<Box<dyn ToolDef>>,
    ) -> Self {
        Self {
            provider,
            config,
            tools,
            messages: Vec::new(),
            budget: BudgetLedger::new(),
            concurrency: Concurrency::new(),
            confirm: ConfirmPolicy::interactive_default(),
            auto_denials: 0,
            side_effected_this_turn: false,
            last_input_tokens: 0,
            compacted_summary: None,
            models_cache: None,
            styled: false,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Enable or disable ANSI styling of meta lines. Wired from the same
    /// TTY check that selects the raw-mode editor.
    pub fn set_styled(&mut self, styled: bool) {
        self.styled = styled;
    }

    /// Share the cancellation flag [`Agent::run`]'s seams poll. See the
    /// `cancel` field for the wiring; the flag's lifecycle (reset at turn
    /// start, consumed at turn end) is `run`'s.
    pub fn set_cancel_flag(&mut self, flag: Arc<AtomicBool>) {
        self.cancel = flag;
    }

    /// Share a process-wide budget ledger, replacing the fresh one
    /// [`Agent::new`] built. Mirrors [`Agent::set_cancel_flag`]: a child agent
    /// spawned by the `task` tool is wired to the same ledger as
    /// its parent, so parent and child draw against one combined ceiling
    /// instead of each enforcing their own.
    pub fn set_budget_ledger(&mut self, ledger: BudgetLedger) {
        self.budget = ledger;
    }

    /// Share a process-wide fan-out concurrency pool, replacing the fresh one
    /// [`Agent::new`] built. Mirrors [`Agent::set_budget_ledger`]: the `task`
    /// tool hands each child the same pool as its parent, so a batch's fan-out
    /// and its children's fan-outs contend for one set of permits rather than
    /// each spawning up to the cap independently.
    pub fn set_concurrency(&mut self, concurrency: Concurrency) {
        self.concurrency = concurrency;
    }

    /// Replace the confirmation policy [`Agent::new`] defaulted to (the
    /// interactive `ask`). The REPL installs the session policy built from
    /// `config.json`; the `task` tool hands each child its variant — the
    /// parent's policy labeled with the delegating profile for an executor
    /// (bubbling prompts to the parent's TTY), pinned when the profile
    /// carries a `confirm` override, an inert deny for a read-only child.
    pub fn set_confirm_policy(&mut self, confirm: ConfirmPolicy) {
        self.confirm = confirm;
    }

    /// The active confirmation policy — the REPL's `/confirm` reads the mode
    /// and switches it through here, and the `task` tool clones it for
    /// children (sharing the mode cell, so a later switch reaches them).
    pub fn confirm_policy(&self) -> &ConfirmPolicy {
        &self.confirm
    }

    /// Whether a cancellation is pending. Relaxed ordering: the flag is a
    /// lone bool guarding no other data, so there is nothing to order.
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Write one bracketed meta line — tool chatter, token counts, error
    /// notices. Dimmed with ANSI faint when styling is on, so the loop's
    /// bookkeeping reads apart from the model's own streamed text.
    fn meta_line(&self, out: &mut dyn Write, text: std::fmt::Arguments) {
        let _ = if self.styled {
            writeln!(out, "\x1b[2m[{text}]\x1b[0m")
        } else {
            writeln!(out, "[{text}]")
        };
    }

    /// The end-of-turn usage meta line: the final round-trip's token counts,
    /// its cache activity when any was reported, and the measured context
    /// against the compaction guard's ceiling — the number the operator
    /// watches climb toward the threshold. Printed once per *completed* turn
    /// (`run_loop` returns before reaching it on a failure, deliberately: a
    /// failed turn's measurement rolls back with its messages, so reporting
    /// it would describe state the rollback discards). Quiet when the
    /// round-trip carried no usage at all — a provider that reports nothing
    /// gets no fabricated zeros.
    fn usage_line(&self, usage: &Usage, out: &mut dyn Write) {
        if usage.input_tokens == 0 && usage.output_tokens == 0 {
            return;
        }
        let read = usage.cache_read_input_tokens.unwrap_or(0);
        let written = usage.cache_creation_input_tokens.unwrap_or(0);
        let cache = if read > 0 || written > 0 {
            format!(
                " · cache: {} read, {} written",
                group_thousands(read.into()),
                group_thousands(written.into())
            )
        } else {
            String::new()
        };
        // The same sum `run_loop` records as `last_input_tokens` — the
        // compaction guard's measure of the prompt that was just sent.
        // `context_token_limit` is validated non-zero at config load.
        let context = usage.prompt_size();
        let limit = u64::from(self.config.context_token_limit);
        self.meta_line(
            out,
            format_args!(
                "usage: {} in, {} out{cache} · context: {}/{} ({}%)",
                group_thousands(usage.input_tokens.into()),
                group_thousands(usage.output_tokens.into()),
                group_thousands(context),
                group_thousands(limit),
                context * 100 / limit
            ),
        );
    }

    /// Names of the registered tools, in registration order. Backs the REPL's
    /// startup banner.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Run the agent loop for one top-level user turn. Streams the response,
    /// executes any tool calls, and loops until the model stops calling tools
    /// (or the turn limit is reached). Streamed text and tool status lines are
    /// written to `out` as they happen; the final text is returned.
    ///
    /// This is the *owning* run: it resets the cancel flag at the start and
    /// consumes it at the end (see [`CancelRole`]). A nested child turn (the
    /// `task` tool) uses [`Agent::run_nested`] instead, which only observes it.
    pub fn run(&mut self, input: &str, out: &mut dyn Write) -> Result<String, AgentError> {
        self.run_with_role(input, out, CancelRole::Owner)
    }

    /// Drive a nested child turn to completion. Same loop as [`Agent::run`],
    /// but the shared cancel flag is only *observed*, never reset or consumed
    /// (see [`CancelRole`]): a set flag stops the child, yet the parent's
    /// owning run stays the one to reset and consume it, so a Ctrl-C that
    /// interrupted the child still surfaces to the parent. The `task` tool
    /// builds a fresh `Agent` for each delegation and drives it through here.
    pub fn run_nested(&mut self, input: &str, out: &mut dyn Write) -> Result<String, AgentError> {
        self.run_with_role(input, out, CancelRole::Observer)
    }

    /// The single run path shared by the owning [`Agent::run`] and the nested
    /// [`Agent::run_nested`]. Only the cancel-flag lifecycle differs by
    /// `role`; everything else — compaction, the rollback snapshot, the tool
    /// loop — is identical, so there is one loop, not two.
    fn run_with_role(
        &mut self,
        input: &str,
        out: &mut dyn Write,
        role: CancelRole,
    ) -> Result<String, AgentError> {
        // Fresh per-turn side-effect tracking — the flag must not leak across
        // turns: a mutation in a prior turn does not protect this one. The
        // breaker count resets with it: consecutive denials are a per-turn
        // signal, and a stuck *turn* is what the breaker ends.
        self.side_effected_this_turn = false;
        self.auto_denials = 0;

        // Owner turns discard a cancellation that landed between turns. During
        // editing the terminal is raw (ISIG off), so Ctrl-C is the editor's own
        // line-cancel byte and never reaches here; in the brief cooked gaps
        // around a read there is no turn to cancel, so a stray SIGINT there
        // must not abort the next turn. (A cancellation *during* compaction
        // below cancels the turn at the first pre-request check — nothing is
        // recorded yet, so there is nothing to roll back.) A nested run must
        // *not* clear the flag: a Ctrl-C raised mid-parent-turn is meant to
        // stop the child too, and only the parent's owning run may consume it.
        if matches!(role, CancelRole::Owner) {
            self.cancel.store(false, Ordering::Relaxed);
        }

        // Compact before the rollback snapshot and before the new input is
        // pushed: compaction is a deliberate, retained state change (you don't
        // un-compact and re-overflow), so a failed turn rolls back to the
        // *compacted* history. A compaction failure surfaces here, before the
        // input is recorded.
        self.maybe_compact(input, out)?;

        // Snapshot for a possible rollback. The new input either pushes a fresh
        // user message or coalesces into a trailing user message a preserved
        // prior turn left behind; recording both the message count and that
        // trailing message's block count lets the rollback undo whichever
        // happened — a push grows the count, a coalesce grows the block vector.
        let snapshot_len = self.messages.len();
        let snapshot_last_blocks = self.messages.last().map(|m| m.content.len());
        // The prompt-size measurement rolls back with the messages it was
        // taken from: an inner tool-loop iteration may have recorded one
        // before a later iteration failed, and keeping it would describe a
        // context the rollback just discarded.
        let snapshot_input_tokens = self.last_input_tokens;

        // A preserved turn can end on a `Role::User` tool_result message; a
        // fresh user message after it would be two consecutive user messages,
        // which Anthropic's alternating-role contract rejects. Coalesce the
        // text into that trailing message instead (a user message may carry
        // both tool_result and text blocks).
        context::append_input(&mut self.messages, input);

        let result = self.run_loop(out);

        // Deliver a pending cancellation. Any failure with a cancellation
        // pending reports as the quiet `Cancelled` — the operator asked to
        // stop, and the interrupted turn's error is often just the fallout of
        // the signal itself (an EINTR'd read). A turn that *completed* before
        // the flag went up is kept as-is: its answer already streamed in full,
        // and cancelling finished work would discard a good turn. The owning
        // run consumes the flag (swap-to-clear) so the next turn starts clean;
        // a nested run only reads it, leaving the parent's owning run to
        // consume it — otherwise the child would swallow the parent's Ctrl-C.
        let cancelled = match role {
            CancelRole::Owner => self.cancel.swap(false, Ordering::Relaxed),
            CancelRole::Observer => self.cancelled(),
        };
        let result = match result {
            Err(_) if cancelled => Err(AgentError::Cancelled),
            other => other,
        };

        // Salvage a nested child's partial findings at the turn wall. A
        // read-only child that exhausted its rounds would otherwise return
        // only an error and have its whole transcript erased by the rollback
        // below — everything it read paid for and discarded. One wrap-up
        // round-trip turns that into a marked partial result. Only nested
        // children (Observer) salvage: the top-level Owner turn errors exactly
        // as before, and a pending cancellation already took precedence above.
        let result = match result {
            Err(AgentError::TurnLimitExceeded(_)) if matches!(role, CancelRole::Observer) => {
                self.salvage_at_wall(out)
            }
            other => other,
        };

        // Roll back only on a pure read/API failure or a cancellation — the
        // two share rollback semantics by design. If a side-effecting tool
        // ran this turn the filesystem already diverged, so the transcript is
        // kept verbatim — losing it would leave the model blind to a mutation
        // it must reason about. The two-part restore mirrors the snapshot:
        // drop messages added this turn, then strip any coalesced text from the
        // (now-trailing) preserved message.
        if result.is_err() && !self.side_effected_this_turn {
            self.messages.truncate(snapshot_len);
            self.last_input_tokens = snapshot_input_tokens;
            if let Some(blocks) = snapshot_last_blocks
                && let Some(last) = self.messages.last_mut()
            {
                last.content.truncate(blocks);
            }
        }
        result
    }

    /// The model id requests are currently built with. Backs the REPL's bare
    /// `/model` command, which reports the active model.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Switch the model used for subsequent turns. Backs the REPL's
    /// `/model <id>` command. The model is a per-request string — `build_request`
    /// reads `self.config.model` afresh each turn — so the swap takes effect on
    /// the next turn with no provider rebuild and no change to conversation
    /// history. The id is not validated: an unknown model surfaces as the
    /// provider's API error on the next request (fail-fast at request time).
    /// `context_token_limit` deliberately does not auto-track the swap — with
    /// no model registry to consult (the same ethos that leaves the id
    /// unvalidated), the compaction limit stays the operator's config knob.
    pub fn set_model(&mut self, model: String) {
        self.config.model = model;
    }

    /// The reasoning effort requests are currently built with, if any. Backs
    /// the REPL's `/effort` report and the effort shown in the `/model` view.
    pub fn effort(&self) -> Option<&str> {
        self.config.effort.as_deref()
    }

    /// Set (or, with `None`, clear) the reasoning effort for subsequent turns.
    /// Backs the REPL's `/effort <value>` command. Like the model id, the
    /// value is a per-request opaque string — `build_request` reads it afresh
    /// each turn — so the change takes effect on the next turn, and a bad value
    /// fails at request time with the provider's own error.
    pub fn set_effort(&mut self, effort: Option<String>) {
        self.config.effort = effort;
    }

    /// The provider requests are currently routed to, paired with
    /// [`Agent::model`] in the REPL's bare `/model` report.
    pub fn provider_kind(&self) -> ProviderKind {
        self.config.provider_kind
    }

    /// Switch the provider — and with it the model, since a model id is
    /// meaningless across vendors — used for subsequent turns. Backs the
    /// REPL's `/model <provider>:<model>` command; the REPL resolves the key
    /// and builds `provider` first, so a failed key lookup never reaches here
    /// and fail-soft stays the caller's concern. Conversation history carries
    /// over untouched — it is provider-agnostic in the turn model, and both
    /// adapters accept any normalized history. The same non-validation
    /// contract as [`Agent::set_model`] applies: a bad model id fails at
    /// request time, and `context_token_limit` does not auto-track the swap.
    /// `last_input_tokens` is also deliberately kept: it measures the very
    /// history that carries over, so it stays an approximately-right
    /// compaction signal under the new provider's tokenizer (the 75%
    /// threshold absorbs the drift) — resetting it as `clear` does for a
    /// genuinely emptied history would instead blind the guard for a turn
    /// over a conversation that still exists. The model-id cache, by
    /// contrast, is provider state and *is* dropped — the old provider's
    /// listing must not complete for the new one. The reasoning effort is
    /// dropped for the same reason: the extremes differ per provider (`max`
    /// on Anthropic, `none` on OpenAI), so a value set for one vendor is
    /// meaningless under another. A same-provider `/model` switch
    /// goes through [`Agent::set_model`] instead and keeps the effort.
    pub fn set_provider(&mut self, kind: ProviderKind, provider: Box<dyn Provider>, model: String) {
        self.provider = provider;
        self.config.provider_kind = kind;
        self.config.model = model;
        self.config.effort = None;
        self.models_cache = None;
    }

    /// The provider's model ids — the completion source for the REPL's
    /// `/model` autocomplete. Fetched from the provider once, on first use,
    /// then held for the session (`set_provider` drops the cache, so a swap
    /// re-fetches from the new provider). Fail-soft by decision: an offline
    /// or non-2xx listing caches as *empty* — no completions, never an error
    /// to the user — and is not retried, so a flaky network cannot add a
    /// timeout to every completion attempt. Commands and provider names
    /// still complete without it.
    pub fn list_models_cached(&mut self) -> &[String] {
        self.models_cache
            .get_or_insert_with(|| self.provider.list_models().unwrap_or_default())
    }

    /// A non-fetching view of the model-id cache: `Some` only once a listing
    /// has been fetched this session. Backs the REPL's switch-time warning,
    /// which must never trigger the fetch [`Agent::list_models_cached`]
    /// performs — an uncached catalog means no opinion, not a lookup.
    pub fn cached_models(&self) -> Option<&[String]> {
        self.models_cache.as_deref()
    }

    /// Inner loop factored out so `run` can conditionally roll back history on
    /// an error path — keeping the transcript when a side-effecting tool ran,
    /// truncating it otherwise.
    fn run_loop(&mut self, out: &mut dyn Write) -> Result<String, AgentError> {
        for _ in 0..self.config.max_turns {
            // The pre-request seam: a cancellation observed here stops the
            // tool loop before another provider round-trip. This is also
            // where a Ctrl-C that arrived during tool execution (including
            // one raised at the confirmation prompt) takes effect — the
            // batch resolves its remaining slots first, so a kept history
            // always carries a result for every tool_use.
            if self.cancelled() {
                return Err(AgentError::Cancelled);
            }
            let request = self.fit_request(self.build_request(), out)?;
            let mut stream = self.provider.stream(&request).map_err(AgentError::Api)?;
            let result = self.process_stream(stream.as_mut(), out)?;

            // Record the measured prompt size. Each inner tool-loop iteration
            // overwrites the last, so after `run` returns this holds the final
            // (largest) measured context of the turn.
            // `prompt_size` sums the untrusted provider counts in `u64`;
            // saturate back into the `u32` store. Saturating high is safe: a
            // count above `u32::MAX` dwarfs any context limit, so the guard
            // compacts rather than being disarmed by a wrap.
            self.last_input_tokens = u32::try_from(result.usage.prompt_size()).unwrap_or(u32::MAX);

            match result.stop_reason {
                StopReason::ToolUse => {
                    // The tool round commits every block unchanged;
                    // `execute_tools` pairs each tool_use with a tool_result in
                    // the User message that immediately follows.
                    self.messages.push(TurnMessage {
                        role: Role::Assistant,
                        content: result.blocks.clone(),
                    });
                    let tool_results = self.execute_tools(&result.blocks, out);
                    self.messages.push(TurnMessage {
                        role: Role::User,
                        content: tool_results,
                    });
                    // The circuit breaker ends the turn *after* the results
                    // are recorded — like a cancellation, a kept history must
                    // never leave a tool_use dangling without its result.
                    if self.auto_denials >= CONFIRM_BREAKER_LIMIT {
                        return Err(AgentError::ConfirmBreaker);
                    }
                }
                stop => {
                    // EndTurn, StopSequence, MaxTokens all end the turn. A
                    // response cut off at max_tokens *while* emitting a tool_use
                    // finalizes a partial, unpaired tool_use; committing it would
                    // leave a tool_use with no tool_result, so the next request is
                    // rejected — and since the rollback snapshot sits after it,
                    // every later turn too (a permanent wedge only /clear escapes,
                    // which autosave then persists to disk). No non-tool-use stop
                    // pairs a tool_use, so drop every one, keep text/thinking, and
                    // skip the message when nothing survives (an empty content
                    // array is itself wire-rejected).
                    let content: Vec<Block> = result
                        .blocks
                        .iter()
                        .filter(|b| !matches!(b, Block::ToolUse { .. }))
                        .cloned()
                        .collect();
                    if !content.is_empty() {
                        self.messages.push(TurnMessage {
                            role: Role::Assistant,
                            content,
                        });
                    }
                    // MaxTokens additionally means the response was cut off at
                    // the token limit — warn so a partial answer isn't mistaken
                    // for a complete one.
                    if stop == StopReason::MaxTokens {
                        self.meta_line(out, format_args!("response truncated: hit max_tokens"));
                    }
                    self.usage_line(&result.usage, out);
                    return Ok(text_of(&result.blocks));
                }
            }
        }

        Err(AgentError::TurnLimitExceeded(self.config.max_turns))
    }

    /// Wring a partial answer out of a nested child that ran out of tool
    /// rounds, instead of throwing its whole exploration away. Runs one extra
    /// round-trip that asks — in text — for the findings so far, and returns
    /// them under [`SALVAGE_MARKER`]. Fallbacks, in order: a text-less wrap-up
    /// reply or a wrap-up API error falls back to the transcript's last
    /// assistant text; when there is no assistant text anywhere, nothing can
    /// be salvaged and the original `TurnLimitExceeded` stands. Called only on
    /// the Observer path, before the error-path rollback that would erase the
    /// transcript this reads.
    fn salvage_at_wall(&mut self, out: &mut dyn Write) -> Result<String, AgentError> {
        // The most recent assistant text already in the transcript — the
        // fallback when the wrap-up round yields no usable text. Captured
        // before the extra round-trip appends anything.
        let fallback = self.last_assistant_text();

        // Coalesce the wrap-up instruction into the trailing tool-result
        // message. An exhausted transcript always ends on a `Role::User`
        // tool-result (the loop only continues on `ToolUse`, pushing assistant
        // then user each round), so a fresh user message would break the
        // alternating-role contract; coalescing keeps the roles alternating.
        self.messages
            .last_mut()
            .expect("an exhausted transcript ends on a user tool-result message")
            .content
            .push(Block::Text(WRAP_UP_INSTRUCTION.to_string()));

        // One more round-trip with the tools still defined (the wire contract
        // requires their definitions while the history carries `tool_use`). A
        // non-empty text reply is the salvage; a text-less reply or an API
        // error falls through to the transcript fallback.
        let reply = self.wrap_up_reply(out).unwrap_or_default();
        let salvaged = if reply.is_empty() {
            match fallback {
                Some(text) => text,
                None => return Err(AgentError::TurnLimitExceeded(self.config.max_turns)),
            }
        } else {
            reply
        };

        Ok(format!("{SALVAGE_MARKER}\n\n{salvaged}"))
    }

    /// The one wrap-up round-trip: build the request (tools still defined),
    /// stream it, and return the reply's joined text. Any `tool_use` blocks
    /// are ignored — the loop is over. A build/stream error propagates for
    /// [`Agent::salvage_at_wall`] to treat as a fallback trigger.
    fn wrap_up_reply(&mut self, out: &mut dyn Write) -> Result<String, AgentError> {
        let request = self.fit_request(self.build_request(), out)?;
        let mut stream = self.provider.stream(&request).map_err(AgentError::Api)?;
        let result = self.process_stream(stream.as_mut(), out)?;
        Ok(text_of(&result.blocks))
    }

    /// The newest assistant message carrying non-empty text, joined. Walks the
    /// history newest-first and skips assistant messages that hold only
    /// `tool_use` blocks; `None` when no assistant text exists at all.
    fn last_assistant_text(&self) -> Option<String> {
        self.messages.iter().rev().find_map(|m| {
            (m.role == Role::Assistant)
                .then(|| text_of(&m.content))
                .filter(|t| !t.is_empty())
        })
    }
}

#[cfg(test)]
impl Agent {
    /// Read the conversation history from a sibling module's tests (the `task`
    /// tool's fan-out reassembly check reads the tool-result blocks the loop
    /// recorded). This module's own tests reach the field directly.
    pub(crate) fn messages_for_test(&self) -> &[TurnMessage] {
        &self.messages
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;
    use crate::testing::{ErrProvider, MockProvider, TestTool, expect_tool_result};
    use crate::turn::{StreamDelta, Usage};

    // ── AgentError ──

    #[test]
    fn agent_error_displays_every_variant() {
        assert_eq!(
            AgentError::TurnLimitExceeded(MAX_TURNS).to_string(),
            format!("agent exceeded maximum of {MAX_TURNS} turns")
        );
        // The rendered limit is the value carried, not a constant — a raised
        // limit reports its own number.
        assert_eq!(
            AgentError::TurnLimitExceeded(40).to_string(),
            "agent exceeded maximum of 40 turns"
        );
        // The Api variant renders its inner error, and From<ApiError> wraps it.
        let err: AgentError = ApiError::Stream("overloaded".to_string()).into();
        assert_eq!(err.to_string(), "stream error: overloaded");
        assert_eq!(AgentError::Cancelled.to_string(), "turn cancelled");
    }

    // ── tool_names ──

    #[test]
    fn tool_names_lists_tools_in_registration_order() {
        let agent = agent_with_tools(vec![mock_tool("alpha", "a"), mock_tool("beta", "b")]);
        assert_eq!(agent.tool_names(), vec!["alpha", "beta"]);
    }

    #[test]
    fn tool_names_empty_when_no_tools() {
        let agent = agent_with_tools(vec![]);
        assert!(agent.tool_names().is_empty());
    }

    // ── Confirmation policy dispatch (allow / judge) ──

    #[test]
    fn run_stops_at_the_confirmation_circuit_breaker() {
        // A model that keeps re-trying a judged-away call forever: the turn
        // ends itself after the limit instead of burning MAX_TURNS provider
        // rounds (and MAX_TURNS paid judge calls) against a wall.
        let stream = tool_use_stream("t1", "guarded_write");
        let mut agent = Agent::new(
            Box::new(MockProvider::repeating(stream)),
            test_config(None),
            vec![confirmed_tool()],
        );
        agent.set_confirm_policy(judged_stub("DENY\nno"));
        let err = agent.run("do it", &mut std::io::sink()).unwrap_err();
        assert!(matches!(err, AgentError::ConfirmBreaker));
        assert!(
            err.to_string().contains("3 consecutive automated denials"),
            "got: {err}"
        );
    }

    // ── run / run_loop (hermetic, via MockProvider) ──

    #[test]
    fn run_returns_text_and_records_history() {
        let stream = vec![
            message_start(),
            StreamDelta::TextStart {
                index: 0,
                text: String::new(),
            },
            StreamDelta::TextDelta {
                index: 0,
                text: "Hello ".to_string(),
            },
            StreamDelta::TextDelta {
                index: 0,
                text: "world".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![stream])),
            test_config(None),
            vec![],
        );

        let text = agent.run("hi", &mut std::io::sink()).unwrap();

        assert_eq!(text, "Hello world");
        assert_eq!(agent.messages.len(), 2);
        assert_eq!(agent.messages[0], user_msg("hi"));
        assert_eq!(agent.messages[1], assistant_msg("Hello world"));
    }

    #[test]
    fn run_final_text_skips_non_text_blocks() {
        // A degenerate end_turn response carrying a tool_use block alongside
        // its text: the caller gets the text only, and the unpaired tool_use is
        // stripped from committed history (an unpaired tool_use would wedge the
        // next request), leaving just the assistant text.
        let stream = vec![
            message_start(),
            StreamDelta::ToolUseStart {
                index: 0,
                id: "toolu_1".to_string(),
                name: "noop".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::TextStart {
                index: 1,
                text: "answer".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![stream])),
            test_config(None),
            vec![],
        );

        let text = agent.run("hi", &mut std::io::sink()).unwrap();

        assert_eq!(text, "answer");
        assert_eq!(
            agent.messages,
            vec![user_msg("hi"), assistant_msg("answer")]
        );
    }

    #[test]
    fn run_max_tokens_returns_partial_text() {
        // A turn cut off at the token limit still returns its partial text and
        // records it — the truncation notice is informational, not an error.
        let stream = vec![
            message_start(),
            StreamDelta::TextStart {
                index: 0,
                text: "partial".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::MaxTokens),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![stream])),
            test_config(None),
            vec![],
        );

        let mut out = Vec::new();
        let text = agent.run("write an essay", &mut out).unwrap();

        assert_eq!(text, "partial");
        assert_eq!(agent.messages.len(), 2);
        assert_eq!(agent.messages[1], assistant_msg("partial"));
        // The cut-off is flagged on the output so a partial answer is never
        // mistaken for a complete one.
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("[response truncated: hit max_tokens]")
        );
    }

    #[test]
    fn run_max_tokens_mid_tool_use_strips_the_orphan_and_survives_the_next_turn() {
        // Cut off at max_tokens while emitting a tool_use: the partial text is
        // kept and returned, the unpaired tool_use is stripped from history, and
        // the *next* turn proceeds normally instead of the provider 400ing on an
        // assistant tool_use with no tool_result.
        let first = vec![
            message_start(),
            StreamDelta::TextStart {
                index: 0,
                text: "let me check".to_string(),
            },
            StreamDelta::ToolUseStart {
                index: 1,
                id: "call_1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::MaxTokens),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![first, text_stream("recovered")])),
            test_config(None),
            vec![mock_tool("echo", "ok")],
        );

        let mut out = Vec::new();
        let text = agent.run("hi", &mut out).unwrap();
        assert_eq!(text, "let me check");
        assert_eq!(
            agent.messages,
            vec![user_msg("hi"), assistant_msg("let me check")]
        );
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("[response truncated: hit max_tokens]")
        );

        // The next turn is not wedged: no orphaned tool_use remains anywhere.
        let text = agent.run("continue", &mut std::io::sink()).unwrap();
        assert_eq!(text, "recovered");
        assert!(
            !agent
                .messages
                .iter()
                .flat_map(|m| &m.content)
                .any(|b| matches!(b, Block::ToolUse { .. }))
        );
    }

    #[test]
    fn run_max_tokens_on_a_bare_tool_use_commits_no_assistant_message() {
        // Cut off at max_tokens with only a tool_use and no text: nothing
        // survives the strip, so no empty assistant message is committed (an
        // empty content array is itself wire-rejected).
        let stream = vec![
            message_start(),
            StreamDelta::ToolUseStart {
                index: 0,
                id: "call_1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::MaxTokens),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![stream])),
            test_config(None),
            vec![mock_tool("echo", "ok")],
        );

        let text = agent.run("hi", &mut std::io::sink()).unwrap();
        assert_eq!(text, "");
        assert_eq!(agent.messages, vec![user_msg("hi")]);
    }

    #[test]
    fn run_records_measured_prompt_size() {
        // The recorded size sums input_tokens with both cache counters —
        // bare input_tokens undercounts a cached conversation.
        let first = vec![
            StreamDelta::MessageStart {
                usage: Usage {
                    input_tokens: 100,
                    output_tokens: 0,
                    cache_creation_input_tokens: Some(7),
                    cache_read_input_tokens: Some(3),
                },
            },
            StreamDelta::TextStart {
                index: 0,
                text: "one".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            },
        ];
        let second = vec![
            StreamDelta::MessageStart {
                usage: Usage {
                    input_tokens: 250,
                    output_tokens: 0,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            },
            StreamDelta::TextStart {
                index: 0,
                text: "two".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![first, second])),
            test_config(None),
            vec![],
        );
        assert_eq!(agent.last_input_tokens, 0);

        agent.run("hi", &mut std::io::sink()).unwrap();
        assert_eq!(agent.last_input_tokens, 110);

        // The next completed turn overwrites — the field holds the latest
        // measurement, not a running sum.
        agent.run("again", &mut std::io::sink()).unwrap();
        assert_eq!(agent.last_input_tokens, 250);
    }

    // ── the end-of-turn usage line ──

    #[test]
    fn run_prints_usage_line_on_completed_turn() {
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![measured_text_stream("hi", 42, 17)])),
            test_config(None),
            vec![],
        );

        let mut out = Vec::new();
        agent.run("hello", &mut out).unwrap();

        let printed = String::from_utf8(out).unwrap();
        // Grouped counts, the guard's context measure against its ceiling —
        // and no cache segment when no cache activity was reported.
        assert!(
            printed.contains("[usage: 42 in, 17 out · context: 42/100,000 (0%)]"),
            "got: {printed}"
        );
        assert!(!printed.contains("cache:"));
    }

    #[test]
    fn run_usage_line_includes_cache_when_present() {
        let stream = vec![
            StreamDelta::MessageStart {
                usage: Usage {
                    input_tokens: 100,
                    output_tokens: 0,
                    cache_creation_input_tokens: Some(7),
                    cache_read_input_tokens: Some(3),
                },
            },
            StreamDelta::TextStart {
                index: 0,
                text: "hi".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 17,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![stream])),
            test_config(None),
            vec![],
        );

        let mut out = Vec::new();
        agent.run("hello", &mut out).unwrap();

        // The context measure sums input with both cache counters — the
        // exact number recorded as last_input_tokens.
        assert!(String::from_utf8(out).unwrap().contains(
            "[usage: 100 in, 17 out · cache: 3 read, 7 written · context: 110/100,000 (0%)]"
        ));
        assert_eq!(agent.last_input_tokens, 110);
    }

    #[test]
    fn run_usage_line_reports_final_round_trip_and_threshold_percent() {
        // A tool-loop turn: the line prints once, with the *final*
        // round-trip's usage (the guard's signal), not the earlier one's.
        let first = vec![
            StreamDelta::MessageStart {
                usage: Usage {
                    input_tokens: 40,
                    output_tokens: 0,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            },
            StreamDelta::ToolUseStart {
                index: 0,
                id: "t1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![
                first,
                measured_text_stream("done", 1500, 10),
            ])),
            AgentConfig {
                provider_kind: ProviderKind::Anthropic,
                model: crate::TEST_MODEL.to_string(),
                max_tokens: 64,
                system: None,
                context_token_limit: 2000,
                effort: None,
                max_turns: MAX_TURNS,
            },
            vec![mock_tool("echo", "echoed")],
        );

        let mut out = Vec::new();
        agent.run("use echo", &mut out).unwrap();

        let printed = String::from_utf8(out).unwrap();
        assert!(
            printed.contains("[usage: 1,500 in, 10 out · context: 1,500/2,000 (75%)]"),
            "got: {printed}"
        );
        assert_eq!(printed.matches("[usage:").count(), 1);
    }

    #[test]
    fn run_zero_usage_prints_no_usage_line() {
        // A round-trip that reported no usage at all (every counter zero)
        // stays quiet — no fabricated zeros.
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![text_stream("hi")])),
            test_config(None),
            vec![],
        );

        let mut out = Vec::new();
        agent.run("hello", &mut out).unwrap();

        assert!(!String::from_utf8(out).unwrap().contains("[usage:"));
    }

    #[test]
    fn run_failed_turn_prints_no_usage_line() {
        // The turn measures usage on its first round-trip, then the next
        // request fails: no usage line — the measurement rolls back with the
        // turn, so reporting it would describe discarded state.
        let first = vec![
            StreamDelta::MessageStart {
                usage: Usage {
                    input_tokens: 40,
                    output_tokens: 0,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            },
            StreamDelta::ToolUseStart {
                index: 0,
                id: "t1".to_string(),
                name: "noop".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(SucceedThenErrProvider::new(vec![first])),
            test_config(None),
            vec![mock_tool("noop", "ok")],
        );

        let mut out = Vec::new();
        assert!(agent.run("go", &mut out).is_err());
        assert!(!String::from_utf8(out).unwrap().contains("[usage:"));
    }

    // ── group_thousands ──

    #[test]
    fn group_thousands_inserts_separators_every_three_digits() {
        assert_eq!(group_thousands(0), "0");
        assert_eq!(group_thousands(999), "999");
        assert_eq!(group_thousands(1_000), "1,000");
        assert_eq!(group_thousands(200_000), "200,000");
        assert_eq!(group_thousands(1_234_567), "1,234,567");
    }

    #[test]
    fn run_executes_tool_then_completes() {
        let first = vec![
            message_start(),
            StreamDelta::ToolUseStart {
                index: 0,
                id: "t1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::ToolArgsDelta {
                index: 0,
                json: "{\"msg\":\"hi\"}".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            },
        ];
        let second = vec![
            message_start(),
            StreamDelta::TextStart {
                index: 0,
                text: "done".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![first, second])),
            test_config(None),
            vec![mock_tool("echo", "echoed")],
        );

        let text = agent.run("use echo", &mut std::io::sink()).unwrap();

        assert_eq!(text, "done");
        // user → assistant(tool_use) → user(tool_result) → assistant(text)
        assert_eq!(agent.messages.len(), 4);
        assert_eq!(agent.messages[2].role, Role::User);
        assert_eq!(
            agent.messages[2].content,
            vec![Block::ToolResult {
                tool_use_id: "t1".to_string(),
                content: "echoed".to_string(),
                is_error: false,
            }]
        );
    }

    #[test]
    fn run_styled_dims_meta_lines_but_not_model_text() {
        let first = vec![
            message_start(),
            StreamDelta::ToolUseStart {
                index: 0,
                id: "t1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            },
        ];
        let second = vec![
            message_start(),
            StreamDelta::TextStart {
                index: 0,
                text: "done".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![first, second])),
            test_config(None),
            vec![mock_tool("echo", "echoed")],
        );
        agent.set_styled(true);

        let mut out = Vec::new();
        agent.run("use echo", &mut out).unwrap();

        let printed = String::from_utf8(out).unwrap();
        // Every bracketed meta line is wrapped in ANSI faint-on/reset…
        assert!(printed.contains("\x1b[2m[tool: echo]\x1b[0m"));
        assert!(printed.contains("\x1b[2m[result: echoed]\x1b[0m"));
        // …while the model's own streamed text stays unstyled.
        assert!(printed.contains("done"));
        assert!(!printed.contains("\x1b[2mdone"));
    }

    #[test]
    fn run_styled_dims_thinking_meta_line_and_hides_reasoning() {
        // Styled mode: the thinking marker is a dimmed meta line like every
        // other bookkeeping line, and the reasoning text stays off-screen.
        let stream = vec![
            message_start(),
            StreamDelta::ThinkingStart {
                index: 0,
                text: String::new(),
                signature: String::new(),
            },
            StreamDelta::ThinkingDelta {
                index: 0,
                text: "secret reasoning".to_string(),
            },
            StreamDelta::SignatureDelta {
                index: 0,
                signature: "sig".to_string(),
            },
            StreamDelta::TextStart {
                index: 1,
                text: "done".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![stream])),
            test_config(None),
            Vec::new(),
        );
        agent.set_styled(true);

        let mut out = Vec::new();
        let text = agent.run("hi", &mut out).unwrap();

        // The returned text is the answer alone — thinking is not the answer.
        assert_eq!(text, "done");
        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("\x1b[2m[thinking]\x1b[0m"));
        assert!(!printed.contains("secret reasoning"));
        // History preserves the full block for the next turn's resend.
        assert_eq!(
            agent.messages[1].content[0],
            Block::Thinking {
                text: "secret reasoning".to_string(),
                signature: "sig".to_string(),
            }
        );
    }

    #[test]
    fn run_thinking_tool_loop_preserves_thinking_across_the_round_trip() {
        // The claude-fable-5 shape hermetically: a thinking block leads a
        // tool-use turn, the loop executes the tool, and the assistant
        // message the next request resends still carries the thinking block
        // with its signature — the multi-turn contract's precondition.
        let first = vec![
            message_start(),
            StreamDelta::ThinkingStart {
                index: 0,
                text: String::new(),
                signature: String::new(),
            },
            StreamDelta::ThinkingDelta {
                index: 0,
                text: "I should echo".to_string(),
            },
            StreamDelta::SignatureDelta {
                index: 0,
                signature: "sig_1".to_string(),
            },
            StreamDelta::ToolUseStart {
                index: 1,
                id: "t1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![first, text_stream("done")])),
            test_config(None),
            vec![mock_tool("echo", "echoed")],
        );

        let mut out = Vec::new();
        let text = agent.run("use echo", &mut out).unwrap();

        assert_eq!(text, "done");
        // user → assistant(thinking + tool_use) → user(tool_result) →
        // assistant(text): the thinking block sits in history verbatim.
        assert_eq!(agent.messages.len(), 4);
        assert_eq!(
            agent.messages[1].content,
            vec![
                Block::Thinking {
                    text: "I should echo".to_string(),
                    signature: "sig_1".to_string(),
                },
                Block::ToolUse {
                    id: "t1".to_string(),
                    name: "echo".to_string(),
                    input: serde_json::json!({}),
                },
            ]
        );
        let (id, content, is_error) = expect_tool_result(&agent.messages[2].content[0]);
        assert_eq!(id, "t1");
        assert_eq!(content, "echoed");
        assert!(!is_error);
    }

    #[test]
    fn run_styled_renders_streamed_markdown() {
        // Markers split across TextStart and TextDelta: the streaming
        // renderer must reassemble them, and the end-of-stream flush must
        // emit the newline-less tail.
        let stream = vec![
            message_start(),
            StreamDelta::TextStart {
                index: 0,
                text: "## Do".to_string(),
            },
            StreamDelta::TextDelta {
                index: 0,
                text: "ne\n**bo".to_string(),
            },
            StreamDelta::TextDelta {
                index: 0,
                text: "ld** and `code`".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![stream])),
            test_config(None),
            Vec::new(),
        );
        agent.set_styled(true);

        let mut out = Vec::new();
        agent.run("hi", &mut out).unwrap();

        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("\x1b[1;4mDone\x1b[0m\n"));
        assert!(printed.contains("\x1b[1mbold\x1b[0m and \x1b[36mcode\x1b[0m\n"));
        // The raw markers themselves never reach the terminal.
        assert!(!printed.contains("**"));
        assert!(!printed.contains("##"));
    }

    #[test]
    fn run_unstyled_leaves_markdown_verbatim() {
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![text_stream("**raw** `md`")])),
            test_config(None),
            Vec::new(),
        );

        let mut out = Vec::new();
        agent.run("hi", &mut out).unwrap();

        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("**raw** `md`"));
        assert!(!printed.contains('\x1b'));
    }

    #[test]
    fn run_unstyled_scrubs_control_bytes() {
        // The unstyled sink (piped/non-TTY) shares the renderer's scrub
        // policy: model-supplied ESC/BEL/C1 bytes and invisible-format
        // characters (here a U+202E bidi override) never reach the terminal on
        // either the TextStart or the TextDelta write, while `\n` survives.
        let stream = vec![
            message_start(),
            StreamDelta::TextStart {
                index: 0,
                text: "a\x1bb".to_string(),
            },
            StreamDelta::TextDelta {
                index: 0,
                text: "c\x07d\u{0080}e\u{202e}\n".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![stream])),
            test_config(None),
            Vec::new(),
        );

        let mut out = Vec::new();
        agent.run("hi", &mut out).unwrap();

        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("a\u{FFFD}bc\u{FFFD}d\u{FFFD}e\u{FFFD}\n"));
        assert!(!printed.contains('\x1b'));
        assert!(!printed.contains('\x07'));
        assert!(!printed.contains('\u{0080}'));
        assert!(!printed.contains('\u{202e}'));
    }

    #[test]
    fn run_styled_flushes_partial_text_line_before_tool_meta_line() {
        // A text block that ends mid-line followed by a tool call: the
        // buffered tail must land before the meta line, mirroring the
        // unstyled path's layout.
        let first = vec![
            message_start(),
            StreamDelta::TextStart {
                index: 0,
                text: "checking".to_string(),
            },
            StreamDelta::ToolUseStart {
                index: 1,
                id: "t1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![first, text_stream("done")])),
            test_config(None),
            vec![mock_tool("echo", "echoed")],
        );
        agent.set_styled(true);

        let mut out = Vec::new();
        agent.run("use echo", &mut out).unwrap();

        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("checking\x1b[2m[tool: echo]\x1b[0m"));
    }

    #[test]
    fn run_exceeding_turn_limit_with_read_only_tool_rolls_back() {
        // A provider that always asks for a tool call loops until the cap. The
        // `echo` tool is read-only (not side-effecting), so nothing on disk
        // diverged — the turn-limit error still rolls the whole turn back.
        let tool_stream = vec![
            message_start(),
            StreamDelta::ToolUseStart {
                index: 0,
                id: "t1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::ToolArgsDelta {
                index: 0,
                json: "{}".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::repeating(tool_stream)),
            test_config(None),
            vec![mock_tool("echo", "ok")],
        );

        let result = agent.run("loop forever", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::TurnLimitExceeded(_))));
        // On error, run() truncates history back to the pre-call snapshot.
        assert!(agent.messages.is_empty());
    }

    #[test]
    fn run_loop_honors_a_configured_non_default_max_turns() {
        // With max_turns lowered to 2, the loop stops after its second round
        // rather than the default ten: a three-long script whose third stream
        // *would* end the turn is never reached, and the error carries the
        // configured limit — 2, not MAX_TURNS.
        let mut config = test_config(None);
        config.max_turns = 2;
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![
                tool_use_stream("t1", "echo"),
                tool_use_stream("t2", "echo"),
                text_stream("would end the turn"),
            ])),
            config,
            vec![mock_tool("echo", "ok")],
        );

        let err = agent.run("survey", &mut std::io::sink()).unwrap_err();
        assert_eq!(err.to_string(), "agent exceeded maximum of 2 turns");
    }

    #[test]
    fn run_loop_a_raised_limit_reaches_a_later_end() {
        // The complement: the same three-long script *completes* under a limit
        // of 3, proving max_turns is the live bound — the third stream, which
        // a limit of 2 never reaches, ends the turn.
        let mut config = test_config(None);
        config.max_turns = 3;
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![
                tool_use_stream("t1", "echo"),
                tool_use_stream("t2", "echo"),
                text_stream("done"),
            ])),
            config,
            vec![mock_tool("echo", "ok")],
        );

        let result = agent.run("survey", &mut std::io::sink());
        assert_eq!(result.unwrap(), "done");
    }

    #[test]
    fn run_provider_error_before_any_tool_rolls_back_to_snapshot() {
        // The stream fails on the first call, before any tool runs, so no side
        // effect occurred and the freshly pushed user message is rolled back.
        let mut agent = Agent::new(Box::new(ErrProvider), test_config(None), vec![]);
        agent.messages.push(assistant_msg("prior history"));
        let before = agent.messages.len();

        let result = agent.run("hi", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(_))));
        // The freshly pushed user message is rolled back; prior history remains.
        assert_eq!(agent.messages.len(), before);
        assert_eq!(agent.messages[0], assistant_msg("prior history"));
    }

    #[test]
    fn run_terminal_stream_error_rolls_partial_answer_out_of_session_snapshot() {
        // Anthropic can stream visible text before the adapter discovers that
        // EOF arrived without message_stop. The terminal error must roll back
        // both the fresh user turn and that partial assistant answer, so the
        // snapshot the REPL hands to autosave contains neither.
        let stream = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: String::new(),
            }),
            Ok(StreamDelta::TextDelta {
                index: 0,
                text: "partial answer".to_string(),
            }),
            Err(ApiError::Stream(
                "stream ended without a terminal message_stop event".to_string(),
            )),
        ];
        let mut agent = Agent::new(
            Box::new(SucceedThenErrProvider::with_result_stream(stream)),
            test_config(None),
            vec![],
        );
        agent.messages.push(user_msg("earlier question"));
        agent.messages.push(assistant_msg("earlier answer"));
        let before = agent.session();
        let mut out = Vec::new();

        let result = agent.run("next question", &mut out);

        assert!(matches!(
            result,
            Err(AgentError::Api(ApiError::Stream(ref message)))
                if message == "stream ended without a terminal message_stop event"
        ));
        assert_eq!(String::from_utf8(out).unwrap(), "partial answer");
        let autosave_input = agent.session();
        assert_eq!(autosave_input, before);
        assert!(
            !serde_json::to_string(&autosave_input)
                .unwrap()
                .contains("partial answer")
        );
    }

    // ── conditional rollback after side effects (D2) ──

    #[test]
    fn run_keeps_history_when_side_effecting_tool_ran_then_stream_errors() {
        // A side-effecting tool runs, then the next turn's stream errors. The
        // mutation already touched disk, so the transcript must survive intact —
        // rolling it back would blind the model to a change it must reason about.
        let mut agent = Agent::new(
            Box::new(SucceedThenErrProvider::new(vec![tool_use_stream(
                "t1", "writer",
            )])),
            test_config(None),
            vec![side_effect_tool("writer", "wrote")],
        );

        let result = agent.run("save it", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(_))));
        // user → assistant(tool_use) → user(tool_result), all preserved.
        assert_eq!(agent.messages.len(), 3);
        assert_eq!(agent.messages[0], user_msg("save it"));
        assert_eq!(agent.messages[1], tool_use_msg("t1", "writer"));
        assert_eq!(agent.messages[2], tool_result_msg("t1", "wrote"));
    }

    #[test]
    fn run_keeps_history_when_side_effecting_tool_errored_then_stream_errors() {
        // Locks in the *pre*-`run` flag timing: a mutating tool that errors may
        // still have touched disk, so the flag is set before `run` and the
        // history survives a later stream error. Moving the flag to only the
        // `Ok` path would regress this case (and pass every other rollback
        // test, which all use a succeeding side-effecting tool).
        let mut agent = Agent::new(
            Box::new(SucceedThenErrProvider::new(vec![tool_use_stream(
                "t1", "writer",
            )])),
            test_config(None),
            vec![failing_side_effect_tool(
                "writer",
                "disk write failed midway",
            )],
        );

        let result = agent.run("save it", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(_))));
        assert_eq!(agent.messages.len(), 3);
        assert_eq!(agent.messages[0], user_msg("save it"));
        assert_eq!(agent.messages[1], tool_use_msg("t1", "writer"));
        // The failed mutation's tool_result is preserved with is_error: true.
        assert_eq!(
            agent.messages[2],
            TurnMessage {
                role: Role::User,
                content: vec![Block::ToolResult {
                    tool_use_id: "t1".to_string(),
                    content: "disk write failed midway".to_string(),
                    is_error: true,
                }],
            }
        );
    }

    #[test]
    fn run_keeps_history_when_side_effecting_tool_then_turn_limit() {
        // A turn-limit hit driven by a side-effecting tool is preserved by the
        // same rule as a stream error — disk diverged, so the history stays.
        let mut agent = Agent::new(
            Box::new(MockProvider::repeating(tool_use_stream("t1", "writer"))),
            test_config(None),
            vec![side_effect_tool("writer", "wrote")],
        );

        let result = agent.run("loop", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::TurnLimitExceeded(_))));
        // Initial user + MAX_TURNS × (assistant tool_use + user tool_result).
        assert_eq!(agent.messages.len(), 1 + 2 * MAX_TURNS as usize);
        assert_eq!(agent.messages[0], user_msg("loop"));
    }

    // ── salvage at the turn wall (nested children only) ──

    #[test]
    fn run_nested_salvages_partial_findings_at_the_wall() {
        // A nested child burns its whole round budget on tool calls, then the
        // wrap-up round returns text. The child returns Ok with that text under
        // the partial-findings marker instead of an error, and its transcript
        // is *kept* (no rollback) — the salvage converted Err→Ok. The kept
        // transcript *is* the wrap-up request `build_request` cloned verbatim,
        // so asserting on it pins the on-wire wrap-up round.
        let mut streams = streams_to_the_wall(tool_use_stream("t1", "echo"));
        streams.push(text_stream("what I found so far"));
        let mut agent = Agent::new(
            Box::new(MockProvider::new(streams)),
            test_config(None),
            vec![mock_tool("echo", "ok")],
        );

        let result = agent.run_nested("survey the repo", &mut std::io::sink());

        assert_eq!(
            result.unwrap(),
            format!("{SALVAGE_MARKER}\n\nwhat I found so far")
        );

        // Converting Err→Ok skips the child's rollback: the transcript stays.
        assert!(!agent.messages.is_empty());

        // The instruction is coalesced into the trailing user(tool_result)
        // message, not appended as a fresh user message — role alternation
        // holds (no two consecutive user messages).
        assert!(no_consecutive_users(&agent.messages));
        let last = agent.messages.last().unwrap();
        assert_eq!(last.role, Role::User);
        assert!(
            last.content
                .iter()
                .any(|b| matches!(b, Block::ToolResult { .. }))
        );
        assert_eq!(
            last.content.last(),
            Some(&Block::Text(WRAP_UP_INSTRUCTION.to_string()))
        );

        // The wrap-up request still defines the tools —
        // stripping them would break the wire contract. `build_request` over
        // the kept transcript reproduces exactly what the wrap-up round sent
        // (mirrors `build_request_includes_tools`).
        let wrap_up = agent.build_request();
        assert_eq!(wrap_up.tools.len(), 1);
        assert_eq!(wrap_up.tools[0].name, "echo");
        assert_eq!(wrap_up.messages, agent.messages);
    }

    #[test]
    fn run_nested_salvage_falls_back_to_last_assistant_text_on_tool_use_only_reply() {
        // The wrap-up reply is another tool_use (no text). Salvage falls back
        // to the transcript's last assistant text — the text each loop round
        // carried alongside its tool call.
        let mut streams =
            streams_to_the_wall(text_and_tool_use_stream("t1", "echo", "partial note"));
        streams.push(tool_use_stream("t1", "echo"));
        let mut agent = Agent::new(
            Box::new(MockProvider::new(streams)),
            test_config(None),
            vec![mock_tool("echo", "ok")],
        );

        let result = agent.run_nested("survey", &mut std::io::sink());

        assert_eq!(result.unwrap(), format!("{SALVAGE_MARKER}\n\npartial note"));
    }

    #[test]
    fn run_nested_salvage_falls_back_to_last_assistant_text_on_wrap_up_api_error() {
        // The queue holds only the MAX_TURNS loop streams; the wrap-up
        // round-trip finds an empty queue and errors (SucceedThenErrProvider
        // fails once its streams run out). Salvage treats the API error the
        // same as a text-less reply and falls back to the last assistant text.
        let streams = streams_to_the_wall(text_and_tool_use_stream("t1", "echo", "partial note"));
        let mut agent = Agent::new(
            Box::new(SucceedThenErrProvider::new(streams)),
            test_config(None),
            vec![mock_tool("echo", "ok")],
        );

        let result = agent.run_nested("survey", &mut std::io::sink());

        assert_eq!(result.unwrap(), format!("{SALVAGE_MARKER}\n\npartial note"));
    }

    #[test]
    fn run_nested_errors_at_the_wall_when_no_assistant_text_to_salvage() {
        // Every round is a pure tool call and the wrap-up reply is one too —
        // no assistant text exists anywhere. Nothing to salvage, so today's
        // TurnLimitExceeded stands and the child's transcript rolls back.
        let mut agent = Agent::new(
            Box::new(MockProvider::repeating(tool_use_stream("t1", "echo"))),
            test_config(None),
            vec![mock_tool("echo", "ok")],
        );

        let result = agent.run_nested("survey", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::TurnLimitExceeded(_))));
        // A pure read/API failure with no side effect rolls the turn back.
        assert!(agent.messages.is_empty());
    }

    #[test]
    fn run_owner_at_the_wall_never_salvages() {
        // The top-level owner path errors exactly as today even when the
        // transcript carries assistant text a child would have salvaged. The
        // provider errors once its streams run out, so *if* the owner wrongly
        // ran a wrap-up round it would fall back to that text and return Ok —
        // the `Err(TurnLimitExceeded)` here proves it never salvages.
        let streams =
            streams_to_the_wall(text_and_tool_use_stream("t1", "echo", "would-be salvage"));
        let mut agent = Agent::new(
            Box::new(SucceedThenErrProvider::new(streams)),
            test_config(None),
            vec![mock_tool("echo", "ok")],
        );

        let result = agent.run("survey", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::TurnLimitExceeded(_))));
        // The owner rolls its read-only transcript back.
        assert!(agent.messages.is_empty());
    }

    #[test]
    fn run_rolls_back_when_only_read_only_tool_ran_then_stream_errors() {
        // A read-only tool ran, but nothing on disk diverged, so a later stream
        // error still rolls the whole turn back — the mirror of the side-effect
        // case. ("a tool ran" alone is not the trigger; mutation is.)
        let mut agent = Agent::new(
            Box::new(SucceedThenErrProvider::new(vec![tool_use_stream(
                "t1", "echo",
            )])),
            test_config(None),
            vec![mock_tool("echo", "ok")],
        );

        let result = agent.run("read it", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(_))));
        assert!(agent.messages.is_empty());
    }

    #[test]
    fn run_rollback_restores_measured_prompt_size() {
        // An inner iteration succeeded and recorded its measurement before a
        // later iteration failed. The rollback that discards those messages
        // must discard the measurement taken from them too, or the compaction
        // guard would act on a context that no longer exists.
        let first = vec![
            StreamDelta::MessageStart {
                usage: Usage {
                    input_tokens: 999,
                    output_tokens: 0,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            },
            StreamDelta::ToolUseStart {
                index: 0,
                id: "t1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(SucceedThenErrProvider::new(vec![first])),
            test_config(None),
            vec![mock_tool("echo", "ok")],
        );
        agent.last_input_tokens = 100;

        let result = agent.run("read it", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(_))));
        assert_eq!(agent.last_input_tokens, 100);
    }

    #[test]
    fn run_side_effect_flag_does_not_leak_across_turns() {
        // Turn 1 runs a side-effecting tool and succeeds. Turn 2 runs only a
        // read-only tool, then errors: it must still roll back, proving the
        // per-turn flag was reset and turn 1's mutation does not protect turn 2.
        let mut agent = Agent::new(
            Box::new(SucceedThenErrProvider::new(vec![
                tool_use_stream("t1", "writer"),
                text_stream("done"),
                tool_use_stream("t2", "echo"),
            ])),
            test_config(None),
            vec![side_effect_tool("writer", "wrote"), mock_tool("echo", "ok")],
        );

        let first = agent.run("first", &mut std::io::sink()).unwrap();
        assert_eq!(first, "done");
        assert_eq!(agent.messages.len(), 4);

        let second = agent.run("second", &mut std::io::sink());

        assert!(matches!(second, Err(AgentError::Api(_))));
        // Turn 2 rolled back to turn 1's end state; turn 1's history survives.
        assert_eq!(agent.messages.len(), 4);
        assert_eq!(agent.messages[0], user_msg("first"));
        assert_eq!(agent.messages[3], assistant_msg("done"));
    }

    #[test]
    fn run_content_aware_rollback_strips_coalesced_text() {
        // A preserved side-effecting turn left history ending on a user
        // tool_result. The next input coalesces into that message; a pure
        // failure before any side effect must strip the coalesced text, not
        // just truncate by length (which would strand it — the length is
        // unchanged by a coalesce).
        let mut agent = Agent::new(Box::new(ErrProvider), test_config(None), vec![]);
        agent.messages.push(user_msg("save"));
        agent.messages.push(tool_use_msg("t1", "writer"));
        agent.messages.push(tool_result_msg("t1", "wrote"));

        let result = agent.run("again", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(_))));
        // The trailing tool_result is restored verbatim, without "again".
        assert_eq!(agent.messages.len(), 3);
        assert_eq!(agent.messages[0], user_msg("save"));
        assert_eq!(agent.messages[2], tool_result_msg("t1", "wrote"));
    }

    #[test]
    fn run_coalesces_next_input_into_preserved_tool_result() {
        // After a preserved turn ending on a user tool_result, the next input
        // is folded into that message (a user message may carry both
        // tool_result and text) rather than pushed as a second consecutive
        // user message, which Anthropic's alternating-role contract rejects.
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![text_stream("ok2")])),
            test_config(None),
            vec![],
        );
        agent.messages.push(user_msg("save"));
        agent.messages.push(tool_use_msg("t1", "writer"));
        agent.messages.push(tool_result_msg("t1", "wrote"));

        let text = agent.run("again", &mut std::io::sink()).unwrap();

        assert_eq!(text, "ok2");
        // No new user message: "again" joined the trailing tool_result message.
        assert_eq!(agent.messages.len(), 4);
        assert_eq!(agent.messages[2].role, Role::User);
        assert_eq!(
            agent.messages[2].content,
            vec![
                Block::ToolResult {
                    tool_use_id: "t1".to_string(),
                    content: "wrote".to_string(),
                    is_error: false,
                },
                Block::Text("again".to_string()),
            ]
        );
        assert_eq!(agent.messages[3], assistant_msg("ok2"));
    }

    // ── turn cancellation ──

    /// A shared flag and a tool whose `run` sets it — the hermetic stand-in
    /// for a Ctrl-C landing while a tool executes.
    fn cancelling_tool(name: &str, response: &str, flag: &Arc<AtomicBool>) -> TestTool {
        let handle = Arc::clone(flag);
        let response = response.to_string();
        TestTool::new(name, "").with_run(move |_| {
            handle.store(true, Ordering::Relaxed);
            Ok(response.clone())
        })
    }

    #[test]
    fn run_cancelled_during_a_tool_skips_the_next_request_and_rolls_back() {
        // One canned stream only: if the loop made a second request after the
        // cancellation, the MockProvider would panic on the missing stream.
        let flag = Arc::new(AtomicBool::new(false));
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![tool_use_stream("t1", "halt")])),
            test_config(None),
            vec![Box::new(cancelling_tool("halt", "done", &flag))],
        );
        agent.set_cancel_flag(Arc::clone(&flag));

        let result = agent.run("go", &mut Vec::new());
        assert!(matches!(result, Err(AgentError::Cancelled)));
        // A read-only turn rolls back like a failed one.
        assert!(agent.messages.is_empty());
        // The flag was consumed with the turn — the next one starts clean.
        assert!(!flag.load(Ordering::Relaxed));
    }

    #[test]
    fn run_cancelled_after_a_mutating_tool_keeps_history_and_skips_the_rest() {
        // The side-effect carve-out holds for cancellation: the mutating tool
        // already ran, so the transcript survives verbatim — and it is valid,
        // because the batch resolved the skipped sibling with an is_error
        // result instead of leaving its tool_use dangling.
        let flag = Arc::new(AtomicBool::new(false));
        let stream = vec![
            message_start(),
            StreamDelta::ToolUseStart {
                index: 0,
                id: "w1".to_string(),
                name: "writer".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::ToolUseStart {
                index: 1,
                id: "e1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            },
        ];
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![stream])),
            test_config(None),
            vec![
                Box::new(cancelling_tool("writer", "wrote", &flag).mutating()),
                mock_tool("echo", "never"),
            ],
        );
        agent.set_cancel_flag(Arc::clone(&flag));

        let result = agent.run("go", &mut Vec::new());
        assert!(matches!(result, Err(AgentError::Cancelled)));
        assert_eq!(agent.messages.len(), 3); // user, assistant, tool results
        let results = &agent.messages[2].content;
        assert_eq!(expect_tool_result(&results[0]), ("w1", "wrote", false));
        assert_eq!(
            expect_tool_result(&results[1]),
            ("e1", "cancelled by user", true)
        );
    }

    #[test]
    fn run_discards_a_cancellation_pending_from_between_turns() {
        // A stray SIGINT with no turn in flight (a cooked-mode gap, an
        // external `kill -INT`) must not abort the next turn: `run` resets
        // the flag before doing anything else.
        let flag = Arc::new(AtomicBool::new(false));
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![text_stream("hello")])),
            test_config(None),
            vec![],
        );
        agent.set_cancel_flag(Arc::clone(&flag));
        flag.store(true, Ordering::Relaxed);

        let text = agent.run("hi", &mut Vec::new()).unwrap();
        assert_eq!(text, "hello");
        assert_eq!(agent.messages.len(), 2);
        assert!(!flag.load(Ordering::Relaxed));
    }

    /// A sink that raises the cancellation flag when a newline is written —
    /// the trailing `writeln!` after the stream's last event, i.e. a Ctrl-C
    /// landing after every seam already passed.
    struct CancelOnNewline {
        flag: Arc<AtomicBool>,
        out: Vec<u8>,
    }
    impl Write for CancelOnNewline {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if buf.contains(&b'\n') {
                self.flag.store(true, Ordering::Relaxed);
            }
            self.out.write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.out.flush()
        }
    }

    #[test]
    fn run_keeps_a_turn_that_completed_before_the_cancellation_landed() {
        // The answer streamed in full before the flag went up, so the turn is
        // kept — cancelling finished work would discard a good answer. The
        // late flag is still consumed so the next turn starts clean.
        let flag = Arc::new(AtomicBool::new(false));
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![text_stream("hi")])),
            test_config(None),
            vec![],
        );
        agent.set_cancel_flag(Arc::clone(&flag));
        let mut out = CancelOnNewline {
            flag: Arc::clone(&flag),
            out: Vec::new(),
        };

        let text = agent.run("q", &mut out).unwrap();
        assert_eq!(text, "hi");
        assert_eq!(agent.messages.len(), 2);
        assert!(!flag.load(Ordering::Relaxed));
    }

    // ── model / set_model ──

    #[test]
    fn model_reports_the_configured_id() {
        let agent = agent_with_tools(vec![]);
        assert_eq!(agent.model(), crate::TEST_MODEL);
    }

    #[test]
    fn set_model_swaps_the_request_model() {
        let mut agent = agent_with_tools(vec![]);
        agent.set_model("some-other-model".to_string());

        assert_eq!(agent.model(), "some-other-model");
        // The swap flows through to the next request, not just the accessor.
        assert_eq!(agent.build_request().model, "some-other-model");
    }

    #[test]
    fn set_model_preserves_conversation_history() {
        // Switching the model is provider-agnostic over the turn model — the
        // history carries over so the next turn keeps its context.
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("hi"));
        agent.messages.push(assistant_msg("hello"));

        agent.set_model("some-other-model".to_string());

        // The exact turns survive, not merely the count — the swap touches only
        // the model, leaving every message intact for the next turn's context.
        assert_eq!(agent.messages.len(), 2);
        assert_eq!(agent.messages[0], user_msg("hi"));
        assert_eq!(agent.messages[1], assistant_msg("hello"));
    }

    // ── effort / set_effort ──

    #[test]
    fn effort_is_none_by_default_and_round_trips_through_set() {
        let mut agent = agent_with_tools(vec![]);
        assert!(agent.effort().is_none());
        agent.set_effort(Some("xhigh".to_string()));
        assert_eq!(agent.effort(), Some("xhigh"));
        // `None` clears it back to the model default.
        agent.set_effort(None);
        assert!(agent.effort().is_none());
    }

    #[test]
    fn set_model_keeps_the_effort() {
        // A same-provider model switch goes through `set_model` and must not
        // disturb the effort — only a provider change clears it.
        let mut agent = agent_with_tools(vec![]);
        agent.set_effort(Some("high".to_string()));
        agent.set_model("some-other-model".to_string());
        assert_eq!(agent.effort(), Some("high"));
    }

    #[test]
    fn set_provider_clears_the_effort() {
        // The extremes differ per provider, so a value set for one vendor is
        // meaningless under another — the switch drops it.
        let mut agent = agent_with_tools(vec![]);
        agent.set_effort(Some("high".to_string()));
        agent.set_provider(
            ProviderKind::Openai,
            Box::new(MockProvider::new(vec![])),
            "gpt-4o".to_string(),
        );
        assert!(agent.effort().is_none());
    }

    // ── provider_kind / set_provider ──

    #[test]
    fn provider_kind_reports_the_configured_provider() {
        let agent = agent_with_tools(vec![]);
        assert_eq!(agent.provider_kind(), ProviderKind::Anthropic);
    }

    #[test]
    fn set_provider_swaps_provider_kind_and_model() {
        // Seed history with a turn answered by the original provider, then
        // swap. The old provider has no second stream, so the post-swap turn
        // completing at all proves the box itself was replaced.
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![text_stream("from anthropic")])),
            test_config(None),
            vec![],
        );
        agent.run("hi", &mut std::io::sink()).unwrap();

        agent.set_provider(
            ProviderKind::Openai,
            Box::new(MockProvider::new(vec![text_stream("from openai")])),
            "gpt-4o".to_string(),
        );

        assert_eq!(agent.provider_kind(), ProviderKind::Openai);
        assert_eq!(agent.model(), "gpt-4o");
        // The swap flows through to the next request, not just the accessors.
        assert_eq!(agent.build_request().model, "gpt-4o");
        let text = agent.run("again", &mut std::io::sink()).unwrap();
        assert_eq!(text, "from openai");
        // History carries over untouched: the pre-swap turn plus the new one.
        assert_eq!(agent.messages.len(), 4);
        assert_eq!(agent.messages[0], user_msg("hi"));
        assert_eq!(agent.messages[1], assistant_msg("from anthropic"));
    }

    // ── list_models_cached ──

    #[test]
    fn list_models_cached_fetches_once_and_reuses() {
        let provider = MockProvider::new(vec![]).with_models(&["claude-a", "claude-b"]);
        let calls = provider.list_models_calls();
        let mut agent = Agent::new(Box::new(provider), test_config(None), vec![]);

        assert_eq!(agent.list_models_cached(), ["claude-a", "claude-b"]);
        assert_eq!(agent.list_models_cached(), ["claude-a", "claude-b"]);
        // The second read served the cache — one fetch for the session.
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn list_models_cached_offline_is_empty_not_an_error() {
        // Fail-soft: a provider that cannot list models yields no completions,
        // never an error to the user.
        let mut agent = Agent::new(Box::new(ErrProvider), test_config(None), vec![]);
        assert!(agent.list_models_cached().is_empty());
    }

    #[test]
    fn list_models_cached_survives_clear() {
        // The model listing is provider state, not conversation state — /clear
        // must not force a refetch.
        let provider = MockProvider::new(vec![]).with_models(&["claude-a"]);
        let calls = provider.list_models_calls();
        let mut agent = Agent::new(Box::new(provider), test_config(None), vec![]);
        agent.list_models_cached();

        agent.clear();

        assert_eq!(agent.list_models_cached(), ["claude-a"]);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn set_provider_drops_the_model_cache() {
        // A provider swap must not let the old provider's listing complete for
        // the new one.
        let provider = MockProvider::new(vec![]).with_models(&["claude-a"]);
        let mut agent = Agent::new(Box::new(provider), test_config(None), vec![]);
        assert_eq!(agent.list_models_cached(), ["claude-a"]);

        agent.set_provider(
            ProviderKind::Openai,
            Box::new(MockProvider::new(vec![]).with_models(&["gpt-4o"])),
            "gpt-4o".to_string(),
        );

        assert_eq!(agent.list_models_cached(), ["gpt-4o"]);
    }

    #[test]
    fn provider_default_list_models_is_empty() {
        // The trait default — a provider without a models endpoint (here the
        // agent-local sequenced double, which never overrides it) degrades to
        // "no completions" rather than erroring.
        let provider = SucceedThenErrProvider::new(vec![]);
        assert_eq!(provider.list_models().unwrap(), Vec::<String>::new());
    }

    // ── format_status ──

    #[test]
    fn format_status_default_returns_none() {
        let tool = mock_tool("echo", "ok");
        assert_eq!(tool.format_status(&serde_json::json!({"x": 1})), None);
    }

    #[test]
    fn format_status_override_returns_message() {
        let tool =
            TestTool::new("status_tool", "ok").with_status(|_| Some("doing something".to_string()));
        assert_eq!(
            tool.format_status(&serde_json::json!({})),
            Some("doing something".to_string())
        );
    }

    // ── confirm_with ──

    #[test]
    fn confirm_with_accepts_affirmative_replies() {
        for reply in ["y\n", "Y\n", "yes\n", "YES\n", "  y  \n"] {
            assert!(
                confirm_with(&mut reply.as_bytes(), &mut std::io::sink(), "shell: $ ls"),
                "expected approval for {reply:?}"
            );
        }
    }

    #[test]
    fn confirm_with_denies_everything_else() {
        // Anything but an explicit yes is a no — including the bare Enter
        // default the [y/N] prompt advertises.
        for reply in ["n\n", "N\n", "no\n", "\n", "yess\n", "y n\n"] {
            assert!(
                !confirm_with(&mut reply.as_bytes(), &mut std::io::sink(), "shell: $ ls"),
                "expected denial for {reply:?}"
            );
        }
    }

    #[test]
    fn confirm_with_denies_on_eof() {
        // A closed stdin (headless embedding, piped input that ran dry) reads
        // Ok(0) — the gate fails closed rather than silently approving.
        assert!(!confirm_with(
            &mut &b""[..],
            &mut std::io::sink(),
            "shell: $ ls"
        ));
    }

    #[test]
    fn confirm_with_denies_on_read_error() {
        struct FailingReader;
        impl std::io::Read for FailingReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("stdin broke"))
            }
        }
        let mut input = std::io::BufReader::new(FailingReader);
        assert!(!confirm_with(
            &mut input,
            &mut std::io::sink(),
            "shell: $ ls"
        ));
    }

    #[test]
    fn confirm_with_writes_prompt_to_the_error_writer() {
        let mut err = Vec::new();
        confirm_with(&mut &b"y\n"[..], &mut err, "shell: $ rm -rf src");
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "Allow shell: $ rm -rf src? [y/N] "
        );
    }

    #[test]
    fn confirm_with_displays_multiline_preview_and_fails_closed_if_display_breaks() {
        let mut err = Vec::new();
        assert!(confirm_with(
            &mut &b"y\n"[..],
            &mut err,
            "write file\n- old\n+ new"
        ));
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "write file\n- old\n+ new\nAllow this change? [y/N] "
        );
        struct BrokenDisplay {
            fail_flush: bool,
        }
        impl Write for BrokenDisplay {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.fail_flush {
                    Ok(bytes.len())
                } else {
                    Err(std::io::Error::other("display broke"))
                }
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::other("flush broke"))
            }
        }
        for fail_flush in [false, true] {
            let mut input = &b"y\n"[..];
            assert!(!confirm_with(
                &mut input,
                &mut BrokenDisplay { fail_flush },
                "preview\n+ change"
            ));
            assert_eq!(
                input, b"y\n",
                "do not consume approval when the preview was not displayed"
            );
        }
    }

    // ── summarize_output ──
}
