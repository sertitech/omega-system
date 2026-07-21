//! The `task` tool: delegate a subtask to a named subagent profile.
//!
//! Each call spawns a fresh [`Agent`] on the profile's provider, model, and
//! effort, drives it to completion, and returns only its final text — the
//! child transcript is ephemeral. A **read-only** profile's
//! toolset never mutates, so [`TaskTool::side_effecting`] is `false` for it and
//! several such `task` calls in one assistant turn parallelize through the
//! agent loop's existing read-only fan-out. An **executor** profile
//! additionally grants a mutating tool: `side_effecting` is then `true`, which
//! forces the whole batch inline/serial and keeps the parent turn's history,
//! and each dangerous child call goes through the parent's confirmation
//! policy — prompting the parent's TTY under `ask`, adjudicated or
//! waved under `judge`/`allow`, with a profile `confirm` field overriding the
//! session mode. Every field is `Send + Sync` so the tool can still fan
//! read-only children across worker threads.
//!
//! Child output is live: everything the child writes — streamed
//! text, tool chatter — passes through a [`PrefixWriter`] to the sink the
//! agent loop hands `run`, each line prefixed with the profile name and
//! scrubbed for the terminal. An inline (serial) child therefore streams to
//! the operator as it works; a fanned-out child writes into the private
//! buffer the loop flushes in block order after the batch. Either way the
//! tool *result* is unchanged — only the child's final text, never the
//! stream.

use super::sandbox::Sandbox;
use super::{
    ToolDef, edit_file, list_directory, read_file, search_files, shell, web_fetch, web_search,
    write_file,
};
use crate::agent::{Agent, AgentConfig, BudgetLedger, Concurrency, ConfirmPolicy};
use crate::config::AgentProfile;
use crate::display::sanitize_for_display;
use crate::provider::ProviderFactory;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// The read-only tools a child profile may be granted — and the default set
/// when a profile omits `tools`. A child with only these is guaranteed
/// non-side-effecting, which is what lets its `task` calls parallelize for free.
const READ_ONLY_CHILD_TOOLS: [&str; 5] = [
    "read_file",
    "list_directory",
    "search_files",
    "web_fetch",
    "web_search",
];

/// The mutating tools an executor profile may be granted. A profile
/// listing any of these becomes side-effecting: the agent loop forces its
/// `task` calls inline/serial and keeps the parent turn's history, and the
/// child prompts the parent TTY before each dangerous call.
const MUTATING_CHILD_TOOLS: [&str; 3] = ["write_file", "edit_file", "shell"];

/// Whether `name` is a tool a child profile's allowlist may list — read-only or
/// mutating. Backs the config-load validation ([`crate::config`]) so a typo
/// fails the parse rather than surfacing at spawn time.
pub(crate) fn is_valid_child_tool(name: &str) -> bool {
    READ_ONLY_CHILD_TOOLS.contains(&name) || MUTATING_CHILD_TOOLS.contains(&name)
}

/// The default child toolset when a profile omits `tools`: the read-only set.
/// `web_search` is included but only materializes when a firecrawl key is
/// available (see [`build_child_tool`]) — matching `default_tools`' rule.
pub(crate) fn default_child_tool_names() -> Vec<String> {
    READ_ONLY_CHILD_TOOLS
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Construct one child tool by name, sandbox-resolved exactly like the parent's
/// own instances. Returns `None` for `web_search` when no firecrawl key is
/// available (the tool cannot be built without it, so the child silently goes
/// without — the `default_tools` convention) and for any name outside the valid
/// set (validation rejects those at load, so this arm is the belt-and-braces
/// default). `cancel` threads the shared turn-cancellation flag into `shell`,
/// exactly as `default_tools` does for the parent.
fn build_child_tool(
    name: &str,
    sandbox: &Sandbox,
    firecrawl_key: Option<&str>,
    cancel: &Arc<AtomicBool>,
) -> Option<Box<dyn ToolDef>> {
    match name {
        "read_file" => Some(Box::new(read_file::ReadFileTool::new(sandbox.clone()))),
        "list_directory" => Some(Box::new(list_directory::ListDirectoryTool::new(
            sandbox.clone(),
        ))),
        "search_files" => Some(Box::new(search_files::SearchFilesTool::new(
            sandbox.clone(),
        ))),
        "web_fetch" => Some(Box::new(web_fetch::WebFetchTool::default())),
        "web_search" => firecrawl_key.map(|key| {
            Box::new(web_search::WebSearchTool::new(key.to_string())) as Box<dyn ToolDef>
        }),
        "write_file" => Some(Box::new(write_file::WriteFileTool::new(sandbox.clone()))),
        "edit_file" => Some(Box::new(edit_file::EditFileTool::new(sandbox.clone()))),
        "shell" => Some(Box::new(shell::ShellTool::new(
            sandbox.clone(),
            Arc::clone(cancel),
        ))),
        _ => None,
    }
}

/// Build the child toolset for `profile` — its explicit allowlist, or the
/// read-only default when it omits `tools` — each instance sandbox-resolved
/// exactly like the parent's own. Shared by [`TaskTool::child_tools`], which
/// spawns the child agent over real seams, and [`profile_is_executor`], which
/// only inspects the built tools' markers over throwaway ones.
pub(crate) fn build_child_tools(
    profile: &AgentProfile,
    sandbox: &Sandbox,
    firecrawl_key: Option<&str>,
    cancel: &Arc<AtomicBool>,
) -> Vec<Box<dyn ToolDef>> {
    let names = profile
        .tools
        .clone()
        .unwrap_or_else(default_child_tool_names);
    names
        .iter()
        .filter_map(|name| build_child_tool(name, sandbox, firecrawl_key, cancel))
        .collect()
}

/// Whether `profile` is an executor — its toolset grants a mutating tool. The
/// tools' own [`ToolDef::side_effecting`] markers are the single source of
/// truth (they are input-independent, so no argument is needed), which is why
/// this queries the built toolset rather than a second name list that could
/// drift from it. Shared by the `task` tool — driving both the per-input
/// [`ToolDef::side_effecting`] the agent loop reads and the child confirm
/// policy — and the `/agents` report, so the operator sees exactly the
/// classification a delegation will get. The markers are type-level, so a
/// throwaway read-only sandbox and no firecrawl key classify identically to the
/// live seams (only `web_search`, itself read-only, is key-gated).
pub(crate) fn profile_is_executor(profile: &AgentProfile) -> bool {
    let cancel = Arc::new(AtomicBool::new(false));
    build_child_tools(profile, &Sandbox::unbounded(), None, &cancel)
        .iter()
        .any(|t| t.side_effecting(&serde_json::Value::Null))
}

/// A line-buffered writer that relays a child agent's output onto the parent's
/// display sink, prefixing every line with the delegating profile's name and
/// scrubbing it through [`sanitize_for_display`] — child output is model text,
/// an indirect-injection channel, and this is the one point where it crosses
/// onto the operator's terminal. Lines are emitted only when complete (the
/// child's own mid-line flushes pass through without splitting), buffered on
/// the `\n` byte — which is never part of a multi-byte UTF-8 sequence, so a
/// character split across `write` calls reassembles intact. [`Write::flush`]
/// flushes the inner sink only; [`PrefixWriter::finish`] drains a trailing
/// unterminated line when the child is done.
struct PrefixWriter<'a> {
    inner: &'a mut dyn Write,
    /// `"[name] "`, sanitized once at construction.
    prefix: String,
    /// The current, not-yet-terminated line.
    line: Vec<u8>,
}

impl<'a> PrefixWriter<'a> {
    fn new(inner: &'a mut dyn Write, profile: &str) -> Self {
        Self {
            inner,
            prefix: format!("[{}] ", sanitize_for_display(profile)),
            line: Vec::new(),
        }
    }

    /// Emit one complete line: prefix, scrubbed content, newline.
    fn emit(&mut self, line: &[u8]) -> std::io::Result<()> {
        let text = String::from_utf8_lossy(line);
        writeln!(self.inner, "{}{}", self.prefix, sanitize_for_display(&text))
    }

    /// Drain a trailing unterminated line, if any. Called once the child has
    /// finished writing — mid-stream it would split a line in two.
    fn finish(&mut self) -> std::io::Result<()> {
        if self.line.is_empty() {
            return Ok(());
        }
        let line = std::mem::take(&mut self.line);
        self.emit(&line)
    }
}

impl Write for PrefixWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.line.extend_from_slice(buf);
        while let Some(pos) = self.line.iter().position(|&b| b == b'\n') {
            let rest = self.line.split_off(pos + 1);
            let line = std::mem::replace(&mut self.line, rest);
            self.emit(&line[..line.len() - 1])?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// The `task` tool. Holds the seams a child needs — the provider factory with
/// lazy key resolution, the sandbox, the firecrawl key, the shared cancel
/// flag and budget ledger, and the base confirmation policy executor children
/// prompt through — plus the profile table and the token budgets a
/// child request inherits from the parent. All `Send + Sync`, so the agent loop
/// can fan several read-only `task` calls out across worker threads; the child
/// `Agent` itself is built *inside* [`TaskTool::run`] and never crosses a
/// thread boundary, so [`crate::provider::Provider`] needs no `Send` bound.
pub struct TaskTool {
    factory: ProviderFactory,
    sandbox: Sandbox,
    firecrawl_key: Option<String>,
    cancel: Arc<AtomicBool>,
    budget: BudgetLedger,
    /// The process-wide fan-out concurrency pool, cloned into every child agent
    /// so a child's leaf fan-out contends for the same permits as the parent —
    /// one ceiling across the whole delegation tree.
    concurrency: Concurrency,
    /// The session confirmation policy, cloned per child. An executor child
    /// gets it labeled with its profile (and pinned when the profile carries
    /// a `confirm` override); a read-only child gets its inert deny. The mode
    /// cell is shared, so a `/confirm` switch reaches children spawned later.
    confirm: ConfirmPolicy,
    profiles: Vec<AgentProfile>,
    /// Output-token cap for a child request, inherited from the parent's
    /// `max_tokens` (a profile carries no separate knob).
    max_tokens: u32,
    /// The child's compaction-guard ceiling, inherited from the parent's
    /// `context_token_limit`.
    context_token_limit: u32,
    /// The default tool-loop round budget for a child, inherited from the
    /// parent's `max_turns`. A spawned profile's own `max_turns` overrides it
    /// per child (`profile.max_turns.unwrap_or(self.max_turns)`).
    max_turns: u32,
    /// The tool description, built once at construction so it can name the
    /// configured profiles (the model picks one by name). Held because
    /// [`ToolDef::description`] returns a borrow.
    description: String,
}

impl TaskTool {
    /// Wire a `task` tool over the given seams and profile table. `max_tokens`,
    /// `context_token_limit`, and `max_turns` are the parent agent's, inherited
    /// by every child request (a profile's own `max_turns` overrides the last).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        factory: ProviderFactory,
        sandbox: Sandbox,
        firecrawl_key: Option<String>,
        cancel: Arc<AtomicBool>,
        budget: BudgetLedger,
        concurrency: Concurrency,
        confirm: ConfirmPolicy,
        profiles: Vec<AgentProfile>,
        max_tokens: u32,
        context_token_limit: u32,
        max_turns: u32,
    ) -> Self {
        let names: Vec<&str> = profiles.iter().map(|p| p.name.as_str()).collect();
        let description = format!(
            "Delegate a subtask to a named subagent, which runs to completion on \
             its own provider and model and returns only its final answer. \
             Available profiles: {}. Give a self-contained `prompt` — the child \
             sees none of this conversation. Optionally set `effort` to override \
             the profile's reasoning effort for this call.",
            names.join(", ")
        );
        Self {
            factory,
            sandbox,
            firecrawl_key,
            cancel,
            budget,
            concurrency,
            confirm,
            profiles,
            max_tokens,
            context_token_limit,
            max_turns,
            description,
        }
    }

    /// Find a profile by name.
    fn profile(&self, name: &str) -> Option<&AgentProfile> {
        self.profiles.iter().find(|p| p.name == name)
    }

    /// Build the child toolset for `profile` — its allowlist, or the read-only
    /// default when it omits `tools` — over this tool's live seams.
    fn child_tools(&self, profile: &AgentProfile) -> Vec<Box<dyn ToolDef>> {
        build_child_tools(
            profile,
            &self.sandbox,
            self.firecrawl_key.as_deref(),
            &self.cancel,
        )
    }

    /// Whether `profile` is an executor — see the free [`profile_is_executor`],
    /// the shared classifier the `/agents` report reuses so the two never
    /// drift. Drives both the per-input [`ToolDef::side_effecting`] the loop
    /// reads and the choice of child confirm in [`TaskTool::run`].
    fn profile_is_executor(&self, profile: &AgentProfile) -> bool {
        profile_is_executor(profile)
    }

    /// The confirmation policy the spawned child runs under. An executor
    /// child runs inline on the parent's main thread (forced there by
    /// `side_effecting`), so its `ask` prompts reach the parent's terminal —
    /// labeled with the profile name, so the operator sees which delegation
    /// is asking. It inherits the session policy (the shared mode cell, so a
    /// `/confirm` switch reaches it) unless the profile pins its own
    /// `confirm` override, which takes precedence — e.g. an executor that
    /// always goes through the judge even when the session runs `allow`. A
    /// read-only child fans out on a worker thread and reaches no confirmable
    /// tool, so it carries the policy's inert deny — a quiet refusal rather
    /// than a prompt interleaving with the main thread's stdin.
    fn child_policy(&self, profile: &AgentProfile, executor: bool) -> ConfirmPolicy {
        if !executor {
            return self.confirm.inert_deny();
        }
        let labeled = self.confirm.labeled(&profile.name);
        match &profile.confirm {
            Some(mode) => labeled.pinned(mode.clone()),
            None => labeled,
        }
    }
}

impl ToolDef for TaskTool {
    fn name(&self) -> &str {
        "task"
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "profile": {
                    "type": "string",
                    "description": "The subagent profile to delegate to."
                },
                "prompt": {
                    "type": "string",
                    "description": "The self-contained task for the subagent."
                },
                "effort": {
                    "type": "string",
                    "description": "Optional reasoning-effort override for this call."
                }
            },
            "required": ["profile", "prompt"]
        })
    }

    /// Tier 4 — the most expensive tier, so the shared budget ledger's tier-4
    /// ceiling (10/session) bounds how many children a session may spawn.
    fn cost(&self) -> u8 {
        4
    }

    /// The `task` tool does not draw a fan-out permit: its worker drives a
    /// child agent whose own leaf tools acquire from the same pool, so holding
    /// a permit here would risk deadlocking a batch of tasks against their
    /// children (see [`ToolDef::gates_concurrency`]). Task breadth stays bounded
    /// by the tier-4 budget ceiling instead.
    fn gates_concurrency(&self) -> bool {
        false
    }

    /// Per-input: a delegation to an executor profile (its toolset grants a
    /// mutating tool) is side-effecting, so the agent loop forces the whole
    /// batch inline/serial and marks the parent turn rollback-exempt — the same
    /// treatment a direct `write_file`/`shell` call gets. A read-only or unknown
    /// profile is not: read-only tasks keep parallelizing, and an unknown
    /// profile is rejected by `validate`/`run` regardless.
    fn side_effecting(&self, input: &serde_json::Value) -> bool {
        input["profile"]
            .as_str()
            .and_then(|name| self.profile(name))
            .is_some_and(|profile| self.profile_is_executor(profile))
    }

    /// Reject an unknown profile or an empty prompt before the call runs, so
    /// the model self-corrects via `is_error` instead of the spawn failing.
    fn validate(&self, input: &serde_json::Value) -> Result<(), String> {
        let profile = input["profile"]
            .as_str()
            .ok_or("missing required field: profile")?;
        if self.profile(profile).is_none() {
            return Err(format!("unknown profile: {profile}"));
        }
        let prompt = input["prompt"]
            .as_str()
            .ok_or("missing required field: prompt")?;
        if prompt.trim().is_empty() {
            return Err("prompt must not be empty".to_string());
        }
        Ok(())
    }

    /// Name the profile and the model it resolves to, e.g.
    /// `task(reviewer: gpt-5.6-sol)`, so the operator sees each delegation
    /// before it runs.
    fn format_status(&self, input: &serde_json::Value) -> Option<String> {
        let name = input["profile"].as_str()?;
        let profile = self.profile(name)?;
        Some(format!("task({}: {})", profile.name, profile.model))
    }

    fn run(&self, input: serde_json::Value, out: &mut dyn Write) -> Result<String, String> {
        let profile_name = input["profile"]
            .as_str()
            .ok_or("missing required field: profile")?;
        let prompt = input["prompt"]
            .as_str()
            .ok_or("missing required field: prompt")?;
        // A per-call effort argument beats the profile's; absent both, the
        // child runs at the provider's default.
        let call_effort = input["effort"].as_str().map(str::to_string);

        let profile = self
            .profile(profile_name)
            .ok_or_else(|| format!("unknown profile: {profile_name}"))?;

        // Resolve the child provider's key lazily, at spawn time, through the
        // step-2 seam — a provider the operator never delegates to costs no
        // lookup, and a key added to `.env` mid-session is still seen.
        let env = profile.provider.api_key_env();
        let key = self
            .factory
            .resolve_key(env)
            .ok_or_else(|| format!("no API key for provider {} (set {env})", profile.provider))?;
        let provider = self.factory.build(profile.provider, key);

        let config = AgentConfig {
            provider_kind: profile.provider,
            model: profile.model.clone(),
            max_tokens: self.max_tokens,
            system: profile.system.clone(),
            context_token_limit: self.context_token_limit,
            effort: call_effort.or_else(|| profile.effort.clone()),
            // The profile's own limit wins; absent, the child inherits the
            // parent's — a survey-shaped profile can run more rounds than the
            // top-level executor default.
            max_turns: profile.max_turns.unwrap_or(self.max_turns),
        };
        let mut agent = Agent::new(provider, config, self.child_tools(profile));
        // Share the parent's budget ledger and cancel flag: children draw
        // against the same process-wide ceilings, and a Ctrl-C stops parent
        // and children together (the child only observes the flag — see
        // `run_nested`).
        agent.set_budget_ledger(self.budget.clone());
        agent.set_cancel_flag(Arc::clone(&self.cancel));
        // Share the parent's fan-out pool: the child's own leaf fan-out draws
        // permits from the same ceiling, so the whole delegation tree stays
        // bounded by one process-wide cap.
        agent.set_concurrency(self.concurrency.clone());
        // An executor child inherits the session confirmation policy — its
        // `ask` prompts land at the parent's TTY naming this profile, unless
        // the profile pins its own mode; a read-only child carries an inert
        // deny. Because `side_effecting` forces an executor `task` inline,
        // an interactive prompt only ever fires on the parent's main thread.
        agent.set_confirm_policy(self.child_policy(profile, self.profile_is_executor(profile)));

        // The child's streamed text and tool chatter relay live onto the
        // loop-provided sink, each line prefixed with the profile and
        // scrubbed. Write failures are ignored here as everywhere display
        // output is written — the display is best-effort, the tool result
        // below is the contract.
        let mut live = PrefixWriter::new(out, &profile.name);
        let result = agent
            .run_nested(prompt, &mut live)
            .map_err(|e| format!("subagent '{}' failed: {e}", profile.name));
        let _ = live.finish();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ProviderKind;
    use crate::testing::{ThreadSafeProvider, expect_tool_result};
    use crate::turn::{Block, Role, StopReason, StreamDelta, Usage};

    // ── Fixtures ──

    /// A profile with the read-only default toolset.
    fn profile(name: &str, provider: ProviderKind, model: &str) -> AgentProfile {
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

    /// A factory that builds prompt-echoing `ThreadSafeProvider`s and resolves
    /// keys through `resolve`. One shared definition of the build closure so
    /// every test's provider construction runs the same covered line.
    fn echo_factory(
        resolve: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
    ) -> ProviderFactory {
        ProviderFactory::from_fns(|_kind, _key| Box::new(ThreadSafeProvider::echo()), resolve)
    }

    /// A `TaskTool` over a factory that builds `ThreadSafeProvider`s (echoing
    /// the prompt) and resolves every key. The shared `cancel`/`budget` handles
    /// are returned so a test can observe them.
    fn echo_task(profiles: Vec<AgentProfile>) -> (TaskTool, Arc<AtomicBool>, BudgetLedger) {
        echo_task_keyed(profiles, None)
    }

    /// [`echo_task`] with an explicit firecrawl key, for the child-toolset
    /// `web_search` case. The resolver here is exercised by the running
    /// `echo_task` tests, so it carries coverage even when a caller only
    /// inspects the child toolset.
    fn echo_task_keyed(
        profiles: Vec<AgentProfile>,
        firecrawl_key: Option<String>,
    ) -> (TaskTool, Arc<AtomicBool>, BudgetLedger) {
        let factory = echo_factory(|_env| Some("stub-key".to_string()));
        let cancel = Arc::new(AtomicBool::new(false));
        let budget = BudgetLedger::new();
        let tool = TaskTool::new(
            factory,
            Sandbox::unbounded(),
            firecrawl_key,
            Arc::clone(&cancel),
            budget.clone(),
            Concurrency::new(),
            approve_all(),
            profiles,
            256,
            100_000,
            crate::agent::MAX_TURNS,
        );
        (tool, cancel, budget)
    }

    /// A session policy that approves everything at the `ask` prompt — the
    /// read-only fixtures never reach it (their children carry the inert
    /// deny), so its answer is immaterial there; the executor tests build
    /// their own recording policies instead.
    fn approve_all() -> ConfirmPolicy {
        ask_session(|_| true)
    }

    /// An `ask`-mode session policy answering through `f`, over the shared
    /// inert seams — what `main` builds from a default config, minus the
    /// real TTY.
    fn ask_session(f: impl Fn(&str) -> bool + Send + Sync + 'static) -> ConfirmPolicy {
        ConfirmPolicy::new(
            crate::config::ConfirmMode::Ask,
            f,
            ProviderFactory::from_fns(crate::testing::stub_provider, crate::testing::no_key),
            Vec::new(),
            Sandbox::unbounded(),
        )
    }

    fn anthropic_profiles() -> Vec<AgentProfile> {
        vec![profile("reviewer", ProviderKind::Anthropic, "some-model")]
    }

    // ── Metadata & schema ──

    #[test]
    fn metadata_names_the_tool_and_lists_profiles() {
        let (tool, _, _) = echo_task(anthropic_profiles());
        assert_eq!(tool.name(), "task");
        assert_eq!(tool.cost(), 4);
        assert!(!tool.requires_confirmation());
        // A read-only profile makes the delegation non-side-effecting.
        assert!(!tool.side_effecting(&serde_json::json!({"profile": "reviewer"})));
        // The description names the configured profiles so the model can pick.
        assert!(tool.description().contains("reviewer"));

        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "profile"));
        assert!(required.iter().any(|v| v == "prompt"));
        assert!(schema["properties"]["effort"].is_object());
    }

    #[test]
    fn format_status_names_profile_and_model() {
        let (tool, _, _) = echo_task(anthropic_profiles());
        let status = tool.format_status(&serde_json::json!({"profile": "reviewer"}));
        assert_eq!(status.as_deref(), Some("task(reviewer: some-model)"));
    }

    #[test]
    fn format_status_none_for_unknown_or_absent_profile() {
        let (tool, _, _) = echo_task(anthropic_profiles());
        assert!(tool.format_status(&serde_json::json!({})).is_none());
        assert!(
            tool.format_status(&serde_json::json!({"profile": "ghost"}))
                .is_none()
        );
    }

    // ── validate ──

    #[test]
    fn validate_accepts_a_known_profile_and_prompt() {
        let (tool, _, _) = echo_task(anthropic_profiles());
        assert!(
            tool.validate(&serde_json::json!({"profile": "reviewer", "prompt": "hi"}))
                .is_ok()
        );
    }

    #[test]
    fn validate_rejects_unknown_profile() {
        let (tool, _, _) = echo_task(anthropic_profiles());
        let err = tool
            .validate(&serde_json::json!({"profile": "ghost", "prompt": "hi"}))
            .unwrap_err();
        assert!(err.contains("unknown profile: ghost"), "got: {err}");
    }

    #[test]
    fn validate_rejects_missing_profile_field() {
        let (tool, _, _) = echo_task(anthropic_profiles());
        let err = tool
            .validate(&serde_json::json!({"prompt": "hi"}))
            .unwrap_err();
        assert!(
            err.contains("missing required field: profile"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_rejects_missing_prompt_field() {
        let (tool, _, _) = echo_task(anthropic_profiles());
        let err = tool
            .validate(&serde_json::json!({"profile": "reviewer"}))
            .unwrap_err();
        assert!(err.contains("missing required field: prompt"), "got: {err}");
    }

    #[test]
    fn validate_rejects_blank_prompt() {
        let (tool, _, _) = echo_task(anthropic_profiles());
        let err = tool
            .validate(&serde_json::json!({"profile": "reviewer", "prompt": "   "}))
            .unwrap_err();
        assert!(err.contains("prompt must not be empty"), "got: {err}");
    }

    // ── run: success, echo, effort ──

    #[test]
    fn run_returns_the_child_final_text() {
        // The echo provider replies with the child's prompt, so a successful
        // delegation returns exactly that prompt as the tool result.
        let (tool, _, _) = echo_task(anthropic_profiles());
        let out = tool
            .run(
                serde_json::json!({"profile": "reviewer", "prompt": "review this"}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert_eq!(out, "review this");
    }

    #[test]
    fn run_rejects_unknown_profile_directly() {
        // `run` re-checks the profile (it may be called without `validate`).
        let (tool, _, _) = echo_task(anthropic_profiles());
        let err = tool
            .run(
                serde_json::json!({"profile": "ghost", "prompt": "hi"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("unknown profile: ghost"), "got: {err}");
    }

    #[test]
    fn run_rejects_missing_fields_directly() {
        let (tool, _, _) = echo_task(anthropic_profiles());
        assert!(
            tool.run(serde_json::json!({"prompt": "hi"}), &mut std::io::sink())
                .unwrap_err()
                .contains("missing required field: profile")
        );
        assert!(
            tool.run(
                serde_json::json!({"profile": "reviewer"}),
                &mut std::io::sink()
            )
            .unwrap_err()
            .contains("missing required field: prompt")
        );
    }

    /// A `task` tool whose child provider echoes but records each request's
    /// effort into the returned log — so a test can assert the effort the tool
    /// resolved for the child.
    #[allow(clippy::type_complexity)]
    fn effort_task(
        profiles: Vec<AgentProfile>,
    ) -> (TaskTool, Arc<std::sync::Mutex<Vec<Option<String>>>>) {
        let log: Arc<std::sync::Mutex<Vec<Option<String>>>> = Arc::default();
        let factory = {
            let log = Arc::clone(&log);
            ProviderFactory::from_fns(
                move |_kind, _key| {
                    Box::new(ThreadSafeProvider::echo().with_effort_log(Arc::clone(&log)))
                },
                |_env| Some("k".to_string()),
            )
        };
        let tool = TaskTool::new(
            factory,
            Sandbox::unbounded(),
            None,
            Arc::new(AtomicBool::new(false)),
            BudgetLedger::new(),
            Concurrency::new(),
            approve_all(),
            profiles,
            256,
            100_000,
            crate::agent::MAX_TURNS,
        );
        (tool, log)
    }

    #[test]
    fn per_call_effort_overrides_the_profile() {
        let mut p = profile("reviewer", ProviderKind::Anthropic, "m");
        p.effort = Some("low".to_string());
        let (tool, seen) = effort_task(vec![p]);
        tool.run(
            serde_json::json!({"profile": "reviewer", "prompt": "hi", "effort": "xhigh"}),
            &mut std::io::sink(),
        )
        .unwrap();
        assert_eq!(seen.lock().unwrap().as_slice(), [Some("xhigh".to_string())]);
    }

    #[test]
    fn profile_effort_applies_when_no_override() {
        let mut p = profile("reviewer", ProviderKind::Anthropic, "m");
        p.effort = Some("high".to_string());
        let (tool, seen) = effort_task(vec![p]);
        tool.run(
            serde_json::json!({"profile": "reviewer", "prompt": "hi"}),
            &mut std::io::sink(),
        )
        .unwrap();
        assert_eq!(seen.lock().unwrap().as_slice(), [Some("high".to_string())]);
    }

    // ── run: child failure & missing key ──

    #[test]
    fn run_surfaces_child_api_error() {
        // The child provider's stream fails at the transport layer; the tool
        // wraps it as an error result naming the profile.
        let factory = ProviderFactory::from_fns(
            |_kind, _key| Box::new(ThreadSafeProvider::failing()),
            |_env| Some("k".to_string()),
        );
        let tool = TaskTool::new(
            factory,
            Sandbox::unbounded(),
            None,
            Arc::new(AtomicBool::new(false)),
            BudgetLedger::new(),
            Concurrency::new(),
            approve_all(),
            anthropic_profiles(),
            256,
            100_000,
            crate::agent::MAX_TURNS,
        );
        let err = tool
            .run(
                serde_json::json!({"profile": "reviewer", "prompt": "hi"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("subagent 'reviewer' failed"), "got: {err}");
    }

    #[test]
    fn run_reports_a_missing_key() {
        // The resolver misses the child provider's key — a fail-fast error, not
        // a silent spawn.
        let factory = echo_factory(|_env| None);
        let tool = TaskTool::new(
            factory,
            Sandbox::unbounded(),
            None,
            Arc::new(AtomicBool::new(false)),
            BudgetLedger::new(),
            Concurrency::new(),
            approve_all(),
            anthropic_profiles(),
            256,
            100_000,
            crate::agent::MAX_TURNS,
        );
        let err = tool
            .run(
                serde_json::json!({"profile": "reviewer", "prompt": "hi"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("no API key"), "got: {err}");
        assert!(err.contains("ANTHROPIC_API_KEY"), "got: {err}");
    }

    // ── run: cancellation observed mid-child ──

    #[test]
    fn run_observes_a_pending_cancel_without_consuming_it() {
        // A flag already set before the spawn stops the child at its first
        // pre-request check, and the observer child must leave it set so the
        // parent's owning run still sees the Ctrl-C.
        let (tool, cancel, _) = echo_task(anthropic_profiles());
        cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        let err = tool
            .run(
                serde_json::json!({"profile": "reviewer", "prompt": "hi"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("subagent 'reviewer' failed"), "got: {err}");
        assert!(
            cancel.load(std::sync::atomic::Ordering::Relaxed),
            "the child must not consume the parent's cancel flag"
        );
    }

    // ── child toolset construction ──

    #[test]
    fn child_tools_default_to_the_read_only_set() {
        let (tool, _, _) = echo_task(anthropic_profiles());
        let tools = tool.child_tools(&anthropic_profiles()[0]);
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        // No firecrawl key here, so web_search is silently omitted.
        assert_eq!(
            names,
            ["read_file", "list_directory", "search_files", "web_fetch"]
        );
    }

    #[test]
    fn child_tools_include_web_search_when_keyed() {
        let (tool, _, _) = echo_task_keyed(anthropic_profiles(), Some("fc-key".to_string()));
        let tools = tool.child_tools(&anthropic_profiles()[0]);
        assert!(tools.iter().any(|t| t.name() == "web_search"));
    }

    #[test]
    fn child_tools_honor_an_explicit_allowlist() {
        let mut p = profile("reader", ProviderKind::Anthropic, "m");
        p.tools = Some(vec!["read_file".to_string()]);
        let (tool, _, _) = echo_task(vec![p.clone()]);
        let tools = tool.child_tools(&p);
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(names, ["read_file"]);
    }

    #[test]
    fn child_tools_build_the_mutating_set_for_executor_profiles() {
        // Step 5: an executor profile may grant the mutating tools, and the
        // builder constructs them sandbox-resolved like the parent's own.
        let mut p = profile("editor", ProviderKind::Anthropic, "m");
        p.tools = Some(vec![
            "write_file".to_string(),
            "edit_file".to_string(),
            "shell".to_string(),
        ]);
        let (tool, _, _) = echo_task(vec![p.clone()]);
        let tools = tool.child_tools(&p);
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(names, ["write_file", "edit_file", "shell"]);
        // Any mutating tool makes the profile an executor.
        assert!(tool.profile_is_executor(&p));
    }

    // ── side-effecting classification (per input) ──

    #[test]
    fn side_effecting_true_for_a_mutating_profile() {
        let mut p = profile("editor", ProviderKind::Anthropic, "m");
        p.tools = Some(vec!["edit_file".to_string()]);
        let (tool, _, _) = echo_task(vec![p]);
        assert!(tool.side_effecting(&serde_json::json!({"profile": "editor", "prompt": "x"})));
    }

    #[test]
    fn side_effecting_false_for_read_only_unknown_and_missing_profiles() {
        // A read-only profile parallelizes; an unknown or absent profile is not
        // side-effecting (the call is rejected by validate/run regardless).
        let (tool, _, _) = echo_task(anthropic_profiles());
        assert!(!tool.side_effecting(&serde_json::json!({"profile": "reviewer"})));
        assert!(!tool.side_effecting(&serde_json::json!({"profile": "ghost"})));
        assert!(!tool.side_effecting(&serde_json::json!({})));
    }

    // ── helpers: names/validation ──

    #[test]
    fn is_valid_child_tool_covers_read_only_and_mutating_sets() {
        assert!(is_valid_child_tool("read_file"));
        assert!(is_valid_child_tool("web_search"));
        // Step 5: the mutating tools are now valid child-allowlist entries.
        assert!(is_valid_child_tool("write_file"));
        assert!(is_valid_child_tool("edit_file"));
        assert!(is_valid_child_tool("shell"));
        // `task` (no recursion) and typos stay rejected.
        assert!(!is_valid_child_tool("task"));
        assert!(!is_valid_child_tool("frobnicate"));
    }

    #[test]
    fn default_child_tool_names_are_the_read_only_set() {
        assert_eq!(
            default_child_tool_names(),
            [
                "read_file",
                "list_directory",
                "search_files",
                "web_fetch",
                "web_search"
            ]
        );
    }

    #[test]
    fn build_child_tool_skips_web_search_without_key_and_unknown_names() {
        let sandbox = Sandbox::unbounded();
        let cancel = Arc::new(AtomicBool::new(false));
        assert!(build_child_tool("web_search", &sandbox, None, &cancel).is_none());
        // A name outside the valid set is the belt-and-braces `None` arm.
        assert!(build_child_tool("frobnicate", &sandbox, None, &cancel).is_none());
    }

    // ── Parent-level fan-out: two task calls, ordered results ──

    /// A parent-agent stream calling `task` for `(id, profile, prompt)` pairs,
    /// stopping on `ToolUse` so the loop runs them.
    fn task_calls_stream(calls: &[(&str, &str, &str)]) -> Vec<StreamDelta> {
        let mut deltas = vec![StreamDelta::MessageStart {
            usage: Usage::default(),
        }];
        for (i, (id, prof, prompt)) in calls.iter().enumerate() {
            deltas.push(StreamDelta::ToolUseStart {
                index: i,
                id: id.to_string(),
                name: "task".to_string(),
                input: serde_json::json!({"profile": prof, "prompt": prompt}),
            });
        }
        deltas.push(StreamDelta::MessageDelta {
            stop_reason: Some(StopReason::ToolUse),
            usage: Usage::default(),
        });
        deltas
    }

    fn end_stream() -> Vec<StreamDelta> {
        vec![
            StreamDelta::MessageStart {
                usage: Usage::default(),
            },
            StreamDelta::TextStart {
                index: 0,
                text: "all done".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage::default(),
            },
        ]
    }

    /// A text-less child turn that calls `read_file` and stops on `ToolUse`,
    /// driving one more loop round. Text-less on purpose: a child built only
    /// from these has no assistant text for the salvage fallback, so an
    /// exhausted child surfaces the turn-limit error (naming its limit) rather
    /// than a salvaged partial.
    fn child_tool_use_stream(id: &str) -> Vec<StreamDelta> {
        vec![
            StreamDelta::MessageStart {
                usage: Usage::default(),
            },
            StreamDelta::ToolUseStart {
                index: 0,
                id: id.to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: Usage::default(),
            },
        ]
    }

    /// Drive a parent [`Agent`] whose only tool is the given `TaskTool` through
    /// one turn that issues `calls`, returning the tool-result blocks the fan-out
    /// produced (in the model's block order) and everything the turn wrote to
    /// the parent's display sink.
    fn run_parent_with_tasks(
        tool: TaskTool,
        budget: BudgetLedger,
        calls: &[(&str, &str, &str)],
    ) -> (Vec<Block>, String) {
        use crate::testing::MockProvider;
        let parent = MockProvider::new(vec![task_calls_stream(calls), end_stream()]);
        let config = AgentConfig {
            provider_kind: ProviderKind::Anthropic,
            model: "parent-model".to_string(),
            max_tokens: 64,
            system: None,
            context_token_limit: 100_000,
            effort: None,
            max_turns: crate::agent::MAX_TURNS,
        };
        let mut agent = Agent::new(Box::new(parent), config, vec![Box::new(tool)]);
        agent.set_budget_ledger(budget);
        let mut out = Vec::new();
        agent.run("go", &mut out).unwrap();
        // The tool-result message is the one the loop pushed after the tool turn.
        let results = agent
            .messages_for_test()
            .iter()
            .find(|m| {
                m.role == Role::User
                    && m.content
                        .iter()
                        .any(|b| matches!(b, Block::ToolResult { .. }))
            })
            .expect("a tool-result message")
            .content
            .clone();
        (results, String::from_utf8(out).unwrap())
    }

    #[test]
    fn profile_max_turns_overrides_the_top_level_for_the_child() {
        // The profile pins max_turns: 2; the tool's top-level default is 10. A
        // child that never stops calling tools hits the wall at *2* rounds —
        // the profile's limit, not the parent's — and, with no assistant text
        // to salvage, surfaces the turn-limit error naming 2. Were the
        // top-level 10 wrongly applied, the three-stream script would run dry
        // and echo an Ok result instead of erroring.
        let factory = ProviderFactory::from_fns(
            |_kind, _key| {
                Box::new(ThreadSafeProvider::scripted(vec![
                    child_tool_use_stream("c1"),
                    child_tool_use_stream("c2"),
                    child_tool_use_stream("c3"),
                ]))
            },
            |_env| Some("k".to_string()),
        );
        let mut p = profile("surveyor", ProviderKind::Anthropic, "m");
        p.max_turns = Some(2);
        let tool = TaskTool::new(
            factory,
            Sandbox::unbounded(),
            None,
            Arc::new(AtomicBool::new(false)),
            BudgetLedger::new(),
            Concurrency::new(),
            approve_all(),
            vec![p],
            256,
            100_000,
            10,
        );

        let err = tool
            .run(
                serde_json::json!({"profile": "surveyor", "prompt": "survey the repo"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("maximum of 2 turns"), "got: {err}");
    }

    #[test]
    fn child_inherits_the_top_level_max_turns_when_the_profile_omits_it() {
        // The complement: a profile with no max_turns inherits the tool's
        // top-level value. Pinned at 2 here so the same three-stream script
        // hits the wall at the second round and names 2 — proving the
        // `unwrap_or(self.max_turns)` inheritance, not a hardcoded default.
        let factory = ProviderFactory::from_fns(
            |_kind, _key| {
                Box::new(ThreadSafeProvider::scripted(vec![
                    child_tool_use_stream("c1"),
                    child_tool_use_stream("c2"),
                    child_tool_use_stream("c3"),
                ]))
            },
            |_env| Some("k".to_string()),
        );
        let p = profile("surveyor", ProviderKind::Anthropic, "m");
        let tool = TaskTool::new(
            factory,
            Sandbox::unbounded(),
            None,
            Arc::new(AtomicBool::new(false)),
            BudgetLedger::new(),
            Concurrency::new(),
            approve_all(),
            vec![p],
            256,
            100_000,
            2,
        );

        let err = tool
            .run(
                serde_json::json!({"profile": "surveyor", "prompt": "survey the repo"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("maximum of 2 turns"), "got: {err}");
    }

    #[test]
    fn two_task_calls_return_results_in_block_order() {
        // Read-only children ⇒ the two `task` calls fan out concurrently; the
        // loop reassembles the echoed replies in the model's block order,
        // regardless of which worker finishes first.
        let (tool, _, budget) = echo_task(anthropic_profiles());
        let (results, _) = run_parent_with_tasks(
            tool,
            budget,
            &[
                ("t0", "reviewer", "first task"),
                ("t1", "reviewer", "second task"),
            ],
        );
        let (id0, content0, err0) = expect_tool_result(&results[0]);
        let (id1, content1, err1) = expect_tool_result(&results[1]);
        assert_eq!((id0, content0, err0), ("t0", "first task", false));
        assert_eq!((id1, content1, err1), ("t1", "second task", false));
    }

    // ── Fan-out concurrency exemption ──

    #[test]
    fn task_calls_do_not_consume_fan_out_permits() {
        // `task` opts out of the concurrency gate, so a batch of read-only
        // delegations fans out without drawing a single permit. Pinned
        // deterministically against a *zero*-permit pool: a gated tool's
        // acquire would block forever on it, so the batch completing at all
        // proves the exemption — and the deadlock a gating `task` would cause
        // (its worker holding a permit while the child needs one) cannot form.
        let (tool, _, budget) = echo_task(anthropic_profiles());
        assert!(!tool.gates_concurrency(), "the task tool must not gate");

        use crate::testing::MockProvider;
        let calls = [
            ("t0", "reviewer", "first task"),
            ("t1", "reviewer", "second task"),
        ];
        let parent = MockProvider::new(vec![task_calls_stream(&calls), end_stream()]);
        let config = AgentConfig {
            provider_kind: ProviderKind::Anthropic,
            model: "parent-model".to_string(),
            max_tokens: 64,
            system: None,
            context_token_limit: 100_000,
            effort: None,
            max_turns: crate::agent::MAX_TURNS,
        };
        let mut agent = Agent::new(Box::new(parent), config, vec![Box::new(tool)]);
        agent.set_budget_ledger(budget);
        agent.set_concurrency(Concurrency::with_permits(0));
        let mut out = Vec::new();
        // Completes rather than deadlocks: neither the two `task` workers nor
        // their (leaf-free) children ever acquire a permit from the empty pool.
        agent.run("go", &mut out).unwrap();
    }

    // ── Shared budget ledger ──

    #[test]
    fn task_spawns_draw_tier_four_from_the_shared_ledger() {
        // Each `task` call spends tier 4 on the ledger the parent and tool
        // share, so the count reflects both spawns.
        let (tool, _, budget) = echo_task(anthropic_profiles());
        run_parent_with_tasks(
            tool,
            budget.clone(),
            &[("t0", "reviewer", "a"), ("t1", "reviewer", "b")],
        );
        assert_eq!(budget.count(4), 2);
    }

    #[test]
    fn shared_ledger_caps_the_number_of_spawns() {
        // With the tier-4 ceiling at one, the second `task` in a batch is
        // budget-rejected — the spawn cap enforced on the shared ledger.
        let (tool, _, budget) = echo_task(anthropic_profiles());
        budget.set_limit(4, 1);
        let (results, _) = run_parent_with_tasks(
            tool,
            budget.clone(),
            &[("t0", "reviewer", "a"), ("t1", "reviewer", "b")],
        );
        let (_, first, err_first) = expect_tool_result(&results[0]);
        let (_, second, err_second) = expect_tool_result(&results[1]);
        assert_eq!((first, err_first), ("a", false));
        assert!(err_second, "the second spawn is over budget");
        assert!(second.contains("budget exceeded"), "got: {second}");
        assert_eq!(budget.count(4), 1);
    }

    #[test]
    fn child_tool_calls_draw_from_the_shared_ledger() {
        // A child granted `web_fetch` and scripted to call it draws tier 3 on
        // the *same* ledger the parent spends tier 4 on — proving parent and
        // child share one process-wide budget.
        let mut p = profile("fetcher", ProviderKind::Anthropic, "m");
        p.tools = Some(vec!["web_fetch".to_string()]);
        // The child's first turn calls web_fetch on a blocked (hermetic) URL,
        // then its second turn ends. The fetch draws tier 3 before failing
        // validation, which is all this test needs.
        let child_turn1 = vec![
            StreamDelta::MessageStart {
                usage: Usage::default(),
            },
            StreamDelta::ToolUseStart {
                index: 0,
                id: "c0".to_string(),
                name: "web_fetch".to_string(),
                input: serde_json::json!({"url": "http://127.0.0.1/"}),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: Usage::default(),
            },
        ];
        let child_turn2 = vec![
            StreamDelta::MessageStart {
                usage: Usage::default(),
            },
            StreamDelta::TextStart {
                index: 0,
                text: "fetched".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage::default(),
            },
        ];
        let factory = ProviderFactory::from_fns(
            move |_kind, _key| {
                Box::new(ThreadSafeProvider::scripted(vec![
                    child_turn1.clone(),
                    child_turn2.clone(),
                ]))
            },
            |_env| Some("k".to_string()),
        );
        let budget = BudgetLedger::new();
        let tool = TaskTool::new(
            factory,
            Sandbox::unbounded(),
            None,
            Arc::new(AtomicBool::new(false)),
            budget.clone(),
            Concurrency::new(),
            approve_all(),
            vec![p],
            256,
            100_000,
            crate::agent::MAX_TURNS,
        );
        run_parent_with_tasks(tool, budget.clone(), &[("t0", "fetcher", "fetch it")]);
        assert_eq!(budget.count(4), 1, "the parent's task spawn");
        assert_eq!(
            budget.count(3),
            1,
            "the child's web_fetch, on the shared ledger"
        );
    }

    // ── Executor children: confirmations bubble to the parent TTY ──

    /// The two child turns of an executor delegation: a `write_file` call to
    /// `out.txt`, then a plain-text end. The write only lands if the child's
    /// confirm — the parent-TTY one the `task` tool installs — approves.
    fn write_file_child_streams() -> Vec<Vec<StreamDelta>> {
        vec![
            vec![
                StreamDelta::MessageStart {
                    usage: Usage::default(),
                },
                StreamDelta::ToolUseStart {
                    index: 0,
                    id: "w0".to_string(),
                    name: "write_file".to_string(),
                    input: serde_json::json!({"path": "out.txt", "content": "hi"}),
                },
                StreamDelta::MessageDelta {
                    stop_reason: Some(StopReason::ToolUse),
                    usage: Usage::default(),
                },
            ],
            vec![
                StreamDelta::MessageStart {
                    usage: Usage::default(),
                },
                StreamDelta::TextStart {
                    index: 0,
                    text: "done".to_string(),
                },
                StreamDelta::MessageDelta {
                    stop_reason: Some(StopReason::EndTurn),
                    usage: Usage::default(),
                },
            ],
        ]
    }

    /// A `task` tool with one executor profile (`editor`, granted
    /// `write_file`) whose child is scripted to call `write_file`, over
    /// `sandbox` and the given session policy. Drives the policy-inheritance
    /// path directly; `confirm_override` pins the profile's own mode, the
    /// precedence case.
    fn executor_task(sandbox: Sandbox, confirm: ConfirmPolicy) -> TaskTool {
        executor_task_pinned(sandbox, confirm, None)
    }

    fn executor_task_pinned(
        sandbox: Sandbox,
        confirm: ConfirmPolicy,
        confirm_override: Option<crate::config::ConfirmMode>,
    ) -> TaskTool {
        let factory = ProviderFactory::from_fns(
            |_kind, _key| Box::new(ThreadSafeProvider::scripted(write_file_child_streams())),
            |_env| Some("k".to_string()),
        );
        let mut p = profile("editor", ProviderKind::Anthropic, "m");
        p.tools = Some(vec!["write_file".to_string()]);
        p.confirm = confirm_override;
        TaskTool::new(
            factory,
            sandbox,
            None,
            Arc::new(AtomicBool::new(false)),
            BudgetLedger::new(),
            Concurrency::new(),
            confirm,
            vec![p],
            256,
            100_000,
            crate::agent::MAX_TURNS,
        )
    }

    #[test]
    fn executor_child_confirm_approves_and_names_the_profile() {
        // The dangerous child call prompts through the base confirm, which sees
        // a summary naming the delegating profile; approval lets the write land.
        let dir = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let seen: Arc<std::sync::Mutex<Vec<String>>> = Arc::default();
        let confirm = {
            let seen = Arc::clone(&seen);
            ask_session(move |summary| {
                seen.lock().unwrap().push(summary.to_string());
                true
            })
        };
        let tool = executor_task(sandbox, confirm);
        let out = tool
            .run(
                serde_json::json!({"profile": "editor", "prompt": "write it"}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert_eq!(out, "done");

        let prompts = seen.lock().unwrap();
        assert_eq!(prompts.len(), 1, "one confirm for the one write");
        let prompt = &prompts[0];
        assert!(prompt.contains("editor"), "must name the profile: {prompt}");
        assert!(
            prompt.contains("write_file"),
            "must name the tool: {prompt}"
        );
        // Approved ⇒ the child's write reached disk.
        assert_eq!(
            std::fs::read_to_string(dir.path().join("out.txt")).unwrap(),
            "hi"
        );
    }

    #[test]
    fn executor_child_confirm_denial_blocks_the_write() {
        // A denial at the parent TTY rejects the child's write before it runs.
        let dir = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let tool = executor_task(sandbox, ask_session(|_| false));
        let out = tool
            .run(
                serde_json::json!({"profile": "editor", "prompt": "write it"}),
                &mut std::io::sink(),
            )
            .unwrap();
        // The child recovers from the denied tool and still finishes its turn.
        assert_eq!(out, "done");
        assert!(
            !dir.path().join("out.txt").exists(),
            "denied write must not land"
        );
    }

    #[test]
    fn executor_child_inherits_the_session_allow_mode() {
        // The session runs `allow`: the child's dangerous call is waved with
        // no prompt anywhere (the session stub panics if consulted) — an AFK
        // run delegates end-to-end.
        let dir = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let session = ask_session(crate::testing::no_prompt);
        session.set_mode(crate::config::ConfirmMode::Allow);
        let tool = executor_task(sandbox, session);
        let out = tool
            .run(
                serde_json::json!({"profile": "editor", "prompt": "write it"}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert_eq!(out, "done");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("out.txt")).unwrap(),
            "hi"
        );
    }

    #[test]
    fn executor_profile_confirm_override_beats_the_session() {
        // The session would deny at the prompt, but the profile pins `allow`
        // — the override wins for this child, and the pin means a later
        // session switch could not move it either.
        let dir = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let tool = executor_task_pinned(
            sandbox,
            ask_session(|_| false),
            Some(crate::config::ConfirmMode::Allow),
        );
        let out = tool
            .run(
                serde_json::json!({"profile": "editor", "prompt": "write it"}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert_eq!(out, "done");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("out.txt")).unwrap(),
            "hi"
        );
    }

    #[test]
    fn session_confirm_switch_reaches_children_spawned_later() {
        // The task tool holds a clone of the session policy; the shared mode
        // cell means a /confirm switch after construction still governs the
        // next child.
        let dir = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let session = ask_session(|_| false);
        let tool = executor_task(sandbox, session.clone());
        session.set_mode(crate::config::ConfirmMode::Allow);
        let out = tool
            .run(
                serde_json::json!({"profile": "editor", "prompt": "write it"}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert_eq!(out, "done");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("out.txt")).unwrap(),
            "hi"
        );
    }

    // ── Live child output: the PrefixWriter ──

    #[test]
    fn prefix_writer_prefixes_and_scrubs_every_line() {
        let mut sink: Vec<u8> = Vec::new();
        let mut w = PrefixWriter::new(&mut sink, "rev");
        // A line split across writes reassembles; a control byte is scrubbed;
        // every complete line gets the prefix; finish drains the tail.
        w.write_all(b"hel").unwrap();
        w.write_all(b"lo\nwo\x1brld\ntail").unwrap();
        w.finish().unwrap();
        assert_eq!(
            String::from_utf8(sink).unwrap(),
            "[rev] hello\n[rev] wo?rld\n[rev] tail\n"
        );
    }

    #[test]
    fn prefix_writer_finish_is_quiet_after_a_terminated_line() {
        let mut sink: Vec<u8> = Vec::new();
        let mut w = PrefixWriter::new(&mut sink, "rev");
        w.write_all(b"done\n").unwrap();
        w.finish().unwrap();
        assert_eq!(String::from_utf8(sink).unwrap(), "[rev] done\n");
    }

    #[test]
    fn prefix_writer_reassembles_a_char_split_across_writes() {
        // '€' is three bytes; buffering on the `\n` byte (never part of a
        // multi-byte sequence) must keep the split character intact.
        let mut sink: Vec<u8> = Vec::new();
        let mut w = PrefixWriter::new(&mut sink, "rev");
        let euro = "€".as_bytes();
        w.write_all(&euro[..1]).unwrap();
        w.write_all(&euro[1..]).unwrap();
        w.write_all(b"\n").unwrap();
        assert_eq!(String::from_utf8(sink).unwrap(), "[rev] €\n");
    }

    #[test]
    fn prefix_writer_flush_does_not_split_the_pending_line() {
        // The child agent flushes after every streamed delta; a mid-line flush
        // must reach the inner sink without emitting the partial line.
        let mut sink: Vec<u8> = Vec::new();
        let mut w = PrefixWriter::new(&mut sink, "rev");
        w.write_all(b"par").unwrap();
        w.flush().unwrap();
        w.write_all(b"tial\n").unwrap();
        assert_eq!(String::from_utf8(sink).unwrap(), "[rev] partial\n");
    }

    #[test]
    fn prefix_writer_sanitizes_the_profile_name_itself() {
        // The name lands on the terminal once per line, so it goes through the
        // same policy as the content.
        let mut sink: Vec<u8> = Vec::new();
        let mut w = PrefixWriter::new(&mut sink, "re\x07v");
        w.write_all(b"x\n").unwrap();
        assert_eq!(String::from_utf8(sink).unwrap(), "[re?v] x\n");
    }

    /// A sink that fails every write and flush, driving the writer's error
    /// propagation paths.
    struct FailWriter;
    impl Write for FailWriter {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("sink broke"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("sink broke"))
        }
    }

    #[test]
    fn prefix_writer_surfaces_inner_errors() {
        let mut broken = FailWriter;
        let mut w = PrefixWriter::new(&mut broken, "rev");
        // A completed line emits — and the inner failure propagates.
        assert!(w.write(b"x\n").is_err());
        // A partial line defers the failure to finish, and flush delegates.
        assert!(w.write(b"tail").is_ok());
        assert!(w.flush().is_err());
        assert!(w.finish().is_err());
    }

    // ── Live child output: through the task tool ──

    #[test]
    fn run_streams_prefixed_child_output_to_the_sink() {
        // The echo child streams its prompt; the tool relays it live to the
        // provided sink, prefixed and line-terminated — while the tool result
        // stays the child's final text, unchanged.
        let (tool, _, _) = echo_task(anthropic_profiles());
        let mut sink: Vec<u8> = Vec::new();
        let result = tool
            .run(
                serde_json::json!({"profile": "reviewer", "prompt": "review this"}),
                &mut sink,
            )
            .unwrap();
        assert_eq!(result, "review this");
        assert_eq!(String::from_utf8(sink).unwrap(), "[reviewer] review this\n");
    }

    #[test]
    fn parallel_children_flush_prefixed_output_in_block_order() {
        // Two read-only delegations fan out concurrently; each worker's stream
        // lands in a private buffer, flushed in block order — child t0's lines,
        // then its result line, then child t1's — never interleaved.
        let (tool, _, budget) = echo_task(anthropic_profiles());
        let (_, shown) = run_parent_with_tasks(
            tool,
            budget,
            &[
                ("t0", "reviewer", "first task"),
                ("t1", "reviewer", "second task"),
            ],
        );
        let stream0 = shown.find("[reviewer] first task").expect("child 0 stream");
        let result0 = shown.find("[result: first task]").expect("result 0");
        let stream1 = shown
            .find("[reviewer] second task")
            .expect("child 1 stream");
        let result1 = shown.find("[result: second task]").expect("result 1");
        assert!(stream0 < result0, "child 0 streams before its result line");
        assert!(result0 < stream1, "block order: call 0 fully before call 1");
        assert!(stream1 < result1, "child 1 streams before its result line");
    }

    // ── Rollback exemption for an executor `task` ──

    #[test]
    fn executor_task_failure_keeps_parent_history() {
        // A parent turn that ran an executor `task` (side-effecting) and then
        // failed keeps its transcript verbatim — the same rollback carve-out a
        // direct write_file gets, reached through per-input `side_effecting`. To
        // fail *after* the tool ran, the parent keeps re-issuing the executor
        // `task` until the loop hits its turn limit.
        use crate::testing::MockProvider;
        let mut p = profile("editor", ProviderKind::Anthropic, "m");
        p.tools = Some(vec!["write_file".to_string()]);
        let (tool, _, budget) = echo_task(vec![p]);

        let parent = MockProvider::repeating(task_calls_stream(&[("t0", "editor", "do it")]));
        let config = AgentConfig {
            provider_kind: ProviderKind::Anthropic,
            model: "parent-model".to_string(),
            max_tokens: 64,
            system: None,
            context_token_limit: 100_000,
            effort: None,
            max_turns: crate::agent::MAX_TURNS,
        };
        let mut agent = Agent::new(Box::new(parent), config, vec![Box::new(tool)]);
        agent.set_budget_ledger(budget);
        let mut out = Vec::new();
        let err = agent.run("go", &mut out).unwrap_err();
        assert!(matches!(
            err,
            crate::agent::AgentError::TurnLimitExceeded(_)
        ));
        // The executor task's tool_use survived the failed turn's rollback.
        let kept_task = agent.messages_for_test().iter().any(|m| {
            m.content
                .iter()
                .any(|b| matches!(b, Block::ToolUse { name, .. } if name == "task"))
        });
        assert!(kept_task, "a side-effecting task turn must not roll back");
    }
}
