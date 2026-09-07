use crate::display::{sanitize_for_display, sanitize_multiline, scrub_controls};
use crate::markdown::MarkdownRenderer;
use crate::provider::{ApiError, Provider, ProviderKind};
use crate::session::{SESSION_VERSION, Session};
use crate::tools::ToolDef;
use crate::turn::{
    Block, Role, StopReason, StreamDelta, ToolSpec, TurnMessage, TurnRequest, Usage,
};
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

mod budget;
mod concurrency;
mod confirm;
mod confirm_live;
mod context;
#[cfg(test)]
mod tests_live;

pub use budget::BudgetLedger;
pub use concurrency::Concurrency;
pub use confirm::ConfirmPolicy;
pub use confirm_live::interactive_confirm;

use confirm::{ConfirmCall, ConfirmOutcome};

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

const TOOL_POLICY: &str = "\
\n\n## Tool selection policy\n\
Prefer the cheapest tool that can answer the question:\n\
1. Your own knowledge (free — no tool call needed)\n\
2. read_file / list_directory / search_files (local, fast, read-only)\n\
3. edit_file / write_file (local, mutates files)\n\
4. web_fetch (network, targeted — use when you have a specific URL)\n\
5. web_search (external API, slow, noisy — last resort only)\n\
6. shell (arbitrary local execution — only when no dedicated tool fits)\n\
\n\
Never use an expensive tool when a cheaper one suffices.";

// ── Context compaction ──

/// Compaction fires when the last measured prompt size reaches this percentage
/// of the configured `context_token_limit`. A hardcoded constant by design:
/// the limit is the configurable knob; the ratio is not a second one.
const COMPACT_THRESHOLD_PERCENT: u64 = 75;

/// How many of the most recent turn-groups survive a compaction verbatim.
/// Everything older is folded into the rolling summary.
const KEEP_RECENT_TURN_GROUPS: usize = 2;

/// Output budget for the summary sub-call, deliberately decoupled from the
/// operator's `max_tokens`: that knob caps normal replies, and a small value
/// there would truncate the summary — which the fail-fast check rejects,
/// wedging every later turn into the same failure until `/clear`. The
/// summary is prompted to be concise, so this is a generous fixed budget.
const SUMMARIZE_MAX_TOKENS: u32 = 1024;

/// System prompt for the compaction summary call.
const SUMMARIZE_SYSTEM: &str = "\
You are compacting the oldest part of a longer conversation so it can be \
replaced by a short record. Write a concise summary that preserves the user's \
goals and constraints, decisions made, key facts and file paths, and any \
unresolved questions. Reply with the summary text only.";

/// Where to cut the history so the dropped prefix `messages[..cut]` consists
/// of whole turn-groups and the most recent [`KEEP_RECENT_TURN_GROUPS`] groups
/// survive verbatim. `None` when there are not enough groups to drop anything.
///
/// A turn-group starts at a `Role::User` message carrying no `ToolResult`
/// block — a fresh user turn. A user message *with* tool results belongs to
/// the tool loop of the group in progress (including the coalesced case,
/// where the next turn's text rides in a trailing tool-result message), so
/// cutting there would separate a `tool_result` from its `tool_use`, which
/// the API rejects. Cutting only at group starts makes orphans impossible.
fn compaction_cut(messages: &[TurnMessage], keep: usize) -> Option<usize> {
    let starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            m.role == Role::User
                && !m
                    .content
                    .iter()
                    .any(|b| matches!(b, Block::ToolResult { .. }))
        })
        .map(|(i, _)| i)
        .collect();
    (starts.len() > keep).then(|| starts[starts.len() - keep])
}

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

// ── Stream processing ──

/// Hard cap on a single block's accumulated bytes — streamed text or
/// tool-args JSON. After headers arrive the transport deliberately leaves the
/// response body un-timed (a long turn must stream to completion), so without
/// a cap a broken or hostile endpoint that keeps sending deltas for one block
/// grows the process without bound. A full turn at the configured
/// max_tokens=8192 tops out around a few tens of KB; 8 MiB — matching the SSE
/// layer's per-event cap — is orders of magnitude above any legitimate block.
const MAX_BLOCK_BYTES: usize = 8 * 1024 * 1024;

/// The error an oversized block surfaces as, named after the cap so the
/// operator sees which bound tripped.
fn block_overflow() -> AgentError {
    AgentError::Api(ApiError::Stream(format!(
        "content block exceeded MAX_BLOCK_BYTES ({MAX_BLOCK_BYTES} bytes)"
    )))
}

/// Close the styled renderer's half-printed line before an early return.
/// Text streams ahead of its newline, so an error or cancellation can land
/// mid-line — without this, whatever prints next lands on the partial line,
/// inside its style.
fn abort_line(renderer: &mut Option<MarkdownRenderer>, out: &mut dyn Write) {
    if let Some(r) = renderer.as_mut() {
        r.interrupt(out);
    }
}

/// Intermediate state for accumulating a content block from streaming events.
enum BlockBuilder {
    Text(String),
    Thinking {
        text: String,
        signature: String,
    },
    ToolUse {
        id: String,
        name: String,
        json: String,
    },
}

/// Fold one usage sighting into the accumulator. The wire counts are
/// cumulative, so max-of-sightings converges on the true totals; for the
/// optional cache counters a sighted `Some` also beats `None`, so a trailing
/// event that omits them cannot erase an earlier report.
fn merge_usage(acc: &mut Usage, seen: &Usage) {
    fn max_cache(acc: Option<u32>, seen: Option<u32>) -> Option<u32> {
        match (acc, seen) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => b.or(a),
        }
    }
    acc.input_tokens = acc.input_tokens.max(seen.input_tokens);
    acc.output_tokens = acc.output_tokens.max(seen.output_tokens);
    acc.cache_creation_input_tokens = max_cache(
        acc.cache_creation_input_tokens,
        seen.cache_creation_input_tokens,
    );
    acc.cache_read_input_tokens =
        max_cache(acc.cache_read_input_tokens, seen.cache_read_input_tokens);
}

/// Result of processing a streamed response.
struct StreamResult {
    blocks: Vec<Block>,
    stop_reason: StopReason,
    /// Measured token accounting for this response, accumulated across the
    /// stream's usage-bearing events.
    usage: Usage,
}

// ── Tool execution ──

/// A tool call that cleared every pre-flight check in `execute_tools` and is
/// ready to run. Borrows the block's id/input and the tool itself; `slot`
/// indexes the turn's results vector, so reassembly restores the model's
/// block order no matter which worker finishes first.
struct ApprovedCall<'a> {
    slot: usize,
    id: &'a str,
    tool: &'a dyn ToolDef,
    input: &'a serde_json::Value,
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

/// Cap on the verbatim result line: a result that fits — one line within
/// the cap — prints as-is; anything larger collapses to a size-annotated
/// first-line preview. Only the operator's copy is summarized; the model
/// always receives the full output in its tool_result block.
const RESULT_PREVIEW_MAX_CHARS: usize = 100;

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

/// Human-readable byte count for the result summary line.
fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Collapse a tool result for display. Short output — one line within
/// [`RESULT_PREVIEW_MAX_CHARS`], ignoring a trailing newline — passes
/// through sanitized but verbatim; anything larger becomes
/// `N lines, SIZE — first-line preview…`. Sanitization is not optional
/// cosmetics: results carry tool-fetched content (web pages, file reads),
/// and raw control characters here could repaint the terminal or forge
/// model output.
fn summarize_output(output: &str) -> String {
    let body = output.trim_end_matches('\n');
    let mut lines = body.lines();
    let first = lines.next().unwrap_or("");
    let multiline = lines.next().is_some();
    if !multiline && first.chars().count() <= RESULT_PREVIEW_MAX_CHARS {
        return sanitize_for_display(body);
    }
    let count = body.lines().count();
    let noun = if count == 1 { "line" } else { "lines" };
    let preview: String = first.chars().take(RESULT_PREVIEW_MAX_CHARS).collect();
    format!(
        "{count} {noun}, {} — {}…",
        format_size(output.len()),
        sanitize_for_display(&preview)
    )
}

/// Cap on the confirmation summary, so an adversarial input cannot scroll
/// the prompt (and the dangerous part of a command) off-screen. The inverse
/// trade-off is accepted: anything past the cap is not shown, so the gate
/// is sound only for commands whose dangerous part is visible within it.
const CONFIRM_SUMMARY_MAX_CHARS: usize = 200;

/// One-line summary for the confirmation prompt: the tool name plus the
/// tool's own `format_status` line. Deriving the detail from the tool —
/// rather than guessing input fields here — guarantees the prompt shows the
/// same field the tool will act on, never a decoy field.
fn confirm_summary(name: &str, status: Option<&str>) -> String {
    let summary = match status {
        Some(status) => format!("{name}: {}", sanitize_for_display(status)),
        None => name.to_string(),
    };
    if summary.chars().count() > CONFIRM_SUMMARY_MAX_CHARS {
        let capped: String = summary.chars().take(CONFIRM_SUMMARY_MAX_CHARS).collect();
        format!("{capped}…")
    } else {
        summary
    }
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

    /// Convert registered ToolDef impls into the normalized tool spec.
    fn tool_definitions(&self) -> Vec<ToolSpec> {
        self.tools
            .iter()
            .map(|t| ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }

    /// Names of the registered tools, in registration order. Backs the REPL's
    /// startup banner.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Consume a stream of events, printing text to `out` as it arrives and
    /// accumulating all content blocks (text and tool use). Returns the
    /// completed blocks, the final stop reason, and the measured token usage.
    ///
    /// Takes the stream as a trait object rather than `impl Iterator`: the
    /// error-path closures inside would otherwise monomorphize per caller,
    /// and the copy in the instantiation no test drives directly reads as
    /// uncovered under the scoped-100 gate.
    fn process_stream(
        &self,
        stream: &mut dyn Iterator<Item = Result<StreamDelta, ApiError>>,
        out: &mut dyn Write,
    ) -> Result<StreamResult, AgentError> {
        let mut builders: Vec<BlockBuilder> = Vec::new();
        let mut stop_reason = None;
        // On a TTY the model's Markdown habits render as ANSI styling; piped
        // output stays byte-for-byte verbatim. Fresh per stream, so fence
        // state can't leak across turns.
        let mut renderer = self.styled.then(MarkdownRenderer::default);
        // Usage arrives split across events and providers: Anthropic reports
        // input up front (`MessageStart`) and cumulative totals in
        // `MessageDelta`; OpenAI reports everything only in the trailing
        // `MessageDelta`. Merging the max of every sighting yields the true
        // per-turn totals regardless of which event carried them.
        let mut usage = Usage::default();

        for delta in stream {
            // The between-events seam: returning here drops the stream —
            // and with it the underlying response body, aborting the
            // transfer. Nothing has been pushed to history yet (the caller
            // pushes only completed results), so the turn stays clean.
            if self.cancelled() {
                abort_line(&mut renderer, out);
                return Err(AgentError::Cancelled);
            }
            let delta = match delta {
                Ok(delta) => delta,
                Err(e) => {
                    abort_line(&mut renderer, out);
                    return Err(AgentError::Api(e));
                }
            };
            match delta {
                StreamDelta::MessageStart { usage: u } => merge_usage(&mut usage, &u),
                StreamDelta::TextStart { text, .. } => {
                    if !text.is_empty() {
                        match renderer.as_mut() {
                            Some(r) => r.push(&text, out),
                            None => {
                                // No renderer (piped/non-TTY): scrub through
                                // the same policy the renderer applies, so
                                // both text sinks share one guarantee.
                                let _ = out.write_all(scrub_controls(&text).as_bytes());
                            }
                        }
                        let _ = out.flush();
                    }
                    builders.push(BlockBuilder::Text(text));
                }
                StreamDelta::ThinkingStart {
                    text, signature, ..
                } => {
                    // Display policy: thinking is not the answer. A thinking
                    // model's reasoning can dwarf its reply, so the full text
                    // is never streamed to the terminal — one quiet meta line
                    // marks the block (dimmed on a TTY, like every other meta
                    // line; plain and grep-able when piped). The text itself
                    // is still accumulated: it exists for continuity, resent
                    // with its signature on tool-use turns.
                    if let Some(r) = renderer.as_mut() {
                        r.finish(out);
                    }
                    self.meta_line(out, format_args!("thinking"));
                    builders.push(BlockBuilder::Thinking { text, signature });
                }
                StreamDelta::ToolUseStart {
                    id, name, input, ..
                } => {
                    // Flush any buffered partial text line so the meta line
                    // lands after it, matching the unstyled path's layout.
                    if let Some(r) = renderer.as_mut() {
                        r.finish(out);
                    }
                    self.meta_line(out, format_args!("tool: {name}"));
                    // Seed with the start event's input. The API sends {} here
                    // for streaming, but if it ever sends a non-empty object we
                    // preserve it.
                    let seed = if input.is_object() && !input.as_object().unwrap().is_empty() {
                        input.to_string()
                    } else {
                        String::new()
                    };
                    builders.push(BlockBuilder::ToolUse {
                        id,
                        name,
                        json: seed,
                    });
                }
                StreamDelta::TextDelta { index, text } => {
                    if index < builders.len()
                        && let BlockBuilder::Text(ref mut buf) = builders[index]
                    {
                        buf.push_str(&text);
                        if buf.len() > MAX_BLOCK_BYTES {
                            abort_line(&mut renderer, out);
                            return Err(block_overflow());
                        }
                        match renderer.as_mut() {
                            Some(r) => r.push(&text, out),
                            None => {
                                // No renderer (piped/non-TTY): scrub through
                                // the same policy the renderer applies, so
                                // both text sinks share one guarantee.
                                let _ = out.write_all(scrub_controls(&text).as_bytes());
                            }
                        }
                        let _ = out.flush();
                    }
                }
                StreamDelta::ThinkingDelta { index, text } => {
                    if index < builders.len()
                        && let BlockBuilder::Thinking {
                            text: ref mut buf, ..
                        } = builders[index]
                    {
                        buf.push_str(&text);
                        // Reasoning rides the same cap as text and tool args:
                        // it is never displayed, so nothing else bounds it.
                        if buf.len() > MAX_BLOCK_BYTES {
                            abort_line(&mut renderer, out);
                            return Err(block_overflow());
                        }
                    }
                }
                StreamDelta::SignatureDelta {
                    index,
                    signature: sig,
                } => {
                    if index < builders.len()
                        && let BlockBuilder::Thinking {
                            ref mut signature, ..
                        } = builders[index]
                    {
                        signature.push_str(&sig);
                        if signature.len() > MAX_BLOCK_BYTES {
                            abort_line(&mut renderer, out);
                            return Err(block_overflow());
                        }
                    }
                }
                StreamDelta::ToolArgsDelta { index, json: part } => {
                    if index < builders.len()
                        && let BlockBuilder::ToolUse { ref mut json, .. } = builders[index]
                    {
                        json.push_str(&part);
                        if json.len() > MAX_BLOCK_BYTES {
                            abort_line(&mut renderer, out);
                            return Err(block_overflow());
                        }
                    }
                }
                StreamDelta::MessageDelta {
                    stop_reason: sr,
                    usage: u,
                } => {
                    stop_reason = sr;
                    merge_usage(&mut usage, &u);
                }
            }
        }

        let Some(stop_reason) = stop_reason else {
            // A provider adapter must normalize the protocol's terminal state
            // explicitly. Treat a synthetically exhausted or future malformed
            // adapter stream as incomplete rather than inventing EndTurn and
            // committing partial output.
            abort_line(&mut renderer, out);
            return Err(AgentError::Api(ApiError::Stream(
                "stream ended without a normalized stop reason".to_string(),
            )));
        };

        if let Some(r) = renderer.as_mut() {
            r.finish(out);
        }
        let _ = writeln!(out);

        // Usage is accumulated here but reported once per user turn, at the
        // end of `run_loop` — a per-round-trip line would repeat the same
        // numbers on the common single-round-trip turn and drown a tool loop
        // in bookkeeping.

        // Convert builders into finalized blocks.
        let mut blocks = Vec::with_capacity(builders.len());
        for b in builders {
            let block = match b {
                BlockBuilder::Text(text) => Block::Text(text),
                BlockBuilder::Thinking { text, signature } => Block::Thinking { text, signature },
                BlockBuilder::ToolUse { id, name, json } => {
                    // Tool inputs are JSON objects: an empty stream body means no
                    // arguments (`{}`). A non-empty body that fails to parse is
                    // malformed model output — log it and leave the input Null so
                    // execute_tools rejects the call with is_error (the model then
                    // retries), rather than silently dispatching a default input.
                    let input = if json.is_empty() {
                        serde_json::json!({})
                    } else {
                        serde_json::from_str(&json).unwrap_or_else(|e| {
                            self.meta_line(
                                out,
                                format_args!("malformed tool input for {name}: {e}"),
                            );
                            serde_json::Value::Null
                        })
                    };
                    Block::ToolUse { id, name, input }
                }
            };
            blocks.push(block);
        }

        Ok(StreamResult {
            blocks,
            stop_reason,
            usage,
        })
    }

    /// Build the normalized request for the next turn. Provider-neutral: the
    /// system prompt is plain text and messages are carried verbatim. Wire
    /// concerns — prompt-cache breakpoints, the string-or-blocks content shape —
    /// are the adapter's job, not the loop's.
    fn build_request(&self) -> TurnRequest {
        self.build_request_with_summary(self.compacted_summary.as_deref())
    }

    fn build_request_with_summary(&self, summary: Option<&str>) -> TurnRequest {
        let has_tools = !self.tools.is_empty();

        // Append the tool-selection policy to the system prompt when tools exist.
        let system = match (&self.config.system, has_tools) {
            (Some(text), true) => Some(format!("{text}{TOOL_POLICY}")),
            (None, true) => Some(TOOL_POLICY.trim_start().to_string()),
            (Some(text), false) => Some(text.clone()),
            (None, false) => None,
        };

        // The rolling compaction summary rides in the system prompt, after the
        // configured text (see the `compacted_summary` field for why it is not
        // a message).
        let system = match (system, summary) {
            (Some(text), Some(summary)) => Some(format!(
                "{text}\n\n## Earlier conversation (summarized)\n{summary}"
            )),
            (None, Some(summary)) => {
                Some(format!("## Earlier conversation (summarized)\n{summary}"))
            }
            (system, None) => system,
        };

        TurnRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            system,
            messages: self.messages.clone(),
            tools: self.tool_definitions(),
            effort: self.config.effort.clone(),
        }
    }

    /// Execute each tool call in the content blocks and return ToolResult
    /// blocks in the model's block order. Three phases: a serial pre-flight
    /// that runs every check and prompt in order on the main thread, a fan-out
    /// that runs approved calls concurrently on scoped threads when all of
    /// them are read-only (a batch containing a side-effecting call runs
    /// inline in block order, as does a batch of one — the common path spawns
    /// nothing), and a serial reassembly keyed by block index. Budget
    /// exceeded, validation failure, denial, and run errors all return
    /// `is_error: true`. Status and result lines are written to `out`, as is
    /// each tool's own live output — streamed directly by inline calls,
    /// buffered per worker and flushed in block order by fanned-out ones.
    /// The text that initiated the current turn: the newest user message
    /// carrying a `Text` block. Later user messages in a tool loop hold only
    /// tool results, so this walks past them to the typed prompt (or, in a
    /// child, the parent's task description). Quoted to the judge as the
    /// intent the proposed call should serve — trusted *relative to tool
    /// output*: it came from the operator or the delegating parent, never
    /// from fetched content.
    fn initiating_request(&self) -> Option<String> {
        self.messages.iter().rev().find_map(|m| {
            if m.role != Role::User {
                return None;
            }
            m.content.iter().find_map(|b| match b {
                Block::Text(text) => Some(text.clone()),
                _ => None,
            })
        })
    }

    fn execute_tools(&mut self, blocks: &[Block], out: &mut dyn Write) -> Vec<Block> {
        // A rejection resolves its slot immediately, before the fan-out.
        fn rejected(id: &str, content: String) -> Option<Block> {
            Some(Block::ToolResult {
                tool_use_id: id.to_string(),
                content,
                is_error: true,
            })
        }

        // Phase 1 — serial pre-flight. Everything that touches agent state or
        // the operator's terminal stays on the main thread, in block order:
        // budget accounting, the side-effect flag, and above all the
        // confirmation decisions, which share one stdin (`ask`) or one judge
        // budget and must resolve one at a time. The policy being `Send +
        // Sync` means the compiler no longer holds this line — the
        // serial placement here does. Each tool_use block either resolves to
        // a rejection here or is approved for the fan-out.
        let mut slots: Vec<Option<Block>> = Vec::new();
        let mut approved: Vec<ApprovedCall> = Vec::new();

        // The turn text the judge weighs intent against, captured before the
        // borrow of `self` inside the loop. One lookup per batch, and only
        // when something in it is actually confirmable.
        let initiating: Option<String> = blocks
            .iter()
            .any(|b| {
                matches!(b, Block::ToolUse { name, .. }
                if self.tools.iter().any(|t| t.name() == name && t.requires_confirmation()))
            })
            .then(|| self.initiating_request())
            .flatten();

        for block in blocks {
            let Block::ToolUse { id, name, input } = block else {
                continue;
            };

            // A pending cancellation resolves every remaining call without
            // prompting or running it. The slot still gets an is_error
            // tool_result rather than the turn erroring out mid-batch: a
            // kept history (the side-effect carve-out) must never leave a
            // tool_use dangling without its result, so the loop surfaces
            // the quiet outcome at its next pre-request check instead.
            if self.cancelled() {
                self.meta_line(out, format_args!("cancelled: {name}"));
                slots.push(rejected(id, format!("{name} cancelled by user")));
                continue;
            }

            // Tool inputs must be JSON objects (API contract). A non-object
            // means the streamed input was malformed or absent; reject with
            // is_error so the model retries instead of running on bad input.
            if !input.is_object() {
                self.meta_line(out, format_args!("invalid input: {name}"));
                slots.push(rejected(
                    id,
                    format!("invalid input for {name}: expected a JSON object"),
                ));
                continue;
            }

            // Unknown tool name — a hallucinated tool. Self-correct via
            // is_error (the model can pick a real tool) instead of aborting
            // the whole turn, matching every other failure path here.
            let Some(tool) = self.tools.iter().find(|t| t.name() == name) else {
                self.meta_line(out, format_args!("unknown tool: {name}"));
                slots.push(rejected(id, format!("unknown tool: {name}")));
                continue;
            };

            let cost = tool.cost();

            let status = tool.format_status(input);
            if let Some(status) = &status {
                self.meta_line(out, format_args!("{}", sanitize_for_display(status)));
            }

            // Budget check-and-reserve — one atomic ledger operation, so a
            // sibling call in this turn (or another agent sharing the
            // ledger) can never race between a check and an increment done
            // as two steps. Reserved here, before validation and
            // confirmation, so a call already over budget is rejected before
            // either runs — matching the pre-shared-ledger ordering.
            if let Err((count, limit)) = self.budget.draw(cost) {
                let reason =
                    format!("budget exceeded: {count} calls at cost tier {cost} (limit {limit})");
                self.meta_line(out, format_args!("budget error: {reason}"));
                slots.push(rejected(id, reason));
                continue;
            }

            // Pre-execution validation — reject before confirmation so
            // obviously-malformed calls never reach the user prompt. The
            // budget only bounds calls that actually run, so a rejection
            // here releases the reservation just made above.
            if let Err(reason) = tool.validate(input) {
                self.budget.release(cost);
                self.meta_line(out, format_args!("validation error: {reason}"));
                slots.push(rejected(id, reason));
                continue;
            }

            // Confirmation gate — the policy decides (ask prompts, allow
            // notices, judge adjudicates), and a denial releases the
            // reservation for the same reason as the validation gate above.
            // The summary is derived from the tool's own format_status, so
            // the decider approves the field the tool acts on. A tripped
            // circuit breaker short-circuits the rest of the batch without
            // consulting the policy again: the turn is already over, and
            // each further consult would be another paid judge call.
            if tool.requires_confirmation() {
                if self.auto_denials >= CONFIRM_BREAKER_LIMIT {
                    self.budget.release(cost);
                    slots.push(rejected(
                        id,
                        format!("{name} rejected: confirmation circuit breaker tripped"),
                    ));
                    continue;
                }
                let mut summary = confirm_summary(name, status.as_deref());
                if self.confirm.mode() == crate::config::ConfirmMode::Ask {
                    match tool.confirmation_preview(input) {
                        Ok(Some(preview)) => {
                            // Put review text in the prompt itself: a child's
                            // ordinary output may still be buffered when it asks.
                            summary = format!("{summary}\n{}", sanitize_multiline(&preview));
                        }
                        Ok(None) => {}
                        Err(reason) => {
                            self.budget.release(cost);
                            self.meta_line(
                                out,
                                format_args!("preview error: {}", sanitize_multiline(&reason)),
                            );
                            slots.push(rejected(id, reason));
                            continue;
                        }
                    }
                }
                let call = ConfirmCall {
                    tool: name,
                    input,
                    summary: &summary,
                    request: initiating.as_deref(),
                };
                match self.confirm.decide(&call) {
                    ConfirmOutcome::Approved { notice } => {
                        self.auto_denials = 0;
                        if let Some(notice) = notice {
                            self.meta_line(out, format_args!("{notice}"));
                        }
                    }
                    ConfirmOutcome::Denied { detail, automated } => {
                        self.budget.release(cost);
                        if automated {
                            self.auto_denials += 1;
                            // The automated detail carries the judge's or the
                            // floor's reason — record it where the operator
                            // reads the transcript, not just in the model's
                            // tool_result.
                            self.meta_line(out, format_args!("denied: {name} — {detail}"));
                        } else {
                            self.auto_denials = 0;
                            self.meta_line(out, format_args!("denied: {name}"));
                        }
                        slots.push(rejected(id, format!("{name} {detail}")));
                        continue;
                    }
                }
            }

            // The call is approved and already counted (reserved above, count
            // before run so retries also count) — no further budget bookkeeping.

            // Mark the turn as side-effecting at approval, before the fan-out:
            // a mutating tool may touch disk even if it then returns `Err`,
            // so the conversation history must be kept verbatim on a later
            // error this turn (see `run`'s conditional rollback).
            if tool.side_effecting(input) {
                self.side_effected_this_turn = true;
            }

            approved.push(ApprovedCall {
                slot: slots.len(),
                id,
                tool: tool.as_ref(),
                input,
            });
            slots.push(None);
        }

        // Phase 2 — fan-out, but only when every approved call is read-only.
        // Side-effecting calls never run concurrently: two mutations in one
        // batch can interfere through the filesystem (both read pre-turn
        // state; the later write silently drops the earlier one while both
        // results report success), so any mutation keeps the whole batch
        // inline, in block order — the pre-fan-out semantics. The case this
        // phase exists for — several independent I/O-bound reads — still
        // fans out: `run` is `&self` everywhere, `thread::scope` lets workers
        // borrow the tools with no `Arc`/`'static` ceremony, and inputs are
        // cloned on the worker (`run` takes ownership). A worker panic is a
        // tool bug — `run` reports failure as `Err` — and is re-raised on
        // the main thread, never swallowed.
        let read_only = approved
            .iter()
            .all(|call| !call.tool.side_effecting(call.input));
        // Each outcome carries the live output the call wrote to its sink. An
        // inline call writes straight to `out` — a serial `task` child streams
        // to the operator as it works — and carries an empty (never-allocated)
        // buffer; a fanned-out call writes into a private buffer instead,
        // flushed in block order in phase 3, so concurrent children cannot
        // interleave lines.
        //
        // A gated tool (`gates_concurrency`, the default for leaf tools) takes
        // a fan-out permit for the whole of its run, bounding concurrent leaf
        // executions process-wide; the `task` tool opts out, so a `task` worker
        // never holds a permit while its child fans out — the nested-semaphore
        // deadlock cannot form, since leaf tools never run nested tools. The
        // inline arm acquires per call, one at a time; the parallel arm acquires
        // *before* `scope.spawn` and moves the guard into the worker, so a batch
        // wider than the cap parks the extra acquisitions on the main thread
        // instead of spawning them — bounding live workers, not just running
        // executions.
        let concurrency = &self.concurrency;
        let outcomes: Vec<(Result<String, String>, Vec<u8>)> = if approved.len() <= 1 || !read_only
        {
            approved
                .iter()
                .map(|call| {
                    // The between-executions seam (inline batches): a
                    // cancellation during one tool skips the rest of the
                    // batch. The parallel arm below deliberately has no
                    // counterpart — its calls are all in flight at once,
                    // and letting read-only workers finish is simpler than
                    // a kill seam; cancellation lands at the next check.
                    let outcome = if self.cancelled() {
                        Err("cancelled by user".to_string())
                    } else {
                        let _permit = call.tool.gates_concurrency().then(|| concurrency.acquire());
                        call.tool.run(call.input.clone(), &mut *out)
                    };
                    (outcome, Vec::new())
                })
                .collect()
        } else {
            std::thread::scope(|scope| {
                let workers: Vec<_> = approved
                    .iter()
                    .map(|call| {
                        // Acquire on the main thread before spawning: the
                        // (K+1)th call parks here rather than becoming a
                        // spawned-but-waiting worker, so live workers and
                        // running executions are bounded together.
                        let permit = call.tool.gates_concurrency().then(|| concurrency.acquire());
                        scope.spawn(move || {
                            // The guard rides the worker for its whole run and
                            // releases on drop, panic included.
                            let _permit = permit;
                            let mut buffered = Vec::new();
                            let outcome = call.tool.run(call.input.clone(), &mut buffered);
                            (outcome, buffered)
                        })
                    })
                    .collect();
                workers
                    .into_iter()
                    .map(|w| w.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
                    .collect()
            })
        };

        // Phase 3 — serial reassembly, keyed by block index so the
        // tool_result order always matches the model's tool_use order.
        // Buffered live output and result lines print here, not in the
        // workers, so output stays deterministic regardless of completion
        // order.
        for (call, (outcome, buffered)) in approved.iter().zip(outcomes) {
            let _ = out.write_all(&buffered);
            let (content, is_error) = match outcome {
                Ok(output) => {
                    self.meta_line(out, format_args!("result: {}", summarize_output(&output)));
                    (output, false)
                }
                Err(error) => {
                    self.meta_line(
                        out,
                        format_args!("tool error: {}", sanitize_multiline(&error)),
                    );
                    (error, true)
                }
            };
            slots[call.slot] = Some(Block::ToolResult {
                tool_use_id: call.id.to_string(),
                content,
                is_error,
            });
        }

        // Every slot is now filled — rejected in pre-flight or computed above.
        slots
            .into_iter()
            .map(|slot| slot.expect("tool_use slot left unresolved"))
            .collect()
    }

    /// One compaction-summary attempt: the non-streaming `send` plus the
    /// completeness check. A response that didn't finish as plain text is
    /// not a summary — a tool-use or max_tokens turn yields empty or
    /// truncated text, and committing it would permanently lose the dropped
    /// context — so it fails exactly like a transport error.
    fn summarize(&self, request: &TurnRequest) -> Result<String, ApiError> {
        let turn = self.provider.send(request)?;
        let summary = turn
            .blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        if turn.stop_reason != StopReason::EndTurn || summary.is_empty() {
            return Err(ApiError::Stream(format!(
                "compaction summary incomplete or empty (stop reason {:?})",
                turn.stop_reason
            )));
        }
        Ok(summary)
    }

    /// Compact the conversation if the last measured prompt size crossed the
    /// threshold: summarize the oldest turn-groups via a non-streaming
    /// `provider.send()` call, fold the result into the rolling summary, and
    /// drop those messages.
    ///
    /// Summaries run between user turns, using measured usage and the current
    /// raw history size. Within a turn, every outbound request independently
    /// fits its tool-result text to the budget without changing the history.
    fn maybe_compact(&mut self, input: &str, out: &mut dyn Write) -> Result<(), AgentError> {
        let limit = self.config.context_token_limit as u64;
        let mut prospective = self.build_request();
        context::append_input(&mut prospective.messages, input);
        let oversized = context::input_size(&prospective)
            > context::input_budget(&prospective, self.config.context_token_limit);
        if (self.last_input_tokens as u64) * 100 < limit * COMPACT_THRESHOLD_PERCENT && !oversized {
            return Ok(());
        }
        // Preserve two recent groups normally. When immutable prompt content
        // cannot fit, allow one older complete group to be summarized rather
        // than preventing the next user turn from making any progress.
        let cut = compaction_cut(&self.messages, KEEP_RECENT_TURN_GROUPS).or_else(|| {
            if oversized
                && context::fit_request(prospective, self.config.context_token_limit).is_err()
            {
                compaction_cut(&self.messages, 1)
            } else {
                None
            }
        });
        let Some(cut) = cut else {
            return Ok(());
        };

        // The summary must be cumulative: this prefix's predecessors are
        // already gone, so the sub-call folds the existing summary in with the
        // prefix — otherwise each compaction would forget everything before
        // the previous one.
        let instruction = match &self.compacted_summary {
            Some(prior) => format!(
                "An earlier part of this conversation was already summarized \
                 as:\n\n{prior}\n\nWrite one replacement summary covering both \
                 that summary and the messages above."
            ),
            None => "Summarize the messages above.".to_string(),
        };
        let mut messages = self.messages[..cut].to_vec();
        messages.push(TurnMessage {
            role: Role::User,
            content: vec![Block::Text(instruction)],
        });
        let request = TurnRequest {
            model: self.config.model.clone(),
            max_tokens: SUMMARIZE_MAX_TOKENS.min(self.config.context_token_limit / 4),
            system: Some(SUMMARIZE_SYSTEM.to_string()),
            messages,
            // The prefix may carry tool_use blocks, and requests containing
            // them must define the tools that produced them.
            tools: self.tool_definitions(),
            // Summarization is routine work: it always runs at the model's
            // default effort, never the configured conversation effort.
            effort: None,
        };

        // Fail fast on a summary failure (settled decision): surfacing the
        // error beats silently sending the known-over-limit conversation.
        // One retry — no more — absorbs a transient blip (transport error or
        // malformed summary) before that policy applies; a second failure
        // surfaces unchanged, and nothing is drained.
        let request = self.fit_request(request, out)?;
        let summary = self
            .summarize(&request)
            .or_else(|_| self.summarize(&request))
            .map_err(AgentError::Api)?;
        // Verify the replacement before discarding its source. An oversized
        // summary or incoming prompt must leave the full old history available
        // for /save or a narrower follow-up.
        let mut prospective = self.build_request_with_summary(Some(&summary));
        prospective.messages.drain(..cut);
        context::append_input(&mut prospective.messages, input);
        context::fit_request(prospective, self.config.context_token_limit)?;
        self.compacted_summary = Some(summary);
        self.messages.drain(..cut);
        // The post-compaction size is unmeasurable until the next response
        // reports it, so reset and accept one turn of guard blindness — the
        // same trade-off as the startup/`/clear` case.
        self.last_input_tokens = 0;
        self.meta_line(out, format_args!("compacted {cut} messages into summary"));
        Ok(())
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

    /// Reset the conversation history, starting a fresh context. Backs the
    /// REPL's `/clear` command. The measured prompt size and the compaction
    /// summary are per-conversation state and reset with it — the summary
    /// reset is load-bearing, since `build_request` folds it into the system
    /// prompt and a stale one would leak the previous conversation into the
    /// fresh one. The tool budget ledger is intentionally left untouched —
    /// it bounds cost across the whole process, not per conversation, so
    /// clearing the history must not refill it.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.last_input_tokens = 0;
        self.compacted_summary = None;
    }

    /// Snapshot the conversation state — history, rolling summary, and the
    /// compaction guard's last measurement — as a persistable [`Session`].
    /// Pure, like [`Agent::restore`]: no disk I/O here, and none in `run()`
    /// either — the agent stays a pure state machine, and the REPL
    /// orchestrates persistence through its injected sink, mirroring how the
    /// provider-switch key/build seam lives in `run_repl`, not the agent.
    pub fn session(&self) -> Session {
        Session {
            version: SESSION_VERSION,
            messages: self.messages.clone(),
            compacted_summary: self.compacted_summary.clone(),
            last_input_tokens: self.last_input_tokens,
        }
    }

    /// Load a previously snapshotted conversation, replacing the current one.
    /// The restored `last_input_tokens` measures exactly the history being
    /// restored, so the compaction guard is armed from the first post-restore
    /// turn instead of riding one turn blind. The tool budget ledger is
    /// untouched for the same reason [`Agent::clear`] leaves it alone: it
    /// bounds cost across the whole process, not per conversation.
    pub fn restore(&mut self, session: Session) {
        self.messages = session.messages;
        // A pre-fix build could persist a trailing assistant message whose
        // tool_use was orphaned by a max_tokens cut-off (see `run_loop`);
        // loading it verbatim would 400 the first post-restore request. A
        // legitimately saved session never trails on an assistant tool_use (a
        // real tool round trails on the following User tool_result), so this
        // fires only on genuine poison: drop the unpaired tool_use, and drop the
        // message if nothing else survives.
        if let Some(last) = self.messages.last_mut()
            && last.role == Role::Assistant
            && last
                .content
                .iter()
                .any(|b| matches!(b, Block::ToolUse { .. }))
        {
            last.content.retain(|b| !matches!(b, Block::ToolUse { .. }));
            if last.content.is_empty() {
                self.messages.pop();
            }
        }
        self.compacted_summary = session.compacted_summary;
        self.last_input_tokens = session.last_input_tokens;
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

    /// Check every outbound request, including tool-loop continuations and
    /// summaries. Shortening only the outbound copy preserves the complete
    /// session for recovery and never separates a tool call from its result.
    fn fit_request(
        &self,
        request: TurnRequest,
        out: &mut dyn Write,
    ) -> Result<TurnRequest, AgentError> {
        let (request, shortened) = context::fit_request(request, self.config.context_token_limit)?;
        if shortened > 0 {
            self.meta_line(
                out,
                format_args!("context: shortened {shortened} tool results in outgoing request"),
            );
        }
        Ok(request)
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
    use super::*;
    use crate::provider::DeltaStream;
    use crate::testing::{
        ErrProvider, MockProvider, TestTool, expect_text, expect_tool_result, expect_tool_use,
    };
    use crate::turn::{Turn, Usage};
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::rc::Rc;

    // ── Test providers ──
    // MockProvider and ErrProvider live in crate::testing (shared with the
    // REPL tests); only the sequenced provider below is agent-specific.

    /// `stream` replays its queued streams in order, then fails every later
    /// call. Neither `MockProvider` (always succeeds) nor `ErrProvider`
    /// (always fails) can drive a "a tool ran, then a *later* turn's request
    /// errors" sequence — the case the conditional rollback turns on. `send`
    /// (the compaction summary path) always succeeds, so with no queued
    /// streams this is the compaction-succeeds-then-stream-errors provider;
    /// the stray tool-use block in its response pins the summary extraction's
    /// text-only filter.
    struct SucceedThenErrProvider {
        streams: RefCell<VecDeque<Vec<Result<StreamDelta, ApiError>>>>,
    }
    impl SucceedThenErrProvider {
        fn new(streams: Vec<Vec<StreamDelta>>) -> Self {
            Self {
                streams: RefCell::new(
                    streams
                        .into_iter()
                        .map(|stream| stream.into_iter().map(Ok).collect())
                        .collect(),
                ),
            }
        }

        fn with_result_stream(stream: Vec<Result<StreamDelta, ApiError>>) -> Self {
            Self {
                streams: RefCell::new(VecDeque::from([stream])),
            }
        }
    }
    impl Provider for SucceedThenErrProvider {
        fn send(&self, _request: &TurnRequest) -> Result<Turn, ApiError> {
            Ok(Turn {
                blocks: vec![
                    Block::ToolUse {
                        id: "t0".to_string(),
                        name: "noop".to_string(),
                        input: serde_json::json!({}),
                    },
                    Block::Text("prefix summary".to_string()),
                ],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
        fn stream(&self, _request: &TurnRequest) -> Result<DeltaStream, ApiError> {
            match self.streams.borrow_mut().pop_front() {
                Some(deltas) => Ok(Box::new(deltas.into_iter())),
                None => Err(ApiError::Io(std::io::Error::other("boom"))),
            }
        }
    }

    /// A canned stream carrying both a text block and a tool call, stopping on
    /// `ToolUse` — the assistant message it yields has text the salvage
    /// fallback can later find, while still driving the loop another round.
    fn text_and_tool_use_stream(id: &str, name: &str, text: &str) -> Vec<StreamDelta> {
        vec![
            message_start(),
            StreamDelta::TextStart {
                index: 0,
                text: text.to_string(),
            },
            StreamDelta::ToolUseStart {
                index: 1,
                id: id.to_string(),
                name: name.to_string(),
                input: serde_json::json!({}),
            },
            StreamDelta::ToolArgsDelta {
                index: 1,
                json: "{}".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            },
        ]
    }

    /// `MAX_TURNS` identical tool-call streams — enough to burn a nested
    /// child's whole round budget and reach the turn wall.
    fn streams_to_the_wall(stream: Vec<StreamDelta>) -> Vec<Vec<StreamDelta>> {
        std::iter::repeat_n(stream, MAX_TURNS as usize).collect()
    }

    /// True when no two adjacent messages share the `User` role — the
    /// alternating-role contract the coalescing must preserve.
    fn no_consecutive_users(messages: &[TurnMessage]) -> bool {
        messages
            .windows(2)
            .all(|w| !(w[0].role == Role::User && w[1].role == Role::User))
    }

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

    // ── Test tools & fixtures ──
    // Tool doubles are the shared, closure-configurable [`TestTool`] — a
    // bespoke impl per test would leave its unused trait methods as
    // permanently-dead lines under the coverage gate.

    fn mock_tool(name: &str, response: &str) -> Box<dyn ToolDef> {
        Box::new(TestTool::new(name, response))
    }

    /// A tool that reports itself side-effecting, like write_file/shell. Its
    /// `run` succeeds with a fixed string — the mutation it stands in for is
    /// notional; the `side_effecting` flag is what drives the rollback logic.
    fn side_effect_tool(name: &str, response: &str) -> Box<dyn ToolDef> {
        Box::new(TestTool::new(name, response).mutating())
    }

    /// A side-effecting tool whose `run` fails, modeling a mutation that
    /// touched disk and then errored. The flag is set *before* `run`, so
    /// history must still be preserved on a later error this turn.
    fn failing_side_effect_tool(name: &str, error: &str) -> Box<dyn ToolDef> {
        let error = error.to_string();
        Box::new(
            TestTool::new(name, "")
                .mutating()
                .with_run(move |_| Err(error.clone())),
        )
    }

    /// A canned stream that calls tool `name` (with id `id`) and stops on
    /// `ToolUse`, so the agent loop will execute the tool and loop again.
    fn tool_use_stream(id: &str, name: &str) -> Vec<StreamDelta> {
        vec![
            message_start(),
            StreamDelta::ToolUseStart {
                index: 0,
                id: id.to_string(),
                name: name.to_string(),
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
        ]
    }

    /// A canned stream that emits `text` and ends the turn (`EndTurn`).
    fn text_stream(text: &str) -> Vec<StreamDelta> {
        vec![
            message_start(),
            StreamDelta::TextStart {
                index: 0,
                text: text.to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            },
        ]
    }

    fn test_config(system: Option<&str>) -> AgentConfig {
        AgentConfig {
            provider_kind: ProviderKind::Anthropic,
            model: crate::TEST_MODEL.to_string(),
            max_tokens: 64,
            system: system.map(str::to_string),
            // Large enough that no test reaches the compaction threshold
            // unless it arms the guard itself.
            context_token_limit: 100_000,
            effort: None,
            max_turns: MAX_TURNS,
        }
    }

    /// An agent primed for the compaction tests: a `limit`-token context
    /// window and three complete turn-groups of plain text history
    /// (`q0`/`a0` … `q2`/`a2`). The guard itself stays dormant until the test
    /// sets `last_input_tokens`.
    fn compaction_agent(provider: Box<dyn Provider>, limit: u32) -> Agent {
        let mut agent = Agent::new(
            provider,
            AgentConfig {
                provider_kind: ProviderKind::Anthropic,
                model: crate::TEST_MODEL.to_string(),
                max_tokens: 64,
                system: None,
                context_token_limit: limit,
                effort: None,
                max_turns: MAX_TURNS,
            },
            vec![],
        );
        for i in 0..3 {
            agent.messages.push(user_msg(&format!("q{i}")));
            agent.messages.push(assistant_msg(&format!("a{i}")));
        }
        agent
    }

    fn agent_with_tools(tools: Vec<Box<dyn ToolDef>>) -> Agent {
        Agent::new(
            Box::new(MockProvider::new(vec![])),
            test_config(None),
            tools,
        )
    }

    /// An `ask` policy answering through `f` — what the pre-policy tests
    /// installed as a bare closure. `Send + Sync` now, so recorders use
    /// `Arc`, never `Rc`.
    fn ask_stub(f: impl Fn(&str) -> bool + Send + Sync + 'static) -> ConfirmPolicy {
        ConfirmPolicy::new(
            crate::config::ConfirmMode::Ask,
            f,
            crate::provider::ProviderFactory::from_fns(
                crate::testing::stub_provider,
                crate::testing::no_key,
            ),
            Vec::new(),
            crate::tools::sandbox::Sandbox::unbounded(),
        )
    }

    /// An `allow` policy over the shared inert seams — the prompt must never
    /// fire.
    fn allow_stub() -> ConfirmPolicy {
        ConfirmPolicy::new(
            crate::config::ConfirmMode::Allow,
            crate::testing::no_prompt,
            crate::provider::ProviderFactory::from_fns(
                crate::testing::stub_provider,
                crate::testing::no_key,
            ),
            Vec::new(),
            crate::tools::sandbox::Sandbox::unbounded(),
        )
    }

    /// A judge-mode policy whose provider is built by `build` per
    /// adjudication — the fixture behind the gate-dispatch, breaker, and
    /// payload tests.
    fn judged_policy(
        build: impl Fn() -> Box<dyn Provider> + Send + Sync + 'static,
    ) -> ConfirmPolicy {
        ConfirmPolicy::new(
            crate::config::ConfirmMode::Judge("sentinel".to_string()),
            crate::testing::no_prompt,
            crate::provider::ProviderFactory::from_fns(
                move |_kind, _key| build(),
                |_env| Some("k".to_string()),
            ),
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
        )
    }

    /// [`judged_policy`] replying `verdict` to every adjudication.
    fn judged_stub(verdict: &'static str) -> ConfirmPolicy {
        judged_policy(move || {
            Box::new(crate::testing::ThreadSafeProvider::echo().with_send_text(verdict))
        })
    }

    fn agent_with_system(system: &str) -> Agent {
        Agent::new(
            Box::new(MockProvider::new(vec![])),
            test_config(Some(system)),
            vec![],
        )
    }

    fn user_msg(text: &str) -> TurnMessage {
        TurnMessage {
            role: Role::User,
            content: vec![Block::Text(text.to_string())],
        }
    }

    fn assistant_msg(text: &str) -> TurnMessage {
        TurnMessage {
            role: Role::Assistant,
            content: vec![Block::Text(text.to_string())],
        }
    }

    fn stub_usage() -> Usage {
        Usage::default()
    }

    fn message_start() -> StreamDelta {
        StreamDelta::MessageStart {
            usage: stub_usage(),
        }
    }

    // ── tool_definitions ──

    #[test]
    fn tool_definitions_converts_trait_to_spec() {
        let agent = agent_with_tools(vec![mock_tool("echo", "ok")]);
        let defs = agent.tool_definitions();

        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");
        assert_eq!(defs[0].description, "A configurable test tool");
        assert_eq!(defs[0].input_schema["type"], "object");
    }

    #[test]
    fn tool_definitions_empty_when_no_tools() {
        let agent = agent_with_tools(vec![]);
        assert!(agent.tool_definitions().is_empty());
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

    // ── build_request ──

    #[test]
    fn build_request_includes_tools() {
        let mut agent = agent_with_tools(vec![mock_tool("echo", "ok")]);
        agent.messages.push(user_msg("hello"));
        let req = agent.build_request();

        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].name, "echo");
    }

    #[test]
    fn build_request_tools_empty_when_none() {
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("hello"));
        assert!(agent.build_request().tools.is_empty());
    }

    #[test]
    fn build_request_effort_is_none_by_default() {
        // With no effort configured the request carries none — the guarantee
        // that a default turn is byte-identical to the pre-effort wire shape.
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("hi"));
        assert!(agent.build_request().effort.is_none());
    }

    #[test]
    fn build_request_copies_configured_effort() {
        let mut agent = agent_with_tools(vec![]);
        agent.set_effort(Some("high".to_string()));
        agent.messages.push(user_msg("hi"));
        assert_eq!(agent.build_request().effort.as_deref(), Some("high"));
    }

    #[test]
    fn build_request_no_system_when_unset_and_no_tools() {
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("hi"));
        assert!(agent.build_request().system.is_none());
    }

    #[test]
    fn build_request_injects_tool_policy_when_tools_present() {
        let mut agent = agent_with_tools(vec![mock_tool("echo", "ok")]);
        agent.messages.push(user_msg("hi"));
        let req = agent.build_request();

        // System is set even though AgentConfig.system is None.
        let system = req.system.expect("system present");
        assert!(system.contains("Tool selection policy"));
    }

    #[test]
    fn build_request_appends_tool_policy_to_existing_system() {
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![])),
            test_config(Some("You are helpful.")),
            vec![mock_tool("echo", "ok")],
        );
        agent.messages.push(user_msg("hi"));
        let req = agent.build_request();

        let system = req.system.expect("system present");
        assert!(system.starts_with("You are helpful."));
        assert!(system.contains("Tool selection policy"));
    }

    #[test]
    fn build_request_carries_system_text_only() {
        // The agent emits plain system text; cache breakpoints are the adapter's
        // job, so nothing here wraps it in blocks.
        let mut agent = agent_with_system("You are helpful.");
        agent.messages.push(user_msg("hi"));
        assert_eq!(
            agent.build_request().system.as_deref(),
            Some("You are helpful.")
        );
    }

    #[test]
    fn build_request_does_not_mutate_agent_messages() {
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("hello"));
        agent.build_request();

        assert_eq!(
            agent.messages[0].content,
            vec![Block::Text("hello".to_string())]
        );
    }

    #[test]
    fn build_request_carries_messages_verbatim() {
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("a"));
        agent.messages.push(assistant_msg("b"));
        let req = agent.build_request();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0], user_msg("a"));
        assert_eq!(req.messages[1], assistant_msg("b"));
    }

    // ── process_stream ──

    #[test]
    fn process_stream_text_only() {
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: String::new(),
            }),
            Ok(StreamDelta::TextDelta {
                index: 0,
                text: "Hello ".to_string(),
            }),
            Ok(StreamDelta::TextDelta {
                index: 0,
                text: "world".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            }),
        ];

        let mut out = Vec::new();
        let result = agent
            .process_stream(&mut events.into_iter(), &mut out)
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(result.blocks, vec![Block::Text("Hello world".to_string())]);
        // The text is echoed to the writer as it streams, then the stream-end
        // newline; zero usage means no token line.
        assert_eq!(String::from_utf8(out).unwrap(), "Hello world\n");
    }

    #[test]
    fn process_stream_thinking_accumulates_text_and_signature_quietly() {
        let agent = agent_with_tools(vec![]);
        // Seeds from the start event, then reasoning split across deltas and
        // a signature split across two signature_delta events.
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ThinkingStart {
                index: 0,
                text: "seed ".to_string(),
                signature: "s0".to_string(),
            }),
            Ok(StreamDelta::ThinkingDelta {
                index: 0,
                text: "let me ".to_string(),
            }),
            Ok(StreamDelta::ThinkingDelta {
                index: 0,
                text: "reason".to_string(),
            }),
            Ok(StreamDelta::SignatureDelta {
                index: 0,
                signature: "s1".to_string(),
            }),
            Ok(StreamDelta::SignatureDelta {
                index: 0,
                signature: "s2".to_string(),
            }),
            Ok(StreamDelta::TextStart {
                index: 1,
                text: "the answer".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            }),
        ];

        let mut out = Vec::new();
        let result = agent
            .process_stream(&mut events.into_iter(), &mut out)
            .unwrap();

        assert_eq!(
            result.blocks,
            vec![
                Block::Thinking {
                    text: "seed let me reason".to_string(),
                    signature: "s0s1s2".to_string(),
                },
                Block::Text("the answer".to_string()),
            ]
        );
        // Display policy: one quiet meta line marks the block; the reasoning
        // text never reaches the terminal — only the answer does.
        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("[thinking]"));
        assert!(!printed.contains("let me reason"));
        assert!(printed.contains("the answer"));
    }

    #[test]
    fn process_stream_thinking_deltas_to_missing_or_mismatched_builders_are_ignored() {
        let agent = agent_with_tools(vec![]);
        // A thinking delta and a signature delta addressed past the builders,
        // plus both addressed to a *text* builder: all four are dropped
        // without corrupting the text block.
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "hi".to_string(),
            }),
            Ok(StreamDelta::ThinkingDelta {
                index: 0,
                text: "stray".to_string(),
            }),
            Ok(StreamDelta::SignatureDelta {
                index: 0,
                signature: "stray".to_string(),
            }),
            Ok(StreamDelta::ThinkingDelta {
                index: 9,
                text: "stray".to_string(),
            }),
            Ok(StreamDelta::SignatureDelta {
                index: 9,
                signature: "stray".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            }),
        ];

        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();
        assert_eq!(result.blocks, vec![Block::Text("hi".to_string())]);
    }

    #[test]
    fn process_stream_with_tool_use() {
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: String::new(),
            }),
            Ok(StreamDelta::TextDelta {
                index: 0,
                text: "Let me check.".to_string(),
            }),
            Ok(StreamDelta::ToolUseStart {
                index: 1,
                id: "toolu_abc".to_string(),
                name: "get_weather".to_string(),
                input: serde_json::json!({}),
            }),
            Ok(StreamDelta::ToolArgsDelta {
                index: 1,
                json: "{\"loc".to_string(),
            }),
            Ok(StreamDelta::ToolArgsDelta {
                index: 1,
                json: "ation\": \"SF\"}".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            }),
        ];

        let mut out = Vec::new();
        let result = agent
            .process_stream(&mut events.into_iter(), &mut out)
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::ToolUse);
        assert_eq!(result.blocks.len(), 2);
        assert_eq!(result.blocks[0], Block::Text("Let me check.".to_string()));
        // The tool call is announced on the output as it starts streaming.
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("[tool: get_weather]")
        );
        let (id, name, input) = expect_tool_use(&result.blocks[1]);
        assert_eq!(id, "toolu_abc");
        assert_eq!(name, "get_weather");
        assert_eq!(input["location"], "SF");
    }

    #[test]
    fn process_stream_multiple_tool_uses() {
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ToolUseStart {
                index: 0,
                id: "toolu_1".to_string(),
                name: "tool_a".to_string(),
                input: serde_json::json!({}),
            }),
            Ok(StreamDelta::ToolArgsDelta {
                index: 0,
                json: "{\"x\": 1}".to_string(),
            }),
            Ok(StreamDelta::ToolUseStart {
                index: 1,
                id: "toolu_2".to_string(),
                name: "tool_b".to_string(),
                input: serde_json::json!({}),
            }),
            Ok(StreamDelta::ToolArgsDelta {
                index: 1,
                json: "{\"y\": 2}".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            }),
        ];

        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();

        assert_eq!(result.blocks.len(), 2);
        let (_, name, input) = expect_tool_use(&result.blocks[0]);
        assert_eq!(name, "tool_a");
        assert_eq!(input["x"], 1);
        let (_, name, input) = expect_tool_use(&result.blocks[1]);
        assert_eq!(name, "tool_b");
        assert_eq!(input["y"], 2);
    }

    #[test]
    fn process_stream_empty_tool_input_yields_empty_object() {
        let agent = agent_with_tools(vec![]);
        // Tool with no parameters: a ToolUseStart with input {} and no arg deltas.
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ToolUseStart {
                index: 0,
                id: "toolu_1".to_string(),
                name: "noop".to_string(),
                input: serde_json::json!({}),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            }),
        ];

        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();

        let (_, _, input) = expect_tool_use(&result.blocks[0]);
        assert!(input.is_object(), "expected object, got: {input}");
        assert!(input.as_object().unwrap().is_empty());
    }

    #[test]
    fn process_stream_seeds_tool_input_from_start_event() {
        // Streaming sends {} on ToolUseStart, but a non-empty start input must
        // be preserved as the seed the arg deltas append to.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ToolUseStart {
                index: 0,
                id: "toolu_1".to_string(),
                name: "noop".to_string(),
                input: serde_json::json!({"pre": 1}),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            }),
        ];

        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();

        let (_, _, input) = expect_tool_use(&result.blocks[0]);
        assert_eq!(input["pre"], 1);
    }

    #[test]
    fn process_stream_malformed_tool_input_yields_null() {
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ToolUseStart {
                index: 0,
                id: "toolu_1".to_string(),
                name: "noop".to_string(),
                input: serde_json::json!({}),
            }),
            // Malformed JSON body — never assembles into a valid object.
            Ok(StreamDelta::ToolArgsDelta {
                index: 0,
                json: "{not valid json".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            }),
        ];

        let mut out = Vec::new();
        let result = agent
            .process_stream(&mut events.into_iter(), &mut out)
            .unwrap();

        // Malformed input is left Null; execute_tools turns that into an
        // is_error tool_result (see execute_tools_rejects_non_object_input).
        let (_, _, input) = expect_tool_use(&result.blocks[0]);
        assert!(input.is_null(), "expected Null, got: {input}");
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("[malformed tool input for noop:")
        );
    }

    #[test]
    fn process_stream_ignores_out_of_range_deltas() {
        // A delta whose index has no builder is harmlessly dropped.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextDelta {
                index: 5,
                text: "orphan".to_string(),
            }),
            Ok(StreamDelta::ToolArgsDelta {
                index: 9,
                json: "{}".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            }),
        ];

        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();
        assert!(result.blocks.is_empty());
    }

    #[test]
    fn process_stream_text_block_over_the_cap_errors() {
        // A text block that keeps growing past MAX_BLOCK_BYTES aborts the
        // turn with an error naming the cap. The first delta lands exactly on
        // the cap (inclusive), so only the byte after it overflows.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: String::new(),
            }),
            Ok(StreamDelta::TextDelta {
                index: 0,
                text: "x".repeat(MAX_BLOCK_BYTES),
            }),
            Ok(StreamDelta::TextDelta {
                index: 0,
                text: "y".to_string(),
            }),
        ];
        let result = agent.process_stream(&mut events.into_iter(), &mut std::io::sink());
        assert!(matches!(
            result,
            Err(AgentError::Api(ApiError::Stream(ref msg))) if msg.contains("MAX_BLOCK_BYTES")
        ));
    }

    #[test]
    fn process_stream_tool_args_over_the_cap_errors() {
        // Tool-args JSON accumulates under the same cap as text.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ToolUseStart {
                index: 0,
                id: "t1".to_string(),
                name: "big".to_string(),
                input: serde_json::json!({}),
            }),
            Ok(StreamDelta::ToolArgsDelta {
                index: 0,
                json: "x".repeat(MAX_BLOCK_BYTES + 1),
            }),
        ];
        let result = agent.process_stream(&mut events.into_iter(), &mut std::io::sink());
        assert!(matches!(
            result,
            Err(AgentError::Api(ApiError::Stream(ref msg))) if msg.contains("MAX_BLOCK_BYTES")
        ));
    }

    #[test]
    fn process_stream_thinking_over_the_cap_errors() {
        // Reasoning text accumulates under the same cap as text and tool
        // args — it is never displayed, so nothing else bounds it.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ThinkingStart {
                index: 0,
                text: String::new(),
                signature: String::new(),
            }),
            Ok(StreamDelta::ThinkingDelta {
                index: 0,
                text: "x".repeat(MAX_BLOCK_BYTES + 1),
            }),
        ];
        let result = agent.process_stream(&mut events.into_iter(), &mut std::io::sink());
        assert!(matches!(
            result,
            Err(AgentError::Api(ApiError::Stream(ref msg))) if msg.contains("MAX_BLOCK_BYTES")
        ));
    }

    #[test]
    fn process_stream_signature_over_the_cap_errors() {
        // The signature accumulator is bounded too: a hostile stream of
        // endless signature_delta events must not grow the process.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ThinkingStart {
                index: 0,
                text: String::new(),
                signature: String::new(),
            }),
            Ok(StreamDelta::SignatureDelta {
                index: 0,
                signature: "s".repeat(MAX_BLOCK_BYTES + 1),
            }),
        ];
        let result = agent.process_stream(&mut events.into_iter(), &mut std::io::sink());
        assert!(matches!(
            result,
            Err(AgentError::Api(ApiError::Stream(ref msg))) if msg.contains("MAX_BLOCK_BYTES")
        ));
    }

    #[test]
    fn process_stream_accumulates_cache_usage() {
        // Cache counters ride the accumulated usage; nothing prints here —
        // the report is the end-of-turn usage line, not stream chatter.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(StreamDelta::MessageStart {
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 0,
                    cache_creation_input_tokens: Some(10),
                    cache_read_input_tokens: Some(20),
                },
            }),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "hi".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            }),
        ];
        let mut out = Vec::new();
        let result = agent
            .process_stream(&mut events.into_iter(), &mut out)
            .unwrap();
        assert_eq!(result.blocks, vec![Block::Text("hi".to_string())]);
        // The cache counters survive on the returned usage, `Option`-ness intact.
        assert_eq!(
            result.usage,
            Usage {
                input_tokens: 1,
                output_tokens: 0,
                cache_creation_input_tokens: Some(10),
                cache_read_input_tokens: Some(20),
            }
        );
        assert_eq!(String::from_utf8(out).unwrap(), "hi\n");
    }

    #[test]
    fn process_stream_merges_cache_usage_from_message_delta() {
        // Anthropic's message_delta carries cumulative usage, cache counters
        // included. A trailing sighting merges by max — it must neither be
        // dropped nor double-counted — and a counter first sighted there
        // (read: None then Some) still lands on the result.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(StreamDelta::MessageStart {
                usage: Usage {
                    input_tokens: 40,
                    output_tokens: 0,
                    cache_creation_input_tokens: Some(5),
                    cache_read_input_tokens: None,
                },
            }),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "hi".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 40,
                    output_tokens: 9,
                    cache_creation_input_tokens: Some(12),
                    cache_read_input_tokens: Some(30),
                },
            }),
        ];
        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();
        assert_eq!(
            result.usage,
            Usage {
                input_tokens: 40,
                output_tokens: 9,
                cache_creation_input_tokens: Some(12),
                cache_read_input_tokens: Some(30),
            }
        );
    }

    #[test]
    fn process_stream_accumulates_token_usage() {
        // Anthropic-shaped split: input up front, output in the trailing delta.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(StreamDelta::MessageStart {
                usage: Usage {
                    input_tokens: 42,
                    output_tokens: 1,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            }),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "hi".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 17,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            }),
        ];
        let mut out = Vec::new();
        let result = agent
            .process_stream(&mut events.into_iter(), &mut out)
            .unwrap();
        assert_eq!(result.blocks, vec![Block::Text("hi".to_string())]);
        // Max-accumulation across events: 42 in from MessageStart, 17 out
        // from the trailing MessageDelta.
        assert_eq!(
            result.usage,
            Usage {
                input_tokens: 42,
                output_tokens: 17,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            }
        );
        // The stream itself stays quiet about usage; the per-turn line is
        // run_loop's job.
        assert_eq!(String::from_utf8(out).unwrap(), "hi\n");
    }

    #[test]
    fn process_stream_accumulates_token_usage_trailing_only() {
        // OpenAI-shaped: MessageStart carries no usage, both counts arrive in
        // the trailing MessageDelta. The max-accumulation must pick up both
        // from the delta — the path the Anthropic-shaped test doesn't exercise.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(StreamDelta::MessageStart {
                usage: Usage::default(),
            }),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "hi".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 30,
                    output_tokens: 12,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            }),
        ];
        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();
        assert_eq!(result.blocks, vec![Block::Text("hi".to_string())]);
        assert_eq!(
            result.usage,
            Usage {
                input_tokens: 30,
                output_tokens: 12,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            }
        );
    }

    #[test]
    fn process_stream_propagates_error() {
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> =
            vec![Err(ApiError::Io(std::io::Error::other("nope")))];
        assert!(matches!(
            agent.process_stream(&mut events.into_iter(), &mut std::io::sink()),
            Err(AgentError::Api(_))
        ));
    }

    #[test]
    fn process_stream_rejects_a_missing_stop_reason() {
        // A provider-neutral stream that ends without explicitly normalizing
        // its terminal reason is incomplete, not an implicit EndTurn.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "done".to_string(),
            }),
        ];
        let result = agent.process_stream(&mut events.into_iter(), &mut std::io::sink());
        assert!(matches!(
            result,
            Err(AgentError::Api(ApiError::Stream(ref message)))
                if message == "stream ended without a normalized stop reason"
        ));
    }

    // ── execute_tools ──

    #[test]
    fn execute_tools_dispatches_and_returns_result() {
        let mut agent = agent_with_tools(vec![mock_tool("echo", "hello back")]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "echo".to_string(),
            input: serde_json::json!({"msg": "hi"}),
        }];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(
            results,
            vec![Block::ToolResult {
                tool_use_id: "toolu_1".to_string(),
                content: "hello back".to_string(),
                is_error: false,
            }]
        );
    }

    #[test]
    fn execute_tools_inline_call_streams_live_output_to_the_sink() {
        // An inline (batch-of-one) call gets the parent's own sink: its live
        // output lands on the terminal as it runs, before the result line.
        let streamy = TestTool::new("streamy", "done").emitting("live line\n");
        let mut agent = agent_with_tools(vec![Box::new(streamy)]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "streamy".to_string(),
            input: serde_json::json!({}),
        }];

        let mut out = Vec::new();
        let results = agent.execute_tools(&blocks, &mut out);
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!((content, is_error), ("done", false));

        let shown = String::from_utf8(out).unwrap();
        let live = shown.find("live line").expect("live output shown");
        let result = shown.find("[result: done]").expect("result line shown");
        assert!(live < result, "live output precedes the result line");
    }

    #[test]
    fn execute_tools_parallel_batch_flushes_live_output_in_block_order() {
        // Fanned-out calls write into private buffers; the loop flushes them
        // in block order after the batch — each call's stream, then its
        // result line — regardless of which worker finished first.
        let first = TestTool::new("first", "first done").emitting("first stream\n");
        let second = TestTool::new("second", "second done").emitting("second stream\n");
        let mut agent = agent_with_tools(vec![Box::new(first), Box::new(second)]);
        let blocks = vec![
            Block::ToolUse {
                id: "toolu_1".to_string(),
                name: "first".to_string(),
                input: serde_json::json!({}),
            },
            Block::ToolUse {
                id: "toolu_2".to_string(),
                name: "second".to_string(),
                input: serde_json::json!({}),
            },
        ];

        let mut out = Vec::new();
        let results = agent.execute_tools(&blocks, &mut out);
        assert_eq!(results.len(), 2);

        let shown = String::from_utf8(out).unwrap();
        let stream_a = shown.find("first stream").expect("first call's stream");
        let result_a = shown.find("[result: first done]").expect("first result");
        let stream_b = shown.find("second stream").expect("second call's stream");
        let result_b = shown.find("[result: second done]").expect("second result");
        assert!(stream_a < result_a, "call 0 streams before its result line");
        assert!(
            result_a < stream_b,
            "block order: call 0 fully before call 1"
        );
        assert!(stream_b < result_b, "call 1 streams before its result line");
    }

    #[test]
    fn execute_tools_unknown_tool_self_corrects_with_error() {
        let mut agent = agent_with_tools(vec![]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "nonexistent".to_string(),
            input: serde_json::json!({}),
        }];

        // A hallucinated tool name must not abort the turn — it returns a
        // tool_result with is_error so the model can retry with a real tool.
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (tool_use_id, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(tool_use_id, "toolu_1");
        assert!(content.contains("unknown tool: nonexistent"));
        assert!(is_error);
    }

    #[test]
    fn execute_tools_rejects_non_object_input() {
        // A malformed streamed input surfaces as a non-object Value (Null).
        // execute_tools must reject it rather than dispatch on bad input.
        let mut agent = agent_with_tools(vec![mock_tool("echo", "hello back")]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "echo".to_string(),
            input: serde_json::Value::Null,
        }];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert!(content.contains("expected a JSON object"));
        assert!(is_error);
    }

    #[test]
    fn execute_tools_sends_is_error_on_failure() {
        let failing = TestTool::new("fail", "").with_run(|_| Err("something broke".to_string()));
        let mut agent = agent_with_tools(vec![Box::new(failing)]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "fail".to_string(),
            input: serde_json::json!({}),
        }];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "something broke");
        assert!(is_error);
    }

    #[test]
    fn execute_tools_rejects_on_validation_failure() {
        let guarded = TestTool::new("guarded", "guarded ran").with_validate(|input| {
            if input["bad"].as_bool() == Some(true) {
                Err("bad input rejected".to_string())
            } else {
                Ok(())
            }
        });
        let mut agent = agent_with_tools(vec![Box::new(guarded)]);

        // Rejected by validate — run should NOT be called.
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "guarded".to_string(),
            input: serde_json::json!({"bad": true}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "bad input rejected");
        assert!(is_error);

        // Passes validation — run proceeds normally.
        let blocks = vec![Block::ToolUse {
            id: "toolu_2".to_string(),
            name: "guarded".to_string(),
            input: serde_json::json!({"bad": false}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "guarded ran");
        assert!(!is_error);
    }

    #[test]
    fn execute_tools_enforces_budget_limit() {
        let costly = TestTool::new("costly", "done").with_cost(4);
        let mut agent = agent_with_tools(vec![Box::new(costly)]);
        agent.budget.set_limit(4, 2);

        let block = |id: &str| Block::ToolUse {
            id: id.to_string(),
            name: "costly".to_string(),
            input: serde_json::json!({}),
        };
        let is_error =
            |result: &Block| -> bool { matches!(result, Block::ToolResult { is_error: true, .. }) };

        // First two calls succeed.
        let mut out = std::io::sink();
        assert!(!is_error(&agent.execute_tools(&[block("t1")], &mut out)[0]));
        assert!(!is_error(&agent.execute_tools(&[block("t2")], &mut out)[0]));

        // Third call is rejected — budget exceeded.
        let results = agent.execute_tools(&[block("t3")], &mut out);
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert!(content.contains("budget exceeded"));
        assert!(is_error);
    }

    /// A confirmation-gated double. The canned response goes through
    /// [`TestTool::new`]'s shared default `run`, so the denied test (where
    /// `run` must not fire) adds no never-executed closure.
    fn confirmed_tool() -> Box<dyn ToolDef> {
        Box::new(TestTool::new("guarded_write", "wrote something").confirmed())
    }

    #[test]
    fn execute_tools_denied_by_confirmation() {
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(ask_stub(|_summary| false));

        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "guarded_write".to_string(),
            input: serde_json::json!({}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert!(content.contains("denied by user"));
        assert!(is_error);
    }

    #[test]
    fn execute_tools_approved_by_confirmation() {
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(ask_stub(|_summary| true));

        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "guarded_write".to_string(),
            input: serde_json::json!({}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "wrote something");
        assert!(!is_error);
    }

    #[test]
    fn file_approval_receives_full_preview_before_mutation_and_can_decline() {
        use crate::tools::{edit_file::EditFileTool, sandbox::Sandbox, write_file::WriteFileTool};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        std::fs::write(&path, "old\n").unwrap();
        let sandbox = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let mut agent = agent_with_tools(vec![
            Box::new(WriteFileTool::new(sandbox.clone())),
            Box::new(EditFileTool::new(sandbox)),
        ]);
        let inspected_path = path.clone();
        agent.set_confirm_policy(ask_stub(move |summary| {
            assert!(summary.contains("-     1 | old\n+     1 | new"));
            assert_eq!(std::fs::read_to_string(&inspected_path).unwrap(), "old\n");
            false
        }));
        let call = Block::ToolUse {
            id: "preview".to_string(),
            name: "write_file".to_string(),
            input: serde_json::json!({"path": "file", "content": "new\n"}),
        };
        let results = agent.execute_tools(&[call], &mut std::io::sink());
        assert!(expect_tool_result(&results[0]).2);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "old\n");
        let inspected_path = path.clone();
        agent.set_confirm_policy(ask_stub(move |summary| {
            assert!(summary.contains("-     1 | old\n+     1 | new"));
            assert_eq!(std::fs::read_to_string(&inspected_path).unwrap(), "old\n");
            true
        }));
        let call = Block::ToolUse {
            id: "edit".to_string(),
            name: "edit_file".to_string(),
            input: serde_json::json!({"path": "file", "old_str": "old", "new_str": "new"}),
        };
        let results = agent.execute_tools(&[call], &mut std::io::sink());
        assert!(!expect_tool_result(&results[0]).2);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new\n");
    }

    #[test]
    fn oversized_preview_rejects_without_prompt_or_write_and_releases_budget() {
        use crate::tools::{sandbox::Sandbox, write_file::WriteFileTool};
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(Sandbox::rooted(dir.path().to_path_buf()).unwrap());
        let mut agent = agent_with_tools(vec![Box::new(tool)]);
        agent.budget.set_limit(2, 1);
        let prompts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let prompt_count = prompts.clone();
        agent.set_confirm_policy(ask_stub(move |_| {
            prompt_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            true
        }));
        let call = |content: String| Block::ToolUse {
            id: "write".to_string(),
            name: "write_file".to_string(),
            input: serde_json::json!({"path": "file", "content": content}),
        };
        let large = "x".repeat(20 * 1024);
        let results = agent.execute_tools(&[call(large.clone())], &mut std::io::sink());
        let (_, text, error) = expect_tool_result(&results[0]);
        assert!(error && text.contains("preview omitted") && text.contains("smaller edits"));
        assert_eq!(prompts.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert!(!dir.path().join("file").exists());
        let results = agent.execute_tools(&[call("small".to_string())], &mut std::io::sink());
        assert!(!expect_tool_result(&results[0]).2);
        assert_eq!(prompts.load(std::sync::atomic::Ordering::Relaxed), 1);
        // The display cap does not silently become a write-size cap for an
        // explicitly unattended policy, which still receives the exact input.
        agent.budget.set_limit(2, 2);
        agent.set_confirm_policy(allow_stub());
        let results = agent.execute_tools(&[call(large.clone())], &mut std::io::sink());
        assert!(!expect_tool_result(&results[0]).2);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("file")).unwrap(),
            large
        );
    }

    #[test]
    fn execute_tools_validation_before_confirmation() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let guarded = TestTool::new("guarded", "ran")
            .confirmed()
            .with_validate(|input| {
                if input["valid"].as_bool() == Some(true) {
                    Ok(())
                } else {
                    Err("invalid input".to_string())
                }
            });
        let mut agent = agent_with_tools(vec![Box::new(guarded)]);

        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();
        agent.set_confirm_policy(ask_stub(move |_summary| {
            called_clone.store(true, Ordering::SeqCst);
            true
        }));

        // Invalid input — should fail validation WITHOUT triggering confirm.
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "guarded".to_string(),
            input: serde_json::json!({"valid": false}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert!(content.contains("invalid input"));
        assert!(is_error);
        assert!(
            !called.load(Ordering::SeqCst),
            "confirm should not be called for invalid input"
        );

        // Valid input — confirm fires, then run proceeds.
        let blocks = vec![Block::ToolUse {
            id: "toolu_2".to_string(),
            name: "guarded".to_string(),
            input: serde_json::json!({"valid": true}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "ran");
        assert!(!is_error);
        assert!(
            called.load(Ordering::SeqCst),
            "confirm must be called for valid input"
        );
    }

    #[test]
    fn execute_tools_skips_confirmation_when_not_required() {
        // TestTool has requires_confirmation() = false unless .confirmed().
        let mut agent = agent_with_tools(vec![mock_tool("echo", "ok")]);
        agent.set_confirm_policy(ask_stub(|_summary| false));

        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "echo".to_string(),
            input: serde_json::json!({}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "ok");
        assert!(!is_error);
    }

    #[test]
    fn execute_tools_handles_multiple_tool_calls() {
        let mut agent = agent_with_tools(vec![
            mock_tool("tool_a", "result_a"),
            mock_tool("tool_b", "result_b"),
        ]);
        let blocks = vec![
            Block::Text("thinking...".to_string()),
            Block::ToolUse {
                id: "toolu_1".to_string(),
                name: "tool_a".to_string(),
                input: serde_json::json!({}),
            },
            Block::ToolUse {
                id: "toolu_2".to_string(),
                name: "tool_b".to_string(),
                input: serde_json::json!({}),
            },
        ];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        // Only ToolUse blocks produce results — Text blocks are skipped.
        assert_eq!(results.len(), 2);
        assert_eq!(expect_tool_result(&results[0]).1, "result_a");
        assert_eq!(expect_tool_result(&results[1]).1, "result_b");
    }

    // ── execute_tools: parallel fan-out ──

    /// A bare tool_use block for the fan-out tests.
    fn tool_use_block(id: &str, name: &str) -> Block {
        Block::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input: serde_json::json!({}),
        }
    }

    #[test]
    fn execute_tools_runs_approved_calls_concurrently_in_block_order() {
        use std::sync::{Mutex, mpsc};
        use std::time::Duration;

        // A rendezvous pins genuine concurrency without timing assertions:
        // the FIRST block's tool cannot finish until the SECOND block's tool
        // has run. Sequential in-block-order execution would time the waiter
        // out into a worker panic; only a parallel fan-out completes it. The
        // results must still come back in block order, not completion order.
        // The unwraps (rather than mapping to Err) keep the failure paths
        // free of never-executed closures under the coverage gate.
        let (tx, rx) = mpsc::channel();
        let rx = Mutex::new(rx);
        let waiter = TestTool::new("waiter", "").with_run(move |_| {
            rx.lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
            Ok("waiter done".to_string())
        });
        let signaler = TestTool::new("signaler", "").with_run(move |_| {
            tx.send(()).unwrap();
            Ok("signaler done".to_string())
        });
        let mut agent = agent_with_tools(vec![Box::new(waiter), Box::new(signaler)]);

        let blocks = vec![
            tool_use_block("t_wait", "waiter"),
            tool_use_block("t_sig", "signaler"),
        ];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(results.len(), 2);
        assert_eq!(
            expect_tool_result(&results[0]),
            ("t_wait", "waiter done", false)
        );
        assert_eq!(
            expect_tool_result(&results[1]),
            ("t_sig", "signaler done", false)
        );
    }

    // ── execute_tools: fan-out concurrency cap ──

    /// Shared instrumentation for the concurrency-cap rendezvous tests: a latch
    /// recording the peak number of tools inside `run` at once, holding every
    /// worker until at least `gate` have arrived together. Forcing that overlap
    /// makes the recorded peak deterministic — it equals the permit ceiling
    /// when the cap works, and would exceed it if the cap were removed.
    struct Rendezvous {
        active: usize,
        peak: usize,
        open: bool,
    }

    /// A tool whose `run` reports itself into a shared [`Rendezvous`]: it counts
    /// in, records the peak, opens the latch once `gate` workers overlap, then
    /// waits for the latch before counting out.
    fn rendezvous_tool(
        name: &str,
        state: std::sync::Arc<(std::sync::Mutex<Rendezvous>, std::sync::Condvar)>,
        gate: usize,
    ) -> Box<dyn ToolDef> {
        Box::new(TestTool::new(name, "done").with_run(move |_| {
            let (lock, cvar) = &*state;
            let mut s = lock.lock().unwrap();
            s.active += 1;
            s.peak = s.peak.max(s.active);
            if s.active >= gate {
                s.open = true;
                cvar.notify_all();
            }
            while !s.open {
                s = cvar.wait(s).unwrap();
            }
            s.active -= 1;
            Ok("done".to_string())
        }))
    }

    fn fresh_rendezvous() -> std::sync::Arc<(std::sync::Mutex<Rendezvous>, std::sync::Condvar)> {
        std::sync::Arc::new((
            std::sync::Mutex::new(Rendezvous {
                active: 0,
                peak: 0,
                open: false,
            }),
            std::sync::Condvar::new(),
        ))
    }

    #[test]
    fn execute_tools_caps_concurrent_leaf_executions_at_the_permit_count() {
        // More approved gated calls than permits: the fan-out must never run
        // more than K at once. The latch forces exactly K to overlap, so the
        // peak is deterministic — K when the cap holds, and (with the cap
        // removed) N, which this assertion would catch. Because permits are
        // acquired *before* spawn, this is the live-worker bound too: a worker
        // exists only once it holds a permit, so spawned-and-running leaf
        // workers never exceed K either.
        const K: usize = 2;
        const N: usize = 4;
        let state = fresh_rendezvous();
        let tools: Vec<Box<dyn ToolDef>> = (0..N)
            .map(|i| rendezvous_tool(&format!("rz{i}"), std::sync::Arc::clone(&state), K))
            .collect();
        let mut agent = agent_with_tools(tools);
        agent.set_concurrency(Concurrency::with_permits(K));

        let blocks: Vec<Block> = (0..N)
            .map(|i| tool_use_block(&format!("t{i}"), &format!("rz{i}")))
            .collect();
        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(results.len(), N);
        assert!(
            results.iter().all(|b| matches!(
                b,
                Block::ToolResult {
                    is_error: false,
                    ..
                }
            )),
            "every gated call completes"
        );
        assert_eq!(
            state.0.lock().unwrap().peak,
            K,
            "at most (and exactly) K leaf tools run concurrently"
        );
    }

    /// A non-gating dispatch tool (like `task`) that, inside its `run`, builds
    /// a fresh child [`Agent`] sharing `pool` and drives *its* fan-out of one
    /// gated leaf — the child is built on the worker thread and never sent
    /// across it, the parent→child shape the `task` tool creates. Not gating is
    /// what lets the parent fan several out at once *and* avoids the nested
    /// deadlock (a gating dispatcher would hold a permit while its child needs
    /// one from the same pool).
    fn child_leaf_dispatch(
        name: &str,
        leaf: &str,
        pool: Concurrency,
        state: std::sync::Arc<(std::sync::Mutex<Rendezvous>, std::sync::Condvar)>,
        gate: usize,
    ) -> Box<dyn ToolDef> {
        let leaf = leaf.to_string();
        Box::new(
            TestTool::new(name, "dispatched")
                .ungated()
                .with_run(move |_| {
                    let tool = rendezvous_tool(&leaf, std::sync::Arc::clone(&state), gate);
                    let mut child = agent_with_tools(vec![tool]);
                    child.set_concurrency(pool.clone());
                    let block = tool_use_block("leaf", &leaf);
                    child.execute_tools(std::slice::from_ref(&block), &mut std::io::sink());
                    Ok("dispatched".to_string())
                }),
        )
    }

    #[test]
    fn parent_and_children_share_one_fan_out_pool() {
        // The parent fans out N non-gating dispatch tools; each builds a child
        // agent — sharing the ONE pool — that runs a gated leaf. So N children
        // run concurrently, and their leaves all draw from the parent's permits.
        // With the pool at K, the latch forces exactly K leaves (belonging to
        // K different child agents) to overlap: peak == K. A regression giving
        // each agent its own pool would let all N leaves run at once — peak N —
        // which this catches. This is the `task` tool's parent→child pool
        // sharing, exercised through the real fan-out path.
        const K: usize = 2;
        const N: usize = 4;
        let pool = Concurrency::with_permits(K);
        let state = fresh_rendezvous();
        let tools: Vec<Box<dyn ToolDef>> = (0..N)
            .map(|i| {
                child_leaf_dispatch(
                    &format!("dispatch{i}"),
                    &format!("leaf{i}"),
                    pool.clone(),
                    std::sync::Arc::clone(&state),
                    K,
                )
            })
            .collect();
        let mut parent = agent_with_tools(tools);
        parent.set_concurrency(pool.clone());

        let blocks: Vec<Block> = (0..N)
            .map(|i| tool_use_block(&format!("t{i}"), &format!("dispatch{i}")))
            .collect();
        let results = parent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(results.len(), N);
        assert_eq!(
            state.0.lock().unwrap().peak,
            K,
            "leaves across child agents share the parent's permits"
        );
    }

    #[test]
    fn execute_tools_serializes_the_batch_when_any_call_is_side_effecting() {
        use std::sync::{Arc, Mutex};

        // One mutation in the batch keeps the WHOLE batch sequential: both
        // runs must happen on the calling thread (the fan-out would put them
        // on scoped workers), in block order — the pre-5a semantics that stop
        // two mutations racing each other through the filesystem. Thread
        // identity makes the proof deterministic in both directions: a
        // regression to fan-out cannot produce the main thread's id.
        let log = Arc::new(Mutex::new(Vec::new()));
        let mutator_log = Arc::clone(&log);
        let mutator = TestTool::new("mutator", "").mutating().with_run(move |_| {
            mutator_log
                .lock()
                .unwrap()
                .push(("mutator", std::thread::current().id()));
            Ok("mutated".to_string())
        });
        let reader_log = Arc::clone(&log);
        let reader = TestTool::new("reader", "").with_run(move |_| {
            reader_log
                .lock()
                .unwrap()
                .push(("reader", std::thread::current().id()));
            Ok("read".to_string())
        });
        let mut agent = agent_with_tools(vec![Box::new(mutator), Box::new(reader)]);

        let blocks = vec![
            tool_use_block("t_mut", "mutator"),
            tool_use_block("t_read", "reader"),
        ];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(expect_tool_result(&results[0]), ("t_mut", "mutated", false));
        assert_eq!(expect_tool_result(&results[1]), ("t_read", "read", false));
        let main = std::thread::current().id();
        assert_eq!(
            *log.lock().unwrap(),
            vec![("mutator", main), ("reader", main)],
            "a batch containing a mutation must run inline, in block order"
        );
    }

    #[test]
    #[should_panic(expected = "tool exploded")]
    fn execute_tools_reraises_a_worker_panic() {
        // A panicking `run` is a tool bug — failure is reported as `Err` —
        // so the fan-out re-raises the original payload on the main thread
        // instead of swallowing it into a tool_result.
        let bomb = TestTool::new("bomb", "").with_run(|_| panic!("tool exploded"));
        let mut agent = agent_with_tools(vec![Box::new(bomb), mock_tool("calm", "ok")]);
        let blocks = vec![tool_use_block("t1", "bomb"), tool_use_block("t2", "calm")];

        let _ = agent.execute_tools(&blocks, &mut std::io::sink());
    }

    #[test]
    fn execute_tools_reassembles_mixed_approved_and_rejected_in_order() {
        // A rejected call BETWEEN two approved ones: the rejection resolves in
        // pre-flight, the approved pair fans out, and reassembly must slot all
        // three back in the model's block order under their own ids.
        let mut agent = agent_with_tools(vec![
            mock_tool("tool_a", "result_a"),
            mock_tool("tool_b", "result_b"),
        ]);
        let blocks = vec![
            tool_use_block("t1", "tool_a"),
            tool_use_block("t2", "ghost"),
            tool_use_block("t3", "tool_b"),
        ];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(results.len(), 3);
        assert_eq!(expect_tool_result(&results[0]), ("t1", "result_a", false));
        let (id, content, is_error) = expect_tool_result(&results[1]);
        assert_eq!(id, "t2");
        assert!(content.contains("unknown tool: ghost"));
        assert!(is_error);
        assert_eq!(expect_tool_result(&results[2]), ("t3", "result_b", false));
    }

    #[test]
    fn execute_tools_all_rejected_yields_only_preflight_errors() {
        // Every call fails pre-flight, so the fan-out has nothing to run and
        // the results are exactly the pre-flight rejections, in block order.
        let mut agent = agent_with_tools(vec![]);
        let blocks = vec![
            tool_use_block("t1", "ghost_a"),
            tool_use_block("t2", "ghost_b"),
        ];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(results.len(), 2);
        let (id, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(id, "t1");
        assert!(content.contains("unknown tool: ghost_a"));
        assert!(is_error);
        let (id, content, is_error) = expect_tool_result(&results[1]);
        assert_eq!(id, "t2");
        assert!(content.contains("unknown tool: ghost_b"));
        assert!(is_error);
    }

    #[test]
    fn execute_tools_budget_counts_siblings_within_one_turn() {
        // Three calls to a limit-2 tier arrive in ONE turn: approval bumps the
        // count during pre-flight, so the third call must see its two siblings
        // and reject — a regression the one-call-per-turn budget test above
        // cannot catch.
        let costly = TestTool::new("costly", "done").with_cost(4);
        let mut agent = agent_with_tools(vec![Box::new(costly)]);
        agent.budget.set_limit(4, 2);
        let blocks = vec![
            tool_use_block("t1", "costly"),
            tool_use_block("t2", "costly"),
            tool_use_block("t3", "costly"),
        ];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(results.len(), 3);
        assert_eq!(expect_tool_result(&results[0]), ("t1", "done", false));
        assert_eq!(expect_tool_result(&results[1]), ("t2", "done", false));
        let (id, content, is_error) = expect_tool_result(&results[2]);
        assert_eq!(id, "t3");
        assert!(content.contains("budget exceeded: 2 calls at cost tier 4 (limit 2)"));
        assert!(is_error);
    }

    #[test]
    fn execute_tools_two_agents_sharing_a_ledger_cannot_jointly_exceed_the_limit() {
        // Two independent `Agent`s wired to the SAME ledger — the shape a
        // parent/child pair spawned through the `task` tool has.
        // Interleaving calls between them proves the ceiling is combined,
        // not per-agent: a regression back to each `Agent` owning its own
        // budget would let both agents separately reach the limit, doubling
        // the effective ceiling.
        let ledger = BudgetLedger::new();
        ledger.set_limit(4, 2);

        let mut agent_a =
            agent_with_tools(vec![Box::new(TestTool::new("costly", "done").with_cost(4))]);
        agent_a.set_budget_ledger(ledger.clone());
        let mut agent_b =
            agent_with_tools(vec![Box::new(TestTool::new("costly", "done").with_cost(4))]);
        agent_b.set_budget_ledger(ledger.clone());

        let block = tool_use_block("t1", "costly");
        let mut out = std::io::sink();

        // One call from each agent exhausts the shared limit of 2.
        let (_, _, is_error) =
            expect_tool_result(&agent_a.execute_tools(std::slice::from_ref(&block), &mut out)[0]);
        assert!(!is_error);
        let (_, _, is_error) =
            expect_tool_result(&agent_b.execute_tools(std::slice::from_ref(&block), &mut out)[0]);
        assert!(!is_error);

        // A further call from EITHER agent is rejected — the ceiling is
        // shared, not doubled.
        let results = agent_a.execute_tools(std::slice::from_ref(&block), &mut out);
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert!(is_error);
        assert!(content.contains("budget exceeded: 2 calls at cost tier 4 (limit 2)"));
        let (_, _, is_error) =
            expect_tool_result(&agent_b.execute_tools(std::slice::from_ref(&block), &mut out)[0]);
        assert!(is_error);

        assert_eq!(ledger.count(4), 2);
    }

    #[test]
    fn execute_tools_confirmation_prompts_stay_ordered_and_serial() {
        // Two gated tools in one turn: both prompts fire in block order during
        // the serial pre-flight (they share one stdin — never a worker), and
        // denying only the second leaves the first's approval intact.
        let mut agent = agent_with_tools(vec![
            Box::new(TestTool::new("first", "first ran").confirmed()),
            Box::new(TestTool::new("second", "second ran").confirmed()),
        ]);
        let prompts = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorder = Arc::clone(&prompts);
        agent.set_confirm_policy(ask_stub(move |summary| {
            recorder.lock().unwrap().push(summary.to_string());
            summary != "second"
        }));
        let blocks = vec![
            tool_use_block("t1", "first"),
            tool_use_block("t2", "second"),
        ];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(*prompts.lock().unwrap(), vec!["first", "second"]);
        assert_eq!(expect_tool_result(&results[0]), ("t1", "first ran", false));
        let (_, content, is_error) = expect_tool_result(&results[1]);
        assert!(content.contains("second denied by user"));
        assert!(is_error);
    }

    // ── Confirmation policy dispatch (allow / judge) ──

    #[test]
    fn execute_tools_allow_mode_runs_with_a_notice() {
        // The human prompt is waved (the stub panics if consulted); the call
        // runs, and the transcript records what was waved through.
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(allow_stub());
        let mut out = Vec::new();
        let results = agent.execute_tools(&[tool_use_block("t1", "guarded_write")], &mut out);
        assert_eq!(
            expect_tool_result(&results[0]),
            ("t1", "wrote something", false)
        );
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("[auto-approved: guarded_write]"), "got: {out}");
    }

    #[test]
    fn execute_tools_judge_allow_runs_and_notices_the_reason() {
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(judged_stub("ALLOW\nroutine write"));
        let mut out = Vec::new();
        let results = agent.execute_tools(&[tool_use_block("t1", "guarded_write")], &mut out);
        assert_eq!(
            expect_tool_result(&results[0]),
            ("t1", "wrote something", false)
        );
        let out = String::from_utf8(out).unwrap();
        assert!(
            out.contains("[judge allowed: guarded_write — routine write]"),
            "got: {out}"
        );
    }

    #[test]
    fn execute_tools_judge_deny_reports_reason_and_releases_budget() {
        let mut agent = agent_with_tools(vec![Box::new(
            TestTool::new("guarded_write", "wrote something")
                .confirmed()
                .with_cost(4),
        )]);
        agent.set_confirm_policy(judged_stub("DENY\ntoo destructive"));
        let mut out = Vec::new();
        let results = agent.execute_tools(&[tool_use_block("t1", "guarded_write")], &mut out);
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "guarded_write denied by judge: too destructive");
        assert!(is_error);
        // The reservation is released, exactly like a human "no" — the
        // budget only bounds calls that run.
        assert_eq!(agent.budget.count(4), 0);
        // The operator's transcript carries the recorded reason, not just
        // the model's tool_result.
        let out = String::from_utf8(out).unwrap();
        assert!(
            out.contains("[denied: guarded_write — denied by judge: too destructive]"),
            "got: {out}"
        );
    }

    #[test]
    fn execute_tools_breaker_short_circuits_the_batch() {
        // Three consecutive automated denials trip the breaker; the fourth
        // call is rejected without another (paid) adjudication.
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(judged_stub("DENY\nno"));
        let blocks: Vec<Block> = (0..4)
            .map(|i| tool_use_block(&format!("t{i}"), "guarded_write"))
            .collect();
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        for result in &results[..3] {
            let (_, content, _) = expect_tool_result(result);
            assert!(content.contains("denied by judge"), "got: {content}");
        }
        let (_, content, is_error) = expect_tool_result(&results[3]);
        assert_eq!(
            content,
            "guarded_write rejected: confirmation circuit breaker tripped"
        );
        assert!(is_error);
        assert_eq!(agent.auto_denials, 3);
    }

    #[test]
    fn execute_tools_approval_resets_the_breaker_count() {
        // DENY, DENY, ALLOW, DENY, DENY: never three *consecutive* automated
        // denials, so all five calls are adjudicated and no breaker trips.
        let verdicts = ["DENY\na", "DENY\nb", "ALLOW", "DENY\nc", "DENY\nd"];
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(judged_policy({
            let calls = Arc::clone(&calls);
            move || {
                let verdict = verdicts[calls.fetch_add(1, Ordering::Relaxed)];
                Box::new(crate::testing::ThreadSafeProvider::echo().with_send_text(verdict))
            }
        }));
        // Each call carries a distinct input so the verdict cache treats them
        // as five separate adjudications — the scenario under test is a run of
        // *different* operations, not one operation replayed.
        let blocks: Vec<Block> = (0..5)
            .map(|i| Block::ToolUse {
                id: format!("t{i}"),
                name: "guarded_write".to_string(),
                input: serde_json::json!({ "n": i }),
            })
            .collect();
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        assert_eq!(calls.load(Ordering::Relaxed), 5, "all five adjudicated");
        assert!(
            expect_tool_result(&results[2])
                .1
                .contains("wrote something")
        );
        assert!(
            expect_tool_result(&results[4])
                .1
                .contains("denied by judge")
        );
        assert_eq!(agent.auto_denials, 2);
    }

    #[test]
    fn execute_tools_human_denials_never_trip_the_breaker() {
        // Four human "no"s in one batch: someone is present and answering,
        // so every call still reaches the prompt and the count stays zero.
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(ask_stub(|_| false));
        let blocks: Vec<Block> = (0..4)
            .map(|i| tool_use_block(&format!("t{i}"), "guarded_write"))
            .collect();
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        for result in &results {
            let (_, content, _) = expect_tool_result(result);
            assert!(content.contains("denied by user"), "got: {content}");
        }
        assert_eq!(agent.auto_denials, 0);
    }

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

    #[test]
    fn judge_payload_carries_the_turns_initiating_request() {
        // The judge weighs the proposed call against the intent that started
        // the turn — the typed prompt, quoted with trusted provenance.
        let log: Arc<std::sync::Mutex<Vec<TurnRequest>>> = Arc::default();
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![
                tool_use_stream("t1", "guarded_write"),
                text_stream("done"),
            ])),
            test_config(None),
            vec![confirmed_tool()],
        );
        agent.set_confirm_policy(judged_policy({
            let log = Arc::clone(&log);
            move || {
                Box::new(
                    crate::testing::ThreadSafeProvider::echo()
                        .with_send_text("ALLOW")
                        .with_send_log(Arc::clone(&log)),
                )
            }
        }));
        agent
            .run("clean the build tree", &mut std::io::sink())
            .unwrap();
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 1);
        let payload = crate::testing::expect_text(&log[0].messages[0].content[0]);
        let payload: serde_json::Value = serde_json::from_str(payload).unwrap();
        assert_eq!(payload["initiating_request"], "clean the build tree");
        assert_eq!(payload["proposed_call"]["tool"], "guarded_write");
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

    /// A canned stream that reports `input` up front, emits `text`, and ends
    /// the turn reporting `output` — the Anthropic-shaped usage split the
    /// usage-line tests drive.
    fn measured_text_stream(text: &str, input: u32, output: u32) -> Vec<StreamDelta> {
        vec![
            StreamDelta::MessageStart {
                usage: Usage {
                    input_tokens: input,
                    output_tokens: 0,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            },
            StreamDelta::TextStart {
                index: 0,
                text: text.to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: output,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            },
        ]
    }

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
    fn process_stream_styled_error_mid_line_closes_the_partial_line() {
        // Text streams ahead of its newline, so a stream that dies mid-line
        // leaves a half-printed, styled line; the abort must reset the style
        // and newline so whatever prints next starts clean.
        let mut agent = agent_with_tools(vec![]);
        agent.set_styled(true);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "## Par".to_string(),
            }),
            Err(ApiError::Stream("connection reset".to_string())),
        ];

        let mut out = Vec::new();
        let result = agent.process_stream(&mut events.into_iter(), &mut out);

        assert!(matches!(result, Err(AgentError::Api(_))));
        assert_eq!(String::from_utf8(out).unwrap(), "\x1b[1;4mPar\x1b[0m\n");
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

    fn tool_use_msg(id: &str, name: &str) -> TurnMessage {
        TurnMessage {
            role: Role::Assistant,
            content: vec![Block::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input: serde_json::json!({}),
            }],
        }
    }

    fn tool_result_msg(id: &str, content: &str) -> TurnMessage {
        TurnMessage {
            role: Role::User,
            content: vec![Block::ToolResult {
                tool_use_id: id.to_string(),
                content: content.to_string(),
                is_error: false,
            }],
        }
    }

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

    #[test]
    fn process_stream_cancelled_between_events_stops_consuming() {
        // The stream raises the flag while yielding its third event — the
        // between-events check fires before that event is processed, so the
        // partial text stays partial and the tail is never pulled.
        let agent = agent_with_tools(vec![]);
        let flag = Arc::clone(&agent.cancel);
        let pulled = Rc::new(Cell::new(0usize));
        let counter = Rc::clone(&pulled);
        let events = vec![
            message_start(),
            StreamDelta::TextStart {
                index: 0,
                text: "par".to_string(),
            },
            StreamDelta::TextDelta {
                index: 0,
                text: "tial".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            },
        ];
        let mut stream = events.into_iter().map(move |e| {
            counter.set(counter.get() + 1);
            if counter.get() == 3 {
                flag.store(true, Ordering::Relaxed);
            }
            Ok(e)
        });

        let mut out = Vec::new();
        let result = agent.process_stream(&mut stream, &mut out);
        assert!(matches!(result, Err(AgentError::Cancelled)));
        assert_eq!(String::from_utf8(out).unwrap(), "par");
        assert_eq!(pulled.get(), 3); // the fourth event was never pulled
    }

    #[test]
    fn execute_tools_cancelled_at_the_confirmation_gate_runs_nothing() {
        // A Ctrl-C while the gate waits on the operator: the answer no
        // longer matters — the already-approved call is not run, and the
        // sibling is resolved in pre-flight without ever being prompted.
        let flag = Arc::new(AtomicBool::new(false));
        let mut agent = agent_with_tools(vec![
            Box::new(TestTool::new("t1", "ran-1").confirmed()),
            Box::new(TestTool::new("t2", "ran-2").confirmed()),
        ]);
        agent.set_cancel_flag(Arc::clone(&flag));
        let prompts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = Arc::clone(&prompts);
        agent.set_confirm_policy(ask_stub(move |_summary| {
            count.fetch_add(1, Ordering::Relaxed);
            flag.store(true, Ordering::Relaxed);
            true
        }));

        let blocks = vec![tool_use_block("1", "t1"), tool_use_block("2", "t2")];
        let mut out = Vec::new();
        let results = agent.execute_tools(&blocks, &mut out);
        assert_eq!(prompts.load(Ordering::Relaxed), 1); // t2 was never prompted
        assert_eq!(
            expect_tool_result(&results[0]),
            ("1", "cancelled by user", true)
        );
        assert_eq!(
            expect_tool_result(&results[1]),
            ("2", "t2 cancelled by user", true)
        );
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("[cancelled: t2]"));
    }

    // ── context compaction ──

    #[test]
    fn compaction_cut_empty_history_is_none() {
        assert!(compaction_cut(&[], KEEP_RECENT_TURN_GROUPS).is_none());
    }

    #[test]
    fn compaction_cut_single_group_is_none() {
        let msgs = vec![user_msg("q0"), assistant_msg("a0")];
        assert!(compaction_cut(&msgs, KEEP_RECENT_TURN_GROUPS).is_none());
    }

    #[test]
    fn compaction_cut_exactly_keep_window_is_none() {
        // Two groups fill the keep window on the nose — nothing older exists.
        let msgs = vec![
            user_msg("q0"),
            assistant_msg("a0"),
            user_msg("q1"),
            assistant_msg("a1"),
        ];
        assert!(compaction_cut(&msgs, KEEP_RECENT_TURN_GROUPS).is_none());
    }

    #[test]
    fn compaction_cut_drops_groups_beyond_keep_window() {
        // Four fully-summarizable groups, keep 2 → the cut lands at the third
        // group's start, dropping the first two whole.
        let mut msgs = Vec::new();
        for i in 0..4 {
            msgs.push(user_msg(&format!("q{i}")));
            msgs.push(assistant_msg(&format!("a{i}")));
        }
        assert_eq!(compaction_cut(&msgs, KEEP_RECENT_TURN_GROUPS), Some(4));
    }

    #[test]
    fn compaction_cut_never_splits_a_tool_loop() {
        // Group 0 contains a tool loop. Its tool_result message (index 2) is
        // not a boundary — only the fresh user turns at 0, 4, and 6 are — so
        // the cut keeps the loop intact and lands on group 1's start.
        let msgs = vec![
            user_msg("q0"),
            tool_use_msg("t1", "echo"),
            tool_result_msg("t1", "ok"),
            assistant_msg("a0"),
            user_msg("q1"),
            assistant_msg("a1"),
            user_msg("q2"),
            assistant_msg("a2"),
        ];
        assert_eq!(compaction_cut(&msgs, KEEP_RECENT_TURN_GROUPS), Some(4));
    }

    #[test]
    fn compaction_cut_skips_coalesced_turn_start() {
        // The turn asking "q2" starts mid-message — its text was coalesced
        // into the trailing tool_result message at index 4 (see `run`'s
        // coalescing). That message is not a safe boundary: were it one, the
        // cut would land there (index 4) and orphan the tool_result from its
        // tool_use; instead the boundaries are 0, 2, and 6, cutting at 2.
        let msgs = vec![
            user_msg("q0"),
            assistant_msg("a0"),
            user_msg("q1"),
            tool_use_msg("t1", "writer"),
            TurnMessage {
                role: Role::User,
                content: vec![
                    Block::ToolResult {
                        tool_use_id: "t1".to_string(),
                        content: "wrote".to_string(),
                        is_error: false,
                    },
                    Block::Text("q2".to_string()),
                ],
            },
            assistant_msg("a2"),
            user_msg("q3"),
            assistant_msg("a3"),
        ];
        assert_eq!(compaction_cut(&msgs, KEEP_RECENT_TURN_GROUPS), Some(2));
    }

    #[test]
    fn compaction_summarizes_away_a_thinking_prefix() {
        // A thinking block in the droppable prefix is summarized away like
        // everything else: it rides into the summary sub-call verbatim, and
        // no signature survives compaction — only live tool-use turns resend
        // one, and the dropped messages no longer exist to be resent.
        let provider = MockProvider::new(vec![text_stream("next")]).with_send_text("S1");
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.messages[1].content.insert(
            0,
            Block::Thinking {
                text: "old reasoning".to_string(),
                signature: "old_sig".to_string(),
            },
        );
        agent.last_input_tokens = 7500;

        agent.run("q3", &mut std::io::sink()).unwrap();

        // The sub-call carried the thinking block inside the dropped prefix…
        let log = log.borrow();
        assert_eq!(
            log[0].messages[1].content[0],
            Block::Thinking {
                text: "old reasoning".to_string(),
                signature: "old_sig".to_string(),
            }
        );
        // …and after compaction neither the block nor its signature remains.
        assert_eq!(agent.compacted_summary.as_deref(), Some("S1"));
        assert!(agent.messages.iter().all(|m| {
            m.content
                .iter()
                .all(|b| !matches!(b, Block::Thinking { .. }))
        }));
    }

    #[test]
    fn run_compacts_at_threshold_and_folds_summary_into_system() {
        // 7,500 measured tokens against a 10,000-token limit sits exactly on the
        // 75% threshold — the guard fires at >=, not >.
        let provider = MockProvider::new(vec![text_stream("next")]).with_send_text("S1");
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.last_input_tokens = 7500;

        let mut out = Vec::new();
        agent.run("q3", &mut out).unwrap();

        // The summary sub-call carried the dropped prefix (group 0) plus the
        // summarize instruction, under the dedicated system prompt.
        let log = log.borrow();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].system.as_deref(), Some(SUMMARIZE_SYSTEM));
        // The sub-call runs on its own output budget, not the operator's
        // reply cap — a small max_tokens must not truncate the summary and
        // wedge compaction into a permanent fail-fast loop.
        assert_eq!(log[0].max_tokens, SUMMARIZE_MAX_TOKENS);
        assert_eq!(log[0].messages.len(), 3);
        assert_eq!(log[0].messages[0], user_msg("q0"));
        assert_eq!(log[0].messages[1], assistant_msg("a0"));
        assert_eq!(
            log[0].messages[2],
            user_msg("Summarize the messages above.")
        );

        // History: the two kept groups plus this turn's exchange; the guard
        // resets to blind until the next measured response.
        assert_eq!(agent.messages.len(), 6);
        assert_eq!(agent.messages[0], user_msg("q1"));
        assert_eq!(agent.compacted_summary.as_deref(), Some("S1"));
        assert_eq!(agent.last_input_tokens, 0);
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("[compacted 2 messages into summary]")
        );

        // The next request folds the summary into the (otherwise absent)
        // system prompt.
        assert_eq!(
            agent.build_request().system.as_deref(),
            Some("## Earlier conversation (summarized)\nS1")
        );
    }

    #[test]
    fn compaction_request_carries_no_effort_even_when_configured() {
        // Summarization is routine work: it runs at the model default, never
        // the configured conversation effort. The sub-call's
        // recorded request must carry `effort: None` despite the agent's set.
        let provider = MockProvider::new(vec![text_stream("next")]).with_send_text("S1");
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.set_effort(Some("high".to_string()));
        agent.last_input_tokens = 7500;

        agent.run("q3", &mut std::io::sink()).unwrap();

        let log = log.borrow();
        assert_eq!(log.len(), 1);
        assert!(log[0].effort.is_none());
        // The conversation effort is untouched — only the summary sub-call
        // opts out, so subsequent turns still carry it.
        assert_eq!(agent.effort(), Some("high"));
    }

    #[test]
    fn run_below_threshold_does_not_compact() {
        let provider = MockProvider::new(vec![text_stream("next")]).with_send_text("S1");
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.last_input_tokens = 7499; // one below the 75% threshold

        agent.run("q3", &mut std::io::sink()).unwrap();

        assert!(log.borrow().is_empty());
        assert!(agent.compacted_summary.is_none());
        assert_eq!(agent.messages.len(), 8); // nothing dropped
    }

    #[test]
    fn run_over_threshold_without_droppable_prefix_skips_compaction() {
        // Only the keep-window's worth of groups exists: the guard fires but
        // nothing is safe to drop, so the turn proceeds uncompacted and the
        // oversized context rides until enough turns accumulate.
        let provider = MockProvider::new(vec![text_stream("next")]);
        let log = provider.send_log();
        let mut agent = Agent::new(Box::new(provider), test_config(None), vec![]);
        agent.config.context_token_limit = 10000;
        agent.messages.push(user_msg("q0"));
        agent.messages.push(assistant_msg("a0"));
        agent.messages.push(user_msg("q1"));
        agent.messages.push(assistant_msg("a1"));
        agent.last_input_tokens = 9000;

        agent.run("q2", &mut std::io::sink()).unwrap();

        assert!(log.borrow().is_empty());
        assert!(agent.compacted_summary.is_none());
        assert_eq!(agent.messages.len(), 6);
    }

    #[test]
    fn run_second_compaction_folds_prior_summary() {
        // Cumulative summarizing: the second compaction drops turn-groups
        // whose predecessors are already gone, so its sub-call must carry the
        // first summary — overwriting with a summary of only the new prefix
        // would silently forget the oldest context.
        let provider =
            MockProvider::new(vec![text_stream("r1"), text_stream("r2")]).with_send_text("S");
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);

        agent.last_input_tokens = 8000;
        agent.run("q3", &mut std::io::sink()).unwrap(); // first compaction
        agent.last_input_tokens = 8000; // re-arm: the next turn measured over
        agent.run("q4", &mut std::io::sink()).unwrap(); // second compaction

        let log = log.borrow();
        assert_eq!(log.len(), 2);
        // The second sub-call drops the now-oldest prefix (q1/a1)…
        assert_eq!(log[1].messages.len(), 3);
        assert_eq!(log[1].messages[0], user_msg("q1"));
        // …and its instruction folds in the summary from the first pass.
        let instruction = expect_text(&log[1].messages[2].content[0]);
        assert!(instruction.contains("already summarized"));
        assert!(instruction.contains("S"), "prior summary missing");
        assert_eq!(agent.compacted_summary.as_deref(), Some("S"));
    }

    #[test]
    fn run_summary_failure_fails_fast_without_consuming_input() {
        // Settled decision: a failed summary sub-call surfaces as an API
        // error rather than silently sending the known-over-limit request.
        // It fails before the new input is recorded, so nothing changes.
        let mut agent = compaction_agent(Box::new(ErrProvider), 10000);
        agent.last_input_tokens = 8000;

        let result = agent.run("q3", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(_))));
        assert_eq!(agent.messages.len(), 6); // all three groups intact, no "q3"
        assert!(agent.compacted_summary.is_none());
        assert_eq!(agent.last_input_tokens, 8000);
    }

    #[test]
    fn run_summary_transient_failure_retries_once_and_compacts() {
        // One transient blip must not abort the user's turn: the summary
        // sub-call is retried once in place, and the second attempt
        // compacts and lets the turn proceed as if nothing failed.
        let provider = MockProvider::new(vec![text_stream("next")])
            .with_send_text("S1")
            .with_send_failures(1);
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.last_input_tokens = 8000;

        agent.run("q3", &mut std::io::sink()).unwrap();

        assert_eq!(log.borrow().len(), 2); // the failed attempt plus the retry
        assert_eq!(agent.compacted_summary.as_deref(), Some("S1"));
        assert_eq!(agent.messages.len(), 6); // group 0 dropped, q3 exchange on
    }

    #[test]
    fn run_summary_failing_twice_surfaces_the_error() {
        // The retry is bounded at one: two consecutive failures surface the
        // existing error unchanged — no third attempt is made even though
        // this provider would have succeeded on it — and nothing is drained
        // or recorded.
        let provider = MockProvider::new(vec![])
            .with_send_text("S1")
            .with_send_failures(2);
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.last_input_tokens = 8000;

        let result = agent.run("q3", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(ApiError::Io(_)))));
        assert_eq!(log.borrow().len(), 2); // exactly one retry
        assert_eq!(agent.messages.len(), 6); // intact, no "q3"
        assert!(agent.compacted_summary.is_none());
        assert_eq!(agent.last_input_tokens, 8000);
    }

    #[test]
    fn run_truncated_summary_fails_fast_without_draining() {
        // A summary cut off at max_tokens (or diverted into a tool call) is
        // not a summary: committing it and draining would permanently lose
        // the dropped context. Same fail-fast as a transport failure —
        // including the one retry, which this shape also gets.
        let provider = MockProvider::new(vec![]).with_send_stop_reason(StopReason::MaxTokens);
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.last_input_tokens = 8000;

        let result = agent.run("q3", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(ApiError::Stream(_)))));
        assert_eq!(log.borrow().len(), 2); // the retry covers this shape too
        assert_eq!(agent.messages.len(), 6); // nothing drained, no "q3"
        assert!(agent.compacted_summary.is_none());
    }

    #[test]
    fn run_empty_summary_fails_fast_without_draining() {
        // A clean EndTurn that carries no usable text is equally not a
        // summary — an empty rolling summary would silently forget the
        // dropped context behind a bare header.
        let provider = MockProvider::new(vec![]).with_send_text("");
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.last_input_tokens = 8000;

        let result = agent.run("q3", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(ApiError::Stream(_)))));
        assert_eq!(agent.messages.len(), 6);
        assert!(agent.compacted_summary.is_none());
    }

    #[test]
    fn run_rollback_after_compaction_keeps_compacted_history() {
        // Compaction succeeds, then the main stream errors before any side
        // effect. Compaction is a retained state change — the rollback
        // snapshot is taken after it — so the failed turn rolls back to the
        // *compacted* history, and the summary keeps only the text block of
        // the sub-call's response.
        let mut agent = compaction_agent(Box::new(SucceedThenErrProvider::new(vec![])), 10000);
        agent.last_input_tokens = 8000;

        let result = agent.run("q3", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(_))));
        assert_eq!(
            agent.messages,
            vec![
                user_msg("q1"),
                assistant_msg("a1"),
                user_msg("q2"),
                assistant_msg("a2"),
            ]
        );
        assert_eq!(agent.compacted_summary.as_deref(), Some("prefix summary"));
        // The measurement rolls back to the post-compaction reset, not to 80.
        assert_eq!(agent.last_input_tokens, 0);
    }

    #[test]
    fn build_request_folds_summary_after_configured_system() {
        let mut agent = agent_with_system("You are helpful.");
        agent.compacted_summary = Some("old stuff".to_string());
        agent.messages.push(user_msg("hi"));

        let system = agent.build_request().system.unwrap();

        assert!(system.starts_with("You are helpful."));
        assert!(system.ends_with("## Earlier conversation (summarized)\nold stuff"));
    }

    #[test]
    fn compaction_recovers_a_prospective_request_at_the_actual_input_budget() {
        for (groups, reply_bytes, max_tokens, incoming_bytes) in
            [(2, 4000, 64, 8), (3, 1800, 5000, 8), (3, 2100, 64, 2000)]
        {
            let provider = MockProvider::new(vec![text_stream("done")]).with_send_text("summary");
            let summaries = provider.send_log();
            let streams = provider.stream_log();
            let mut config = test_config(None);
            config.context_token_limit = 10_000;
            config.max_tokens = max_tokens;
            let mut agent = Agent::new(Box::new(provider), config, vec![]);
            for i in 0..groups {
                agent.messages.push(user_msg(&format!("q{i}")));
                agent.messages.push(assistant_msg(&"a".repeat(reply_bytes)));
            }
            agent.last_input_tokens = 500;
            agent
                .run(&"q".repeat(incoming_bytes), &mut Vec::new())
                .unwrap();
            assert_eq!(summaries.borrow().len(), 1);
            assert_eq!(streams.borrow().len(), 1);
            let streams = streams.borrow();
            assert!(context::input_size(&streams[0]) <= context::input_budget(&streams[0], 10_000));
            assert_eq!(agent.compacted_summary.as_deref(), Some("summary"));
        }
    }

    #[test]
    fn oversized_summary_does_not_discard_original_history() {
        let provider = MockProvider::new(vec![]).with_send_text(&"s".repeat(10_000));
        let log = provider.stream_log();
        let mut agent = compaction_agent(Box::new(provider), 10_000);
        agent.last_input_tokens = 8000;
        let before = agent.session();
        assert!(matches!(
            agent.run("next", &mut Vec::new()),
            Err(AgentError::ContextLimit(10_000))
        ));
        assert_eq!(agent.session(), before);
        assert!(log.borrow().is_empty());
    }

    #[test]
    fn tool_loop_checks_large_real_file_before_continuing() {
        use crate::tools::{read_file::ReadFileTool, sandbox::Sandbox};
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("large.txt"), "word ".repeat(13_000)).unwrap();
        let mut first = tool_use_stream("read", "read_file");
        if let StreamDelta::ToolArgsDelta { json, .. } = &mut first[2] {
            *json = serde_json::json!({"path": "large.txt"}).to_string();
        }
        let provider = MockProvider::new(vec![first, text_stream("read a narrower range")]);
        let log = provider.stream_log();
        let mut config = test_config(None);
        config.context_token_limit = 8000;
        let mut agent = Agent::new(
            Box::new(provider),
            config,
            vec![Box::new(ReadFileTool::new(
                Sandbox::rooted(directory.path().into()).unwrap(),
            ))],
        );
        let mut output = Vec::new();
        agent.run("read large.txt", &mut output).unwrap();
        let log = log.borrow();
        assert_eq!(log.len(), 2);
        assert!(context::input_size(&log[1]) <= 6000);
        let (id, content, error) = expect_tool_result(&log[1].messages[2].content[0]);
        assert_eq!(id, "read");
        assert!(!error);
        assert!(content.contains("tool output shortened"));
        assert_eq!(expect_tool_use(&log[1].messages[1].content[0]).0, id);
        assert_eq!(
            expect_tool_result(&agent.session().messages[2].content[0])
                .1
                .len(),
            65_000
        );
        assert!(
            String::from_utf8(output)
                .unwrap()
                .contains("shortened 1 tool results in outgoing request")
        );
    }

    #[test]
    fn oversized_first_input_fails_before_network_and_rolls_back() {
        let provider = MockProvider::new(vec![]);
        let log = provider.stream_log();
        let mut agent = Agent::new(Box::new(provider), test_config(None), vec![]);
        let error = agent
            .run(&"x".repeat(100_000), &mut Vec::new())
            .unwrap_err();
        assert!(matches!(error, AgentError::ContextLimit(100_000)));
        assert!(error.to_string().contains("smaller read_file ranges"));
        assert!(log.borrow().is_empty());
        assert!(agent.session().messages.is_empty());
    }

    #[test]
    fn context_failure_preserves_tool_pairs_after_mutation_only() {
        for mutates in [false, true] {
            let first = text_and_tool_use_stream("call", "tool", &"x".repeat(8000));
            let provider = MockProvider::new(vec![first]);
            let log = provider.stream_log();
            let tool = if mutates {
                TestTool::new("tool", "changed").mutating()
            } else {
                TestTool::new("tool", "read")
            };
            let mut config = test_config(None);
            config.context_token_limit = 8000;
            let mut agent = Agent::new(Box::new(provider), config, vec![Box::new(tool)]);
            assert!(matches!(
                agent.run("work", &mut Vec::new()),
                Err(AgentError::ContextLimit(8000))
            ));
            assert_eq!(log.borrow().len(), 1);
            if mutates {
                let session = agent.session();
                assert_eq!(session.messages.len(), 3);
                assert_eq!(
                    expect_tool_result(&session.messages[2].content[0]).0,
                    "call"
                );
            } else {
                assert!(agent.session().messages.is_empty());
            }
        }
    }

    // ── clear ──

    #[test]
    fn clear_resets_message_history() {
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("hi"));
        agent.messages.push(assistant_msg("hello"));
        assert_eq!(agent.messages.len(), 2);

        agent.clear();

        assert!(agent.messages.is_empty());
    }

    #[test]
    fn clear_on_empty_history_is_a_noop() {
        let mut agent = agent_with_tools(vec![]);
        agent.clear();
        assert!(agent.messages.is_empty());
    }

    #[test]
    fn clear_preserves_tool_budget() {
        // Budgets bound cost across the whole process, not per conversation —
        // clearing the history must not refill a spent tier count.
        let mut agent = agent_with_tools(vec![]);
        agent.budget.set_count(4, 5);

        agent.clear();

        assert_eq!(agent.budget.count(4), 5);
    }

    #[test]
    fn clear_resets_measured_prompt_size() {
        // The measurement describes the conversation it was taken from; a
        // fresh context must not inherit it, or the compaction guard would
        // fire off a context that no longer exists.
        let mut agent = agent_with_tools(vec![]);
        agent.last_input_tokens = 42;

        agent.clear();

        assert_eq!(agent.last_input_tokens, 0);
    }

    #[test]
    fn clear_resets_compacted_summary() {
        // Load-bearing: build_request folds the summary into the system
        // prompt, so a stale one would leak the previous conversation into
        // the fresh one.
        let mut agent = agent_with_tools(vec![]);
        agent.compacted_summary = Some("stale".to_string());

        agent.clear();

        assert!(agent.compacted_summary.is_none());
    }

    // ── session / restore ──

    #[test]
    fn session_snapshots_conversation_state() {
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("hi"));
        agent.messages.push(assistant_msg("hello"));
        agent.compacted_summary = Some("earlier".to_string());
        agent.last_input_tokens = 42;

        let session = agent.session();

        assert_eq!(session.version, crate::session::SESSION_VERSION);
        assert_eq!(
            session.messages,
            vec![user_msg("hi"), assistant_msg("hello")]
        );
        assert_eq!(session.compacted_summary.as_deref(), Some("earlier"));
        assert_eq!(session.last_input_tokens, 42);
        // A snapshot, not a drain: the agent's own state is untouched.
        assert_eq!(agent.messages.len(), 2);
    }

    #[test]
    fn restore_round_trips_a_snapshot_into_a_fresh_agent() {
        let mut donor = agent_with_tools(vec![]);
        donor.messages.push(user_msg("hi"));
        donor.messages.push(assistant_msg("hello"));
        donor.compacted_summary = Some("earlier".to_string());
        donor.last_input_tokens = 42;

        let mut agent = agent_with_tools(vec![]);
        agent.restore(donor.session());

        assert_eq!(agent.session(), donor.session());
    }

    #[test]
    fn restore_replaces_the_current_conversation() {
        // Restoring over live state must not merge the two conversations.
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("stale"));
        agent.compacted_summary = Some("stale summary".to_string());
        agent.last_input_tokens = 9;

        agent.restore(Session {
            version: crate::session::SESSION_VERSION,
            messages: vec![user_msg("restored")],
            compacted_summary: None,
            last_input_tokens: 0,
        });

        assert_eq!(agent.messages, vec![user_msg("restored")]);
        assert!(agent.compacted_summary.is_none());
        assert_eq!(agent.last_input_tokens, 0);
    }

    #[test]
    fn restore_strips_a_trailing_orphan_tool_use_keeping_partial_text() {
        // A pre-fix session persisted a trailing assistant message whose
        // tool_use was orphaned by a max_tokens cut-off. Restore drops the
        // unpaired tool_use but keeps the partial text, so the first
        // post-restore request is not 400ed.
        let mut agent = agent_with_tools(vec![]);
        agent.restore(Session {
            version: crate::session::SESSION_VERSION,
            messages: vec![
                user_msg("hi"),
                TurnMessage {
                    role: Role::Assistant,
                    content: vec![
                        Block::Text("partial".to_string()),
                        Block::ToolUse {
                            id: "call_1".to_string(),
                            name: "echo".to_string(),
                            input: serde_json::json!({}),
                        },
                    ],
                },
            ],
            compacted_summary: None,
            last_input_tokens: 0,
        });

        assert_eq!(
            agent.messages,
            vec![user_msg("hi"), assistant_msg("partial")]
        );
    }

    #[test]
    fn restore_drops_a_trailing_assistant_left_empty_by_the_orphan_strip() {
        // The trailing assistant message is a bare orphaned tool_use with no
        // text: stripping it leaves the message empty, so the whole message is
        // dropped rather than committing an empty (wire-rejected) content array.
        let mut agent = agent_with_tools(vec![]);
        agent.restore(Session {
            version: crate::session::SESSION_VERSION,
            messages: vec![
                user_msg("hi"),
                TurnMessage {
                    role: Role::Assistant,
                    content: vec![Block::ToolUse {
                        id: "call_1".to_string(),
                        name: "echo".to_string(),
                        input: serde_json::json!({}),
                    }],
                },
            ],
            compacted_summary: None,
            last_input_tokens: 0,
        });

        assert_eq!(agent.messages, vec![user_msg("hi")]);
    }

    #[test]
    fn restore_preserves_tool_budget() {
        // Same contract as clear: budgets are process state, not
        // conversation state, so a restore must not refill a spent tier.
        let mut agent = agent_with_tools(vec![]);
        agent.budget.set_count(4, 5);

        agent.restore(Session {
            version: crate::session::SESSION_VERSION,
            messages: vec![],
            compacted_summary: None,
            last_input_tokens: 0,
        });

        assert_eq!(agent.budget.count(4), 5);
    }

    #[test]
    fn restore_continues_the_conversation_with_summary_and_armed_guard() {
        // The acceptance shape for persistence: a restored session's next
        // turn runs over the restored history and summary, and the restored
        // measurement arms the compaction guard from turn one — 75 measured
        // tokens against a 100-token limit fires the summarizing sub-call
        // immediately, exactly as if the process had never restarted.
        let provider = MockProvider::new(vec![text_stream("next")]).with_send_text("S2");
        let log = provider.send_log();
        let mut agent = Agent::new(Box::new(provider), test_config(None), vec![]);
        agent.config.context_token_limit = 10000;

        agent.restore(Session {
            version: crate::session::SESSION_VERSION,
            messages: (0..3)
                .flat_map(|i| [user_msg(&format!("q{i}")), assistant_msg(&format!("a{i}"))])
                .collect(),
            compacted_summary: Some("S1".to_string()),
            last_input_tokens: 7500,
        });

        agent.run("q3", &mut std::io::sink()).unwrap();

        // The guard fired on the first post-restore turn; its sub-call
        // summarized the restored oldest turn-group and folded the restored
        // summary into the instruction.
        let log = log.borrow();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].messages[0], user_msg("q0"));
        let instruction = expect_text(&log[0].messages.last().unwrap().content[0]);
        assert!(instruction.contains("S1"), "got: {instruction}");
        assert_eq!(agent.compacted_summary.as_deref(), Some("S2"));
        // The turn itself ran over the restored (now compacted) history.
        assert_eq!(agent.messages[0], user_msg("q1"));
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

    #[test]
    fn execute_tools_prints_status_when_provided() {
        let tool = TestTool::new("status_tool", "done")
            .with_status(|_| Some("doing something".to_string()));
        let mut agent = agent_with_tools(vec![Box::new(tool)]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "status_tool".to_string(),
            input: serde_json::json!({}),
        }];

        let mut out = Vec::new();
        let results = agent.execute_tools(&blocks, &mut out);
        assert_eq!(expect_tool_result(&results[0]).1, "done");
        // Both the pre-run status line and the result line reach the writer.
        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("[doing something]"));
        assert!(printed.contains("[result: done]"));
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

    // ── confirm_summary ──

    #[test]
    fn confirm_summary_includes_tool_status() {
        assert_eq!(
            confirm_summary("shell", Some("$ cargo test")),
            "shell: $ cargo test"
        );
        assert_eq!(
            confirm_summary("write_file", Some("writing src/main.rs")),
            "write_file: writing src/main.rs"
        );
    }

    #[test]
    fn confirm_summary_falls_back_to_name() {
        assert_eq!(confirm_summary("mystery", None), "mystery");
    }

    #[test]
    fn confirm_summary_sanitizes_control_characters() {
        assert_eq!(
            confirm_summary("shell", Some("$ echo hi\x1b[31m")),
            "shell: $ echo hi?[31m"
        );
    }

    #[test]
    fn confirm_summary_strips_invisible_unicode() {
        // U+202E (right-to-left override) could visually reverse the prompt.
        assert_eq!(
            confirm_summary("shell", Some("$ echo \u{202E}fr- mr\u{202C}")),
            "shell: $ echo ?fr- mr?"
        );
    }

    #[test]
    fn confirm_summary_caps_length() {
        let long = "x".repeat(500);
        let summary = confirm_summary("shell", Some(&long));
        assert_eq!(summary.chars().count(), CONFIRM_SUMMARY_MAX_CHARS + 1);
        assert!(summary.ends_with('…'));
    }

    #[test]
    fn confirm_summary_at_cap_unchanged() {
        // name "shell" + ": " + 193 chars = exactly 200 — no ellipsis.
        let status = "x".repeat(CONFIRM_SUMMARY_MAX_CHARS - 7);
        let summary = confirm_summary("shell", Some(&status));
        assert_eq!(summary.chars().count(), CONFIRM_SUMMARY_MAX_CHARS);
        assert!(!summary.ends_with('…'));
    }

    #[test]
    fn execute_tools_confirmation_summary_uses_tool_status() {
        use std::sync::{Arc, Mutex};

        let shell_like = TestTool::new("shell", "ran")
            .confirmed()
            .with_status(|input| input["command"].as_str().map(|c| format!("$ {c}")));
        let mut agent = agent_with_tools(vec![Box::new(shell_like)]);
        let seen = Arc::new(Mutex::new(String::new()));
        let seen_clone = seen.clone();
        agent.set_confirm_policy(ask_stub(move |summary| {
            *seen_clone.lock().unwrap() = summary.to_string();
            true
        }));

        // A decoy `path` field must not reach the prompt — the summary comes
        // from format_status, which reads the field the tool executes.
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "shell".to_string(),
            input: serde_json::json!({"command": "rm -rf src", "path": "README.md"}),
        }];
        agent.execute_tools(&blocks, &mut std::io::sink());
        assert_eq!(*seen.lock().unwrap(), "shell: $ rm -rf src");
    }

    // ── format_size ──

    #[test]
    fn format_size_picks_the_unit() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1023), "1023 B");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(3 * 1024 * 1024), "3.0 MB");
    }

    // ── summarize_output ──

    #[test]
    fn summarize_output_passes_short_results_verbatim() {
        assert_eq!(summarize_output("done"), "done");
        assert_eq!(summarize_output(""), "");
        // A trailing newline is shell-output dressing, not a second line.
        assert_eq!(summarize_output("done\n"), "done");
        // Verbatim still means sanitized: short output is model-visible data.
        assert_eq!(summarize_output("a\x1b[31mb"), "a?[31mb");
    }

    #[test]
    fn summarize_output_collapses_multiline_results() {
        assert_eq!(
            summarize_output("alpha\nbeta\ngamma\n"),
            "3 lines, 17 B — alpha…"
        );
    }

    #[test]
    fn summarize_output_truncates_a_long_single_line() {
        let long = "x".repeat(150);
        let summary = summarize_output(&long);
        assert_eq!(
            summary,
            format!("1 line, 150 B — {}…", "x".repeat(RESULT_PREVIEW_MAX_CHARS))
        );
    }

    #[test]
    fn summarize_output_truncates_preview_on_char_boundaries() {
        // 150 multibyte chars: the preview cap counts chars, not bytes, so
        // this must not split a UTF-8 sequence (a byte-indexed slice would).
        let long = "日".repeat(150);
        let summary = summarize_output(&long);
        assert!(summary.contains(&"日".repeat(RESULT_PREVIEW_MAX_CHARS)));
        assert!(!summary.contains(&"日".repeat(RESULT_PREVIEW_MAX_CHARS + 1)));
    }

    #[test]
    fn summarize_output_sanitizes_the_preview() {
        let sneaky = format!("\x1b[2K[result: fake]\n{}", "padding\n".repeat(5));
        let summary = summarize_output(&sneaky);
        assert!(summary.starts_with("6 lines"));
        assert!(!summary.contains('\x1b'));
    }

    #[test]
    fn execute_tools_summarizes_long_results_but_returns_them_whole() {
        let long = format!("first line\n{}", "body\n".repeat(50));
        let response = long.clone();
        let tool = TestTool::new("reader", "").with_run(move |_| Ok(response.clone()));
        let mut agent = agent_with_tools(vec![Box::new(tool)]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "reader".to_string(),
            input: serde_json::json!({}),
        }];

        let mut out = Vec::new();
        let results = agent.execute_tools(&blocks, &mut out);
        // The model's copy is untouched — only the printed line collapses.
        assert_eq!(expect_tool_result(&results[0]).1, long);
        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("[result: 51 lines, 261 B — first line…]"));
        assert!(!printed.contains("body"));
    }

    #[test]
    fn execute_tools_sanitizes_error_lines() {
        let tool =
            TestTool::new("boom", "").with_run(|_| Err("line one\x1b[31m\nline two".to_string()));
        let mut agent = agent_with_tools(vec![Box::new(tool)]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "boom".to_string(),
            input: serde_json::json!({}),
        }];

        let mut out = Vec::new();
        let results = agent.execute_tools(&blocks, &mut out);
        // The model's copy keeps the raw error; the display copy is scrubbed
        // but complete — errors are read in full, not summarized.
        assert_eq!(
            expect_tool_result(&results[0]).1,
            "line one\x1b[31m\nline two"
        );
        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("[tool error: line one?[31m\nline two]"));
    }
}
