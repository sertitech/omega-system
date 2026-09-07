//! Shared test doubles and assertion helpers. Compiled only into the crate's
//! own test build (`#[cfg(test)]` at the declaration), so nothing here ships.
//! The agent and REPL test suites both drive the loop through these providers;
//! keeping them in one place stops the fixtures from drifting apart — and,
//! under the coverage gate, stops every test from growing its own single-use
//! double whose unexercised methods would count as permanently-dead lines.

use crate::provider::{ApiError, DeltaStream, Provider};
use crate::tools::ToolDef;
use crate::turn::{Block, Role, StopReason, StreamDelta, Turn, TurnMessage, TurnRequest, Usage};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

/// Replays canned streams. Each `stream` call pops the next canned delta
/// sequence; with `repeat`, it replays the front sequence forever (used to
/// drive the turn limit). `send` — the compaction summary path — records each
/// request in `send_log` and returns `send_text` as a completed turn, after
/// failing its first `send_failures` calls (drives the summary retry).
/// `list_models` returns the `with_models` list and counts its calls — the
/// agent's model-cache tests read both.
pub(crate) struct MockProvider {
    streams: RefCell<VecDeque<Vec<StreamDelta>>>,
    repeat: bool,
    send_text: String,
    send_stop: StopReason,
    send_failures: Cell<u32>,
    send_log: Rc<RefCell<Vec<TurnRequest>>>,
    stream_log: Rc<RefCell<Vec<TurnRequest>>>,
    models: Vec<String>,
    list_models_calls: Rc<Cell<usize>>,
}

impl MockProvider {
    pub(crate) fn new(streams: Vec<Vec<StreamDelta>>) -> Self {
        Self {
            streams: RefCell::new(streams.into()),
            repeat: false,
            send_text: "sent".to_string(),
            send_stop: StopReason::EndTurn,
            send_failures: Cell::new(0),
            send_log: Rc::default(),
            stream_log: Rc::default(),
            models: Vec::new(),
            list_models_calls: Rc::default(),
        }
    }
    pub(crate) fn repeating(stream: Vec<StreamDelta>) -> Self {
        Self {
            streams: RefCell::new(VecDeque::from([stream])),
            repeat: true,
            send_text: "sent".to_string(),
            send_stop: StopReason::EndTurn,
            send_failures: Cell::new(0),
            send_log: Rc::default(),
            stream_log: Rc::default(),
            models: Vec::new(),
            list_models_calls: Rc::default(),
        }
    }
    /// Replace the canned `list_models` ids (default empty).
    pub(crate) fn with_models(mut self, models: &[&str]) -> Self {
        self.models = models.iter().map(|m| m.to_string()).collect();
        self
    }
    /// A shared handle to the `list_models` call count, cloned out before the
    /// provider is boxed into the agent — proves the agent's cache fetches
    /// exactly once.
    pub(crate) fn list_models_calls(&self) -> Rc<Cell<usize>> {
        Rc::clone(&self.list_models_calls)
    }
    /// Replace the canned `send` response text (the compaction summary).
    pub(crate) fn with_send_text(mut self, text: &str) -> Self {
        self.send_text = text.to_string();
        self
    }
    /// Replace the canned `send` stop reason (default `EndTurn`) — drives the
    /// truncated-summary rejection path.
    pub(crate) fn with_send_stop_reason(mut self, stop: StopReason) -> Self {
        self.send_stop = stop;
        self
    }
    /// Fail the first `n` `send` calls at the transport layer (default 0)
    /// before the canned response resumes — drives the compaction summary
    /// retry. Failed attempts are still recorded in `send_log`.
    pub(crate) fn with_send_failures(mut self, n: u32) -> Self {
        self.send_failures = Cell::new(n);
        self
    }
    /// Capture the requests actually sent, including tool-loop continuations.
    pub(crate) fn stream_log(&self) -> Rc<RefCell<Vec<TurnRequest>>> {
        Rc::clone(&self.stream_log)
    }
    /// A shared handle to the recorded `send` requests, cloned out before the
    /// provider is boxed into the agent.
    pub(crate) fn send_log(&self) -> Rc<RefCell<Vec<TurnRequest>>> {
        Rc::clone(&self.send_log)
    }
}

impl Provider for MockProvider {
    fn send(&self, request: &TurnRequest) -> Result<Turn, ApiError> {
        self.send_log.borrow_mut().push(request.clone());
        if self.send_failures.get() > 0 {
            self.send_failures.set(self.send_failures.get() - 1);
            return Err(ApiError::Io(std::io::Error::other("transient boom")));
        }
        Ok(Turn {
            blocks: vec![Block::Text(self.send_text.clone())],
            stop_reason: self.send_stop.clone(),
            usage: Usage::default(),
        })
    }
    fn stream(&self, request: &TurnRequest) -> Result<DeltaStream, ApiError> {
        self.stream_log.borrow_mut().push(request.clone());
        let deltas = if self.repeat {
            self.streams
                .borrow()
                .front()
                .expect("repeating mock needs a stream")
                .clone()
        } else {
            self.streams
                .borrow_mut()
                .pop_front()
                .expect("MockProvider: no canned streams left")
        };
        Ok(Box::new(deltas.into_iter().map(Ok)))
    }
    fn list_models(&self) -> Result<Vec<String>, ApiError> {
        self.list_models_calls.set(self.list_models_calls.get() + 1);
        Ok(self.models.clone())
    }
}

/// The assistant text a [`ThreadSafeProvider`] echoes: the last user-message
/// text in the request. A child agent's final reply is then a deterministic
/// function of its prompt, which lets the `task` fan-out tests tell concurrent
/// children apart and pin their result order.
fn last_user_text(request: &TurnRequest) -> String {
    request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .and_then(|m| {
            m.content.iter().rev().find_map(|b| match b {
                Block::Text(text) => Some(text.clone()),
                _ => None,
            })
        })
        .unwrap_or_default()
}

/// A stream that emits `text` as one assistant text block and ends the turn.
fn echo_stream(text: String) -> Vec<StreamDelta> {
    vec![
        StreamDelta::TextStart {
            index: 0,
            text: String::new(),
        },
        StreamDelta::TextDelta { index: 0, text },
        StreamDelta::MessageDelta {
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage::default(),
        },
    ]
}

/// A `Send + Sync` provider double (Mutex-backed, no `Rc`/`Cell`) so it can be
/// built and driven inside the `task` tool's parallel worker threads — the
/// `Rc`-based [`MockProvider`] cannot cross a thread boundary. With no canned
/// streams it *echoes* the prompt (see [`last_user_text`]); with `scripted`
/// streams it replays them in order, falling back to echoing once they run
/// out. `failing` fails every call at the transport layer, driving the child
/// API-error path. Each `stream` request's effort is recorded so a test can
/// assert what the tool passed the child. `send` echoes too unless
/// `with_send_text` cans a reply — the judge fixtures ([`crate::agent`]'s
/// confirm-policy tests) drive verdicts through it, observing the one-shot
/// request via `with_send_log` (the `MockProvider` equivalents are `Rc`-based
/// and cannot live inside a `Send + Sync` [`crate::provider::ProviderFactory`]
/// build closure).
pub(crate) struct ThreadSafeProvider {
    streams: Mutex<VecDeque<Vec<StreamDelta>>>,
    fail: bool,
    efforts: Arc<Mutex<Vec<Option<String>>>>,
    send_text: Option<String>,
    send_stop: StopReason,
    send_log: Arc<Mutex<Vec<TurnRequest>>>,
}

impl ThreadSafeProvider {
    /// Echoes the prompt back as the assistant reply.
    pub(crate) fn echo() -> Self {
        Self {
            streams: Mutex::new(VecDeque::new()),
            fail: false,
            efforts: Arc::default(),
            send_text: None,
            send_stop: StopReason::EndTurn,
            send_log: Arc::default(),
        }
    }
    /// Replays `streams` in order, then echoes once they are exhausted.
    pub(crate) fn scripted(streams: Vec<Vec<StreamDelta>>) -> Self {
        Self {
            streams: Mutex::new(streams.into()),
            ..Self::echo()
        }
    }
    /// Fails every `send`/`stream` at the transport layer.
    pub(crate) fn failing() -> Self {
        Self {
            fail: true,
            ..Self::echo()
        }
    }
    /// Record each streamed request's effort into `log` — a handle the test
    /// holds so it can assert the effort the `task` tool resolved for the child.
    pub(crate) fn with_effort_log(mut self, log: Arc<Mutex<Vec<Option<String>>>>) -> Self {
        self.efforts = log;
        self
    }
    /// Replace the echoed `send` reply with canned text — a judge verdict.
    pub(crate) fn with_send_text(mut self, text: &str) -> Self {
        self.send_text = Some(text.to_string());
        self
    }
    /// Replace the canned `send` stop reason (default `EndTurn`) — drives the
    /// truncated-verdict rejection path.
    pub(crate) fn with_send_stop_reason(mut self, stop: StopReason) -> Self {
        self.send_stop = stop;
        self
    }
    /// Record each `send` request into `log`, so a test can assert what the
    /// judge actually asked (model, effort, payload).
    pub(crate) fn with_send_log(mut self, log: Arc<Mutex<Vec<TurnRequest>>>) -> Self {
        self.send_log = log;
        self
    }
}

impl Provider for ThreadSafeProvider {
    fn send(&self, request: &TurnRequest) -> Result<Turn, ApiError> {
        self.send_log.lock().unwrap().push(request.clone());
        if self.fail {
            return Err(ApiError::Io(std::io::Error::other("boom")));
        }
        let text = self
            .send_text
            .clone()
            .unwrap_or_else(|| last_user_text(request));
        Ok(Turn {
            blocks: vec![Block::Text(text)],
            stop_reason: self.send_stop.clone(),
            usage: Usage::default(),
        })
    }
    fn stream(&self, request: &TurnRequest) -> Result<DeltaStream, ApiError> {
        self.efforts.lock().unwrap().push(request.effort.clone());
        if self.fail {
            return Err(ApiError::Io(std::io::Error::other("boom")));
        }
        let deltas = self
            .streams
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| echo_stream(last_user_text(request)));
        Ok(Box::new(deltas.into_iter().map(Ok)))
    }
}

/// A provider whose calls always fail at the transport layer — including
/// `list_models`, overriding the fail-soft default so the agent's
/// offline-listing path is drivable.
pub(crate) struct ErrProvider;
impl Provider for ErrProvider {
    fn send(&self, _request: &TurnRequest) -> Result<Turn, ApiError> {
        Err(ApiError::Io(std::io::Error::other("boom")))
    }
    fn stream(&self, _request: &TurnRequest) -> Result<DeltaStream, ApiError> {
        Err(ApiError::Io(std::io::Error::other("boom")))
    }
    fn list_models(&self) -> Result<Vec<String>, ApiError> {
        Err(ApiError::Io(std::io::Error::other("boom")))
    }
}

// ── Confirmation-policy fixture seams ──
//
// Shared across the policy fixtures in `agent`, `agent::confirm`, and
// `tools::subagent` for the same reason [`TestTool`] is shared: a bespoke
// never-fired closure per fixture would be a permanently-dead line under the
// coverage gate, while one shared fn is pinned by the tests below.

/// The prompt for automated-mode policy fixtures, where consulting the human
/// is itself the bug the test would catch.
pub(crate) fn no_prompt(_summary: &str) -> bool {
    panic!("prompt must not be consulted")
}

/// An inert provider builder in [`crate::provider::ProviderFactory::from_fns`]
/// shape, for policy fixtures whose judge is never (or not always) reached.
pub(crate) fn stub_provider(
    _kind: crate::provider::ProviderKind,
    _key: String,
) -> Box<dyn Provider> {
    Box::new(ThreadSafeProvider::echo())
}

/// An inert key resolver for the same fixtures.
pub(crate) fn no_key(_env: &str) -> Option<String> {
    None
}

/// [`TestTool`]'s injectable hooks, one alias per [`ToolDef`] method shape.
type RunFn = Box<dyn Fn(serde_json::Value) -> Result<String, String> + Send + Sync>;
type ValidateFn = Box<dyn Fn(&serde_json::Value) -> Result<(), String> + Send + Sync>;
type StatusFn = Box<dyn Fn(&serde_json::Value) -> Option<String> + Send + Sync>;

/// The one configurable [`ToolDef`] double. Behavior is injected per test via
/// the builder methods; unset hooks keep shared defaults that other tests
/// exercise. One shared type (rather than a bespoke impl per test) means every
/// trait method body is executed *somewhere* in the suite — a single-use impl
/// leaves its unused methods as permanently-dead lines under the coverage gate.
///
/// In a test where a hook must **not** fire (a denied confirmation, a rejected
/// validation), rely on the shared default rather than passing a bespoke
/// closure: a closure passed only there would itself never execute.
pub(crate) struct TestTool {
    name: String,
    cost: u8,
    requires_confirmation: bool,
    side_effecting: bool,
    /// Whether the tool draws a fan-out permit. `true` by default (the leaf
    /// behavior); `false` models a dispatching tool like `task`, whose worker
    /// must not hold a permit while a nested agent acquires its own.
    gates_concurrency: bool,
    /// Text `run` writes to its out sink before returning — the live-output
    /// seam. `None` (the default) writes nothing, like most real tools.
    emits: Option<String>,
    run: RunFn,
    validate: ValidateFn,
    format_status: StatusFn,
}

impl TestTool {
    /// A tool named `name` whose `run` returns `Ok(response)`.
    pub(crate) fn new(name: &str, response: &str) -> Self {
        let response = response.to_string();
        Self {
            name: name.to_string(),
            cost: 0,
            requires_confirmation: false,
            side_effecting: false,
            gates_concurrency: true,
            emits: None,
            run: Box::new(move |_| Ok(response.clone())),
            validate: Box::new(|_| Ok(())),
            format_status: Box::new(|_| None),
        }
    }

    /// Override the cost tier (default 0 — never budget-limited).
    pub(crate) fn with_cost(mut self, cost: u8) -> Self {
        self.cost = cost;
        self
    }

    /// Require operator confirmation before `run`.
    pub(crate) fn confirmed(mut self) -> Self {
        self.requires_confirmation = true;
        self
    }

    /// Mark the tool side-effecting, like write_file/shell: its turn's
    /// history is rollback-exempt, and its presence in a batch keeps the
    /// batch sequential.
    pub(crate) fn mutating(mut self) -> Self {
        self.side_effecting = true;
        self
    }

    /// Opt out of the fan-out concurrency gate, like the `task` tool: the
    /// tool's worker takes no permit, so a nested fan-out it drives is free to
    /// acquire from the same pool.
    pub(crate) fn ungated(mut self) -> Self {
        self.gates_concurrency = false;
        self
    }

    /// Write `text` to the out sink during `run`, like a tool streaming live
    /// output (the `task` tool's child stream).
    pub(crate) fn emitting(mut self, text: &str) -> Self {
        self.emits = Some(text.to_string());
        self
    }

    /// Replace the canned `run` response with `f`.
    pub(crate) fn with_run(
        mut self,
        f: impl Fn(serde_json::Value) -> Result<String, String> + Send + Sync + 'static,
    ) -> Self {
        self.run = Box::new(f);
        self
    }

    /// Replace the always-`Ok` validation with `f`.
    pub(crate) fn with_validate(
        mut self,
        f: impl Fn(&serde_json::Value) -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        self.validate = Box::new(f);
        self
    }

    /// Replace the `None` status line with `f`.
    pub(crate) fn with_status(
        mut self,
        f: impl Fn(&serde_json::Value) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        self.format_status = Box::new(f);
        self
    }
}

impl ToolDef for TestTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        "A configurable test tool"
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "input": { "type": "string" }
            }
        })
    }
    fn cost(&self) -> u8 {
        self.cost
    }
    fn requires_confirmation(&self) -> bool {
        self.requires_confirmation
    }
    fn side_effecting(&self, _input: &serde_json::Value) -> bool {
        self.side_effecting
    }
    fn gates_concurrency(&self) -> bool {
        self.gates_concurrency
    }
    fn validate(&self, input: &serde_json::Value) -> Result<(), String> {
        (self.validate)(input)
    }
    fn format_status(&self, input: &serde_json::Value) -> Option<String> {
        (self.format_status)(input)
    }
    fn run(
        &self,
        input: serde_json::Value,
        out: &mut dyn std::io::Write,
    ) -> Result<String, String> {
        if let Some(text) = &self.emits {
            let _ = out.write_all(text.as_bytes());
        }
        (self.run)(input)
    }
}

// ── Assertion helpers ──

/// Unwrap a [`Block::ToolUse`], panicking with the actual block otherwise.
/// The panic arm is exercised by its own `#[should_panic]` test below, so the
/// helper carries no dead line — unlike a per-call-site `match … => panic!`.
#[track_caller]
pub(crate) fn expect_tool_use(block: &Block) -> (&str, &str, &serde_json::Value) {
    match block {
        Block::ToolUse { id, name, input } => (id, name, input),
        other => panic!("expected ToolUse, got {other:?}"),
    }
}

/// Unwrap a [`Block::Text`], panicking with the actual block otherwise.
#[track_caller]
pub(crate) fn expect_text(block: &Block) -> &str {
    match block {
        Block::Text(text) => text,
        other => panic!("expected Text, got {other:?}"),
    }
}

/// Unwrap a [`Block::ToolResult`] into `(tool_use_id, content, is_error)`,
/// panicking with the actual block otherwise.
#[track_caller]
pub(crate) fn expect_tool_result(block: &Block) -> (&str, &str, bool) {
    match block {
        Block::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => (tool_use_id, content, *is_error),
        other => panic!("expected ToolResult, got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> TurnRequest {
        TurnRequest {
            model: "m".to_string(),
            max_tokens: 16,
            system: None,
            messages: vec![],
            tools: vec![],
            effort: None,
        }
    }

    #[test]
    fn mock_provider_send_returns_turn() {
        let provider = MockProvider::new(vec![]);
        let turn = provider.send(&sample_request()).unwrap();
        assert_eq!(turn.stop_reason, StopReason::EndTurn);
        assert_eq!(turn.blocks, vec![Block::Text("sent".to_string())]);
    }

    #[test]
    fn mock_provider_send_records_requests_and_returns_canned_text() {
        let provider = MockProvider::new(vec![]).with_send_text("a summary");
        let log = provider.send_log();
        let turn = provider.send(&sample_request()).unwrap();
        assert_eq!(turn.blocks, vec![Block::Text("a summary".to_string())]);
        // The handle observes the request recorded through the boxed side.
        assert_eq!(log.borrow().len(), 1);
        assert_eq!(log.borrow()[0].model, "m");
    }

    #[test]
    fn mock_provider_scripted_send_failures_fail_then_resume() {
        let provider = MockProvider::new(vec![]).with_send_failures(1);
        let log = provider.send_log();
        assert!(provider.send(&sample_request()).is_err());
        assert!(provider.send(&sample_request()).is_ok());
        // Failed attempts are recorded too, so tests can count retries.
        assert_eq!(log.borrow().len(), 2);
    }

    #[test]
    fn thread_safe_provider_send_echoes_the_last_user_text() {
        let provider = ThreadSafeProvider::echo();
        let mut request = sample_request();
        request.messages.push(TurnMessage {
            role: Role::User,
            content: vec![Block::Text("echo me".to_string())],
        });
        let turn = provider.send(&request).unwrap();
        assert_eq!(turn.blocks, vec![Block::Text("echo me".to_string())]);
    }

    #[test]
    fn thread_safe_provider_send_with_no_user_text_echoes_empty() {
        // Covers `last_user_text`'s empty fallback (no user message present).
        let provider = ThreadSafeProvider::echo();
        let turn = provider.send(&sample_request()).unwrap();
        assert_eq!(turn.blocks, vec![Block::Text(String::new())]);
    }

    #[test]
    fn thread_safe_provider_failing_errors_on_send_and_stream() {
        let provider = ThreadSafeProvider::failing();
        assert!(provider.send(&sample_request()).is_err());
        assert!(provider.stream(&sample_request()).is_err());
    }

    #[test]
    fn thread_safe_provider_scripted_replays_then_echoes() {
        let provider = ThreadSafeProvider::scripted(vec![echo_stream("canned".to_string())]);
        // First stream call replays the canned deltas.
        let first: Vec<_> = provider.stream(&sample_request()).unwrap().collect();
        assert!(first.iter().any(|d| matches!(
            d,
            Ok(StreamDelta::TextDelta { text, .. }) if text == "canned"
        )));
        // Exhausted, the next call falls back to echoing the prompt.
        let mut request = sample_request();
        request.messages.push(TurnMessage {
            role: Role::User,
            content: vec![Block::Text("fallback".to_string())],
        });
        let second: Vec<_> = provider.stream(&request).unwrap().collect();
        assert!(second.iter().any(|d| matches!(
            d,
            Ok(StreamDelta::TextDelta { text, .. }) if text == "fallback"
        )));
    }

    #[test]
    fn last_user_text_skips_non_text_blocks() {
        // A trailing non-Text block (a tool result) is skipped so the actual
        // prompt text is echoed — exercising the `find_map` miss arm.
        let mut request = sample_request();
        request.messages.push(TurnMessage {
            role: Role::User,
            content: vec![
                Block::Text("the prompt".to_string()),
                Block::ToolResult {
                    tool_use_id: "t".to_string(),
                    content: "r".to_string(),
                    is_error: false,
                },
            ],
        });
        assert_eq!(last_user_text(&request), "the prompt");
    }

    #[test]
    fn thread_safe_provider_send_canned_text_stop_and_log() {
        // The judge-fixture shape: canned verdict text, an overridable stop
        // reason, and a request log observable from outside the factory.
        let log = Arc::new(Mutex::new(Vec::new()));
        let provider = ThreadSafeProvider::echo()
            .with_send_text("ALLOW")
            .with_send_stop_reason(StopReason::MaxTokens)
            .with_send_log(Arc::clone(&log));
        let turn = provider.send(&sample_request()).unwrap();
        assert_eq!(turn.blocks, vec![Block::Text("ALLOW".to_string())]);
        assert_eq!(turn.stop_reason, StopReason::MaxTokens);
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[test]
    fn thread_safe_provider_records_request_effort() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let provider = ThreadSafeProvider::echo().with_effort_log(Arc::clone(&log));
        let mut request = sample_request();
        request.effort = Some("high".to_string());
        let _ = provider.stream(&request).unwrap();
        assert_eq!(log.lock().unwrap().as_slice(), [Some("high".to_string())]);
    }

    #[test]
    fn err_provider_fails_on_every_path() {
        let provider = ErrProvider;
        assert!(provider.send(&sample_request()).is_err());
        assert!(provider.stream(&sample_request()).is_err());
        assert!(provider.list_models().is_err());
    }

    #[test]
    #[should_panic(expected = "prompt must not be consulted")]
    fn no_prompt_panics() {
        no_prompt("any summary");
    }

    #[test]
    fn inert_policy_seams_are_inert() {
        // Direct drives of the shared fixture seams: the resolver misses,
        // and the builder yields a working echo double.
        assert!(no_key("ANY_KEY").is_none());
        let provider = stub_provider(crate::provider::ProviderKind::Anthropic, "k".to_string());
        assert!(provider.list_models().unwrap().is_empty());
    }

    #[test]
    #[should_panic(expected = "expected ToolUse")]
    fn expect_tool_use_panics_on_other_block() {
        expect_tool_use(&Block::Text("not a tool use".to_string()));
    }

    #[test]
    #[should_panic(expected = "expected ToolResult")]
    fn expect_tool_result_panics_on_other_block() {
        expect_tool_result(&Block::Text("not a tool result".to_string()));
    }

    #[test]
    #[should_panic(expected = "expected Text")]
    fn expect_text_panics_on_other_block() {
        expect_text(&Block::ToolResult {
            tool_use_id: "t".to_string(),
            content: "c".to_string(),
            is_error: false,
        });
    }
}
