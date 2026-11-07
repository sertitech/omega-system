//! Shared setup for the agent's inline unit and integration tests.

use super::*;
use crate::provider::DeltaStream;
use crate::testing::{MockProvider, TestTool};
use crate::turn::{StreamDelta, Turn, TurnRequest};
use std::cell::RefCell;
use std::collections::VecDeque;

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
pub(super) struct SucceedThenErrProvider {
    streams: RefCell<VecDeque<Vec<Result<StreamDelta, ApiError>>>>,
}
impl SucceedThenErrProvider {
    pub(super) fn new(streams: Vec<Vec<StreamDelta>>) -> Self {
        Self {
            streams: RefCell::new(
                streams
                    .into_iter()
                    .map(|stream| stream.into_iter().map(Ok).collect())
                    .collect(),
            ),
        }
    }

    pub(super) fn with_result_stream(stream: Vec<Result<StreamDelta, ApiError>>) -> Self {
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
pub(super) fn text_and_tool_use_stream(id: &str, name: &str, text: &str) -> Vec<StreamDelta> {
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
pub(super) fn streams_to_the_wall(stream: Vec<StreamDelta>) -> Vec<Vec<StreamDelta>> {
    std::iter::repeat_n(stream, MAX_TURNS as usize).collect()
}

/// True when no two adjacent messages share the `User` role — the
/// alternating-role contract the coalescing must preserve.
pub(super) fn no_consecutive_users(messages: &[TurnMessage]) -> bool {
    messages
        .windows(2)
        .all(|w| !(w[0].role == Role::User && w[1].role == Role::User))
}

pub(super) fn mock_tool(name: &str, response: &str) -> Box<dyn ToolDef> {
    Box::new(TestTool::new(name, response))
}

/// A tool that reports itself side-effecting, like write_file/shell. Its
/// `run` succeeds with a fixed string — the mutation it stands in for is
/// notional; the `side_effecting` flag is what drives the rollback logic.
pub(super) fn side_effect_tool(name: &str, response: &str) -> Box<dyn ToolDef> {
    Box::new(TestTool::new(name, response).mutating())
}

/// A side-effecting tool whose `run` fails, modeling a mutation that
/// touched disk and then errored. The flag is set *before* `run`, so
/// history must still be preserved on a later error this turn.
pub(super) fn failing_side_effect_tool(name: &str, error: &str) -> Box<dyn ToolDef> {
    let error = error.to_string();
    Box::new(
        TestTool::new(name, "")
            .mutating()
            .with_run(move |_| Err(error.clone())),
    )
}

/// A canned stream that calls tool `name` (with id `id`) and stops on
/// `ToolUse`, so the agent loop will execute the tool and loop again.
pub(super) fn tool_use_stream(id: &str, name: &str) -> Vec<StreamDelta> {
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
pub(super) fn text_stream(text: &str) -> Vec<StreamDelta> {
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

pub(super) fn test_config(system: Option<&str>) -> AgentConfig {
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

pub(super) fn agent_with_tools(tools: Vec<Box<dyn ToolDef>>) -> Agent {
    Agent::new(
        Box::new(MockProvider::new(vec![])),
        test_config(None),
        tools,
    )
}

/// An `ask` policy answering through `f` — what the pre-policy tests
/// installed as a bare closure. `Send + Sync` now, so recorders use
/// `Arc`, never `Rc`.
pub(super) fn ask_stub(f: impl Fn(&str) -> bool + Send + Sync + 'static) -> ConfirmPolicy {
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
pub(super) fn allow_stub() -> ConfirmPolicy {
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
pub(super) fn judged_policy(
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
pub(super) fn judged_stub(verdict: &'static str) -> ConfirmPolicy {
    judged_policy(move || {
        Box::new(crate::testing::ThreadSafeProvider::echo().with_send_text(verdict))
    })
}

pub(super) fn agent_with_system(system: &str) -> Agent {
    Agent::new(
        Box::new(MockProvider::new(vec![])),
        test_config(Some(system)),
        vec![],
    )
}

pub(super) fn user_msg(text: &str) -> TurnMessage {
    TurnMessage {
        role: Role::User,
        content: vec![Block::Text(text.to_string())],
    }
}

pub(super) fn assistant_msg(text: &str) -> TurnMessage {
    TurnMessage {
        role: Role::Assistant,
        content: vec![Block::Text(text.to_string())],
    }
}

pub(super) fn stub_usage() -> Usage {
    Usage::default()
}

pub(super) fn message_start() -> StreamDelta {
    StreamDelta::MessageStart {
        usage: stub_usage(),
    }
}

/// A confirmation-gated double. The canned response goes through
/// [`TestTool::new`]'s shared default `run`, so the denied test (where
/// `run` must not fire) adds no never-executed closure.
pub(super) fn confirmed_tool() -> Box<dyn ToolDef> {
    Box::new(TestTool::new("guarded_write", "wrote something").confirmed())
}

/// A bare tool_use block for the fan-out tests.
pub(super) fn tool_use_block(id: &str, name: &str) -> Block {
    Block::ToolUse {
        id: id.to_string(),
        name: name.to_string(),
        input: serde_json::json!({}),
    }
}

/// A canned stream that reports `input` up front, emits `text`, and ends
/// the turn reporting `output` — the Anthropic-shaped usage split the
/// usage-line tests drive.
pub(super) fn measured_text_stream(text: &str, input: u32, output: u32) -> Vec<StreamDelta> {
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

pub(super) fn tool_use_msg(id: &str, name: &str) -> TurnMessage {
    TurnMessage {
        role: Role::Assistant,
        content: vec![Block::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input: serde_json::json!({}),
        }],
    }
}

pub(super) fn tool_result_msg(id: &str, content: &str) -> TurnMessage {
    TurnMessage {
        role: Role::User,
        content: vec![Block::ToolResult {
            tool_use_id: id.to_string(),
            content: content.to_string(),
            is_error: false,
        }],
    }
}
