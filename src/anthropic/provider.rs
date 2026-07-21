//! The Anthropic adapter's pure mapping layer: the normalized [`crate::turn`]
//! model onto the Messages API wire types and back. The [`Provider`] impl that
//! drives it over HTTP lives in [`super::provider_live`], which the coverage
//! gate excludes wholesale — everything here is hermetic.
//!
//! Two Anthropic-specific concerns live here rather than in the agent loop:
//!   * **Prompt caching.** [`to_wire`] places cache breakpoints on the system
//!     prompt and the conversation tail by emitting
//!     `cache_control: ephemeral`. The normalized model has no cache concept, so
//!     a future automatic-caching provider (OpenAI) simply does nothing here.
//!   * **The string-or-blocks content shape.** A plain user-text message
//!     serializes as a bare string and everything else as a block array.

use crate::anthropic::types::{
    self, CacheControl, ContentBlock, Delta, Message, MessageContent, MessagesRequest,
    MessagesResponse, OutputConfig, StreamEvent, SystemBlock, SystemContent, Tool,
};
use crate::provider::ApiError;
use crate::provider::index_map::DenseIndexMap;
use crate::turn::{Block, Role, StopReason, StreamDelta, Turn, TurnMessage, TurnRequest, Usage};

// ── Request: normalized → wire ──

/// Build the Anthropic request, applying the two prompt-cache breakpoints.
///
/// Guardrail: no sampling parameters (`temperature`, `top_p`, `top_k`) are
/// sent, and any future sampling knob must not be serialized unconditionally:
/// some reasoning models reject non-default sampling values.
pub(crate) fn to_wire(request: &TurnRequest) -> MessagesRequest {
    // The system prompt is always a cached block (breakpoint #1).
    let system = request
        .system
        .as_ref()
        .map(|text| SystemContent::Blocks(vec![SystemBlock::cached(text.clone())]));

    let tools = if request.tools.is_empty() {
        None
    } else {
        Some(
            request
                .tools
                .iter()
                .map(|t| Tool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t.input_schema.clone(),
                })
                .collect(),
        )
    };

    let mut messages: Vec<Message> = request.messages.iter().map(message_to_wire).collect();
    apply_cache_breakpoint(&mut messages);

    MessagesRequest {
        model: request.model.clone(),
        max_tokens: request.max_tokens,
        system,
        messages,
        tools,
        stream: None,
        // Set only when configured, so a no-effort turn omits the field and
        // stays byte-identical to the pre-effort wire shape.
        output_config: request.effort.clone().map(|effort| OutputConfig { effort }),
    }
}

fn message_to_wire(message: &TurnMessage) -> Message {
    let role = role_to_wire(&message.role);
    // Preserve the wire shape: a plain user-text message is a bare string;
    // everything else (assistant turns, tool results) is a block array.
    let content = match (&role, message.content.as_slice()) {
        (types::Role::User, [Block::Text(text)]) => MessageContent::Text(text.clone()),
        (_, blocks) => MessageContent::Blocks(blocks.iter().map(block_to_wire).collect()),
    };
    Message { role, content }
}

/// Mark the conversation tail as a cache breakpoint (breakpoint #2): the last
/// block of the last message gets `cache_control: ephemeral`, converting a
/// bare-string last message into block form so the marker has somewhere to go.
fn apply_cache_breakpoint(messages: &mut [Message]) {
    let Some(last) = messages.last_mut() else {
        return;
    };
    if let MessageContent::Text(text) = &last.content {
        last.content = MessageContent::Blocks(vec![ContentBlock::Text {
            text: text.clone(),
            cache_control: Some(CacheControl::Ephemeral),
        }]);
        return;
    }
    if let MessageContent::Blocks(blocks) = &mut last.content
        && let Some(block) = blocks.last_mut()
    {
        match block {
            ContentBlock::Text { cache_control, .. }
            | ContentBlock::ToolUse { cache_control, .. }
            | ContentBlock::ToolResult { cache_control, .. } => {
                *cache_control = Some(CacheControl::Ephemeral);
            }
            // A thinking block rejects cache markers (the API 400s on them),
            // and an unknown block has no cache_control to mark. Neither ends
            // a request the adapter builds — the last message is always
            // user-role — so skipping is defensive, not a lost breakpoint.
            ContentBlock::Thinking { .. } | ContentBlock::Unknown => {}
        }
    }
}

fn block_to_wire(block: &Block) -> ContentBlock {
    match block {
        Block::Text(text) => ContentBlock::Text {
            text: text.clone(),
            cache_control: None,
        },
        // Resent verbatim, signature included: the multi-turn thinking
        // contract requires the assistant's thinking blocks back unmodified
        // on tool-use turns, or the request is rejected.
        Block::Thinking { text, signature } => ContentBlock::Thinking {
            thinking: text.clone(),
            signature: signature.clone(),
        },
        Block::ToolUse { id, name, input } => ContentBlock::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
            cache_control: None,
        },
        Block::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => ContentBlock::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: content.clone(),
            // The wire omits the flag when the call succeeded.
            is_error: is_error.then_some(true),
            cache_control: None,
        },
    }
}

fn role_to_wire(role: &Role) -> types::Role {
    match role {
        Role::User => types::Role::User,
        Role::Assistant => types::Role::Assistant,
    }
}

// ── Response: wire → normalized ──

pub(crate) fn from_wire(response: MessagesResponse) -> Turn {
    Turn {
        blocks: response
            .content
            .into_iter()
            .filter_map(block_from_wire)
            .collect(),
        stop_reason: response
            .stop_reason
            .map(stop_reason_from_wire)
            .unwrap_or(StopReason::EndTurn),
        usage: usage_from_wire(&response.usage),
    }
}

/// Map a wire block to its normalized counterpart. Thinking blocks survive —
/// the multi-turn contract needs them (with their signature) back on the wire
/// in tool-use turns. Truly unknown block types (e.g. `redacted_thinking`)
/// still map to `None` and are dropped from the turn: forward compatibility is
/// unchanged for types this build has no normalized block for.
fn block_from_wire(block: ContentBlock) -> Option<Block> {
    match block {
        ContentBlock::Text { text, .. } => Some(Block::Text(text)),
        ContentBlock::Thinking {
            thinking,
            signature,
        } => Some(Block::Thinking {
            text: thinking,
            signature,
        }),
        ContentBlock::ToolUse {
            id, name, input, ..
        } => Some(Block::ToolUse { id, name, input }),
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
            ..
        } => Some(Block::ToolResult {
            tool_use_id,
            content,
            is_error: is_error.unwrap_or(false),
        }),
        ContentBlock::Unknown => None,
    }
}

fn stop_reason_from_wire(reason: types::StopReason) -> StopReason {
    match reason {
        types::StopReason::EndTurn => StopReason::EndTurn,
        types::StopReason::ToolUse => StopReason::ToolUse,
        types::StopReason::MaxTokens => StopReason::MaxTokens,
        types::StopReason::StopSequence => StopReason::StopSequence,
        // An unknown reason behaves like the absent one: the turn simply
        // ended. This mirrors the OpenAI adapter's wildcard finish_reason arm.
        types::StopReason::Other => StopReason::EndTurn,
    }
}

fn usage_from_wire(usage: &types::Usage) -> Usage {
    Usage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_creation_input_tokens: usage.cache_creation_input_tokens,
        cache_read_input_tokens: usage.cache_read_input_tokens,
    }
}

// ── Stream: wire events → normalized deltas ──

/// Translates Anthropic's SSE event stream into the normalized [`StreamDelta`]
/// vocabulary. Errors pass through; events the agent does not consume are
/// dropped. Stateful because the agent's accumulator pushes builders in
/// arrival order while keying deltas by index: a dropped block start (a
/// `tool_result`, or an unknown type like `redacted_thinking`) would leave
/// every later wire index pointing one builder too far, so surviving blocks —
/// thinking included — are re-keyed onto a dense index space via the shared
/// [`DenseIndexMap`]. The live adapter's `stream()` is exactly
/// `NormalizedStream::new(events)`, so the whole mapping stays hermetically
/// covered here.
pub(crate) struct NormalizedStream {
    /// The raw event source, boxed (not generic) so there is exactly one
    /// instantiation of the mapping logic — fully covered by the hermetic
    /// tests below rather than split across per-caller copies.
    events: Box<dyn Iterator<Item = Result<StreamEvent, ApiError>>>,
    /// Wire content-block index → flat builder position, for the surviving
    /// block starts.
    indices: DenseIndexMap,
    /// Whether Anthropic's terminal `message_stop` event closed the turn.
    /// A source EOF without it is a truncated stream, even when a preceding
    /// `message_delta` carried a stop reason.
    finished: bool,
    /// Whether iteration has terminated, either cleanly at `message_stop` or
    /// after surfacing one upstream/truncation error.
    ended: bool,
}

impl NormalizedStream {
    pub(crate) fn new(events: Box<dyn Iterator<Item = Result<StreamEvent, ApiError>>>) -> Self {
        Self {
            events,
            indices: DenseIndexMap::default(),
            finished: false,
            ended: false,
        }
    }

    /// Map one Anthropic stream event to its normalized delta. Returns `None`
    /// for events the agent does not consume: `content_block_stop`,
    /// block starts it has no builder for (the unreachable `tool_result` and
    /// unknown types like `redacted_thinking`), and deltas addressed to a
    /// dropped block. `message_stop` changes the stream's terminal state but
    /// produces no normalized delta.
    fn delta_from_event(&mut self, event: StreamEvent) -> Option<StreamDelta> {
        match event {
            StreamEvent::MessageStart { message } => Some(StreamDelta::MessageStart {
                usage: usage_from_wire(&message.usage),
            }),
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                ContentBlock::Text { text, .. } => Some(StreamDelta::TextStart {
                    index: self.indices.alloc(index),
                    text,
                }),
                ContentBlock::Thinking {
                    thinking,
                    signature,
                } => Some(StreamDelta::ThinkingStart {
                    index: self.indices.alloc(index),
                    text: thinking,
                    signature,
                }),
                ContentBlock::ToolUse {
                    id, name, input, ..
                } => Some(StreamDelta::ToolUseStart {
                    index: self.indices.alloc(index),
                    id,
                    name,
                    input,
                }),
                // A tool_result never opens a streamed assistant block, and an
                // unknown block type has no normalized counterpart. Neither
                // allocates a builder position, so their deltas are dropped
                // below and the blocks that follow stay aligned.
                ContentBlock::ToolResult { .. } | ContentBlock::Unknown => None,
            },
            StreamEvent::ContentBlockDelta { index, delta } => {
                let index = self.indices.get(index)?;
                match delta {
                    Delta::TextDelta { text } => Some(StreamDelta::TextDelta { index, text }),
                    Delta::ThinkingDelta { thinking } => Some(StreamDelta::ThinkingDelta {
                        index,
                        text: thinking,
                    }),
                    Delta::SignatureDelta { signature } => {
                        Some(StreamDelta::SignatureDelta { index, signature })
                    }
                    Delta::InputJsonDelta { partial_json } => Some(StreamDelta::ToolArgsDelta {
                        index,
                        json: partial_json,
                    }),
                    // An unknown delta type extends a block the agent does
                    // not build.
                    Delta::Unknown => None,
                }
            }
            StreamEvent::MessageDelta { delta, usage } => Some(StreamDelta::MessageDelta {
                stop_reason: delta.stop_reason.map(stop_reason_from_wire),
                usage: usage_from_wire(&usage),
            }),
            StreamEvent::ContentBlockStop { .. } => None,
            StreamEvent::MessageStop => {
                self.finished = true;
                None
            }
        }
    }
}

impl Iterator for NormalizedStream {
    type Item = Result<StreamDelta, ApiError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.ended {
                return None;
            }
            if self.finished {
                // `message_stop` is terminal. Do not pull or normalize any
                // stray source events after it.
                self.ended = true;
                return None;
            }
            match self.events.next() {
                Some(Err(e)) => {
                    // Surface the upstream failure exactly once. It already
                    // explains why the turn is incomplete, so the next poll
                    // must not add a misleading missing-terminal error.
                    self.ended = true;
                    return Some(Err(e));
                }
                Some(Ok(event)) => {
                    if let Some(delta) = self.delta_from_event(event) {
                        return Some(Ok(delta));
                    }
                }
                None => {
                    self.ended = true;
                    return Some(Err(ApiError::Stream(
                        "stream ended without a terminal message_stop event".to_string(),
                    )));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::types::{MessageDeltaBody, Usage as WireUsage};
    use crate::turn::ToolSpec;

    fn text_msg(role: Role, text: &str) -> TurnMessage {
        TurnMessage {
            role,
            content: vec![Block::Text(text.to_string())],
        }
    }

    /// Unwrap block-form message content, panicking on the bare-string form.
    /// The panic arm is exercised by its own `#[should_panic]` test, so the
    /// helper carries no dead line.
    #[track_caller]
    fn expect_blocks(content: &MessageContent) -> &[ContentBlock] {
        match content {
            MessageContent::Blocks(blocks) => blocks,
            other => panic!("expected Blocks, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "expected Blocks")]
    fn expect_blocks_panics_on_bare_string() {
        expect_blocks(&MessageContent::Text("bare".to_string()));
    }

    // ── to_wire: system & tools ──

    #[test]
    fn to_wire_caches_system_prompt() {
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: Some("You are helpful.".to_string()),
            messages: vec![text_msg(Role::User, "hi")],
            tools: vec![],
            effort: None,
        };
        let wire = to_wire(&request);
        let json = serde_json::to_value(&wire.system).unwrap();
        assert_eq!(json[0]["type"], "text");
        assert_eq!(json[0]["text"], "You are helpful.");
        assert_eq!(json[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn to_wire_no_system_when_none() {
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![text_msg(Role::User, "hi")],
            tools: vec![],
            effort: None,
        };
        assert!(to_wire(&request).system.is_none());
    }

    #[test]
    fn to_wire_maps_tools() {
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![text_msg(Role::User, "hi")],
            tools: vec![ToolSpec {
                name: "echo".to_string(),
                description: "Echo it".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            effort: None,
        };
        let wire = to_wire(&request);
        let tools = wire.tools.expect("tools present");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        assert_eq!(tools[0].description, "Echo it");
        assert_eq!(tools[0].input_schema["type"], "object");
    }

    #[test]
    fn to_wire_omits_tools_when_empty() {
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![text_msg(Role::User, "hi")],
            tools: vec![],
            effort: None,
        };
        assert!(to_wire(&request).tools.is_none());
    }

    #[test]
    fn to_wire_passes_model_and_max_tokens() {
        let request = TurnRequest {
            model: "claude-x".to_string(),
            max_tokens: 999,
            system: None,
            messages: vec![text_msg(Role::User, "hi")],
            tools: vec![],
            effort: None,
        };
        let wire = to_wire(&request);
        assert_eq!(wire.model, "claude-x");
        assert_eq!(wire.max_tokens, 999);
        assert!(wire.stream.is_none());
    }

    #[test]
    fn to_wire_sets_output_config_when_effort_present() {
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![text_msg(Role::User, "hi")],
            tools: vec![],
            effort: Some("high".to_string()),
        };
        let json = serde_json::to_value(to_wire(&request)).unwrap();
        assert_eq!(json["output_config"], serde_json::json!({"effort": "high"}));
    }

    #[test]
    fn to_wire_omits_output_config_when_effort_absent() {
        // The no-effort request must stay byte-identical to the pre-effort
        // wire shape: the field is omitted entirely, not sent as null.
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![text_msg(Role::User, "hi")],
            tools: vec![],
            effort: None,
        };
        let json = serde_json::to_value(to_wire(&request)).unwrap();
        assert!(json.get("output_config").is_none());
    }

    // ── to_wire: message shapes & cache breakpoint ──

    #[test]
    fn to_wire_user_text_message_is_bare_string_when_not_last() {
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![
                text_msg(Role::User, "first"),
                text_msg(Role::Assistant, "second"),
                text_msg(Role::User, "third"),
            ],
            tools: vec![],
            effort: None,
        };
        let wire = to_wire(&request);
        // The first (non-last) user message stays a bare string.
        assert!(matches!(
            &wire.messages[0].content,
            MessageContent::Text(t) if t == "first"
        ));
    }

    #[test]
    fn to_wire_caches_last_message_text_form() {
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![
                text_msg(Role::User, "a"),
                text_msg(Role::Assistant, "b"),
                text_msg(Role::User, "c"),
            ],
            tools: vec![],
            effort: None,
        };
        let wire = to_wire(&request);
        // Last message converts to block form with a cache breakpoint.
        let blocks = expect_blocks(&wire.messages[2].content);
        assert!(matches!(
            &blocks[0],
            ContentBlock::Text { text, cache_control }
                if text == "c" && *cache_control == Some(CacheControl::Ephemeral)
        ));
        // The non-last user message stays a bare string.
        assert!(matches!(&wire.messages[0].content, MessageContent::Text(_)));
        // The non-last assistant single-text message is a BLOCK ARRAY (not a
        // bare string — that shape is reserved for user text) and carries no
        // cache breakpoint: only the conversation tail does.
        let blocks = expect_blocks(&wire.messages[1].content);
        assert!(matches!(
            &blocks[0],
            ContentBlock::Text { text, cache_control }
                if text == "b" && cache_control.is_none()
        ));
    }

    #[test]
    fn to_wire_caches_last_block_form_text() {
        // A block-form last message ending in a *text* block (an assistant
        // tail) takes the in-place marker path, not the bare-string conversion
        // reserved for user text.
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![text_msg(Role::Assistant, "tail")],
            tools: vec![],
            effort: None,
        };
        let wire = to_wire(&request);
        let blocks = expect_blocks(&wire.messages[0].content);
        assert!(matches!(
            &blocks[0],
            ContentBlock::Text { text, cache_control }
                if text == "tail" && *cache_control == Some(CacheControl::Ephemeral)
        ));
    }

    #[test]
    fn to_wire_caches_last_tool_use_block() {
        // A last message ending in a tool_use block anchors the breakpoint on
        // that block — the third arm of the marker's or-pattern.
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![TurnMessage {
                role: Role::Assistant,
                content: vec![Block::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "echo".to_string(),
                    input: serde_json::json!({}),
                }],
            }],
            tools: vec![],
            effort: None,
        };
        let wire = to_wire(&request);
        let blocks = expect_blocks(&wire.messages[0].content);
        assert!(matches!(
            &blocks[0],
            ContentBlock::ToolUse { cache_control, .. }
                if *cache_control == Some(CacheControl::Ephemeral)
        ));
    }

    #[test]
    fn to_wire_caches_last_block_form_message() {
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![TurnMessage {
                role: Role::User,
                content: vec![Block::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: "result".to_string(),
                    is_error: false,
                }],
            }],
            tools: vec![],
            effort: None,
        };
        let wire = to_wire(&request);
        let blocks = expect_blocks(&wire.messages[0].content);
        assert!(matches!(
            &blocks[0],
            ContentBlock::ToolResult { cache_control, .. }
                if *cache_control == Some(CacheControl::Ephemeral)
        ));
    }

    #[test]
    fn cache_breakpoint_skips_unknown_last_block() {
        // No request the adapter builds ends in an Unknown block, but the
        // marker must still handle the variant: nothing to mark, no panic.
        let mut messages = vec![Message {
            role: types::Role::Assistant,
            content: MessageContent::Blocks(vec![ContentBlock::Unknown]),
        }];
        apply_cache_breakpoint(&mut messages);
        assert!(matches!(
            expect_blocks(&messages[0].content),
            [ContentBlock::Unknown]
        ));
    }

    #[test]
    fn to_wire_empty_messages_is_safe() {
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![],
            tools: vec![],
            effort: None,
        };
        assert!(to_wire(&request).messages.is_empty());
    }

    #[test]
    fn to_wire_last_message_empty_blocks_is_safe() {
        // A last message with no blocks has nowhere to anchor the tail cache
        // breakpoint; to_wire must skip it without panicking.
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![TurnMessage {
                role: Role::Assistant,
                content: vec![],
            }],
            tools: vec![],
            effort: None,
        };
        assert!(expect_blocks(&to_wire(&request).messages[0].content).is_empty());
    }

    #[test]
    fn to_wire_user_tool_result_then_text_is_a_block_array() {
        // The agent's conditional rollback (D2) can leave a user message
        // carrying both a tool_result and a coalesced text block. Anthropic
        // accepts that mixed shape as a block array — assert the adapter emits
        // both blocks in order rather than collapsing or rejecting them.
        let message = TurnMessage {
            role: Role::User,
            content: vec![
                Block::ToolResult {
                    tool_use_id: "t1".to_string(),
                    content: "wrote".to_string(),
                    is_error: false,
                },
                Block::Text("again".to_string()),
            ],
        };
        let wire = message_to_wire(&message);
        assert_eq!(wire.role, types::Role::User);
        let blocks = expect_blocks(&wire.content);
        assert_eq!(blocks.len(), 2);
        assert!(matches!(blocks[0], ContentBlock::ToolResult { .. }));
        assert!(matches!(blocks[1], ContentBlock::Text { .. }));
    }

    // ── block_to_wire ──

    #[test]
    fn block_to_wire_tool_result_error_sets_flag() {
        let block = block_to_wire(&Block::ToolResult {
            tool_use_id: "t1".to_string(),
            content: "boom".to_string(),
            is_error: true,
        });
        assert!(matches!(
            block,
            ContentBlock::ToolResult {
                is_error: Some(true),
                ..
            }
        ));
    }

    #[test]
    fn block_to_wire_tool_result_success_omits_flag() {
        let block = block_to_wire(&Block::ToolResult {
            tool_use_id: "t1".to_string(),
            content: "ok".to_string(),
            is_error: false,
        });
        assert!(matches!(
            block,
            ContentBlock::ToolResult { is_error: None, .. }
        ));
    }

    #[test]
    fn block_to_wire_tool_use_roundtrips_input() {
        let block = block_to_wire(&Block::ToolUse {
            id: "t1".to_string(),
            name: "echo".to_string(),
            input: serde_json::json!({"x": 1}),
        });
        assert!(matches!(
            block,
            ContentBlock::ToolUse { id, name, input, .. }
                if id == "t1" && name == "echo" && input["x"] == 1
        ));
    }

    #[test]
    fn block_to_wire_thinking_resends_verbatim() {
        // The multi-turn thinking contract: text and signature return to the
        // wire exactly as received, in the API's own shape.
        let block = block_to_wire(&Block::Thinking {
            text: "let me reason".to_string(),
            signature: "sig_abc".to_string(),
        });
        assert_eq!(
            serde_json::to_value(&block).unwrap(),
            serde_json::json!({
                "type": "thinking",
                "thinking": "let me reason",
                "signature": "sig_abc",
            })
        );
    }

    #[test]
    fn to_wire_assistant_thinking_survives_with_breakpoint_on_the_tail() {
        // A tool-use turn's assistant message leads with its thinking block;
        // to_wire must carry it through untouched while the tail cache
        // breakpoint anchors on the *last* block, never the thinking one.
        let request = TurnRequest {
            model: "m".to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![TurnMessage {
                role: Role::Assistant,
                content: vec![
                    Block::Thinking {
                        text: "reasoning".to_string(),
                        signature: "sig".to_string(),
                    },
                    Block::Text("answer".to_string()),
                ],
            }],
            tools: vec![],
            effort: None,
        };
        let wire = to_wire(&request);
        let blocks = expect_blocks(&wire.messages[0].content);
        assert!(matches!(
            &blocks[0],
            ContentBlock::Thinking { thinking, signature }
                if thinking == "reasoning" && signature == "sig"
        ));
        assert!(matches!(
            &blocks[1],
            ContentBlock::Text { cache_control, .. }
                if *cache_control == Some(CacheControl::Ephemeral)
        ));
    }

    #[test]
    fn cache_breakpoint_skips_thinking_last_block() {
        // No request the adapter builds ends in a thinking block (the last
        // message is always user-role), but the marker must still skip it:
        // the API rejects cache_control on thinking blocks.
        let mut messages = vec![Message {
            role: types::Role::Assistant,
            content: MessageContent::Blocks(vec![ContentBlock::Thinking {
                thinking: "hmm".to_string(),
                signature: "sig".to_string(),
            }]),
        }];
        apply_cache_breakpoint(&mut messages);
        assert_eq!(
            serde_json::to_value(expect_blocks(&messages[0].content)).unwrap(),
            serde_json::json!([{
                "type": "thinking",
                "thinking": "hmm",
                "signature": "sig",
            }])
        );
    }

    // ── from_wire ──

    #[test]
    fn from_wire_text_response() {
        let response = MessagesResponse {
            id: "msg_1".to_string(),
            role: types::Role::Assistant,
            content: vec![ContentBlock::Text {
                text: "hello".to_string(),
                cache_control: None,
            }],
            model: "m".to_string(),
            stop_reason: Some(types::StopReason::EndTurn),
            usage: WireUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: Some(2),
                cache_read_input_tokens: Some(3),
            },
        };
        let turn = from_wire(response);
        assert_eq!(turn.blocks, vec![Block::Text("hello".to_string())]);
        assert_eq!(turn.stop_reason, StopReason::EndTurn);
        assert_eq!(turn.usage.input_tokens, 10);
        assert_eq!(turn.usage.output_tokens, 5);
        assert_eq!(turn.usage.cache_creation_input_tokens, Some(2));
        assert_eq!(turn.usage.cache_read_input_tokens, Some(3));
    }

    #[test]
    fn from_wire_tool_use_and_result_blocks() {
        let response = MessagesResponse {
            id: "msg_1".to_string(),
            role: types::Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "echo".to_string(),
                    input: serde_json::json!({"v": 2}),
                    cache_control: None,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "t0".to_string(),
                    content: "prior".to_string(),
                    is_error: Some(true),
                    cache_control: None,
                },
            ],
            model: "m".to_string(),
            stop_reason: Some(types::StopReason::ToolUse),
            usage: WireUsage {
                input_tokens: 1,
                output_tokens: 1,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        let turn = from_wire(response);
        assert_eq!(turn.stop_reason, StopReason::ToolUse);
        assert_eq!(
            turn.blocks[0],
            Block::ToolUse {
                id: "t1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({"v": 2}),
            }
        );
        assert_eq!(
            turn.blocks[1],
            Block::ToolResult {
                tool_use_id: "t0".to_string(),
                content: "prior".to_string(),
                is_error: true,
            }
        );
    }

    #[test]
    fn from_wire_tool_result_absent_error_flag_is_false() {
        // An omitted wire `is_error` (None) maps to the normalized `false`,
        // mirroring block_to_wire's success path in the other direction.
        let response = MessagesResponse {
            id: "msg_1".to_string(),
            role: types::Role::Assistant,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t0".to_string(),
                content: "ok".to_string(),
                is_error: None,
                cache_control: None,
            }],
            model: "m".to_string(),
            stop_reason: Some(types::StopReason::EndTurn),
            usage: WireUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        assert_eq!(
            from_wire(response).blocks[0],
            Block::ToolResult {
                tool_use_id: "t0".to_string(),
                content: "ok".to_string(),
                is_error: false,
            }
        );
    }

    #[test]
    fn from_wire_parses_thinking_block() {
        // A thinking-first response: the reasoning block survives, in order,
        // as the normalized thinking block — text and signature intact.
        let response = MessagesResponse {
            id: "msg_1".to_string(),
            role: types::Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "let me think".to_string(),
                    signature: "sig_abc".to_string(),
                },
                ContentBlock::Text {
                    text: "answer".to_string(),
                    cache_control: None,
                },
            ],
            model: "m".to_string(),
            stop_reason: Some(types::StopReason::EndTurn),
            usage: WireUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        assert_eq!(
            from_wire(response).blocks,
            vec![
                Block::Thinking {
                    text: "let me think".to_string(),
                    signature: "sig_abc".to_string(),
                },
                Block::Text("answer".to_string()),
            ]
        );
    }

    #[test]
    fn from_wire_drops_unknown_blocks() {
        // An unknown leading block (e.g. `redacted_thinking`) is dropped; the
        // text block carrying the answer must survive it.
        let response = MessagesResponse {
            id: "msg_1".to_string(),
            role: types::Role::Assistant,
            content: vec![
                ContentBlock::Unknown,
                ContentBlock::Text {
                    text: "answer".to_string(),
                    cache_control: None,
                },
            ],
            model: "m".to_string(),
            stop_reason: Some(types::StopReason::EndTurn),
            usage: WireUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        assert_eq!(
            from_wire(response).blocks,
            vec![Block::Text("answer".to_string())]
        );
    }

    #[test]
    fn from_wire_missing_stop_reason_defaults_to_end_turn() {
        let response = MessagesResponse {
            id: "msg_1".to_string(),
            role: types::Role::Assistant,
            content: vec![],
            model: "m".to_string(),
            stop_reason: None,
            usage: WireUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        assert_eq!(from_wire(response).stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn stop_reason_maps_all_variants() {
        assert_eq!(
            stop_reason_from_wire(types::StopReason::EndTurn),
            StopReason::EndTurn
        );
        assert_eq!(
            stop_reason_from_wire(types::StopReason::ToolUse),
            StopReason::ToolUse
        );
        assert_eq!(
            stop_reason_from_wire(types::StopReason::MaxTokens),
            StopReason::MaxTokens
        );
        assert_eq!(
            stop_reason_from_wire(types::StopReason::StopSequence),
            StopReason::StopSequence
        );
        // The catch-all for reasons this build doesn't know (`refusal`,
        // `pause_turn`, …) behaves like an ordinary end of turn.
        assert_eq!(
            stop_reason_from_wire(types::StopReason::Other),
            StopReason::EndTurn
        );
    }

    #[test]
    fn role_maps_both_variants() {
        assert_eq!(role_to_wire(&Role::User), types::Role::User);
        assert_eq!(role_to_wire(&Role::Assistant), types::Role::Assistant);
    }

    // ── NormalizedStream ──

    fn wire_usage(cache_read: u32, cache_created: u32) -> WireUsage {
        WireUsage {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: Some(cache_created),
            cache_read_input_tokens: Some(cache_read),
        }
    }

    /// Collect the deltas a fresh [`NormalizedStream`] yields for `events`,
    /// panicking on any error item.
    fn normalize(mut events: Vec<StreamEvent>) -> Vec<StreamDelta> {
        // Adapter mapping fixtures represent complete Anthropic streams. Tests
        // of the truncation/error boundary below construct the iterator
        // directly so they can deliberately omit `message_stop`.
        events.push(StreamEvent::MessageStop);
        NormalizedStream::new(Box::new(events.into_iter().map(Ok)))
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    fn text_start(index: u32, text: &str) -> StreamEvent {
        StreamEvent::ContentBlockStart {
            index,
            content_block: ContentBlock::Text {
                text: text.to_string(),
                cache_control: None,
            },
        }
    }

    fn text_delta(index: u32, text: &str) -> StreamEvent {
        StreamEvent::ContentBlockDelta {
            index,
            delta: Delta::TextDelta {
                text: text.to_string(),
            },
        }
    }

    fn tool_use_start(index: u32) -> StreamEvent {
        StreamEvent::ContentBlockStart {
            index,
            content_block: ContentBlock::ToolUse {
                id: "t1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
                cache_control: None,
            },
        }
    }

    fn thinking_start(index: u32) -> StreamEvent {
        StreamEvent::ContentBlockStart {
            index,
            content_block: ContentBlock::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
        }
    }

    #[test]
    fn delta_from_message_start_carries_usage() {
        let event = StreamEvent::MessageStart {
            message: MessagesResponse {
                id: "m".to_string(),
                role: types::Role::Assistant,
                content: vec![],
                model: "m".to_string(),
                stop_reason: None,
                usage: wire_usage(5, 7),
            },
        };
        assert!(matches!(
            normalize(vec![event]).as_slice(),
            [StreamDelta::MessageStart { usage }]
                if usage.cache_read_input_tokens == Some(5)
                    && usage.cache_creation_input_tokens == Some(7)
        ));
    }

    #[test]
    fn delta_from_text_start() {
        assert_eq!(
            normalize(vec![text_start(0, "hi")]),
            vec![StreamDelta::TextStart {
                index: 0,
                text: "hi".to_string()
            }]
        );
    }

    #[test]
    fn delta_from_tool_use_start_rekeys_index() {
        // The wire index (2) is not echoed: builder positions are dense,
        // counting only the block starts that survive normalization.
        assert_eq!(
            normalize(vec![tool_use_start(2)]),
            vec![StreamDelta::ToolUseStart {
                index: 0,
                id: "t1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            }]
        );
    }

    #[test]
    fn delta_from_thinking_start_and_deltas() {
        // A streamed thinking block: the start opens a builder position and
        // the reasoning/signature deltas land on it.
        let deltas = normalize(vec![
            thinking_start(0),
            StreamEvent::ContentBlockDelta {
                index: 0,
                delta: Delta::ThinkingDelta {
                    thinking: "let me".to_string(),
                },
            },
            StreamEvent::ContentBlockDelta {
                index: 0,
                delta: Delta::SignatureDelta {
                    signature: "sig".to_string(),
                },
            },
        ]);
        assert_eq!(
            deltas,
            vec![
                StreamDelta::ThinkingStart {
                    index: 0,
                    text: String::new(),
                    signature: String::new(),
                },
                StreamDelta::ThinkingDelta {
                    index: 0,
                    text: "let me".to_string(),
                },
                StreamDelta::SignatureDelta {
                    index: 0,
                    signature: "sig".to_string(),
                },
            ]
        );
    }

    #[test]
    fn leading_thinking_block_and_text_both_survive_with_dense_indices() {
        // The historical misalignment case, now with the thinking block *kept*:
        // claude-fable-5 streams thinking at wire index 0 before the text at
        // wire index 1. Both survive normalization, each on its own dense
        // builder position, and every delta follows its block.
        let deltas = normalize(vec![
            thinking_start(0),
            StreamEvent::ContentBlockDelta {
                index: 0,
                delta: Delta::ThinkingDelta {
                    thinking: "reasoning".to_string(),
                },
            },
            StreamEvent::ContentBlockDelta {
                index: 0,
                delta: Delta::SignatureDelta {
                    signature: "sig".to_string(),
                },
            },
            text_start(1, ""),
            text_delta(1, "the answer"),
        ]);
        assert_eq!(
            deltas,
            vec![
                StreamDelta::ThinkingStart {
                    index: 0,
                    text: String::new(),
                    signature: String::new(),
                },
                StreamDelta::ThinkingDelta {
                    index: 0,
                    text: "reasoning".to_string(),
                },
                StreamDelta::SignatureDelta {
                    index: 0,
                    signature: "sig".to_string(),
                },
                StreamDelta::TextStart {
                    index: 1,
                    text: String::new(),
                },
                StreamDelta::TextDelta {
                    index: 1,
                    text: "the answer".to_string(),
                },
            ]
        );
    }

    #[test]
    fn delta_from_tool_result_start_is_dropped() {
        let event = StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::ToolResult {
                tool_use_id: "t1".to_string(),
                content: "x".to_string(),
                is_error: None,
                cache_control: None,
            },
        };
        assert_eq!(normalize(vec![event]), vec![]);
    }

    #[test]
    fn delta_from_text_and_args_deltas() {
        // Deltas land at the builder position their block start was assigned.
        let deltas = normalize(vec![
            text_start(0, ""),
            tool_use_start(1),
            text_delta(0, "yo"),
            StreamEvent::ContentBlockDelta {
                index: 1,
                delta: Delta::InputJsonDelta {
                    partial_json: "{\"x\":".to_string(),
                },
            },
        ]);
        assert_eq!(
            deltas[2],
            StreamDelta::TextDelta {
                index: 0,
                text: "yo".to_string()
            }
        );
        assert_eq!(
            deltas[3],
            StreamDelta::ToolArgsDelta {
                index: 1,
                json: "{\"x\":".to_string()
            }
        );
    }

    #[test]
    fn delta_from_message_delta_maps_stop_reason() {
        let event = StreamEvent::MessageDelta {
            delta: MessageDeltaBody {
                stop_reason: Some(types::StopReason::ToolUse),
            },
            usage: wire_usage(0, 0),
        };
        assert!(matches!(
            normalize(vec![event]).as_slice(),
            [StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                ..
            }]
        ));
    }

    #[test]
    fn delta_from_stop_events_is_dropped() {
        assert_eq!(
            normalize(vec![
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::MessageStop,
            ]),
            vec![]
        );
    }

    #[test]
    fn partial_text_then_eof_surfaces_one_terminal_error() {
        let mut stream = NormalizedStream::new(Box::new(
            vec![Ok(text_start(0, "")), Ok(text_delta(0, "partial"))].into_iter(),
        ));

        assert!(matches!(
            stream.next(),
            Some(Ok(StreamDelta::TextStart { .. }))
        ));
        assert!(matches!(
            stream.next(),
            Some(Ok(StreamDelta::TextDelta { ref text, .. })) if text == "partial"
        ));
        assert!(matches!(
            stream.next(),
            Some(Err(ApiError::Stream(ref message)))
                if message == "stream ended without a terminal message_stop event"
        ));
        assert!(stream.next().is_none());
    }

    #[test]
    fn stop_reason_without_message_stop_still_errors() {
        let mut stream = NormalizedStream::new(Box::new(
            vec![Ok(StreamEvent::MessageDelta {
                delta: MessageDeltaBody {
                    stop_reason: Some(types::StopReason::EndTurn),
                },
                usage: wire_usage(0, 0),
            })]
            .into_iter(),
        ));

        assert!(matches!(
            stream.next(),
            Some(Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                ..
            }))
        ));
        assert!(matches!(stream.next(), Some(Err(ApiError::Stream(_)))));
        assert!(stream.next().is_none());
    }

    #[test]
    fn empty_stream_is_a_terminal_error() {
        let mut stream = NormalizedStream::new(Box::new(std::iter::empty()));

        assert!(matches!(
            stream.next(),
            Some(Err(ApiError::Stream(ref message)))
                if message == "stream ended without a terminal message_stop event"
        ));
        assert!(stream.next().is_none());
    }

    #[test]
    fn message_stop_ends_a_valid_stream_cleanly() {
        let results: Vec<_> = NormalizedStream::new(Box::new(
            vec![
                Ok(text_start(0, "answer")),
                Ok(StreamEvent::MessageDelta {
                    delta: MessageDeltaBody {
                        stop_reason: Some(types::StopReason::EndTurn),
                    },
                    usage: wire_usage(0, 0),
                }),
                Ok(StreamEvent::MessageStop),
            ]
            .into_iter(),
        ))
        .collect();

        assert_eq!(results.len(), 2);
        assert!(matches!(results[0], Ok(StreamDelta::TextStart { .. })));
        assert!(matches!(
            results[1],
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                ..
            })
        ));
    }

    #[test]
    fn leading_unknown_block_does_not_misalign_the_text() {
        // An unknown block (e.g. `redacted_thinking`) streams at wire index 0
        // before the text at wire index 1. The agent's accumulator pushes
        // builders in arrival order but keys deltas by index, so simply
        // dropping the unknown start would leave the text's deltas pointing
        // one builder too far and silently swallow the visible answer. The
        // surviving text block must be re-keyed to position 0 — its start and
        // deltas alike.
        let deltas = normalize(vec![
            StreamEvent::ContentBlockStart {
                index: 0,
                content_block: ContentBlock::Unknown,
            },
            StreamEvent::ContentBlockDelta {
                index: 0,
                delta: Delta::Unknown,
            },
            text_start(1, ""),
            text_delta(1, "the answer"),
            StreamEvent::MessageDelta {
                delta: MessageDeltaBody {
                    stop_reason: Some(types::StopReason::EndTurn),
                },
                usage: wire_usage(0, 0),
            },
        ]);
        assert_eq!(
            deltas[0],
            StreamDelta::TextStart {
                index: 0,
                text: String::new()
            }
        );
        assert_eq!(
            deltas[1],
            StreamDelta::TextDelta {
                index: 0,
                text: "the answer".to_string()
            }
        );
        assert!(matches!(deltas[2], StreamDelta::MessageDelta { .. }));
        assert_eq!(deltas.len(), 3);
    }

    #[test]
    fn unknown_delta_on_a_known_block_is_dropped() {
        // A future delta type addressed to a block the agent does build must
        // be dropped too, leaving the block itself intact.
        assert_eq!(
            normalize(vec![
                text_start(0, "hi"),
                StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: Delta::Unknown,
                },
            ]),
            vec![StreamDelta::TextStart {
                index: 0,
                text: "hi".to_string()
            }]
        );
    }

    #[test]
    fn upstream_error_is_not_followed_by_a_terminal_error() {
        let mut stream = NormalizedStream::new(Box::new(
            vec![
                Ok(text_start(0, "partial")),
                Err(ApiError::Io(std::io::Error::other("reset"))),
                Ok(StreamEvent::MessageStop),
            ]
            .into_iter(),
        ));

        assert!(matches!(
            stream.next(),
            Some(Ok(StreamDelta::TextStart { .. }))
        ));
        assert!(matches!(stream.next(), Some(Err(ApiError::Io(_)))));
        assert!(stream.next().is_none());
    }
}
