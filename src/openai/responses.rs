//! The OpenAI adapter's pure mapping layer: the normalized [`crate::turn`]
//! model onto the `/v1/responses` wire types and back — requests, responses,
//! and the [`NormalizedStream`] that translates the semantic SSE events. The
//! live HTTP calls sit in [`super::client_live`]; everything here is hermetic.
//!
//! The Responses API is the endpoint OpenAI serves its whole current catalog
//! on — including the ids Chat Completions rejected outright (`gpt-5.5-pro`,
//! `gpt-5.3-codex`: "Use the v1/responses endpoint instead", verified live
//! 2026-07-06). The adapter therefore routes all OpenAI traffic here, with no
//! per-id routing table — no model registry, one endpoint.
//!
//! The structural mismatches with the normalized model:
//!   * **Items, not role-messages.** The wire wants a flat item array; a
//!     normalized user message fans out into `message` / `function_call_output`
//!     items, an assistant message into `message` / `function_call` items.
//!   * **`instructions`, not a system message.** The system prompt is a
//!     top-level request field.
//!   * **No `is_error` flag** on `function_call_output` — a failed result is
//!     marked in the output string via [`tool_result_content`]'s
//!     `[tool error] ` marker.
//!   * **Reasoning items are skipped**, both directions (see
//!     [`super::responses_types::OutputItem`] for the decision record), and
//!     normalized `Thinking` blocks — another provider's preserved reasoning —
//!     are likewise skipped on send, keeping history portable.
//!   * **No `finish_reason`.** The stop reason is derived from the
//!     response-level `status` plus the presence of `function_call` items
//!     ([`stop_reason_of`]), for the non-streaming response and the stream's
//!     terminal event alike.
//!   * **Stream index re-keying.** The wire keys deltas by `output_index`
//!     over *all* items, including the reasoning items this adapter skips;
//!     [`NormalizedStream`] maps the indices of the items it keeps onto the
//!     agent's flat builder positions and drops deltas for the rest.
//!
//! Prompt caching remains automatic (positional prefix cache); the cache-read
//! count arrives as `usage.input_tokens_details.cached_tokens` and maps into
//! the normalized disjoint accounting exactly as #102 settled for Chat
//! Completions (see [`usage_from_wire`]).

use std::collections::VecDeque;

use crate::openai::responses_types::{
    ContentPart, InputItem, OutputItem, Reasoning, ResponsesApiError, ResponsesRequest,
    ResponsesResponse, ResponsesStreamEvent, ResponsesTool, ResponsesUsage,
};
use crate::provider::ApiError;
use crate::provider::index_map::DenseIndexMap;
use crate::turn::{Block, Role, StopReason, StreamDelta, Turn, TurnRequest, Usage};

/// Encode a tool result for the wire. `function_call_output` items have no
/// `is_error` flag, so a failure is marked in the output string itself.
///
/// Marker choice: `[tool error] ` — bracketed and space-terminated, so no
/// legitimate tool output plausibly opens with it, unlike the natural prose
/// prefix `Error: ` that real output (compiler diagnostics, HTTP bodies,
/// stderr dumps) trivially starts with. Encode-only and one-way: nothing on
/// the receive side parses the marker back out — the normalized model carries
/// the real `is_error` flag, and the marker exists solely for the model to
/// read. Non-error content is passed through byte-for-byte, even when it
/// happens to start with `Error: ` or with the marker itself.
fn tool_result_content(content: &str, is_error: bool) -> String {
    if is_error {
        format!("[tool error] {content}")
    } else {
        content.to_string()
    }
}

/// Parse a tool call's `arguments` string into the normalized object input.
/// Mirrors the agent's streaming accumulator: an empty string is no arguments
/// (`{}`); a non-empty string that fails to parse becomes `Null`, which the
/// agent rejects with `is_error` so the model retries (rather than dispatching
/// on a silently-defaulted input).
fn parse_tool_arguments(arguments: &str) -> serde_json::Value {
    if arguments.is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(arguments).unwrap_or(serde_json::Value::Null)
    }
}

// ── Request: normalized → wire ──

/// Map the normalized request onto the Responses wire request. Fallible: a
/// block under a role the wire cannot express fails request construction
/// rather than being silently dropped.
///
/// Guardrail: no sampling or reasoning parameters (`temperature`,
/// `reasoning.effort`, verbosity) are sent — model defaults hold because
/// reasoning models reject
/// non-default sampling with a 400.
pub(crate) fn to_wire(request: &TurnRequest) -> Result<ResponsesRequest, ApiError> {
    let tools = if request.tools.is_empty() {
        None
    } else {
        Some(
            request
                .tools
                .iter()
                .map(|t| {
                    ResponsesTool::function(
                        t.name.clone(),
                        t.description.clone(),
                        t.input_schema.clone(),
                    )
                })
                .collect(),
        )
    };

    Ok(ResponsesRequest {
        model: request.model.clone(),
        instructions: request.system.clone(),
        input: items_to_wire(request)?,
        max_output_tokens: request.max_tokens,
        tools,
        stream: None,
        store: false,
        // Set only when configured, so a no-effort turn omits the field and
        // stays byte-identical to the pre-effort wire shape (F8 guardrail).
        reasoning: request.effort.clone().map(|effort| Reasoning { effort }),
    })
}

/// Fan the normalized messages out into the flat wire item array. A
/// block/role pairing the agent never produces (a tool-use under the user
/// role, a tool-result under the assistant role) is a contract violation,
/// surfaced as [`ApiError::InvalidRequest`] rather than silently dropped —
/// the same policy as the Chat Completions adapter.
fn items_to_wire(request: &TurnRequest) -> Result<Vec<InputItem>, ApiError> {
    let mut out = Vec::new();
    for message in &request.messages {
        for block in &message.content {
            match (&message.role, block) {
                (role, Block::Text(text)) => out.push(InputItem::Message {
                    role: match role {
                        Role::User => "user".to_string(),
                        Role::Assistant => "assistant".to_string(),
                    },
                    content: text.clone(),
                }),
                (
                    Role::User,
                    Block::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    },
                ) => out.push(InputItem::FunctionCallOutput {
                    call_id: tool_use_id.clone(),
                    output: tool_result_content(content, *is_error),
                }),
                (Role::Assistant, Block::ToolUse { id, name, input }) => {
                    out.push(InputItem::FunctionCall {
                        call_id: id.clone(),
                        name: name.clone(),
                        arguments: input.to_string(),
                    });
                }
                // Preserved reasoning from a thinking model has no portable
                // wire slot here (the skip treatment) — legitimate history,
                // skipped on send, exactly like the Chat Completions adapter.
                (Role::Assistant, Block::Thinking { .. }) => {}
                (Role::User, Block::ToolUse { .. }) => {
                    return Err(ApiError::InvalidRequest(
                        "tool_use block in a user-role message".to_string(),
                    ));
                }
                (Role::User, Block::Thinking { .. }) => {
                    return Err(ApiError::InvalidRequest(
                        "thinking block in a user-role message".to_string(),
                    ));
                }
                (Role::Assistant, Block::ToolResult { .. }) => {
                    return Err(ApiError::InvalidRequest(
                        "tool_result block in an assistant-role message".to_string(),
                    ));
                }
            }
        }
    }
    Ok(out)
}

// ── Response: wire → normalized ──

/// Both the non-streaming `from_wire` guard and the streaming terminal guard
/// (`NormalizedStream::ingest`) hit this on a turn with no usable content —
/// same wording, different state checked (`blocks.is_empty()` vs the stream's
/// content-level `!produced_text && !emitted_tool_use`). One constant keeps a
/// reword from drifting between the two.
const NO_USABLE_CONTENT: &str = "response contained no usable content";

/// `format!` needs a literal format string, so the "ended with status"
/// message — shared by `from_wire`'s status guard and the stream's `Failed`
/// handler — takes a helper instead of a `const`.
fn ended_with_status(status: &str) -> String {
    format!("response ended with status {status}")
}

/// Map the wire response onto a normalized turn. Fallible: a failed status
/// (or one this adapter doesn't recognize) and an empty `output` both surface
/// as named errors rather than a silent empty success — the Chat Completions
/// "no choices" policy, translated.
pub(crate) fn from_wire(response: ResponsesResponse) -> Result<Turn, ApiError> {
    if let Some(error) = &response.error {
        return Err(ApiError::Stream(error_text(error)));
    }
    if !matches!(response.status.as_str(), "completed" | "incomplete") {
        return Err(ApiError::Stream(ended_with_status(&response.status)));
    }
    if response.output.is_empty() {
        return Err(ApiError::Stream(
            "response contained no output items".to_string(),
        ));
    }

    let stop_reason = stop_reason_of(&response);
    let usage = response
        .usage
        .as_ref()
        .map(usage_from_wire)
        .unwrap_or_default();

    let mut blocks = Vec::new();
    for item in response.output {
        match item {
            OutputItem::Message { content } => {
                let text: String = content
                    .into_iter()
                    .filter_map(|part| match part {
                        ContentPart::OutputText { text } => Some(text),
                        ContentPart::Other => None,
                    })
                    .collect();
                if !text.is_empty() {
                    blocks.push(Block::Text(text));
                }
            }
            OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                blocks.push(Block::ToolUse {
                    id: call_id,
                    name,
                    input: parse_tool_arguments(&arguments),
                });
            }
            OutputItem::Reasoning | OutputItem::Other => {}
        }
    }

    // Output items arrived but none carried usable content — a refusal-only
    // response (its sole message part is a `refusal`, which maps to nothing),
    // or one wholly composed of skipped items. Fail fast rather than handing
    // the agent a degenerate turn it would report as a completed empty answer.
    if blocks.is_empty() {
        return Err(ApiError::Stream(NO_USABLE_CONTENT.to_string()));
    }

    Ok(Turn {
        blocks,
        stop_reason,
        usage,
    })
}

/// Render the in-body error object as one line, code-prefixed when present —
/// the same `type: message` shape [`crate::provider::error_message`] gives
/// the non-2xx envelope.
fn error_text(error: &ResponsesApiError) -> String {
    match &error.code {
        Some(code) => format!("{code}: {}", error.message),
        None => error.message.clone(),
    }
}

/// The `MaxTokens`/`EndTurn` distinction for an `incomplete` response, read
/// from its `incomplete_details`. This lives only in the recap, not in stream
/// state, so both the non-streaming and streaming paths derive it here:
/// `max_output_tokens` maps to `MaxTokens`, any other reason (content filter,
/// future kinds, a missing details object) folds to `EndTurn` — the response
/// stopped, and there is no normalized variant for it.
fn incomplete_stop_reason(response: &ResponsesResponse) -> StopReason {
    let is_tokens = response
        .incomplete_details
        .as_ref()
        .and_then(|d| d.reason.as_deref())
        .is_some_and(|reason| reason == "max_output_tokens");
    if is_tokens {
        StopReason::MaxTokens
    } else {
        StopReason::EndTurn
    }
}

/// The normalized stop reason of a finished non-streaming response. An
/// incomplete status takes precedence — whatever items came through, the turn
/// was cut short, and dispatching a possibly-truncated tool call would be
/// worse. A completed response is `ToolUse` iff it carries a `function_call`
/// item. The streaming path derives the completed case from its own emitted
/// state instead (see [`NormalizedStream::ingest`]) rather than trusting the
/// recap to echo the calls, but shares [`incomplete_stop_reason`].
fn stop_reason_of(response: &ResponsesResponse) -> StopReason {
    if response.status == "incomplete" {
        return incomplete_stop_reason(response);
    }
    if response
        .output
        .iter()
        .any(|item| matches!(item, OutputItem::FunctionCall { .. }))
    {
        StopReason::ToolUse
    } else {
        StopReason::EndTurn
    }
}

/// The same subset-vs-disjoint conversion as the Chat Completions adapter:
/// the wire's cached count is a subset of `input_tokens`, the normalized
/// model wants them disjoint, so the cached portion is subtracted out
/// (saturating). No cache-write counter exists — the prefix cache is
/// automatic — so `cache_creation_input_tokens` stays `None`.
fn usage_from_wire(usage: &ResponsesUsage) -> Usage {
    let cached = usage.input_tokens_details.as_ref().map(|d| d.cached_tokens);
    Usage {
        input_tokens: usage.input_tokens.saturating_sub(cached.unwrap_or(0)),
        output_tokens: usage.output_tokens,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: cached,
    }
}

// ── Stream: wire events → normalized deltas ──

/// Translates the Responses SSE event stream into the normalized
/// [`StreamDelta`] vocabulary. Stateful for three reasons:
///   * The wire emits no explicit message-start, so one
///     [`StreamDelta::MessageStart`] is synthesized on the first event.
///   * Wire `output_index` values count *every* item — including the
///     reasoning items this adapter skips — so the indices of the items that
///     are kept are re-keyed onto the agent's flat builder positions, and
///     deltas addressed to unmapped indices (a skipped item's) are dropped.
///   * The terminal event carries the complete final response; its status,
///     items, and usage fold into the single trailing
///     [`StreamDelta::MessageDelta`] (via [`stop_reason_of`] /
///     [`usage_from_wire`]) or into the turn-failing error.
///
/// One event can yield several deltas, so produced deltas are buffered and
/// drained one per `next()`.
pub(crate) struct NormalizedStream {
    /// The raw event source, boxed (not generic) so there is exactly one
    /// instantiation of the mapping logic — fully covered by the hermetic
    /// tests below rather than split across per-caller copies.
    events: Box<dyn Iterator<Item = Result<ResponsesStreamEvent, ApiError>>>,
    buffer: VecDeque<Result<StreamDelta, ApiError>>,
    started: bool,
    ended: bool,
    /// Whether a terminal event (completed/incomplete/failed/error) closed
    /// the turn — the source ending without one is a truncation error.
    finished: bool,
    /// Whether a `ToolUseStart` was emitted. The completed stop reason is read
    /// from this streamed state, not from re-scanning the terminal recap's
    /// `output`: the server is not obliged to echo the full item list, so tool
    /// dispatch must not hinge on the recap carrying the calls back.
    emitted_tool_use: bool,
    /// Whether a non-empty text delta was emitted. With `emitted_tool_use`, this
    /// is the content-level "usable output produced" signal the terminal guard
    /// uses — an opened item slot alone is not usable content (a refusal opens a
    /// message item but streams its text on dropped events), mirroring the
    /// non-streaming `from_wire`, which makes a `Block::Text` only for non-empty
    /// text.
    produced_text: bool,
    /// Wire `output_index` → flat builder position, for the kept items. Deltas
    /// addressed to an unmapped index (a skipped item's) are dropped.
    indices: DenseIndexMap,
}

impl NormalizedStream {
    pub(crate) fn new(
        events: Box<dyn Iterator<Item = Result<ResponsesStreamEvent, ApiError>>>,
    ) -> Self {
        Self {
            events,
            buffer: VecDeque::new(),
            started: false,
            ended: false,
            finished: false,
            emitted_tool_use: false,
            produced_text: false,
            indices: DenseIndexMap::default(),
        }
    }

    /// Translate one event into 0+ buffered deltas; a terminal event closes
    /// the turn (`finished`) and stops the pull (`ended`).
    fn ingest(&mut self, event: ResponsesStreamEvent) {
        if !self.started {
            self.started = true;
            self.buffer.push_back(Ok(StreamDelta::MessageStart {
                usage: Usage::default(),
            }));
        }
        match event {
            ResponsesStreamEvent::OutputItemAdded { item, output_index } => match item {
                OutputItem::Message { .. } => {
                    let index = self.indices.alloc(output_index);
                    self.buffer.push_back(Ok(StreamDelta::TextStart {
                        index,
                        text: String::new(),
                    }));
                }
                OutputItem::FunctionCall { call_id, name, .. } => {
                    let index = self.indices.alloc(output_index);
                    self.emitted_tool_use = true;
                    self.buffer.push_back(Ok(StreamDelta::ToolUseStart {
                        index,
                        id: call_id,
                        name,
                        input: serde_json::json!({}),
                    }));
                }
                // Skipped items get no builder slot; their deltas (if any)
                // are dropped by the unmapped-index check below.
                OutputItem::Reasoning | OutputItem::Other => {}
            },
            ResponsesStreamEvent::OutputTextDelta {
                delta,
                output_index,
            } => {
                if let Some(index) = self.indices.get(output_index) {
                    if !delta.is_empty() {
                        self.produced_text = true;
                    }
                    self.buffer
                        .push_back(Ok(StreamDelta::TextDelta { index, text: delta }));
                }
            }
            ResponsesStreamEvent::FunctionCallArgumentsDelta {
                delta,
                output_index,
            } => {
                if let Some(index) = self.indices.get(output_index) {
                    self.buffer
                        .push_back(Ok(StreamDelta::ToolArgsDelta { index, json: delta }));
                }
            }
            ResponsesStreamEvent::Completed { response }
            | ResponsesStreamEvent::Incomplete { response } => {
                self.finished = true;
                self.ended = true;
                if !self.produced_text && !self.emitted_tool_use {
                    // No usable content streamed: a refusal-only turn (its text
                    // rode dropped events) or an all-skipped turn. Mirror
                    // `from_wire`'s content-level guard so the agent fails fast
                    // instead of reporting a blank completed answer and poisoning
                    // history with an empty assistant block.
                    self.buffer
                        .push_back(Err(ApiError::Stream(NO_USABLE_CONTENT.to_string())));
                    return;
                }
                let usage = response
                    .usage
                    .as_ref()
                    .map(usage_from_wire)
                    .unwrap_or_default();
                // The completed stop reason comes from what actually streamed
                // (`emitted_tool_use`), not from re-scanning the recap; only
                // the incomplete truncation reason still lives in the recap.
                let stop_reason = if response.status == "incomplete" {
                    incomplete_stop_reason(&response)
                } else if self.emitted_tool_use {
                    StopReason::ToolUse
                } else {
                    StopReason::EndTurn
                };
                self.buffer.push_back(Ok(StreamDelta::MessageDelta {
                    stop_reason: Some(stop_reason),
                    usage,
                }));
            }
            ResponsesStreamEvent::Failed { response } => {
                self.finished = true;
                self.ended = true;
                let message = response
                    .error
                    .as_ref()
                    .map(error_text)
                    .unwrap_or_else(|| ended_with_status(&response.status));
                self.buffer.push_back(Err(ApiError::Stream(message)));
            }
            ResponsesStreamEvent::Error { message, code } => {
                self.finished = true;
                self.ended = true;
                let message = match code {
                    Some(code) => format!("{code}: {message}"),
                    None => message,
                };
                self.buffer.push_back(Err(ApiError::Stream(message)));
            }
            ResponsesStreamEvent::Other => {}
        }
    }
}

impl Iterator for NormalizedStream {
    type Item = Result<StreamDelta, ApiError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.buffer.pop_front() {
                return Some(item);
            }
            if self.ended {
                return None;
            }
            match self.events.next() {
                Some(Err(e)) => {
                    // Fail fast: surface the error and stop. No trailing
                    // MessageDelta — the turn aborts.
                    self.ended = true;
                    return Some(Err(e));
                }
                Some(Ok(event)) => self.ingest(event),
                None => {
                    self.ended = true;
                    // The source closed without a terminal event: whatever
                    // streamed is not a complete turn — surface it rather
                    // than reporting partial content as a finished EndTurn.
                    return if self.started && !self.finished {
                        Some(Err(ApiError::Stream(
                            "stream ended without a terminal response event".to_string(),
                        )))
                    } else {
                        None
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::responses_types::{IncompleteDetails, ResponsesApiError};
    use crate::turn::{ToolSpec, TurnMessage};

    fn request_with(messages: Vec<TurnMessage>) -> TurnRequest {
        TurnRequest {
            model: crate::TEST_MODEL_OPENAI.to_string(),
            max_tokens: 512,
            system: None,
            messages,
            tools: vec![],
            effort: None,
        }
    }

    fn user_text(text: &str) -> TurnMessage {
        TurnMessage {
            role: Role::User,
            content: vec![Block::Text(text.to_string())],
        }
    }

    // ── to_wire ──

    #[test]
    fn to_wire_maps_model_tokens_and_store() {
        let wire = to_wire(&request_with(vec![user_text("hi")])).unwrap();
        assert_eq!(wire.model, crate::TEST_MODEL_OPENAI);
        assert_eq!(wire.max_output_tokens, 512);
        assert!(!wire.store, "state is client-managed; store must be false");
        assert!(wire.stream.is_none());
        assert!(wire.tools.is_none());
        assert_eq!(
            wire.input,
            vec![InputItem::Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }]
        );
    }

    #[test]
    fn to_wire_sets_reasoning_when_effort_present() {
        let mut request = request_with(vec![user_text("hi")]);
        request.effort = Some("high".to_string());
        let json = serde_json::to_value(to_wire(&request).unwrap()).unwrap();
        assert_eq!(json["reasoning"], serde_json::json!({"effort": "high"}));
    }

    #[test]
    fn to_wire_omits_reasoning_when_effort_absent() {
        // The no-effort request must stay byte-identical to the pre-effort
        // wire shape: the field is omitted entirely, not sent as null.
        let json =
            serde_json::to_value(to_wire(&request_with(vec![user_text("hi")])).unwrap()).unwrap();
        assert!(json.get("reasoning").is_none());
    }

    #[test]
    fn to_wire_carries_system_as_instructions() {
        let mut request = request_with(vec![user_text("hi")]);
        request.system = Some("be terse".to_string());
        let wire = to_wire(&request).unwrap();
        assert_eq!(wire.instructions.as_deref(), Some("be terse"));
        // No leading system message item — instructions is the only carrier.
        assert_eq!(wire.input.len(), 1);
    }

    #[test]
    fn to_wire_maps_tools_flat() {
        let mut request = request_with(vec![user_text("hi")]);
        request.tools = vec![ToolSpec {
            name: "get_weather".to_string(),
            description: "weather".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let wire = to_wire(&request).unwrap();
        let tools = wire.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].kind, "function");
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].parameters, serde_json::json!({"type": "object"}));
    }

    #[test]
    fn to_wire_fans_out_a_tool_loop_exchange() {
        // The canonical loop: user asks, assistant calls, user carries the
        // result. Each normalized block becomes its own wire item, in order.
        let wire = to_wire(&request_with(vec![
            user_text("weather in Paris?"),
            TurnMessage {
                role: Role::Assistant,
                content: vec![
                    Block::Text("checking".to_string()),
                    Block::ToolUse {
                        id: "call_1".to_string(),
                        name: "get_weather".to_string(),
                        input: serde_json::json!({"city": "Paris"}),
                    },
                ],
            },
            TurnMessage {
                role: Role::User,
                content: vec![Block::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: "15C".to_string(),
                    is_error: false,
                }],
            },
        ]))
        .unwrap();
        assert_eq!(
            wire.input,
            vec![
                InputItem::Message {
                    role: "user".to_string(),
                    content: "weather in Paris?".to_string(),
                },
                InputItem::Message {
                    role: "assistant".to_string(),
                    content: "checking".to_string(),
                },
                InputItem::FunctionCall {
                    call_id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: "{\"city\":\"Paris\"}".to_string(),
                },
                InputItem::FunctionCallOutput {
                    call_id: "call_1".to_string(),
                    output: "15C".to_string(),
                },
            ]
        );
    }

    #[test]
    fn to_wire_marks_error_results_in_the_output_string() {
        let wire = to_wire(&request_with(vec![TurnMessage {
            role: Role::User,
            content: vec![Block::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "no such file".to_string(),
                is_error: true,
            }],
        }]))
        .unwrap();
        assert_eq!(
            wire.input,
            vec![InputItem::FunctionCallOutput {
                call_id: "call_1".to_string(),
                output: "[tool error] no such file".to_string(),
            }]
        );
    }

    #[test]
    fn to_wire_skips_thinking_blocks_in_assistant_history() {
        // Another provider's preserved reasoning is legitimate history with
        // no wire slot — skipped, not rejected, so history stays portable.
        let wire = to_wire(&request_with(vec![TurnMessage {
            role: Role::Assistant,
            content: vec![
                Block::Thinking {
                    text: "hmm".to_string(),
                    signature: "sig".to_string(),
                },
                Block::Text("answer".to_string()),
            ],
        }]))
        .unwrap();
        assert_eq!(
            wire.input,
            vec![InputItem::Message {
                role: "assistant".to_string(),
                content: "answer".to_string(),
            }]
        );
    }

    #[test]
    fn to_wire_rejects_tool_use_under_user_role() {
        let err = to_wire(&request_with(vec![TurnMessage {
            role: Role::User,
            content: vec![Block::ToolUse {
                id: "x".to_string(),
                name: "t".to_string(),
                input: serde_json::json!({}),
            }],
        }]))
        .unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(m) if m.contains("tool_use")));
    }

    #[test]
    fn to_wire_rejects_thinking_under_user_role() {
        let err = to_wire(&request_with(vec![TurnMessage {
            role: Role::User,
            content: vec![Block::Thinking {
                text: "t".to_string(),
                signature: "s".to_string(),
            }],
        }]))
        .unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(m) if m.contains("thinking")));
    }

    #[test]
    fn to_wire_rejects_tool_result_under_assistant_role() {
        let err = to_wire(&request_with(vec![TurnMessage {
            role: Role::Assistant,
            content: vec![Block::ToolResult {
                tool_use_id: "x".to_string(),
                content: "c".to_string(),
                is_error: false,
            }],
        }]))
        .unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(m) if m.contains("tool_result")));
    }

    // ── from_wire ──

    fn completed(output: Vec<OutputItem>) -> ResponsesResponse {
        ResponsesResponse {
            status: "completed".to_string(),
            incomplete_details: None,
            error: None,
            output,
            usage: Some(ResponsesUsage {
                input_tokens: 20,
                input_tokens_details: None,
                output_tokens: 5,
            }),
        }
    }

    fn text_item(text: &str) -> OutputItem {
        OutputItem::Message {
            content: vec![ContentPart::OutputText {
                text: text.to_string(),
            }],
        }
    }

    #[test]
    fn from_wire_maps_a_text_turn() {
        let turn = from_wire(completed(vec![text_item("hello")])).unwrap();
        assert_eq!(turn.blocks, vec![Block::Text("hello".to_string())]);
        assert_eq!(turn.stop_reason, StopReason::EndTurn);
        assert_eq!(turn.usage.input_tokens, 20);
        assert_eq!(turn.usage.output_tokens, 5);
    }

    #[test]
    fn from_wire_maps_a_tool_call_turn() {
        let turn = from_wire(completed(vec![OutputItem::FunctionCall {
            call_id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: "{\"city\":\"Paris\"}".to_string(),
        }]))
        .unwrap();
        assert_eq!(
            turn.blocks,
            vec![Block::ToolUse {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                input: serde_json::json!({"city": "Paris"}),
            }]
        );
        assert_eq!(turn.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn from_wire_skips_reasoning_and_unknown_items() {
        // The reasoning item (the skip treatment) and an unknown item both
        // vanish; the presence of text alone leaves EndTurn.
        let turn = from_wire(completed(vec![
            OutputItem::Reasoning,
            OutputItem::Other,
            text_item("hi"),
        ]))
        .unwrap();
        assert_eq!(turn.blocks, vec![Block::Text("hi".to_string())]);
        assert_eq!(turn.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn from_wire_concatenates_text_parts_and_drops_unknown_parts() {
        let turn = from_wire(completed(vec![OutputItem::Message {
            content: vec![
                ContentPart::OutputText {
                    text: "a".to_string(),
                },
                ContentPart::Other,
                ContentPart::OutputText {
                    text: "b".to_string(),
                },
            ],
        }]))
        .unwrap();
        assert_eq!(turn.blocks, vec![Block::Text("ab".to_string())]);
    }

    #[test]
    fn from_wire_message_with_no_text_yields_no_block() {
        // A message item whose parts carry no text (all unknown) produces no
        // empty text block — but the turn still stands on the other items.
        let turn = from_wire(completed(vec![
            OutputItem::Message {
                content: vec![ContentPart::Other],
            },
            text_item("x"),
        ]))
        .unwrap();
        assert_eq!(turn.blocks, vec![Block::Text("x".to_string())]);
    }

    #[test]
    fn from_wire_incomplete_on_tokens_is_max_tokens() {
        let mut response = completed(vec![text_item("truncated…")]);
        response.status = "incomplete".to_string();
        response.incomplete_details = Some(IncompleteDetails {
            reason: Some("max_output_tokens".to_string()),
        });
        let turn = from_wire(response).unwrap();
        assert_eq!(turn.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn from_wire_incomplete_takes_precedence_over_tool_calls() {
        // A truncated turn that got as far as opening a tool call is still a
        // truncation — dispatching a possibly half-formed call would be worse.
        let mut response = completed(vec![OutputItem::FunctionCall {
            call_id: "c".to_string(),
            name: "t".to_string(),
            arguments: String::new(),
        }]);
        response.status = "incomplete".to_string();
        response.incomplete_details = Some(IncompleteDetails {
            reason: Some("max_output_tokens".to_string()),
        });
        let turn = from_wire(response).unwrap();
        assert_eq!(turn.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn from_wire_incomplete_other_reason_folds_to_end_turn() {
        // content_filter (or a future reason, or a missing details object):
        // the response stopped; there is no normalized variant for it.
        for details in [
            Some(IncompleteDetails {
                reason: Some("content_filter".to_string()),
            }),
            Some(IncompleteDetails { reason: None }),
            None,
        ] {
            let mut response = completed(vec![text_item("x")]);
            response.status = "incomplete".to_string();
            response.incomplete_details = details;
            assert_eq!(
                from_wire(response).unwrap().stop_reason,
                StopReason::EndTurn
            );
        }
    }

    #[test]
    fn from_wire_error_object_fails_the_turn() {
        let mut response = completed(vec![]);
        response.status = "failed".to_string();
        response.error = Some(ResponsesApiError {
            message: "The model had an error".to_string(),
            code: Some("server_error".to_string()),
        });
        let err = from_wire(response).unwrap_err();
        assert!(
            matches!(&err, ApiError::Stream(m) if m == "server_error: The model had an error"),
            "got {err:?}"
        );
    }

    #[test]
    fn from_wire_error_without_code_is_bare_message() {
        let mut response = completed(vec![]);
        response.error = Some(ResponsesApiError {
            message: "boom".to_string(),
            code: None,
        });
        let err = from_wire(response).unwrap_err();
        assert!(matches!(&err, ApiError::Stream(m) if m == "boom"));
    }

    #[test]
    fn from_wire_unrecognized_status_fails_the_turn() {
        // `failed` without an error object, `cancelled`, `queued` — anything
        // that isn't a finished generation surfaces rather than passing a
        // degenerate turn to the agent.
        let mut response = completed(vec![text_item("x")]);
        response.status = "failed".to_string();
        let err = from_wire(response).unwrap_err();
        assert!(matches!(&err, ApiError::Stream(m) if m.contains("status failed")));
    }

    #[test]
    fn from_wire_empty_output_fails_the_turn() {
        let err = from_wire(completed(vec![])).unwrap_err();
        assert!(matches!(&err, ApiError::Stream(m) if m.contains("no output items")));
    }

    #[test]
    fn from_wire_refusal_only_response_fails_the_turn() {
        // A refusal-only turn: the message's sole content part is a refusal
        // (→ ContentPart::Other), so output is non-empty but no usable block
        // comes through. It must error, not return a silent empty turn.
        let err = from_wire(completed(vec![OutputItem::Message {
            content: vec![ContentPart::Other],
        }]))
        .unwrap_err();
        assert!(
            matches!(&err, ApiError::Stream(m) if m == NO_USABLE_CONTENT),
            "got {err:?}"
        );
    }

    #[test]
    fn from_wire_empty_arguments_parse_as_empty_object() {
        let turn = from_wire(completed(vec![OutputItem::FunctionCall {
            call_id: "c".to_string(),
            name: "ping".to_string(),
            arguments: String::new(),
        }]))
        .unwrap();
        assert_eq!(
            turn.blocks,
            vec![Block::ToolUse {
                id: "c".to_string(),
                name: "ping".to_string(),
                input: serde_json::json!({}),
            }]
        );
    }

    #[test]
    fn from_wire_malformed_arguments_parse_as_null() {
        // Same policy as the Chat Completions adapter: Null input reaches the
        // agent, which rejects the call back to the model as an error.
        let turn = from_wire(completed(vec![OutputItem::FunctionCall {
            call_id: "c".to_string(),
            name: "t".to_string(),
            arguments: "{not json".to_string(),
        }]))
        .unwrap();
        assert_eq!(
            turn.blocks,
            vec![Block::ToolUse {
                id: "c".to_string(),
                name: "t".to_string(),
                input: serde_json::Value::Null,
            }]
        );
    }

    #[test]
    fn from_wire_missing_usage_defaults_to_zero() {
        let mut response = completed(vec![text_item("x")]);
        response.usage = None;
        let turn = from_wire(response).unwrap();
        assert_eq!(turn.usage, Usage::default());
    }

    #[test]
    fn from_wire_usage_subtracts_cached_from_input() {
        // The disjoint accounting: wire input 2006 with 1920 cached becomes
        // normalized input 86 + cache_read 1920 (the agent sums them back).
        let mut response = completed(vec![text_item("x")]);
        response.usage = Some(ResponsesUsage {
            input_tokens: 2006,
            input_tokens_details: Some(crate::openai::responses_types::InputTokensDetails {
                cached_tokens: 1920,
            }),
            output_tokens: 300,
        });
        let turn = from_wire(response).unwrap();
        assert_eq!(turn.usage.input_tokens, 86);
        assert_eq!(turn.usage.cache_read_input_tokens, Some(1920));
        assert_eq!(turn.usage.cache_creation_input_tokens, None);
        assert_eq!(turn.usage.output_tokens, 300);
    }

    #[test]
    fn from_wire_usage_oversized_cached_saturates() {
        // A malformed wire reporting more cached than input must not
        // underflow — same saturation as the Chat Completions adapter.
        let mut response = completed(vec![text_item("x")]);
        response.usage = Some(ResponsesUsage {
            input_tokens: 10,
            input_tokens_details: Some(crate::openai::responses_types::InputTokensDetails {
                cached_tokens: 50,
            }),
            output_tokens: 1,
        });
        let turn = from_wire(response).unwrap();
        assert_eq!(turn.usage.input_tokens, 0);
        assert_eq!(turn.usage.cache_read_input_tokens, Some(50));
    }

    #[test]
    fn from_wire_usage_without_details_passes_input_through() {
        let turn = from_wire(completed(vec![text_item("x")])).unwrap();
        assert_eq!(turn.usage.input_tokens, 20);
        assert_eq!(turn.usage.cache_read_input_tokens, None);
    }

    // ── tool_result_content (the [tool error] marker) ──

    #[test]
    fn tool_result_error_prose_content_is_not_marked() {
        // Real output that merely *reads* like an error ("Error: …" prose)
        // passes through byte-for-byte when is_error is false.
        assert_eq!(
            tool_result_content("Error: not really an error", false),
            "Error: not really an error"
        );
    }

    #[test]
    fn tool_result_marker_shaped_content_passes_through() {
        // Content that happens to start with the marker is not de-duplicated
        // or escaped — encode-only, one-way.
        assert_eq!(
            tool_result_content("[tool error] already looks marked", false),
            "[tool error] already looks marked"
        );
        assert_eq!(
            tool_result_content("[tool error] already looks marked", true),
            "[tool error] [tool error] already looks marked"
        );
    }

    // ── parse_tool_arguments ──

    #[test]
    fn parse_tool_arguments_empty_malformed_and_object() {
        // Empty → `{}`, malformed → Null (the agent rejects the call back to
        // the model), valid → the object.
        assert_eq!(parse_tool_arguments(""), serde_json::json!({}));
        assert_eq!(parse_tool_arguments("{not json"), serde_json::Value::Null);
        assert_eq!(
            parse_tool_arguments(r#"{"location":"SF"}"#),
            serde_json::json!({"location": "SF"})
        );
    }

    // ── NormalizedStream ──

    /// Drive [`NormalizedStream`] over a scripted event sequence, collecting
    /// the produced deltas (errors unwrapped by the caller's expectations).
    fn normalize(
        events: Vec<Result<ResponsesStreamEvent, ApiError>>,
    ) -> Vec<Result<StreamDelta, ApiError>> {
        NormalizedStream::new(Box::new(events.into_iter())).collect()
    }

    /// Shorthand for an item-added event.
    fn added(item: OutputItem, output_index: u32) -> Result<ResponsesStreamEvent, ApiError> {
        Ok(ResponsesStreamEvent::OutputItemAdded { item, output_index })
    }

    fn message_item() -> OutputItem {
        OutputItem::Message { content: vec![] }
    }

    fn call_item(call_id: &str, name: &str) -> OutputItem {
        OutputItem::FunctionCall {
            call_id: call_id.to_string(),
            name: name.to_string(),
            arguments: String::new(),
        }
    }

    fn text_delta(delta: &str, output_index: u32) -> Result<ResponsesStreamEvent, ApiError> {
        Ok(ResponsesStreamEvent::OutputTextDelta {
            delta: delta.to_string(),
            output_index,
        })
    }

    fn args_delta(delta: &str, output_index: u32) -> Result<ResponsesStreamEvent, ApiError> {
        Ok(ResponsesStreamEvent::FunctionCallArgumentsDelta {
            delta: delta.to_string(),
            output_index,
        })
    }

    fn completed_event(output: Vec<OutputItem>) -> Result<ResponsesStreamEvent, ApiError> {
        Ok(ResponsesStreamEvent::Completed {
            response: completed(output),
        })
    }

    #[test]
    fn stream_text_only_turn() {
        // The live shape: bookkeeping events (→ Other), a reasoning item at
        // wire index 0 (skipped), the message at wire index 1 — re-keyed to
        // flat 0 — then the terminal recap with usage.
        let deltas: Vec<StreamDelta> = normalize(vec![
            Ok(ResponsesStreamEvent::Other), // response.created
            added(OutputItem::Reasoning, 0),
            added(message_item(), 1),
            text_delta("1\n", 1),
            text_delta("2", 1),
            completed_event(vec![text_item("1\n2")]),
        ])
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap();
        assert_eq!(
            deltas,
            vec![
                StreamDelta::MessageStart {
                    usage: Usage::default()
                },
                StreamDelta::TextStart {
                    index: 0,
                    text: String::new()
                },
                StreamDelta::TextDelta {
                    index: 0,
                    text: "1\n".to_string()
                },
                StreamDelta::TextDelta {
                    index: 0,
                    text: "2".to_string()
                },
                StreamDelta::MessageDelta {
                    stop_reason: Some(StopReason::EndTurn),
                    usage: Usage {
                        input_tokens: 20,
                        output_tokens: 5,
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: None,
                    },
                },
            ]
        );
    }

    #[test]
    fn stream_tool_call_turn() {
        // A function_call item announces call_id/name up front, arguments
        // stream as deltas, and the terminal recap (carrying the call) makes
        // the stop reason ToolUse.
        let deltas: Vec<StreamDelta> = normalize(vec![
            added(call_item("call_9", "get_weather"), 0),
            args_delta("{\"city\":", 0),
            args_delta("\"Paris\"}", 0),
            completed_event(vec![call_item("call_9", "get_weather")]),
        ])
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap();
        assert_eq!(
            deltas[1],
            StreamDelta::ToolUseStart {
                index: 0,
                id: "call_9".to_string(),
                name: "get_weather".to_string(),
                input: serde_json::json!({}),
            }
        );
        assert_eq!(
            deltas[2],
            StreamDelta::ToolArgsDelta {
                index: 0,
                json: "{\"city\":".to_string()
            }
        );
        assert!(matches!(
            deltas[4],
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                ..
            }
        ));
    }

    #[test]
    fn stream_tool_call_with_truncated_recap_is_tool_use() {
        // The stop reason comes from the streamed ToolUseStart, not the recap:
        // a function_call streamed in full, but the terminal response.completed
        // carries "output": [] — the server is not obliged to echo the items.
        // Tool dispatch must still classify this as ToolUse.
        let deltas: Vec<StreamDelta> = normalize(vec![
            added(call_item("call_9", "get_weather"), 0),
            args_delta("{}", 0),
            completed_event(vec![]),
        ])
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap();
        assert!(matches!(
            deltas.last(),
            Some(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                ..
            })
        ));
    }

    #[test]
    fn stream_no_content_block_fails_fast() {
        // Only skipped items (bookkeeping, a reasoning item) reached the stream
        // before the terminal event: no block-start was emitted, so the turn
        // fails fast rather than closing with an empty MessageDelta — the
        // streaming mirror of from_wire's no-usable-content guard.
        let results = normalize(vec![
            Ok(ResponsesStreamEvent::Other),
            added(OutputItem::Reasoning, 0),
            completed_event(vec![]),
        ]);
        assert!(
            matches!(
                results.last().unwrap(),
                Err(ApiError::Stream(m)) if m == NO_USABLE_CONTENT
            ),
            "got {results:?}"
        );
        assert!(
            !results
                .iter()
                .any(|r| matches!(r, Ok(StreamDelta::MessageDelta { .. }))),
            "no terminal delta on an empty turn"
        );
    }

    #[test]
    fn stream_message_item_without_text_fails_fast() {
        // A message item opened a builder slot (allocating an index and emitting
        // an empty TextStart) but streamed no text — the refusal-only shape,
        // where the text rides dropped `Other` events. The terminal guard now
        // keys on content produced, not merely an index allocated, so this fails
        // fast instead of closing with an empty completed answer.
        let results = normalize(vec![added(message_item(), 0), completed_event(vec![])]);
        assert!(
            matches!(
                results.last().unwrap(),
                Err(ApiError::Stream(m)) if m == NO_USABLE_CONTENT
            ),
            "got {results:?}"
        );
        assert!(
            !results
                .iter()
                .any(|r| matches!(r, Ok(StreamDelta::MessageDelta { .. }))),
            "no terminal delta on a no-content turn"
        );
    }

    #[test]
    fn stream_rekeys_mixed_items_to_flat_indices() {
        // Wire indices 0 (reasoning, skipped), 1 (message), 2 (function_call)
        // land on flat builder positions 0 and 1.
        let deltas: Vec<StreamDelta> = normalize(vec![
            added(OutputItem::Reasoning, 0),
            added(message_item(), 1),
            text_delta("checking", 1),
            added(call_item("c1", "probe"), 2),
            args_delta("{}", 2),
            completed_event(vec![text_item("checking"), call_item("c1", "probe")]),
        ])
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap();
        assert!(matches!(deltas[1], StreamDelta::TextStart { index: 0, .. }));
        assert!(matches!(deltas[2], StreamDelta::TextDelta { index: 0, .. }));
        assert!(matches!(
            deltas[3],
            StreamDelta::ToolUseStart { index: 1, .. }
        ));
        assert!(matches!(
            deltas[4],
            StreamDelta::ToolArgsDelta { index: 1, .. }
        ));
    }

    #[test]
    fn stream_drops_deltas_for_unmapped_indices() {
        // Deltas addressed to a skipped item's index (a reasoning summary
        // stream, a future item kind) vanish rather than corrupting the
        // builder positions.
        let deltas: Vec<StreamDelta> = normalize(vec![
            added(OutputItem::Reasoning, 0),
            text_delta("private reasoning", 0),
            args_delta("{}", 7),
            added(message_item(), 1),
            text_delta("public answer", 1),
            completed_event(vec![text_item("public answer")]),
        ])
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap();
        assert_eq!(deltas.len(), 4, "got {deltas:?}");
        assert!(matches!(
            &deltas[2],
            StreamDelta::TextDelta { index: 0, text } if text == "public answer"
        ));
    }

    #[test]
    fn stream_incomplete_terminal_is_max_tokens() {
        let mut response = completed(vec![text_item("truncat")]);
        response.status = "incomplete".to_string();
        response.incomplete_details = Some(IncompleteDetails {
            reason: Some("max_output_tokens".to_string()),
        });
        let deltas: Vec<StreamDelta> = normalize(vec![
            added(message_item(), 0),
            text_delta("truncat", 0),
            Ok(ResponsesStreamEvent::Incomplete { response }),
        ])
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap();
        assert!(matches!(
            deltas.last(),
            Some(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::MaxTokens),
                ..
            })
        ));
    }

    #[test]
    fn stream_failed_terminal_surfaces_the_error() {
        let mut response = completed(vec![]);
        response.status = "failed".to_string();
        response.error = Some(ResponsesApiError {
            message: "The model had an error".to_string(),
            code: Some("server_error".to_string()),
        });
        let results = normalize(vec![
            added(message_item(), 0),
            Ok(ResponsesStreamEvent::Failed { response }),
        ]);
        let last = results.last().unwrap();
        assert!(
            matches!(last, Err(ApiError::Stream(m)) if m == "server_error: The model had an error"),
            "got {last:?}"
        );
    }

    #[test]
    fn stream_failed_terminal_without_error_object_names_the_status() {
        let mut response = completed(vec![]);
        response.status = "failed".to_string();
        let results = normalize(vec![Ok(ResponsesStreamEvent::Failed { response })]);
        assert!(matches!(
            results.last().unwrap(),
            Err(ApiError::Stream(m)) if m == "response ended with status failed"
        ));
    }

    #[test]
    fn stream_error_event_fails_the_turn() {
        // With and without a code — the code prefixes like the error body.
        let results = normalize(vec![Ok(ResponsesStreamEvent::Error {
            message: "slow down".to_string(),
            code: Some("rate_limit_exceeded".to_string()),
        })]);
        assert!(matches!(
            results.last().unwrap(),
            Err(ApiError::Stream(m)) if m == "rate_limit_exceeded: slow down"
        ));

        let results = normalize(vec![Ok(ResponsesStreamEvent::Error {
            message: "bare".to_string(),
            code: None,
        })]);
        assert!(matches!(
            results.last().unwrap(),
            Err(ApiError::Stream(m)) if m == "bare"
        ));
    }

    #[test]
    fn stream_source_ending_without_terminal_is_an_error() {
        // Content streamed but the connection dropped before a terminal
        // event: the partial turn must not pass as a finished one.
        let results = normalize(vec![added(message_item(), 0), text_delta("par", 0)]);
        assert!(matches!(
            results.last().unwrap(),
            Err(ApiError::Stream(m)) if m.contains("without a terminal")
        ));
    }

    #[test]
    fn stream_empty_source_yields_nothing() {
        assert!(normalize(vec![]).is_empty());
    }

    #[test]
    fn stream_source_error_propagates_and_stops() {
        let results = normalize(vec![
            added(message_item(), 0),
            Err(ApiError::Io(std::io::Error::other("reset"))),
        ]);
        // MessageStart + TextStart, then the error; nothing after.
        assert_eq!(results.len(), 3);
        assert!(matches!(results.last().unwrap(), Err(ApiError::Io(_))));
    }

    #[test]
    fn stream_events_after_terminal_are_not_pulled() {
        // `ended` stops the pull once the terminal event lands; a stray
        // trailing event is never ingested (the script would grow deltas). The
        // message streams a text delta so the turn carries usable content and
        // closes with a MessageDelta rather than the no-content fail-fast.
        let results = normalize(vec![
            added(message_item(), 0),
            text_delta("x", 0),
            completed_event(vec![text_item("x")]),
            text_delta("stray", 0),
        ]);
        assert_eq!(results.len(), 4, "got {results:?}");
        assert!(matches!(
            results.last().unwrap(),
            Ok(StreamDelta::MessageDelta { .. })
        ));
    }
}
