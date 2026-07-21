//! OpenAI Responses API wire types (`POST /v1/responses`). These touch the
//! network directly (serde derives) and are deliberately kept separate from the
//! normalized [`crate::turn`] model — the adapter in [`super::responses`] maps
//! between them. Shapes verified against the live API on 2026-07-06 (gpt-5.5,
//! the newest general model, plus gpt-4o / chat-latest / gpt-5.3-codex).
//!
//! Structural differences from the Chat Completions types
//! ([`super::types`]): the conversation is a flat array of typed *items*
//! (`message`, `function_call`, `function_call_output`) rather than
//! role-disambiguated messages; the system prompt travels as a top-level
//! `instructions` string; tool definitions are flat (no `function` nesting);
//! the response carries typed `output` items rather than `choices`; and there
//! is no `finish_reason` — the response-level `status` plus the presence of
//! `function_call` items carry the same information.
//!
//! Wire tolerance follows the existing `#[serde(other)]` precedent: unknown
//! item and content-part types — including the server-managed kinds this
//! agent never requests (web search, file search, computer use) — parse to an
//! ignored variant instead of failing the turn.

use serde::{Deserialize, Serialize};

// ── Request ──

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ResponsesRequest {
    pub model: String,
    /// The system prompt. The Responses API carries it as a top-level field
    /// (like Anthropic's `system`), not a leading wire message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub input: Vec<InputItem>,
    pub max_output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Always `false`: the agent manages conversation state itself (the same
    /// client-side history it keeps for every provider), so responses must
    /// not be persisted server-side. Serialized unconditionally — opting out
    /// of storage is the deliberate choice, not a default to omit.
    pub store: bool,
    /// Opt-in reasoning effort. Omitted when absent, so a no-effort request is
    /// byte-identical to today's; set, it carries the opaque effort string the
    /// API validates (`reasoning: {"effort": ...}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
}

/// The Responses API `reasoning` object — carries the reasoning `effort` in
/// OpenAI's wire location. The value is an opaque string validated server-side
/// (see [`crate::turn::TurnRequest::effort`]).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Reasoning {
    pub effort: String,
}

/// One input item. Item `id`s from previous responses are deliberately never
/// sent back: with client-managed state the API accepts id-less items
/// (verified live), and omitting them avoids the server's id-pairing rules
/// (a `function_call` id demands its sibling `reasoning` id) that would
/// otherwise force reasoning items into portable history.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    /// A user or assistant text message.
    Message { role: String, content: String },
    /// A prior model tool call, echoed back as history.
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    /// The tool's result, keyed to its call by `call_id`.
    FunctionCallOutput { call_id: String, output: String },
}

/// A tool definition — flat, unlike Chat Completions' `{type, function:{…}}`
/// nesting.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ResponsesTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

impl ResponsesTool {
    pub fn function(name: String, description: String, parameters: serde_json::Value) -> Self {
        Self {
            kind: "function".to_string(),
            name,
            description,
            parameters,
        }
    }
}

// ── Response ──

#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesResponse {
    /// `completed`, `incomplete`, `failed`, … — the response-level outcome;
    /// there is no per-choice `finish_reason`.
    pub status: String,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    /// Populated (with `status: "failed"`) when generation itself failed
    /// after the HTTP 200 — the Responses analog of Anthropic's mid-stream
    /// `error` event.
    #[serde(default)]
    pub error: Option<ResponsesApiError>,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
}

/// Why an `incomplete` response stopped: `max_output_tokens` or
/// `content_filter`.
#[derive(Debug, Clone, Deserialize)]
pub struct IncompleteDetails {
    #[serde(default)]
    pub reason: Option<String>,
}

/// The in-body error object of a failed response. Distinct from the non-2xx
/// envelope ([`crate::provider::error_message`] handles that one, unchanged —
/// the Responses API uses the same `{"error":{…}}` shape there).
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesApiError {
    pub message: String,
    #[serde(default)]
    pub code: Option<String>,
}

/// One output item. `reasoning` is parsed but deliberately carried no further
/// (the *skip treatment*, decided against the live wire): without a summary
/// request the items arrive empty (`content: [], summary: []`), their only
/// substance being a model-private `encrypted_content` blob — resending that
/// through the portable normalized history would leak one provider's opaque
/// state into another's requests, exactly what [`crate::turn::Block`] keeps
/// out. The tool loop round-trips fine without them (verified live).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    Message {
        #[serde(default)]
        content: Vec<ContentPart>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        #[serde(default)]
        arguments: String,
    },
    Reasoning,
    /// Any item type this agent doesn't consume — the `#[serde(other)]`
    /// same forward-compatible wire tolerance.
    #[serde(other)]
    Other,
}

/// One part of a message item's `content`. Only `output_text` carries the
/// turn's text; annotations and future part kinds are ignored.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    OutputText {
        text: String,
    },
    #[serde(other)]
    Other,
}

/// Token accounting. The Responses API spells these `input_tokens` /
/// `output_tokens` (Chat Completions says `prompt` / `completion`); like
/// there, the cached count nested in `input_tokens_details` is a *subset* of
/// `input_tokens`, which the adapter converts to the normalized disjoint
/// accounting. The details object is defaulted fail-soft: its absence must
/// not fail the turn.
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    #[serde(default)]
    pub input_tokens_details: Option<InputTokensDetails>,
    pub output_tokens: u32,
}

/// The `input_tokens_details` object; only the cached-token count is
/// consumed.
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

// ── Streaming events ──

/// One Responses SSE event, keyed by its `type` field. The dialect is
/// *semantic events*, not Chat-Completions chunk deltas: items are announced
/// (`response.output_item.added`), their content streams in typed deltas
/// keyed by `output_index`, and the stream closes with a terminal event
/// carrying the complete final response — there is no `[DONE]` sentinel.
///
/// Only the events the normalization consumes are named; everything else
/// (`response.created`, `response.in_progress`, `response.content_part.*`,
/// the `*.done` recaps, reasoning-summary deltas, …) parses to [`Other`] and
/// is skipped — the same `#[serde(other)]` tolerance as the item types.
/// Vocabulary captured live 2026-07-06 (text and tool-call streams).
///
/// [`Other`]: ResponsesStreamEvent::Other
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesStreamEvent {
    /// A new output item opened at `output_index`. For `message` items the
    /// content follows as text deltas; for `function_call` items the
    /// announcement itself carries `call_id`/`name`, then the arguments
    /// stream.
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded { item: OutputItem, output_index: u32 },
    /// Text appended to the message item at `output_index`.
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta { delta: String, output_index: u32 },
    /// A JSON fragment appended to the function-call arguments at
    /// `output_index`.
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta { delta: String, output_index: u32 },
    /// Terminal: the turn finished; carries the complete final response
    /// (output recap + usage).
    #[serde(rename = "response.completed")]
    Completed { response: ResponsesResponse },
    /// Terminal: the turn was cut short (`max_output_tokens`,
    /// content filter); the final response carries `incomplete_details`.
    #[serde(rename = "response.incomplete")]
    Incomplete { response: ResponsesResponse },
    /// Terminal: generation failed after the stream opened; the final
    /// response carries the `error` object.
    #[serde(rename = "response.failed")]
    Failed { response: ResponsesResponse },
    /// Terminal: a stream-level error event (the SSE analog of the non-2xx
    /// envelope).
    #[serde(rename = "error")]
    Error {
        message: String,
        #[serde(default)]
        code: Option<String>,
    },
    /// Any event type the normalization doesn't consume.
    #[serde(other)]
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unwrap a message item's parts, panicking with the actual item
    /// otherwise. The panic arms of these helpers are exercised by their own
    /// `#[should_panic]` tests, so they carry no dead line.
    #[track_caller]
    fn expect_message(item: &OutputItem) -> &[ContentPart] {
        match item {
            OutputItem::Message { content } => content,
            other => panic!("expected message item, got {other:?}"),
        }
    }

    /// Unwrap a function-call item as `(call_id, name, arguments)`.
    #[track_caller]
    fn expect_function_call(item: &OutputItem) -> (&str, &str, &str) {
        match item {
            OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => (call_id, name, arguments),
            other => panic!("expected function_call item, got {other:?}"),
        }
    }

    /// Unwrap an output_text part's text.
    #[track_caller]
    fn expect_output_text(part: &ContentPart) -> &str {
        match part {
            ContentPart::OutputText { text } => text,
            other => panic!("expected output_text part, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "expected message item")]
    fn expect_message_panics_on_other() {
        expect_message(&OutputItem::Reasoning);
    }

    #[test]
    #[should_panic(expected = "expected function_call item")]
    fn expect_function_call_panics_on_other() {
        expect_function_call(&OutputItem::Reasoning);
    }

    #[test]
    #[should_panic(expected = "expected output_text part")]
    fn expect_output_text_panics_on_other() {
        expect_output_text(&ContentPart::Other);
    }

    /// Unwrap an item-added stream event as `(item, output_index)`.
    #[track_caller]
    fn expect_added(event: ResponsesStreamEvent) -> (OutputItem, u32) {
        match event {
            ResponsesStreamEvent::OutputItemAdded { item, output_index } => (item, output_index),
            other => panic!("expected OutputItemAdded, got {other:?}"),
        }
    }

    /// Unwrap a terminal stream event's final response.
    #[track_caller]
    fn expect_terminal_response(event: ResponsesStreamEvent) -> ResponsesResponse {
        match event {
            ResponsesStreamEvent::Completed { response }
            | ResponsesStreamEvent::Incomplete { response }
            | ResponsesStreamEvent::Failed { response } => response,
            other => panic!("expected a terminal event, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "expected OutputItemAdded")]
    fn expect_added_panics_on_other() {
        expect_added(ResponsesStreamEvent::Other);
    }

    #[test]
    #[should_panic(expected = "expected a terminal event")]
    fn expect_terminal_response_panics_on_other() {
        expect_terminal_response(ResponsesStreamEvent::Other);
    }

    // ── Request serialization ──

    #[test]
    fn serialize_minimal_request() {
        let req = ResponsesRequest {
            model: "gpt-5.5".to_string(),
            instructions: None,
            input: vec![InputItem::Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            max_output_tokens: 64,
            tools: None,
            stream: None,
            store: false,
            reasoning: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "gpt-5.5");
        assert_eq!(json["max_output_tokens"], 64);
        // store is always on the wire — opting out of server-side state is
        // the point, not an omittable default.
        assert_eq!(json["store"], false);
        assert!(json.get("instructions").is_none());
        assert!(json.get("tools").is_none());
        assert!(json.get("stream").is_none());
        assert_eq!(json["input"][0]["type"], "message");
        assert_eq!(json["input"][0]["role"], "user");
        assert_eq!(json["input"][0]["content"], "hi");
    }

    #[test]
    fn serialize_request_with_instructions_and_stream() {
        let req = ResponsesRequest {
            model: "m".to_string(),
            instructions: Some("be terse".to_string()),
            input: vec![],
            max_output_tokens: 1,
            tools: None,
            stream: Some(true),
            store: false,
            reasoning: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["instructions"], "be terse");
        assert_eq!(json["stream"], true);
    }

    #[test]
    fn serialize_function_call_items() {
        // The tool-loop history items: the model's call echoed back, then the
        // result keyed by call_id. No `id` field — items are sent id-less.
        let call = InputItem::FunctionCall {
            call_id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: "{\"city\":\"Paris\"}".to_string(),
        };
        let json = serde_json::to_value(&call).unwrap();
        assert_eq!(json["type"], "function_call");
        assert_eq!(json["call_id"], "call_1");
        assert_eq!(json["name"], "get_weather");
        assert_eq!(json["arguments"], "{\"city\":\"Paris\"}");
        assert!(json.get("id").is_none());

        let output = InputItem::FunctionCallOutput {
            call_id: "call_1".to_string(),
            output: "15C, sunny".to_string(),
        };
        let json = serde_json::to_value(&output).unwrap();
        assert_eq!(json["type"], "function_call_output");
        assert_eq!(json["call_id"], "call_1");
        assert_eq!(json["output"], "15C, sunny");
    }

    #[test]
    fn serialize_tool_definition_is_flat() {
        // Unlike Chat Completions, no `function` nesting.
        let tool = ResponsesTool::function(
            "get_weather".to_string(),
            "Get current weather".to_string(),
            serde_json::json!({"type": "object"}),
        );
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["name"], "get_weather");
        assert_eq!(json["description"], "Get current weather");
        assert_eq!(json["parameters"]["type"], "object");
        assert!(json.get("function").is_none());
    }

    // ── Response deserialization (shapes captured live, 2026-07-06) ──

    #[test]
    fn deserialize_text_response_with_reasoning_item() {
        // The live gpt-5.5 shape: a (contentless) reasoning item precedes the
        // message; extra fields (`phase`, `annotations`, ids) are ignored.
        let json = r#"{
            "id": "resp_1", "object": "response", "status": "completed",
            "incomplete_details": null, "error": null,
            "output": [
                {"id": "rs_1", "type": "reasoning", "content": [], "summary": []},
                {"id": "msg_1", "type": "message", "status": "completed", "phase": "final_answer",
                 "role": "assistant",
                 "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "hello"}]}
            ],
            "usage": {"input_tokens": 21, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens": 17, "output_tokens_details": {"reasoning_tokens": 10},
                      "total_tokens": 38}
        }"#;
        let resp: ResponsesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "completed");
        assert!(resp.error.is_none());
        assert!(resp.incomplete_details.is_none());
        assert_eq!(resp.output.len(), 2);
        assert!(matches!(resp.output[0], OutputItem::Reasoning));
        let content = expect_message(&resp.output[1]);
        assert_eq!(expect_output_text(&content[0]), "hello");
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 21);
        assert_eq!(usage.output_tokens, 17);
        assert_eq!(usage.input_tokens_details.unwrap().cached_tokens, 0);
    }

    #[test]
    fn deserialize_function_call_response() {
        // The live tool-call shape: `call_id` keys the loop, `id` is ignored.
        let json = r#"{
            "status": "completed",
            "output": [{"id": "fc_1", "type": "function_call", "status": "completed",
                        "arguments": "{\"city\":\"Paris\"}", "call_id": "call_X", "name": "get_weather"}],
            "usage": {"input_tokens": 56, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens": 18, "total_tokens": 74}
        }"#;
        let resp: ResponsesResponse = serde_json::from_str(json).unwrap();
        let (call_id, name, arguments) = expect_function_call(&resp.output[0]);
        assert_eq!(call_id, "call_X");
        assert_eq!(name, "get_weather");
        assert_eq!(arguments, "{\"city\":\"Paris\"}");
    }

    #[test]
    fn deserialize_incomplete_response() {
        // The live max_output_tokens truncation shape.
        let json = r#"{
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [], "usage": null
        }"#;
        let resp: ResponsesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "incomplete");
        assert_eq!(
            resp.incomplete_details.unwrap().reason.as_deref(),
            Some("max_output_tokens")
        );
        assert!(resp.usage.is_none());
    }

    #[test]
    fn deserialize_failed_response_error_object() {
        let json = r#"{
            "status": "failed",
            "error": {"code": "server_error", "message": "The model had an error"},
            "output": []
        }"#;
        let resp: ResponsesResponse = serde_json::from_str(json).unwrap();
        let err = resp.error.unwrap();
        assert_eq!(err.message, "The model had an error");
        assert_eq!(err.code.as_deref(), Some("server_error"));
    }

    #[test]
    fn deserialize_unknown_item_and_part_types_are_tolerated() {
        // Server-managed tool kinds and future part types must not fail the
        // turn — the #[serde(other)] tolerance.
        let json = r#"{
            "status": "completed",
            "output": [
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
                {"type": "message", "content": [
                    {"type": "refusal", "refusal": "no"},
                    {"type": "output_text", "annotations": [], "text": "ok"}
                ]}
            ]
        }"#;
        let resp: ResponsesResponse = serde_json::from_str(json).unwrap();
        assert!(matches!(resp.output[0], OutputItem::Other));
        let content = expect_message(&resp.output[1]);
        assert!(matches!(content[0], ContentPart::Other));
        assert!(matches!(content[1], ContentPart::OutputText { .. }));
    }

    #[test]
    fn deserialize_usage_without_details_defaults_fail_soft() {
        let json = r#"{"status": "completed", "output": [],
                       "usage": {"input_tokens": 5, "output_tokens": 1}}"#;
        let resp: ResponsesResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage.unwrap();
        assert!(usage.input_tokens_details.is_none());
    }

    #[test]
    fn deserialize_usage_details_without_cached_tokens_defaults_to_zero() {
        let json = r#"{"status": "completed", "output": [],
                       "usage": {"input_tokens": 5, "input_tokens_details": {},
                                 "output_tokens": 1}}"#;
        let resp: ResponsesResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens_details.unwrap().cached_tokens, 0);
    }

    #[test]
    fn deserialize_function_call_without_arguments_defaults_empty() {
        // `arguments` is defaulted: a zero-argument call must not fail parse.
        let json = r#"{"status": "completed",
                       "output": [{"type": "function_call", "call_id": "c", "name": "ping"}]}"#;
        let resp: ResponsesResponse = serde_json::from_str(json).unwrap();
        let (_, _, arguments) = expect_function_call(&resp.output[0]);
        assert_eq!(arguments, "");
    }

    // ── Streaming events (shapes captured live, 2026-07-06) ──

    #[test]
    fn deserialize_output_item_added_events() {
        // A message item opening (extra fields ignored)…
        let json = r#"{"type":"response.output_item.added",
                       "item":{"id":"msg_1","type":"message","status":"in_progress","content":[],"role":"assistant"},
                       "output_index":1,"sequence_number":4}"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        let (item, output_index) = expect_added(event);
        assert!(matches!(item, OutputItem::Message { .. }));
        assert_eq!(output_index, 1);

        // …and a function_call item, whose announcement carries call_id/name.
        let json = r#"{"type":"response.output_item.added",
                       "item":{"id":"fc_1","type":"function_call","status":"in_progress",
                               "arguments":"","call_id":"call_9","name":"get_weather"},
                       "output_index":0,"sequence_number":2}"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        let (item, _) = expect_added(event);
        let (call_id, name, _) = expect_function_call(&item);
        assert_eq!(call_id, "call_9");
        assert_eq!(name, "get_weather");
    }

    #[test]
    fn deserialize_text_and_arguments_deltas() {
        let json = r#"{"type":"response.output_text.delta","content_index":0,
                       "delta":"Hi","item_id":"msg_1","logprobs":[],"output_index":1,
                       "sequence_number":7,"obfuscation":"xx"}"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            ResponsesStreamEvent::OutputTextDelta { ref delta, output_index: 1 } if delta == "Hi"
        ));

        let json = r#"{"type":"response.function_call_arguments.delta","delta":"{\"",
                       "item_id":"fc_1","obfuscation":"y","output_index":0,"sequence_number":4}"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            ResponsesStreamEvent::FunctionCallArgumentsDelta { ref delta, output_index: 0 } if delta == "{\""
        ));
    }

    #[test]
    fn deserialize_terminal_events() {
        let json = r#"{"type":"response.completed",
                       "response":{"id":"resp_1","status":"completed","output":[],
                                   "usage":{"input_tokens":9,"output_tokens":17}},
                       "sequence_number":16}"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, ResponsesStreamEvent::Completed { .. }));
        let response = expect_terminal_response(event);
        assert_eq!(response.usage.unwrap().output_tokens, 17);

        let json = r#"{"type":"response.incomplete",
                       "response":{"status":"incomplete",
                                   "incomplete_details":{"reason":"max_output_tokens"},"output":[]}}"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, ResponsesStreamEvent::Incomplete { .. }));
        let response = expect_terminal_response(event);
        assert_eq!(
            response.incomplete_details.unwrap().reason.as_deref(),
            Some("max_output_tokens")
        );

        let json = r#"{"type":"response.failed",
                       "response":{"status":"failed",
                                   "error":{"code":"server_error","message":"boom"},"output":[]}}"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, ResponsesStreamEvent::Failed { .. }));
        let response = expect_terminal_response(event);
        assert_eq!(response.error.unwrap().message, "boom");

        let json =
            r#"{"type":"error","code":"rate_limit_exceeded","message":"slow down","param":null}"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            ResponsesStreamEvent::Error { ref message, ref code }
                if message == "slow down" && code.as_deref() == Some("rate_limit_exceeded")
        ));
    }

    #[test]
    fn deserialize_unconsumed_event_types_as_other() {
        // The bookkeeping events the normalization skips: creation, progress,
        // content-part brackets, done-recaps — and anything future.
        for json in [
            r#"{"type":"response.created","response":{"status":"in_progress","output":[]}}"#,
            r#"{"type":"response.in_progress","response":{"status":"in_progress","output":[]}}"#,
            r#"{"type":"response.content_part.added","content_index":0,"item_id":"m","output_index":1,"part":{"type":"output_text","text":""}}"#,
            r#"{"type":"response.output_text.done","content_index":0,"item_id":"m","output_index":1,"text":"1\n2"}"#,
            r#"{"type":"response.function_call_arguments.done","arguments":"{}","item_id":"f","output_index":0}"#,
            r#"{"type":"response.output_item.done","item":{"id":"m","type":"message","content":[]},"output_index":1}"#,
            r#"{"type":"response.reasoning_summary_text.delta","delta":"…","output_index":0}"#,
        ] {
            let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
            assert!(matches!(event, ResponsesStreamEvent::Other), "for {json}");
        }
    }
}
