use serde::{Deserialize, Serialize};

// ── Cache control ──

/// Marks a content block as a cache breakpoint. The API caches everything from
/// the start of the prompt up to this block, with a 5-minute TTL.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum CacheControl {
    #[serde(rename = "ephemeral")]
    Ephemeral,
}

// ── Request types ──

#[derive(Debug, Clone, Serialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemContent>,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Opt-in reasoning effort. Omitted entirely when absent, so a no-effort
    /// request is byte-identical to today's; set, it carries the opaque effort
    /// string the API validates (`output_config: {"effort": ...}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
}

/// The Anthropic `output_config` object — carries the reasoning `effort` in
/// the Messages API's wire location. The value is an opaque string validated
/// server-side (see [`crate::turn::TurnRequest::effort`]).
#[derive(Debug, Clone, Serialize)]
pub struct OutputConfig {
    pub effort: String,
}

/// A tool definition sent to the API in the `tools` array.
#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// System prompt — either a plain string or an array of blocks (needed for cache_control).
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum SystemContent {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

/// A single block in a system prompt array. Always type "text".
#[derive(Debug, Clone, Serialize)]
pub struct SystemBlock {
    r#type: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl SystemBlock {
    pub fn new(text: String) -> Self {
        Self {
            r#type: "text".to_string(),
            text,
            cache_control: None,
        }
    }

    pub fn cached(text: String) -> Self {
        Self {
            r#type: "text".to_string(),
            text,
            cache_control: Some(CacheControl::Ephemeral),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

// ── Response types ──

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<StopReason>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    /// Any stop reason this build does not know (`refusal`, `pause_turn`, …).
    /// Newer models add reasons over time; an unknown one must not crash the
    /// turn, so it deserializes here and normalizes as an ordinary end of turn.
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    // The documented minimal `message_delta` shape omits `input_tokens` (and
    // may omit `output_tokens`); default them to 0 so a partial `usage` object
    // still deserializes rather than failing the whole turn. The live API
    // currently sends both, so this is hardening, not a live bug.
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

// ── Content blocks ──

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// The model's extended-thinking reasoning. Parsed so it can be resent
    /// verbatim — `signature` included — with the assistant message on
    /// tool-use turns, per the API's multi-turn thinking contract. Carries
    /// no `cache_control`: the API rejects cache markers on thinking blocks.
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    /// Any block type this build does not know (e.g. `redacted_thinking`).
    /// Its payload is discarded at parse time; the adapter drops the block
    /// from the normalized turn rather than crashing it. Never serialized —
    /// requests are built from normalized blocks, which have no unknown kind.
    #[serde(other)]
    Unknown,
}

// ── Streaming event types ──

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessagesResponse },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u32,
        content_block: ContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: u32, delta: Delta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u32 },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaBody,
        usage: Usage,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum Delta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    /// Reasoning text extending a streamed thinking block.
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    /// A signature fragment closing a streamed thinking block.
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
    /// Any delta type this build does not know. Dropped by the adapter.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MessageDeltaBody {
    pub stop_reason: Option<StopReason>,
}

// ── Model listing ──

/// One page of `GET /v1/models`. The endpoint is paginated (default page size
/// is small), so a page carries a continuation cursor alongside its entries;
/// [`super::client::collect_model_ids`] walks the cursor chain.
#[derive(Debug, Deserialize)]
pub struct ModelsPage {
    pub data: Vec<ModelEntry>,
    pub has_more: bool,
    /// Cursor for the next page (`after_id`); absent on an empty listing.
    pub last_id: Option<String>,
}

/// A single model in the listing: the id that feeds the REPL's model-id
/// completion, plus the RFC 3339 release timestamp that orders the drained
/// listing newest-first (uniform stamps compare chronologically as plain
/// strings, so no date parsing is needed). `created_at` is defaulted, not
/// required — a missing timestamp sorts the entry last rather than failing
/// the whole listing.
#[derive(Debug, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    #[serde(default)]
    pub created_at: String,
}

// ── Helpers ──

impl MessagesResponse {
    /// Extracts the first text content from the response, if any.
    pub fn text(&self) -> Option<&str> {
        self.content.iter().find_map(|block| match block {
            ContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Assertion helpers ──
    // Each unwraps one wire variant, panicking with the actual value otherwise.
    // The panic arms are exercised by the `#[should_panic]` tests below, so no
    // helper carries a dead line — unlike a per-call-site `match … => panic!`.

    #[track_caller]
    fn expect_text_block(block: ContentBlock) -> (String, Option<CacheControl>) {
        match block {
            ContentBlock::Text {
                text,
                cache_control,
            } => (text, cache_control),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[track_caller]
    fn expect_tool_use_block(block: &ContentBlock) -> (&str, &str, &serde_json::Value) {
        match block {
            ContentBlock::ToolUse {
                id, name, input, ..
            } => (id, name, input),
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[track_caller]
    fn expect_tool_result_block(block: ContentBlock) -> (String, String, Option<bool>) {
        match block {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
                ..
            } => (tool_use_id, content, is_error),
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[track_caller]
    fn expect_message_start(event: StreamEvent) -> MessagesResponse {
        match event {
            StreamEvent::MessageStart { message } => message,
            other => panic!("expected MessageStart, got {other:?}"),
        }
    }

    #[track_caller]
    fn expect_block_delta(event: StreamEvent) -> (u32, Delta) {
        match event {
            StreamEvent::ContentBlockDelta { index, delta } => (index, delta),
            other => panic!("expected ContentBlockDelta, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "expected Text")]
    fn expect_text_block_panics_on_other_variant() {
        expect_text_block(ContentBlock::ToolUse {
            id: "t1".to_string(),
            name: "x".to_string(),
            input: serde_json::json!({}),
            cache_control: None,
        });
    }

    #[test]
    #[should_panic(expected = "expected ToolUse")]
    fn expect_tool_use_block_panics_on_other_variant() {
        expect_tool_use_block(&ContentBlock::Text {
            text: "x".to_string(),
            cache_control: None,
        });
    }

    #[test]
    #[should_panic(expected = "expected ToolResult")]
    fn expect_tool_result_block_panics_on_other_variant() {
        expect_tool_result_block(ContentBlock::Text {
            text: "x".to_string(),
            cache_control: None,
        });
    }

    #[test]
    #[should_panic(expected = "expected MessageStart")]
    fn expect_message_start_panics_on_other_variant() {
        expect_message_start(StreamEvent::MessageStop);
    }

    #[test]
    #[should_panic(expected = "expected ContentBlockDelta")]
    fn expect_block_delta_panics_on_other_variant() {
        expect_block_delta(StreamEvent::MessageStop);
    }

    // ── Request serialization ──

    #[test]
    fn serialize_simple_request() {
        let req = MessagesRequest {
            model: crate::TEST_MODEL.to_string(),
            max_tokens: 1024,
            system: None,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Hello".to_string()),
            }],
            tools: None,
            stream: None,
            output_config: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], crate::TEST_MODEL);
        assert_eq!(json["max_tokens"], 1024);
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "Hello");
        assert!(json.get("system").is_none());
        assert!(json.get("stream").is_none());
    }

    #[test]
    fn serialize_request_with_system_text() {
        let req = MessagesRequest {
            model: crate::TEST_MODEL.to_string(),
            max_tokens: 512,
            system: Some(SystemContent::Text("You are helpful.".to_string())),
            messages: vec![],
            tools: None,
            stream: Some(true),
            output_config: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["system"], "You are helpful.");
        assert_eq!(json["stream"], true);
    }

    // ── Cache control ──

    #[test]
    fn serialize_cache_control() {
        let cc = CacheControl::Ephemeral;
        let json = serde_json::to_value(&cc).unwrap();
        assert_eq!(json, serde_json::json!({"type": "ephemeral"}));
    }

    #[test]
    fn deserialize_cache_control() {
        let json = r#"{"type": "ephemeral"}"#;
        let cc: CacheControl = serde_json::from_str(json).unwrap();
        assert_eq!(cc, CacheControl::Ephemeral);
    }

    #[test]
    fn serialize_system_blocks_cached() {
        let system =
            SystemContent::Blocks(vec![SystemBlock::cached("You are helpful.".to_string())]);
        let json = serde_json::to_value(&system).unwrap();
        assert_eq!(json[0]["type"], "text");
        assert_eq!(json[0]["text"], "You are helpful.");
        assert_eq!(json[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn serialize_system_blocks_uncached() {
        let system = SystemContent::Blocks(vec![SystemBlock::new("No caching.".to_string())]);
        let json = serde_json::to_value(&system).unwrap();
        assert_eq!(json[0]["type"], "text");
        assert_eq!(json[0]["text"], "No caching.");
        assert!(json[0].get("cache_control").is_none());
    }

    #[test]
    fn serialize_content_block_with_cache() {
        let block = ContentBlock::Text {
            text: "Hello".to_string(),
            cache_control: Some(CacheControl::Ephemeral),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "Hello");
        assert_eq!(json["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn serialize_content_block_without_cache() {
        let block = ContentBlock::Text {
            text: "Hello".to_string(),
            cache_control: None,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "Hello");
        assert!(json.get("cache_control").is_none());
    }

    #[test]
    fn deserialize_content_block_without_cache_field() {
        let json = r#"{"type": "text", "text": "Hello"}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        let (text, cache_control) = expect_text_block(block);
        assert_eq!(text, "Hello");
        assert_eq!(cache_control, None);
    }

    #[test]
    fn deserialize_content_block_with_cache_field() {
        let json = r#"{"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        let (text, cache_control) = expect_text_block(block);
        assert_eq!(text, "Hello");
        assert_eq!(cache_control, Some(CacheControl::Ephemeral));
    }

    // ── Usage with cache fields ──

    #[test]
    fn deserialize_usage_without_cache_fields() {
        let json = r#"{"input_tokens": 10, "output_tokens": 5}"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.cache_creation_input_tokens, None);
        assert_eq!(usage.cache_read_input_tokens, None);
    }

    #[test]
    fn deserialize_minimal_message_delta_usage_without_input_tokens() {
        // The documented minimal `message_delta` shape carries only the
        // output counter; `input_tokens` must default rather than fail parsing.
        let json = r#"{"output_tokens": 42}"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 42);
        assert_eq!(usage.cache_creation_input_tokens, None);
        assert_eq!(usage.cache_read_input_tokens, None);
    }

    #[test]
    fn deserialize_empty_usage_object_defaults_all_fields() {
        let usage: Usage = serde_json::from_str("{}").unwrap();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
    }

    #[test]
    fn deserialize_usage_with_cache_fields() {
        let json = r#"{
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 1000,
            "cache_read_input_tokens": 500
        }"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.cache_creation_input_tokens, Some(1000));
        assert_eq!(usage.cache_read_input_tokens, Some(500));
    }

    // ── Response deserialization ──

    #[test]
    fn deserialize_text_response() {
        let json = r#"{
            "id": "msg_123",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-5-20250929",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;
        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "msg_123");
        assert_eq!(resp.role, Role::Assistant);
        assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
        assert_eq!(resp.text(), Some("Hello!"));
        assert_eq!(resp.usage.input_tokens, 10);
    }

    #[test]
    fn deserialize_tool_use_response() {
        let json = r#"{
            "id": "msg_456",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"location": "SF"}}
            ],
            "model": "claude-sonnet-4-5-20250929",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15}
        }"#;
        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
        assert_eq!(resp.content.len(), 2);
        let (id, name, input) = expect_tool_use_block(&resp.content[1]);
        assert_eq!(id, "toolu_abc");
        assert_eq!(name, "get_weather");
        assert_eq!(input["location"], "SF");
    }

    // ── Thinking blocks: known wire values ──
    // claude-fable-5 (and every extended-thinking model) returns `thinking`
    // blocks by default; both directions of the wire shape are pinned here —
    // the resend must be byte-compatible with what the API sent.

    #[test]
    fn deserialize_thinking_content_block() {
        let json = r#"{"type": "thinking", "thinking": "hmm", "signature": "sig"}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        assert!(matches!(
            block,
            ContentBlock::Thinking { thinking, signature }
                if thinking == "hmm" && signature == "sig"
        ));
    }

    #[test]
    fn serialize_thinking_content_block_verbatim() {
        // The resend shape the multi-turn contract requires: exactly the
        // three fields the API sent, nothing else (no cache_control slot).
        let block = ContentBlock::Thinking {
            thinking: "hmm".to_string(),
            signature: "sig".to_string(),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"type": "thinking", "thinking": "hmm", "signature": "sig"})
        );
    }

    #[test]
    fn deserialize_response_with_thinking_block() {
        // The non-streaming send path: a thinking-first response parses with
        // both the reasoning and the answer intact.
        let json = r#"{
            "id": "msg_1",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "let me think", "signature": "sig"},
                {"type": "text", "text": "Hello!"}
            ],
            "model": "claude-fable-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;
        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert!(matches!(
            &resp.content[0],
            ContentBlock::Thinking { thinking, signature }
                if thinking == "let me think" && signature == "sig"
        ));
        assert_eq!(resp.text(), Some("Hello!"));
    }

    #[test]
    fn deserialize_stream_thinking_block_start() {
        let json = r#"{
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""}
        }"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            StreamEvent::ContentBlockStart {
                index: 0,
                content_block: ContentBlock::Thinking { thinking, signature }
            } if thinking.is_empty() && signature.is_empty()
        ));
    }

    #[test]
    fn deserialize_stream_thinking_and_signature_deltas() {
        let json = r#"{"type": "content_block_delta", "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hmm"}}"#;
        let (index, delta) = expect_block_delta(serde_json::from_str(json).unwrap());
        assert_eq!(index, 0);
        assert!(matches!(delta, Delta::ThinkingDelta { thinking } if thinking == "hmm"));

        let json = r#"{"type": "content_block_delta", "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"}}"#;
        let (index, delta) = expect_block_delta(serde_json::from_str(json).unwrap());
        assert_eq!(index, 0);
        assert!(matches!(delta, Delta::SignatureDelta { signature } if signature == "sig"));
    }

    // ── Forward compatibility: unknown wire values ──
    // Newer models emit block types, delta types, and stop reasons this build
    // does not know — e.g. `redacted_thinking` blocks, or a `refusal` stop.
    // Each closed enum must fold them into its catch-all instead of failing
    // the whole parse.

    #[test]
    fn deserialize_unknown_stop_reason() {
        let reason: StopReason = serde_json::from_str(r#""refusal""#).unwrap();
        assert_eq!(reason, StopReason::Other);
    }

    #[test]
    fn deserialize_unknown_content_block() {
        let json = r#"{"type": "redacted_thinking", "data": "opaque"}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        assert!(matches!(block, ContentBlock::Unknown));
    }

    #[test]
    fn deserialize_response_with_unknown_block_and_stop_reason() {
        // Both unknowns in one response shape must leave the known content
        // readable.
        let json = r#"{
            "id": "msg_1",
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "Hello!"}
            ],
            "model": "claude-fable-5",
            "stop_reason": "refusal",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;
        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert!(matches!(resp.content[0], ContentBlock::Unknown));
        assert_eq!(resp.text(), Some("Hello!"));
        assert_eq!(resp.stop_reason, Some(StopReason::Other));
    }

    #[test]
    fn deserialize_stream_unknown_delta() {
        let json = r#"{"type": "content_block_delta", "index": 0,
            "delta": {"type": "some_future_delta", "payload": "x"}}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        let (index, delta) = expect_block_delta(event);
        assert_eq!(index, 0);
        assert!(matches!(delta, Delta::Unknown));
    }

    #[test]
    fn deserialize_stream_message_delta_with_unknown_stop_reason() {
        let json = r#"{
            "type": "message_delta",
            "delta": {"stop_reason": "pause_turn"},
            "usage": {"input_tokens": 0, "output_tokens": 1}
        }"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            StreamEvent::MessageDelta { delta, .. }
                if delta.stop_reason == Some(StopReason::Other)
        ));
    }

    // ── Streaming events ──

    #[test]
    fn deserialize_stream_message_start() {
        let json = r#"{
            "type": "message_start",
            "message": {
                "id": "msg_789",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": null,
                "usage": {"input_tokens": 10, "output_tokens": 1}
            }
        }"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        let message = expect_message_start(event);
        assert_eq!(message.id, "msg_789");
        assert_eq!(message.stop_reason, None);
    }

    #[test]
    fn deserialize_stream_message_start_with_cache_usage() {
        let json = r#"{
            "type": "message_start",
            "message": {
                "id": "msg_cached",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": null,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "cache_creation_input_tokens": 500,
                    "cache_read_input_tokens": 0
                }
            }
        }"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        let message = expect_message_start(event);
        assert_eq!(message.usage.cache_creation_input_tokens, Some(500));
        assert_eq!(message.usage.cache_read_input_tokens, Some(0));
    }

    #[test]
    fn deserialize_stream_text_delta() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}
        }"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        let (index, delta) = expect_block_delta(event);
        assert_eq!(index, 0);
        assert!(matches!(delta, Delta::TextDelta { text } if text == "Hello"));
    }

    #[test]
    fn deserialize_stream_input_json_delta() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{\"location\":"}
        }"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        let (index, delta) = expect_block_delta(event);
        assert_eq!(index, 1);
        assert!(matches!(
            delta,
            Delta::InputJsonDelta { partial_json } if partial_json == "{\"location\":"
        ));
    }

    #[test]
    fn deserialize_stream_message_delta() {
        let json = r#"{
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 0, "output_tokens": 42}
        }"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            StreamEvent::MessageDelta { delta, usage }
                if delta.stop_reason == Some(StopReason::EndTurn) && usage.output_tokens == 42
        ));
    }

    #[test]
    fn deserialize_stream_message_stop() {
        let json = r#"{"type": "message_stop"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::MessageStop));
    }

    // ── MessageContent roundtrips ──

    #[test]
    fn message_content_text_roundtrip() {
        let msg = Message {
            role: Role::User,
            content: MessageContent::Text("hi".to_string()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, Role::User);
        assert!(matches!(parsed.content, MessageContent::Text(t) if t == "hi"));
    }

    #[test]
    fn message_content_blocks_roundtrip() {
        let msg = Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "toolu_abc".to_string(),
                content: "72\u{00b0}F".to_string(),
                is_error: None,
                cache_control: None,
            }]),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("tool_result"));
        assert!(json.contains("toolu_abc"));
    }

    // ── Helpers ──

    #[test]
    fn text_helper_returns_none_for_no_text() {
        let resp = MessagesResponse {
            id: "msg_1".to_string(),
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "t1".to_string(),
                name: "foo".to_string(),
                input: serde_json::json!({}),
                cache_control: None,
            }],
            model: crate::TEST_MODEL.to_string(),
            stop_reason: Some(StopReason::ToolUse),
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        assert_eq!(resp.text(), None);
    }

    // ── Tool definitions ──

    #[test]
    fn serialize_tool_definition() {
        let tool = Tool {
            name: "get_weather".to_string(),
            description: "Get current weather".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }),
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["name"], "get_weather");
        assert_eq!(json["description"], "Get current weather");
        assert_eq!(json["input_schema"]["type"], "object");
        assert_eq!(
            json["input_schema"]["properties"]["location"]["type"],
            "string"
        );
    }

    #[test]
    fn serialize_request_with_tools() {
        let req = MessagesRequest {
            model: crate::TEST_MODEL.to_string(),
            max_tokens: 1024,
            system: None,
            messages: vec![],
            tools: Some(vec![Tool {
                name: "echo".to_string(),
                description: "Echo input".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }]),
            stream: None,
            output_config: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["tools"][0]["name"], "echo");
    }

    #[test]
    fn serialize_request_without_tools_omits_field() {
        let req = MessagesRequest {
            model: crate::TEST_MODEL.to_string(),
            max_tokens: 1024,
            system: None,
            messages: vec![],
            tools: None,
            stream: None,
            output_config: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("tools").is_none());
    }

    // ── ToolResult is_error ──

    #[test]
    fn serialize_tool_result_with_is_error() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "toolu_1".to_string(),
            content: "something went wrong".to_string(),
            is_error: Some(true),
            cache_control: None,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_result");
        assert_eq!(json["is_error"], true);
        assert_eq!(json["content"], "something went wrong");
    }

    #[test]
    fn serialize_tool_result_without_is_error_omits_field() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "toolu_1".to_string(),
            content: "ok".to_string(),
            is_error: None,
            cache_control: None,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert!(json.get("is_error").is_none());
    }

    #[test]
    fn deserialize_tool_result_with_is_error() {
        let json = r#"{"type": "tool_result", "tool_use_id": "toolu_1", "content": "failed", "is_error": true}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        let (tool_use_id, content, is_error) = expect_tool_result_block(block);
        assert_eq!(tool_use_id, "toolu_1");
        assert_eq!(content, "failed");
        assert_eq!(is_error, Some(true));
    }

    #[test]
    fn deserialize_tool_result_without_is_error() {
        let json = r#"{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        let (_, _, is_error) = expect_tool_result_block(block);
        assert_eq!(is_error, None);
    }
}
