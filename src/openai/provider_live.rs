//! The live half of the OpenAI adapter: [`OpenAiProvider`] owns the HTTP
//! [`Client`] and implements [`Provider`] as pure delegation — all mapping
//! lives (tested and covered) in [`super::responses`]. Nothing here is
//! reachable without network access and a real key, so the `_live.rs` suffix
//! marks the file for wholesale exclusion from the coverage gate — anything
//! beyond this delegation and the live tests belongs in a covered module.
//!
//! **There is no endpoint routing.** Both paths
//! speak the Responses API — the endpoint OpenAI serves its whole current
//! catalog on, including the ids Chat Completions rejected outright. No
//! model registry, no id lists, one endpoint; the Chat Completions wire
//! path is retired.

use crate::openai::client_live::Client;
use crate::openai::responses::{NormalizedStream, from_wire, to_wire};
use crate::provider::{ApiError, DeltaStream, Provider};
use crate::turn::{Turn, TurnRequest};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// A [`Provider`] backed by OpenAI's Responses API.
pub struct OpenAiProvider {
    client: Client,
}

impl OpenAiProvider {
    pub fn new(api_key: String, cancel: Arc<AtomicBool>) -> Self {
        Self {
            client: Client::new(api_key, cancel),
        }
    }
}

impl Provider for OpenAiProvider {
    fn send(&self, request: &TurnRequest) -> Result<Turn, ApiError> {
        let response = self.client.send_responses(&to_wire(request)?)?;
        from_wire(response)
    }

    fn stream(&self, request: &TurnRequest) -> Result<DeltaStream, ApiError> {
        let events = self.client.stream_responses(&to_wire(request)?)?;
        Ok(Box::new(NormalizedStream::new(Box::new(events))))
    }

    fn list_models(&self) -> Result<Vec<String>, ApiError> {
        self.client.list_models()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn::{Block, Role, StopReason, StreamDelta, ToolSpec, TurnMessage};

    fn api_key() -> Option<String> {
        crate::load_env_var("OPENAI_API_KEY")
    }

    fn ping_request() -> TurnRequest {
        TurnRequest {
            model: crate::TEST_MODEL_OPENAI.to_string(),
            max_tokens: 128,
            system: None,
            messages: vec![TurnMessage {
                role: Role::User,
                content: vec![Block::Text("Reply with exactly: hi".to_string())],
            }],
            tools: vec![],
            effort: None,
        }
    }

    #[test]
    #[ignore = "hits the live OpenAI API; run with --ignored"]
    fn provider_send_returns_turn() {
        let Some(key) = api_key() else {
            eprintln!("OPENAI_API_KEY not set, skipping integration test");
            return;
        };
        let turn = OpenAiProvider::new(key, Arc::default())
            .send(&ping_request())
            .unwrap();
        assert!(turn.usage.output_tokens > 0);
        assert!(
            turn.blocks
                .iter()
                .any(|b| matches!(b, Block::Text(t) if !t.is_empty()))
        );
    }

    #[test]
    #[ignore = "hits the live OpenAI API; run with --ignored"]
    fn provider_stream_yields_deltas() {
        let Some(key) = api_key() else {
            eprintln!("OPENAI_API_KEY not set, skipping integration test");
            return;
        };
        let deltas: Vec<StreamDelta> = OpenAiProvider::new(key, Arc::default())
            .stream(&ping_request())
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(matches!(
            deltas.first(),
            Some(StreamDelta::MessageStart { .. })
        ));
        assert!(
            deltas
                .iter()
                .any(|d| matches!(d, StreamDelta::MessageDelta { .. }))
        );
    }

    /// A multi-step tool-loop acceptance test against a
    /// Responses-only id, streamed — the model calls the tool, the result
    /// goes back with the call echoed as id-less history items (and no
    /// reasoning items, per the skip treatment), and the follow-up turn
    /// answers from the result.
    #[test]
    #[ignore = "hits the live OpenAI API; run with --ignored"]
    fn provider_tool_loop_round_trips_on_a_responses_only_id() {
        let Some(key) = api_key() else {
            eprintln!("OPENAI_API_KEY not set, skipping integration test");
            return;
        };
        let provider = OpenAiProvider::new(key, Arc::default());
        let tool = ToolSpec {
            name: "get_weather".to_string(),
            description: "Get the current weather for a city".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
        };
        let ask = TurnMessage {
            role: Role::User,
            content: vec![Block::Text(
                "What is the weather in Paris? You must call the get_weather tool.".to_string(),
            )],
        };

        // Step 1: streamed turn ends in a tool call.
        let mut request = TurnRequest {
            model: crate::TEST_MODEL_OPENAI_RESPONSES_ONLY.to_string(),
            max_tokens: 1024,
            system: None,
            messages: vec![ask],
            tools: vec![tool],
            effort: None,
        };
        let deltas: Vec<StreamDelta> = provider
            .stream(&request)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let Some(StreamDelta::ToolUseStart { id, name, .. }) = deltas
            .iter()
            .find(|d| matches!(d, StreamDelta::ToolUseStart { .. }))
        else {
            panic!("expected a tool call, got {deltas:?}");
        };
        assert_eq!(name, "get_weather");
        assert!(matches!(
            deltas.last(),
            Some(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                ..
            })
        ));

        // Step 2: echo the call, carry the result, get the final answer.
        request.messages.push(TurnMessage {
            role: Role::Assistant,
            content: vec![Block::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: serde_json::json!({"city": "Paris"}),
            }],
        });
        request.messages.push(TurnMessage {
            role: Role::User,
            content: vec![Block::ToolResult {
                tool_use_id: id.clone(),
                content: "15C, sunny".to_string(),
                is_error: false,
            }],
        });
        let deltas: Vec<StreamDelta> = provider
            .stream(&request)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let text: String = deltas
            .iter()
            .filter_map(|d| match d {
                StreamDelta::TextStart { text, .. } | StreamDelta::TextDelta { text, .. } => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect();
        assert!(
            text.contains("15") || text.to_lowercase().contains("sunny"),
            "expected the tool result to inform the answer, got: {text}"
        );
    }
}
