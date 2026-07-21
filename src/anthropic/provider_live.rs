//! The live half of the Anthropic adapter: [`AnthropicProvider`] owns the HTTP
//! [`Client`] and implements [`Provider`] as pure delegation — [`to_wire`] on
//! the way out, [`from_wire`]/[`NormalizedStream`] on the way back, all of
//! which live (tested and covered) in [`super::provider`]. Nothing here is
//! reachable without network access and a real key, so the `_live.rs` suffix
//! marks the file for wholesale exclusion from the coverage gate — anything
//! beyond this delegation and the live tests belongs in a covered module.

use crate::anthropic::client_live::Client;
use crate::anthropic::provider::{NormalizedStream, from_wire, to_wire};
use crate::provider::{ApiError, DeltaStream, Provider};
use crate::turn::{Turn, TurnRequest};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// A [`Provider`] backed by Anthropic's Messages API.
pub struct AnthropicProvider {
    client: Client,
}

impl AnthropicProvider {
    pub fn new(api_key: String, cancel: Arc<AtomicBool>) -> Self {
        Self {
            client: Client::new(api_key, cancel),
        }
    }
}

impl Provider for AnthropicProvider {
    fn send(&self, request: &TurnRequest) -> Result<Turn, ApiError> {
        let response = self.client.send(&to_wire(request))?;
        Ok(from_wire(response))
    }

    fn stream(&self, request: &TurnRequest) -> Result<DeltaStream, ApiError> {
        let events = self.client.stream(&to_wire(request))?;
        Ok(Box::new(NormalizedStream::new(Box::new(events))))
    }

    fn list_models(&self) -> Result<Vec<String>, ApiError> {
        self.client.list_models()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn::{Block, Role, StreamDelta, TurnMessage};

    fn api_key() -> Option<String> {
        crate::load_env_var("ANTHROPIC_API_KEY")
    }

    fn ping_request() -> TurnRequest {
        TurnRequest {
            model: crate::TEST_MODEL.to_string(),
            max_tokens: 16,
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
    #[ignore = "hits the live Anthropic API; run with --ignored"]
    fn provider_send_returns_turn() {
        let Some(key) = api_key() else {
            eprintln!("ANTHROPIC_API_KEY not set, skipping integration test");
            return;
        };
        let turn = AnthropicProvider::new(key, Arc::default())
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
    #[ignore = "hits the live Anthropic API; run with --ignored"]
    fn provider_stream_yields_deltas() {
        let Some(key) = api_key() else {
            eprintln!("ANTHROPIC_API_KEY not set, skipping integration test");
            return;
        };
        let deltas: Vec<StreamDelta> = AnthropicProvider::new(key, Arc::default())
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
}
