//! The live half of the Anthropic transport: the [`Client`] that builds and
//! sends real `ureq` requests, plus the `#[ignore]`'d live-API tests. Nothing
//! here is reachable without network access and a real key, so the `_live.rs`
//! suffix marks the file for wholesale exclusion from the coverage gate —
//! anything beyond request construction and live tests belongs in a covered
//! module (the SSE/event parsing lives in [`super::client`]).

use crate::anthropic::client::{StreamIterator, collect_model_ids};
use crate::anthropic::types::{MessagesRequest, MessagesResponse};
use crate::provider::ApiError;
use crate::provider::transport;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const MODELS_URL: &str = "https://api.anthropic.com/v1/models";
const API_VERSION: &str = "2023-06-01";

/// Page size for the models listing — the endpoint's maximum, so the whole
/// catalog realistically arrives in one page and the pagination walk in
/// [`collect_model_ids`] is a safety net, not the common path.
const MODELS_PAGE_LIMIT: &str = "1000";

pub struct Client {
    api_key: String,
    /// Bounds the whole non-streaming `send()` call.
    blocking: ureq::Agent,
    /// Bounds connection setup and time-to-first-byte for `stream()`, but leaves
    /// the response body unbounded so a long turn is never cut off mid-stream.
    streaming: ureq::Agent,
    /// The shared turn-cancellation flag; [`transport::execute_with_retry`]
    /// polls it so a Ctrl-C interrupts retry backoff instead of sleeping
    /// through it.
    cancel: Arc<AtomicBool>,
}

impl Client {
    pub fn new(api_key: String, cancel: Arc<AtomicBool>) -> Self {
        Self {
            api_key,
            blocking: transport::blocking_agent(),
            streaming: transport::streaming_agent(),
            cancel,
        }
    }

    /// Send a non-streaming message request to the Anthropic API.
    pub fn send(&self, request: &MessagesRequest) -> Result<MessagesResponse, ApiError> {
        let body = serde_json::to_string(request)?;

        let response = transport::execute_with_retry(&self.cancel, || {
            self.blocking
                .post(API_URL)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION)
                .header("content-type", "application/json")
                .send(body.as_bytes())
        })?;

        let body = response.into_body().read_to_string()?;
        let parsed: MessagesResponse = serde_json::from_str(&body)?;
        Ok(parsed)
    }

    /// Send a streaming message request. Returns an iterator of parsed SSE events.
    pub fn stream(&self, request: &MessagesRequest) -> Result<StreamIterator, ApiError> {
        let mut streaming_request = request.clone();
        streaming_request.stream = Some(true);
        let body = serde_json::to_string(&streaming_request)?;

        let response = transport::execute_with_retry(&self.cancel, || {
            self.streaming
                .post(API_URL)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION)
                .header("content-type", "application/json")
                .send(body.as_bytes())
        })?;

        Ok(StreamIterator::over(Box::new(
            response.into_body().into_reader(),
        )))
    }

    /// Fetch every model id the API serves, following the pagination cursor.
    /// The cursor walk and page parsing live (covered) in
    /// [`collect_model_ids`]; only this page fetch is live.
    pub fn list_models(&self) -> Result<Vec<String>, ApiError> {
        collect_model_ids(|cursor| {
            let response = transport::execute_with_retry(&self.cancel, || {
                let mut request = self
                    .blocking
                    .get(MODELS_URL)
                    .header("x-api-key", &self.api_key)
                    .header("anthropic-version", API_VERSION)
                    .query("limit", MODELS_PAGE_LIMIT);
                if let Some(after_id) = cursor {
                    request = request.query("after_id", after_id);
                }
                request.call()
            })?;
            Ok(response.into_body().read_to_string()?)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::types::*;

    fn api_key() -> Option<String> {
        crate::load_env_var("ANTHROPIC_API_KEY")
    }

    #[test]
    fn client_creation() {
        let client = Client::new("sk-test-key".to_string(), Arc::default());
        assert_eq!(client.api_key, "sk-test-key");
    }

    #[test]
    #[ignore = "hits the live Anthropic API; run with --ignored"]
    fn send_simple_message() {
        let Some(key) = api_key() else {
            eprintln!("ANTHROPIC_API_KEY not set, skipping integration test");
            return;
        };

        let client = Client::new(key, Arc::default());
        let request = MessagesRequest {
            model: crate::TEST_MODEL.to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Reply with exactly: hello".to_string()),
            }],
            tools: None,
            stream: None,
            output_config: None,
        };

        let response = client.send(&request).unwrap();
        assert_eq!(response.role, Role::Assistant);
        assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
        assert!(response.text().is_some());
        assert!(response.usage.output_tokens > 0);
    }

    #[test]
    #[ignore = "hits the live Anthropic API; run with --ignored"]
    fn send_with_system_prompt() {
        let Some(key) = api_key() else {
            eprintln!("ANTHROPIC_API_KEY not set, skipping integration test");
            return;
        };

        let client = Client::new(key, Arc::default());
        let request = MessagesRequest {
            model: crate::TEST_MODEL.to_string(),
            max_tokens: 64,
            system: Some(SystemContent::Text(
                "You only respond with the word 'pong'.".to_string(),
            )),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("ping".to_string()),
            }],
            tools: None,
            stream: None,
            output_config: None,
        };

        let response = client.send(&request).unwrap();
        let text = response.text().unwrap().to_lowercase();
        assert!(text.contains("pong"), "expected 'pong', got: {text}");
    }

    #[test]
    #[ignore = "hits the live Anthropic API; run with --ignored"]
    fn stream_simple_message() {
        let Some(key) = api_key() else {
            eprintln!("ANTHROPIC_API_KEY not set, skipping integration test");
            return;
        };

        let client = Client::new(key, Arc::default());
        let request = MessagesRequest {
            model: crate::TEST_MODEL.to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Reply with exactly: hi".to_string()),
            }],
            tools: None,
            stream: None,
            output_config: None,
        };

        let stream = client.stream(&request).unwrap();
        let events: Vec<StreamEvent> = stream.collect::<Result<Vec<_>, _>>().unwrap();

        // Must have at least: message_start, content_block_start, >=1 delta, content_block_stop, message_delta, message_stop
        assert!(events.len() >= 5, "got {} events", events.len());

        // First event is message_start.
        assert!(matches!(&events[0], StreamEvent::MessageStart { .. }));

        // Last event is message_stop.
        assert!(matches!(
            &events[events.len() - 1],
            StreamEvent::MessageStop
        ));

        // Accumulate text from deltas.
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockDelta {
                    delta: crate::anthropic::types::Delta::TextDelta { text },
                    ..
                } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(!text.is_empty(), "expected streamed text, got nothing");
    }

    #[test]
    #[ignore = "hits the live Anthropic API; run with --ignored"]
    fn prompt_caching_creates_and_reads_cache() {
        let Some(key) = api_key() else {
            eprintln!("ANTHROPIC_API_KEY not set, skipping integration test");
            return;
        };

        let client = Client::new(key, Arc::default());

        // System prompt must comfortably exceed the minimum cacheable token
        // count (1024 for Sonnet). ~300 repetitions clears it with margin; 150
        // sat right at the threshold and cached inconsistently.
        let system_text = "You are a helpful assistant. ".repeat(300);
        let system = SystemContent::Blocks(vec![SystemBlock::cached(system_text)]);

        let request = MessagesRequest {
            model: crate::TEST_MODEL.to_string(),
            max_tokens: 8,
            system: Some(system),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Reply with exactly: ok".to_string()),
            }],
            tools: None,
            stream: None,
            output_config: None,
        };

        // First request — writes to cache on a cold prefix, or reads it if a
        // prior run primed the same prefix within the 5-minute TTL. Either way
        // the cache must engage.
        let resp1 = client.send(&request).unwrap();
        let created1 = resp1.usage.cache_creation_input_tokens.unwrap_or(0);
        let read1 = resp1.usage.cache_read_input_tokens.unwrap_or(0);
        assert!(
            created1 > 0 || read1 > 0,
            "expected cache creation or read tokens on first request, got neither"
        );

        // Second request — same prefix, must hit the cache.
        let resp2 = client.send(&request).unwrap();
        let read2 = resp2.usage.cache_read_input_tokens.unwrap_or(0);
        assert!(
            read2 > 0,
            "expected cache read tokens on second request, got 0"
        );
    }

    #[test]
    #[ignore = "hits the live Anthropic API; run with --ignored"]
    fn list_models_returns_current_ids() {
        let Some(key) = api_key() else {
            eprintln!("ANTHROPIC_API_KEY not set, skipping integration test");
            return;
        };

        let ids = Client::new(key, Arc::default()).list_models().unwrap();
        // The listing must include the model the test suite itself pins —
        // also confirming the documented response shape (data[].id).
        assert!(
            ids.iter().any(|id| id == crate::TEST_MODEL),
            "expected {} in the live listing, got: {ids:?}",
            crate::TEST_MODEL
        );
    }

    #[test]
    #[ignore = "makes a live request to the Anthropic API; run with --ignored"]
    fn invalid_api_key_returns_error() {
        let client = Client::new("sk-invalid-key".to_string(), Arc::default());
        let request = MessagesRequest {
            model: crate::TEST_MODEL.to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("hello".to_string()),
            }],
            tools: None,
            stream: None,
            output_config: None,
        };

        let result = client.send(&request);
        assert!(result.is_err());
    }
}
