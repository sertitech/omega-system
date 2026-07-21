//! The live half of the OpenAI transport: the [`Client`] that builds and sends
//! real `ureq` requests, plus the `#[ignore]`'d live-API tests. Nothing here is
//! reachable without network access and a real key, so the `_live.rs` suffix
//! marks the file for wholesale exclusion from the coverage gate — anything
//! beyond request construction and live tests belongs in a covered module (the
//! SSE/event parsing lives in [`super::client`]).

use crate::openai::client::{ResponsesStreamIterator, parse_model_ids};
use crate::openai::responses_types::{ResponsesRequest, ResponsesResponse};
use crate::provider::ApiError;
use crate::provider::transport;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const RESPONSES_URL: &str = "https://api.openai.com/v1/responses";
const MODELS_URL: &str = "https://api.openai.com/v1/models";

pub struct Client {
    api_key: String,
    /// Bounds the whole non-streaming `send_responses()` call.
    blocking: ureq::Agent,
    /// Bounds connection setup and time-to-first-byte for
    /// `stream_responses()`, but leaves the response body unbounded so a long
    /// turn is never cut off mid-stream.
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

    /// Send a non-streaming Responses API request — the endpoint that serves
    /// OpenAI's whole current catalog, including the ids Chat Completions
    /// rejected (`gpt-5.5-pro`, `gpt-5.3-codex`).
    pub fn send_responses(
        &self,
        request: &ResponsesRequest,
    ) -> Result<ResponsesResponse, ApiError> {
        let body = serde_json::to_string(request)?;

        let response = transport::execute_with_retry(&self.cancel, || {
            self.blocking
                .post(RESPONSES_URL)
                .header("authorization", &format!("Bearer {}", self.api_key))
                .header("content-type", "application/json")
                .send(body.as_bytes())
        })?;

        let body = response.into_body().read_to_string()?;
        let parsed: ResponsesResponse = serde_json::from_str(&body)?;
        Ok(parsed)
    }

    /// Send a streaming Responses request. Sets `stream`, then returns an
    /// iterator of parsed semantic events (the stream ends with the terminal
    /// `response.*` event — no sentinel).
    pub fn stream_responses(
        &self,
        request: &ResponsesRequest,
    ) -> Result<ResponsesStreamIterator, ApiError> {
        let mut streaming_request = request.clone();
        streaming_request.stream = Some(true);
        let body = serde_json::to_string(&streaming_request)?;

        let response = transport::execute_with_retry(&self.cancel, || {
            self.streaming
                .post(RESPONSES_URL)
                .header("authorization", &format!("Bearer {}", self.api_key))
                .header("content-type", "application/json")
                .send(body.as_bytes())
        })?;

        Ok(ResponsesStreamIterator::over(Box::new(
            response.into_body().into_reader(),
        )))
    }

    /// Fetch every model id the API serves — one unpaginated page. The body
    /// parsing lives (covered) in [`parse_model_ids`]; only this GET is live.
    pub fn list_models(&self) -> Result<Vec<String>, ApiError> {
        let response = transport::execute_with_retry(&self.cancel, || {
            self.blocking
                .get(MODELS_URL)
                .header("authorization", &format!("Bearer {}", self.api_key))
                .call()
        })?;
        parse_model_ids(&response.into_body().read_to_string()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::responses_types::{
        ContentPart, InputItem, OutputItem, ResponsesStreamEvent,
    };

    fn api_key() -> Option<String> {
        crate::load_env_var("OPENAI_API_KEY")
    }

    fn responses_ping(model: &str) -> ResponsesRequest {
        ResponsesRequest {
            model: model.to_string(),
            instructions: None,
            input: vec![InputItem::Message {
                role: "user".to_string(),
                content: "Reply with exactly: hello".to_string(),
            }],
            max_output_tokens: 128,
            tools: None,
            stream: None,
            store: false,
            reasoning: None,
        }
    }

    /// The `output_text` of the response's message items, concatenated.
    fn response_text(response: &ResponsesResponse) -> String {
        response
            .output
            .iter()
            .filter_map(|item| match item {
                OutputItem::Message { content } => Some(content),
                _ => None,
            })
            .flatten()
            .filter_map(|part| match part {
                ContentPart::OutputText { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn client_creation() {
        let client = Client::new("sk-test-key".to_string(), Arc::default());
        assert_eq!(client.api_key, "sk-test-key");
    }

    #[test]
    #[ignore = "hits the live OpenAI API; run with --ignored"]
    fn send_responses_simple_message() {
        let Some(key) = api_key() else {
            eprintln!("OPENAI_API_KEY not set, skipping integration test");
            return;
        };

        let response = Client::new(key, Arc::default())
            .send_responses(&responses_ping(crate::TEST_MODEL_OPENAI))
            .unwrap();
        assert_eq!(response.status, "completed");
        assert!(response.error.is_none());
        assert!(response_text(&response).to_lowercase().contains("hello"));
        assert!(response.usage.unwrap().output_tokens > 0);
    }

    #[test]
    #[ignore = "hits the live OpenAI API; run with --ignored"]
    fn send_responses_reaches_a_responses_only_id() {
        // The headline reach: an id Chat Completions cannot serve at all
        // answers here.
        let Some(key) = api_key() else {
            eprintln!("OPENAI_API_KEY not set, skipping integration test");
            return;
        };

        let response = Client::new(key, Arc::default())
            .send_responses(&responses_ping(crate::TEST_MODEL_OPENAI_RESPONSES_ONLY))
            .unwrap();
        assert_eq!(response.status, "completed");
        assert!(!response_text(&response).is_empty());
    }

    #[test]
    #[ignore = "hits the live OpenAI API; run with --ignored"]
    fn stream_responses_simple_message() {
        let Some(key) = api_key() else {
            eprintln!("OPENAI_API_KEY not set, skipping integration test");
            return;
        };

        let events: Vec<ResponsesStreamEvent> = Client::new(key, Arc::default())
            .stream_responses(&responses_ping(crate::TEST_MODEL_OPENAI))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(!events.is_empty());
        // Text arrives as deltas and the stream closes on the terminal
        // completed event carrying usage.
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                ResponsesStreamEvent::OutputTextDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert!(!text.is_empty(), "expected streamed text, got nothing");
        let Some(ResponsesStreamEvent::Completed { response }) = events.last() else {
            panic!("expected a trailing completed event");
        };
        assert!(response.usage.as_ref().unwrap().output_tokens > 0);
    }

    #[test]
    #[ignore = "hits the live OpenAI API; run with --ignored"]
    fn list_models_returns_current_ids() {
        let Some(key) = api_key() else {
            eprintln!("OPENAI_API_KEY not set, skipping integration test");
            return;
        };

        let ids = Client::new(key, Arc::default()).list_models().unwrap();
        // The listing must include the model the live tests pin — also
        // confirming the documented response shape (data[].id).
        assert!(
            ids.iter().any(|id| id == crate::TEST_MODEL_OPENAI),
            "expected {} in the live listing, got {} ids",
            crate::TEST_MODEL_OPENAI,
            ids.len()
        );
    }

    #[test]
    #[ignore = "makes a live request to the OpenAI API; run with --ignored"]
    fn invalid_api_key_returns_error() {
        let client = Client::new("sk-invalid-key".to_string(), Arc::default());
        assert!(
            client
                .send_responses(&responses_ping(crate::TEST_MODEL_OPENAI))
                .is_err()
        );
    }
}
