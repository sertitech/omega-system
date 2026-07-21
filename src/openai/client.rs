//! The hermetic half of the OpenAI transport: the Responses SSE dialect for
//! streamed responses, exercised entirely over in-memory bodies, plus the
//! models-listing parse. The `ureq` request construction and the live-API
//! tests live in [`super::client_live`], which the coverage gate excludes
//! wholesale.
//!
//! The streaming dialect differs from Anthropic's in one way the iterator
//! absorbs: there is no terminating sentinel (Anthropic ends on
//! `message_stop`; the retired Chat Completions dialect ended on
//! `data: [DONE]`) — a Responses stream simply closes after its terminal
//! `response.*` event, so iteration runs to end-of-body. Each event's
//! `data:` payload carries its own `type` field, which
//! [`ResponsesStreamEvent`]'s tagged enum keys on — the SSE `event:` line
//! duplicates it and is ignored.

use crate::openai::responses_types::ResponsesStreamEvent;
use crate::provider::ApiError;
use crate::sse::SseParser;
use serde::Deserialize;

/// Iterator over streaming Responses events. Yields one parsed
/// [`ResponsesStreamEvent`] per `data:` payload, skipping keepalives, until
/// the body ends.
pub struct ResponsesStreamIterator {
    parser: SseParser<Box<dyn std::io::Read + Send + Sync>>,
}

impl ResponsesStreamIterator {
    /// Parse Responses SSE events out of `reader` — the response body on the
    /// live path, an in-memory cursor in the hermetic tests. Boxed (not
    /// generic) so there is exactly one instantiation, fully covered here.
    pub(crate) fn over(reader: Box<dyn std::io::Read + Send + Sync>) -> Self {
        Self {
            parser: SseParser::new(reader),
        }
    }
}

impl Iterator for ResponsesStreamIterator {
    type Item = Result<ResponsesStreamEvent, ApiError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.parser.next()? {
                Err(e) => return Some(Err(ApiError::Io(e))),
                Ok(sse_event) => {
                    // Skip keepalives (blank data).
                    if sse_event.data.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<ResponsesStreamEvent>(&sse_event.data) {
                        Ok(event) => return Some(Ok(event)),
                        Err(e) => return Some(Err(ApiError::Json(e))),
                    }
                }
            }
        }
    }
}

// ── Model listing ──

/// The `GET /v1/models` response — unpaginated, unlike Anthropic's. The
/// listing includes non-chat ids (embeddings, TTS) and dated snapshots;
/// [`crate::repl`]'s curation hides those from the completion surfaces.
#[derive(Debug, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<ModelEntry>,
}

/// A single model in the listing: the id that feeds the REPL's model-id
/// completion, plus the release epoch that orders the listing newest-first.
/// `created` is defaulted, not required — a missing timestamp sorts the
/// entry last rather than failing the whole listing.
#[derive(Debug, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    #[serde(default)]
    pub created: i64,
}

/// Parse a `GET /v1/models` body into its id list, newest release first.
/// Hermetic counterpart of [`super::client_live::Client::list_models`],
/// which supplies the live body.
///
/// The endpoint's own order is not reliably chronological, so the entries
/// are re-sorted by `created` descending. The sort is stable: ties — and
/// entries whose `created` is absent, which default to 0 — keep the
/// server's relative order.
pub fn parse_model_ids(body: &str) -> Result<Vec<String>, ApiError> {
    let response: ModelsResponse = serde_json::from_str(body)?;
    let mut entries = response.data;
    entries.sort_by(|a, b| b.created.cmp(&a.created));
    Ok(entries.into_iter().map(|entry| entry.id).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a [`ResponsesStreamIterator`] over an in-memory SSE body so the
    /// parsing dialect (event/data pairing, skip-empty, run-to-EOF, JSON
    /// errors) is covered without the network.
    fn stream_over(body: &str) -> ResponsesStreamIterator {
        ResponsesStreamIterator::over(Box::new(std::io::Cursor::new(body.as_bytes().to_vec())))
    }

    #[test]
    fn stream_iterator_parses_events_to_end_of_body() {
        // The live dialect: `event:` lines name the type, the JSON `type`
        // field repeats it (the parse keys on the JSON), and the stream just
        // ends after the terminal event — no sentinel.
        let body = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"status\":\"in_progress\",\"output\":[]}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\",\"output_index\":0}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[]}}\n\n",
        );
        let events: Vec<ResponsesStreamEvent> =
            stream_over(body).collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], ResponsesStreamEvent::Other));
        assert!(matches!(
            events[1],
            ResponsesStreamEvent::OutputTextDelta { ref delta, .. } if delta == "Hi"
        ));
        assert!(matches!(events[2], ResponsesStreamEvent::Completed { .. }));
    }

    #[test]
    fn stream_iterator_skips_empty_keepalive_data() {
        // A comment line is dropped inside the SSE parser; a bare `data:`
        // line yields an event with *empty* data, which the iterator must
        // skip rather than fail to parse.
        let body = concat!(
            ": keepalive\n\n",
            "data:\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"x\",\"output_index\":0}\n\n",
        );
        let events: Vec<ResponsesStreamEvent> =
            stream_over(body).collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn stream_iterator_surfaces_malformed_json() {
        let body = "data: {not json}\n\n";
        let mut iter = stream_over(body);
        assert!(matches!(iter.next(), Some(Err(ApiError::Json(_)))));
    }

    #[test]
    fn stream_iterator_surfaces_io_error() {
        // A reader that errors mid-stream must surface as ApiError::Io, not
        // be silently swallowed.
        struct FailingReader;
        impl std::io::Read for FailingReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("boom"))
            }
        }
        let mut iter = ResponsesStreamIterator::over(Box::new(FailingReader));
        assert!(matches!(iter.next(), Some(Err(ApiError::Io(_)))));
    }

    // ── parse_model_ids ──

    #[test]
    fn parse_model_ids_sorts_newest_first() {
        // The server's order (oldest first here) is discarded in favor of
        // `created` descending.
        let body = r#"{"object":"list","data":[{"id":"gpt-4o","object":"model","created":1715000000,"owned_by":"openai"},{"id":"gpt-5.4","object":"model","created":1741000000,"owned_by":"openai"}]}"#;
        assert_eq!(parse_model_ids(body).unwrap(), ["gpt-5.4", "gpt-4o"]);
    }

    #[test]
    fn parse_model_ids_keeps_server_order_on_ties_and_missing_created() {
        // A missing `created` defaults to 0 rather than failing the listing,
        // and the stable sort preserves the server's relative order for it.
        let body = r#"{"object":"list","data":[{"id":"gpt-4o","object":"model","owned_by":"openai"},{"id":"gpt-4o-mini","object":"model","owned_by":"openai"}]}"#;
        assert_eq!(parse_model_ids(body).unwrap(), ["gpt-4o", "gpt-4o-mini"]);
    }

    #[test]
    fn parse_model_ids_empty_listing_is_empty() {
        assert_eq!(
            parse_model_ids(r#"{"object":"list","data":[]}"#).unwrap(),
            Vec::<String>::new()
        );
    }

    #[test]
    fn parse_model_ids_surfaces_malformed_bodies() {
        assert!(matches!(
            parse_model_ids("{not json}"),
            Err(ApiError::Json(_))
        ));
    }
}
