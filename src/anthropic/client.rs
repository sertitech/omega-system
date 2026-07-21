//! The hermetic half of the Anthropic transport: the SSE event dialect —
//! keepalive skipping, mid-stream `error` events, JSON decoding — exercised
//! entirely over in-memory bodies. The `ureq` request construction and the
//! live-API tests live in [`super::client_live`], which the coverage gate
//! excludes wholesale.

use crate::anthropic::types::{ModelsPage, StreamEvent};
use crate::provider::{ApiError, error_message};
use crate::sse::SseParser;

/// Iterator over streaming Anthropic events.
pub struct StreamIterator {
    parser: SseParser<Box<dyn std::io::Read + Send + Sync>>,
}

impl StreamIterator {
    /// Parse Anthropic SSE events out of `reader` — the response body on the
    /// live path, an in-memory cursor in the hermetic tests. Boxed (not
    /// generic) so there is exactly one instantiation, fully covered here.
    pub(crate) fn over(reader: Box<dyn std::io::Read + Send + Sync>) -> Self {
        Self {
            parser: SseParser::new(reader),
        }
    }
}

impl Iterator for StreamIterator {
    type Item = Result<StreamEvent, ApiError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.parser.next()? {
                Err(e) => return Some(Err(ApiError::Io(e))),
                Ok(sse_event) => {
                    match sse_event.event.as_deref() {
                        // A mid-stream `error` event (e.g. overloaded_error,
                        // api_error) aborts the turn. Surface it instead of
                        // letting it fall through as a silent truncation the
                        // agent would report as a complete answer. Checked
                        // before the empty-data skip below so an error event
                        // with an empty body still aborts rather than being
                        // mistaken for a keepalive and swallowed.
                        Some("error") => return Some(Err(stream_error(&sse_event.data))),
                        // Skip events with empty data (keepalives).
                        _ if sse_event.data.is_empty() => continue,
                        // Skip events we don't consume (e.g. "ping" keepalives).
                        Some(other) if !is_content_event(other) => continue,
                        // A content event, or a bare `data:` line with no
                        // event type — parse it as a stream event.
                        _ => {
                            return Some(
                                match serde_json::from_str::<StreamEvent>(&sse_event.data) {
                                    Ok(event) => Ok(event),
                                    Err(e) => Err(ApiError::Json(e)),
                                },
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The stream events the agent consumes — content blocks and turn metadata.
/// Anything else (a `ping` keepalive) is skipped; the `error` event is handled
/// separately as a failure, not skipped.
fn is_content_event(event: &str) -> bool {
    matches!(
        event,
        "message_start"
            | "content_block_start"
            | "content_block_delta"
            | "content_block_stop"
            | "message_delta"
            | "message_stop"
    )
}

/// Turn the body of a mid-stream `error` event into an [`ApiError::Stream`],
/// reusing the shared [`error_message`] envelope parser — the same one the HTTP
/// non-2xx path uses. A body that doesn't parse still surfaces verbatim, so the
/// failure is never swallowed.
fn stream_error(data: &str) -> ApiError {
    ApiError::Stream(error_message(data))
}

/// Runaway guard on the models-listing pagination walk. At the 1000-per-page
/// `limit` the live client requests, one page covers the real catalog many
/// times over; a server that keeps answering `has_more: true` gets cut off
/// here rather than looping forever. Hitting the cap truncates silently —
/// acceptable for completion data, which is best-effort by design.
const MAX_MODEL_PAGES: usize = 20;

/// Drain the paginated `GET /v1/models` listing into a flat id list, newest
/// release first.
///
/// Seamed over `fetch_page` — which returns one page's raw JSON body for a
/// given `after_id` cursor (`None` for the first page) — so the cursor walk
/// and parsing stay hermetic; the `ureq` request construction is the live
/// half ([`super::client_live::Client::list_models`]). A page that claims
/// more results but carries no cursor cannot be advanced past — the walk
/// returns what it has rather than erroring or spinning.
///
/// The drained entries are re-sorted by `created_at` descending — the API
/// already lists newest-first, but the contract is made explicit rather
/// than trusted. The sort is stable: ties — and entries whose `created_at`
/// is absent, which default to empty — keep the listing's relative order.
pub fn collect_model_ids(
    mut fetch_page: impl FnMut(Option<&str>) -> Result<String, ApiError>,
) -> Result<Vec<String>, ApiError> {
    let mut entries = Vec::new();
    let mut cursor: Option<String> = None;
    for _ in 0..MAX_MODEL_PAGES {
        let page: ModelsPage = serde_json::from_str(&fetch_page(cursor.as_deref())?)?;
        entries.extend(page.data);
        match (page.has_more, page.last_id) {
            (true, Some(id)) => cursor = Some(id),
            _ => break,
        }
    }
    entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(entries.into_iter().map(|entry| entry.id).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::types::*;

    /// Build a [`StreamIterator`] over an in-memory SSE body so the parsing
    /// dialect (error events, keepalive skipping, JSON errors) is covered
    /// without the network.
    fn stream_over(body: &str) -> StreamIterator {
        StreamIterator::over(Box::new(std::io::Cursor::new(body.as_bytes().to_vec())))
    }

    /// Unwrap the stream item as an [`ApiError::Stream`] message, panicking
    /// with the actual item otherwise. The panic arm is exercised by its own
    /// `#[should_panic]` test, so the helper carries no dead line.
    #[track_caller]
    fn expect_stream_err(item: Option<Result<StreamEvent, ApiError>>) -> String {
        match item {
            Some(Err(ApiError::Stream(msg))) => msg,
            other => panic!("expected Stream error, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "expected Stream error")]
    fn expect_stream_err_panics_on_other_item() {
        expect_stream_err(None);
    }

    #[test]
    fn stream_surfaces_error_event() {
        // A mid-stream `error` event must abort with its type and message, not
        // be skipped as an unknown event (the fail-fast bug this fixes).
        let body = concat!(
            "event: error\n",
            "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
        );
        let msg = expect_stream_err(stream_over(body).next());
        assert_eq!(msg, "overloaded_error: Overloaded");
    }

    #[test]
    fn stream_error_event_with_unparseable_body_still_errors() {
        // A malformed error body must still surface as an error carrying the
        // raw payload — never swallowed.
        let body = "event: error\ndata: not json\n\n";
        assert_eq!(expect_stream_err(stream_over(body).next()), "not json");
    }

    #[test]
    fn stream_error_event_with_empty_data_still_aborts() {
        // An `error` event carrying no data must abort the turn, not be
        // mistaken for an empty-data keepalive and skipped (finding F10). The
        // empty body surfaces verbatim through the shared envelope parser.
        let body = "event: error\ndata:\n\n";
        assert_eq!(expect_stream_err(stream_over(body).next()), "");
    }

    #[test]
    fn stream_parses_content_events_and_skips_ping() {
        let body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"role\":\"assistant\",\"content\":[],\"model\":\"x\",\"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
            "event: ping\n",
            "data: {\"type\":\"ping\"}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let events: Vec<StreamEvent> = stream_over(body).collect::<Result<Vec<_>, _>>().unwrap();
        // The ping is dropped; the three content events come through in order.
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], StreamEvent::MessageStart { .. }));
        assert!(matches!(
            &events[1],
            StreamEvent::ContentBlockDelta {
                delta: Delta::TextDelta { text },
                ..
            } if text == "hi"
        ));
        assert!(matches!(events[2], StreamEvent::MessageStop));
    }

    #[test]
    fn stream_skips_empty_data_event() {
        // A `data:` line with an empty value yields an event with empty data,
        // which is skipped (keepalive) rather than parsed.
        let body = concat!(
            "event: ping\ndata:\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );
        let events: Vec<StreamEvent> = stream_over(body).collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::MessageStop));
    }

    #[test]
    fn stream_surfaces_malformed_json() {
        let body = "event: message_start\ndata: {not json}\n\n";
        assert!(matches!(
            stream_over(body).next(),
            Some(Err(ApiError::Json(_)))
        ));
    }

    #[test]
    fn stream_surfaces_io_error() {
        // A reader that errors mid-stream must surface as ApiError::Io, not be
        // silently swallowed.
        struct FailingReader;
        impl std::io::Read for FailingReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("boom"))
            }
        }
        let mut iter = StreamIterator::over(Box::new(FailingReader));
        assert!(matches!(iter.next(), Some(Err(ApiError::Io(_)))));
    }

    // ── collect_model_ids ──

    #[test]
    fn collect_model_ids_single_page() {
        let ids = collect_model_ids(|cursor| {
            assert_eq!(cursor, None);
            Ok(r#"{"data":[{"id":"claude-a"},{"id":"claude-b"}],"has_more":false,"last_id":"claude-b"}"#.to_string())
        })
        .unwrap();
        assert_eq!(ids, ["claude-a", "claude-b"]);
    }

    #[test]
    fn collect_model_ids_follows_the_cursor() {
        // The second fetch must receive the first page's last_id, and the
        // ids must concatenate in page order.
        let ids = collect_model_ids(|cursor| {
            Ok(match cursor {
                None => r#"{"data":[{"id":"claude-a"}],"has_more":true,"last_id":"claude-a"}"#
                    .to_string(),
                Some(id) => {
                    // Both arms execute — an assert, not a dead panic arm,
                    // which the gate would count as a permanently-missed line.
                    assert_eq!(id, "claude-a");
                    r#"{"data":[{"id":"claude-b"}],"has_more":false,"last_id":"claude-b"}"#
                        .to_string()
                }
            })
        })
        .unwrap();
        assert_eq!(ids, ["claude-a", "claude-b"]);
    }

    #[test]
    fn collect_model_ids_sorts_newest_first_across_pages() {
        // An older release on the first page must list after a newer one on
        // the second: the sort runs over the whole drained catalog, not per
        // page. RFC 3339 stamps order chronologically as plain strings.
        let ids = collect_model_ids(|cursor| {
            Ok(match cursor {
                None => r#"{"data":[{"id":"claude-a","created_at":"2025-08-05T00:00:00Z"}],"has_more":true,"last_id":"claude-a"}"#
                    .to_string(),
                Some(_) => r#"{"data":[{"id":"claude-b","created_at":"2026-06-01T00:00:00Z"}],"has_more":false,"last_id":"claude-b"}"#
                    .to_string(),
            })
        })
        .unwrap();
        assert_eq!(ids, ["claude-b", "claude-a"]);
    }

    #[test]
    fn collect_model_ids_stops_on_more_without_cursor() {
        // `has_more` with no cursor cannot be advanced past — the walk keeps
        // what it has instead of erroring or refetching the same page.
        let mut calls = 0;
        let ids = collect_model_ids(|_| {
            calls += 1;
            Ok(r#"{"data":[{"id":"claude-a"}],"has_more":true,"last_id":null}"#.to_string())
        })
        .unwrap();
        assert_eq!(ids, ["claude-a"]);
        assert_eq!(calls, 1);
    }

    #[test]
    fn collect_model_ids_page_cap_stops_a_runaway_listing() {
        // A server that always claims another page is cut off at the cap.
        let mut calls = 0;
        let ids = collect_model_ids(|_| {
            calls += 1;
            Ok(r#"{"data":[{"id":"m"}],"has_more":true,"last_id":"m"}"#.to_string())
        })
        .unwrap();
        assert_eq!(calls, 20);
        assert_eq!(ids.len(), 20);
    }

    #[test]
    fn collect_model_ids_propagates_fetch_errors() {
        let result = collect_model_ids(|_| Err(ApiError::Stream("offline".to_string())));
        assert!(matches!(result, Err(ApiError::Stream(_))));
    }

    #[test]
    fn collect_model_ids_surfaces_malformed_pages() {
        let result = collect_model_ids(|_| Ok("{not json}".to_string()));
        assert!(matches!(result, Err(ApiError::Json(_))));
    }
}
