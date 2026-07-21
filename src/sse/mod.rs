/// A single Server-Sent Event parsed from an SSE stream.
#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    /// The event type (from the `event:` field). None if not specified.
    pub event: Option<String>,
    /// The data payload (from `data:` fields, joined by newlines).
    pub data: String,
}

/// Hard cap on a single event's accumulated bytes — the `data` lines gathered
/// so far plus the line currently being read. Once response headers arrive the
/// transport deliberately leaves the body unbounded (a long turn must stream
/// to completion), so this is the only thing standing between a broken or
/// hostile endpoint and unbounded memory growth. A real provider event carries
/// one streaming delta (a few KB); even an entire max_tokens=8192 turn packed
/// into a single event would stay under ~1 MiB. 8 MiB is far above anything
/// legitimate while still bounding the process.
const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;

/// The error an oversized event surfaces as, named after the cap so the
/// operator sees which bound tripped. It rides the parser's existing
/// `io::Error` channel, which both clients map to `ApiError::Io`.
fn event_overflow() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("SSE event exceeded MAX_EVENT_BYTES ({MAX_EVENT_BYTES} bytes)"),
    )
}

/// Parses a stream of SSE bytes into events.
///
/// Reads line by line, accumulating `event:` and `data:` fields.
/// Emits an `SseEvent` on each blank line boundary.
pub struct SseParser<R> {
    reader: std::io::BufReader<R>,
    event: Option<String>,
    data: Vec<String>,
    /// Bytes accumulated in `data` (each value plus its joining newline),
    /// checked against [`MAX_EVENT_BYTES`] as lines are read.
    data_bytes: usize,
}

impl<R: std::io::Read> SseParser<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader: std::io::BufReader::new(reader),
            event: None,
            data: Vec::new(),
            data_bytes: 0,
        }
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        if self.data.is_empty() {
            // No data accumulated — reset and skip (per SSE spec).
            self.event = None;
            return None;
        }

        let event = SseEvent {
            event: self.event.take(),
            data: self.data.join("\n"),
        };
        self.data.clear();
        self.data_bytes = 0;
        Some(event)
    }

    /// Read one line (through `\n`) into `line`, returning the bytes read. A
    /// capped stand-in for `BufRead::read_line`: the cap must hold while a
    /// line is still arriving — a stream that never sends a newline would
    /// otherwise grow the buffer without bound before any per-event check
    /// could run — so the event's accumulated bytes plus the partial line are
    /// checked against [`MAX_EVENT_BYTES`] on every buffered chunk.
    fn read_line(&mut self, line: &mut String) -> std::io::Result<usize> {
        use std::io::BufRead;

        let mut bytes = Vec::new();
        loop {
            let (done, used) = {
                let chunk = self.reader.fill_buf()?;
                match chunk.iter().position(|&b| b == b'\n') {
                    Some(i) => {
                        bytes.extend_from_slice(&chunk[..=i]);
                        (true, i + 1)
                    }
                    None => {
                        bytes.extend_from_slice(chunk);
                        (chunk.is_empty(), chunk.len())
                    }
                }
            };
            self.reader.consume(used);
            if self.data_bytes + bytes.len() > MAX_EVENT_BYTES {
                return Err(event_overflow());
            }
            if done {
                break;
            }
        }
        let text = std::str::from_utf8(&bytes).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "stream did not contain valid UTF-8",
            )
        })?;
        line.push_str(text);
        Ok(bytes.len())
    }
}

impl<R: std::io::Read> Iterator for SseParser<R> {
    type Item = std::io::Result<SseEvent>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        loop {
            line.clear();
            match self.read_line(&mut line) {
                Ok(0) => {
                    // EOF — dispatch any remaining event.
                    return self.dispatch().map(Ok);
                }
                Ok(_) => {
                    let line = line.trim_end_matches(['\r', '\n']);

                    if line.is_empty() {
                        // Blank line — dispatch event.
                        if let Some(event) = self.dispatch() {
                            return Some(Ok(event));
                        }
                        continue;
                    }

                    // Comment line — skip.
                    if line.starts_with(':') {
                        continue;
                    }

                    // Parse field: value
                    let (field, value) = match line.find(':') {
                        Some(pos) => {
                            let value = &line[pos + 1..];
                            // Strip single leading space per SSE spec.
                            let value = value.strip_prefix(' ').unwrap_or(value);
                            (&line[..pos], value)
                        }
                        // Field with no colon — treat entire line as field, value is empty.
                        None => (line, ""),
                    };

                    match field {
                        "event" => self.event = Some(value.to_string()),
                        "data" => {
                            self.data.push(value.to_string());
                            // +1 per line for the newline `dispatch` joins
                            // with, so a flood of empty `data:` lines still
                            // counts toward the cap.
                            self.data_bytes += value.len() + 1;
                        }
                        _ => {} // Ignore unknown fields per SSE spec.
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reader type the clients hand to `SseParser` in production. Tests
    /// parse over the same type so the whole crate drives a single
    /// monomorphization — a branch covered here is covered, full stop, rather
    /// than covered in a `&[u8]` instantiation that production never runs.
    type ProductionReader = Box<dyn std::io::Read + Send + Sync>;

    fn parse(input: &str) -> Vec<SseEvent> {
        let reader: ProductionReader = Box::new(std::io::Cursor::new(input.to_string()));
        SseParser::new(reader)
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    #[test]
    fn simple_data_event() {
        let events = parse("data: hello\n\n");
        assert_eq!(
            events,
            vec![SseEvent {
                event: None,
                data: "hello".to_string(),
            }]
        );
    }

    #[test]
    fn event_with_type() {
        let events = parse("event: message_start\ndata: {\"type\":\"message_start\"}\n\n");
        assert_eq!(
            events,
            vec![SseEvent {
                event: Some("message_start".to_string()),
                data: "{\"type\":\"message_start\"}".to_string(),
            }]
        );
    }

    #[test]
    fn multiple_data_lines_joined_with_newline() {
        let events = parse("data: line one\ndata: line two\n\n");
        assert_eq!(
            events,
            vec![SseEvent {
                event: None,
                data: "line one\nline two".to_string(),
            }]
        );
    }

    #[test]
    fn multiple_events() {
        let input = "data: first\n\ndata: second\n\n";
        let events = parse(input);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "first");
        assert_eq!(events[1].data, "second");
    }

    #[test]
    fn comments_are_ignored() {
        let events = parse(": this is a comment\ndata: hello\n\n");
        assert_eq!(
            events,
            vec![SseEvent {
                event: None,
                data: "hello".to_string(),
            }]
        );
    }

    #[test]
    fn blank_lines_without_data_produce_no_event() {
        let events = parse("\n\n\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn data_with_no_trailing_blank_line_dispatches_on_eof() {
        let events = parse("data: eof event");
        assert_eq!(
            events,
            vec![SseEvent {
                event: None,
                data: "eof event".to_string(),
            }]
        );
    }

    #[test]
    fn data_field_with_no_space_after_colon() {
        let events = parse("data:no space\n\n");
        assert_eq!(events[0].data, "no space");
    }

    #[test]
    fn data_field_with_extra_spaces_preserved() {
        let events = parse("data:  two spaces\n\n");
        // Only one leading space is stripped per spec.
        assert_eq!(events[0].data, " two spaces");
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let events = parse("id: 123\nretry: 5000\ndata: hello\n\n");
        assert_eq!(
            events,
            vec![SseEvent {
                event: None,
                data: "hello".to_string(),
            }]
        );
    }

    #[test]
    fn field_with_no_colon() {
        // Per SSE spec: line with no colon is treated as field name with empty value.
        let events = parse("data\n\n");
        assert_eq!(
            events,
            vec![SseEvent {
                event: None,
                data: "".to_string(),
            }]
        );
    }

    #[test]
    fn carriage_return_line_endings() {
        let events = parse("event: test\r\ndata: hello\r\n\r\n");
        assert_eq!(
            events,
            vec![SseEvent {
                event: Some("test".to_string()),
                data: "hello".to_string(),
            }]
        );
    }

    /// A reader that always fails — the parser's `Err` arm is otherwise only
    /// reachable through a real transport failure mid-stream.
    struct FailingReader;

    impl std::io::Read for FailingReader {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("connection reset"))
        }
    }

    #[test]
    fn read_error_is_surfaced() {
        let reader: ProductionReader = Box::new(FailingReader);
        let mut parser = SseParser::new(reader);
        let err = parser.next().unwrap().unwrap_err();
        assert_eq!(err.to_string(), "connection reset");
    }

    fn parser_over(input: Vec<u8>) -> SseParser<ProductionReader> {
        SseParser::new(Box::new(std::io::Cursor::new(input)) as ProductionReader)
    }

    #[test]
    fn event_data_over_the_cap_errors() {
        // Data lines whose accumulated size crosses MAX_EVENT_BYTES abort the
        // stream with an error naming the cap, instead of growing without
        // bound toward an event boundary that may never come.
        let line = format!("data: {}\n", "x".repeat(1024 * 1024));
        let mut parser = parser_over(line.repeat(9).into_bytes());
        let err = parser.next().unwrap().unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("MAX_EVENT_BYTES"));
    }

    #[test]
    fn unterminated_line_over_the_cap_errors() {
        // The cap must hold mid-line: a stream that never sends a newline
        // would otherwise grow read_line's buffer unboundedly before any
        // per-event check could run.
        let input = format!("data: {}", "x".repeat(MAX_EVENT_BYTES + 1));
        let mut parser = parser_over(input.into_bytes());
        let err = parser.next().unwrap().unwrap_err();
        assert!(err.to_string().contains("MAX_EVENT_BYTES"));
    }

    #[test]
    fn invalid_utf8_is_an_error() {
        // The capped read_line preserves BufRead::read_line's contract: a
        // line that is not valid UTF-8 is an InvalidData error, not a panic
        // or a lossy decode.
        let mut parser = parser_over(vec![b'd', 0xff, b'\n']);
        let err = parser.next().unwrap().unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("valid UTF-8"));
    }

    #[test]
    fn anthropic_style_stream() {
        let input = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\"}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let events = parse(input);
        assert_eq!(events.len(), 6);
        assert_eq!(events[0].event.as_deref(), Some("message_start"));
        assert_eq!(events[2].event.as_deref(), Some("content_block_delta"));
        assert_eq!(events[5].event.as_deref(), Some("message_stop"));
    }
}
