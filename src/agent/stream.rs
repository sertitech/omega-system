//! Streaming response assembly, terminal rendering, and per-block limits.

use super::{Agent, AgentError};
use crate::display::scrub_controls;
use crate::markdown::MarkdownRenderer;
use crate::provider::ApiError;
use crate::turn::{Block, StopReason, StreamDelta, Usage};
use std::io::Write;

// ── Stream processing ──

/// Hard cap on a single block's accumulated bytes — streamed text or
/// tool-args JSON. After headers arrive the transport deliberately leaves the
/// response body un-timed (a long turn must stream to completion), so without
/// a cap a broken or hostile endpoint that keeps sending deltas for one block
/// grows the process without bound. A full turn at the configured
/// max_tokens=8192 tops out around a few tens of KB; 8 MiB — matching the SSE
/// layer's per-event cap — is orders of magnitude above any legitimate block.
const MAX_BLOCK_BYTES: usize = 8 * 1024 * 1024;

/// The error an oversized block surfaces as, named after the cap so the
/// operator sees which bound tripped.
fn block_overflow() -> AgentError {
    AgentError::Api(ApiError::Stream(format!(
        "content block exceeded MAX_BLOCK_BYTES ({MAX_BLOCK_BYTES} bytes)"
    )))
}

/// Close the styled renderer's half-printed line before an early return.
/// Text streams ahead of its newline, so an error or cancellation can land
/// mid-line — without this, whatever prints next lands on the partial line,
/// inside its style.
fn abort_line(renderer: &mut Option<MarkdownRenderer>, out: &mut dyn Write) {
    if let Some(r) = renderer.as_mut() {
        r.interrupt(out);
    }
}

/// Intermediate state for accumulating a content block from streaming events.
enum BlockBuilder {
    Text(String),
    Thinking {
        text: String,
        signature: String,
    },
    ToolUse {
        id: String,
        name: String,
        json: String,
    },
}

/// Fold one usage sighting into the accumulator. The wire counts are
/// cumulative, so max-of-sightings converges on the true totals; for the
/// optional cache counters a sighted `Some` also beats `None`, so a trailing
/// event that omits them cannot erase an earlier report.
fn merge_usage(acc: &mut Usage, seen: &Usage) {
    fn max_cache(acc: Option<u32>, seen: Option<u32>) -> Option<u32> {
        match (acc, seen) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => b.or(a),
        }
    }
    acc.input_tokens = acc.input_tokens.max(seen.input_tokens);
    acc.output_tokens = acc.output_tokens.max(seen.output_tokens);
    acc.cache_creation_input_tokens = max_cache(
        acc.cache_creation_input_tokens,
        seen.cache_creation_input_tokens,
    );
    acc.cache_read_input_tokens =
        max_cache(acc.cache_read_input_tokens, seen.cache_read_input_tokens);
}

/// Result of processing a streamed response.
pub(super) struct StreamResult {
    pub(super) blocks: Vec<Block>,
    pub(super) stop_reason: StopReason,
    /// Measured token accounting for this response, accumulated across the
    /// stream's usage-bearing events.
    pub(super) usage: Usage,
}

impl Agent {
    /// Consume a stream of events, printing text to `out` as it arrives and
    /// accumulating all content blocks (text and tool use). Returns the
    /// completed blocks, the final stop reason, and the measured token usage.
    ///
    /// Takes the stream as a trait object rather than `impl Iterator`: the
    /// error-path closures inside would otherwise monomorphize per caller,
    /// and the copy in the instantiation no test drives directly reads as
    /// uncovered under the scoped-100 gate.
    pub(super) fn process_stream(
        &self,
        stream: &mut dyn Iterator<Item = Result<StreamDelta, ApiError>>,
        out: &mut dyn Write,
    ) -> Result<StreamResult, AgentError> {
        let mut builders: Vec<BlockBuilder> = Vec::new();
        let mut stop_reason = None;
        // On a TTY the model's Markdown habits render as ANSI styling; piped
        // output stays byte-for-byte verbatim. Fresh per stream, so fence
        // state can't leak across turns.
        let mut renderer = self.styled.then(MarkdownRenderer::default);
        // Usage arrives split across events and providers: Anthropic reports
        // input up front (`MessageStart`) and cumulative totals in
        // `MessageDelta`; OpenAI reports everything only in the trailing
        // `MessageDelta`. Merging the max of every sighting yields the true
        // per-turn totals regardless of which event carried them.
        let mut usage = Usage::default();

        for delta in stream {
            // The between-events seam: returning here drops the stream —
            // and with it the underlying response body, aborting the
            // transfer. Nothing has been pushed to history yet (the caller
            // pushes only completed results), so the turn stays clean.
            if self.cancelled() {
                abort_line(&mut renderer, out);
                return Err(AgentError::Cancelled);
            }
            let delta = match delta {
                Ok(delta) => delta,
                Err(e) => {
                    abort_line(&mut renderer, out);
                    return Err(AgentError::Api(e));
                }
            };
            match delta {
                StreamDelta::MessageStart { usage: u } => merge_usage(&mut usage, &u),
                StreamDelta::TextStart { text, .. } => {
                    if !text.is_empty() {
                        match renderer.as_mut() {
                            Some(r) => r.push(&text, out),
                            None => {
                                // No renderer (piped/non-TTY): scrub through
                                // the same policy the renderer applies, so
                                // both text sinks share one guarantee.
                                let _ = out.write_all(scrub_controls(&text).as_bytes());
                            }
                        }
                        let _ = out.flush();
                    }
                    builders.push(BlockBuilder::Text(text));
                }
                StreamDelta::ThinkingStart {
                    text, signature, ..
                } => {
                    // Display policy: thinking is not the answer. A thinking
                    // model's reasoning can dwarf its reply, so the full text
                    // is never streamed to the terminal — one quiet meta line
                    // marks the block (dimmed on a TTY, like every other meta
                    // line; plain and grep-able when piped). The text itself
                    // is still accumulated: it exists for continuity, resent
                    // with its signature on tool-use turns.
                    if let Some(r) = renderer.as_mut() {
                        r.finish(out);
                    }
                    self.meta_line(out, format_args!("thinking"));
                    builders.push(BlockBuilder::Thinking { text, signature });
                }
                StreamDelta::ToolUseStart {
                    id, name, input, ..
                } => {
                    // Flush any buffered partial text line so the meta line
                    // lands after it, matching the unstyled path's layout.
                    if let Some(r) = renderer.as_mut() {
                        r.finish(out);
                    }
                    self.meta_line(out, format_args!("tool: {name}"));
                    // Seed with the start event's input. The API sends {} here
                    // for streaming, but if it ever sends a non-empty object we
                    // preserve it.
                    let seed = if input.is_object() && !input.as_object().unwrap().is_empty() {
                        input.to_string()
                    } else {
                        String::new()
                    };
                    builders.push(BlockBuilder::ToolUse {
                        id,
                        name,
                        json: seed,
                    });
                }
                StreamDelta::TextDelta { index, text } => {
                    if index < builders.len()
                        && let BlockBuilder::Text(ref mut buf) = builders[index]
                    {
                        buf.push_str(&text);
                        if buf.len() > MAX_BLOCK_BYTES {
                            abort_line(&mut renderer, out);
                            return Err(block_overflow());
                        }
                        match renderer.as_mut() {
                            Some(r) => r.push(&text, out),
                            None => {
                                // No renderer (piped/non-TTY): scrub through
                                // the same policy the renderer applies, so
                                // both text sinks share one guarantee.
                                let _ = out.write_all(scrub_controls(&text).as_bytes());
                            }
                        }
                        let _ = out.flush();
                    }
                }
                StreamDelta::ThinkingDelta { index, text } => {
                    if index < builders.len()
                        && let BlockBuilder::Thinking {
                            text: ref mut buf, ..
                        } = builders[index]
                    {
                        buf.push_str(&text);
                        // Reasoning rides the same cap as text and tool args:
                        // it is never displayed, so nothing else bounds it.
                        if buf.len() > MAX_BLOCK_BYTES {
                            abort_line(&mut renderer, out);
                            return Err(block_overflow());
                        }
                    }
                }
                StreamDelta::SignatureDelta {
                    index,
                    signature: sig,
                } => {
                    if index < builders.len()
                        && let BlockBuilder::Thinking {
                            ref mut signature, ..
                        } = builders[index]
                    {
                        signature.push_str(&sig);
                        if signature.len() > MAX_BLOCK_BYTES {
                            abort_line(&mut renderer, out);
                            return Err(block_overflow());
                        }
                    }
                }
                StreamDelta::ToolArgsDelta { index, json: part } => {
                    if index < builders.len()
                        && let BlockBuilder::ToolUse { ref mut json, .. } = builders[index]
                    {
                        json.push_str(&part);
                        if json.len() > MAX_BLOCK_BYTES {
                            abort_line(&mut renderer, out);
                            return Err(block_overflow());
                        }
                    }
                }
                StreamDelta::MessageDelta {
                    stop_reason: sr,
                    usage: u,
                } => {
                    stop_reason = sr;
                    merge_usage(&mut usage, &u);
                }
            }
        }

        let Some(stop_reason) = stop_reason else {
            // A provider adapter must normalize the protocol's terminal state
            // explicitly. Treat a synthetically exhausted or future malformed
            // adapter stream as incomplete rather than inventing EndTurn and
            // committing partial output.
            abort_line(&mut renderer, out);
            return Err(AgentError::Api(ApiError::Stream(
                "stream ended without a normalized stop reason".to_string(),
            )));
        };

        if let Some(r) = renderer.as_mut() {
            r.finish(out);
        }
        let _ = writeln!(out);

        // Usage is accumulated here but reported once per user turn, at the
        // end of `run_loop` — a per-round-trip line would repeat the same
        // numbers on the common single-round-trip turn and drown a tool loop
        // in bookkeeping.

        // Convert builders into finalized blocks.
        let mut blocks = Vec::with_capacity(builders.len());
        for b in builders {
            let block = match b {
                BlockBuilder::Text(text) => Block::Text(text),
                BlockBuilder::Thinking { text, signature } => Block::Thinking { text, signature },
                BlockBuilder::ToolUse { id, name, json } => {
                    // Tool inputs are JSON objects: an empty stream body means no
                    // arguments (`{}`). A non-empty body that fails to parse is
                    // malformed model output — log it and leave the input Null so
                    // execute_tools rejects the call with is_error (the model then
                    // retries), rather than silently dispatching a default input.
                    let input = if json.is_empty() {
                        serde_json::json!({})
                    } else {
                        serde_json::from_str(&json).unwrap_or_else(|e| {
                            self.meta_line(
                                out,
                                format_args!("malformed tool input for {name}: {e}"),
                            );
                            serde_json::Value::Null
                        })
                    };
                    Block::ToolUse { id, name, input }
                }
            };
            blocks.push(block);
        }

        Ok(StreamResult {
            blocks,
            stop_reason,
            usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::fixtures::*;
    use crate::testing::expect_tool_use;
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    #[test]
    fn process_stream_text_only() {
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: String::new(),
            }),
            Ok(StreamDelta::TextDelta {
                index: 0,
                text: "Hello ".to_string(),
            }),
            Ok(StreamDelta::TextDelta {
                index: 0,
                text: "world".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            }),
        ];

        let mut out = Vec::new();
        let result = agent
            .process_stream(&mut events.into_iter(), &mut out)
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(result.blocks, vec![Block::Text("Hello world".to_string())]);
        // The text is echoed to the writer as it streams, then the stream-end
        // newline; zero usage means no token line.
        assert_eq!(String::from_utf8(out).unwrap(), "Hello world\n");
    }

    #[test]
    fn process_stream_thinking_accumulates_text_and_signature_quietly() {
        let agent = agent_with_tools(vec![]);
        // Seeds from the start event, then reasoning split across deltas and
        // a signature split across two signature_delta events.
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ThinkingStart {
                index: 0,
                text: "seed ".to_string(),
                signature: "s0".to_string(),
            }),
            Ok(StreamDelta::ThinkingDelta {
                index: 0,
                text: "let me ".to_string(),
            }),
            Ok(StreamDelta::ThinkingDelta {
                index: 0,
                text: "reason".to_string(),
            }),
            Ok(StreamDelta::SignatureDelta {
                index: 0,
                signature: "s1".to_string(),
            }),
            Ok(StreamDelta::SignatureDelta {
                index: 0,
                signature: "s2".to_string(),
            }),
            Ok(StreamDelta::TextStart {
                index: 1,
                text: "the answer".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            }),
        ];

        let mut out = Vec::new();
        let result = agent
            .process_stream(&mut events.into_iter(), &mut out)
            .unwrap();

        assert_eq!(
            result.blocks,
            vec![
                Block::Thinking {
                    text: "seed let me reason".to_string(),
                    signature: "s0s1s2".to_string(),
                },
                Block::Text("the answer".to_string()),
            ]
        );
        // Display policy: one quiet meta line marks the block; the reasoning
        // text never reaches the terminal — only the answer does.
        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("[thinking]"));
        assert!(!printed.contains("let me reason"));
        assert!(printed.contains("the answer"));
    }

    #[test]
    fn process_stream_thinking_deltas_to_missing_or_mismatched_builders_are_ignored() {
        let agent = agent_with_tools(vec![]);
        // A thinking delta and a signature delta addressed past the builders,
        // plus both addressed to a *text* builder: all four are dropped
        // without corrupting the text block.
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "hi".to_string(),
            }),
            Ok(StreamDelta::ThinkingDelta {
                index: 0,
                text: "stray".to_string(),
            }),
            Ok(StreamDelta::SignatureDelta {
                index: 0,
                signature: "stray".to_string(),
            }),
            Ok(StreamDelta::ThinkingDelta {
                index: 9,
                text: "stray".to_string(),
            }),
            Ok(StreamDelta::SignatureDelta {
                index: 9,
                signature: "stray".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            }),
        ];

        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();
        assert_eq!(result.blocks, vec![Block::Text("hi".to_string())]);
    }

    #[test]
    fn process_stream_with_tool_use() {
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: String::new(),
            }),
            Ok(StreamDelta::TextDelta {
                index: 0,
                text: "Let me check.".to_string(),
            }),
            Ok(StreamDelta::ToolUseStart {
                index: 1,
                id: "toolu_abc".to_string(),
                name: "get_weather".to_string(),
                input: serde_json::json!({}),
            }),
            Ok(StreamDelta::ToolArgsDelta {
                index: 1,
                json: "{\"loc".to_string(),
            }),
            Ok(StreamDelta::ToolArgsDelta {
                index: 1,
                json: "ation\": \"SF\"}".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            }),
        ];

        let mut out = Vec::new();
        let result = agent
            .process_stream(&mut events.into_iter(), &mut out)
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::ToolUse);
        assert_eq!(result.blocks.len(), 2);
        assert_eq!(result.blocks[0], Block::Text("Let me check.".to_string()));
        // The tool call is announced on the output as it starts streaming.
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("[tool: get_weather]")
        );
        let (id, name, input) = expect_tool_use(&result.blocks[1]);
        assert_eq!(id, "toolu_abc");
        assert_eq!(name, "get_weather");
        assert_eq!(input["location"], "SF");
    }

    #[test]
    fn process_stream_multiple_tool_uses() {
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ToolUseStart {
                index: 0,
                id: "toolu_1".to_string(),
                name: "tool_a".to_string(),
                input: serde_json::json!({}),
            }),
            Ok(StreamDelta::ToolArgsDelta {
                index: 0,
                json: "{\"x\": 1}".to_string(),
            }),
            Ok(StreamDelta::ToolUseStart {
                index: 1,
                id: "toolu_2".to_string(),
                name: "tool_b".to_string(),
                input: serde_json::json!({}),
            }),
            Ok(StreamDelta::ToolArgsDelta {
                index: 1,
                json: "{\"y\": 2}".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            }),
        ];

        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();

        assert_eq!(result.blocks.len(), 2);
        let (_, name, input) = expect_tool_use(&result.blocks[0]);
        assert_eq!(name, "tool_a");
        assert_eq!(input["x"], 1);
        let (_, name, input) = expect_tool_use(&result.blocks[1]);
        assert_eq!(name, "tool_b");
        assert_eq!(input["y"], 2);
    }

    #[test]
    fn process_stream_empty_tool_input_yields_empty_object() {
        let agent = agent_with_tools(vec![]);
        // Tool with no parameters: a ToolUseStart with input {} and no arg deltas.
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ToolUseStart {
                index: 0,
                id: "toolu_1".to_string(),
                name: "noop".to_string(),
                input: serde_json::json!({}),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            }),
        ];

        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();

        let (_, _, input) = expect_tool_use(&result.blocks[0]);
        assert!(input.is_object(), "expected object, got: {input}");
        assert!(input.as_object().unwrap().is_empty());
    }

    #[test]
    fn process_stream_seeds_tool_input_from_start_event() {
        // Streaming sends {} on ToolUseStart, but a non-empty start input must
        // be preserved as the seed the arg deltas append to.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ToolUseStart {
                index: 0,
                id: "toolu_1".to_string(),
                name: "noop".to_string(),
                input: serde_json::json!({"pre": 1}),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            }),
        ];

        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();

        let (_, _, input) = expect_tool_use(&result.blocks[0]);
        assert_eq!(input["pre"], 1);
    }

    #[test]
    fn process_stream_malformed_tool_input_yields_null() {
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ToolUseStart {
                index: 0,
                id: "toolu_1".to_string(),
                name: "noop".to_string(),
                input: serde_json::json!({}),
            }),
            // Malformed JSON body — never assembles into a valid object.
            Ok(StreamDelta::ToolArgsDelta {
                index: 0,
                json: "{not valid json".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::ToolUse),
                usage: stub_usage(),
            }),
        ];

        let mut out = Vec::new();
        let result = agent
            .process_stream(&mut events.into_iter(), &mut out)
            .unwrap();

        // Malformed input is left Null; execute_tools turns that into an
        // is_error tool_result (see execute_tools_rejects_non_object_input).
        let (_, _, input) = expect_tool_use(&result.blocks[0]);
        assert!(input.is_null(), "expected Null, got: {input}");
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("[malformed tool input for noop:")
        );
    }

    #[test]
    fn process_stream_ignores_out_of_range_deltas() {
        // A delta whose index has no builder is harmlessly dropped.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextDelta {
                index: 5,
                text: "orphan".to_string(),
            }),
            Ok(StreamDelta::ToolArgsDelta {
                index: 9,
                json: "{}".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            }),
        ];

        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();
        assert!(result.blocks.is_empty());
    }

    #[test]
    fn process_stream_text_block_over_the_cap_errors() {
        // A text block that keeps growing past MAX_BLOCK_BYTES aborts the
        // turn with an error naming the cap. The first delta lands exactly on
        // the cap (inclusive), so only the byte after it overflows.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: String::new(),
            }),
            Ok(StreamDelta::TextDelta {
                index: 0,
                text: "x".repeat(MAX_BLOCK_BYTES),
            }),
            Ok(StreamDelta::TextDelta {
                index: 0,
                text: "y".to_string(),
            }),
        ];
        let result = agent.process_stream(&mut events.into_iter(), &mut std::io::sink());
        assert!(matches!(
            result,
            Err(AgentError::Api(ApiError::Stream(ref msg))) if msg.contains("MAX_BLOCK_BYTES")
        ));
    }

    #[test]
    fn process_stream_tool_args_over_the_cap_errors() {
        // Tool-args JSON accumulates under the same cap as text.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ToolUseStart {
                index: 0,
                id: "t1".to_string(),
                name: "big".to_string(),
                input: serde_json::json!({}),
            }),
            Ok(StreamDelta::ToolArgsDelta {
                index: 0,
                json: "x".repeat(MAX_BLOCK_BYTES + 1),
            }),
        ];
        let result = agent.process_stream(&mut events.into_iter(), &mut std::io::sink());
        assert!(matches!(
            result,
            Err(AgentError::Api(ApiError::Stream(ref msg))) if msg.contains("MAX_BLOCK_BYTES")
        ));
    }

    #[test]
    fn process_stream_thinking_over_the_cap_errors() {
        // Reasoning text accumulates under the same cap as text and tool
        // args — it is never displayed, so nothing else bounds it.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ThinkingStart {
                index: 0,
                text: String::new(),
                signature: String::new(),
            }),
            Ok(StreamDelta::ThinkingDelta {
                index: 0,
                text: "x".repeat(MAX_BLOCK_BYTES + 1),
            }),
        ];
        let result = agent.process_stream(&mut events.into_iter(), &mut std::io::sink());
        assert!(matches!(
            result,
            Err(AgentError::Api(ApiError::Stream(ref msg))) if msg.contains("MAX_BLOCK_BYTES")
        ));
    }

    #[test]
    fn process_stream_signature_over_the_cap_errors() {
        // The signature accumulator is bounded too: a hostile stream of
        // endless signature_delta events must not grow the process.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::ThinkingStart {
                index: 0,
                text: String::new(),
                signature: String::new(),
            }),
            Ok(StreamDelta::SignatureDelta {
                index: 0,
                signature: "s".repeat(MAX_BLOCK_BYTES + 1),
            }),
        ];
        let result = agent.process_stream(&mut events.into_iter(), &mut std::io::sink());
        assert!(matches!(
            result,
            Err(AgentError::Api(ApiError::Stream(ref msg))) if msg.contains("MAX_BLOCK_BYTES")
        ));
    }

    #[test]
    fn process_stream_accumulates_cache_usage() {
        // Cache counters ride the accumulated usage; nothing prints here —
        // the report is the end-of-turn usage line, not stream chatter.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(StreamDelta::MessageStart {
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 0,
                    cache_creation_input_tokens: Some(10),
                    cache_read_input_tokens: Some(20),
                },
            }),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "hi".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            }),
        ];
        let mut out = Vec::new();
        let result = agent
            .process_stream(&mut events.into_iter(), &mut out)
            .unwrap();
        assert_eq!(result.blocks, vec![Block::Text("hi".to_string())]);
        // The cache counters survive on the returned usage, `Option`-ness intact.
        assert_eq!(
            result.usage,
            Usage {
                input_tokens: 1,
                output_tokens: 0,
                cache_creation_input_tokens: Some(10),
                cache_read_input_tokens: Some(20),
            }
        );
        assert_eq!(String::from_utf8(out).unwrap(), "hi\n");
    }

    #[test]
    fn process_stream_merges_cache_usage_from_message_delta() {
        // Anthropic's message_delta carries cumulative usage, cache counters
        // included. A trailing sighting merges by max — it must neither be
        // dropped nor double-counted — and a counter first sighted there
        // (read: None then Some) still lands on the result.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(StreamDelta::MessageStart {
                usage: Usage {
                    input_tokens: 40,
                    output_tokens: 0,
                    cache_creation_input_tokens: Some(5),
                    cache_read_input_tokens: None,
                },
            }),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "hi".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 40,
                    output_tokens: 9,
                    cache_creation_input_tokens: Some(12),
                    cache_read_input_tokens: Some(30),
                },
            }),
        ];
        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();
        assert_eq!(
            result.usage,
            Usage {
                input_tokens: 40,
                output_tokens: 9,
                cache_creation_input_tokens: Some(12),
                cache_read_input_tokens: Some(30),
            }
        );
    }

    #[test]
    fn process_stream_accumulates_token_usage() {
        // Anthropic-shaped split: input up front, output in the trailing delta.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(StreamDelta::MessageStart {
                usage: Usage {
                    input_tokens: 42,
                    output_tokens: 1,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            }),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "hi".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 17,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            }),
        ];
        let mut out = Vec::new();
        let result = agent
            .process_stream(&mut events.into_iter(), &mut out)
            .unwrap();
        assert_eq!(result.blocks, vec![Block::Text("hi".to_string())]);
        // Max-accumulation across events: 42 in from MessageStart, 17 out
        // from the trailing MessageDelta.
        assert_eq!(
            result.usage,
            Usage {
                input_tokens: 42,
                output_tokens: 17,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            }
        );
        // The stream itself stays quiet about usage; the per-turn line is
        // run_loop's job.
        assert_eq!(String::from_utf8(out).unwrap(), "hi\n");
    }

    #[test]
    fn process_stream_accumulates_token_usage_trailing_only() {
        // OpenAI-shaped: MessageStart carries no usage, both counts arrive in
        // the trailing MessageDelta. The max-accumulation must pick up both
        // from the delta — the path the Anthropic-shaped test doesn't exercise.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(StreamDelta::MessageStart {
                usage: Usage::default(),
            }),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "hi".to_string(),
            }),
            Ok(StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: Usage {
                    input_tokens: 30,
                    output_tokens: 12,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            }),
        ];
        let result = agent
            .process_stream(&mut events.into_iter(), &mut std::io::sink())
            .unwrap();
        assert_eq!(result.blocks, vec![Block::Text("hi".to_string())]);
        assert_eq!(
            result.usage,
            Usage {
                input_tokens: 30,
                output_tokens: 12,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            }
        );
    }

    #[test]
    fn process_stream_propagates_error() {
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> =
            vec![Err(ApiError::Io(std::io::Error::other("nope")))];
        assert!(matches!(
            agent.process_stream(&mut events.into_iter(), &mut std::io::sink()),
            Err(AgentError::Api(_))
        ));
    }

    #[test]
    fn process_stream_rejects_a_missing_stop_reason() {
        // A provider-neutral stream that ends without explicitly normalizing
        // its terminal reason is incomplete, not an implicit EndTurn.
        let agent = agent_with_tools(vec![]);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "done".to_string(),
            }),
        ];
        let result = agent.process_stream(&mut events.into_iter(), &mut std::io::sink());
        assert!(matches!(
            result,
            Err(AgentError::Api(ApiError::Stream(ref message)))
                if message == "stream ended without a normalized stop reason"
        ));
    }

    #[test]
    fn process_stream_styled_error_mid_line_closes_the_partial_line() {
        // Text streams ahead of its newline, so a stream that dies mid-line
        // leaves a half-printed, styled line; the abort must reset the style
        // and newline so whatever prints next starts clean.
        let mut agent = agent_with_tools(vec![]);
        agent.set_styled(true);
        let events: Vec<Result<StreamDelta, ApiError>> = vec![
            Ok(message_start()),
            Ok(StreamDelta::TextStart {
                index: 0,
                text: "## Par".to_string(),
            }),
            Err(ApiError::Stream("connection reset".to_string())),
        ];

        let mut out = Vec::new();
        let result = agent.process_stream(&mut events.into_iter(), &mut out);

        assert!(matches!(result, Err(AgentError::Api(_))));
        assert_eq!(String::from_utf8(out).unwrap(), "\x1b[1;4mPar\x1b[0m\n");
    }

    #[test]
    fn process_stream_cancelled_between_events_stops_consuming() {
        // The stream raises the flag while yielding its third event — the
        // between-events check fires before that event is processed, so the
        // partial text stays partial and the tail is never pulled.
        let agent = agent_with_tools(vec![]);
        let flag = Arc::clone(&agent.cancel);
        let pulled = Rc::new(Cell::new(0usize));
        let counter = Rc::clone(&pulled);
        let events = vec![
            message_start(),
            StreamDelta::TextStart {
                index: 0,
                text: "par".to_string(),
            },
            StreamDelta::TextDelta {
                index: 0,
                text: "tial".to_string(),
            },
            StreamDelta::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: stub_usage(),
            },
        ];
        let mut stream = events.into_iter().map(move |e| {
            counter.set(counter.get() + 1);
            if counter.get() == 3 {
                flag.store(true, Ordering::Relaxed);
            }
            Ok(e)
        });

        let mut out = Vec::new();
        let result = agent.process_stream(&mut stream, &mut out);
        assert!(matches!(result, Err(AgentError::Cancelled)));
        assert_eq!(String::from_utf8(out).unwrap(), "par");
        assert_eq!(pulled.get(), 3); // the fourth event was never pulled
    }
}
