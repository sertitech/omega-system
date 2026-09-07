//! Tool approval, bounded fan-out, and ordered result collection.

use super::confirm::{ConfirmCall, ConfirmOutcome};
use super::{Agent, CONFIRM_BREAKER_LIMIT};
use crate::display::{sanitize_for_display, sanitize_multiline};
use crate::tools::ToolDef;
use crate::turn::{Block, Role};
use std::io::Write;

// ── Tool execution ──

/// A tool call that cleared every pre-flight check in `execute_tools` and is
/// ready to run. Borrows the block's id/input and the tool itself; `slot`
/// indexes the turn's results vector, so reassembly restores the model's
/// block order no matter which worker finishes first.
struct ApprovedCall<'a> {
    slot: usize,
    id: &'a str,
    tool: &'a dyn ToolDef,
    input: &'a serde_json::Value,
}

/// Cap on the verbatim result line: a result that fits — one line within
/// the cap — prints as-is; anything larger collapses to a size-annotated
/// first-line preview. Only the operator's copy is summarized; the model
/// always receives the full output in its tool_result block.
const RESULT_PREVIEW_MAX_CHARS: usize = 100;

/// Human-readable byte count for the result summary line.
fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Collapse a tool result for display. Short output — one line within
/// [`RESULT_PREVIEW_MAX_CHARS`], ignoring a trailing newline — passes
/// through sanitized but verbatim; anything larger becomes
/// `N lines, SIZE — first-line preview…`. Sanitization is not optional
/// cosmetics: results carry tool-fetched content (web pages, file reads),
/// and raw control characters here could repaint the terminal or forge
/// model output.
fn summarize_output(output: &str) -> String {
    let body = output.trim_end_matches('\n');
    let mut lines = body.lines();
    let first = lines.next().unwrap_or("");
    let multiline = lines.next().is_some();
    if !multiline && first.chars().count() <= RESULT_PREVIEW_MAX_CHARS {
        return sanitize_for_display(body);
    }
    let count = body.lines().count();
    let noun = if count == 1 { "line" } else { "lines" };
    let preview: String = first.chars().take(RESULT_PREVIEW_MAX_CHARS).collect();
    format!(
        "{count} {noun}, {} — {}…",
        format_size(output.len()),
        sanitize_for_display(&preview)
    )
}

/// Cap on the confirmation summary, so an adversarial input cannot scroll
/// the prompt (and the dangerous part of a command) off-screen. The inverse
/// trade-off is accepted: anything past the cap is not shown, so the gate
/// is sound only for commands whose dangerous part is visible within it.
const CONFIRM_SUMMARY_MAX_CHARS: usize = 200;

/// One-line summary for the confirmation prompt: the tool name plus the
/// tool's own `format_status` line. Deriving the detail from the tool —
/// rather than guessing input fields here — guarantees the prompt shows the
/// same field the tool will act on, never a decoy field.
fn confirm_summary(name: &str, status: Option<&str>) -> String {
    let summary = match status {
        Some(status) => format!("{name}: {}", sanitize_for_display(status)),
        None => name.to_string(),
    };
    if summary.chars().count() > CONFIRM_SUMMARY_MAX_CHARS {
        let capped: String = summary.chars().take(CONFIRM_SUMMARY_MAX_CHARS).collect();
        format!("{capped}…")
    } else {
        summary
    }
}

impl Agent {
    /// The text that initiated the current turn: the newest user message
    /// carrying a `Text` block. Later user messages in a tool loop hold only
    /// tool results, so this walks past them to the typed prompt (or, in a
    /// child, the parent's task description). Quoted to the judge as the
    /// intent the proposed call should serve — trusted *relative to tool
    /// output*: it came from the operator or the delegating parent, never
    /// from fetched content.
    fn initiating_request(&self) -> Option<String> {
        self.messages.iter().rev().find_map(|m| {
            if m.role != Role::User {
                return None;
            }
            m.content.iter().find_map(|b| match b {
                Block::Text(text) => Some(text.clone()),
                _ => None,
            })
        })
    }

    /// Execute each tool call in the content blocks and return ToolResult
    /// blocks in the model's block order. Three phases: a serial pre-flight
    /// that runs every check and prompt in order on the main thread, a fan-out
    /// that runs approved calls concurrently on scoped threads when all of
    /// them are read-only (a batch containing a side-effecting call runs
    /// inline in block order, as does a batch of one — the common path spawns
    /// nothing), and a serial reassembly keyed by block index. Budget
    /// exceeded, validation failure, denial, and run errors all return
    /// `is_error: true`. Status and result lines are written to `out`, as is
    /// each tool's own live output — streamed directly by inline calls,
    /// buffered per worker and flushed in block order by fanned-out ones.
    pub(super) fn execute_tools(&mut self, blocks: &[Block], out: &mut dyn Write) -> Vec<Block> {
        // A rejection resolves its slot immediately, before the fan-out.
        fn rejected(id: &str, content: String) -> Option<Block> {
            Some(Block::ToolResult {
                tool_use_id: id.to_string(),
                content,
                is_error: true,
            })
        }

        // Phase 1 — serial pre-flight. Everything that touches agent state or
        // the operator's terminal stays on the main thread, in block order:
        // budget accounting, the side-effect flag, and above all the
        // confirmation decisions, which share one stdin (`ask`) or one judge
        // budget and must resolve one at a time. The policy being `Send +
        // Sync` means the compiler no longer holds this line — the
        // serial placement here does. Each tool_use block either resolves to
        // a rejection here or is approved for the fan-out.
        let mut slots: Vec<Option<Block>> = Vec::new();
        let mut approved: Vec<ApprovedCall> = Vec::new();

        // The turn text the judge weighs intent against, captured before the
        // borrow of `self` inside the loop. One lookup per batch, and only
        // when something in it is actually confirmable.
        let initiating: Option<String> = blocks
            .iter()
            .any(|b| {
                matches!(b, Block::ToolUse { name, .. }
                if self.tools.iter().any(|t| t.name() == name && t.requires_confirmation()))
            })
            .then(|| self.initiating_request())
            .flatten();

        for block in blocks {
            let Block::ToolUse { id, name, input } = block else {
                continue;
            };

            // A pending cancellation resolves every remaining call without
            // prompting or running it. The slot still gets an is_error
            // tool_result rather than the turn erroring out mid-batch: a
            // kept history (the side-effect carve-out) must never leave a
            // tool_use dangling without its result, so the loop surfaces
            // the quiet outcome at its next pre-request check instead.
            if self.cancelled() {
                self.meta_line(out, format_args!("cancelled: {name}"));
                slots.push(rejected(id, format!("{name} cancelled by user")));
                continue;
            }

            // Tool inputs must be JSON objects (API contract). A non-object
            // means the streamed input was malformed or absent; reject with
            // is_error so the model retries instead of running on bad input.
            if !input.is_object() {
                self.meta_line(out, format_args!("invalid input: {name}"));
                slots.push(rejected(
                    id,
                    format!("invalid input for {name}: expected a JSON object"),
                ));
                continue;
            }

            // Unknown tool name — a hallucinated tool. Self-correct via
            // is_error (the model can pick a real tool) instead of aborting
            // the whole turn, matching every other failure path here.
            let Some(tool) = self.tools.iter().find(|t| t.name() == name) else {
                self.meta_line(out, format_args!("unknown tool: {name}"));
                slots.push(rejected(id, format!("unknown tool: {name}")));
                continue;
            };

            let cost = tool.cost();

            let status = tool.format_status(input);
            if let Some(status) = &status {
                self.meta_line(out, format_args!("{}", sanitize_for_display(status)));
            }

            // Budget check-and-reserve — one atomic ledger operation, so a
            // sibling call in this turn (or another agent sharing the
            // ledger) can never race between a check and an increment done
            // as two steps. Reserved here, before validation and
            // confirmation, so a call already over budget is rejected before
            // either runs — matching the pre-shared-ledger ordering.
            if let Err((count, limit)) = self.budget.draw(cost) {
                let reason =
                    format!("budget exceeded: {count} calls at cost tier {cost} (limit {limit})");
                self.meta_line(out, format_args!("budget error: {reason}"));
                slots.push(rejected(id, reason));
                continue;
            }

            // Pre-execution validation — reject before confirmation so
            // obviously-malformed calls never reach the user prompt. The
            // budget only bounds calls that actually run, so a rejection
            // here releases the reservation just made above.
            if let Err(reason) = tool.validate(input) {
                self.budget.release(cost);
                self.meta_line(out, format_args!("validation error: {reason}"));
                slots.push(rejected(id, reason));
                continue;
            }

            // Confirmation gate — the policy decides (ask prompts, allow
            // notices, judge adjudicates), and a denial releases the
            // reservation for the same reason as the validation gate above.
            // The summary is derived from the tool's own format_status, so
            // the decider approves the field the tool acts on. A tripped
            // circuit breaker short-circuits the rest of the batch without
            // consulting the policy again: the turn is already over, and
            // each further consult would be another paid judge call.
            if tool.requires_confirmation() {
                if self.auto_denials >= CONFIRM_BREAKER_LIMIT {
                    self.budget.release(cost);
                    slots.push(rejected(
                        id,
                        format!("{name} rejected: confirmation circuit breaker tripped"),
                    ));
                    continue;
                }
                let mut summary = confirm_summary(name, status.as_deref());
                if self.confirm.mode() == crate::config::ConfirmMode::Ask {
                    match tool.confirmation_preview(input) {
                        Ok(Some(preview)) => {
                            // Put review text in the prompt itself: a child's
                            // ordinary output may still be buffered when it asks.
                            summary = format!("{summary}\n{}", sanitize_multiline(&preview));
                        }
                        Ok(None) => {}
                        Err(reason) => {
                            self.budget.release(cost);
                            self.meta_line(
                                out,
                                format_args!("preview error: {}", sanitize_multiline(&reason)),
                            );
                            slots.push(rejected(id, reason));
                            continue;
                        }
                    }
                }
                let call = ConfirmCall {
                    tool: name,
                    input,
                    summary: &summary,
                    request: initiating.as_deref(),
                };
                match self.confirm.decide(&call) {
                    ConfirmOutcome::Approved { notice } => {
                        self.auto_denials = 0;
                        if let Some(notice) = notice {
                            self.meta_line(out, format_args!("{notice}"));
                        }
                    }
                    ConfirmOutcome::Denied { detail, automated } => {
                        self.budget.release(cost);
                        if automated {
                            self.auto_denials += 1;
                            // The automated detail carries the judge's or the
                            // floor's reason — record it where the operator
                            // reads the transcript, not just in the model's
                            // tool_result.
                            self.meta_line(out, format_args!("denied: {name} — {detail}"));
                        } else {
                            self.auto_denials = 0;
                            self.meta_line(out, format_args!("denied: {name}"));
                        }
                        slots.push(rejected(id, format!("{name} {detail}")));
                        continue;
                    }
                }
            }

            // The call is approved and already counted (reserved above, count
            // before run so retries also count) — no further budget bookkeeping.

            // Mark the turn as side-effecting at approval, before the fan-out:
            // a mutating tool may touch disk even if it then returns `Err`,
            // so the conversation history must be kept verbatim on a later
            // error this turn (see `run`'s conditional rollback).
            if tool.side_effecting(input) {
                self.side_effected_this_turn = true;
            }

            approved.push(ApprovedCall {
                slot: slots.len(),
                id,
                tool: tool.as_ref(),
                input,
            });
            slots.push(None);
        }

        // Phase 2 — fan-out, but only when every approved call is read-only.
        // Side-effecting calls never run concurrently: two mutations in one
        // batch can interfere through the filesystem (both read pre-turn
        // state; the later write silently drops the earlier one while both
        // results report success), so any mutation keeps the whole batch
        // inline, in block order — the pre-fan-out semantics. The case this
        // phase exists for — several independent I/O-bound reads — still
        // fans out: `run` is `&self` everywhere, `thread::scope` lets workers
        // borrow the tools with no `Arc`/`'static` ceremony, and inputs are
        // cloned on the worker (`run` takes ownership). A worker panic is a
        // tool bug — `run` reports failure as `Err` — and is re-raised on
        // the main thread, never swallowed.
        let read_only = approved
            .iter()
            .all(|call| !call.tool.side_effecting(call.input));
        // Each outcome carries the live output the call wrote to its sink. An
        // inline call writes straight to `out` — a serial `task` child streams
        // to the operator as it works — and carries an empty (never-allocated)
        // buffer; a fanned-out call writes into a private buffer instead,
        // flushed in block order in phase 3, so concurrent children cannot
        // interleave lines.
        //
        // A gated tool (`gates_concurrency`, the default for leaf tools) takes
        // a fan-out permit for the whole of its run, bounding concurrent leaf
        // executions process-wide; the `task` tool opts out, so a `task` worker
        // never holds a permit while its child fans out — the nested-semaphore
        // deadlock cannot form, since leaf tools never run nested tools. The
        // inline arm acquires per call, one at a time; the parallel arm acquires
        // *before* `scope.spawn` and moves the guard into the worker, so a batch
        // wider than the cap parks the extra acquisitions on the main thread
        // instead of spawning them — bounding live workers, not just running
        // executions.
        let concurrency = &self.concurrency;
        let outcomes: Vec<(Result<String, String>, Vec<u8>)> = if approved.len() <= 1 || !read_only
        {
            approved
                .iter()
                .map(|call| {
                    // The between-executions seam (inline batches): a
                    // cancellation during one tool skips the rest of the
                    // batch. The parallel arm below deliberately has no
                    // counterpart — its calls are all in flight at once,
                    // and letting read-only workers finish is simpler than
                    // a kill seam; cancellation lands at the next check.
                    let outcome = if self.cancelled() {
                        Err("cancelled by user".to_string())
                    } else {
                        let _permit = call.tool.gates_concurrency().then(|| concurrency.acquire());
                        call.tool.run(call.input.clone(), &mut *out)
                    };
                    (outcome, Vec::new())
                })
                .collect()
        } else {
            std::thread::scope(|scope| {
                let workers: Vec<_> = approved
                    .iter()
                    .map(|call| {
                        // Acquire on the main thread before spawning: the
                        // (K+1)th call parks here rather than becoming a
                        // spawned-but-waiting worker, so live workers and
                        // running executions are bounded together.
                        let permit = call.tool.gates_concurrency().then(|| concurrency.acquire());
                        scope.spawn(move || {
                            // The guard rides the worker for its whole run and
                            // releases on drop, panic included.
                            let _permit = permit;
                            let mut buffered = Vec::new();
                            let outcome = call.tool.run(call.input.clone(), &mut buffered);
                            (outcome, buffered)
                        })
                    })
                    .collect();
                workers
                    .into_iter()
                    .map(|w| w.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
                    .collect()
            })
        };

        // Phase 3 — serial reassembly, keyed by block index so the
        // tool_result order always matches the model's tool_use order.
        // Buffered live output and result lines print here, not in the
        // workers, so output stays deterministic regardless of completion
        // order.
        for (call, (outcome, buffered)) in approved.iter().zip(outcomes) {
            let _ = out.write_all(&buffered);
            let (content, is_error) = match outcome {
                Ok(output) => {
                    self.meta_line(out, format_args!("result: {}", summarize_output(&output)));
                    (output, false)
                }
                Err(error) => {
                    self.meta_line(
                        out,
                        format_args!("tool error: {}", sanitize_multiline(&error)),
                    );
                    (error, true)
                }
            };
            slots[call.slot] = Some(Block::ToolResult {
                tool_use_id: call.id.to_string(),
                content,
                is_error,
            });
        }

        // Every slot is now filled — rejected in pre-flight or computed above.
        slots
            .into_iter()
            .map(|slot| slot.expect("tool_use slot left unresolved"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::fixtures::*;
    use crate::agent::*;
    use crate::testing::{MockProvider, TestTool, expect_tool_result};
    use crate::turn::TurnRequest;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn execute_tools_dispatches_and_returns_result() {
        let mut agent = agent_with_tools(vec![mock_tool("echo", "hello back")]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "echo".to_string(),
            input: serde_json::json!({"msg": "hi"}),
        }];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(
            results,
            vec![Block::ToolResult {
                tool_use_id: "toolu_1".to_string(),
                content: "hello back".to_string(),
                is_error: false,
            }]
        );
    }

    #[test]
    fn execute_tools_inline_call_streams_live_output_to_the_sink() {
        // An inline (batch-of-one) call gets the parent's own sink: its live
        // output lands on the terminal as it runs, before the result line.
        let streamy = TestTool::new("streamy", "done").emitting("live line\n");
        let mut agent = agent_with_tools(vec![Box::new(streamy)]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "streamy".to_string(),
            input: serde_json::json!({}),
        }];

        let mut out = Vec::new();
        let results = agent.execute_tools(&blocks, &mut out);
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!((content, is_error), ("done", false));

        let shown = String::from_utf8(out).unwrap();
        let live = shown.find("live line").expect("live output shown");
        let result = shown.find("[result: done]").expect("result line shown");
        assert!(live < result, "live output precedes the result line");
    }

    #[test]
    fn execute_tools_parallel_batch_flushes_live_output_in_block_order() {
        // Fanned-out calls write into private buffers; the loop flushes them
        // in block order after the batch — each call's stream, then its
        // result line — regardless of which worker finished first.
        let first = TestTool::new("first", "first done").emitting("first stream\n");
        let second = TestTool::new("second", "second done").emitting("second stream\n");
        let mut agent = agent_with_tools(vec![Box::new(first), Box::new(second)]);
        let blocks = vec![
            Block::ToolUse {
                id: "toolu_1".to_string(),
                name: "first".to_string(),
                input: serde_json::json!({}),
            },
            Block::ToolUse {
                id: "toolu_2".to_string(),
                name: "second".to_string(),
                input: serde_json::json!({}),
            },
        ];

        let mut out = Vec::new();
        let results = agent.execute_tools(&blocks, &mut out);
        assert_eq!(results.len(), 2);

        let shown = String::from_utf8(out).unwrap();
        let stream_a = shown.find("first stream").expect("first call's stream");
        let result_a = shown.find("[result: first done]").expect("first result");
        let stream_b = shown.find("second stream").expect("second call's stream");
        let result_b = shown.find("[result: second done]").expect("second result");
        assert!(stream_a < result_a, "call 0 streams before its result line");
        assert!(
            result_a < stream_b,
            "block order: call 0 fully before call 1"
        );
        assert!(stream_b < result_b, "call 1 streams before its result line");
    }

    #[test]
    fn execute_tools_unknown_tool_self_corrects_with_error() {
        let mut agent = agent_with_tools(vec![]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "nonexistent".to_string(),
            input: serde_json::json!({}),
        }];

        // A hallucinated tool name must not abort the turn — it returns a
        // tool_result with is_error so the model can retry with a real tool.
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (tool_use_id, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(tool_use_id, "toolu_1");
        assert!(content.contains("unknown tool: nonexistent"));
        assert!(is_error);
    }

    #[test]
    fn execute_tools_rejects_non_object_input() {
        // A malformed streamed input surfaces as a non-object Value (Null).
        // execute_tools must reject it rather than dispatch on bad input.
        let mut agent = agent_with_tools(vec![mock_tool("echo", "hello back")]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "echo".to_string(),
            input: serde_json::Value::Null,
        }];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert!(content.contains("expected a JSON object"));
        assert!(is_error);
    }

    #[test]
    fn execute_tools_sends_is_error_on_failure() {
        let failing = TestTool::new("fail", "").with_run(|_| Err("something broke".to_string()));
        let mut agent = agent_with_tools(vec![Box::new(failing)]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "fail".to_string(),
            input: serde_json::json!({}),
        }];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "something broke");
        assert!(is_error);
    }

    #[test]
    fn execute_tools_rejects_on_validation_failure() {
        let guarded = TestTool::new("guarded", "guarded ran").with_validate(|input| {
            if input["bad"].as_bool() == Some(true) {
                Err("bad input rejected".to_string())
            } else {
                Ok(())
            }
        });
        let mut agent = agent_with_tools(vec![Box::new(guarded)]);

        // Rejected by validate — run should NOT be called.
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "guarded".to_string(),
            input: serde_json::json!({"bad": true}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "bad input rejected");
        assert!(is_error);

        // Passes validation — run proceeds normally.
        let blocks = vec![Block::ToolUse {
            id: "toolu_2".to_string(),
            name: "guarded".to_string(),
            input: serde_json::json!({"bad": false}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "guarded ran");
        assert!(!is_error);
    }

    #[test]
    fn execute_tools_enforces_budget_limit() {
        let costly = TestTool::new("costly", "done").with_cost(4);
        let mut agent = agent_with_tools(vec![Box::new(costly)]);
        agent.budget.set_limit(4, 2);

        let block = |id: &str| Block::ToolUse {
            id: id.to_string(),
            name: "costly".to_string(),
            input: serde_json::json!({}),
        };
        let is_error =
            |result: &Block| -> bool { matches!(result, Block::ToolResult { is_error: true, .. }) };

        // First two calls succeed.
        let mut out = std::io::sink();
        assert!(!is_error(&agent.execute_tools(&[block("t1")], &mut out)[0]));
        assert!(!is_error(&agent.execute_tools(&[block("t2")], &mut out)[0]));

        // Third call is rejected — budget exceeded.
        let results = agent.execute_tools(&[block("t3")], &mut out);
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert!(content.contains("budget exceeded"));
        assert!(is_error);
    }

    #[test]
    fn execute_tools_denied_by_confirmation() {
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(ask_stub(|_summary| false));

        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "guarded_write".to_string(),
            input: serde_json::json!({}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert!(content.contains("denied by user"));
        assert!(is_error);
    }

    #[test]
    fn execute_tools_approved_by_confirmation() {
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(ask_stub(|_summary| true));

        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "guarded_write".to_string(),
            input: serde_json::json!({}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "wrote something");
        assert!(!is_error);
    }

    #[test]
    fn execute_tools_validation_before_confirmation() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let guarded = TestTool::new("guarded", "ran")
            .confirmed()
            .with_validate(|input| {
                if input["valid"].as_bool() == Some(true) {
                    Ok(())
                } else {
                    Err("invalid input".to_string())
                }
            });
        let mut agent = agent_with_tools(vec![Box::new(guarded)]);

        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();
        agent.set_confirm_policy(ask_stub(move |_summary| {
            called_clone.store(true, Ordering::SeqCst);
            true
        }));

        // Invalid input — should fail validation WITHOUT triggering confirm.
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "guarded".to_string(),
            input: serde_json::json!({"valid": false}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert!(content.contains("invalid input"));
        assert!(is_error);
        assert!(
            !called.load(Ordering::SeqCst),
            "confirm should not be called for invalid input"
        );

        // Valid input — confirm fires, then run proceeds.
        let blocks = vec![Block::ToolUse {
            id: "toolu_2".to_string(),
            name: "guarded".to_string(),
            input: serde_json::json!({"valid": true}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "ran");
        assert!(!is_error);
        assert!(
            called.load(Ordering::SeqCst),
            "confirm must be called for valid input"
        );
    }

    #[test]
    fn execute_tools_skips_confirmation_when_not_required() {
        // TestTool has requires_confirmation() = false unless .confirmed().
        let mut agent = agent_with_tools(vec![mock_tool("echo", "ok")]);
        agent.set_confirm_policy(ask_stub(|_summary| false));

        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "echo".to_string(),
            input: serde_json::json!({}),
        }];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "ok");
        assert!(!is_error);
    }

    #[test]
    fn execute_tools_handles_multiple_tool_calls() {
        let mut agent = agent_with_tools(vec![
            mock_tool("tool_a", "result_a"),
            mock_tool("tool_b", "result_b"),
        ]);
        let blocks = vec![
            Block::Text("thinking...".to_string()),
            Block::ToolUse {
                id: "toolu_1".to_string(),
                name: "tool_a".to_string(),
                input: serde_json::json!({}),
            },
            Block::ToolUse {
                id: "toolu_2".to_string(),
                name: "tool_b".to_string(),
                input: serde_json::json!({}),
            },
        ];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        // Only ToolUse blocks produce results — Text blocks are skipped.
        assert_eq!(results.len(), 2);
        assert_eq!(expect_tool_result(&results[0]).1, "result_a");
        assert_eq!(expect_tool_result(&results[1]).1, "result_b");
    }

    #[test]
    fn execute_tools_runs_approved_calls_concurrently_in_block_order() {
        use std::sync::{Mutex, mpsc};
        use std::time::Duration;

        // A rendezvous pins genuine concurrency without timing assertions:
        // the FIRST block's tool cannot finish until the SECOND block's tool
        // has run. Sequential in-block-order execution would time the waiter
        // out into a worker panic; only a parallel fan-out completes it. The
        // results must still come back in block order, not completion order.
        // The unwraps (rather than mapping to Err) keep the failure paths
        // free of never-executed closures under the coverage gate.
        let (tx, rx) = mpsc::channel();
        let rx = Mutex::new(rx);
        let waiter = TestTool::new("waiter", "").with_run(move |_| {
            rx.lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
            Ok("waiter done".to_string())
        });
        let signaler = TestTool::new("signaler", "").with_run(move |_| {
            tx.send(()).unwrap();
            Ok("signaler done".to_string())
        });
        let mut agent = agent_with_tools(vec![Box::new(waiter), Box::new(signaler)]);

        let blocks = vec![
            tool_use_block("t_wait", "waiter"),
            tool_use_block("t_sig", "signaler"),
        ];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(results.len(), 2);
        assert_eq!(
            expect_tool_result(&results[0]),
            ("t_wait", "waiter done", false)
        );
        assert_eq!(
            expect_tool_result(&results[1]),
            ("t_sig", "signaler done", false)
        );
    }

    #[test]
    fn execute_tools_caps_concurrent_leaf_executions_at_the_permit_count() {
        // More approved gated calls than permits: the fan-out must never run
        // more than K at once. The latch forces exactly K to overlap, so the
        // peak is deterministic — K when the cap holds, and (with the cap
        // removed) N, which this assertion would catch. Because permits are
        // acquired *before* spawn, this is the live-worker bound too: a worker
        // exists only once it holds a permit, so spawned-and-running leaf
        // workers never exceed K either.
        const K: usize = 2;
        const N: usize = 4;
        let state = fresh_rendezvous();
        let tools: Vec<Box<dyn ToolDef>> = (0..N)
            .map(|i| rendezvous_tool(&format!("rz{i}"), std::sync::Arc::clone(&state), K))
            .collect();
        let mut agent = agent_with_tools(tools);
        agent.set_concurrency(Concurrency::with_permits(K));

        let blocks: Vec<Block> = (0..N)
            .map(|i| tool_use_block(&format!("t{i}"), &format!("rz{i}")))
            .collect();
        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(results.len(), N);
        assert!(
            results.iter().all(|b| matches!(
                b,
                Block::ToolResult {
                    is_error: false,
                    ..
                }
            )),
            "every gated call completes"
        );
        assert_eq!(
            state.0.lock().unwrap().peak,
            K,
            "at most (and exactly) K leaf tools run concurrently"
        );
    }

    #[test]
    fn execute_tools_serializes_the_batch_when_any_call_is_side_effecting() {
        use std::sync::{Arc, Mutex};

        // One mutation in the batch keeps the WHOLE batch sequential: both
        // runs must happen on the calling thread (the fan-out would put them
        // on scoped workers), in block order — the pre-5a semantics that stop
        // two mutations racing each other through the filesystem. Thread
        // identity makes the proof deterministic in both directions: a
        // regression to fan-out cannot produce the main thread's id.
        let log = Arc::new(Mutex::new(Vec::new()));
        let mutator_log = Arc::clone(&log);
        let mutator = TestTool::new("mutator", "").mutating().with_run(move |_| {
            mutator_log
                .lock()
                .unwrap()
                .push(("mutator", std::thread::current().id()));
            Ok("mutated".to_string())
        });
        let reader_log = Arc::clone(&log);
        let reader = TestTool::new("reader", "").with_run(move |_| {
            reader_log
                .lock()
                .unwrap()
                .push(("reader", std::thread::current().id()));
            Ok("read".to_string())
        });
        let mut agent = agent_with_tools(vec![Box::new(mutator), Box::new(reader)]);

        let blocks = vec![
            tool_use_block("t_mut", "mutator"),
            tool_use_block("t_read", "reader"),
        ];
        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(expect_tool_result(&results[0]), ("t_mut", "mutated", false));
        assert_eq!(expect_tool_result(&results[1]), ("t_read", "read", false));
        let main = std::thread::current().id();
        assert_eq!(
            *log.lock().unwrap(),
            vec![("mutator", main), ("reader", main)],
            "a batch containing a mutation must run inline, in block order"
        );
    }

    #[test]
    fn execute_tools_reassembles_mixed_approved_and_rejected_in_order() {
        // A rejected call BETWEEN two approved ones: the rejection resolves in
        // pre-flight, the approved pair fans out, and reassembly must slot all
        // three back in the model's block order under their own ids.
        let mut agent = agent_with_tools(vec![
            mock_tool("tool_a", "result_a"),
            mock_tool("tool_b", "result_b"),
        ]);
        let blocks = vec![
            tool_use_block("t1", "tool_a"),
            tool_use_block("t2", "ghost"),
            tool_use_block("t3", "tool_b"),
        ];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(results.len(), 3);
        assert_eq!(expect_tool_result(&results[0]), ("t1", "result_a", false));
        let (id, content, is_error) = expect_tool_result(&results[1]);
        assert_eq!(id, "t2");
        assert!(content.contains("unknown tool: ghost"));
        assert!(is_error);
        assert_eq!(expect_tool_result(&results[2]), ("t3", "result_b", false));
    }

    #[test]
    fn execute_tools_all_rejected_yields_only_preflight_errors() {
        // Every call fails pre-flight, so the fan-out has nothing to run and
        // the results are exactly the pre-flight rejections, in block order.
        let mut agent = agent_with_tools(vec![]);
        let blocks = vec![
            tool_use_block("t1", "ghost_a"),
            tool_use_block("t2", "ghost_b"),
        ];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(results.len(), 2);
        let (id, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(id, "t1");
        assert!(content.contains("unknown tool: ghost_a"));
        assert!(is_error);
        let (id, content, is_error) = expect_tool_result(&results[1]);
        assert_eq!(id, "t2");
        assert!(content.contains("unknown tool: ghost_b"));
        assert!(is_error);
    }

    #[test]
    fn execute_tools_budget_counts_siblings_within_one_turn() {
        // Three calls to a limit-2 tier arrive in ONE turn: approval bumps the
        // count during pre-flight, so the third call must see its two siblings
        // and reject — a regression the one-call-per-turn budget test above
        // cannot catch.
        let costly = TestTool::new("costly", "done").with_cost(4);
        let mut agent = agent_with_tools(vec![Box::new(costly)]);
        agent.budget.set_limit(4, 2);
        let blocks = vec![
            tool_use_block("t1", "costly"),
            tool_use_block("t2", "costly"),
            tool_use_block("t3", "costly"),
        ];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(results.len(), 3);
        assert_eq!(expect_tool_result(&results[0]), ("t1", "done", false));
        assert_eq!(expect_tool_result(&results[1]), ("t2", "done", false));
        let (id, content, is_error) = expect_tool_result(&results[2]);
        assert_eq!(id, "t3");
        assert!(content.contains("budget exceeded: 2 calls at cost tier 4 (limit 2)"));
        assert!(is_error);
    }

    #[test]
    fn execute_tools_two_agents_sharing_a_ledger_cannot_jointly_exceed_the_limit() {
        // Two independent `Agent`s wired to the SAME ledger — the shape a
        // parent/child pair spawned through the `task` tool has.
        // Interleaving calls between them proves the ceiling is combined,
        // not per-agent: a regression back to each `Agent` owning its own
        // budget would let both agents separately reach the limit, doubling
        // the effective ceiling.
        let ledger = BudgetLedger::new();
        ledger.set_limit(4, 2);

        let mut agent_a =
            agent_with_tools(vec![Box::new(TestTool::new("costly", "done").with_cost(4))]);
        agent_a.set_budget_ledger(ledger.clone());
        let mut agent_b =
            agent_with_tools(vec![Box::new(TestTool::new("costly", "done").with_cost(4))]);
        agent_b.set_budget_ledger(ledger.clone());

        let block = tool_use_block("t1", "costly");
        let mut out = std::io::sink();

        // One call from each agent exhausts the shared limit of 2.
        let (_, _, is_error) =
            expect_tool_result(&agent_a.execute_tools(std::slice::from_ref(&block), &mut out)[0]);
        assert!(!is_error);
        let (_, _, is_error) =
            expect_tool_result(&agent_b.execute_tools(std::slice::from_ref(&block), &mut out)[0]);
        assert!(!is_error);

        // A further call from EITHER agent is rejected — the ceiling is
        // shared, not doubled.
        let results = agent_a.execute_tools(std::slice::from_ref(&block), &mut out);
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert!(is_error);
        assert!(content.contains("budget exceeded: 2 calls at cost tier 4 (limit 2)"));
        let (_, _, is_error) =
            expect_tool_result(&agent_b.execute_tools(std::slice::from_ref(&block), &mut out)[0]);
        assert!(is_error);

        assert_eq!(ledger.count(4), 2);
    }

    #[test]
    fn execute_tools_confirmation_prompts_stay_ordered_and_serial() {
        // Two gated tools in one turn: both prompts fire in block order during
        // the serial pre-flight (they share one stdin — never a worker), and
        // denying only the second leaves the first's approval intact.
        let mut agent = agent_with_tools(vec![
            Box::new(TestTool::new("first", "first ran").confirmed()),
            Box::new(TestTool::new("second", "second ran").confirmed()),
        ]);
        let prompts = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorder = Arc::clone(&prompts);
        agent.set_confirm_policy(ask_stub(move |summary| {
            recorder.lock().unwrap().push(summary.to_string());
            summary != "second"
        }));
        let blocks = vec![
            tool_use_block("t1", "first"),
            tool_use_block("t2", "second"),
        ];

        let results = agent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(*prompts.lock().unwrap(), vec!["first", "second"]);
        assert_eq!(expect_tool_result(&results[0]), ("t1", "first ran", false));
        let (_, content, is_error) = expect_tool_result(&results[1]);
        assert!(content.contains("second denied by user"));
        assert!(is_error);
    }

    #[test]
    fn execute_tools_allow_mode_runs_with_a_notice() {
        // The human prompt is waved (the stub panics if consulted); the call
        // runs, and the transcript records what was waved through.
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(allow_stub());
        let mut out = Vec::new();
        let results = agent.execute_tools(&[tool_use_block("t1", "guarded_write")], &mut out);
        assert_eq!(
            expect_tool_result(&results[0]),
            ("t1", "wrote something", false)
        );
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("[auto-approved: guarded_write]"), "got: {out}");
    }

    #[test]
    fn execute_tools_judge_allow_runs_and_notices_the_reason() {
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(judged_stub("ALLOW\nroutine write"));
        let mut out = Vec::new();
        let results = agent.execute_tools(&[tool_use_block("t1", "guarded_write")], &mut out);
        assert_eq!(
            expect_tool_result(&results[0]),
            ("t1", "wrote something", false)
        );
        let out = String::from_utf8(out).unwrap();
        assert!(
            out.contains("[judge allowed: guarded_write — routine write]"),
            "got: {out}"
        );
    }

    #[test]
    fn execute_tools_judge_deny_reports_reason_and_releases_budget() {
        let mut agent = agent_with_tools(vec![Box::new(
            TestTool::new("guarded_write", "wrote something")
                .confirmed()
                .with_cost(4),
        )]);
        agent.set_confirm_policy(judged_stub("DENY\ntoo destructive"));
        let mut out = Vec::new();
        let results = agent.execute_tools(&[tool_use_block("t1", "guarded_write")], &mut out);
        let (_, content, is_error) = expect_tool_result(&results[0]);
        assert_eq!(content, "guarded_write denied by judge: too destructive");
        assert!(is_error);
        // The reservation is released, exactly like a human "no" — the
        // budget only bounds calls that run.
        assert_eq!(agent.budget.count(4), 0);
        // The operator's transcript carries the recorded reason, not just
        // the model's tool_result.
        let out = String::from_utf8(out).unwrap();
        assert!(
            out.contains("[denied: guarded_write — denied by judge: too destructive]"),
            "got: {out}"
        );
    }

    #[test]
    fn execute_tools_breaker_short_circuits_the_batch() {
        // Three consecutive automated denials trip the breaker; the fourth
        // call is rejected without another (paid) adjudication.
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(judged_stub("DENY\nno"));
        let blocks: Vec<Block> = (0..4)
            .map(|i| tool_use_block(&format!("t{i}"), "guarded_write"))
            .collect();
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        for result in &results[..3] {
            let (_, content, _) = expect_tool_result(result);
            assert!(content.contains("denied by judge"), "got: {content}");
        }
        let (_, content, is_error) = expect_tool_result(&results[3]);
        assert_eq!(
            content,
            "guarded_write rejected: confirmation circuit breaker tripped"
        );
        assert!(is_error);
        assert_eq!(agent.auto_denials, 3);
    }

    #[test]
    fn execute_tools_approval_resets_the_breaker_count() {
        // DENY, DENY, ALLOW, DENY, DENY: never three *consecutive* automated
        // denials, so all five calls are adjudicated and no breaker trips.
        let verdicts = ["DENY\na", "DENY\nb", "ALLOW", "DENY\nc", "DENY\nd"];
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(judged_policy({
            let calls = Arc::clone(&calls);
            move || {
                let verdict = verdicts[calls.fetch_add(1, Ordering::Relaxed)];
                Box::new(crate::testing::ThreadSafeProvider::echo().with_send_text(verdict))
            }
        }));
        // Each call carries a distinct input so the verdict cache treats them
        // as five separate adjudications — the scenario under test is a run of
        // *different* operations, not one operation replayed.
        let blocks: Vec<Block> = (0..5)
            .map(|i| Block::ToolUse {
                id: format!("t{i}"),
                name: "guarded_write".to_string(),
                input: serde_json::json!({ "n": i }),
            })
            .collect();
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        assert_eq!(calls.load(Ordering::Relaxed), 5, "all five adjudicated");
        assert!(
            expect_tool_result(&results[2])
                .1
                .contains("wrote something")
        );
        assert!(
            expect_tool_result(&results[4])
                .1
                .contains("denied by judge")
        );
        assert_eq!(agent.auto_denials, 2);
    }

    #[test]
    fn execute_tools_human_denials_never_trip_the_breaker() {
        // Four human "no"s in one batch: someone is present and answering,
        // so every call still reaches the prompt and the count stays zero.
        let mut agent = agent_with_tools(vec![confirmed_tool()]);
        agent.set_confirm_policy(ask_stub(|_| false));
        let blocks: Vec<Block> = (0..4)
            .map(|i| tool_use_block(&format!("t{i}"), "guarded_write"))
            .collect();
        let results = agent.execute_tools(&blocks, &mut std::io::sink());
        for result in &results {
            let (_, content, _) = expect_tool_result(result);
            assert!(content.contains("denied by user"), "got: {content}");
        }
        assert_eq!(agent.auto_denials, 0);
    }

    #[test]
    fn execute_tools_cancelled_at_the_confirmation_gate_runs_nothing() {
        // A Ctrl-C while the gate waits on the operator: the answer no
        // longer matters — the already-approved call is not run, and the
        // sibling is resolved in pre-flight without ever being prompted.
        let flag = Arc::new(AtomicBool::new(false));
        let mut agent = agent_with_tools(vec![
            Box::new(TestTool::new("t1", "ran-1").confirmed()),
            Box::new(TestTool::new("t2", "ran-2").confirmed()),
        ]);
        agent.set_cancel_flag(Arc::clone(&flag));
        let prompts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = Arc::clone(&prompts);
        agent.set_confirm_policy(ask_stub(move |_summary| {
            count.fetch_add(1, Ordering::Relaxed);
            flag.store(true, Ordering::Relaxed);
            true
        }));

        let blocks = vec![tool_use_block("1", "t1"), tool_use_block("2", "t2")];
        let mut out = Vec::new();
        let results = agent.execute_tools(&blocks, &mut out);
        assert_eq!(prompts.load(Ordering::Relaxed), 1); // t2 was never prompted
        assert_eq!(
            expect_tool_result(&results[0]),
            ("1", "cancelled by user", true)
        );
        assert_eq!(
            expect_tool_result(&results[1]),
            ("2", "t2 cancelled by user", true)
        );
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("[cancelled: t2]"));
    }

    #[test]
    fn execute_tools_prints_status_when_provided() {
        let tool = TestTool::new("status_tool", "done")
            .with_status(|_| Some("doing something".to_string()));
        let mut agent = agent_with_tools(vec![Box::new(tool)]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "status_tool".to_string(),
            input: serde_json::json!({}),
        }];

        let mut out = Vec::new();
        let results = agent.execute_tools(&blocks, &mut out);
        assert_eq!(expect_tool_result(&results[0]).1, "done");
        // Both the pre-run status line and the result line reach the writer.
        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("[doing something]"));
        assert!(printed.contains("[result: done]"));
    }

    #[test]
    fn confirm_summary_includes_tool_status() {
        assert_eq!(
            confirm_summary("shell", Some("$ cargo test")),
            "shell: $ cargo test"
        );
        assert_eq!(
            confirm_summary("write_file", Some("writing src/main.rs")),
            "write_file: writing src/main.rs"
        );
    }

    #[test]
    fn confirm_summary_falls_back_to_name() {
        assert_eq!(confirm_summary("mystery", None), "mystery");
    }

    #[test]
    fn confirm_summary_sanitizes_control_characters() {
        assert_eq!(
            confirm_summary("shell", Some("$ echo hi\x1b[31m")),
            "shell: $ echo hi?[31m"
        );
    }

    #[test]
    fn confirm_summary_strips_invisible_unicode() {
        // U+202E (right-to-left override) could visually reverse the prompt.
        assert_eq!(
            confirm_summary("shell", Some("$ echo \u{202E}fr- mr\u{202C}")),
            "shell: $ echo ?fr- mr?"
        );
    }

    #[test]
    fn confirm_summary_caps_length() {
        let long = "x".repeat(500);
        let summary = confirm_summary("shell", Some(&long));
        assert_eq!(summary.chars().count(), CONFIRM_SUMMARY_MAX_CHARS + 1);
        assert!(summary.ends_with('…'));
    }

    #[test]
    fn confirm_summary_at_cap_unchanged() {
        // name "shell" + ": " + 193 chars = exactly 200 — no ellipsis.
        let status = "x".repeat(CONFIRM_SUMMARY_MAX_CHARS - 7);
        let summary = confirm_summary("shell", Some(&status));
        assert_eq!(summary.chars().count(), CONFIRM_SUMMARY_MAX_CHARS);
        assert!(!summary.ends_with('…'));
    }

    #[test]
    fn execute_tools_confirmation_summary_uses_tool_status() {
        use std::sync::{Arc, Mutex};

        let shell_like = TestTool::new("shell", "ran")
            .confirmed()
            .with_status(|input| input["command"].as_str().map(|c| format!("$ {c}")));
        let mut agent = agent_with_tools(vec![Box::new(shell_like)]);
        let seen = Arc::new(Mutex::new(String::new()));
        let seen_clone = seen.clone();
        agent.set_confirm_policy(ask_stub(move |summary| {
            *seen_clone.lock().unwrap() = summary.to_string();
            true
        }));

        // A decoy `path` field must not reach the prompt — the summary comes
        // from format_status, which reads the field the tool executes.
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "shell".to_string(),
            input: serde_json::json!({"command": "rm -rf src", "path": "README.md"}),
        }];
        agent.execute_tools(&blocks, &mut std::io::sink());
        assert_eq!(*seen.lock().unwrap(), "shell: $ rm -rf src");
    }

    #[test]
    fn format_size_picks_the_unit() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1023), "1023 B");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(3 * 1024 * 1024), "3.0 MB");
    }

    #[test]
    fn summarize_output_passes_short_results_verbatim() {
        assert_eq!(summarize_output("done"), "done");
        assert_eq!(summarize_output(""), "");
        // A trailing newline is shell-output dressing, not a second line.
        assert_eq!(summarize_output("done\n"), "done");
        // Verbatim still means sanitized: short output is model-visible data.
        assert_eq!(summarize_output("a\x1b[31mb"), "a?[31mb");
    }

    #[test]
    fn summarize_output_collapses_multiline_results() {
        assert_eq!(
            summarize_output("alpha\nbeta\ngamma\n"),
            "3 lines, 17 B — alpha…"
        );
    }

    #[test]
    fn summarize_output_truncates_a_long_single_line() {
        let long = "x".repeat(150);
        let summary = summarize_output(&long);
        assert_eq!(
            summary,
            format!("1 line, 150 B — {}…", "x".repeat(RESULT_PREVIEW_MAX_CHARS))
        );
    }

    #[test]
    fn summarize_output_truncates_preview_on_char_boundaries() {
        // 150 multibyte chars: the preview cap counts chars, not bytes, so
        // this must not split a UTF-8 sequence (a byte-indexed slice would).
        let long = "日".repeat(150);
        let summary = summarize_output(&long);
        assert!(summary.contains(&"日".repeat(RESULT_PREVIEW_MAX_CHARS)));
        assert!(!summary.contains(&"日".repeat(RESULT_PREVIEW_MAX_CHARS + 1)));
    }

    #[test]
    fn summarize_output_sanitizes_the_preview() {
        let sneaky = format!("\x1b[2K[result: fake]\n{}", "padding\n".repeat(5));
        let summary = summarize_output(&sneaky);
        assert!(summary.starts_with("6 lines"));
        assert!(!summary.contains('\x1b'));
    }

    #[test]
    fn execute_tools_summarizes_long_results_but_returns_them_whole() {
        let long = format!("first line\n{}", "body\n".repeat(50));
        let response = long.clone();
        let tool = TestTool::new("reader", "").with_run(move |_| Ok(response.clone()));
        let mut agent = agent_with_tools(vec![Box::new(tool)]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "reader".to_string(),
            input: serde_json::json!({}),
        }];

        let mut out = Vec::new();
        let results = agent.execute_tools(&blocks, &mut out);
        // The model's copy is untouched — only the printed line collapses.
        assert_eq!(expect_tool_result(&results[0]).1, long);
        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("[result: 51 lines, 261 B — first line…]"));
        assert!(!printed.contains("body"));
    }

    #[test]
    fn execute_tools_sanitizes_error_lines() {
        let tool =
            TestTool::new("boom", "").with_run(|_| Err("line one\x1b[31m\nline two".to_string()));
        let mut agent = agent_with_tools(vec![Box::new(tool)]);
        let blocks = vec![Block::ToolUse {
            id: "toolu_1".to_string(),
            name: "boom".to_string(),
            input: serde_json::json!({}),
        }];

        let mut out = Vec::new();
        let results = agent.execute_tools(&blocks, &mut out);
        // The model's copy keeps the raw error; the display copy is scrubbed
        // but complete — errors are read in full, not summarized.
        assert_eq!(
            expect_tool_result(&results[0]).1,
            "line one\x1b[31m\nline two"
        );
        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("[tool error: line one?[31m\nline two]"));
    }

    #[test]
    fn file_approval_receives_full_preview_before_mutation_and_can_decline() {
        use crate::tools::{edit_file::EditFileTool, sandbox::Sandbox, write_file::WriteFileTool};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        std::fs::write(&path, "old\n").unwrap();
        let sandbox = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let mut agent = agent_with_tools(vec![
            Box::new(WriteFileTool::new(sandbox.clone())),
            Box::new(EditFileTool::new(sandbox)),
        ]);
        let inspected_path = path.clone();
        agent.set_confirm_policy(ask_stub(move |summary| {
            assert!(summary.contains("-     1 | old\n+     1 | new"));
            assert_eq!(std::fs::read_to_string(&inspected_path).unwrap(), "old\n");
            false
        }));
        let call = Block::ToolUse {
            id: "preview".to_string(),
            name: "write_file".to_string(),
            input: serde_json::json!({"path": "file", "content": "new\n"}),
        };
        let results = agent.execute_tools(&[call], &mut std::io::sink());
        assert!(expect_tool_result(&results[0]).2);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "old\n");
        let inspected_path = path.clone();
        agent.set_confirm_policy(ask_stub(move |summary| {
            assert!(summary.contains("-     1 | old\n+     1 | new"));
            assert_eq!(std::fs::read_to_string(&inspected_path).unwrap(), "old\n");
            true
        }));
        let call = Block::ToolUse {
            id: "edit".to_string(),
            name: "edit_file".to_string(),
            input: serde_json::json!({"path": "file", "old_str": "old", "new_str": "new"}),
        };
        let results = agent.execute_tools(&[call], &mut std::io::sink());
        assert!(!expect_tool_result(&results[0]).2);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new\n");
    }

    #[test]
    fn oversized_preview_rejects_without_prompt_or_write_and_releases_budget() {
        use crate::tools::{sandbox::Sandbox, write_file::WriteFileTool};
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(Sandbox::rooted(dir.path().to_path_buf()).unwrap());
        let mut agent = agent_with_tools(vec![Box::new(tool)]);
        agent.budget.set_limit(2, 1);
        let prompts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let prompt_count = prompts.clone();
        agent.set_confirm_policy(ask_stub(move |_| {
            prompt_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            true
        }));
        let call = |content: String| Block::ToolUse {
            id: "write".to_string(),
            name: "write_file".to_string(),
            input: serde_json::json!({"path": "file", "content": content}),
        };
        let large = "x".repeat(20 * 1024);
        let results = agent.execute_tools(&[call(large.clone())], &mut std::io::sink());
        let (_, text, error) = expect_tool_result(&results[0]);
        assert!(error && text.contains("preview omitted") && text.contains("smaller edits"));
        assert_eq!(prompts.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert!(!dir.path().join("file").exists());
        let results = agent.execute_tools(&[call("small".to_string())], &mut std::io::sink());
        assert!(!expect_tool_result(&results[0]).2);
        assert_eq!(prompts.load(std::sync::atomic::Ordering::Relaxed), 1);
        // The display cap does not silently become a write-size cap for an
        // explicitly unattended policy, which still receives the exact input.
        agent.budget.set_limit(2, 2);
        agent.set_confirm_policy(allow_stub());
        let results = agent.execute_tools(&[call(large.clone())], &mut std::io::sink());
        assert!(!expect_tool_result(&results[0]).2);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("file")).unwrap(),
            large
        );
    }

    #[test]
    fn parent_and_children_share_one_fan_out_pool() {
        // The parent fans out N non-gating dispatch tools; each builds a child
        // agent — sharing the ONE pool — that runs a gated leaf. So N children
        // run concurrently, and their leaves all draw from the parent's permits.
        // With the pool at K, the latch forces exactly K leaves (belonging to
        // K different child agents) to overlap: peak == K. A regression giving
        // each agent its own pool would let all N leaves run at once — peak N —
        // which this catches. This is the `task` tool's parent→child pool
        // sharing, exercised through the real fan-out path.
        const K: usize = 2;
        const N: usize = 4;
        let pool = Concurrency::with_permits(K);
        let state = fresh_rendezvous();
        let tools: Vec<Box<dyn ToolDef>> = (0..N)
            .map(|i| {
                child_leaf_dispatch(
                    &format!("dispatch{i}"),
                    &format!("leaf{i}"),
                    pool.clone(),
                    std::sync::Arc::clone(&state),
                    K,
                )
            })
            .collect();
        let mut parent = agent_with_tools(tools);
        parent.set_concurrency(pool.clone());

        let blocks: Vec<Block> = (0..N)
            .map(|i| tool_use_block(&format!("t{i}"), &format!("dispatch{i}")))
            .collect();
        let results = parent.execute_tools(&blocks, &mut std::io::sink());

        assert_eq!(results.len(), N);
        assert_eq!(
            state.0.lock().unwrap().peak,
            K,
            "leaves across child agents share the parent's permits"
        );
    }

    #[test]
    fn judge_payload_carries_the_turns_initiating_request() {
        // The judge weighs the proposed call against the intent that started
        // the turn — the typed prompt, quoted with trusted provenance.
        let log: Arc<std::sync::Mutex<Vec<TurnRequest>>> = Arc::default();
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![
                tool_use_stream("t1", "guarded_write"),
                text_stream("done"),
            ])),
            test_config(None),
            vec![confirmed_tool()],
        );
        agent.set_confirm_policy(judged_policy({
            let log = Arc::clone(&log);
            move || {
                Box::new(
                    crate::testing::ThreadSafeProvider::echo()
                        .with_send_text("ALLOW")
                        .with_send_log(Arc::clone(&log)),
                )
            }
        }));
        agent
            .run("clean the build tree", &mut std::io::sink())
            .unwrap();
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 1);
        let payload = crate::testing::expect_text(&log[0].messages[0].content[0]);
        let payload: serde_json::Value = serde_json::from_str(payload).unwrap();
        assert_eq!(payload["initiating_request"], "clean the build tree");
        assert_eq!(payload["proposed_call"]["tool"], "guarded_write");
    }

    /// Shared instrumentation for the concurrency-cap rendezvous tests: a latch
    /// recording the peak number of tools inside `run` at once, holding every
    /// worker until at least `gate` have arrived together. Forcing that overlap
    /// makes the recorded peak deterministic — it equals the permit ceiling
    /// when the cap works, and would exceed it if the cap were removed.
    struct Rendezvous {
        active: usize,
        peak: usize,
        open: bool,
    }

    /// A tool whose `run` reports itself into a shared [`Rendezvous`]: it counts
    /// in, records the peak, opens the latch once `gate` workers overlap, then
    /// waits for the latch before counting out.
    fn rendezvous_tool(
        name: &str,
        state: std::sync::Arc<(std::sync::Mutex<Rendezvous>, std::sync::Condvar)>,
        gate: usize,
    ) -> Box<dyn ToolDef> {
        Box::new(TestTool::new(name, "done").with_run(move |_| {
            let (lock, cvar) = &*state;
            let mut s = lock.lock().unwrap();
            s.active += 1;
            s.peak = s.peak.max(s.active);
            if s.active >= gate {
                s.open = true;
                cvar.notify_all();
            }
            while !s.open {
                s = cvar.wait(s).unwrap();
            }
            s.active -= 1;
            Ok("done".to_string())
        }))
    }

    fn fresh_rendezvous() -> std::sync::Arc<(std::sync::Mutex<Rendezvous>, std::sync::Condvar)> {
        std::sync::Arc::new((
            std::sync::Mutex::new(Rendezvous {
                active: 0,
                peak: 0,
                open: false,
            }),
            std::sync::Condvar::new(),
        ))
    }

    /// A non-gating dispatch tool (like `task`) that, inside its `run`, builds
    /// a fresh child [`Agent`] sharing `pool` and drives *its* fan-out of one
    /// gated leaf — the child is built on the worker thread and never sent
    /// across it, the parent→child shape the `task` tool creates. Not gating is
    /// what lets the parent fan several out at once *and* avoids the nested
    /// deadlock (a gating dispatcher would hold a permit while its child needs
    /// one from the same pool).
    fn child_leaf_dispatch(
        name: &str,
        leaf: &str,
        pool: Concurrency,
        state: std::sync::Arc<(std::sync::Mutex<Rendezvous>, std::sync::Condvar)>,
        gate: usize,
    ) -> Box<dyn ToolDef> {
        let leaf = leaf.to_string();
        Box::new(
            TestTool::new(name, "dispatched")
                .ungated()
                .with_run(move |_| {
                    let tool = rendezvous_tool(&leaf, std::sync::Arc::clone(&state), gate);
                    let mut child = agent_with_tools(vec![tool]);
                    child.set_concurrency(pool.clone());
                    let block = tool_use_block("leaf", &leaf);
                    child.execute_tools(std::slice::from_ref(&block), &mut std::io::sink());
                    Ok("dispatched".to_string())
                }),
        )
    }

    #[test]
    #[should_panic(expected = "tool exploded")]
    fn execute_tools_reraises_a_worker_panic() {
        // A panicking `run` is a tool bug — failure is reported as `Err` —
        // so the fan-out re-raises the original payload on the main thread
        // instead of swallowing it into a tool_result.
        let bomb = TestTool::new("bomb", "").with_run(|_| panic!("tool exploded"));
        let mut agent = agent_with_tools(vec![Box::new(bomb), mock_tool("calm", "ok")]);
        let blocks = vec![tool_use_block("t1", "bomb"), tool_use_block("t2", "calm")];

        let _ = agent.execute_tools(&blocks, &mut std::io::sink());
    }
}
