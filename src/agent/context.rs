//! Request sizing, whole-turn compaction, and conversation persistence.
//!
//! UTF-8 bytes provide a conservative estimate without a model tokenizer.
//! Request fitting shortens only outbound tool-result copies; successful
//! compaction replaces complete older groups with a validated summary.

use super::{Agent, AgentError};
use crate::provider::ApiError;
use crate::session::{SESSION_VERSION, Session};
use crate::turn::{Block, Role, StopReason, ToolSpec, TurnMessage, TurnRequest};
use std::io::Write;

const TOOL_POLICY: &str = "\
\n\n## Tool selection policy\n\
Prefer the cheapest tool that can answer the question:\n\
1. Your own knowledge (free — no tool call needed)\n\
2. read_file / list_directory / search_files (local, fast, read-only)\n\
3. edit_file / write_file (local, mutates files)\n\
4. web_fetch (network, targeted — use when you have a specific URL)\n\
5. web_search (external API, slow, noisy — last resort only)\n\
6. shell (arbitrary local execution — only when no dedicated tool fits)\n\
\n\
Never use an expensive tool when a cheaper one suffices.";

// ── Context compaction ──

/// Compaction fires when the last measured prompt size reaches this percentage
/// of the configured `context_token_limit`. A hardcoded constant by design:
/// the limit is the configurable knob; the ratio is not a second one.
const COMPACT_THRESHOLD_PERCENT: u64 = 75;

/// How many of the most recent turn-groups survive a compaction verbatim.
/// Everything older is folded into the rolling summary.
const KEEP_RECENT_TURN_GROUPS: usize = 2;

/// Output budget for the summary sub-call, deliberately decoupled from the
/// operator's `max_tokens`: that knob caps normal replies, and a small value
/// there would truncate the summary — which the fail-fast check rejects,
/// wedging every later turn into the same failure until `/clear`. The
/// summary is prompted to be concise, so this is a generous fixed budget.
const SUMMARIZE_MAX_TOKENS: u32 = 1024;

/// System prompt for the compaction summary call.
const SUMMARIZE_SYSTEM: &str = "\
You are compacting the oldest part of a longer conversation so it can be \
replaced by a short record. Write a concise summary that preserves the user's \
goals and constraints, decisions made, key facts and file paths, and any \
unresolved questions. Reply with the summary text only.";

/// Where to cut the history so the dropped prefix `messages[..cut]` consists
/// of whole turn-groups and the most recent [`KEEP_RECENT_TURN_GROUPS`] groups
/// survive verbatim. `None` when there are not enough groups to drop anything.
///
/// A turn-group starts at a `Role::User` message carrying no `ToolResult`
/// block — a fresh user turn. A user message *with* tool results belongs to
/// the tool loop of the group in progress (including the coalesced case,
/// where the next turn's text rides in a trailing tool-result message), so
/// cutting there would separate a `tool_result` from its `tool_use`, which
/// the API rejects. Cutting only at group starts makes orphans impossible.
fn compaction_cut(messages: &[TurnMessage], keep: usize) -> Option<usize> {
    let starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            m.role == Role::User
                && !m
                    .content
                    .iter()
                    .any(|b| matches!(b, Block::ToolResult { .. }))
        })
        .map(|(i, _)| i)
        .collect();
    (starts.len() > keep).then(|| starts[starts.len() - keep])
}

const OMITTED: &str =
    "\n[tool output shortened to fit context; request a smaller range or narrower query]";

/// Estimated input size, including tool schemas and message structure. This is
/// a local byte budget, not an exact count from the provider's tokenizer.
fn input_size(request: &TurnRequest) -> usize {
    let messages = serde_json::to_string(&request.messages)
        .expect("normalized messages serialize infallibly")
        .len();
    request.tools.iter().fold(
        messages.saturating_add(request.system.as_ref().map_or(0, String::len)),
        |size, tool| {
            size.saturating_add(tool.name.len())
                .saturating_add(tool.description.len())
                .saturating_add(tool.input_schema.to_string().len())
        },
    )
}

/// Share the same output reserve with compaction's preflight decision.
fn input_budget(request: &TurnRequest, limit: u32) -> usize {
    ((u64::from(limit) * COMPACT_THRESHOLD_PERCENT / 100) as usize)
        .min(limit.saturating_sub(request.max_tokens) as usize)
}

/// Keep continuation text attached to a pending tool-result message.
pub(super) fn append_input(messages: &mut Vec<TurnMessage>, input: &str) {
    match messages.last_mut() {
        Some(last) if last.role == Role::User => last.content.push(Block::Text(input.to_string())),
        _ => messages.push(TurnMessage {
            role: Role::User,
            content: vec![Block::Text(input.to_string())],
        }),
    }
}

/// Fit the outbound copy without breaking tool pairing or rewriting the
/// operator's instructions. Large results lose their suffix first; if the
/// non-reducible content alone exceeds the budget, no request is sent.
fn fit_request(mut request: TurnRequest, limit: u32) -> Result<(TurnRequest, usize), AgentError> {
    let budget = input_budget(&request, limit);
    let mut shortened = 0;
    while input_size(&request) > budget {
        let excess = input_size(&request) - budget;
        let candidate = request
            .messages
            .iter_mut()
            .flat_map(|message| &mut message.content)
            .filter_map(|block| match block {
                Block::ToolResult { content, .. } if content.len() > OMITTED.len() => Some(content),
                _ => None,
            })
            .max_by_key(|content| content.len());
        let Some(content) = candidate else {
            return Err(AgentError::ContextLimit(limit));
        };
        // Reserve the escaped marker too: JSON escaping can expand a suffix.
        let marker_bytes = serde_json::to_string(OMITTED)
            .expect("an omission marker serializes infallibly")
            .len();
        let mut boundary = content
            .len()
            .saturating_sub(excess.saturating_add(marker_bytes));
        while !content.is_char_boundary(boundary) {
            boundary -= 1;
        }
        content.truncate(boundary);
        content.push_str(OMITTED);
        shortened += 1;
    }
    Ok((request, shortened))
}

impl Agent {
    /// Convert registered ToolDef impls into the normalized tool spec.
    fn tool_definitions(&self) -> Vec<ToolSpec> {
        self.tools
            .iter()
            .map(|t| ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }

    /// Build the normalized request for the next turn. Provider-neutral: the
    /// system prompt is plain text and messages are carried verbatim. Wire
    /// concerns — prompt-cache breakpoints, the string-or-blocks content shape —
    /// are the adapter's job, not the loop's.
    pub(super) fn build_request(&self) -> TurnRequest {
        self.build_request_with_summary(self.compacted_summary.as_deref())
    }

    fn build_request_with_summary(&self, summary: Option<&str>) -> TurnRequest {
        let has_tools = !self.tools.is_empty();

        // Append the tool-selection policy to the system prompt when tools exist.
        let system = match (&self.config.system, has_tools) {
            (Some(text), true) => Some(format!("{text}{TOOL_POLICY}")),
            (None, true) => Some(TOOL_POLICY.trim_start().to_string()),
            (Some(text), false) => Some(text.clone()),
            (None, false) => None,
        };

        // The rolling compaction summary rides in the system prompt, after the
        // configured text (see the `compacted_summary` field for why it is not
        // a message).
        let system = match (system, summary) {
            (Some(text), Some(summary)) => Some(format!(
                "{text}\n\n## Earlier conversation (summarized)\n{summary}"
            )),
            (None, Some(summary)) => {
                Some(format!("## Earlier conversation (summarized)\n{summary}"))
            }
            (system, None) => system,
        };

        TurnRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            system,
            messages: self.messages.clone(),
            tools: self.tool_definitions(),
            effort: self.config.effort.clone(),
        }
    }

    /// One compaction-summary attempt: the non-streaming `send` plus the
    /// completeness check. A response that didn't finish as plain text is
    /// not a summary — a tool-use or max_tokens turn yields empty or
    /// truncated text, and committing it would permanently lose the dropped
    /// context — so it fails exactly like a transport error.
    fn summarize(&self, request: &TurnRequest) -> Result<String, ApiError> {
        let turn = self.provider.send(request)?;
        let summary = turn
            .blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        if turn.stop_reason != StopReason::EndTurn || summary.is_empty() {
            return Err(ApiError::Stream(format!(
                "compaction summary incomplete or empty (stop reason {:?})",
                turn.stop_reason
            )));
        }
        Ok(summary)
    }

    /// Compact the conversation if the last measured prompt size crossed the
    /// threshold: summarize the oldest turn-groups via a non-streaming
    /// `provider.send()` call, fold the result into the rolling summary, and
    /// drop those messages.
    ///
    /// Summaries run between user turns, using measured usage and the current
    /// raw history size. Within a turn, every outbound request independently
    /// fits its tool-result text to the budget without changing the history.
    pub(super) fn maybe_compact(
        &mut self,
        input: &str,
        out: &mut dyn Write,
    ) -> Result<(), AgentError> {
        let limit = self.config.context_token_limit as u64;
        let mut prospective = self.build_request();
        append_input(&mut prospective.messages, input);
        let oversized =
            input_size(&prospective) > input_budget(&prospective, self.config.context_token_limit);
        if (self.last_input_tokens as u64) * 100 < limit * COMPACT_THRESHOLD_PERCENT && !oversized {
            return Ok(());
        }
        // Preserve two recent groups normally. When immutable prompt content
        // cannot fit, allow one older complete group to be summarized rather
        // than preventing the next user turn from making any progress.
        let cut = compaction_cut(&self.messages, KEEP_RECENT_TURN_GROUPS).or_else(|| {
            if oversized && fit_request(prospective, self.config.context_token_limit).is_err() {
                compaction_cut(&self.messages, 1)
            } else {
                None
            }
        });
        let Some(cut) = cut else {
            return Ok(());
        };

        // The summary must be cumulative: this prefix's predecessors are
        // already gone, so the sub-call folds the existing summary in with the
        // prefix — otherwise each compaction would forget everything before
        // the previous one.
        let instruction = match &self.compacted_summary {
            Some(prior) => format!(
                "An earlier part of this conversation was already summarized \
                 as:\n\n{prior}\n\nWrite one replacement summary covering both \
                 that summary and the messages above."
            ),
            None => "Summarize the messages above.".to_string(),
        };
        let mut messages = self.messages[..cut].to_vec();
        messages.push(TurnMessage {
            role: Role::User,
            content: vec![Block::Text(instruction)],
        });
        let request = TurnRequest {
            model: self.config.model.clone(),
            max_tokens: SUMMARIZE_MAX_TOKENS.min(self.config.context_token_limit / 4),
            system: Some(SUMMARIZE_SYSTEM.to_string()),
            messages,
            // The prefix may carry tool_use blocks, and requests containing
            // them must define the tools that produced them.
            tools: self.tool_definitions(),
            // Summarization is routine work: it always runs at the model's
            // default effort, never the configured conversation effort.
            effort: None,
        };

        // Fail fast on a summary failure (settled decision): surfacing the
        // error beats silently sending the known-over-limit conversation.
        // One retry — no more — absorbs a transient blip (transport error or
        // malformed summary) before that policy applies; a second failure
        // surfaces unchanged, and nothing is drained.
        let request = self.fit_request(request, out)?;
        let summary = self
            .summarize(&request)
            .or_else(|_| self.summarize(&request))
            .map_err(AgentError::Api)?;
        // Verify the replacement before discarding its source. An oversized
        // summary or incoming prompt must leave the full old history available
        // for /save or a narrower follow-up.
        let mut prospective = self.build_request_with_summary(Some(&summary));
        prospective.messages.drain(..cut);
        append_input(&mut prospective.messages, input);
        fit_request(prospective, self.config.context_token_limit)?;
        self.compacted_summary = Some(summary);
        self.messages.drain(..cut);
        // The post-compaction size is unmeasurable until the next response
        // reports it, so reset and accept one turn of guard blindness — the
        // same trade-off as the startup/`/clear` case.
        self.last_input_tokens = 0;
        self.meta_line(out, format_args!("compacted {cut} messages into summary"));
        Ok(())
    }

    /// Check every outbound request, including tool-loop continuations and
    /// summaries. Shortening only the outbound copy preserves the complete
    /// session for recovery and never separates a tool call from its result.
    pub(super) fn fit_request(
        &self,
        request: TurnRequest,
        out: &mut dyn Write,
    ) -> Result<TurnRequest, AgentError> {
        let (request, shortened) = fit_request(request, self.config.context_token_limit)?;
        if shortened > 0 {
            self.meta_line(
                out,
                format_args!("context: shortened {shortened} tool results in outgoing request"),
            );
        }
        Ok(request)
    }

    /// Reset the conversation history, starting a fresh context. Backs the
    /// REPL's `/clear` command. The measured prompt size and the compaction
    /// summary are per-conversation state and reset with it — the summary
    /// reset is load-bearing, since `build_request` folds it into the system
    /// prompt and a stale one would leak the previous conversation into the
    /// fresh one. The tool budget ledger is intentionally left untouched —
    /// it bounds cost across the whole process, not per conversation, so
    /// clearing the history must not refill it.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.last_input_tokens = 0;
        self.compacted_summary = None;
    }

    /// Snapshot the conversation state — history, rolling summary, and the
    /// compaction guard's last measurement — as a persistable [`Session`].
    /// Pure, like [`Agent::restore`]: no disk I/O here, and none in `run()`
    /// either — the agent stays a pure state machine, and the REPL
    /// orchestrates persistence through its injected sink, mirroring how the
    /// provider-switch key/build seam lives in `run_repl`, not the agent.
    pub fn session(&self) -> Session {
        Session {
            version: SESSION_VERSION,
            messages: self.messages.clone(),
            compacted_summary: self.compacted_summary.clone(),
            last_input_tokens: self.last_input_tokens,
        }
    }

    /// Load a previously snapshotted conversation, replacing the current one.
    /// The restored `last_input_tokens` measures exactly the history being
    /// restored, so the compaction guard is armed from the first post-restore
    /// turn instead of riding one turn blind. The tool budget ledger is
    /// untouched for the same reason [`Agent::clear`] leaves it alone: it
    /// bounds cost across the whole process, not per conversation.
    pub fn restore(&mut self, session: Session) {
        self.messages = session.messages;
        // A pre-fix build could persist a trailing assistant message whose
        // tool_use was orphaned by a max_tokens cut-off (see `run_loop`);
        // loading it verbatim would 400 the first post-restore request. A
        // legitimately saved session never trails on an assistant tool_use (a
        // real tool round trails on the following User tool_result), so this
        // fires only on genuine poison: drop the unpaired tool_use, and drop the
        // message if nothing else survives.
        if let Some(last) = self.messages.last_mut()
            && last.role == Role::Assistant
            && last
                .content
                .iter()
                .any(|b| matches!(b, Block::ToolUse { .. }))
        {
            last.content.retain(|b| !matches!(b, Block::ToolUse { .. }));
            if last.content.is_empty() {
                self.messages.pop();
            }
        }
        self.compacted_summary = session.compacted_summary;
        self.last_input_tokens = session.last_input_tokens;
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::*;
    use super::super::{AgentConfig, MAX_TURNS};
    use super::*;
    use crate::provider::{Provider, ProviderKind};
    use crate::testing::{
        ErrProvider, MockProvider, TestTool, expect_text, expect_tool_result, expect_tool_use,
    };
    use crate::turn::{Role, StreamDelta, ToolSpec, TurnMessage};

    fn request(blocks: Vec<Block>) -> TurnRequest {
        TurnRequest {
            model: crate::TEST_MODEL.to_string(),
            max_tokens: 64,
            system: None,
            messages: vec![TurnMessage {
                role: Role::User,
                content: blocks,
            }],
            tools: vec![],
            effort: None,
        }
    }

    fn result(text: String) -> Block {
        Block::ToolResult {
            tool_use_id: "call".into(),
            content: text,
            is_error: false,
        }
    }

    #[test]
    fn ordinary_request_is_unchanged() {
        let request = request(vec![Block::Text("hello".into())]);
        let (fitted, shortened) = fit_request(request.clone(), 1000).unwrap();
        assert_eq!(fitted, request);
        assert_eq!(shortened, 0);
    }

    #[test]
    fn sizes_system_and_tool_schema() {
        let mut request = request(vec![]);
        let base = input_size(&request);
        request.system = Some("rules".into());
        request.tools.push(ToolSpec {
            name: "read".into(),
            description: "read text".into(),
            input_schema: serde_json::json!({}),
        });
        assert_eq!(input_size(&request), base + 5 + 4 + 9 + 2);
    }

    #[test]
    fn large_results_fit_without_changing_ids_or_original() {
        let request = request(vec![
            Block::Text("keep request".into()),
            result("é\"".repeat(1000)),
            result("x".repeat(3000)),
        ]);
        let (fitted, shortened) = fit_request(request.clone(), 1000).unwrap();
        assert!(input_size(&fitted) <= 750);
        assert_eq!(shortened, 2);
        assert_eq!(
            fitted.messages[0].content[0],
            request.messages[0].content[0]
        );
        for block in &fitted.messages[0].content[1..] {
            let (tool_use_id, content, is_error) = crate::testing::expect_tool_result(block);
            assert_eq!(tool_use_id, "call");
            assert!(!is_error);
            assert!(content.ends_with(OMITTED));
        }
        assert!(!format!("{:?}", request).contains(OMITTED));
    }

    #[test]
    fn refuses_oversized_instructions_or_unshrinkable_results() {
        for blocks in [
            vec![Block::Text("q".repeat(1000))],
            vec![result("small".into())],
        ] {
            assert!(matches!(
                fit_request(request(blocks), 80),
                Err(AgentError::ContextLimit(80))
            ));
        }
        assert!(matches!(
            fit_request(request(vec![]), 64),
            Err(AgentError::ContextLimit(64))
        ));
    }

    #[test]
    fn refuses_when_even_omission_marker_cannot_fit() {
        assert!(matches!(
            fit_request(request(vec![result("x".repeat(1000))]), 100),
            Err(AgentError::ContextLimit(100))
        ));
    }

    #[test]
    fn tool_text_cannot_impersonate_internal_shortening_state() {
        let request = request(vec![result(format!("{}{OMITTED}", "x".repeat(2000)))]);
        let (fitted, count) = fit_request(request, 1000).unwrap();
        assert_eq!(count, 1);
        assert!(input_size(&fitted) <= 750);
    }

    #[test]
    fn shortening_respects_utf8_boundary() {
        for limit in 1000..1005 {
            let request = request(vec![result("é".repeat(1000))]);
            let (fitted, _) = fit_request(request, limit).unwrap();
            assert!(input_size(&fitted) <= limit as usize * 75 / 100);
        }
    }

    /// An agent primed for the compaction tests: a `limit`-token context
    /// window and three complete turn-groups of plain text history
    /// (`q0`/`a0` … `q2`/`a2`). The guard itself stays dormant until the test
    /// sets `last_input_tokens`.
    fn compaction_agent(provider: Box<dyn Provider>, limit: u32) -> Agent {
        let mut agent = Agent::new(
            provider,
            AgentConfig {
                provider_kind: ProviderKind::Anthropic,
                model: crate::TEST_MODEL.to_string(),
                max_tokens: 64,
                system: None,
                context_token_limit: limit,
                effort: None,
                max_turns: MAX_TURNS,
            },
            vec![],
        );
        for i in 0..3 {
            agent.messages.push(user_msg(&format!("q{i}")));
            agent.messages.push(assistant_msg(&format!("a{i}")));
        }
        agent
    }

    #[test]
    fn tool_definitions_converts_trait_to_spec() {
        let agent = agent_with_tools(vec![mock_tool("echo", "ok")]);
        let defs = agent.tool_definitions();

        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");
        assert_eq!(defs[0].description, "A configurable test tool");
        assert_eq!(defs[0].input_schema["type"], "object");
    }

    #[test]
    fn tool_definitions_empty_when_no_tools() {
        let agent = agent_with_tools(vec![]);
        assert!(agent.tool_definitions().is_empty());
    }

    #[test]
    fn build_request_includes_tools() {
        let mut agent = agent_with_tools(vec![mock_tool("echo", "ok")]);
        agent.messages.push(user_msg("hello"));
        let req = agent.build_request();

        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].name, "echo");
    }

    #[test]
    fn build_request_tools_empty_when_none() {
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("hello"));
        assert!(agent.build_request().tools.is_empty());
    }

    #[test]
    fn build_request_effort_is_none_by_default() {
        // With no effort configured the request carries none — the guarantee
        // that a default turn is byte-identical to the pre-effort wire shape.
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("hi"));
        assert!(agent.build_request().effort.is_none());
    }

    #[test]
    fn build_request_copies_configured_effort() {
        let mut agent = agent_with_tools(vec![]);
        agent.set_effort(Some("high".to_string()));
        agent.messages.push(user_msg("hi"));
        assert_eq!(agent.build_request().effort.as_deref(), Some("high"));
    }

    #[test]
    fn build_request_no_system_when_unset_and_no_tools() {
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("hi"));
        assert!(agent.build_request().system.is_none());
    }

    #[test]
    fn build_request_injects_tool_policy_when_tools_present() {
        let mut agent = agent_with_tools(vec![mock_tool("echo", "ok")]);
        agent.messages.push(user_msg("hi"));
        let req = agent.build_request();

        // System is set even though AgentConfig.system is None.
        let system = req.system.expect("system present");
        assert!(system.contains("Tool selection policy"));
    }

    #[test]
    fn build_request_appends_tool_policy_to_existing_system() {
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![])),
            test_config(Some("You are helpful.")),
            vec![mock_tool("echo", "ok")],
        );
        agent.messages.push(user_msg("hi"));
        let req = agent.build_request();

        let system = req.system.expect("system present");
        assert!(system.starts_with("You are helpful."));
        assert!(system.contains("Tool selection policy"));
    }

    #[test]
    fn build_request_carries_system_text_only() {
        // The agent emits plain system text; cache breakpoints are the adapter's
        // job, so nothing here wraps it in blocks.
        let mut agent = agent_with_system("You are helpful.");
        agent.messages.push(user_msg("hi"));
        assert_eq!(
            agent.build_request().system.as_deref(),
            Some("You are helpful.")
        );
    }

    #[test]
    fn build_request_does_not_mutate_agent_messages() {
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("hello"));
        agent.build_request();

        assert_eq!(
            agent.messages[0].content,
            vec![Block::Text("hello".to_string())]
        );
    }

    #[test]
    fn build_request_carries_messages_verbatim() {
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("a"));
        agent.messages.push(assistant_msg("b"));
        let req = agent.build_request();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0], user_msg("a"));
        assert_eq!(req.messages[1], assistant_msg("b"));
    }

    #[test]
    fn compaction_cut_empty_history_is_none() {
        assert!(compaction_cut(&[], KEEP_RECENT_TURN_GROUPS).is_none());
    }

    #[test]
    fn compaction_cut_single_group_is_none() {
        let msgs = vec![user_msg("q0"), assistant_msg("a0")];
        assert!(compaction_cut(&msgs, KEEP_RECENT_TURN_GROUPS).is_none());
    }

    #[test]
    fn compaction_cut_exactly_keep_window_is_none() {
        // Two groups fill the keep window on the nose — nothing older exists.
        let msgs = vec![
            user_msg("q0"),
            assistant_msg("a0"),
            user_msg("q1"),
            assistant_msg("a1"),
        ];
        assert!(compaction_cut(&msgs, KEEP_RECENT_TURN_GROUPS).is_none());
    }

    #[test]
    fn compaction_cut_drops_groups_beyond_keep_window() {
        // Four fully-summarizable groups, keep 2 → the cut lands at the third
        // group's start, dropping the first two whole.
        let mut msgs = Vec::new();
        for i in 0..4 {
            msgs.push(user_msg(&format!("q{i}")));
            msgs.push(assistant_msg(&format!("a{i}")));
        }
        assert_eq!(compaction_cut(&msgs, KEEP_RECENT_TURN_GROUPS), Some(4));
    }

    #[test]
    fn compaction_cut_never_splits_a_tool_loop() {
        // Group 0 contains a tool loop. Its tool_result message (index 2) is
        // not a boundary — only the fresh user turns at 0, 4, and 6 are — so
        // the cut keeps the loop intact and lands on group 1's start.
        let msgs = vec![
            user_msg("q0"),
            tool_use_msg("t1", "echo"),
            tool_result_msg("t1", "ok"),
            assistant_msg("a0"),
            user_msg("q1"),
            assistant_msg("a1"),
            user_msg("q2"),
            assistant_msg("a2"),
        ];
        assert_eq!(compaction_cut(&msgs, KEEP_RECENT_TURN_GROUPS), Some(4));
    }

    #[test]
    fn compaction_cut_skips_coalesced_turn_start() {
        // The turn asking "q2" starts mid-message — its text was coalesced
        // into the trailing tool_result message at index 4 (see `run`'s
        // coalescing). That message is not a safe boundary: were it one, the
        // cut would land there (index 4) and orphan the tool_result from its
        // tool_use; instead the boundaries are 0, 2, and 6, cutting at 2.
        let msgs = vec![
            user_msg("q0"),
            assistant_msg("a0"),
            user_msg("q1"),
            tool_use_msg("t1", "writer"),
            TurnMessage {
                role: Role::User,
                content: vec![
                    Block::ToolResult {
                        tool_use_id: "t1".to_string(),
                        content: "wrote".to_string(),
                        is_error: false,
                    },
                    Block::Text("q2".to_string()),
                ],
            },
            assistant_msg("a2"),
            user_msg("q3"),
            assistant_msg("a3"),
        ];
        assert_eq!(compaction_cut(&msgs, KEEP_RECENT_TURN_GROUPS), Some(2));
    }

    #[test]
    fn compaction_summarizes_away_a_thinking_prefix() {
        // A thinking block in the droppable prefix is summarized away like
        // everything else: it rides into the summary sub-call verbatim, and
        // no signature survives compaction — only live tool-use turns resend
        // one, and the dropped messages no longer exist to be resent.
        let provider = MockProvider::new(vec![text_stream("next")]).with_send_text("S1");
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.messages[1].content.insert(
            0,
            Block::Thinking {
                text: "old reasoning".to_string(),
                signature: "old_sig".to_string(),
            },
        );
        agent.last_input_tokens = 7500;

        agent.run("q3", &mut std::io::sink()).unwrap();

        // The sub-call carried the thinking block inside the dropped prefix…
        let log = log.borrow();
        assert_eq!(
            log[0].messages[1].content[0],
            Block::Thinking {
                text: "old reasoning".to_string(),
                signature: "old_sig".to_string(),
            }
        );
        // …and after compaction neither the block nor its signature remains.
        assert_eq!(agent.compacted_summary.as_deref(), Some("S1"));
        assert!(agent.messages.iter().all(|m| {
            m.content
                .iter()
                .all(|b| !matches!(b, Block::Thinking { .. }))
        }));
    }

    #[test]
    fn run_compacts_at_threshold_and_folds_summary_into_system() {
        // 7,500 measured tokens against a 10,000-token limit sits exactly on the
        // 75% threshold — the guard fires at >=, not >.
        let provider = MockProvider::new(vec![text_stream("next")]).with_send_text("S1");
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.last_input_tokens = 7500;

        let mut out = Vec::new();
        agent.run("q3", &mut out).unwrap();

        // The summary sub-call carried the dropped prefix (group 0) plus the
        // summarize instruction, under the dedicated system prompt.
        let log = log.borrow();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].system.as_deref(), Some(SUMMARIZE_SYSTEM));
        // The sub-call runs on its own output budget, not the operator's
        // reply cap — a small max_tokens must not truncate the summary and
        // wedge compaction into a permanent fail-fast loop.
        assert_eq!(log[0].max_tokens, SUMMARIZE_MAX_TOKENS);
        assert_eq!(log[0].messages.len(), 3);
        assert_eq!(log[0].messages[0], user_msg("q0"));
        assert_eq!(log[0].messages[1], assistant_msg("a0"));
        assert_eq!(
            log[0].messages[2],
            user_msg("Summarize the messages above.")
        );

        // History: the two kept groups plus this turn's exchange; the guard
        // resets to blind until the next measured response.
        assert_eq!(agent.messages.len(), 6);
        assert_eq!(agent.messages[0], user_msg("q1"));
        assert_eq!(agent.compacted_summary.as_deref(), Some("S1"));
        assert_eq!(agent.last_input_tokens, 0);
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("[compacted 2 messages into summary]")
        );

        // The next request folds the summary into the (otherwise absent)
        // system prompt.
        assert_eq!(
            agent.build_request().system.as_deref(),
            Some("## Earlier conversation (summarized)\nS1")
        );
    }

    #[test]
    fn compaction_request_carries_no_effort_even_when_configured() {
        // Summarization is routine work: it runs at the model default, never
        // the configured conversation effort. The sub-call's
        // recorded request must carry `effort: None` despite the agent's set.
        let provider = MockProvider::new(vec![text_stream("next")]).with_send_text("S1");
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.set_effort(Some("high".to_string()));
        agent.last_input_tokens = 7500;

        agent.run("q3", &mut std::io::sink()).unwrap();

        let log = log.borrow();
        assert_eq!(log.len(), 1);
        assert!(log[0].effort.is_none());
        // The conversation effort is untouched — only the summary sub-call
        // opts out, so subsequent turns still carry it.
        assert_eq!(agent.effort(), Some("high"));
    }

    #[test]
    fn run_below_threshold_does_not_compact() {
        let provider = MockProvider::new(vec![text_stream("next")]).with_send_text("S1");
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.last_input_tokens = 7499; // one below the 75% threshold

        agent.run("q3", &mut std::io::sink()).unwrap();

        assert!(log.borrow().is_empty());
        assert!(agent.compacted_summary.is_none());
        assert_eq!(agent.messages.len(), 8); // nothing dropped
    }

    #[test]
    fn run_over_threshold_without_droppable_prefix_skips_compaction() {
        // Only the keep-window's worth of groups exists: the guard fires but
        // nothing is safe to drop, so the turn proceeds uncompacted and the
        // oversized context rides until enough turns accumulate.
        let provider = MockProvider::new(vec![text_stream("next")]);
        let log = provider.send_log();
        let mut agent = Agent::new(Box::new(provider), test_config(None), vec![]);
        agent.config.context_token_limit = 10000;
        agent.messages.push(user_msg("q0"));
        agent.messages.push(assistant_msg("a0"));
        agent.messages.push(user_msg("q1"));
        agent.messages.push(assistant_msg("a1"));
        agent.last_input_tokens = 9000;

        agent.run("q2", &mut std::io::sink()).unwrap();

        assert!(log.borrow().is_empty());
        assert!(agent.compacted_summary.is_none());
        assert_eq!(agent.messages.len(), 6);
    }

    #[test]
    fn run_second_compaction_folds_prior_summary() {
        // Cumulative summarizing: the second compaction drops turn-groups
        // whose predecessors are already gone, so its sub-call must carry the
        // first summary — overwriting with a summary of only the new prefix
        // would silently forget the oldest context.
        let provider =
            MockProvider::new(vec![text_stream("r1"), text_stream("r2")]).with_send_text("S");
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);

        agent.last_input_tokens = 8000;
        agent.run("q3", &mut std::io::sink()).unwrap(); // first compaction
        agent.last_input_tokens = 8000; // re-arm: the next turn measured over
        agent.run("q4", &mut std::io::sink()).unwrap(); // second compaction

        let log = log.borrow();
        assert_eq!(log.len(), 2);
        // The second sub-call drops the now-oldest prefix (q1/a1)…
        assert_eq!(log[1].messages.len(), 3);
        assert_eq!(log[1].messages[0], user_msg("q1"));
        // …and its instruction folds in the summary from the first pass.
        let instruction = expect_text(&log[1].messages[2].content[0]);
        assert!(instruction.contains("already summarized"));
        assert!(instruction.contains("S"), "prior summary missing");
        assert_eq!(agent.compacted_summary.as_deref(), Some("S"));
    }

    #[test]
    fn run_summary_failure_fails_fast_without_consuming_input() {
        // Settled decision: a failed summary sub-call surfaces as an API
        // error rather than silently sending the known-over-limit request.
        // It fails before the new input is recorded, so nothing changes.
        let mut agent = compaction_agent(Box::new(ErrProvider), 10000);
        agent.last_input_tokens = 8000;

        let result = agent.run("q3", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(_))));
        assert_eq!(agent.messages.len(), 6); // all three groups intact, no "q3"
        assert!(agent.compacted_summary.is_none());
        assert_eq!(agent.last_input_tokens, 8000);
    }

    #[test]
    fn run_summary_transient_failure_retries_once_and_compacts() {
        // One transient blip must not abort the user's turn: the summary
        // sub-call is retried once in place, and the second attempt
        // compacts and lets the turn proceed as if nothing failed.
        let provider = MockProvider::new(vec![text_stream("next")])
            .with_send_text("S1")
            .with_send_failures(1);
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.last_input_tokens = 8000;

        agent.run("q3", &mut std::io::sink()).unwrap();

        assert_eq!(log.borrow().len(), 2); // the failed attempt plus the retry
        assert_eq!(agent.compacted_summary.as_deref(), Some("S1"));
        assert_eq!(agent.messages.len(), 6); // group 0 dropped, q3 exchange on
    }

    #[test]
    fn run_summary_failing_twice_surfaces_the_error() {
        // The retry is bounded at one: two consecutive failures surface the
        // existing error unchanged — no third attempt is made even though
        // this provider would have succeeded on it — and nothing is drained
        // or recorded.
        let provider = MockProvider::new(vec![])
            .with_send_text("S1")
            .with_send_failures(2);
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.last_input_tokens = 8000;

        let result = agent.run("q3", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(ApiError::Io(_)))));
        assert_eq!(log.borrow().len(), 2); // exactly one retry
        assert_eq!(agent.messages.len(), 6); // intact, no "q3"
        assert!(agent.compacted_summary.is_none());
        assert_eq!(agent.last_input_tokens, 8000);
    }

    #[test]
    fn run_truncated_summary_fails_fast_without_draining() {
        // A summary cut off at max_tokens (or diverted into a tool call) is
        // not a summary: committing it and draining would permanently lose
        // the dropped context. Same fail-fast as a transport failure —
        // including the one retry, which this shape also gets.
        let provider = MockProvider::new(vec![]).with_send_stop_reason(StopReason::MaxTokens);
        let log = provider.send_log();
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.last_input_tokens = 8000;

        let result = agent.run("q3", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(ApiError::Stream(_)))));
        assert_eq!(log.borrow().len(), 2); // the retry covers this shape too
        assert_eq!(agent.messages.len(), 6); // nothing drained, no "q3"
        assert!(agent.compacted_summary.is_none());
    }

    #[test]
    fn run_empty_summary_fails_fast_without_draining() {
        // A clean EndTurn that carries no usable text is equally not a
        // summary — an empty rolling summary would silently forget the
        // dropped context behind a bare header.
        let provider = MockProvider::new(vec![]).with_send_text("");
        let mut agent = compaction_agent(Box::new(provider), 10000);
        agent.last_input_tokens = 8000;

        let result = agent.run("q3", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(ApiError::Stream(_)))));
        assert_eq!(agent.messages.len(), 6);
        assert!(agent.compacted_summary.is_none());
    }

    #[test]
    fn run_rollback_after_compaction_keeps_compacted_history() {
        // Compaction succeeds, then the main stream errors before any side
        // effect. Compaction is a retained state change — the rollback
        // snapshot is taken after it — so the failed turn rolls back to the
        // *compacted* history, and the summary keeps only the text block of
        // the sub-call's response.
        let mut agent = compaction_agent(Box::new(SucceedThenErrProvider::new(vec![])), 10000);
        agent.last_input_tokens = 8000;

        let result = agent.run("q3", &mut std::io::sink());

        assert!(matches!(result, Err(AgentError::Api(_))));
        assert_eq!(
            agent.messages,
            vec![
                user_msg("q1"),
                assistant_msg("a1"),
                user_msg("q2"),
                assistant_msg("a2"),
            ]
        );
        assert_eq!(agent.compacted_summary.as_deref(), Some("prefix summary"));
        // The measurement rolls back to the post-compaction reset, not to 80.
        assert_eq!(agent.last_input_tokens, 0);
    }

    #[test]
    fn build_request_folds_summary_after_configured_system() {
        let mut agent = agent_with_system("You are helpful.");
        agent.compacted_summary = Some("old stuff".to_string());
        agent.messages.push(user_msg("hi"));

        let system = agent.build_request().system.unwrap();

        assert!(system.starts_with("You are helpful."));
        assert!(system.ends_with("## Earlier conversation (summarized)\nold stuff"));
    }

    #[test]
    fn compaction_recovers_a_prospective_request_at_the_actual_input_budget() {
        for (groups, reply_bytes, max_tokens, incoming_bytes) in
            [(2, 4000, 64, 8), (3, 1800, 5000, 8), (3, 2100, 64, 2000)]
        {
            let provider = MockProvider::new(vec![text_stream("done")]).with_send_text("summary");
            let summaries = provider.send_log();
            let streams = provider.stream_log();
            let mut config = test_config(None);
            config.context_token_limit = 10_000;
            config.max_tokens = max_tokens;
            let mut agent = Agent::new(Box::new(provider), config, vec![]);
            for i in 0..groups {
                agent.messages.push(user_msg(&format!("q{i}")));
                agent.messages.push(assistant_msg(&"a".repeat(reply_bytes)));
            }
            agent.last_input_tokens = 500;
            agent
                .run(&"q".repeat(incoming_bytes), &mut Vec::new())
                .unwrap();
            assert_eq!(summaries.borrow().len(), 1);
            assert_eq!(streams.borrow().len(), 1);
            let streams = streams.borrow();
            assert!(input_size(&streams[0]) <= input_budget(&streams[0], 10_000));
            assert_eq!(agent.compacted_summary.as_deref(), Some("summary"));
        }
    }

    #[test]
    fn oversized_summary_does_not_discard_original_history() {
        let provider = MockProvider::new(vec![]).with_send_text(&"s".repeat(10_000));
        let log = provider.stream_log();
        let mut agent = compaction_agent(Box::new(provider), 10_000);
        agent.last_input_tokens = 8000;
        let before = agent.session();
        assert!(matches!(
            agent.run("next", &mut Vec::new()),
            Err(AgentError::ContextLimit(10_000))
        ));
        assert_eq!(agent.session(), before);
        assert!(log.borrow().is_empty());
    }

    #[test]
    fn tool_loop_checks_large_real_file_before_continuing() {
        use crate::tools::{read_file::ReadFileTool, sandbox::Sandbox};
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("large.txt"), "word ".repeat(13_000)).unwrap();
        let mut first = tool_use_stream("read", "read_file");
        if let StreamDelta::ToolArgsDelta { json, .. } = &mut first[2] {
            *json = serde_json::json!({"path": "large.txt"}).to_string();
        }
        let provider = MockProvider::new(vec![first, text_stream("read a narrower range")]);
        let log = provider.stream_log();
        let mut config = test_config(None);
        config.context_token_limit = 8000;
        let mut agent = Agent::new(
            Box::new(provider),
            config,
            vec![Box::new(ReadFileTool::new(
                Sandbox::rooted(directory.path().into()).unwrap(),
            ))],
        );
        let mut output = Vec::new();
        agent.run("read large.txt", &mut output).unwrap();
        let log = log.borrow();
        assert_eq!(log.len(), 2);
        assert!(input_size(&log[1]) <= 6000);
        let (id, content, error) = expect_tool_result(&log[1].messages[2].content[0]);
        assert_eq!(id, "read");
        assert!(!error);
        assert!(content.contains("tool output shortened"));
        assert_eq!(expect_tool_use(&log[1].messages[1].content[0]).0, id);
        assert_eq!(
            expect_tool_result(&agent.session().messages[2].content[0])
                .1
                .len(),
            65_000
        );
        assert!(
            String::from_utf8(output)
                .unwrap()
                .contains("shortened 1 tool results in outgoing request")
        );
    }

    #[test]
    fn oversized_first_input_fails_before_network_and_rolls_back() {
        let provider = MockProvider::new(vec![]);
        let log = provider.stream_log();
        let mut agent = Agent::new(Box::new(provider), test_config(None), vec![]);
        let error = agent
            .run(&"x".repeat(100_000), &mut Vec::new())
            .unwrap_err();
        assert!(matches!(error, AgentError::ContextLimit(100_000)));
        assert!(error.to_string().contains("smaller read_file ranges"));
        assert!(log.borrow().is_empty());
        assert!(agent.session().messages.is_empty());
    }

    #[test]
    fn context_failure_preserves_tool_pairs_after_mutation_only() {
        for mutates in [false, true] {
            let first = text_and_tool_use_stream("call", "tool", &"x".repeat(8000));
            let provider = MockProvider::new(vec![first]);
            let log = provider.stream_log();
            let tool = if mutates {
                TestTool::new("tool", "changed").mutating()
            } else {
                TestTool::new("tool", "read")
            };
            let mut config = test_config(None);
            config.context_token_limit = 8000;
            let mut agent = Agent::new(Box::new(provider), config, vec![Box::new(tool)]);
            assert!(matches!(
                agent.run("work", &mut Vec::new()),
                Err(AgentError::ContextLimit(8000))
            ));
            assert_eq!(log.borrow().len(), 1);
            if mutates {
                let session = agent.session();
                assert_eq!(session.messages.len(), 3);
                assert_eq!(
                    expect_tool_result(&session.messages[2].content[0]).0,
                    "call"
                );
            } else {
                assert!(agent.session().messages.is_empty());
            }
        }
    }

    #[test]
    fn clear_resets_message_history() {
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("hi"));
        agent.messages.push(assistant_msg("hello"));
        assert_eq!(agent.messages.len(), 2);

        agent.clear();

        assert!(agent.messages.is_empty());
    }

    #[test]
    fn clear_on_empty_history_is_a_noop() {
        let mut agent = agent_with_tools(vec![]);
        agent.clear();
        assert!(agent.messages.is_empty());
    }

    #[test]
    fn clear_preserves_tool_budget() {
        // Budgets bound cost across the whole process, not per conversation —
        // clearing the history must not refill a spent tier count.
        let mut agent = agent_with_tools(vec![]);
        agent.budget.set_count(4, 5);

        agent.clear();

        assert_eq!(agent.budget.count(4), 5);
    }

    #[test]
    fn clear_resets_measured_prompt_size() {
        // The measurement describes the conversation it was taken from; a
        // fresh context must not inherit it, or the compaction guard would
        // fire off a context that no longer exists.
        let mut agent = agent_with_tools(vec![]);
        agent.last_input_tokens = 42;

        agent.clear();

        assert_eq!(agent.last_input_tokens, 0);
    }

    #[test]
    fn clear_resets_compacted_summary() {
        // Load-bearing: build_request folds the summary into the system
        // prompt, so a stale one would leak the previous conversation into
        // the fresh one.
        let mut agent = agent_with_tools(vec![]);
        agent.compacted_summary = Some("stale".to_string());

        agent.clear();

        assert!(agent.compacted_summary.is_none());
    }

    #[test]
    fn session_snapshots_conversation_state() {
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("hi"));
        agent.messages.push(assistant_msg("hello"));
        agent.compacted_summary = Some("earlier".to_string());
        agent.last_input_tokens = 42;

        let session = agent.session();

        assert_eq!(session.version, crate::session::SESSION_VERSION);
        assert_eq!(
            session.messages,
            vec![user_msg("hi"), assistant_msg("hello")]
        );
        assert_eq!(session.compacted_summary.as_deref(), Some("earlier"));
        assert_eq!(session.last_input_tokens, 42);
        // A snapshot, not a drain: the agent's own state is untouched.
        assert_eq!(agent.messages.len(), 2);
    }

    #[test]
    fn restore_round_trips_a_snapshot_into_a_fresh_agent() {
        let mut donor = agent_with_tools(vec![]);
        donor.messages.push(user_msg("hi"));
        donor.messages.push(assistant_msg("hello"));
        donor.compacted_summary = Some("earlier".to_string());
        donor.last_input_tokens = 42;

        let mut agent = agent_with_tools(vec![]);
        agent.restore(donor.session());

        assert_eq!(agent.session(), donor.session());
    }

    #[test]
    fn restore_replaces_the_current_conversation() {
        // Restoring over live state must not merge the two conversations.
        let mut agent = agent_with_tools(vec![]);
        agent.messages.push(user_msg("stale"));
        agent.compacted_summary = Some("stale summary".to_string());
        agent.last_input_tokens = 9;

        agent.restore(Session {
            version: crate::session::SESSION_VERSION,
            messages: vec![user_msg("restored")],
            compacted_summary: None,
            last_input_tokens: 0,
        });

        assert_eq!(agent.messages, vec![user_msg("restored")]);
        assert!(agent.compacted_summary.is_none());
        assert_eq!(agent.last_input_tokens, 0);
    }

    #[test]
    fn restore_strips_a_trailing_orphan_tool_use_keeping_partial_text() {
        // A pre-fix session persisted a trailing assistant message whose
        // tool_use was orphaned by a max_tokens cut-off. Restore drops the
        // unpaired tool_use but keeps the partial text, so the first
        // post-restore request is not 400ed.
        let mut agent = agent_with_tools(vec![]);
        agent.restore(Session {
            version: crate::session::SESSION_VERSION,
            messages: vec![
                user_msg("hi"),
                TurnMessage {
                    role: Role::Assistant,
                    content: vec![
                        Block::Text("partial".to_string()),
                        Block::ToolUse {
                            id: "call_1".to_string(),
                            name: "echo".to_string(),
                            input: serde_json::json!({}),
                        },
                    ],
                },
            ],
            compacted_summary: None,
            last_input_tokens: 0,
        });

        assert_eq!(
            agent.messages,
            vec![user_msg("hi"), assistant_msg("partial")]
        );
    }

    #[test]
    fn restore_drops_a_trailing_assistant_left_empty_by_the_orphan_strip() {
        // The trailing assistant message is a bare orphaned tool_use with no
        // text: stripping it leaves the message empty, so the whole message is
        // dropped rather than committing an empty (wire-rejected) content array.
        let mut agent = agent_with_tools(vec![]);
        agent.restore(Session {
            version: crate::session::SESSION_VERSION,
            messages: vec![
                user_msg("hi"),
                TurnMessage {
                    role: Role::Assistant,
                    content: vec![Block::ToolUse {
                        id: "call_1".to_string(),
                        name: "echo".to_string(),
                        input: serde_json::json!({}),
                    }],
                },
            ],
            compacted_summary: None,
            last_input_tokens: 0,
        });

        assert_eq!(agent.messages, vec![user_msg("hi")]);
    }

    #[test]
    fn restore_preserves_tool_budget() {
        // Same contract as clear: budgets are process state, not
        // conversation state, so a restore must not refill a spent tier.
        let mut agent = agent_with_tools(vec![]);
        agent.budget.set_count(4, 5);

        agent.restore(Session {
            version: crate::session::SESSION_VERSION,
            messages: vec![],
            compacted_summary: None,
            last_input_tokens: 0,
        });

        assert_eq!(agent.budget.count(4), 5);
    }

    #[test]
    fn restore_continues_the_conversation_with_summary_and_armed_guard() {
        // The acceptance shape for persistence: a restored session's next
        // turn runs over the restored history and summary, and the restored
        // measurement arms the compaction guard from turn one — 75 measured
        // tokens against a 100-token limit fires the summarizing sub-call
        // immediately, exactly as if the process had never restarted.
        let provider = MockProvider::new(vec![text_stream("next")]).with_send_text("S2");
        let log = provider.send_log();
        let mut agent = Agent::new(Box::new(provider), test_config(None), vec![]);
        agent.config.context_token_limit = 10000;

        agent.restore(Session {
            version: crate::session::SESSION_VERSION,
            messages: (0..3)
                .flat_map(|i| [user_msg(&format!("q{i}")), assistant_msg(&format!("a{i}"))])
                .collect(),
            compacted_summary: Some("S1".to_string()),
            last_input_tokens: 7500,
        });

        agent.run("q3", &mut std::io::sink()).unwrap();

        // The guard fired on the first post-restore turn; its sub-call
        // summarized the restored oldest turn-group and folded the restored
        // summary into the instruction.
        let log = log.borrow();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].messages[0], user_msg("q0"));
        let instruction = expect_text(&log[0].messages.last().unwrap().content[0]);
        assert!(instruction.contains("S1"), "got: {instruction}");
        assert_eq!(agent.compacted_summary.as_deref(), Some("S2"));
        // The turn itself ran over the restored (now compacted) history.
        assert_eq!(agent.messages[0], user_msg("q1"));
    }
}
