//! Request sizing before network I/O. Without a model tokenizer, UTF-8 bytes
//! provide a deliberately conservative estimate; a quarter of the context is
//! reserved for framing/tokenizer differences, and output has its own reserve.
//! Only tool-result text may be shortened. The complete session stays intact.

use super::{AgentError, COMPACT_THRESHOLD_PERCENT};
use crate::turn::{Block, Role, TurnMessage, TurnRequest};

const OMITTED: &str =
    "\n[tool output shortened to fit context; request a smaller range or narrower query]";

/// Estimated input size, including tool schemas and message structure. This is
/// a local byte budget, not an exact count from the provider's tokenizer.
pub(super) fn input_size(request: &TurnRequest) -> usize {
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
pub(super) fn input_budget(request: &TurnRequest, limit: u32) -> usize {
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
pub(super) fn fit_request(
    mut request: TurnRequest,
    limit: u32,
) -> Result<(TurnRequest, usize), AgentError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn::{Role, ToolSpec, TurnMessage};

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
}
