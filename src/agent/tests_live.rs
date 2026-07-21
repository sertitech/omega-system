//! The agent's `#[ignore]`'d live-API tests: full `run()` round-trips against
//! the real Anthropic API. Nothing here runs without network access and a real
//! key, so the `_live.rs` suffix marks the file for wholesale exclusion from
//! the coverage gate — the same loop logic is covered hermetically by the
//! `MockProvider` tests in the parent module.

use super::{Agent, AgentConfig};
use crate::anthropic::provider_live::AnthropicProvider;
use crate::tools::ToolDef;
use crate::turn::{Block, Role};
use std::sync::Arc;

#[test]
#[ignore = "hits the live Anthropic API; run with --ignored"]
fn agent_run_text_only() {
    let Some(key) = crate::load_env_var("ANTHROPIC_API_KEY") else {
        eprintln!("ANTHROPIC_API_KEY not set, skipping integration test");
        return;
    };

    let provider = AnthropicProvider::new(key, Arc::default());
    let config = AgentConfig {
        provider_kind: crate::provider::ProviderKind::Anthropic,
        model: crate::TEST_MODEL.to_string(),
        max_tokens: 64,
        system: None,
        context_token_limit: 100_000,
        effort: None,
        max_turns: super::MAX_TURNS,
    };
    let mut agent = Agent::new(Box::new(provider), config, vec![]);
    let text = agent
        .run("Reply with exactly: hello", &mut std::io::sink())
        .unwrap();
    assert!(
        text.to_lowercase().contains("hello"),
        "expected 'hello' in response, got: {text}"
    );
}

#[test]
#[ignore = "hits the live Anthropic API; run with --ignored"]
fn agent_run_with_tool_use() {
    let Some(key) = crate::load_env_var("ANTHROPIC_API_KEY") else {
        eprintln!("ANTHROPIC_API_KEY not set, skipping integration test");
        return;
    };

    struct EchoTool;
    impl ToolDef for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes the input message back"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo"
                    }
                },
                "required": ["message"]
            })
        }
        fn run(
            &self,
            input: serde_json::Value,
            _out: &mut dyn std::io::Write,
        ) -> Result<String, String> {
            Ok(input["message"]
                .as_str()
                .unwrap_or("no message")
                .to_string())
        }
    }

    let provider = AnthropicProvider::new(key, Arc::default());
    let config = AgentConfig {
        provider_kind: crate::provider::ProviderKind::Anthropic,
        model: crate::TEST_MODEL.to_string(),
        max_tokens: 256,
        system: Some("You have an echo tool. When asked to echo something, use the echo tool. After getting the result, report it.".to_string()),
        context_token_limit: 100_000,
        effort: None,
        max_turns: super::MAX_TURNS,
    };
    let mut agent = Agent::new(Box::new(provider), config, vec![Box::new(EchoTool)]);
    let text = agent
        .run(
            "Use the echo tool with the message 'test123'",
            &mut std::io::sink(),
        )
        .unwrap();

    // user → assistant (tool_use) → user (tool_result) → assistant (end_turn).
    assert!(
        agent.messages.len() >= 4,
        "expected at least 4 messages, got {}",
        agent.messages.len()
    );
    assert_eq!(agent.messages[0].role, Role::User);
    assert_eq!(agent.messages[1].role, Role::Assistant);
    assert_eq!(agent.messages[2].role, Role::User); // tool result
    assert_eq!(agent.messages[3].role, Role::Assistant); // final response

    assert!(
        text.contains("test123"),
        "expected 'test123' in final response, got: {text}"
    );
}

#[test]
#[ignore = "hits the live Anthropic API; run with --ignored"]
fn agent_run_extended_thinking_tool_loop() {
    // The acceptance probe: claude-fable-5 thinks *adaptively* by
    // default (no request parameter needed — live probe 2026-07-05: a
    // reasoning-shaped prompt returns thinking/text/tool_use; a trivial one
    // may skip the thinking block), so a tool-use turn produces thinking
    // blocks that must be parsed, kept in history with their signature, and
    // resent verbatim on the follow-up request — which the API rejects if
    // the signature is missing or altered. A completed multi-step loop is
    // therefore the proof of the whole path. The prompt asks for arithmetic
    // before the tool call to make the adaptive thinker actually think.
    let Some(key) = crate::load_env_var("ANTHROPIC_API_KEY") else {
        eprintln!("ANTHROPIC_API_KEY not set, skipping integration test");
        return;
    };

    struct EchoTool;
    impl ToolDef for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes the input message back"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo"
                    }
                },
                "required": ["message"]
            })
        }
        fn run(
            &self,
            input: serde_json::Value,
            _out: &mut dyn std::io::Write,
        ) -> Result<String, String> {
            Ok(input["message"]
                .as_str()
                .unwrap_or("no message")
                .to_string())
        }
    }

    let provider = AnthropicProvider::new(key, Arc::default());
    let config = AgentConfig {
        provider_kind: crate::provider::ProviderKind::Anthropic,
        // The thinking budget spends from max_tokens, so it needs headroom a
        // non-thinking turn would not.
        model: "claude-fable-5".to_string(),
        max_tokens: 8192,
        system: Some("You have an echo tool. When asked to echo something, use the echo tool. After getting the result, report it.".to_string()),
        context_token_limit: 200_000,
        effort: None,
        max_turns: super::MAX_TURNS,
    };
    let mut agent = Agent::new(Box::new(provider), config, vec![Box::new(EchoTool)]);
    let text = agent
        .run(
            "First work out 111*111 carefully, then use the echo tool with \
             the product as the message.",
            &mut std::io::sink(),
        )
        .unwrap();

    // user → assistant (thinking + tool_use) → user (tool_result) →
    // assistant (end_turn): the second round-trip succeeding means the API
    // accepted the resent thinking block.
    assert!(
        agent.messages.len() >= 4,
        "expected at least 4 messages, got {}",
        agent.messages.len()
    );
    assert_eq!(agent.messages[1].role, Role::Assistant);
    let signature = agent.messages[1]
        .content
        .iter()
        .find_map(|b| match b {
            Block::Thinking { signature, .. } => Some(signature),
            _ => None,
        })
        .expect("tool-use turn carries no thinking block");
    // The signature is the load-bearing half of the contract. The text may
    // legitimately be *empty*: the live stream (probe 2026-07-05) can carry a
    // thinking block as a bare `signature_delta` with no `thinking_delta` at
    // all, and the resend must reproduce that shape too.
    assert!(!signature.is_empty(), "thinking signature is empty");
    assert!(
        text.contains("12321"),
        "expected '12321' in final response, got: {text}"
    );
}
