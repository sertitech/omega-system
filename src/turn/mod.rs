//! The normalized turn model — the provider-agnostic vocabulary the agent loop
//! speaks. Each provider has a thin adapter that maps these types to and from
//! its own wire format (see [`crate::provider::Provider`]).
//!
//! These are deliberately free of vendor concerns: no `cache_control` (prompt
//! caching is an Anthropic wire detail the adapter applies, not a concept the
//! loop reasons about), and the serde derives on [`Role`], [`Block`], and
//! [`TurnMessage`] serve **on-disk session persistence**
//! ([`crate::session`]) only — a local durability format, not a wire one.
//! The wire mapping stays the adapter's job; the turn model still never
//! serializes to a *provider*.

/// Who produced a message.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Role {
    User,
    Assistant,
}

/// Why the model stopped generating this turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// The model finished its response normally.
    EndTurn,
    /// The model wants to call one or more tools.
    ToolUse,
    /// The response was cut off at the token limit.
    MaxTokens,
    /// A configured stop sequence was emitted.
    StopSequence,
}

/// Token accounting for one turn. `input_tokens` counts the *uncached*
/// prompt tokens — both adapters normalize to that (Anthropic reports it
/// directly; OpenAI folds cached tokens into its prompt count, so its
/// adapter subtracts them out) — and the full prompt size is the sum of
/// input and the cache counters. The cache counters are optional: `None`
/// when the provider did not report them (OpenAI never reports a cache
/// *write* — its prefix cache is automatic, with nothing to place or bill).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_creation_input_tokens: Option<u32>,
    pub cache_read_input_tokens: Option<u32>,
}

impl Usage {
    /// The full prompt size for this turn: uncached input plus both cache
    /// counters. Summed in `u64` because the three fields are untrusted
    /// provider counts — a raw `u32` sum could overflow (a debug panic, or a
    /// release wrap that silently disarms the compaction guard). This is the
    /// single measure both the usage line and the compaction guard read.
    pub fn prompt_size(&self) -> u64 {
        u64::from(self.input_tokens)
            + u64::from(self.cache_creation_input_tokens.unwrap_or(0))
            + u64::from(self.cache_read_input_tokens.unwrap_or(0))
    }
}

/// A unit of message content.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Block {
    /// Free text, from either the user or the model.
    Text(String),
    /// The model's extended-thinking reasoning, preserved for *continuity*
    /// rather than display: Anthropic's multi-turn contract requires thinking
    /// blocks to be resent verbatim — `signature` included — with the
    /// assistant message on tool-use turns, or the follow-up request is
    /// rejected. The signature is an opaque server-side integrity token; the
    /// agent never inspects it. Providers with no equivalent wire slot
    /// (OpenAI) skip the block on send, so history stays portable.
    Thinking { text: String, signature: String },
    /// The model's request to call a tool.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// The result of a tool call, sent back to the model. `is_error` is a plain
    /// `bool`: the agent always decides it deliberately, so there is no
    /// "absent" state to model (the adapter omits it on the wire when `false`).
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
}

/// One message in the conversation: a role and its content blocks.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TurnMessage {
    pub role: Role,
    pub content: Vec<Block>,
}

/// A tool the model may call, in provider-neutral form. Adapters rename fields
/// to each provider's spelling (Anthropic `input_schema`, OpenAI `parameters`).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Everything a provider needs to produce the next turn.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnRequest {
    pub model: String,
    pub max_tokens: u32,
    /// The system prompt as plain text; the adapter decides how to carry it
    /// (Anthropic top-level `system`, OpenAI a leading `developer` message).
    pub system: Option<String>,
    pub messages: Vec<TurnMessage>,
    /// Available tools. Empty means "no tools"; the adapter omits the field.
    pub tools: Vec<ToolSpec>,
    /// Optional reasoning-effort level. Opaque and unvalidated client-side,
    /// exactly like `model`: each adapter serializes it into its own wire
    /// location (Anthropic `output_config.effort`, OpenAI `reasoning.effort`)
    /// only when set, and the provider validates the value at request time —
    /// there is no client enum or registry. `None` leaves the request
    /// byte-identical to a no-effort turn.
    pub effort: Option<String>,
}

/// A completed, non-streamed assistant turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub blocks: Vec<Block>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

/// One normalized streaming event. Adapters translate their provider's SSE
/// dialect into this vocabulary; the agent accumulates a sequence of these into
/// finished [`Block`]s. The `index` fields key deltas to the block they extend,
/// so out-of-order tool-argument fragments still land in the right place.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamDelta {
    /// The stream opened; carries whatever usage is known up front (for
    /// Anthropic, the input and cache-token counts).
    MessageStart { usage: Usage },
    /// A new text block began at `index`, seeded with any initial text.
    TextStart { index: usize, text: String },
    /// A new thinking block began at `index`, seeded with whatever text and
    /// signature the start event carried (both usually empty on the wire).
    /// Only the Anthropic adapter emits the thinking deltas; the OpenAI
    /// stream has no equivalent events.
    ThinkingStart {
        index: usize,
        text: String,
        signature: String,
    },
    /// A new tool-use block began at `index`, with its id, name, and any seed
    /// input (usually an empty object; arguments arrive as `ToolArgsDelta`s).
    ToolUseStart {
        index: usize,
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Text appended to the text block at `index`.
    TextDelta { index: usize, text: String },
    /// Reasoning text appended to the thinking block at `index`.
    ThinkingDelta { index: usize, text: String },
    /// A signature fragment appended to the thinking block at `index`. The
    /// wire delivers it as trailing `signature_delta` events once the
    /// reasoning text is complete.
    SignatureDelta { index: usize, signature: String },
    /// A JSON fragment appended to the tool-use arguments at `index`.
    ToolArgsDelta { index: usize, json: String },
    /// The turn finished; carries the stop reason and the final usage.
    MessageDelta {
        stop_reason: Option<StopReason>,
        usage: Usage,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_size_sums_all_three_fields() {
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 7,
            cache_creation_input_tokens: Some(20),
            cache_read_input_tokens: Some(3),
        };
        assert_eq!(usage.prompt_size(), 123);
    }

    #[test]
    fn prompt_size_treats_absent_cache_counters_as_zero() {
        let usage = Usage {
            input_tokens: 50,
            output_tokens: 0,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        };
        assert_eq!(usage.prompt_size(), 50);
    }

    #[test]
    fn prompt_size_does_not_overflow_near_u32_max() {
        // Three untrusted `u32::MAX` counts would wrap a raw `u32` sum; the
        // `u64` widening must carry the full total instead.
        let usage = Usage {
            input_tokens: u32::MAX,
            output_tokens: 0,
            cache_creation_input_tokens: Some(u32::MAX),
            cache_read_input_tokens: Some(u32::MAX),
        };
        assert_eq!(usage.prompt_size(), u64::from(u32::MAX) * 3);
    }
}
