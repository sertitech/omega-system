//! The [`Provider`] seam. The agent loop talks to an LLM only through this
//! trait, over the normalized [`crate::turn`] types — never a vendor's wire
//! format. Each provider ships a thin adapter implementing it.

use crate::turn::{StreamDelta, Turn, TurnRequest};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

pub(crate) mod index_map;
pub mod transport;

/// A failure talking to a provider. Shared across adapters: both use `ureq` for
/// HTTP and `serde_json` for bodies, so the error shapes are the same.
#[derive(Debug)]
pub enum ApiError {
    /// A transport-level failure with no usable HTTP response — a connection
    /// reset, DNS failure, timeout, or protocol error. Distinct from
    /// [`ApiError::Status`], which *did* get an HTTP response carrying a non-2xx
    /// code and an error body.
    Http(Box<ureq::Error>),
    Json(serde_json::Error),
    Io(std::io::Error),
    /// The normalized request could not be expressed in the provider's wire
    /// format — e.g. a block/role pairing the agent should never produce. The
    /// adapter fails request construction rather than silently dropping it.
    InvalidRequest(String),
    /// A non-2xx HTTP response from the provider, after any retries were
    /// exhausted. Carries the status code and the provider's parsed error type +
    /// message (via [`error_message`]), so the user sees the real cause
    /// (`rate_limit_error: …`) instead of a bare status code.
    Status {
        code: u16,
        message: String,
    },
    /// The provider returned a 2xx but the response body carries no usable
    /// turn. Anthropic emits an `error` event mid-stream (e.g.
    /// `overloaded_error`, `api_error`); an OpenAI stream can end without a
    /// finish reason; a non-streaming OpenAI response can arrive with no
    /// `choices`. Either way the turn is incomplete — surfacing it here keeps
    /// the agent from reporting a degenerate response as a complete answer.
    Stream(String),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Http(e) => write!(f, "HTTP error: {e}"),
            ApiError::Json(e) => write!(f, "JSON error: {e}"),
            ApiError::Io(e) => write!(f, "IO error: {e}"),
            ApiError::InvalidRequest(msg) => write!(f, "invalid request: {msg}"),
            ApiError::Status { code, message } => write!(f, "HTTP {code}: {message}"),
            ApiError::Stream(msg) => write!(f, "stream error: {msg}"),
        }
    }
}

/// A provider's error body. Both vendors wrap the failure in a nested `error`
/// object — Anthropic as `{"type":"error","error":{"type",message}}`, OpenAI as
/// `{"error":{"message","type","code"}}` — so one permissive shape parses both.
/// The outer `type` and OpenAI's `code`/`param` are ignored; only the
/// human-readable `message` (and the machine `type`, when present) are kept.
#[derive(serde::Deserialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(serde::Deserialize)]
struct ErrorDetail {
    /// `rate_limit_error`, `overloaded_error`, … — always present on Anthropic,
    /// present but nullable on OpenAI. Optional so one type covers both.
    r#type: Option<String>,
    message: String,
}

/// Render a provider error body into one human-readable line: `"{type}: {message}"`
/// when a machine type is present, otherwise just the message. A body that
/// doesn't match the shared `{error:{type?,message}}` envelope is returned
/// trimmed and verbatim, so the failure detail is never swallowed. Shared by the
/// HTTP non-2xx path ([`transport::execute_with_retry`]) and the Anthropic
/// mid-stream `error` event.
pub(crate) fn error_message(body: &str) -> String {
    match serde_json::from_str::<ErrorBody>(body) {
        Ok(ErrorBody {
            error:
                ErrorDetail {
                    r#type: Some(t),
                    message,
                },
        }) => format!("{t}: {message}"),
        Ok(ErrorBody {
            error: ErrorDetail { message, .. },
        }) => message,
        Err(_) => body.trim().to_string(),
    }
}

impl From<ureq::Error> for ApiError {
    fn from(e: ureq::Error) -> Self {
        ApiError::Http(Box::new(e))
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(e: serde_json::Error) -> Self {
        ApiError::Json(e)
    }
}

impl From<std::io::Error> for ApiError {
    fn from(e: std::io::Error) -> Self {
        ApiError::Io(e)
    }
}

/// A normalized stream of deltas produced by [`Provider::stream`]. Boxed so the
/// trait stays object-safe and each adapter can return its own iterator type.
pub type DeltaStream = Box<dyn Iterator<Item = Result<StreamDelta, ApiError>>>;

/// An LLM provider behind the normalized turn interface. An implementor
/// translates a [`TurnRequest`] into its wire format, makes the call, and maps
/// the reply back into normalized types.
pub trait Provider {
    /// Make a single non-streaming request and return the completed turn.
    fn send(&self, request: &TurnRequest) -> Result<Turn, ApiError>;

    /// Make a streaming request and return an iterator of normalized deltas.
    fn stream(&self, request: &TurnRequest) -> Result<DeltaStream, ApiError>;

    /// List the model ids this provider currently serves — the data source
    /// for the REPL's model-id completion. The default returns an empty list,
    /// so a provider without a models endpoint (or a test double that never
    /// needs one) compiles unchanged and degrades to "no completions"; the
    /// Anthropic and OpenAI adapters override it with a `GET /v1/models`.
    fn list_models(&self) -> Result<Vec<String>, ApiError> {
        Ok(Vec::new())
    }
}

/// The providers the agent can be configured to use, deserialized from the
/// `provider` field of `config.json`. Adding a provider is a new variant here
/// plus matching arms in [`build`] and [`ProviderKind::from_name`] — the
/// compiler forces every other `match` to grow an arm, but `from_name`
/// matches on strings, so its arm must be kept in step by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Anthropic,
    Openai,
}

impl ProviderKind {
    /// Every provider, in the order the REPL's Tab completion lists them.
    /// Enumerated by hand, like [`ProviderKind::from_name`]'s arms — a new
    /// variant must be added to both (the round-trip test walks this array,
    /// pinning the two together).
    pub const ALL: [ProviderKind; 2] = [ProviderKind::Anthropic, ProviderKind::Openai];

    /// The environment variable that holds this provider's API key. The REPL
    /// resolves the key from here (process env, then `.env`) before building the
    /// provider, so the key never lands in `config.json`.
    pub fn api_key_env(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "ANTHROPIC_API_KEY",
            ProviderKind::Openai => "OPENAI_API_KEY",
        }
    }

    /// Parse the lowercase provider name — the one spelling `config.json`'s
    /// `provider` field accepts and [`std::fmt::Display`] renders, so the pair
    /// round-trips. Case-sensitive by design. Returns `None` for anything
    /// else; the caller decides what a non-name means (for `/model` argument
    /// parsing it means "the whole argument is a model id", not an error).
    pub fn from_name(name: &str) -> Option<ProviderKind> {
        match name {
            "anthropic" => Some(ProviderKind::Anthropic),
            "openai" => Some(ProviderKind::Openai),
            _ => None,
        }
    }
}

impl std::fmt::Display for ProviderKind {
    /// Renders the lowercase name the `provider` field uses in `config.json`, so
    /// the REPL's `/model` report shows the operator the same spelling they
    /// configured.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::Openai => "openai",
        };
        f.write_str(name)
    }
}

/// Construct the [`Provider`] selected by `kind`, owning its API key and the
/// shared turn-cancellation flag (polled during retry backoff, so a Ctrl-C
/// interrupts the wait). The single place that maps a configured provider to
/// its concrete adapter.
pub fn build(kind: ProviderKind, api_key: String, cancel: Arc<AtomicBool>) -> Box<dyn Provider> {
    match kind {
        ProviderKind::Anthropic => Box::new(
            crate::anthropic::provider_live::AnthropicProvider::new(api_key, cancel),
        ),
        ProviderKind::Openai => Box::new(crate::openai::provider_live::OpenAiProvider::new(
            api_key, cancel,
        )),
    }
}

/// The provider-construction seam: a cheaply-cloneable, `Send + Sync` handle
/// that resolves a provider's API key lazily and builds the provider on demand.
/// The REPL's `/model` switch holds one to swap providers mid-session (and to
/// list a non-active provider's models), and a later nested-agent tool will
/// hold the same handle to build a second provider from inside a `run()` call —
/// hence `Send + Sync` and `Clone`, both cheap over the shared `Arc`s.
///
/// **Key resolution stays lazy.** The resolver is *stored*, never called at
/// construction, so no key is read from the environment or `.env` until a
/// switch or spawn actually needs it — a provider the operator never selects
/// costs no lookup, and a key added to `.env` mid-session is still seen.
/// Builds the provider for a `(kind, key)`. Shared (`Arc`) so [`ProviderFactory`]
/// stays cheaply cloneable; `Send + Sync` so a nested-agent tool can hold it.
type BuildFn = Arc<dyn Fn(ProviderKind, String) -> Box<dyn Provider> + Send + Sync>;

/// Resolves an API key by environment-variable name, lazily. Same `Arc` +
/// `Send + Sync` rationale as [`BuildFn`].
type ResolveKeyFn = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

#[derive(Clone)]
pub struct ProviderFactory {
    /// In production a closure over [`build`] that threads the shared
    /// cancellation flag into each provider; tests inject a stub returning a mock.
    build: BuildFn,
    /// Production wires [`crate::load_env_var`]; tests inject a stub.
    resolve_key: ResolveKeyFn,
}

impl ProviderFactory {
    /// The production seam: build every provider through [`build`], threading
    /// the shared `cancel` flag into each (so a Ctrl-C interrupts its retry
    /// backoff), and resolve keys through `resolve_key` — typically
    /// [`crate::load_env_var`] — lazily, at switch/spawn time.
    pub fn new(
        cancel: Arc<AtomicBool>,
        resolve_key: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            build: Arc::new(move |kind, key| build(kind, key, Arc::clone(&cancel))),
            resolve_key: Arc::new(resolve_key),
        }
    }

    /// Resolve `env`'s API key, lazily — invoked only when a switch or spawn
    /// needs it, never eagerly at construction.
    pub fn resolve_key(&self, env: &str) -> Option<String> {
        (self.resolve_key)(env)
    }

    /// Build the provider selected by `kind`, handing it the already-resolved
    /// `key`.
    pub fn build(&self, kind: ProviderKind, key: String) -> Box<dyn Provider> {
        (self.build)(kind, key)
    }

    /// A seam over explicit factories — the injection point tests use to build a
    /// mock provider and drive a stub key resolver, keeping the switch hermetic
    /// (no network, no real environment). Production always goes through
    /// [`ProviderFactory::new`].
    #[cfg(test)]
    pub(crate) fn from_fns(
        build: impl Fn(ProviderKind, String) -> Box<dyn Provider> + Send + Sync + 'static,
        resolve_key: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            build: Arc::new(build),
            resolve_key: Arc::new(resolve_key),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_request_error_displays_message() {
        let err = ApiError::InvalidRequest("bad block".to_string());
        assert_eq!(err.to_string(), "invalid request: bad block");
    }

    #[test]
    fn transport_errors_convert_and_display() {
        // The three From impls wrap library errors verbatim; each Display arm
        // prefixes its source. Driven directly — the live paths that hit them
        // via `?` are excluded from coverage.
        let err: ApiError = ureq::Error::StatusCode(500).into();
        assert!(matches!(err, ApiError::Http(_)));
        assert!(err.to_string().starts_with("HTTP error: "));

        let json = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let err: ApiError = json.into();
        assert!(matches!(err, ApiError::Json(_)));
        assert!(err.to_string().starts_with("JSON error: "));

        let err: ApiError = std::io::Error::other("boom").into();
        assert!(matches!(err, ApiError::Io(_)));
        assert_eq!(err.to_string(), "IO error: boom");
    }

    #[test]
    fn stream_error_displays_message() {
        let err = ApiError::Stream("overloaded_error: Overloaded".to_string());
        assert_eq!(
            err.to_string(),
            "stream error: overloaded_error: Overloaded"
        );
    }

    #[test]
    fn status_error_displays_code_and_message() {
        let err = ApiError::Status {
            code: 429,
            message: "rate_limit_error: slow down".to_string(),
        };
        assert_eq!(err.to_string(), "HTTP 429: rate_limit_error: slow down");
    }

    #[test]
    fn error_message_parses_anthropic_envelope() {
        // Anthropic wraps the detail in an outer `{"type":"error", …}`; the outer
        // type is ignored and the nested type prefixes the message.
        let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        assert_eq!(error_message(body), "overloaded_error: Overloaded");
    }

    #[test]
    fn error_message_parses_openai_envelope() {
        // OpenAI carries extra `code`/`param` fields, which are ignored.
        let body = r#"{"error":{"message":"Incorrect API key","type":"invalid_request_error","code":"invalid_api_key"}}"#;
        assert_eq!(
            error_message(body),
            "invalid_request_error: Incorrect API key"
        );
    }

    #[test]
    fn error_message_without_type_is_just_the_message() {
        // OpenAI's `type` is nullable; a null type yields the bare message.
        let body = r#"{"error":{"message":"server had an error","type":null}}"#;
        assert_eq!(error_message(body), "server had an error");
    }

    #[test]
    fn error_message_falls_back_to_raw_body_when_unparseable() {
        // A body that isn't the shared envelope (HTML, plain text, truncated
        // JSON) is surfaced verbatim — never swallowed.
        assert_eq!(
            error_message("  <html>502 Bad Gateway</html>  "),
            "<html>502 Bad Gateway</html>"
        );
        assert_eq!(error_message(""), "");
    }

    #[test]
    fn provider_kind_deserializes_from_lowercase() {
        let kind: ProviderKind = serde_json::from_str("\"anthropic\"").unwrap();
        assert_eq!(kind, ProviderKind::Anthropic);
        let kind: ProviderKind = serde_json::from_str("\"openai\"").unwrap();
        assert_eq!(kind, ProviderKind::Openai);
    }

    #[test]
    fn api_key_env_per_provider() {
        assert_eq!(ProviderKind::Anthropic.api_key_env(), "ANTHROPIC_API_KEY");
        assert_eq!(ProviderKind::Openai.api_key_env(), "OPENAI_API_KEY");
    }

    #[test]
    fn from_name_parses_the_config_spelling() {
        assert_eq!(
            ProviderKind::from_name("anthropic"),
            Some(ProviderKind::Anthropic)
        );
        assert_eq!(
            ProviderKind::from_name("openai"),
            Some(ProviderKind::Openai)
        );
    }

    #[test]
    fn from_name_rejects_unknown_and_miscased_names() {
        // Case-sensitive: only the exact lowercase config spelling parses, so
        // `Anthropic:` in a `/model` argument reads as part of a model id.
        assert_eq!(ProviderKind::from_name("Anthropic"), None);
        assert_eq!(ProviderKind::from_name("OPENAI"), None);
        assert_eq!(ProviderKind::from_name("opneai"), None);
        assert_eq!(ProviderKind::from_name(""), None);
    }

    #[test]
    fn from_name_round_trips_with_display_for_all() {
        // Walking ALL keeps the three hand-enumerated lists — the array,
        // `from_name`'s arms, and `Display` — pinned to each other.
        for kind in ProviderKind::ALL {
            assert_eq!(ProviderKind::from_name(&kind.to_string()), Some(kind));
        }
    }

    #[test]
    fn display_matches_config_spelling() {
        // The rendered name round-trips with the lowercase `provider` field, so
        // the `/model` report echoes what the operator wrote in config.json.
        assert_eq!(ProviderKind::Anthropic.to_string(), "anthropic");
        assert_eq!(ProviderKind::Openai.to_string(), "openai");
    }

    #[test]
    fn build_constructs_the_selected_provider() {
        // Construction must not touch the network; building the provider and
        // referencing it as a trait object is enough to cover the factory arm.
        let provider = build(
            ProviderKind::Anthropic,
            "sk-test".to_string(),
            Arc::default(),
        );
        let _: &dyn Provider = provider.as_ref();
        let provider = build(ProviderKind::Openai, "sk-test".to_string(), Arc::default());
        let _: &dyn Provider = provider.as_ref();
    }

    #[test]
    fn provider_factory_is_send_sync() {
        // The nested-agent tool (a later step) will hold this across threads —
        // pin the guarantee so a non-`Send`/`Sync` field can't slip in.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ProviderFactory>();
    }

    #[test]
    fn production_factory_builds_and_resolves_lazily() {
        // `new` wires the real `build` (construction stays offline) over a
        // resolver that records each lookup — proving nothing is read until a
        // build actually asks for a key.
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let factory = {
            let seen = Arc::clone(&seen);
            ProviderFactory::new(Arc::default(), move |env: &str| {
                seen.lock().unwrap().push(env.to_string());
                Some(format!("key-for-{env}"))
            })
        };
        // No key touched at construction — laziness.
        assert!(seen.lock().unwrap().is_empty());

        let key = factory.resolve_key("ANTHROPIC_API_KEY").unwrap();
        assert_eq!(key, "key-for-ANTHROPIC_API_KEY");
        assert_eq!(seen.lock().unwrap().as_slice(), ["ANTHROPIC_API_KEY"]);

        let provider = factory.build(ProviderKind::Anthropic, key);
        let _: &dyn Provider = provider.as_ref();
    }

    #[test]
    fn stub_factory_builds_a_mock_and_reports_a_missing_key() {
        // The test seam: a stub builder returns a `MockProvider`, and the
        // resolver misses one key (fail-soft switch path) while hitting another.
        let factory = ProviderFactory::from_fns(
            |_kind, _key| Box::new(crate::testing::MockProvider::new(vec![])),
            |env: &str| (env == "OPENAI_API_KEY").then(|| "stub-key".to_string()),
        );
        assert_eq!(
            factory.resolve_key("OPENAI_API_KEY").as_deref(),
            Some("stub-key")
        );
        assert_eq!(factory.resolve_key("ANTHROPIC_API_KEY"), None);

        // A clone shares the same closures — the shape a held handle relies on.
        let cloned = factory.clone();
        let provider = cloned.build(ProviderKind::Openai, "stub-key".to_string());
        assert!(provider.list_models().unwrap().is_empty());
    }
}
