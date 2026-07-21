//! Shared HTTP transport configuration for the provider adapters. The Anthropic
//! and OpenAI clients drive `ureq` identically, so the timeout policy lives here
//! once rather than being duplicated — and silently drifting — across two
//! clients.
//!
//! The streaming and non-streaming paths need *different* deadlines, so each
//! gets its own agent:
//!
//! - The **blocking** agent ([`blocking_agent`]) bounds the whole call with a
//!   global deadline. A non-streaming completion arrives in one shot, so a hard
//!   total cap is safe and catches a server that hangs partway through the body.
//! - The **streaming** agent ([`streaming_agent`]) deliberately omits any body
//!   deadline: a turn can legitimately stream for a long time, and a global cap
//!   would cut it off. It instead bounds the time to *receive the response
//!   headers* (first byte) — catching a server that accepts the connection but
//!   never starts responding — while letting the body stream freely.
//!
//! Both agents bound connection setup (DNS resolve + TCP/TLS connect) so a dead
//! host fails fast instead of hanging the REPL loop forever. Both also disable
//! `ureq`'s default "4xx/5xx is an `Err`" behavior so a non-2xx response comes
//! back as an `Ok` whose body [`execute_with_retry`] can read and classify.

use super::{ApiError, error_message};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Bound on DNS resolution. The connect timeout does not cover the resolve
/// phase, so it gets its own deadline.
const RESOLVE_TIMEOUT: Duration = Duration::from_secs(10);

/// Bound on TCP + TLS connection establishment. A stalled connect (a dead host,
/// a dropped SYN) fails here instead of blocking forever.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Whole-call deadline for the non-streaming [`blocking_agent`]. Generous enough
/// for a large completion, but a hard ceiling so a hung body read cannot wedge
/// the loop.
const GLOBAL_TIMEOUT: Duration = Duration::from_secs(120);

/// Deadline for the [`streaming_agent`] to receive the response *headers* (time
/// to first byte). Bounds a server that accepts the connection but never starts
/// responding, while leaving the response *body* unbounded so a long turn
/// streams to completion.
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(120);

/// Agent for non-streaming `send()` calls: connection bounds plus a whole-call
/// global deadline.
pub fn blocking_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_resolve(Some(RESOLVE_TIMEOUT))
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_global(Some(GLOBAL_TIMEOUT))
        .http_status_as_error(false)
        .build()
        .new_agent()
}

/// Agent for streaming `stream()` calls: connection bounds plus a time-to-first-
/// byte deadline, but **no** body or global deadline — a long turn must never be
/// cut off mid-stream.
pub fn streaming_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_resolve(Some(RESOLVE_TIMEOUT))
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_recv_response(Some(RESPONSE_TIMEOUT))
        .http_status_as_error(false)
        .build()
        .new_agent()
}

// ── Retry policy ──
//
// Transient provider failures (rate limits, brief overloads) are common for an
// agent that streams long turns, so a non-2xx with a retryable status is retried
// with bounded exponential backoff before the turn is abandoned. The policy is
// vendor-agnostic — only the error envelope differs, and [`error_message`]
// already unifies that — so it lives here once and both adapters inherit it.

/// Retries after the initial attempt for a retryable status, so up to
/// `MAX_RETRIES + 1` total attempts.
const MAX_RETRIES: u32 = 3;

/// Base of the exponential backoff schedule (`BASE_DELAY_MS * 2^attempt`), and
/// the width of the additive jitter window.
const BASE_DELAY_MS: u64 = 500;

/// Ceiling on a single computed backoff delay, so the exponential schedule
/// cannot grow without bound.
const MAX_DELAY_MS: u64 = 30_000;

/// Ceiling on a server-provided `Retry-After`, so a mistaken or hostile header
/// cannot wedge the loop for minutes.
const MAX_RETRY_AFTER_SECS: u64 = 60;

/// Length of one backoff sleep segment. The backoff sleep is sliced into
/// segments with a cancellation check between them, so a Ctrl-C lands within
/// one segment instead of after the whole delay (up to [`MAX_RETRY_AFTER_SECS`]
/// of it). 150 ms keeps cancellation feeling immediate while polling the flag
/// only a handful of times per second.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(150);

/// Ceiling on the bytes read from a non-2xx error body. The body is consumed
/// only to build a one-line message, so a small cap is ample — and on the
/// streaming path it matters: that agent deliberately leaves the body read
/// unbounded for long turns, so without a cap an oversized error
/// body would be read in full. A size cap bounds memory; it does **not** bound
/// time, so a server that trickles the error body byte-by-byte can still stall
/// this read — the same wall-clock trade-off the streaming *success* path
/// already accepts by design. Exceeding the cap surfaces as a read error, which
/// degrades to an empty message (the status code still carries the failure).
const ERROR_BODY_LIMIT: u64 = 64 * 1024;

/// Statuses worth retrying: 429 (rate limit), 500 (server error), the transient
/// gateway class 502/503/504 (a bad, overloaded, or timed-out upstream — all of
/// which a moment later may succeed), and 529 (Anthropic "overloaded").
/// Everything else (400/401/404/…) is a caller error that won't change on retry,
/// so it fails fast.
fn is_retryable(status: u16) -> bool {
    matches!(status, 429 | 500 | 502 | 503 | 504 | 529)
}

/// Parse a `Retry-After` header as integer seconds. The HTTP-date form is not
/// honored (yields `None`, falling back to computed backoff) — the LLM APIs send
/// integer seconds.
fn parse_retry_after(header: Option<&str>) -> Option<u64> {
    header?.trim().parse().ok()
}

/// The delay before the next attempt. A server `Retry-After` wins (capped at
/// [`MAX_RETRY_AFTER_SECS`]); otherwise an exponential backoff
/// `BASE_DELAY_MS * 2^attempt` (capped at [`MAX_DELAY_MS`]) plus up to
/// `BASE_DELAY_MS` of jitter, so concurrent retriers don't resynchronize.
fn backoff_delay(attempt: u32, retry_after: Option<u64>, jitter_seed: u64) -> Duration {
    if let Some(secs) = retry_after {
        return Duration::from_secs(secs.min(MAX_RETRY_AFTER_SECS));
    }
    let factor = 1u64 << attempt.min(20);
    let base = BASE_DELAY_MS.saturating_mul(factor).min(MAX_DELAY_MS);
    Duration::from_millis(base + jitter_seed % BASE_DELAY_MS)
}

/// Sub-second entropy for jitter, taken from the wall clock — no `rand` crate.
/// Only divergence between concurrent retriers matters, not the exact value.
fn jitter_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0)
}

/// What [`execute_with_retry`] needs from one HTTP attempt's response. Abstracted
/// over a trait so the retry loop is exercised hermetically with a fake response
/// rather than a live network round-trip.
pub trait RetryResponse {
    /// The HTTP status code (e.g. `200`, `429`).
    fn status_code(&self) -> u16;
    /// The `Retry-After` header parsed as integer seconds, if present and valid.
    fn retry_after_secs(&self) -> Option<u64>;
    /// Consume the response and read its body to a string (for the error path).
    fn into_body_string(self) -> String;
}

impl RetryResponse for ureq::http::Response<ureq::Body> {
    fn status_code(&self) -> u16 {
        self.status().as_u16()
    }

    fn retry_after_secs(&self) -> Option<u64> {
        parse_retry_after(
            self.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
        )
    }

    fn into_body_string(self) -> String {
        // Read at most ERROR_BODY_LIMIT bytes (see its doc for why a cap matters
        // on the streaming path). If the read fails — including the cap being
        // exceeded — the status code still carries the failure, so degrade to an
        // empty message rather than masking the status with an IO error.
        self.into_body()
            .with_config()
            .limit(ERROR_BODY_LIMIT)
            .read_to_string()
            .unwrap_or_default()
    }
}

/// The error the retry loop returns when a cancellation interrupts it. An
/// `Interrupted` IO error like any other transport failure: the agent sees the
/// still-raised flag alongside the failed turn and reports its quiet cancelled
/// outcome, so this message is not user-facing on the Ctrl-C path.
fn cancelled_error() -> ApiError {
    ApiError::Io(std::io::Error::new(
        std::io::ErrorKind::Interrupted,
        "cancelled during retry",
    ))
}

/// Run `attempt` with bounded exponential backoff. A 2xx response is returned
/// untouched; a retryable non-2xx ([`is_retryable`]) is retried up to
/// [`MAX_RETRIES`] times honoring `Retry-After`; a non-retryable or
/// retry-exhausted non-2xx is read and surfaced as [`ApiError::Status`]; and a
/// transport-level error fails fast. Each call to `attempt` must issue a fresh
/// request, since `ureq` consumes the request builder on send.
///
/// The retry budget is finite but lives *outside* the per-request transport
/// timeouts: worst-case wall-clock is `(MAX_RETRIES + 1)` attempts (each bounded
/// by the agent's own timeout) plus up to `MAX_RETRIES * MAX_RETRY_AFTER_SECS` of
/// backoff sleep. So a persistently-overloaded provider can still occupy the
/// loop for a few minutes — but not un-cancellably: `cancel` (the shared
/// turn-cancellation flag the SIGINT handler raises) is checked before each
/// attempt and between backoff sleep segments ([`CANCEL_POLL_INTERVAL`]), so a
/// Ctrl-C lands within one segment. On cancellation the loop returns
/// [`cancelled_error`]; the in-flight attempt itself is not interrupted — it
/// stays bounded by the agent's per-request timeouts.
pub fn execute_with_retry<R, E, F>(cancel: &AtomicBool, attempt: F) -> Result<R, ApiError>
where
    R: RetryResponse,
    E: Into<ApiError>,
    F: FnMut() -> Result<R, E>,
{
    retry_loop(attempt, std::thread::sleep, cancel)
}

/// The body of [`execute_with_retry`], with the sleep injected so tests drive
/// the loop deterministically and without real delays. The flag is polled with
/// `Relaxed` ordering — it is a lone boolean carrying no other memory, the
/// same discipline as the agent's seams.
fn retry_loop<R, E, F, S>(mut attempt: F, mut sleep: S, cancel: &AtomicBool) -> Result<R, ApiError>
where
    R: RetryResponse,
    E: Into<ApiError>,
    F: FnMut() -> Result<R, E>,
    S: FnMut(Duration),
{
    let mut retries = 0u32;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(cancelled_error());
        }
        let response = attempt().map_err(Into::into)?;
        let status = response.status_code();
        if (200..300).contains(&status) {
            return Ok(response);
        }
        if is_retryable(status) && retries < MAX_RETRIES {
            let mut remaining = backoff_delay(retries, response.retry_after_secs(), jitter_seed());
            while !remaining.is_zero() {
                let segment = remaining.min(CANCEL_POLL_INTERVAL);
                sleep(segment);
                remaining -= segment;
                if cancel.load(Ordering::Relaxed) {
                    return Err(cancelled_error());
                }
            }
            retries += 1;
            continue;
        }
        return Err(ApiError::Status {
            code: status,
            message: error_message(&response.into_body_string()),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocking_agent_bounds_connection_and_whole_call() {
        let timeouts = blocking_agent().config().timeouts();
        assert_eq!(timeouts.resolve, Some(RESOLVE_TIMEOUT));
        assert_eq!(timeouts.connect, Some(CONNECT_TIMEOUT));
        assert_eq!(timeouts.global, Some(GLOBAL_TIMEOUT));
    }

    #[test]
    fn streaming_agent_bounds_first_byte_but_not_the_body() {
        let timeouts = streaming_agent().config().timeouts();
        assert_eq!(timeouts.resolve, Some(RESOLVE_TIMEOUT));
        assert_eq!(timeouts.connect, Some(CONNECT_TIMEOUT));
        assert_eq!(timeouts.recv_response, Some(RESPONSE_TIMEOUT));
        // The body and the whole call must stay unbounded — either deadline
        // would cut off a legitimately long turn mid-stream.
        assert_eq!(timeouts.recv_body, None);
        assert_eq!(timeouts.global, None);
    }

    #[test]
    fn both_agents_surface_non_2xx_as_response_not_error() {
        // The retry path can only read an error body if a non-2xx comes back as
        // an `Ok` response, so status-as-error must be disabled on both agents.
        assert!(!blocking_agent().config().http_status_as_error());
        assert!(!streaming_agent().config().http_status_as_error());
    }

    // ── Retry policy ──

    #[test]
    fn is_retryable_accepts_only_transient_statuses() {
        // 502/504 join 503 as the transient-gateway class — all retryable.
        for status in [429, 500, 502, 503, 504, 529] {
            assert!(is_retryable(status), "{status} should be retryable");
        }
        // 501 (Not Implemented) is a definite server refusal, not transient.
        for status in [200, 400, 401, 403, 404, 422, 501] {
            assert!(!is_retryable(status), "{status} should not be retryable");
        }
    }

    #[test]
    fn parse_retry_after_reads_integer_seconds_only() {
        assert_eq!(parse_retry_after(Some("5")), Some(5));
        assert_eq!(parse_retry_after(Some("  10  ")), Some(10));
        assert_eq!(parse_retry_after(Some("0")), Some(0));
        // The HTTP-date form is not honored.
        assert_eq!(
            parse_retry_after(Some("Wed, 21 Oct 2015 07:28:00 GMT")),
            None
        );
        assert_eq!(parse_retry_after(Some("")), None);
        assert_eq!(parse_retry_after(None), None);
    }

    #[test]
    fn backoff_delay_honors_retry_after_over_exponential() {
        assert_eq!(
            backoff_delay(0, Some(3), 0),
            Duration::from_secs(3),
            "Retry-After wins, regardless of attempt"
        );
        assert_eq!(
            backoff_delay(2, Some(9999), 0),
            Duration::from_secs(MAX_RETRY_AFTER_SECS),
            "an oversized Retry-After is capped"
        );
    }

    #[test]
    fn backoff_delay_is_exponential_with_bounded_jitter() {
        // No Retry-After: BASE * 2^attempt, plus 0..BASE jitter.
        assert_eq!(backoff_delay(0, None, 0), Duration::from_millis(500));
        assert_eq!(backoff_delay(1, None, 0), Duration::from_millis(1000));
        assert_eq!(backoff_delay(2, None, 0), Duration::from_millis(2000));
        // Jitter is `seed % BASE`, so it stays within one base window.
        assert_eq!(backoff_delay(0, None, 499), Duration::from_millis(999));
        assert_eq!(backoff_delay(0, None, 500), Duration::from_millis(500));
    }

    #[test]
    fn backoff_delay_caps_the_exponential_growth() {
        // A high attempt would overflow the schedule; it saturates at the cap
        // (plus jitter) rather than growing unbounded or panicking on shift.
        assert_eq!(
            backoff_delay(60, None, 0),
            Duration::from_millis(MAX_DELAY_MS)
        );
    }

    #[test]
    fn jitter_seed_stays_within_a_second() {
        // Derived from sub-second nanos, so always < 1e9 — bounded entropy.
        assert!(jitter_seed() < 1_000_000_000);
    }

    /// A scripted response for driving [`retry_loop`] without the network.
    #[derive(Debug)]
    struct FakeResponse {
        code: u16,
        retry_after: Option<u64>,
        body: &'static str,
    }

    impl FakeResponse {
        fn status(code: u16) -> Self {
            Self {
                code,
                retry_after: None,
                body: "",
            }
        }
    }

    impl RetryResponse for FakeResponse {
        fn status_code(&self) -> u16 {
            self.code
        }
        fn retry_after_secs(&self) -> Option<u64> {
            self.retry_after
        }
        fn into_body_string(self) -> String {
            self.body.to_string()
        }
    }

    /// Unwrap the retry outcome as a Status error's `(code, message)`,
    /// panicking with the actual outcome otherwise. The panic arm is exercised
    /// by its own `#[should_panic]` test, so the helper carries no dead line.
    #[track_caller]
    fn expect_status(result: Result<FakeResponse, ApiError>) -> (u16, String) {
        match result {
            Err(ApiError::Status { code, message }) => (code, message),
            other => panic!("expected Status error, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "expected Status error")]
    fn expect_status_panics_on_ok() {
        expect_status(Ok(FakeResponse::status(200)));
    }

    /// Run [`retry_loop`] over a fixed script of attempt outcomes, returning
    /// the final result and the sleep segments the loop requested. `cancel` is
    /// the flag the loop polls; `cancel_after_sleeps` raises it once that many
    /// segments have been slept (`None` = never), so a test can cancel at an
    /// exact point mid-backoff.
    fn drive_cancelling_after(
        script: Vec<Result<FakeResponse, ApiError>>,
        cancel: &AtomicBool,
        cancel_after_sleeps: Option<usize>,
    ) -> (Result<FakeResponse, ApiError>, Vec<Duration>) {
        let mut script = script.into_iter();
        let mut delays = Vec::new();
        let result = retry_loop(
            || {
                script
                    .next()
                    .expect("attempt called more times than scripted")
            },
            |d| {
                delays.push(d);
                if Some(delays.len()) == cancel_after_sleeps {
                    cancel.store(true, Ordering::Relaxed);
                }
            },
            cancel,
        );
        (result, delays)
    }

    /// [`drive_cancelling_after`] with cancellation never raised — the plain
    /// retry scenarios.
    fn drive(
        script: Vec<Result<FakeResponse, ApiError>>,
    ) -> (Result<FakeResponse, ApiError>, Vec<Duration>) {
        drive_cancelling_after(script, &AtomicBool::new(false), None)
    }

    #[test]
    fn success_returns_immediately_without_sleeping() {
        let (result, delays) = drive(vec![Ok(FakeResponse::status(200))]);
        assert_eq!(result.unwrap().code, 200);
        assert!(delays.is_empty());
    }

    #[test]
    fn non_retryable_status_fails_without_retry() {
        let (result, delays) = drive(vec![Ok(FakeResponse {
            code: 401,
            retry_after: None,
            body: r#"{"error":{"type":"authentication_error","message":"bad key"}}"#,
        })]);
        let (code, message) = expect_status(result);
        assert_eq!(code, 401);
        assert_eq!(message, "authentication_error: bad key");
        assert!(delays.is_empty(), "a non-retryable status must not sleep");
    }

    #[test]
    fn retryable_status_retries_then_succeeds() {
        // Retry-After: Some(0) keeps the retry instant — a zero delay has no
        // segments to sleep, so the loop goes straight to the next attempt.
        let (result, delays) = drive(vec![
            Ok(FakeResponse {
                code: 503,
                retry_after: Some(0),
                body: "",
            }),
            Ok(FakeResponse::status(200)),
        ]);
        assert_eq!(result.unwrap().code, 200);
        assert!(delays.is_empty(), "a zero backoff must not sleep at all");
    }

    #[test]
    fn retries_are_exhausted_then_the_status_is_surfaced() {
        let attempt = || {
            Ok::<_, ApiError>(FakeResponse {
                code: 529,
                retry_after: Some(0),
                body: r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            })
        };
        let script = vec![attempt(), attempt(), attempt(), attempt()];
        let (result, delays) = drive(script);
        let (code, message) = expect_status(result);
        assert_eq!(code, 529);
        assert_eq!(message, "overloaded_error: Overloaded");
        // One initial attempt + MAX_RETRIES instant (zero-delay) backoffs
        // before giving up — the script length pins the attempt count.
        assert!(delays.is_empty());
    }

    #[test]
    fn exponential_backoff_used_when_no_retry_after() {
        // No Retry-After header: the delay comes from the exponential schedule,
        // so the slept segments sum into the first base window (BASE..2*BASE
        // with jitter), each segment within the cancellation poll interval.
        let (result, delays) = drive(vec![
            Ok(FakeResponse {
                code: 500,
                retry_after: None,
                body: "",
            }),
            Ok(FakeResponse::status(200)),
        ]);
        assert_eq!(result.unwrap().code, 200);
        let total: Duration = delays.iter().sum();
        assert!(total >= Duration::from_millis(BASE_DELAY_MS));
        assert!(total < Duration::from_millis(BASE_DELAY_MS * 2));
        assert!(delays.iter().all(|d| *d <= CANCEL_POLL_INTERVAL));
    }

    #[test]
    fn backoff_sleep_is_sliced_into_poll_segments() {
        // A 1 s Retry-After sleeps in CANCEL_POLL_INTERVAL slices (with one
        // short remainder), not one uninterruptible block — the slicing is
        // what gives Ctrl-C a seam to land in.
        let (result, delays) = drive(vec![
            Ok(FakeResponse {
                code: 429,
                retry_after: Some(1),
                body: "",
            }),
            Ok(FakeResponse::status(200)),
        ]);
        assert_eq!(result.unwrap().code, 200);
        assert!(delays.len() > 1, "a 1 s backoff must sleep in segments");
        assert_eq!(delays.iter().sum::<Duration>(), Duration::from_secs(1));
        assert!(delays.iter().all(|d| *d <= CANCEL_POLL_INTERVAL));
    }

    /// Assert the retry outcome is the interrupted cancellation error,
    /// panicking with the actual outcome otherwise. The panic arm is exercised
    /// by its own `#[should_panic]` test, so the helper carries no dead line.
    #[track_caller]
    fn expect_interrupted(result: Result<FakeResponse, ApiError>) {
        match result {
            Err(ApiError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::Interrupted),
            other => panic!("expected interrupted IO error, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "expected interrupted IO error")]
    fn expect_interrupted_panics_on_ok() {
        expect_interrupted(Ok(FakeResponse::status(200)));
    }

    #[test]
    fn cancellation_before_the_first_attempt_makes_no_request() {
        // Flag already raised when the loop starts: no attempt is issued at
        // all (the empty script would panic if one were) and the interrupted
        // error surfaces immediately.
        let (result, delays) = drive_cancelling_after(vec![], &AtomicBool::new(true), None);
        expect_interrupted(result);
        assert!(delays.is_empty());
    }

    #[test]
    fn cancellation_mid_backoff_stops_without_another_attempt() {
        // One retryable response starts a long (60 s Retry-After) backoff; the
        // flag flips after two slept segments. The loop must exit right there —
        // ~300 ms in, not 60 s later — without issuing the next attempt (the
        // one-entry script would panic if it did).
        let (result, delays) = drive_cancelling_after(
            vec![Ok(FakeResponse {
                code: 529,
                retry_after: Some(60),
                body: "",
            })],
            &AtomicBool::new(false),
            Some(2),
        );
        expect_interrupted(result);
        assert_eq!(delays, vec![CANCEL_POLL_INTERVAL; 2]);
    }

    #[test]
    fn cancelled_error_is_an_interrupted_io_error() {
        expect_interrupted(Err(cancelled_error()));
        // The full rendering, kind prefix included, for the (unreached in the
        // normal Ctrl-C flow) case where the message does surface.
        assert_eq!(
            cancelled_error().to_string(),
            "IO error: cancelled during retry"
        );
    }

    #[test]
    fn transport_error_fails_fast() {
        let (result, delays) = drive(vec![Err(ApiError::Io(std::io::Error::other("reset")))]);
        assert!(matches!(result, Err(ApiError::Io(_))));
        assert!(delays.is_empty());
    }

    /// A 200 attempt for the [`execute_with_retry`] wrapper tests: the success
    /// path calls it (covering its body); the cancelled path must not — its
    /// 200 leaking through as an `Ok` would fail that test's assertion.
    fn ok_200() -> Result<FakeResponse, ApiError> {
        Ok(FakeResponse::status(200))
    }

    #[test]
    fn execute_with_retry_wraps_the_real_sleep() {
        // The public entry point: a 200 returns without ever sleeping, so it
        // exercises the wrapper hermetically (no live network, no real delay).
        let cancel = AtomicBool::new(false);
        let result = execute_with_retry(&cancel, ok_200);
        assert_eq!(result.unwrap().code, 200);
    }

    #[test]
    fn execute_with_retry_observes_the_shared_flag() {
        // A raised flag short-circuits the wrapper before any attempt: the
        // interrupted error comes back instead of ok_200's success.
        let cancel = AtomicBool::new(true);
        expect_interrupted(execute_with_retry(&cancel, ok_200));
    }

    #[test]
    fn real_response_extracts_status_header_and_body() {
        // Cover the live `RetryResponse` adapter hermetically by handing it a
        // constructed response rather than a network one.
        let response = ureq::http::Response::builder()
            .status(503)
            .header("retry-after", "7")
            .body(ureq::Body::builder().data("upstream is busy"))
            .unwrap();
        assert_eq!(response.status_code(), 503);
        assert_eq!(response.retry_after_secs(), Some(7));
        assert_eq!(response.into_body_string(), "upstream is busy");
    }

    #[test]
    fn real_response_body_over_the_cap_degrades_to_empty() {
        // A body past ERROR_BODY_LIMIT makes the capped read error, which
        // `into_body_string` degrades to an empty message — the status code
        // still surfaces the failure upstream, nothing is swallowed silently.
        let oversized = "x".repeat(ERROR_BODY_LIMIT as usize * 2);
        let response = ureq::http::Response::builder()
            .status(500)
            .body(ureq::Body::builder().data(oversized))
            .unwrap();
        assert_eq!(response.into_body_string(), "");
    }
}
