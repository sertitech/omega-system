pub mod edit_file;
pub mod list_directory;
pub mod read_file;
pub mod sandbox;
pub mod search_files;
pub mod shell;
pub mod shell_guardrails;
pub mod subagent;
pub mod web_fetch;
pub mod web_live;
pub mod web_search;
pub mod write_file;

use sandbox::Sandbox;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};

const MAX_RESPONSE_BYTES: usize = 100 * 1024; // 100 KB

/// Defines a tool the agent can use. Implementations provide the API schema
/// (name, description, input_schema) and the execution logic (run).
/// `Send + Sync` is a supertrait because the agent fans independent calls in
/// one turn out to scoped worker threads, which borrow the tool to `run` it.
pub trait ToolDef: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> serde_json::Value;

    /// Execute the tool. `out` is the live display sink — the same seam
    /// `Agent::run` exposes, pushed one level down: text written to it shows
    /// the operator progress *while* the call runs, in addition to (never
    /// instead of) the returned result the model sees. The agent loop hands
    /// an inline call the parent's own sink and a fanned-out call a private
    /// buffer it flushes in block order after the batch, so a tool may write
    /// freely without interleaving concerns. Most tools have nothing to show
    /// mid-run and ignore it; the `task` tool streams its child's output
    /// through it (see [`subagent::TaskTool`]).
    fn run(&self, input: serde_json::Value, out: &mut dyn std::io::Write)
    -> Result<String, String>;

    /// Pre-execution validation. Called before `run`. Return `Err` to reject
    /// the call and send the error back to the model as `is_error: true`.
    /// Override to add tool-specific input guards (e.g. reject nonsensical queries).
    fn validate(&self, _input: &serde_json::Value) -> Result<(), String> {
        Ok(())
    }

    /// Cost tier for budget tracking. Higher = more expensive.
    /// 0 = free (context/knowledge), 1 = cheap local read, 2 = local write,
    /// 3 = targeted network, 4 = expensive network (search).
    fn cost(&self) -> u8 {
        0
    }

    /// Whether this tool requires user confirmation before execution.
    /// A permanent safety gate for mutating/dangerous tools: static analysis
    /// of tool input is best-effort (see `shell_guardrails`), so the
    /// operator's approval of the exact action is the standing backstop.
    fn requires_confirmation(&self) -> bool {
        false
    }

    /// Whether this tool's `run` can mutate state outside the conversation
    /// (the filesystem, the shell environment). Distinct from
    /// `requires_confirmation`: that is a *safety gate*; this is a *mutation
    /// marker* the agent loop uses to decide whether a failed turn's history
    /// may be rolled back. A turn in which a side-effecting tool ran is kept
    /// verbatim so the model's record never diverges from disk. Override to
    /// `true` on mutating tools.
    ///
    /// Input-aware: most tools mutate (or not) regardless of arguments and
    /// ignore `input`, but a dispatching tool answers per call — the `task`
    /// tool is side-effecting only when its target profile grants a mutating
    /// tool (see [`subagent::TaskTool`]). The loop passes the call's own input.
    fn side_effecting(&self, _input: &serde_json::Value) -> bool {
        false
    }

    /// Optional human-readable status line shown before execution.
    /// Override to display tool-specific context (e.g. the URL being fetched).
    fn format_status(&self, _input: &serde_json::Value) -> Option<String> {
        None
    }

    /// Whether this tool draws a permit from the process-wide fan-out
    /// concurrency pool while it runs. Leaf tools do (the default `true`): the
    /// pool caps how many run at once. A tool that itself spawns a nested
    /// fan-out — the `task` tool — must override to `false`, so its worker
    /// never holds a permit while the child agent it drives acquires its own.
    /// Were it to gate, a batch of `task` calls filling the pool would leave no
    /// permit for their children's leaf tools, deadlocking; because leaf tools
    /// never run nested tools, exempting only the dispatching tool is sufficient
    /// to keep the pool deadlock-free.
    fn gates_concurrency(&self) -> bool {
        true
    }
}

/// Build the list of available tools. If `firecrawl_key` is provided, `web_search`
/// is included; otherwise it is silently omitted. `web_fetch` needs no key and is
/// always available. Filesystem tools are gated by the provided `Sandbox`.
/// `cancel` is the shared turn-cancellation flag: the shell tool polls it to
/// kill an in-flight command's process group on Ctrl-C (the other tools finish
/// their current call — they are bounded by their own timeouts and caps).
pub fn default_tools(
    sandbox: Sandbox,
    firecrawl_key: Option<String>,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Vec<Box<dyn ToolDef>> {
    let mut tools: Vec<Box<dyn ToolDef>> = Vec::new();

    if let Some(key) = firecrawl_key {
        tools.push(Box::new(web_search::WebSearchTool::new(key)));
    }

    // WebFetchTool needs no API key — always available.
    tools.push(Box::new(web_fetch::WebFetchTool::default()));

    // Filesystem tools — sandboxed to the provided root.
    tools.push(Box::new(read_file::ReadFileTool::new(sandbox.clone())));
    tools.push(Box::new(write_file::WriteFileTool::new(sandbox.clone())));
    tools.push(Box::new(edit_file::EditFileTool::new(sandbox.clone())));
    tools.push(Box::new(list_directory::ListDirectoryTool::new(
        sandbox.clone(),
    )));
    tools.push(Box::new(search_files::SearchFilesTool::new(
        sandbox.clone(),
    )));

    // Shell — guarded by the 8i guardrails and the confirmation gate.
    tools.push(Box::new(shell::ShellTool::new(sandbox, cancel)));

    tools
}

// ── Web transport seams ──
//
// The web tools' only genuinely-unreachable-without-network lines are the
// `ureq` calls themselves, so those sit behind these two seams — implemented
// for real in [`web_live`] (excluded from the coverage gate) and by canned
// stubs in the tools' hermetic tests. Everything around the call — input
// validation, status handling, capping, parsing, formatting — stays covered.
// `Send + Sync` matches the `ToolDef` supertrait bound the tool fan-out needs.

/// The one network call `web_fetch` makes: GET `url`, returning the HTTP
/// status and the body reader. Transport-level failures (DNS, connect, TLS)
/// surface as the error string.
pub(crate) trait HttpGet: Send + Sync {
    fn get(&self, url: &str) -> Result<(u16, Box<dyn std::io::Read>), String>;
}

/// The one network call `web_search` makes: POST a JSON `body` to `url` with
/// bearer `api_key`, returning the body reader.
pub(crate) trait HttpPostJson: Send + Sync {
    fn post_json(
        &self,
        url: &str,
        api_key: &str,
        body: &str,
    ) -> Result<Box<dyn std::io::Read>, String>;
}

// ── URL validation (SSRF protection for web_fetch) ──

/// Validate a URL for safe fetching. Rejects non-HTTP(S) schemes, URLs with
/// userinfo (`@`), and URLs that resolve to any non-global address (loopback,
/// private, link-local, CGNAT, documentation, multicast, reserved, …).
///
/// This pre-flight check fails bad URLs early with a clear message, but it is
/// only half the guard: DNS is resolved *again* at connect time, and a
/// rebinding server could answer differently then. The live transport
/// therefore screens that second resolution too — its resolver
/// (`web_live::GuardedResolver`) runs every address it returns, including
/// redirect targets', through the same [`check_resolved_addrs`], so the
/// socket can only ever reach a vetted address.
pub fn validate_url(url: &str) -> Result<(), String> {
    let scheme_end = url.find("://").ok_or("invalid URL: missing scheme")?;
    let scheme = &url[..scheme_end];
    if scheme != "http" && scheme != "https" {
        return Err(format!(
            "blocked URL scheme: {scheme} (only http and https allowed)"
        ));
    }

    let after_scheme = &url[scheme_end + 3..];

    // Extract authority (everything before the first `/`, `?`, or `#`).
    let authority = after_scheme.split('/').next().unwrap_or(after_scheme);
    let authority = authority.split('?').next().unwrap_or(authority);
    let authority = authority.split('#').next().unwrap_or(authority);

    // Reject URLs with userinfo (`@` in authority). The ambiguity between
    // userinfo and host makes hand-parsed URLs exploitable for SSRF bypass.
    if authority.contains('@') {
        return Err("blocked URL: userinfo (@) not allowed".to_string());
    }

    if authority.is_empty() {
        return Err("invalid URL: empty host".to_string());
    }

    // Handle bracket-enclosed IPv6 literals like [::1] or [::1]:8080.
    let host_for_resolve = if authority.starts_with('[') {
        // IPv6 literal: [addr] or [addr]:port
        let bracket_end = authority
            .find(']')
            .ok_or("invalid URL: unclosed bracket in IPv6 address")?;
        let addr = &authority[1..bracket_end];
        let port =
            if authority.len() > bracket_end + 1 && authority.as_bytes()[bracket_end + 1] == b':' {
                &authority[bracket_end + 2..]
            } else {
                "80"
            };
        format!("[{addr}]:{port}")
    } else if authority.contains(':') {
        authority.to_string()
    } else {
        format!("{authority}:80")
    };

    let addrs: Vec<_> = host_for_resolve
        .to_socket_addrs()
        .map_err(|e| format!("DNS resolution failed for {authority}: {e}"))?
        .collect();

    check_resolved_addrs(&addrs, authority)
}

/// Screen every resolved address. An empty (but `Ok`) resolution is rejected
/// explicitly — silently passing it would skip the address screen entirely.
/// Shared by [`validate_url`]'s pre-flight resolve and the live transport's
/// connect-time resolver (`web_live::GuardedResolver`) — one screen, both
/// resolutions, which is what closes the DNS-rebinding window.
pub(crate) fn check_resolved_addrs(addrs: &[SocketAddr], authority: &str) -> Result<(), String> {
    if addrs.is_empty() {
        return Err(format!(
            "DNS resolution returned no addresses for {authority}"
        ));
    }
    for addr in addrs {
        check_global_ip(addr.ip())?;
    }
    Ok(())
}

/// The SSRF screen: allow only globally-routable public addresses. The policy
/// is allow-by-exclusion — every non-global range is refused — so `web_fetch`
/// can reach the public internet and nothing else. "Not RFC1918" is too narrow
/// for SSRF: CGNAT, benchmarking, documentation, multicast, and reserved ranges
/// are equally off-limits.
fn check_global_ip(ip: IpAddr) -> Result<(), String> {
    let global = match ip {
        IpAddr::V4(v4) => is_global_v4(v4),
        // Any v6 form that carries an IPv4 host — IPv4-mapped, IPv4-compatible,
        // NAT64, or 6to4 — is judged by the v4 screen on that embedded address,
        // since that is what the socket ultimately reaches. Native-v6 policy
        // is applied first so a non-global parent allocation cannot be escaped
        // merely by putting an IPv4-looking value in its low bits.
        IpAddr::V6(v6) => {
            // Screen the IPv6 allocation first: local-use NAT64 carries an
            // IPv4-looking tail but the entire /48 is non-global, regardless
            // of the embedded address. Allowed transition forms then inherit
            // the IPv4 policy for the endpoint they ultimately reach.
            is_global_v6(v6)
                && match v6.to_ipv4_mapped().or_else(|| embedded_v4(v6)) {
                    Some(v4) => is_global_v4(v4),
                    None => true,
                }
        }
    };
    if global {
        Ok(())
    } else {
        Err(format!("blocked non-global IP: {ip}"))
    }
}

/// The IPv4 address an IPv6 address tunnels or translates, for the transition
/// forms that ultimately reach a v4 host — so the v4 screen, not the native-v6
/// screen, judges them. Covers the deprecated IPv4-compatible form
/// (`::a.b.c.d`, i.e. `::/96` minus the `::`/`::1` specials already caught as
/// unspecified/loopback), the NAT64 well-known prefix (`64:ff9b::/96`), and
/// 6to4 (`2002::/16`, whose gateway v4 sits in segments 1–2). IPv4-mapped
/// (`::ffff:0:0/96`) is decoded by the caller via the stdlib `to_ipv4_mapped`.
///
/// Without this, an allow-by-exclusion screen lets `::127.0.0.1`, `64:ff9b::`
/// NAT64, and 6to4 literals embedding a non-global v4 pass as "global".
fn embedded_v4(v6: Ipv6Addr) -> Option<Ipv4Addr> {
    let s = v6.segments();
    let octets =
        |hi: u16, lo: u16| Ipv4Addr::new((hi >> 8) as u8, hi as u8, (lo >> 8) as u8, lo as u8);
    // 6to4 (2002::/16) tunnels the gateway v4 in segments 1–2 (bits 16–48).
    if s[0] == 0x2002 {
        return Some(octets(s[1], s[2]));
    }
    // IPv4-compatible (::/96) and the NAT64 well-known prefix (64:ff9b::/96)
    // both carry the v4 in the low 32 bits (segments 6–7). `::` and `::1` are
    // excluded — already screened as unspecified/loopback.
    let high_zero = s[..6].iter().all(|&seg| seg == 0);
    let ipv4_compatible = high_zero && !v6.is_unspecified() && !v6.is_loopback();
    let nat64 = s[0] == 0x0064 && s[1] == 0xff9b && s[2..6].iter().all(|&seg| seg == 0);
    (ipv4_compatible || nat64).then(|| octets(s[6], s[7]))
}

/// Whether an IPv4 address is a globally-routable public address — i.e. in none
/// of the reserved non-global ranges (RFC 6890 and friends).
fn is_global_v4(v4: Ipv4Addr) -> bool {
    let [a, b, c, d] = v4.octets();
    let non_global = a == 0                       // 0.0.0.0/8      "this network" / unspecified
        || a == 10                                // 10.0.0.0/8     RFC1918 private
        || (a == 100 && (64..=127).contains(&b))  // 100.64.0.0/10  CGNAT
        || a == 127                               // 127.0.0.0/8    loopback
        || (a == 169 && b == 254)                 // 169.254.0.0/16 link-local
        || (a == 172 && (16..=31).contains(&b))   // 172.16.0.0/12  RFC1918 private
        // IANA IPv4 Special-Purpose Address Registry, snapshot 2025-10-09.
        // The two anycast addresses are the only globally reachable members.
        || (a == 192 && b == 0 && c == 0 && d != 9 && d != 10)
        || (a == 192 && b == 0 && c == 2)         // 192.0.2.0/24   documentation (TEST-NET-1)
        || (a == 192 && b == 88 && c == 99)       // 192.88.99.0/24 deprecated 6to4 relay anycast
        || (a == 192 && b == 168)                 // 192.168.0.0/16 RFC1918 private
        || (a == 198 && (b == 18 || b == 19))     // 198.18.0.0/15  benchmarking
        || (a == 198 && b == 51 && c == 100)      // 198.51.100.0/24 documentation (TEST-NET-2)
        || (a == 203 && b == 0 && c == 113)       // 203.0.113.0/24 documentation (TEST-NET-3)
        || a >= 224; // 224.0.0.0/4 multicast, 240.0.0.0/4 reserved, 255.255.255.255 broadcast
    !non_global
}

/// Whether an IPv6 address is a globally-routable public address. IPv4-mapped
/// and the other IPv4-embedding forms ([`embedded_v4`]) are handled by the
/// caller (delegated to [`is_global_v4`]); this first screens the allocation
/// represented by the IPv6 address itself.
fn is_global_v6(v6: Ipv6Addr) -> bool {
    let s = v6.segments();
    let value = u128::from(v6);

    // IANA IPv6 Special-Purpose Address Registry, snapshot 2025-10-09.
    // 2001::/23 is non-global except for its explicitly global suballocations.
    let protocol_assignment_exception = value == 0x2001_0001_0000_0000_0000_0000_0000_0001
        || value == 0x2001_0001_0000_0000_0000_0000_0000_0002
        || value == 0x2001_0001_0000_0000_0000_0000_0000_0003
        || ipv6_has_prefix(value, 0x2001_0003_0000_0000_0000_0000_0000_0000, 32)
        || ipv6_has_prefix(value, 0x2001_0004_0112_0000_0000_0000_0000_0000, 48)
        || ipv6_has_prefix(value, 0x2001_0020_0000_0000_0000_0000_0000_0000, 28)
        || ipv6_has_prefix(value, 0x2001_0030_0000_0000_0000_0000_0000_0000, 28);
    let non_global = v6.is_loopback()             // ::1
        || v6.is_unspecified()                    // ::
        || ipv6_has_prefix(value, 0x0064_ff9b_0001_0000_0000_0000_0000_0000, 48) // local-use NAT64
        || ipv6_has_prefix(value, 0x0100_0000_0000_0000_0000_0000_0000_0000, 64) // discard-only
        || ipv6_has_prefix(value, 0x0100_0000_0000_0001_0000_0000_0000_0000, 64) // dummy prefix
        || (ipv6_has_prefix(value, 0x2001_0000_0000_0000_0000_0000_0000_0000, 23)
            && !protocol_assignment_exception)
        || ipv6_has_prefix(value, 0x2001_0db8_0000_0000_0000_0000_0000_0000, 32) // documentation
        || ipv6_has_prefix(value, 0x3fff_0000_0000_0000_0000_0000_0000_0000, 20) // documentation
        || ipv6_has_prefix(value, 0x5f00_0000_0000_0000_0000_0000_0000_0000, 16) // SRv6 SIDs
        || (s[0] & 0xfe00) == 0xfc00              // fc00::/7   unique-local (RFC1918 equivalent)
        || (s[0] & 0xffc0) == 0xfe80              // fe80::/10  link-local
        || (s[0] & 0xffc0) == 0xfec0              // fec0::/10  deprecated site-local
        || (s[0] & 0xff00) == 0xff00; // ff00::/8   multicast
    !non_global
}

/// Whether the IPv6 integer `address` begins with `prefix_len` bits of
/// `prefix`. The registry is expressed as CIDR blocks, so keeping this helper
/// beside the policy makes additions and exception boundaries auditable.
fn ipv6_has_prefix(address: u128, prefix: u128, prefix_len: u32) -> bool {
    let shift = 128 - prefix_len;
    address >> shift == prefix >> shift
}

/// Notice appended to a body that was truncated at the response cap.
const TRUNCATION_NOTICE: &str = "\n\n[truncated at 100 KB]";

/// Truncate a response body to `MAX_RESPONSE_BYTES`, appending a notice if truncated.
pub fn truncate_response(body: String) -> String {
    if body.len() <= MAX_RESPONSE_BYTES {
        return body;
    }
    truncate_with_notice(body)
}

/// Trim `body` to at most `MAX_RESPONSE_BYTES` at a UTF-8 boundary and append the
/// truncation notice — always signalling truncation, even when `body` is already
/// within the cap. `web_fetch` needs this because [`read_capped`] can drop a
/// boundary-straddling char and pull an *overflowed* body back under the cap,
/// making its length an unreliable truncation signal; there the overflow flag is
/// the source of truth.
pub fn truncate_with_notice(mut body: String) -> String {
    let mut end = MAX_RESPONSE_BYTES.min(body.len());
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    body.truncate(end);
    body.push_str(TRUNCATION_NOTICE);
    body
}

/// Read an HTTP response body from `reader`, decoding as UTF-8 (lossily —
/// invalid bytes become U+FFFD) and reading at most `cap + 1` bytes. Returns
/// the decoded body together with whether the source *overflowed* the cap
/// (had more than `cap` bytes).
///
/// This is the shared network read for `web_fetch` and `web_search`. Unlike
/// ureq's `.limit()` — which *errors* past the cap and so hard-fails the whole
/// read — it caps cleanly and reports overflow, leaving each caller to degrade
/// per its own semantics: `web_fetch` truncates-with-notice via
/// [`truncate_with_notice`], while `web_search` (a JSON consumer that cannot use a
/// half-parsed body) fails fast with an actionable error. The `+ 1` byte is what
/// makes overflow detectable. Mirrors the `read_file`/`shell` cap-and-read.
pub fn read_capped(reader: &mut dyn std::io::Read, cap: usize) -> Result<(String, bool), String> {
    use std::io::Read;

    let mut buf = Vec::new();
    reader
        // saturating: `cap` is a free parameter, and `cap as u64 + 1` would
        // overflow to 0 (silent empty read) at `usize::MAX`.
        .take((cap as u64).saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|e| format!("failed to read response body: {e}"))?;

    let overflowed = buf.len() > cap;

    // `take(cap + 1)` can slice a multi-byte UTF-8 character at the boundary.
    // That split is our own artifact, not source malformation, so when we
    // *overflowed* and the first fault is the incomplete *trailing* sequence
    // (`error_len` is None), drop it rather than decode it into a spurious
    // U+FFFD. Everything else — genuinely invalid bytes anywhere in the
    // stream — decodes lossily, the project-wide non-UTF-8 policy shared with
    // `shell` and `read_file`.
    if overflowed
        && let Err(e) = std::str::from_utf8(&buf)
        && e.error_len().is_none()
    {
        buf.truncate(e.valid_up_to());
    }
    let body = String::from_utf8_lossy(&buf).into_owned();

    Ok((body, overflowed))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ToolDef defaults ──

    /// A tool overriding only the four required methods, so the trait's
    /// default bodies are pinned (and covered) through at least one impl
    /// regardless of how the concrete tools override them.
    struct BareTool;
    impl ToolDef for BareTool {
        fn name(&self) -> &str {
            "bare"
        }
        fn description(&self) -> &str {
            "defaults only"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        fn run(
            &self,
            _input: serde_json::Value,
            _out: &mut dyn std::io::Write,
        ) -> Result<String, String> {
            Ok("ok".to_string())
        }
    }

    #[test]
    fn tool_def_defaults_are_permissive_and_free() {
        let tool = BareTool;
        assert_eq!(tool.name(), "bare");
        assert_eq!(tool.description(), "defaults only");
        assert_eq!(tool.input_schema()["type"], "object");
        assert_eq!(
            tool.run(serde_json::json!({}), &mut std::io::sink())
                .unwrap(),
            "ok"
        );
        assert!(tool.validate(&serde_json::json!({})).is_ok());
        assert_eq!(tool.cost(), 0);
        assert!(!tool.requires_confirmation());
        assert!(!tool.side_effecting(&serde_json::json!({})));
        assert_eq!(tool.format_status(&serde_json::json!({})), None);
        // Leaf tools gate the fan-out concurrency pool by default.
        assert!(tool.gates_concurrency());
    }

    // ── validate_url ──

    #[test]
    fn validate_url_accepts_http() {
        assert!(validate_url("http://example.com/path").is_ok());
    }

    #[test]
    fn validate_url_accepts_https() {
        assert!(validate_url("https://example.com/path?q=1").is_ok());
    }

    #[test]
    fn validate_url_rejects_file_scheme() {
        let err = validate_url("file:///etc/passwd").unwrap_err();
        assert!(err.contains("blocked URL scheme"));
    }

    #[test]
    fn validate_url_rejects_ftp_scheme() {
        let err = validate_url("ftp://example.com").unwrap_err();
        assert!(err.contains("blocked URL scheme"));
    }

    #[test]
    fn validate_url_rejects_data_scheme() {
        let err = validate_url("data:text/html,<h1>hi</h1>").unwrap_err();
        assert!(err.contains("blocked URL scheme") || err.contains("missing scheme"));
    }

    #[test]
    fn validate_url_rejects_missing_scheme() {
        let err = validate_url("example.com/path").unwrap_err();
        assert!(err.contains("missing scheme"));
    }

    #[test]
    fn validate_url_rejects_empty_host() {
        let err = validate_url("http:///path").unwrap_err();
        assert!(err.contains("empty host"));
    }

    #[test]
    fn validate_url_rejects_ipv6_loopback_with_port() {
        // The bracketed-IPv6 branch that carries an explicit port.
        let err = validate_url("http://[::1]:8080/admin").unwrap_err();
        assert!(err.contains("blocked"));
    }

    #[test]
    fn validate_url_surfaces_resolution_failure() {
        // A non-numeric port fails to_socket_addrs before any network I/O, so
        // the resolution error path is hermetic.
        let err = validate_url("http://example.com:notaport/").unwrap_err();
        assert!(err.contains("DNS resolution failed"));
    }

    // ── check_resolved_addrs ──

    #[test]
    fn check_resolved_addrs_rejects_empty_resolution() {
        // An empty (but Ok) resolve must fail closed — passing it would skip
        // the address screen entirely.
        let err = check_resolved_addrs(&[], "example.com").unwrap_err();
        assert!(err.contains("no addresses"));
    }

    #[test]
    fn check_resolved_addrs_screens_every_address() {
        let public: SocketAddr = "93.184.216.34:80".parse().unwrap();
        let private: SocketAddr = "10.0.0.1:80".parse().unwrap();
        assert!(check_resolved_addrs(&[public], "example.com").is_ok());
        // One private address among many poisons the whole set.
        assert!(check_resolved_addrs(&[public, private], "example.com").is_err());
    }

    // ── check_global_ip (IPv4) ──

    #[test]
    fn check_global_ip_rejects_loopback() {
        assert!(check_global_ip("127.0.0.1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_10_range() {
        assert!(check_global_ip("10.0.0.1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_172_16_range() {
        assert!(check_global_ip("172.16.0.1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_allows_172_32() {
        assert!(check_global_ip("172.32.0.1".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_rejects_192_168() {
        assert!(check_global_ip("192.168.1.1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_link_local() {
        assert!(check_global_ip("169.254.1.1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_unspecified() {
        assert!(check_global_ip("0.0.0.0".parse().unwrap()).is_err());
    }

    // Ranges the "not RFC1918" screen used to let through — every one must now
    // be blocked. Boundaries (both sides of each /nn) are checked too.

    #[test]
    fn check_global_ip_rejects_cgnat() {
        // 100.64.0.0/10 — carrier-grade NAT.
        assert!(check_global_ip("100.64.0.1".parse().unwrap()).is_err());
        assert!(check_global_ip("100.127.255.254".parse().unwrap()).is_err());
        // Just outside /10 on both sides stays global.
        assert!(check_global_ip("100.63.255.255".parse().unwrap()).is_ok());
        assert!(check_global_ip("100.128.0.0".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_rejects_benchmarking() {
        // 198.18.0.0/15 — network device benchmarking.
        assert!(check_global_ip("198.18.0.1".parse().unwrap()).is_err());
        assert!(check_global_ip("198.19.255.254".parse().unwrap()).is_err());
        assert!(check_global_ip("198.17.255.255".parse().unwrap()).is_ok());
        assert!(check_global_ip("198.20.0.0".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_rejects_documentation_ranges() {
        // The three TEST-NET documentation ranges.
        assert!(check_global_ip("192.0.2.1".parse().unwrap()).is_err());
        assert!(check_global_ip("198.51.100.1".parse().unwrap()).is_err());
        assert!(check_global_ip("203.0.113.1".parse().unwrap()).is_err());
        // Adjacent addresses outside the /24s remain global.
        assert!(check_global_ip("192.0.3.1".parse().unwrap()).is_ok());
        assert!(check_global_ip("203.0.114.1".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_rejects_ipv4_protocol_assignments_except_anycast() {
        // 192.0.0.0/24 is non-global except for the PCP and TURN anycast
        // addresses explicitly marked globally reachable by IANA.
        assert!(check_global_ip("192.0.0.1".parse().unwrap()).is_err());
        assert!(check_global_ip("192.0.0.8".parse().unwrap()).is_err());
        assert!(check_global_ip("192.0.0.9".parse().unwrap()).is_ok());
        assert!(check_global_ip("192.0.0.10".parse().unwrap()).is_ok());
        assert!(check_global_ip("192.0.0.11".parse().unwrap()).is_err());
        assert!(check_global_ip("192.0.0.170".parse().unwrap()).is_err());
        assert!(check_global_ip("192.0.1.0".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_rejects_deprecated_6to4_relay_anycast() {
        assert!(check_global_ip("192.88.99.1".parse().unwrap()).is_err());
        assert!(check_global_ip("192.88.99.2".parse().unwrap()).is_err());
        assert!(check_global_ip("192.88.98.255".parse().unwrap()).is_ok());
        assert!(check_global_ip("192.88.100.0".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_rejects_multicast() {
        // 224.0.0.0/4.
        assert!(check_global_ip("224.0.0.1".parse().unwrap()).is_err());
        assert!(check_global_ip("239.255.255.255".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_reserved_and_broadcast() {
        // 240.0.0.0/4 reserved, and the limited-broadcast 255.255.255.255.
        assert!(check_global_ip("240.0.0.1".parse().unwrap()).is_err());
        assert!(check_global_ip("255.255.255.255".parse().unwrap()).is_err());
        // 223.255.255.255 is the last unicast address below multicast.
        assert!(check_global_ip("223.255.255.255".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_allows_public() {
        assert!(check_global_ip("8.8.8.8".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_error_names_non_global() {
        // The message no longer implies every blocked address is "private".
        let err = check_global_ip("100.64.0.1".parse().unwrap()).unwrap_err();
        assert!(err.contains("blocked non-global IP"), "got: {err}");
    }

    // ── check_global_ip (IPv6) ──

    #[test]
    fn check_global_ip_rejects_ipv6_loopback() {
        assert!(check_global_ip("::1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_ipv6_unspecified() {
        assert!(check_global_ip("::".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_ipv6_unique_local() {
        assert!(check_global_ip("fd12::1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_ipv6_link_local() {
        assert!(check_global_ip("fe80::1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_ipv6_multicast() {
        // ff00::/8 — IPv6 multicast is never a valid fetch target.
        assert!(check_global_ip("ff02::1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_ipv6_documentation() {
        // 2001:db8::/32 — the reserved documentation prefix.
        assert!(check_global_ip("2001:db8::1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_local_use_nat64() {
        // Unlike the globally reachable well-known /96, the whole local-use
        // /48 is non-global even when its tail resembles a public IPv4.
        assert!(check_global_ip("64:ff9b:1::7f00:1".parse().unwrap()).is_err());
        assert!(check_global_ip("64:ff9b:1:1234::1".parse().unwrap()).is_err());
        assert!(check_global_ip("64:ff9b:0:ffff:ffff:ffff:ffff:ffff".parse().unwrap()).is_ok());
        assert!(check_global_ip("64:ff9b:2::".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_rejects_discard_and_dummy_prefixes() {
        assert!(check_global_ip("100::1".parse().unwrap()).is_err());
        assert!(check_global_ip("100::ffff:ffff:ffff:ffff".parse().unwrap()).is_err());
        assert!(check_global_ip("100:0:0:1::1".parse().unwrap()).is_err());
        // The closest addresses outside the consecutive /64 allocations.
        assert!(check_global_ip("ff:ffff:ffff:ffff:ffff:ffff:ffff:ffff".parse().unwrap()).is_ok());
        assert!(check_global_ip("100:0:0:2::".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_screens_ipv6_protocol_assignments_and_exceptions() {
        // 2001::/23 defaults to non-global.
        assert!(check_global_ip("2001::1".parse().unwrap()).is_err());
        assert!(check_global_ip("2001:2::1".parse().unwrap()).is_err()); // benchmarking
        assert!(check_global_ip("2001:10::1".parse().unwrap()).is_err()); // deprecated ORCHID

        // Current globally reachable exceptions inside that parent allocation.
        assert!(check_global_ip("2001:1::1".parse().unwrap()).is_ok());
        assert!(check_global_ip("2001:1::2".parse().unwrap()).is_ok());
        assert!(check_global_ip("2001:1::3".parse().unwrap()).is_ok());
        assert!(check_global_ip("2001:3:ffff::1".parse().unwrap()).is_ok());
        assert!(check_global_ip("2001:4:112:ffff::1".parse().unwrap()).is_ok());
        assert!(check_global_ip("2001:20::1".parse().unwrap()).is_ok());
        assert!(check_global_ip("2001:3f:ffff::1".parse().unwrap()).is_ok());

        // Addresses immediately beside exact and prefix exceptions stay blocked.
        assert!(check_global_ip("2001:1::4".parse().unwrap()).is_err());
        assert!(check_global_ip("2001:4:111:ffff::1".parse().unwrap()).is_err());
        assert!(check_global_ip("2001:4:113::1".parse().unwrap()).is_err());
        assert!(check_global_ip("2001:1f:ffff::1".parse().unwrap()).is_err());
        assert!(check_global_ip("2001:40::1".parse().unwrap()).is_err());

        // Parent-prefix boundaries remain ordinary global unicast.
        assert!(check_global_ip("2000:ffff::1".parse().unwrap()).is_ok());
        assert!(check_global_ip("2001:200::1".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_rejects_new_ipv6_documentation_and_srv6() {
        assert!(check_global_ip("3fff::1".parse().unwrap()).is_err());
        assert!(check_global_ip("3fff:fff::1".parse().unwrap()).is_err());
        assert!(check_global_ip("3ffe:ffff::1".parse().unwrap()).is_ok());
        assert!(check_global_ip("3fff:1000::1".parse().unwrap()).is_ok());

        assert!(check_global_ip("5f00::1".parse().unwrap()).is_err());
        assert!(check_global_ip("5f00:ffff::1".parse().unwrap()).is_err());
        assert!(check_global_ip("5eff:ffff::1".parse().unwrap()).is_ok());
        assert!(check_global_ip("5f01::1".parse().unwrap()).is_ok());

        // Pin both edges of the existing 2001:db8::/32 documentation range.
        assert!(check_global_ip("2001:db7:ffff::1".parse().unwrap()).is_ok());
        assert!(check_global_ip("2001:db9::1".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_rejects_deprecated_site_local() {
        assert!(check_global_ip("fec0::1".parse().unwrap()).is_err());
        assert!(check_global_ip("feff:ffff::1".parse().unwrap()).is_err());
        // The immediately adjacent /10s are independently non-global:
        // link-local below and multicast above.
        assert!(check_global_ip("febf:ffff::1".parse().unwrap()).is_err());
        assert!(check_global_ip("ff00::1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_ipv4_mapped_loopback() {
        // ::ffff:127.0.0.1 — must be caught by delegating to the IPv4 check.
        assert!(check_global_ip("::ffff:127.0.0.1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_ipv4_mapped_private() {
        assert!(check_global_ip("::ffff:10.0.0.1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_rejects_ipv4_mapped_cgnat() {
        // A newly-blocked range reached through the IPv4-mapped delegation.
        assert!(check_global_ip("::ffff:100.64.0.1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_allows_ipv4_mapped_public() {
        assert!(check_global_ip("::ffff:8.8.8.8".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_allows_public_ipv6() {
        // A bare public IPv6 (not IPv4-mapped) must fall through every screen.
        // Without this direct case the fall-through is reached only when live
        // DNS in the validate_url tests happens to return an AAAA record —
        // absent on hosts without IPv6 (e.g. GitHub's macOS runners).
        assert!(check_global_ip("2001:4860:4860::8888".parse().unwrap()).is_ok());
    }

    // ── check_global_ip: IPv4-embedding v6 transition forms ──
    //
    // IPv4-compatible (::/96), NAT64 (64:ff9b::/96), and 6to4 (2002::/16) all
    // tunnel or translate a v4 host the socket ultimately reaches, so each is
    // screened through the v4 policy on its embedded address — not the
    // native-v6 screen, which would wave a non-global embedded v4 through.

    #[test]
    fn check_global_ip_rejects_ipv4_compatible_loopback() {
        // ::127.0.0.1 — the deprecated ::/96 form embedding loopback. Not ::1,
        // so it dodges the loopback check and must be caught by the v4 screen.
        assert!(check_global_ip("::127.0.0.1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_allows_ipv4_compatible_public() {
        // ::8.8.8.8 embeds a global v4, so the compatible form stays allowed.
        assert!(check_global_ip("::8.8.8.8".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_rejects_nat64_embedding_loopback() {
        // 64:ff9b::7f00:1 is the well-known NAT64 prefix over 127.0.0.1.
        assert!(check_global_ip("64:ff9b::7f00:1".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_allows_nat64_embedding_public() {
        // Same prefix over a global v4 (8.8.8.8) must stay reachable.
        assert!(check_global_ip("64:ff9b::8.8.8.8".parse().unwrap()).is_ok());
    }

    #[test]
    fn check_global_ip_rejects_6to4_embedding_private() {
        // 2002:a00:1:: carries the gateway v4 10.0.0.1 in segments 1–2.
        assert!(check_global_ip("2002:a00:1::".parse().unwrap()).is_err());
    }

    #[test]
    fn check_global_ip_allows_6to4_embedding_public() {
        // 2002:808:808::1 carries the global gateway v4 8.8.8.8.
        assert!(check_global_ip("2002:808:808::1".parse().unwrap()).is_ok());
    }

    // ── validate_url with private IPs ──

    #[test]
    fn validate_url_rejects_localhost() {
        let err = validate_url("http://127.0.0.1/admin").unwrap_err();
        assert!(err.contains("blocked non-global IP"));
    }

    #[test]
    fn validate_url_rejects_localhost_name() {
        let err = validate_url("http://localhost/admin").unwrap_err();
        assert!(err.contains("blocked"));
    }

    #[test]
    fn validate_url_rejects_private_with_port() {
        let err = validate_url("http://10.0.0.1:8080/api").unwrap_err();
        assert!(err.contains("blocked non-global IP"));
    }

    #[test]
    fn validate_url_rejects_userinfo() {
        let err = validate_url("http://user:pass@example.com/").unwrap_err();
        assert!(err.contains("userinfo"));
    }

    #[test]
    fn validate_url_rejects_userinfo_ssrf_bypass() {
        // Classic bypass: 127.0.0.1:80@public.com — the @ makes hand-parsed
        // code extract "public.com" but some clients connect to 127.0.0.1.
        let err = validate_url("http://127.0.0.1:80@public.com/").unwrap_err();
        assert!(err.contains("userinfo"));
    }

    #[test]
    fn validate_url_rejects_bracket_ipv6_loopback() {
        let err = validate_url("http://[::1]/path").unwrap_err();
        assert!(err.contains("blocked"));
    }

    #[test]
    fn validate_url_rejects_special_use_literal_reproductions() {
        for url in [
            "http://192.0.0.1/",
            "http://192.88.99.1/",
            "http://[64:ff9b:1::7f00:1]/",
            "http://[fec0::1]/",
        ] {
            let err = validate_url(url).unwrap_err();
            assert!(err.contains("blocked non-global IP"), "{url}: {err}");
        }
    }

    // ── truncate_response ──

    #[test]
    fn truncate_response_short_unchanged() {
        let body = "hello world".to_string();
        assert_eq!(truncate_response(body.clone()), body);
    }

    #[test]
    fn truncate_response_at_limit_unchanged() {
        let body = "a".repeat(MAX_RESPONSE_BYTES);
        assert_eq!(truncate_response(body.clone()), body);
    }

    #[test]
    fn truncate_response_over_limit() {
        let body = "a".repeat(MAX_RESPONSE_BYTES + 100);
        let result = truncate_response(body);
        assert!(result.ends_with("[truncated at 100 KB]"));
        assert!(result.len() < MAX_RESPONSE_BYTES + 50);
    }

    #[test]
    fn truncate_response_preserves_utf8_boundary() {
        let mut body = "a".repeat(MAX_RESPONSE_BYTES - 1);
        body.push('\u{20AC}');
        let result = truncate_response(body);
        assert!(result.ends_with("[truncated at 100 KB]"));
    }

    #[test]
    fn truncate_with_notice_appends_notice_even_when_under_cap() {
        // Unlike truncate_response, this always signals truncation — it is the
        // web_fetch overflow path, where read_capped may hand back an under-cap body.
        let result = truncate_with_notice("short body".to_string());
        assert!(result.starts_with("short body"));
        assert!(result.ends_with("[truncated at 100 KB]"));
    }

    // ── read_capped ──

    #[test]
    fn read_capped_small_body_not_overflowed() {
        let (body, overflowed) = read_capped(&mut &b"hello world"[..], 1024).unwrap();
        assert_eq!(body, "hello world");
        assert!(!overflowed);
    }

    #[test]
    fn read_capped_empty_body() {
        let (body, overflowed) = read_capped(&mut &b""[..], 1024).unwrap();
        assert_eq!(body, "");
        assert!(!overflowed);
    }

    #[test]
    fn read_capped_exactly_at_cap_not_overflowed() {
        let data = vec![b'a'; 8];
        let (body, overflowed) = read_capped(&mut data.as_slice(), 8).unwrap();
        assert_eq!(body.len(), 8);
        assert!(!overflowed);
    }

    #[test]
    fn read_capped_one_byte_over_cap_overflows() {
        let data = vec![b'a'; 9];
        let (body, overflowed) = read_capped(&mut data.as_slice(), 8).unwrap();
        // Reads exactly cap + 1 bytes — enough to know there was more.
        assert_eq!(body.len(), 9);
        assert!(overflowed);
    }

    #[test]
    fn read_capped_far_over_cap_stops_one_past() {
        let data = vec![b'a'; 1000];
        let (body, overflowed) = read_capped(&mut data.as_slice(), 8).unwrap();
        // Never reads more than cap + 1, regardless of how much is available.
        assert_eq!(body.len(), 9);
        assert!(overflowed);
    }

    #[test]
    fn read_capped_usize_max_cap_does_not_overflow() {
        // `cap as u64 + 1` would overflow to 0 at usize::MAX (panicking in debug,
        // a silent empty read in release); the saturating add reads normally.
        let (body, overflowed) = read_capped(&mut &b"hi"[..], usize::MAX).unwrap();
        assert_eq!(body, "hi");
        assert!(!overflowed);
    }

    #[test]
    fn read_capped_decodes_non_utf8_lossily() {
        // Genuinely invalid bytes decode to U+FFFD instead of erroring — the
        // project-wide non-UTF-8 policy shared with shell and read_file.
        let (body, overflowed) = read_capped(&mut &[0xff, 0xfe, 0xfd][..], 1024).unwrap();
        assert_eq!(body, "\u{FFFD}\u{FFFD}\u{FFFD}");
        assert!(!overflowed);
    }

    #[test]
    fn read_capped_decodes_under_cap_trailing_incomplete_char_lossily() {
        // A body that ends mid-character but stays *under* the cap is genuine
        // source malformation, not a cap-boundary slice — so the incomplete
        // sequence is decoded (to U+FFFD), not trimmed as a capping artifact.
        let mut data = b"ok".to_vec();
        data.push(0xe2); // lone lead byte of a 3-byte sequence: incomplete, error_len None
        let (body, overflowed) = read_capped(&mut data.as_slice(), 1024).unwrap();
        assert_eq!(body, "ok\u{FFFD}");
        assert!(!overflowed);
    }

    #[test]
    fn read_capped_overflowed_mid_stream_invalid_byte_decodes_lossily() {
        // Overflowed *and* malformed mid-stream: the first UTF-8 fault is the
        // invalid byte (error_len Some), not a trailing split, so nothing is
        // trimmed — the whole cap+1 read decodes lossily and overflow stands.
        let mut data = vec![b'a'; 4];
        data.push(0xff); // genuinely invalid, mid-stream
        data.extend_from_slice(&[b'b'; 4]); // 9 bytes total, cap 8 → overflow
        let (body, overflowed) = read_capped(&mut data.as_slice(), 8).unwrap();
        assert_eq!(body, "aaaa\u{FFFD}bbbb");
        assert!(overflowed);
    }

    #[test]
    fn read_capped_recovers_when_cap_splits_a_multibyte_char() {
        // '€' is 3 bytes (E2 82 AC). With cap = 8, take(9) reads 7 'a's + the first
        // 2 bytes of '€', slicing it mid-sequence. read_capped should keep the valid
        // prefix and report overflow, not error on the truncated trailing char.
        let mut data = vec![b'a'; 7];
        data.extend_from_slice("€".as_bytes());
        let (body, overflowed) = read_capped(&mut data.as_slice(), 8).unwrap();
        assert_eq!(body, "aaaaaaa");
        assert!(overflowed);
    }

    #[test]
    fn read_capped_surfaces_read_error() {
        struct ErrReader;
        impl std::io::Read for ErrReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("boom"))
            }
        }
        let err = read_capped(&mut ErrReader, 1024).unwrap_err();
        assert!(err.contains("failed to read response body"));
    }

    /// A reader that yields a few bytes per call, like a real socket — exercises
    /// the multi-read reassembly path that single-shot `&[u8]` readers skip.
    struct ChunkReader {
        data: Vec<u8>,
        pos: usize,
        chunk: usize,
    }

    impl std::io::Read for ChunkReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let remaining = &self.data[self.pos..];
            let n = remaining.len().min(buf.len()).min(self.chunk);
            buf[..n].copy_from_slice(&remaining[..n]);
            self.pos += n;
            Ok(n)
        }
    }

    #[test]
    fn read_capped_reassembles_chunked_reads_under_cap() {
        let mut reader = ChunkReader {
            data: b"hello world".to_vec(),
            pos: 0,
            chunk: 3,
        };
        let (body, overflowed) = read_capped(&mut reader, 1024).unwrap();
        assert_eq!(body, "hello world");
        assert!(!overflowed);
    }

    #[test]
    fn read_capped_detects_overflow_across_chunked_reads() {
        let mut reader = ChunkReader {
            data: vec![b'a'; 100],
            pos: 0,
            chunk: 3,
        };
        let (body, overflowed) = read_capped(&mut reader, 8).unwrap();
        assert_eq!(body.len(), 9);
        assert!(overflowed);
    }

    #[test]
    fn read_capped_boundary_split_still_yields_truncation_notice() {
        // The web_fetch edge: a multi-byte char straddling the cap makes read_capped
        // recover a body *under* the cap while `overflowed` stays true. web_fetch
        // drives the notice off the flag, so truncation is still signalled even
        // though the recovered length no longer exceeds the cap.
        let mut data = vec![b'a'; MAX_RESPONSE_BYTES - 1];
        data.extend_from_slice("\u{20AC}".as_bytes()); // 3 bytes, split at the cap
        data.extend_from_slice(&[b'a'; 16]); // ensure the source overflows the cap
        let (body, overflowed) = read_capped(&mut data.as_slice(), MAX_RESPONSE_BYTES).unwrap();
        assert!(overflowed);
        assert!(body.len() <= MAX_RESPONSE_BYTES); // recovery pulled it back under the cap
        let out = truncate_with_notice(body);
        assert!(out.ends_with("[truncated at 100 KB]"));
    }

    // ── default_tools ──

    fn test_sandbox() -> Sandbox {
        Sandbox::rooted(std::env::current_dir().unwrap()).unwrap()
    }

    /// A flag nothing raises — the no-cancellation default for tests.
    fn no_cancel() -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false))
    }

    #[test]
    fn default_tools_always_includes_web_fetch() {
        let tools = default_tools(test_sandbox(), None, no_cancel());
        assert!(tools.iter().any(|t| t.name() == "web_fetch"));
    }

    #[test]
    fn default_tools_includes_filesystem_tools() {
        let tools = default_tools(test_sandbox(), None, no_cancel());
        assert!(tools.iter().any(|t| t.name() == "read_file"));
        assert!(tools.iter().any(|t| t.name() == "write_file"));
        assert!(tools.iter().any(|t| t.name() == "edit_file"));
        assert!(tools.iter().any(|t| t.name() == "list_directory"));
        assert!(tools.iter().any(|t| t.name() == "search_files"));
    }

    #[test]
    fn default_tools_includes_web_search_when_key_present() {
        let tools = default_tools(test_sandbox(), Some("fc-test".to_string()), no_cancel());
        assert!(tools.iter().any(|t| t.name() == "web_search"));
    }

    #[test]
    fn default_tools_omits_web_search_when_no_key() {
        let tools = default_tools(test_sandbox(), None, no_cancel());
        assert!(!tools.iter().any(|t| t.name() == "web_search"));
    }

    #[test]
    fn mutating_tools_require_confirmation() {
        // Cost tier 2 = local write (mutating). All mutating tools must require
        // confirmation. Higher tiers (3, 4) are network reads — expensive but
        // not mutating, so confirmation is not required.
        let tools = default_tools(test_sandbox(), Some("fc-test".to_string()), no_cancel());
        let mutating: Vec<&str> = tools
            .iter()
            .filter(|t| t.cost() == 2)
            .map(|t| t.name())
            .collect();
        assert!(!mutating.is_empty(), "expected mutating tools in the set");
        assert!(
            tools
                .iter()
                .filter(|t| t.cost() == 2)
                .all(|t| t.requires_confirmation()),
            "a mutating tool (cost 2) is missing confirmation, among: {mutating:?}"
        );
    }

    #[test]
    fn write_file_requires_confirmation() {
        let tool = write_file::WriteFileTool::new(test_sandbox());
        assert!(tool.requires_confirmation());
    }

    #[test]
    fn edit_file_requires_confirmation() {
        let tool = edit_file::EditFileTool::new(test_sandbox());
        assert!(tool.requires_confirmation());
    }

    #[test]
    fn default_tools_includes_shell() {
        let tools = default_tools(test_sandbox(), None, no_cancel());
        assert!(tools.iter().any(|t| t.name() == "shell"));
    }

    #[test]
    fn shell_requires_confirmation() {
        let tool = shell::ShellTool::new(test_sandbox(), no_cancel());
        assert!(tool.requires_confirmation());
        // Cost 0: local execution, no API spend. Risk is handled by the
        // guardrails and the confirmation gate, not the budget system.
        assert_eq!(tool.cost(), 0);
    }

    // ── side_effecting ──

    #[test]
    fn mutating_tools_are_side_effecting() {
        // The three tools that touch disk or the shell mark themselves so the
        // agent loop never rolls a turn's history back over a real mutation.
        let dummy = serde_json::json!({});
        assert!(write_file::WriteFileTool::new(test_sandbox()).side_effecting(&dummy));
        assert!(edit_file::EditFileTool::new(test_sandbox()).side_effecting(&dummy));
        assert!(shell::ShellTool::new(test_sandbox(), no_cancel()).side_effecting(&dummy));
    }

    #[test]
    fn read_only_tools_are_not_side_effecting() {
        // Read/network tools leave nothing on disk, so their turns stay
        // rollback-eligible. (web_search defaults to false via the trait.)
        let dummy = serde_json::json!({});
        assert!(!read_file::ReadFileTool::new(test_sandbox()).side_effecting(&dummy));
        assert!(!list_directory::ListDirectoryTool::new(test_sandbox()).side_effecting(&dummy));
        assert!(!search_files::SearchFilesTool::new(test_sandbox()).side_effecting(&dummy));
        assert!(!web_fetch::WebFetchTool::default().side_effecting(&dummy));
    }
}
