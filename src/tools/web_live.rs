//! The live half of the web tools: the shared `ureq` transport behind the
//! [`HttpGet`]/[`HttpPostJson`] seams, plus the `#[ignore]`'d network tests.
//! Nothing here is reachable without the network, so the `_live.rs` suffix
//! marks the file for wholesale exclusion from the coverage gate — anything
//! beyond the raw HTTP calls belongs in the covered tool modules.

use super::{HttpGet, HttpPostJson, check_resolved_addrs};
use ureq::config::Config;
use ureq::http::Uri;
use ureq::unversioned::resolver::{DefaultResolver, ResolvedSocketAddrs, Resolver};
use ureq::unversioned::transport::{DefaultConnector, NextTimeout};

const TIMEOUT_SECS: u64 = 30;

/// The connect-time half of the web_fetch SSRF guard: a [`Resolver`] that runs
/// every address the default resolver returns through the same
/// [`check_resolved_addrs`] screen as `validate_url`'s pre-flight resolve,
/// *before* the connector ever sees it. The socket can only reach a vetted
/// address, so a DNS server that passed validation with a public IP and then
/// rebinds to a private one gets caught at connect — and redirect targets,
/// which ureq resolves through this same resolver, are screened identically.
/// A rejection surfaces as a `PermissionDenied` IO error carrying the screen's
/// own "blocked …" message.
#[derive(Debug, Default)]
struct GuardedResolver {
    inner: DefaultResolver,
}

impl Resolver for GuardedResolver {
    fn resolve(
        &self,
        uri: &Uri,
        config: &Config,
        timeout: NextTimeout,
    ) -> Result<ResolvedSocketAddrs, ureq::Error> {
        let addrs = self.inner.resolve(uri, config, timeout)?;
        check_resolved_addrs(&addrs, uri.host().unwrap_or_default()).map_err(|msg| {
            ureq::Error::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                msg,
            ))
        })?;
        Ok(addrs)
    }
}

/// The web tools' real transport: one `ureq` agent with a whole-call deadline
/// (tool responses are small; nothing streams). Like the provider agents in
/// [`crate::provider::transport`], it disables ureq's "4xx/5xx is an `Err`"
/// default so a non-2xx comes back as an `Ok` carrying its status and body —
/// `web_fetch` reports the status and `web_search` reads Firecrawl's JSON
/// error envelope, instead of both collapsing into a transport failure.
///
/// The agent resolves names through [`GuardedResolver`], so every connection —
/// `web_fetch`'s GETs, redirect hops, and `web_search`'s Firecrawl POSTs alike
/// — can only reach guard-approved addresses.
pub(crate) struct UreqTransport {
    agent: ureq::Agent,
}

impl UreqTransport {
    pub(crate) fn new() -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(TIMEOUT_SECS)))
            .http_status_as_error(false)
            // Every connection must resolve through GuardedResolver for the
            // connect-time SSRF re-screen to hold. ureq's Config::default()
            // otherwise picks up HTTPS_PROXY/HTTP_PROXY/ALL_PROXY and routes
            // through the proxy, which resolves and connects to the target
            // itself — the guard would then only ever see the proxy's address.
            // So environment proxies are deliberately ignored.
            .proxy(None)
            .build();
        let agent = ureq::Agent::with_parts(
            config,
            DefaultConnector::default(),
            GuardedResolver::default(),
        );
        Self { agent }
    }
}

impl HttpGet for UreqTransport {
    fn get(&self, url: &str) -> Result<(u16, Box<dyn std::io::Read>), String> {
        let response = self.agent.get(url).call().map_err(|e| e.to_string())?;
        let status = response.status().as_u16();
        Ok((status, Box::new(response.into_body().into_reader())))
    }
}

impl HttpPostJson for UreqTransport {
    fn post_json(
        &self,
        url: &str,
        api_key: &str,
        body: &str,
    ) -> Result<Box<dyn std::io::Read>, String> {
        let response = self
            .agent
            .post(url)
            .header("Authorization", &format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .send(body.as_bytes())
            .map_err(|e| e.to_string())?;
        Ok(Box::new(response.into_body().into_reader()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_env_var;
    use crate::tools::ToolDef;
    use crate::tools::web_fetch::WebFetchTool;
    use crate::tools::web_search::WebSearchTool;
    use ureq::unversioned::transport::time::Duration;

    /// A no-deadline timeout for driving [`GuardedResolver`] directly.
    fn no_timeout() -> NextTimeout {
        NextTimeout {
            after: Duration::NotHappening,
            reason: ureq::Timeout::Global,
        }
    }

    #[test]
    fn guarded_resolver_blocks_a_private_literal_at_connect_time() {
        // A literal IP resolves without any DNS query, so this exercises the
        // connect-time screen without the network: the resolver itself
        // refuses, before any connector involvement — exactly what a
        // rebinding host would hit at the real connect.
        let uri: Uri = "http://127.0.0.1/admin".parse().unwrap();
        let err = GuardedResolver::default()
            .resolve(&uri, &Config::default(), no_timeout())
            .unwrap_err();
        let ureq::Error::Io(io) = err else {
            panic!("expected Io rejection, got {err:?}");
        };
        assert_eq!(io.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(io.to_string().contains("blocked"), "got: {io}");
    }

    #[test]
    fn guarded_resolver_passes_a_public_literal_through() {
        // A public literal passes the screen and comes back vetted, port and
        // all — the pass-through half of the guard.
        let uri: Uri = "http://8.8.8.8:8080/status".parse().unwrap();
        let addrs = GuardedResolver::default()
            .resolve(&uri, &Config::default(), no_timeout())
            .unwrap();
        assert_eq!(addrs.len(), 1);
        assert_eq!(addrs[0], "8.8.8.8:8080".parse().unwrap());
    }

    #[test]
    #[ignore = "hits httpbin.org over the network; run with --ignored"]
    fn web_fetch_integration() {
        let tool = WebFetchTool::default();
        let result = tool
            .run(
                serde_json::json!({"url": "https://httpbin.org/get"}),
                &mut std::io::sink(),
            )
            .unwrap();

        assert!(!result.is_empty());
        assert!(result.contains("httpbin.org"));
    }

    #[test]
    #[ignore = "hits the live Firecrawl API; run with --ignored"]
    fn web_search_integration() {
        let Some(key) = load_env_var("FIRECRAWL_API_KEY") else {
            eprintln!("FIRECRAWL_API_KEY not set, skipping integration test");
            return;
        };

        let tool = WebSearchTool::new(key);
        let result = tool
            .run(
                serde_json::json!({"query": "rust programming language", "limit": 3}),
                &mut std::io::sink(),
            )
            .unwrap();

        assert!(!result.is_empty());
        assert!(result.contains("http"));
    }
}
