use super::{
    HttpGet, MAX_RESPONSE_BYTES, ToolDef, read_capped, truncate_with_notice, validate_url,
};

pub struct WebFetchTool {
    /// The GET seam: [`super::web_live::UreqTransport`] in production, a
    /// canned stub in the hermetic tests — everything around the network call
    /// stays covered.
    transport: Box<dyn HttpGet>,
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self {
            transport: Box::new(super::web_live::UreqTransport::new()),
        }
    }
}

impl ToolDef for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a URL via HTTP GET and return the response body as text. \
         Good for JSON APIs and plain text. Use when you have a specific URL to retrieve. \
         Do NOT use to search the web — use web_search for discovery. \
         Do NOT use for local files — use read_file instead."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (http or https only)"
                }
            },
            "required": ["url"]
        })
    }

    fn cost(&self) -> u8 {
        3
    }

    fn format_status(&self, input: &serde_json::Value) -> Option<String> {
        input["url"].as_str().map(|url| format!("fetching {url}"))
    }

    fn run(
        &self,
        input: serde_json::Value,
        _out: &mut dyn std::io::Write,
    ) -> Result<String, String> {
        let url = input["url"].as_str().ok_or("missing required field: url")?;

        validate_url(url)?;

        let (status, mut body) = self
            .transport
            .get(url)
            .map_err(|e| format!("HTTP request failed: {e}"))?;

        if status >= 400 {
            return Err(format!("HTTP {status}"));
        }

        // A page is plain text, so an oversized body degrades to its first 100 KB
        // plus a truncation notice. Drive the notice off `read_capped`'s overflow
        // flag, not the body length: a dropped boundary-straddling char can pull an
        // overflowed body back under the cap, which would otherwise hide truncation.
        let (body, overflowed) = read_capped(&mut body, MAX_RESPONSE_BYTES)?;
        Ok(if overflowed {
            truncate_with_notice(body)
        } else {
            body
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A canned transport: one `(status, body)` response for any URL.
    struct StubGet {
        status: u16,
        body: Vec<u8>,
    }
    impl HttpGet for StubGet {
        fn get(&self, _url: &str) -> Result<(u16, Box<dyn std::io::Read>), String> {
            Ok((
                self.status,
                Box::new(std::io::Cursor::new(self.body.clone())),
            ))
        }
    }

    /// A transport that fails before producing a response.
    struct FailingGet;
    impl HttpGet for FailingGet {
        fn get(&self, _url: &str) -> Result<(u16, Box<dyn std::io::Read>), String> {
            Err("connection refused".to_string())
        }
    }

    fn fetch_tool(status: u16, body: &str) -> WebFetchTool {
        WebFetchTool {
            transport: Box::new(StubGet {
                status,
                body: body.as_bytes().to_vec(),
            }),
        }
    }

    #[test]
    fn web_fetch_tool_metadata() {
        let tool = WebFetchTool::default();
        assert_eq!(tool.name(), "web_fetch");
        assert!(!tool.description().is_empty());

        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["url"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "url"));
    }

    #[test]
    fn web_fetch_format_status_with_url() {
        let tool = WebFetchTool::default();
        let input = serde_json::json!({"url": "https://example.com/api"});
        assert_eq!(
            tool.format_status(&input),
            Some("fetching https://example.com/api".to_string())
        );
    }

    #[test]
    fn web_fetch_format_status_without_url() {
        let tool = WebFetchTool::default();
        let input = serde_json::json!({});
        assert_eq!(tool.format_status(&input), None);
    }

    #[test]
    fn web_fetch_missing_url() {
        let tool = WebFetchTool::default();
        let err = tool
            .run(serde_json::json!({}), &mut std::io::sink())
            .unwrap_err();
        assert!(err.contains("missing required field: url"));
    }

    #[test]
    fn web_fetch_rejects_file_scheme() {
        let tool = WebFetchTool::default();
        let err = tool
            .run(
                serde_json::json!({"url": "file:///etc/passwd"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("blocked URL scheme"));
    }

    #[test]
    fn web_fetch_rejects_localhost() {
        let tool = WebFetchTool::default();
        let err = tool
            .run(
                serde_json::json!({"url": "http://127.0.0.1/admin"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("blocked"));
    }

    #[test]
    fn web_fetch_rejects_private_ip() {
        let tool = WebFetchTool::default();
        let err = tool
            .run(
                serde_json::json!({"url": "http://10.0.0.1:8080/internal"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("blocked"));
    }

    // ── run over the transport seam ──

    #[test]
    fn web_fetch_returns_body() {
        let tool = fetch_tool(200, "hello from the page");
        let result = tool
            .run(
                serde_json::json!({"url": "https://example.com/"}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert_eq!(result, "hello from the page");
    }

    #[test]
    fn web_fetch_surfaces_http_error_status() {
        let tool = fetch_tool(404, "not found");
        let err = tool
            .run(
                serde_json::json!({"url": "https://example.com/gone"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert_eq!(err, "HTTP 404");
    }

    #[test]
    fn web_fetch_surfaces_transport_failure() {
        let tool = WebFetchTool {
            transport: Box::new(FailingGet),
        };
        let err = tool
            .run(
                serde_json::json!({"url": "https://example.com/"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert_eq!(err, "HTTP request failed: connection refused");
    }

    #[test]
    fn web_fetch_truncates_oversized_body() {
        let tool = fetch_tool(200, &"x".repeat(MAX_RESPONSE_BYTES + 10));
        let result = tool
            .run(
                serde_json::json!({"url": "https://example.com/big"}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert!(result.contains("[truncated at 100 KB]"));
        assert!(result.len() <= MAX_RESPONSE_BYTES + 30);
    }
}
