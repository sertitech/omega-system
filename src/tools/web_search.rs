use super::{HttpPostJson, ToolDef, read_capped};

const FIRECRAWL_SEARCH_URL: &str = "https://api.firecrawl.dev/v2/search";
const DEFAULT_LIMIT: u32 = 5;
/// Cap for the raw search JSON. Larger than the 100 KB text-tool cap because the
/// tool's *output* is a small title/url/description summary — the budget bounds
/// only the JSON we parse. A body past this fails fast (it cannot be parsed half).
const MAX_SEARCH_BYTES: usize = 1024 * 1024; // 1 MB

pub struct WebSearchTool {
    api_key: String,
    /// The POST seam: [`super::web_live::UreqTransport`] in production, a
    /// canned stub in the hermetic tests — everything around the network call
    /// stays covered.
    transport: Box<dyn HttpPostJson>,
}

impl WebSearchTool {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            transport: Box::new(super::web_live::UreqTransport::new()),
        }
    }
}

impl ToolDef for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web. Returns a list of results with titles, URLs, and descriptions. \
         This is the most expensive tool — use it only as a last resort when the answer \
         cannot be found in local files, your own knowledge, or a targeted web_fetch. \
         Do NOT use for questions answerable from context, local files, or common knowledge."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return (1-20, default 5)"
                }
            },
            "required": ["query"]
        })
    }

    fn cost(&self) -> u8 {
        4
    }

    fn validate(&self, input: &serde_json::Value) -> Result<(), String> {
        if let Some(query) = input["query"].as_str() {
            let trimmed = query.trim();
            // A blank query is a paid no-op at Firecrawl (cost tier 4) — reject it
            // before any request is built. Missing-string handling stays in `run`.
            if trimmed.is_empty() {
                return Err("query must not be empty".to_string());
            }
            // Reject queries that look like filesystem paths.
            if trimmed.starts_with('/') || trimmed.starts_with("./") || trimmed.starts_with("../") {
                return Err(
                    "query looks like a file path — use read_file or list_directory instead"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    fn format_status(&self, input: &serde_json::Value) -> Option<String> {
        input["query"].as_str().map(|q| format!("searching: {q}"))
    }

    fn run(
        &self,
        input: serde_json::Value,
        _out: &mut dyn std::io::Write,
    ) -> Result<String, String> {
        // Trim here too: `run` must be safe when called directly, not only behind
        // the agent's `validate` preflight. The trimmed query is what goes over the
        // wire, so padded input never spends on the whitespace-preserved variant.
        let query = input["query"]
            .as_str()
            .ok_or("missing required field: query")?
            .trim();
        if query.is_empty() {
            return Err("query must not be empty".to_string());
        }

        let limit = input["limit"]
            .as_u64()
            .map(|n| n.clamp(1, 20) as u32)
            .unwrap_or(DEFAULT_LIMIT);

        let body = serde_json::json!({
            "query": query,
            "limit": limit,
        });

        let mut response = self
            .transport
            .post_json(FIRECRAWL_SEARCH_URL, &self.api_key, &body.to_string())
            .map_err(|e| format!("Firecrawl search request failed: {e}"))?;

        // Cap the read (ureq's `.limit()` would instead *error* past the cap).
        // The JSON must be parsed whole, so an overflow can't degrade to a partial
        // body — fail fast with an actionable message rather than feeding serde a
        // truncated document.
        let (resp_body, overflowed) = read_capped(&mut response, MAX_SEARCH_BYTES)?;
        if overflowed {
            return Err(
                "Firecrawl search response exceeded 1 MB — lower the `limit` parameter and retry"
                    .to_string(),
            );
        }

        let parsed: serde_json::Value =
            serde_json::from_str(&resp_body).map_err(|e| format!("invalid JSON response: {e}"))?;

        if parsed.get("success") == Some(&serde_json::Value::Bool(false)) {
            let msg = parsed["error"].as_str().unwrap_or("unknown error");
            return Err(format!("Firecrawl error: {msg}"));
        }

        // Firecrawl returns results under data.web (array of objects).
        // Fall back to data (if flat array) or search_results for resilience.
        let results = parsed["data"]["web"]
            .as_array()
            .or_else(|| parsed["data"].as_array())
            .or_else(|| parsed["search_results"].as_array());

        let Some(results) = results else {
            return Ok("No results found.".to_string());
        };

        if results.is_empty() {
            return Ok("No results found.".to_string());
        }

        let mut output = String::new();
        for (i, result) in results.iter().enumerate() {
            let title = result["title"].as_str().unwrap_or("(no title)");
            let url = result["url"].as_str().unwrap_or("(no url)");
            let description = result["description"]
                .as_str()
                .or_else(|| result["snippet"].as_str())
                .unwrap_or("(no description)");
            output.push_str(&format!(
                "{}. {}\n   {}\n   {}\n\n",
                i + 1,
                title,
                url,
                description
            ));
        }

        Ok(output.trim_end().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A canned transport that records the request body it was handed, so
    /// tests can assert what actually goes over the seam (query, clamped
    /// limit) alongside the canned response driving the parse path. The
    /// recorder is shared via `Arc` so the test keeps a handle after the
    /// stub is boxed into the tool.
    struct StubPost {
        response: String,
        seen_body: Arc<Mutex<Option<String>>>,
    }
    impl HttpPostJson for StubPost {
        fn post_json(
            &self,
            _url: &str,
            _api_key: &str,
            body: &str,
        ) -> Result<Box<dyn std::io::Read>, String> {
            *self.seen_body.lock().unwrap() = Some(body.to_string());
            Ok(Box::new(std::io::Cursor::new(
                self.response.clone().into_bytes(),
            )))
        }
    }

    /// A transport that fails before producing a response.
    struct FailingPost;
    impl HttpPostJson for FailingPost {
        fn post_json(
            &self,
            _url: &str,
            _api_key: &str,
            _body: &str,
        ) -> Result<Box<dyn std::io::Read>, String> {
            Err("connection refused".to_string())
        }
    }

    /// A tool wired to a recording transport, returning both so a test can run
    /// the tool and then assert whether the seam was ever touched. Empty-query
    /// rejections must leave `seen_body` at `None` — proof of zero external spend.
    fn recording_tool() -> (WebSearchTool, Arc<Mutex<Option<String>>>) {
        let seen_body = Arc::new(Mutex::new(None));
        let tool = WebSearchTool {
            api_key: "fc-test".to_string(),
            transport: Box::new(StubPost {
                response: r#"{"success":true,"data":{"web":[]}}"#.to_string(),
                seen_body: seen_body.clone(),
            }),
        };
        (tool, seen_body)
    }

    fn search_tool(response: &str) -> WebSearchTool {
        WebSearchTool {
            api_key: "fc-test".to_string(),
            transport: Box::new(StubPost {
                response: response.to_string(),
                seen_body: Arc::new(Mutex::new(None)),
            }),
        }
    }

    fn run_search(response: &str) -> Result<String, String> {
        search_tool(response).run(serde_json::json!({"query": "rust"}), &mut std::io::sink())
    }

    #[test]
    fn web_search_tool_metadata() {
        let tool = WebSearchTool::new("fc-test".to_string());
        assert_eq!(tool.name(), "web_search");
        assert!(!tool.description().is_empty());

        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["query"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "query"));
    }

    #[test]
    fn web_search_format_status_with_query() {
        let tool = WebSearchTool::new("fc-test".to_string());
        let input = serde_json::json!({"query": "rust lang"});
        assert_eq!(
            tool.format_status(&input),
            Some("searching: rust lang".to_string())
        );
    }

    #[test]
    fn web_search_format_status_without_query() {
        let tool = WebSearchTool::new("fc-test".to_string());
        let input = serde_json::json!({});
        assert_eq!(tool.format_status(&input), None);
    }

    #[test]
    fn web_search_tool_missing_query() {
        let tool = WebSearchTool::new("fc-test".to_string());
        let err = tool
            .run(serde_json::json!({}), &mut std::io::sink())
            .unwrap_err();
        assert!(err.contains("missing required field: query"));
    }

    #[test]
    fn web_search_validate_rejects_absolute_path() {
        let tool = WebSearchTool::new("fc-test".to_string());
        let err = tool
            .validate(&serde_json::json!({"query": "/etc/passwd"}))
            .unwrap_err();
        assert!(err.contains("file path"));
    }

    #[test]
    fn web_search_validate_rejects_relative_path() {
        let tool = WebSearchTool::new("fc-test".to_string());
        assert!(
            tool.validate(&serde_json::json!({"query": "./src/main.rs"}))
                .is_err()
        );
        assert!(
            tool.validate(&serde_json::json!({"query": "../secret.txt"}))
                .is_err()
        );
    }

    #[test]
    fn web_search_validate_rejects_empty_query() {
        let tool = WebSearchTool::new("fc-test".to_string());
        let err = tool
            .validate(&serde_json::json!({"query": ""}))
            .unwrap_err();
        assert!(err.contains("must not be empty"));
    }

    #[test]
    fn web_search_validate_rejects_whitespace_query() {
        let tool = WebSearchTool::new("fc-test".to_string());
        let err = tool
            .validate(&serde_json::json!({"query": "   "}))
            .unwrap_err();
        assert!(err.contains("must not be empty"));
    }

    #[test]
    fn web_search_validate_accepts_normal_query() {
        let tool = WebSearchTool::new("fc-test".to_string());
        assert!(
            tool.validate(&serde_json::json!({"query": "rust programming language"}))
                .is_ok()
        );
    }

    #[test]
    fn web_search_validate_accepts_missing_query() {
        // validate doesn't enforce required fields — run does.
        let tool = WebSearchTool::new("fc-test".to_string());
        assert!(tool.validate(&serde_json::json!({})).is_ok());
    }

    // ── run over the transport seam ──

    #[test]
    fn web_search_formats_results_with_fallbacks() {
        // First result is complete; second exercises every per-field fallback
        // (missing title/url, `snippet` standing in for `description`).
        let response = r#"{"success":true,"data":{"web":[
            {"title":"Rust","url":"https://rust-lang.org","description":"A language"},
            {"snippet":"from snippet"}
        ]}}"#;
        let output = run_search(response).unwrap();
        assert_eq!(
            output,
            "1. Rust\n   https://rust-lang.org\n   A language\n\n\
             2. (no title)\n   (no url)\n   from snippet"
        );
    }

    #[test]
    fn web_search_result_without_any_description_falls_back() {
        let response = r#"{"success":true,"data":{"web":[{"title":"T","url":"u"}]}}"#;
        let output = run_search(response).unwrap();
        assert!(output.contains("(no description)"));
    }

    #[test]
    fn web_search_falls_back_to_flat_data_array() {
        let response = r#"{"success":true,"data":[{"title":"Flat","url":"u","description":"d"}]}"#;
        assert!(run_search(response).unwrap().contains("Flat"));
    }

    #[test]
    fn web_search_falls_back_to_search_results_array() {
        let response = r#"{"search_results":[{"title":"Legacy","url":"u","description":"d"}]}"#;
        assert!(run_search(response).unwrap().contains("Legacy"));
    }

    #[test]
    fn web_search_no_results_when_shape_unknown() {
        assert_eq!(
            run_search(r#"{"success":true}"#).unwrap(),
            "No results found."
        );
    }

    #[test]
    fn web_search_no_results_when_array_empty() {
        let response = r#"{"success":true,"data":{"web":[]}}"#;
        assert_eq!(run_search(response).unwrap(), "No results found.");
    }

    #[test]
    fn web_search_surfaces_firecrawl_error() {
        let response = r#"{"success":false,"error":"invalid API key"}"#;
        assert_eq!(
            run_search(response).unwrap_err(),
            "Firecrawl error: invalid API key"
        );
    }

    #[test]
    fn web_search_firecrawl_error_without_message() {
        assert_eq!(
            run_search(r#"{"success":false}"#).unwrap_err(),
            "Firecrawl error: unknown error"
        );
    }

    #[test]
    fn web_search_surfaces_invalid_json() {
        let err = run_search("{not json").unwrap_err();
        assert!(err.contains("invalid JSON response"));
    }

    #[test]
    fn web_search_surfaces_transport_failure() {
        let tool = WebSearchTool {
            api_key: "fc-test".to_string(),
            transport: Box::new(FailingPost),
        };
        let err = tool
            .run(serde_json::json!({"query": "rust"}), &mut std::io::sink())
            .unwrap_err();
        assert_eq!(err, "Firecrawl search request failed: connection refused");
    }

    #[test]
    fn web_search_run_rejects_empty_query() {
        // The untouched recorder proves no request reached the seam.
        let (tool, seen_body) = recording_tool();
        let err = tool
            .run(serde_json::json!({"query": ""}), &mut std::io::sink())
            .unwrap_err();
        assert!(err.contains("must not be empty"));
        assert!(seen_body.lock().unwrap().is_none());
    }

    #[test]
    fn web_search_run_rejects_whitespace_query() {
        let (tool, seen_body) = recording_tool();
        let err = tool
            .run(serde_json::json!({"query": "   "}), &mut std::io::sink())
            .unwrap_err();
        assert!(err.contains("must not be empty"));
        assert!(seen_body.lock().unwrap().is_none());
    }

    #[test]
    fn web_search_trims_query_on_wire() {
        // A padded query is sent trimmed — spend never lands on the
        // whitespace-preserved variant.
        let (tool, seen_body) = recording_tool();
        tool.run(
            serde_json::json!({"query": "  rust lang  "}),
            &mut std::io::sink(),
        )
        .unwrap();
        let sent = seen_body.lock().unwrap().clone().unwrap();
        let body: serde_json::Value = serde_json::from_str(&sent).unwrap();
        assert_eq!(body["query"], "rust lang");
    }

    #[test]
    fn web_search_oversized_response_fails_fast() {
        // A body past the 1 MB cap cannot be half-parsed — the tool must fail
        // with the actionable lower-the-limit message, not a serde error.
        let big = format!("[{}]", "1,".repeat(600_000));
        let err = run_search(&big).unwrap_err();
        assert!(err.contains("exceeded 1 MB"));
        assert!(err.contains("lower the `limit`"));
    }

    #[test]
    fn web_search_clamps_limit_into_range() {
        // The wire body carries the clamped limit: absent → 5, 0 → 1, 50 → 20.
        for (input, expected) in [
            (serde_json::json!({"query": "q"}), 5),
            (serde_json::json!({"query": "q", "limit": 0}), 1),
            (serde_json::json!({"query": "q", "limit": 50}), 20),
        ] {
            let seen_body = Arc::new(Mutex::new(None));
            let tool = WebSearchTool {
                api_key: "fc-test".to_string(),
                transport: Box::new(StubPost {
                    response: r#"{"success":true,"data":{"web":[]}}"#.to_string(),
                    seen_body: seen_body.clone(),
                }),
            };
            tool.run(input, &mut std::io::sink()).unwrap();
            let sent = seen_body.lock().unwrap().clone().unwrap();
            let body: serde_json::Value = serde_json::from_str(&sent).unwrap();
            assert_eq!(body["limit"], expected);
            assert_eq!(body["query"], "q");
        }
    }
}
