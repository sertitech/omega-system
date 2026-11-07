use super::sandbox::Sandbox;
use super::{MAX_RESPONSE_BYTES, ToolDef, truncate_response};
use std::io::{BufRead, BufReader, Read};

const BINARY_CHECK_BYTES: usize = 8 * 1024; // 8 KB

pub struct ReadFileTool {
    sandbox: Sandbox,
}

impl ReadFileTool {
    pub fn new(sandbox: Sandbox) -> Self {
        Self { sandbox }
    }
}

/// Error text for a `stat` that fails after the path already resolved.
/// `Sandbox::resolve` just canonicalized the path, so this branch is only
/// reachable through a filesystem race; a named fn (passed as a fn pointer)
/// creates no closure for coverage to miss, and the body is covered by its
/// own unit test.
fn stat_error(e: std::io::Error) -> String {
    format!("cannot stat file: {e}")
}

/// Error text for a read that fails after the file was successfully opened —
/// an I/O fault mid-read, not reproducible hermetically. A named fn for the
/// same coverage reason as [`stat_error`].
fn read_error(e: std::io::Error) -> String {
    format!("cannot read file: {e}")
}

/// Parse the optional range without silently accepting zero, fractions, or strings.
fn line_range(input: &serde_json::Value) -> Result<Option<(u64, u64)>, String> {
    let mut values = [1, u64::MAX];
    let mut ranged = false;
    for (i, name) in ["offset", "limit"].iter().enumerate() {
        if let Some(value) = input.get(name) {
            values[i] = value
                .as_u64()
                .filter(|n| *n > 0)
                .ok_or_else(|| format!("{name} must be a positive integer"))?;
            ranged = true;
        }
    }
    Ok(ranged.then_some((values[0], values[1])))
}

/// Scan buffered chunks rather than collecting lines: even a skipped or selected
/// line larger than memory costs only the reader buffer and the output cap.
fn read_text(mut reader: impl BufRead, range: Option<(u64, u64)>) -> std::io::Result<String> {
    let (offset, limit) = range.unwrap_or((1, u64::MAX));
    for _ in 1..offset {
        if reader.skip_until(b'\n')? == 0 {
            return Ok(format!("[EOF; no lines at offset {offset}]"));
        }
    }
    let mut body = Vec::new();
    let mut lines = 0;
    let mut ends_with_newline = false;
    while lines < limit && body.len() <= MAX_RESPONSE_BYTES {
        let chunk = reader.fill_buf()?;
        if chunk.is_empty() {
            break;
        }
        let end = chunk
            .iter()
            .position(|b| *b == b'\n')
            .map_or(chunk.len(), |i| i + 1);
        let take = end.min(MAX_RESPONSE_BYTES + 1 - body.len());
        body.extend_from_slice(&chunk[..take]);
        ends_with_newline = chunk[take - 1] == b'\n';
        lines += u64::from(ends_with_newline);
        reader.consume(take);
    }
    let decoded = String::from_utf8_lossy(&body);
    let truncated = decoded.len() > MAX_RESPONSE_BYTES;
    let text = truncate_response(decoded.into_owned());
    if range.is_none() {
        return Ok(text);
    }
    if body.is_empty() {
        return Ok(format!("[EOF; no lines at offset {offset}]"));
    }
    if truncated {
        return Ok(format!(
            "[from line {offset}; byte limit reached; final displayed line may be incomplete]\n{text}"
        ));
    }
    let end_line = offset.saturating_add(lines + u64::from(!ends_with_newline) - 1);
    let status = if reader.fill_buf()?.is_empty() {
        "EOF".to_string()
    } else {
        format!("more lines; next offset {}", end_line.saturating_add(1))
    };
    Ok(format!("[lines {offset}-{end_line}; {status}]\n{text}"))
}

impl ToolDef for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a text file. Returns an error for binary files. \
         Use offset (1-based line) and limit (line count) to read large files in sections. \
         Prefer this over web_fetch for local files. \
         Do NOT use to check if a file exists — use list_directory on the parent instead."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to working directory)"
                },
                "offset": {
                    "type": "integer", "minimum": 1,
                    "description": "First line to read (1-based; defaults to 1)"
                },
                "limit": {
                    "type": "integer", "minimum": 1,
                    "description": "Maximum lines to return; output is also capped at 100 KB"
                }
            },
            "required": ["path"]
        })
    }

    fn validate(&self, input: &serde_json::Value) -> Result<(), String> {
        line_range(input).map(|_| ())
    }

    fn cost(&self) -> u8 {
        1
    }

    fn format_status(&self, input: &serde_json::Value) -> Option<String> {
        input["path"].as_str().map(|p| format!("reading {p}"))
    }

    fn run(
        &self,
        input: serde_json::Value,
        _out: &mut dyn std::io::Write,
    ) -> Result<String, String> {
        let path = input["path"]
            .as_str()
            .ok_or("missing required field: path")?;

        let range = line_range(&input)?;
        let resolved = self.sandbox.resolve(path)?;

        // The global credentials file is shielded even though this tool is
        // ungated: running Omega from `$HOME` would otherwise put
        // `~/.omega-system/.env` inside the sandbox.
        if self.sandbox.is_protected_read(&resolved) {
            return Err(format!(
                "the agent's own global credentials are shielded from tools: {path}"
            ));
        }

        // Reject directories, FIFOs, device nodes, etc. — only regular files.
        let metadata = std::fs::metadata(&resolved).map_err(stat_error)?;
        if !metadata.is_file() {
            return Err(format!("not a regular file: {path}"));
        }

        let mut file = std::fs::File::open(&resolved)
            .map_err(|e| format!("cannot open '{}': {e}", resolved.display()))?;

        // Binary detection: read the first 8 KB completely (using take to
        // avoid short reads) and check for null bytes.
        let mut head = Vec::new();
        file.by_ref()
            .take(BINARY_CHECK_BYTES as u64)
            .read_to_end(&mut head)
            .map_err(read_error)?;

        if head.contains(&0) {
            return Err(format!("file appears to be binary: {path}"));
        }

        let reader = BufReader::new(std::io::Cursor::new(head).chain(file));
        read_text(reader, range).map_err(read_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn sandbox_in(dir: &std::path::Path) -> Sandbox {
        Sandbox::rooted(dir.to_path_buf()).unwrap()
    }

    #[test]
    fn read_file_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ReadFileTool::new(sandbox_in(dir.path()));
        assert_eq!(tool.name(), "read_file");
        assert!(!tool.description().is_empty());

        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "path"));
    }

    #[test]
    fn read_file_success() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("hello.txt"), "hello world").unwrap();

        let tool = ReadFileTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(
                serde_json::json!({"path": "hello.txt"}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn read_file_missing_path_field() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ReadFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(serde_json::json!({}), &mut std::io::sink())
            .unwrap_err();
        assert!(err.contains("missing required field: path"));
    }

    #[test]
    fn read_file_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ReadFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"path": "nope.txt"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("cannot resolve"));
    }

    #[test]
    fn read_file_rejects_directory() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = ReadFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(serde_json::json!({"path": "subdir"}), &mut std::io::sink())
            .unwrap_err();
        assert!(err.contains("not a regular file"));
    }

    #[test]
    fn read_file_rejects_binary() {
        let dir = tempfile::tempdir().unwrap();
        let mut content = b"some text".to_vec();
        content.push(0); // null byte
        content.extend(b"more text");
        fs::write(dir.path().join("binary.dat"), &content).unwrap();

        let tool = ReadFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"path": "binary.dat"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("binary"));
    }

    #[test]
    fn read_file_rejects_escape() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ReadFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"path": "/etc/hosts"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("escapes sandbox"));
    }

    #[test]
    fn read_file_format_status() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ReadFileTool::new(sandbox_in(dir.path()));
        let input = serde_json::json!({"path": "src/main.rs"});
        assert_eq!(
            tool.format_status(&input),
            Some("reading src/main.rs".to_string())
        );
    }

    #[test]
    fn read_file_truncates_large_file() {
        let dir = tempfile::tempdir().unwrap();
        let content = "a".repeat(200 * 1024); // 200 KB
        fs::write(dir.path().join("big.txt"), &content).unwrap();

        let tool = ReadFileTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"path": "big.txt"}), &mut std::io::sink())
            .unwrap();
        assert!(result.ends_with("[truncated at 100 KB]"));
    }

    #[test]
    fn read_file_rejects_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();

        #[cfg(unix)]
        {
            let link = dir.path().join("escape");
            std::os::unix::fs::symlink("/etc", &link).unwrap();

            let tool = ReadFileTool::new(sandbox_in(dir.path()));
            let err = tool
                .run(
                    serde_json::json!({"path": "escape/hosts"}),
                    &mut std::io::sink(),
                )
                .unwrap_err();
            assert!(err.contains("escapes sandbox"));
        }
    }

    #[test]
    fn read_file_decodes_invalid_utf8_lossily() {
        let dir = tempfile::tempdir().unwrap();
        // Invalid UTF-8 without null bytes — passes the binary gate, decodes
        // lossily (stray byte → U+FFFD) instead of erroring.
        fs::write(dir.path().join("latin1.txt"), b"caf\xE9").unwrap();

        let tool = ReadFileTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(
                serde_json::json!({"path": "latin1.txt"}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert_eq!(result, "caf\u{FFFD}");
    }

    #[cfg(unix)]
    #[test]
    fn read_file_unreadable_file_errors() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("secret.txt");
        fs::write(&file, "hidden").unwrap();
        // Mode 000: stat succeeds (it's a regular file) but open is denied.
        fs::set_permissions(&file, fs::Permissions::from_mode(0o000)).unwrap();

        let tool = ReadFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"path": "secret.txt"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("cannot open"));
    }

    // ── global-credential shielding ──

    /// A tempdir that is both the sandbox root and the fake `$HOME`, with a
    /// created `.omega-system` attached as the protected home.
    fn sandbox_home_in(dir: &std::path::Path) -> Sandbox {
        let home = dir.join(".omega-system");
        fs::create_dir(&home).unwrap();
        Sandbox::rooted(dir.to_path_buf())
            .unwrap()
            .with_protected_home(Some(&home))
    }

    #[test]
    fn read_file_refuses_the_global_env() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_home_in(dir.path());
        fs::write(dir.path().join(".omega-system/.env"), "SECRET=x").unwrap();

        let tool = ReadFileTool::new(sandbox);
        let err = tool
            .run(
                serde_json::json!({"path": ".omega-system/.env"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert_eq!(
            err,
            "the agent's own global credentials are shielded from tools: .omega-system/.env"
        );
    }

    #[test]
    fn read_file_still_reads_a_local_env_at_the_root() {
        // Only the global credentials are shielded — a project-local `.env`
        // stays readable (writes to it are floored elsewhere, reads are not).
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_home_in(dir.path());
        fs::write(dir.path().join(".env"), "LOCAL=y").unwrap();

        let tool = ReadFileTool::new(sandbox);
        let result = tool
            .run(serde_json::json!({"path": ".env"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "LOCAL=y");
    }

    #[cfg(unix)]
    #[test]
    fn read_file_refuses_a_symlink_to_the_global_env() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_home_in(dir.path());
        fs::write(dir.path().join(".omega-system/.env"), "SECRET=x").unwrap();
        // A symlink inside the sandbox pointing at the global `.env`
        // canonicalizes onto the shielded path and is refused.
        std::os::unix::fs::symlink(
            dir.path().join(".omega-system/.env"),
            dir.path().join("link_env"),
        )
        .unwrap();

        let tool = ReadFileTool::new(sandbox);
        let err = tool
            .run(
                serde_json::json!({"path": "link_env"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("shielded"), "got: {err}");
    }

    #[test]
    fn line_ranges_validate_and_report_continuation_and_eof() {
        for input in [
            serde_json::json!({"offset": 0}),
            serde_json::json!({"limit": -1}),
            serde_json::json!({"offset": 1.5}),
            serde_json::json!({"limit": "2"}),
            serde_json::json!({"limit": null}),
        ] {
            let tool = ReadFileTool::new(Sandbox::unbounded());
            assert!(
                tool.validate(&input)
                    .unwrap_err()
                    .contains("positive integer")
            );
        }
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("text"), "one\ntwo\nthree").unwrap();
        let tool = ReadFileTool::new(sandbox_in(dir.path()));
        for (fields, expected) in [
            (
                serde_json::json!({"offset": 2, "limit": 1}),
                "[lines 2-2; more lines; next offset 3]\ntwo\n",
            ),
            (serde_json::json!({"offset": 3}), "[lines 3-3; EOF]\nthree"),
            (
                serde_json::json!({"limit": 3}),
                "[lines 1-3; EOF]\none\ntwo\nthree",
            ),
            (
                serde_json::json!({"offset": 4}),
                "[EOF; no lines at offset 4]",
            ),
            (
                serde_json::json!({"offset": 5}),
                "[EOF; no lines at offset 5]",
            ),
        ] {
            let mut input = fields;
            input["path"] = serde_json::json!("text");
            assert!(tool.validate(&input).is_ok());
            assert_eq!(tool.run(input, &mut std::io::sink()).unwrap(), expected);
        }
        assert!(
            tool.run(
                serde_json::json!({"path": "text", "offset": 0}),
                &mut std::io::sink()
            )
            .is_err()
        );
        assert_eq!(
            read_text(&b""[..], Some((1, 2))).unwrap(),
            "[EOF; no lines at offset 1]"
        );
        assert_eq!(
            read_text(&b"one\n"[..], Some((1, 1))).unwrap(),
            "[lines 1-1; EOF]\none\n"
        );
    }

    #[test]
    fn range_reads_tail_beyond_default_cap_and_skips_a_huge_line() {
        let dir = tempfile::tempdir().unwrap();
        let content = format!("{}\ntail one\ntail two\n", "a".repeat(200 * 1024));
        fs::write(dir.path().join("large"), &content).unwrap();
        let tool = ReadFileTool::new(sandbox_in(dir.path()));
        assert_eq!(
            tool.run(
                serde_json::json!({"path": "large", "offset": 2, "limit": 2}),
                &mut std::io::sink()
            )
            .unwrap(),
            "[lines 2-3; EOF]\ntail one\ntail two\n"
        );
        let result = tool
            .run(
                serde_json::json!({"path": "large", "limit": 1}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert!(result.contains("final displayed line may be incomplete"));
        assert!(result.ends_with("[truncated at 100 KB]"));
        assert!(result.len() < MAX_RESPONSE_BYTES + 200);
        let data = "é".repeat(MAX_RESPONSE_BYTES);
        assert!(
            read_text(data.as_bytes(), Some((1, 1)))
                .unwrap()
                .contains("byte limit reached")
        );
    }

    #[test]
    fn range_read_propagates_io_errors_when_skipping_or_collecting() {
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("injected read failure"))
            }
        }
        for offset in [1, 2] {
            assert!(
                read_text(BufReader::new(Broken), Some((offset, 1)))
                    .unwrap_err()
                    .to_string()
                    .contains("injected read failure")
            );
        }
    }

    #[test]
    fn error_text_helpers_format_reason() {
        let stat = stat_error(std::io::Error::other("race"));
        assert_eq!(stat, "cannot stat file: race");
        let read = read_error(std::io::Error::other("io fault"));
        assert_eq!(read, "cannot read file: io fault");
    }
}
