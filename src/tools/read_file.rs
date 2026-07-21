use super::sandbox::Sandbox;
use super::{MAX_RESPONSE_BYTES, ToolDef, truncate_response};
use std::io::Read;

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

impl ToolDef for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a text file. Returns an error for binary files. \
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
                }
            },
            "required": ["path"]
        })
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

        // Read up to MAX_RESPONSE_BYTES + 1 to detect truncation, then
        // truncate. This caps memory usage rather than reading the whole file.
        let remaining = MAX_RESPONSE_BYTES + 1 - head.len();
        let mut rest = Vec::new();
        file.take(remaining as u64)
            .read_to_end(&mut rest)
            .map_err(read_error)?;

        head.extend(rest);

        // Lossy decode: the null-byte gate above already rejects true
        // binaries, so what reaches here is text with at most stray invalid
        // bytes (e.g. Latin-1) — decode them to U+FFFD rather than error,
        // the project-wide non-UTF-8 policy shared with shell and read_capped.
        let body = String::from_utf8_lossy(&head).into_owned();

        Ok(truncate_response(body))
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
    fn error_text_helpers_format_reason() {
        let stat = stat_error(std::io::Error::other("race"));
        assert_eq!(stat, "cannot stat file: race");
        let read = read_error(std::io::Error::other("io fault"));
        assert_eq!(read, "cannot read file: io fault");
    }
}
