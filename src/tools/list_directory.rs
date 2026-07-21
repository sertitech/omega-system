use super::sandbox::Sandbox;
use super::{ToolDef, truncate_response};
use crate::display::escape_control_chars;

pub struct ListDirectoryTool {
    sandbox: Sandbox,
}

impl ListDirectoryTool {
    pub fn new(sandbox: Sandbox) -> Self {
        Self { sandbox }
    }
}

/// Error text for a directory entry that fails to read mid-iteration — not
/// reproducible hermetically (`readdir` either yields entries or ends), so it
/// lives in a named fn: the fn pointer at the call site creates no closure
/// for coverage to miss, and the body is covered by its own unit test.
fn entry_error(e: std::io::Error) -> String {
    format!("error reading directory entry: {e}")
}

impl ToolDef for ListDirectoryTool {
    fn name(&self) -> &str {
        "list_directory"
    }

    fn description(&self) -> &str {
        "List the contents of a directory. Returns one entry per line with a \
         trailing / for directories. Use to explore project structure or check \
         if a file exists. Do NOT use to read file contents — use read_file instead."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list (relative to working directory). \
                                    Defaults to the working directory if omitted."
                }
            }
        })
    }

    fn cost(&self) -> u8 {
        1
    }

    fn format_status(&self, input: &serde_json::Value) -> Option<String> {
        let path = input["path"].as_str().unwrap_or(".");
        Some(format!("listing {path}"))
    }

    fn run(
        &self,
        input: serde_json::Value,
        _out: &mut dyn std::io::Write,
    ) -> Result<String, String> {
        let path = input["path"].as_str().unwrap_or(".");

        let resolved = self.sandbox.resolve(path)?;

        // Defense in depth: refuse the shielded global `.env` as a target. A
        // directory listing exposes only entry names (not secrets), so
        // `.omega-system` itself and its contents still list — only the
        // credentials file is refused.
        if self.sandbox.is_protected_read(&resolved) {
            return Err(format!(
                "the agent's own global credentials are shielded from tools: {path}"
            ));
        }

        if !resolved.is_dir() {
            return Err(format!("not a directory: {path}"));
        }

        let mut entries: Vec<String> = Vec::new();

        let read_dir = std::fs::read_dir(&resolved)
            .map_err(|e| format!("cannot read directory '{}': {e}", resolved.display()))?;

        for entry in read_dir {
            let entry = entry.map_err(entry_error)?;
            let raw_name = entry.file_name().to_string_lossy().to_string();

            // Escape control characters (newlines, tabs, etc.) so filenames
            // cannot spoof separate entries in the output.
            let name = escape_control_chars(&raw_name);

            // Append / for directories. Use metadata() (follows symlinks) so
            // symlinked directories are correctly tagged with a trailing /.
            let is_dir = entry.path().metadata().map(|m| m.is_dir()).unwrap_or(false);

            if is_dir {
                entries.push(format!("{name}/"));
            } else {
                entries.push(name);
            }
        }

        entries.sort();

        if entries.is_empty() {
            return Ok("(empty directory)".to_string());
        }

        Ok(truncate_response(entries.join("\n")))
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
    fn list_directory_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ListDirectoryTool::new(sandbox_in(dir.path()));
        assert_eq!(tool.name(), "list_directory");
        assert!(!tool.description().is_empty());

        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
    }

    #[test]
    fn list_directory_with_files_and_dirs() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("b.txt"), "").unwrap();
        fs::write(dir.path().join("a.txt"), "").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();

        let tool = ListDirectoryTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"path": "."}), &mut std::io::sink())
            .unwrap();

        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines, vec!["a.txt", "b.txt", "sub/"]);
    }

    #[test]
    fn list_directory_empty() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ListDirectoryTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"path": "."}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "(empty directory)");
    }

    #[test]
    fn list_directory_defaults_to_root() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("file.txt"), "").unwrap();

        let tool = ListDirectoryTool::new(sandbox_in(dir.path()));
        // Omit path — should default to "."
        let result = tool
            .run(serde_json::json!({}), &mut std::io::sink())
            .unwrap();
        assert!(result.contains("file.txt"));
    }

    #[test]
    fn list_directory_not_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("file.txt"), "").unwrap();

        let tool = ListDirectoryTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"path": "file.txt"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("not a directory"));
    }

    #[test]
    fn list_directory_rejects_escape() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ListDirectoryTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(serde_json::json!({"path": "/etc"}), &mut std::io::sink())
            .unwrap_err();
        assert!(err.contains("escapes sandbox"));
    }

    #[test]
    fn list_directory_format_status() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ListDirectoryTool::new(sandbox_in(dir.path()));

        let input = serde_json::json!({"path": "src"});
        assert_eq!(tool.format_status(&input), Some("listing src".to_string()));

        let input_default = serde_json::json!({});
        assert_eq!(
            tool.format_status(&input_default),
            Some("listing .".to_string())
        );
    }

    #[test]
    fn list_directory_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("inner.txt"), "").unwrap();

        let tool = ListDirectoryTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"path": "sub"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "inner.txt");
    }

    #[test]
    fn list_directory_rejects_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();

        #[cfg(unix)]
        {
            let link = dir.path().join("escape");
            std::os::unix::fs::symlink("/etc", &link).unwrap();

            let tool = ListDirectoryTool::new(sandbox_in(dir.path()));
            let err = tool
                .run(serde_json::json!({"path": "escape"}), &mut std::io::sink())
                .unwrap_err();
            assert!(err.contains("escapes sandbox"));
        }
    }

    #[cfg(unix)]
    #[test]
    fn list_directory_unreadable_dir_errors() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let locked = dir.path().join("locked");
        fs::create_dir(&locked).unwrap();
        // Mode 000: the path still resolves (stat needs only parent perms)
        // but opening the directory for listing is denied.
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();

        let tool = ListDirectoryTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(serde_json::json!({"path": "locked"}), &mut std::io::sink())
            .unwrap_err();
        assert!(err.contains("cannot read directory"));

        // Restore permissions so the tempdir can clean itself up.
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn entry_error_formats_reason() {
        let msg = entry_error(std::io::Error::other("io fault"));
        assert_eq!(msg, "error reading directory entry: io fault");
    }

    #[cfg(unix)]
    #[test]
    fn list_directory_scrubs_invisible_format_chars_in_names() {
        // A bidi override, a zero-width space, or a BOM in a filename renders as
        // nothing yet can visually reorder or hide the entry — the same spoofing
        // vector as a raw control char. Each must survive as U+FFFD.
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a\u{202E}b.txt"), "").unwrap(); // bidi override
        fs::write(dir.path().join("c\u{200B}d.txt"), "").unwrap(); // zero-width space
        fs::write(dir.path().join("e\u{FEFF}f.txt"), "").unwrap(); // BOM / ZWNBSP

        let tool = ListDirectoryTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"path": "."}), &mut std::io::sink())
            .unwrap();
        let mut lines: Vec<&str> = result.lines().collect();
        lines.sort();
        assert_eq!(
            lines,
            vec!["a\u{FFFD}b.txt", "c\u{FFFD}d.txt", "e\u{FFFD}f.txt"]
        );
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
    fn list_directory_refuses_the_global_env_target() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_home_in(dir.path());
        fs::write(dir.path().join(".omega-system/.env"), "SECRET=x").unwrap();

        let tool = ListDirectoryTool::new(sandbox);
        let err = tool
            .run(
                serde_json::json!({"path": ".omega-system/.env"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("shielded"), "got: {err}");
    }

    #[test]
    fn list_directory_still_lists_names_including_the_home_and_its_contents() {
        // Names are not secrets — only file content is shielded. The root
        // listing shows `.omega-system/`, and listing it shows `.env`.
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_home_in(dir.path());
        fs::write(dir.path().join(".omega-system/.env"), "SECRET=x").unwrap();

        let tool = ListDirectoryTool::new(sandbox);
        let root = tool
            .run(serde_json::json!({"path": "."}), &mut std::io::sink())
            .unwrap();
        assert!(root.contains(".omega-system/"), "got: {root}");
        let inside = tool
            .run(
                serde_json::json!({"path": ".omega-system"}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert!(inside.contains(".env"), "got: {inside}");
    }

    #[cfg(unix)]
    #[test]
    fn list_directory_passes_plain_unicode_through_untouched() {
        // Accents, CJK, and emoji are ordinary printable characters — they must
        // not be scrubbed, only the invisible/control set is.
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("café_文件_🚀.txt"), "").unwrap();

        let tool = ListDirectoryTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"path": "."}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "café_文件_🚀.txt");
    }
}
