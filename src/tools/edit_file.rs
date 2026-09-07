use super::sandbox::Sandbox;
use super::{ToolDef, file_preview};
use crate::atomic_write::atomic_write_text;

pub struct EditFileTool {
    sandbox: Sandbox,
}

impl EditFileTool {
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

fn replacement(content: &str, old_str: &str, new_str: &str, path: &str) -> Result<String, String> {
    let count = content.matches(old_str).count();
    if count == 0 {
        return Err(format!(
            "old_str not found in {path} (stale context or typo?)"
        ));
    }
    if count > 1 {
        return Err(format!(
            "old_str matches {count} locations in {path} (ambiguous — provide more surrounding context)"
        ));
    }

    Ok(content.replacen(old_str, new_str, 1))
}

impl ToolDef for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replace an exact string in a file. The old_str must match exactly one \
         location in the file — zero matches (stale context or typo) and \
         multiple matches (ambiguous, provide more surrounding context) are \
         both errors. Prefer this over write_file for modifying existing files. \
         Do NOT use for creating new files — use write_file instead."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative to working directory)"
                },
                "old_str": {
                    "type": "string",
                    "description": "The exact string to find (must match exactly once)"
                },
                "new_str": {
                    "type": "string",
                    "description": "The replacement string"
                }
            },
            "required": ["path", "old_str", "new_str"]
        })
    }

    fn cost(&self) -> u8 {
        2
    }

    fn requires_confirmation(&self) -> bool {
        true
    }

    fn side_effecting(&self, _input: &serde_json::Value) -> bool {
        true
    }

    fn validate(&self, input: &serde_json::Value) -> Result<(), String> {
        if input["path"].as_str().is_none() {
            return Err("missing required field: path".to_string());
        }
        if input["old_str"].as_str().is_none() {
            return Err("missing required field: old_str".to_string());
        }
        if input["new_str"].as_str().is_none() {
            return Err("missing required field: new_str".to_string());
        }
        if input["old_str"].as_str() == Some("") {
            return Err("old_str must not be empty".to_string());
        }
        Ok(())
    }

    fn format_status(&self, input: &serde_json::Value) -> Option<String> {
        input["path"].as_str().map(|p| format!("editing {p}"))
    }

    fn confirmation_preview(&self, input: &serde_json::Value) -> Result<Option<String>, String> {
        self.validate(input)?;
        let path = input["path"].as_str().unwrap();
        let resolved = self.sandbox.resolve(path)?;
        let before =
            file_preview::read_existing(&resolved)?.ok_or("file disappeared before preview")?;
        let after = replacement(
            &before,
            input["old_str"].as_str().unwrap(),
            input["new_str"].as_str().unwrap(),
            path,
        )?;
        file_preview::render(path, Some(&before), &after).map(Some)
    }

    fn run(
        &self,
        input: serde_json::Value,
        _out: &mut dyn std::io::Write,
    ) -> Result<String, String> {
        let path = input["path"]
            .as_str()
            .ok_or("missing required field: path")?;
        let old_str = input["old_str"]
            .as_str()
            .ok_or("missing required field: old_str")?;
        let new_str = input["new_str"]
            .as_str()
            .ok_or("missing required field: new_str")?;

        if old_str.is_empty() {
            return Err("old_str must not be empty".to_string());
        }

        let resolved = self.sandbox.resolve(path)?;

        // Reject non-regular files.
        let metadata = std::fs::metadata(&resolved).map_err(stat_error)?;
        if !metadata.is_file() {
            return Err(format!("not a regular file: {path}"));
        }

        let content = std::fs::read_to_string(&resolved)
            .map_err(|e| format!("cannot read '{}': {e}", resolved.display()))?;

        let updated = replacement(&content, old_str, new_str, path)?;

        // Commit through a temp file + rename so a mid-write failure cannot
        // truncate the file we just read.
        atomic_write_text(&resolved, &updated)?;

        Ok(format!("edited {path}"))
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
    fn edit_preview_shows_actual_context_and_rejects_invalid_changes() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("file"), "before\nhello world\nafter\n").unwrap();
        let tool = EditFileTool::new(sandbox_in(dir.path()));
        let mut input = serde_json::json!({"path": "file", "old_str": "world", "new_str": "rust"});
        let preview = tool.confirmation_preview(&input).unwrap().unwrap();
        assert!(preview.contains("-     2 | hello world\n+     2 | hello rust"));
        assert!(preview.contains("      1 | before"));
        assert!(preview.contains("      3 | after"));
        assert_eq!(
            fs::read_to_string(dir.path().join("file")).unwrap(),
            "before\nhello world\nafter\n"
        );
        input["old_str"] = serde_json::json!("absent");
        assert!(
            tool.confirmation_preview(&input)
                .unwrap_err()
                .contains("not found")
        );
        input["old_str"] = serde_json::json!("e");
        assert!(
            tool.confirmation_preview(&input)
                .unwrap_err()
                .contains("ambiguous")
        );
        assert!(tool.confirmation_preview(&serde_json::json!({})).is_err());
        input["path"] = serde_json::json!("absent");
        assert!(tool.confirmation_preview(&input).is_err());
        fs::create_dir(dir.path().join("directory")).unwrap();
        input["path"] = serde_json::json!("directory");
        assert!(tool.confirmation_preview(&input).is_err());
    }

    #[test]
    fn edit_file_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let tool = EditFileTool::new(sandbox_in(dir.path()));
        assert_eq!(tool.name(), "edit_file");
        assert!(!tool.description().is_empty());

        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "path"));
        assert!(required.iter().any(|v| v == "old_str"));
        assert!(required.iter().any(|v| v == "new_str"));
    }

    #[test]
    fn edit_file_single_match() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("file.txt"), "hello world").unwrap();

        let tool = EditFileTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(
                serde_json::json!({
                    "path": "file.txt",
                    "old_str": "world",
                    "new_str": "rust"
                }),
                &mut std::io::sink(),
            )
            .unwrap();

        assert!(result.contains("edited"));
        let content = fs::read_to_string(dir.path().join("file.txt")).unwrap();
        assert_eq!(content, "hello rust");
    }

    #[test]
    fn edit_file_multiline() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("code.rs"),
            "fn main() {\n    println!(\"hi\");\n}\n",
        )
        .unwrap();

        let tool = EditFileTool::new(sandbox_in(dir.path()));
        tool.run(
            serde_json::json!({
                "path": "code.rs",
                "old_str": "    println!(\"hi\");",
                "new_str": "    println!(\"hello\");\n    println!(\"world\");"
            }),
            &mut std::io::sink(),
        )
        .unwrap();

        let content = fs::read_to_string(dir.path().join("code.rs")).unwrap();
        assert_eq!(
            content,
            "fn main() {\n    println!(\"hello\");\n    println!(\"world\");\n}\n"
        );
    }

    #[test]
    fn edit_file_zero_matches() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("file.txt"), "hello world").unwrap();

        let tool = EditFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({
                    "path": "file.txt",
                    "old_str": "nonexistent",
                    "new_str": "replacement"
                }),
                &mut std::io::sink(),
            )
            .unwrap_err();

        assert!(err.contains("not found"));
    }

    #[test]
    fn edit_file_multiple_matches() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("file.txt"), "aaa bbb aaa").unwrap();

        let tool = EditFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({
                    "path": "file.txt",
                    "old_str": "aaa",
                    "new_str": "ccc"
                }),
                &mut std::io::sink(),
            )
            .unwrap_err();

        assert!(err.contains("2 locations"));
        assert!(err.contains("ambiguous"));
    }

    #[test]
    fn edit_file_empty_old_str() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("file.txt"), "content").unwrap();

        let tool = EditFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({
                    "path": "file.txt",
                    "old_str": "",
                    "new_str": "x"
                }),
                &mut std::io::sink(),
            )
            .unwrap_err();

        assert!(err.contains("old_str must not be empty"));
    }

    #[test]
    fn edit_file_missing_fields() {
        let dir = tempfile::tempdir().unwrap();
        let tool = EditFileTool::new(sandbox_in(dir.path()));

        let err = tool
            .run(
                serde_json::json!({"old_str": "a", "new_str": "b"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("missing required field: path"));

        let err = tool
            .run(
                serde_json::json!({"path": "f.txt", "new_str": "b"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("missing required field: old_str"));

        let err = tool
            .run(
                serde_json::json!({"path": "f.txt", "old_str": "a"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("missing required field: new_str"));
    }

    #[test]
    fn edit_file_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let tool = EditFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({
                    "path": "nope.txt",
                    "old_str": "a",
                    "new_str": "b"
                }),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("cannot resolve") || err.contains("cannot stat"));
    }

    #[test]
    fn edit_file_rejects_directory() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = EditFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({
                    "path": "subdir",
                    "old_str": "a",
                    "new_str": "b"
                }),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("not a regular file"));
    }

    #[test]
    fn edit_file_rejects_escape() {
        let dir = tempfile::tempdir().unwrap();
        let tool = EditFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({
                    "path": "/etc/hosts",
                    "old_str": "a",
                    "new_str": "b"
                }),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("escapes sandbox"));
    }

    #[test]
    fn edit_file_rejects_parent_escape() {
        let dir = tempfile::tempdir().unwrap();
        let tool = EditFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({
                    "path": "../../evil.txt",
                    "old_str": "a",
                    "new_str": "b"
                }),
                &mut std::io::sink(),
            )
            .unwrap_err();
        // Sandbox rejects via canonicalize failure (file doesn't exist outside
        // root) or boundary check — either way the edit is blocked.
        assert!(err.contains("escapes sandbox") || err.contains("cannot resolve"));
    }

    #[test]
    fn edit_file_replace_with_empty_deletes_text() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("file.txt"), "hello world").unwrap();

        let tool = EditFileTool::new(sandbox_in(dir.path()));
        tool.run(
            serde_json::json!({
                "path": "file.txt",
                "old_str": " world",
                "new_str": ""
            }),
            &mut std::io::sink(),
        )
        .unwrap();

        let content = fs::read_to_string(dir.path().join("file.txt")).unwrap();
        assert_eq!(content, "hello");
    }

    #[test]
    fn edit_file_format_status() {
        let dir = tempfile::tempdir().unwrap();
        let tool = EditFileTool::new(sandbox_in(dir.path()));
        let input = serde_json::json!({"path": "src/main.rs", "old_str": "a", "new_str": "b"});
        assert_eq!(
            tool.format_status(&input),
            Some("editing src/main.rs".to_string())
        );
    }

    #[test]
    fn edit_file_rejects_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();

        #[cfg(unix)]
        {
            let link = dir.path().join("escape");
            std::os::unix::fs::symlink("/etc", &link).unwrap();

            let tool = EditFileTool::new(sandbox_in(dir.path()));
            let err = tool
                .run(
                    serde_json::json!({
                        "path": "escape/hosts",
                        "old_str": "a",
                        "new_str": "b"
                    }),
                    &mut std::io::sink(),
                )
                .unwrap_err();
            assert!(err.contains("escapes sandbox"));
        }
    }

    #[test]
    fn edit_file_validate_missing_fields() {
        let dir = tempfile::tempdir().unwrap();
        let tool = EditFileTool::new(sandbox_in(dir.path()));

        let err = tool
            .validate(&serde_json::json!({"old_str": "a", "new_str": "b"}))
            .unwrap_err();
        assert!(err.contains("missing required field: path"));

        let err = tool
            .validate(&serde_json::json!({"path": "f.txt", "new_str": "b"}))
            .unwrap_err();
        assert!(err.contains("missing required field: old_str"));

        let err = tool
            .validate(&serde_json::json!({"path": "f.txt", "old_str": "a"}))
            .unwrap_err();
        assert!(err.contains("missing required field: new_str"));
    }

    #[test]
    fn edit_file_validate_empty_old_str() {
        let dir = tempfile::tempdir().unwrap();
        let tool = EditFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .validate(&serde_json::json!({"path": "f.txt", "old_str": "", "new_str": "x"}))
            .unwrap_err();
        assert!(err.contains("old_str must not be empty"));
    }

    #[test]
    fn edit_file_validate_accepts_valid_input() {
        let dir = tempfile::tempdir().unwrap();
        let tool = EditFileTool::new(sandbox_in(dir.path()));
        assert!(
            tool.validate(&serde_json::json!({"path": "f.txt", "old_str": "a", "new_str": "b"}))
                .is_ok()
        );
    }

    #[test]
    fn edit_file_rejects_unreadable_content() {
        let dir = tempfile::tempdir().unwrap();
        // Invalid UTF-8 — `read_to_string` refuses it, driving the read error.
        fs::write(dir.path().join("bin.dat"), b"\xFF\xFE").unwrap();

        let tool = EditFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({
                    "path": "bin.dat",
                    "old_str": "a",
                    "new_str": "b"
                }),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("cannot read"));
    }

    #[cfg(unix)]
    #[test]
    fn edit_file_temp_write_failure_keeps_target() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("ro");
        fs::create_dir(&sub).unwrap();
        let file = sub.join("keep.txt");
        fs::write(&file, "hello world").unwrap();
        // Read-only directory: the match is found and read, but the atomic
        // temp file cannot be created — the original file is left untouched.
        fs::set_permissions(&sub, fs::Permissions::from_mode(0o555)).unwrap();

        let tool = EditFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({
                    "path": "ro/keep.txt",
                    "old_str": "world",
                    "new_str": "rust"
                }),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("cannot write"), "got: {err}");

        fs::set_permissions(&sub, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap(), "hello world");
    }

    #[test]
    fn stat_error_formats_reason() {
        let msg = stat_error(std::io::Error::other("race"));
        assert_eq!(msg, "cannot stat file: race");
    }

    #[test]
    fn edit_file_in_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub/file.txt"), "old content").unwrap();

        let tool = EditFileTool::new(sandbox_in(dir.path()));
        tool.run(
            serde_json::json!({
                "path": "sub/file.txt",
                "old_str": "old content",
                "new_str": "new content"
            }),
            &mut std::io::sink(),
        )
        .unwrap();

        let content = fs::read_to_string(dir.path().join("sub/file.txt")).unwrap();
        assert_eq!(content, "new content");
    }
}
