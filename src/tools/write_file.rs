use super::sandbox::Sandbox;
use super::{ToolDef, file_preview};
use crate::atomic_write::atomic_write_text;

pub struct WriteFileTool {
    sandbox: Sandbox,
}

impl WriteFileTool {
    pub fn new(sandbox: Sandbox) -> Self {
        Self { sandbox }
    }
}

impl ToolDef for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write text content to a file. Creates the file if it doesn't exist, \
         overwrites if it does. Parent directories must already exist. \
         Do NOT use to make small changes to existing files — use edit_file instead. \
         Only use write_file for creating new files or complete rewrites."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (relative to working directory)"
                },
                "content": {
                    "type": "string",
                    "description": "The text content to write to the file"
                }
            },
            "required": ["path", "content"]
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
        match input["content"].as_str() {
            None => return Err("missing required field: content".to_string()),
            Some("") => return Err("content must not be empty".to_string()),
            _ => {}
        }
        Ok(())
    }

    fn format_status(&self, input: &serde_json::Value) -> Option<String> {
        input["path"].as_str().map(|p| format!("writing {p}"))
    }

    fn confirmation_preview(&self, input: &serde_json::Value) -> Result<Option<String>, String> {
        self.validate(input)?;
        let path = input["path"].as_str().unwrap();
        let content = input["content"].as_str().unwrap();
        let resolved = self.sandbox.resolve_for_write(path)?;
        // Execution replaces the symlink itself; review the contents currently
        // observable through it, while retaining the sandbox check on its target.
        let preview_path = if std::fs::read_link(&resolved).is_ok() {
            self.sandbox.resolve(path)?
        } else {
            resolved
        };
        let before = file_preview::read_existing(&preview_path)?;
        file_preview::render(path, before.as_deref(), content).map(Some)
    }

    fn run(
        &self,
        input: serde_json::Value,
        _out: &mut dyn std::io::Write,
    ) -> Result<String, String> {
        let path = input["path"]
            .as_str()
            .ok_or("missing required field: path")?;
        let content = input["content"]
            .as_str()
            .ok_or("missing required field: content")?;

        if content.is_empty() {
            return Err("content must not be empty".to_string());
        }

        let resolved = self.sandbox.resolve_for_write(path)?;

        // If the target already exists, reject non-regular files (FIFOs,
        // device nodes, directories) to prevent hangs or side effects. One
        // `metadata` call replaces the old `exists()`/`metadata()` pair,
        // whose gap between the two calls left a stat-failure arm only a
        // filesystem race could reach. A stat failure here (NotFound
        // included) just falls through: a fresh path is the normal create
        // case, and a genuinely broken one fails fast in the write below.
        if let Ok(metadata) = std::fs::metadata(&resolved)
            && !metadata.is_file()
        {
            return Err(format!("not a regular file: {path}"));
        }

        atomic_write_text(&resolved, content)?;

        Ok(format!("wrote {} bytes to {path}", content.len()))
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
    fn write_previews_creations_and_overwrites_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        let input = serde_json::json!({"path": "file", "content": "new\n"});
        let created = tool.confirmation_preview(&input).unwrap().unwrap();
        assert!(created.contains("create new file"));
        assert!(created.contains("+     1 | new"));
        assert!(!dir.path().join("file").exists());
        fs::write(dir.path().join("file"), "old\n").unwrap();
        let replaced = tool.confirmation_preview(&input).unwrap().unwrap();
        assert!(replaced.contains("replace existing file"));
        assert!(replaced.contains("-     1 | old\n+     1 | new"));
        assert_eq!(
            fs::read_to_string(dir.path().join("file")).unwrap(),
            "old\n"
        );
        assert!(tool.confirmation_preview(&serde_json::json!({})).is_err());
        assert!(
            tool.confirmation_preview(&serde_json::json!({"path": "../escape", "content": "x"}))
                .is_err()
        );
        fs::create_dir(dir.path().join("directory")).unwrap();
        assert!(
            tool.confirmation_preview(&serde_json::json!({"path": "directory", "content": "x"}))
                .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn write_preview_preserves_in_root_symlink_behavior() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target");
        let link = dir.path().join("link");
        fs::write(&target, "old\n").unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        let input = serde_json::json!({"path": "link", "content": "new\n"});
        let preview = tool.confirmation_preview(&input).unwrap().unwrap();
        assert!(preview.contains("-     1 | old\n+     1 | new"));
        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        tool.run(input, &mut std::io::sink()).unwrap();
        assert!(
            !fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_to_string(&link).unwrap(), "new\n");
        assert_eq!(fs::read_to_string(&target).unwrap(), "old\n");
    }

    #[test]
    fn write_file_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        assert_eq!(tool.name(), "write_file");
        assert!(!tool.description().is_empty());

        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "path"));
        assert!(required.iter().any(|v| v == "content"));
    }

    #[test]
    fn write_file_creates_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));

        let result = tool
            .run(
                serde_json::json!({"path": "new.txt", "content": "hello"}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert!(result.contains("5 bytes"));

        let written = fs::read_to_string(dir.path().join("new.txt")).unwrap();
        assert_eq!(written, "hello");
    }

    #[test]
    fn write_file_overwrites_existing() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("exist.txt"), "old").unwrap();

        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        tool.run(
            serde_json::json!({"path": "exist.txt", "content": "new"}),
            &mut std::io::sink(),
        )
        .unwrap();

        let written = fs::read_to_string(dir.path().join("exist.txt")).unwrap();
        assert_eq!(written, "new");
    }

    #[test]
    fn write_file_missing_path() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"content": "hello"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("missing required field: path"));
    }

    #[test]
    fn write_file_missing_content() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"path": "file.txt"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("missing required field: content"));
    }

    #[test]
    fn write_file_rejects_empty_content() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"path": "file.txt", "content": ""}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("content must not be empty"));
    }

    #[test]
    fn write_file_rejects_escape() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"path": "/tmp/evil.txt", "content": "pwned"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("escapes sandbox"));
    }

    #[test]
    fn write_file_rejects_parent_escape() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"path": "../../evil.txt", "content": "pwned"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("escapes sandbox"));
    }

    #[test]
    fn write_file_in_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();

        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        tool.run(
            serde_json::json!({"path": "sub/file.txt", "content": "nested"}),
            &mut std::io::sink(),
        )
        .unwrap();

        let written = fs::read_to_string(dir.path().join("sub/file.txt")).unwrap();
        assert_eq!(written, "nested");
    }

    #[test]
    fn write_file_format_status() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        let input = serde_json::json!({"path": "out.txt", "content": "data"});
        assert_eq!(
            tool.format_status(&input),
            Some("writing out.txt".to_string())
        );
    }

    #[test]
    fn write_file_validate_missing_path() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .validate(&serde_json::json!({"content": "hello"}))
            .unwrap_err();
        assert!(err.contains("missing required field: path"));
    }

    #[test]
    fn write_file_validate_missing_content() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .validate(&serde_json::json!({"path": "f.txt"}))
            .unwrap_err();
        assert!(err.contains("missing required field: content"));
    }

    #[test]
    fn write_file_validate_empty_content() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .validate(&serde_json::json!({"path": "f.txt", "content": ""}))
            .unwrap_err();
        assert!(err.contains("content must not be empty"));
    }

    #[test]
    fn write_file_validate_accepts_valid_input() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        assert!(
            tool.validate(&serde_json::json!({"path": "f.txt", "content": "hello"}))
                .is_ok()
        );
    }

    #[test]
    fn write_file_rejects_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();

        #[cfg(unix)]
        {
            let link = dir.path().join("escape");
            std::os::unix::fs::symlink("/tmp", &link).unwrap();

            let tool = WriteFileTool::new(sandbox_in(dir.path()));
            let err = tool
                .run(
                    serde_json::json!({"path": "escape/evil.txt", "content": "pwned"}),
                    &mut std::io::sink(),
                )
                .unwrap_err();
            assert!(err.contains("escapes sandbox"));
        }
    }

    #[test]
    fn write_file_rejects_directory_target() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("d")).unwrap();

        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"path": "d", "content": "x"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("not a regular file: d"));
    }

    #[cfg(unix)]
    #[test]
    fn write_file_temp_write_failure_keeps_target() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("ro");
        fs::create_dir(&sub).unwrap();
        let file = sub.join("keep.txt");
        fs::write(&file, "orig").unwrap();
        // Read-only directory: the atomic temp file cannot be created, so the
        // write fails before the rename — and the existing target survives.
        fs::set_permissions(&sub, fs::Permissions::from_mode(0o555)).unwrap();

        let tool = WriteFileTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"path": "ro/keep.txt", "content": "new"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("cannot write"), "got: {err}");

        fs::set_permissions(&sub, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap(), "orig");
    }
}
