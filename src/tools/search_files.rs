use super::sandbox::Sandbox;
use super::{ToolDef, truncate_response};
use crate::display::escape_control_chars;
use std::path::{Path, PathBuf};

/// Files larger than this are skipped (and counted in a notice) rather than
/// scanned — a whole-file read is the scan's memory bound, and a legitimate
/// source file over 1 MB is vanishingly rare.
const MAX_FILE_BYTES: u64 = 1024 * 1024; // 1 MB

/// Matched lines are capped so one broad query cannot flood the response;
/// hitting the cap is reported with a truncation notice, never silently.
const MAX_MATCHES: usize = 200;

/// A single matched line is clipped to this many characters (with a visible
/// `…`) so minified one-liners cannot eat the whole response cap.
const MAX_LINE_CHARS: usize = 500;

/// Binary sniff window — same policy as `read_file`: a null byte in the first
/// 8 KB marks the file as binary.
const BINARY_CHECK_BYTES: usize = 8 * 1024; // 8 KB

pub struct SearchFilesTool {
    sandbox: Sandbox,
}

impl SearchFilesTool {
    pub fn new(sandbox: Sandbox) -> Self {
        Self { sandbox }
    }
}

/// Error text for a directory entry that fails mid-iteration — not
/// reproducible hermetically (`readdir` either yields entries or ends), so it
/// lives in a named fn: the fn pointer at the call site creates no closure
/// for coverage to miss, and the body is covered by its own unit test.
fn entry_error(e: std::io::Error) -> String {
    format!("error reading directory entry: {e}")
}

/// Error text for a `stat` that fails after `read_dir` already yielded the
/// entry — only reachable through a filesystem race; a named fn for the same
/// coverage reason as [`entry_error`].
fn stat_error(e: std::io::Error) -> String {
    format!("cannot stat file: {e}")
}

/// The walk's accumulating state: the prepared needle, the filters, and
/// everything the final report needs (matches, cap flag, skip count).
struct Search {
    /// The query — lowercased up front when the match is case-insensitive.
    needle: String,
    case_sensitive: bool,
    /// Extension filter without a leading dot (e.g. `rs`), when given.
    extension: Option<String>,
    /// Canonical sandbox root, for root-relative display paths; `None` when
    /// the sandbox is unbounded (paths then display absolute).
    root: Option<PathBuf>,
    matches: Vec<String>,
    hit_cap: bool,
    skipped_large: usize,
}

impl Search {
    /// Recursively scan `dir`, depth-first in lexicographic entry order (so
    /// output is deterministic). Hidden entries (dot-prefixed, e.g. `.git`)
    /// are skipped, and symlinks are not followed — the walk never leaves the
    /// resolved scope, so no per-entry sandbox re-check is needed.
    fn walk(&mut self, dir: &Path) -> Result<(), String> {
        let read_dir = std::fs::read_dir(dir)
            .map_err(|e| format!("cannot read directory '{}': {e}", dir.display()))?;

        let mut entries = Vec::new();
        for entry in read_dir {
            entries.push(entry.map_err(entry_error)?);
        }
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            if self.hit_cap {
                return Ok(()); // the cap ends the whole walk, not just a file
            }
            if entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            // `file_type` does not follow symlinks, so a symlink is neither
            // dir nor file here and falls through — deliberately skipped
            // (following one could cycle or step outside the resolved scope).
            let file_type = entry.file_type().map_err(entry_error)?;
            if file_type.is_dir() {
                self.walk(&entry.path())?;
            } else if file_type.is_file() {
                self.scan_file(&entry.path())?;
            }
        }
        Ok(())
    }

    /// Scan one file, appending `path:line: text` for every matching line.
    fn scan_file(&mut self, path: &Path) -> Result<(), String> {
        if let Some(ext) = &self.extension
            && path.extension().and_then(|e| e.to_str()) != Some(ext.as_str())
        {
            return Ok(());
        }

        let len = std::fs::metadata(path).map_err(stat_error)?.len();
        if len > MAX_FILE_BYTES {
            self.skipped_large += 1;
            return Ok(());
        }

        let bytes =
            std::fs::read(path).map_err(|e| format!("cannot read '{}': {e}", path.display()))?;

        // Binary sniff: a null byte in the first 8 KB skips the file, same
        // policy as read_file (which errors; here a skip keeps the walk going).
        if bytes.iter().take(BINARY_CHECK_BYTES).any(|&b| b == 0) {
            return Ok(());
        }

        // Lossy decode — the project-wide non-UTF-8 policy shared with
        // read_file and shell: stray invalid bytes become U+FFFD.
        let text = String::from_utf8_lossy(&bytes);
        let display = self.display_path(path);

        for (idx, line) in text.lines().enumerate() {
            let hit = if self.case_sensitive {
                line.contains(&self.needle)
            } else {
                line.to_lowercase().contains(&self.needle)
            };
            if hit {
                // Cap check *before* pushing: the flag only trips when a
                // match beyond the cap actually exists, so a result set of
                // exactly MAX_MATCHES carries no false truncation notice.
                if self.matches.len() == MAX_MATCHES {
                    self.hit_cap = true;
                    return Ok(());
                }
                self.matches
                    .push(format!("{display}:{}: {}", idx + 1, clip_line(line)));
            }
        }
        Ok(())
    }

    /// Display paths relative to the sandbox root (the form the other file
    /// tools accept as input), or absolute when the sandbox is unbounded.
    ///
    /// Control characters in a component are escaped to U+FFFD (the same policy
    /// as `list_directory`): a newline-bearing filename would otherwise forge
    /// extra `path:line:` rows in the output.
    fn display_path(&self, path: &Path) -> String {
        let raw = match &self.root {
            Some(root) => path
                .strip_prefix(root)
                .unwrap_or(path)
                .display()
                .to_string(),
            None => path.display().to_string(),
        };
        escape_control_chars(&raw)
    }
}

/// Clip a matched line to [`MAX_LINE_CHARS`] characters, marking the cut with
/// a visible `…` so truncation is never silent.
fn clip_line(line: &str) -> String {
    match line.char_indices().nth(MAX_LINE_CHARS) {
        Some((byte_idx, _)) => format!("{}…", &line[..byte_idx]),
        None => line.to_string(),
    }
}

impl ToolDef for SearchFilesTool {
    fn name(&self) -> &str {
        "search_files"
    }

    fn description(&self) -> &str {
        "Search file contents for a literal substring (no regex). Recursively \
         walks the given directory in deterministic order, skipping hidden \
         entries (like .git), symlinks, binary files, and files over 1 MB. \
         Returns path:line: matched-line rows. Prefer this over shell grep \
         for code searches."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Literal substring to search for (not a regex)"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search (relative to working directory). \
                                    Defaults to the working directory if omitted."
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Match case-sensitively. Defaults to true."
                },
                "extension": {
                    "type": "string",
                    "description": "Only search files with this extension (e.g. \"rs\")."
                }
            },
            "required": ["query"]
        })
    }

    fn cost(&self) -> u8 {
        1
    }

    fn format_status(&self, input: &serde_json::Value) -> Option<String> {
        input["query"].as_str().map(|q| {
            let path = input["path"].as_str().unwrap_or(".");
            format!("searching {path} for \"{q}\"")
        })
    }

    fn run(
        &self,
        input: serde_json::Value,
        _out: &mut dyn std::io::Write,
    ) -> Result<String, String> {
        let query = input["query"]
            .as_str()
            .ok_or("missing required field: query")?;
        if query.is_empty() {
            return Err("query must not be empty".to_string());
        }
        let scope = input["path"].as_str().unwrap_or(".");
        let case_sensitive = input["case_sensitive"].as_bool().unwrap_or(true);
        let extension = input["extension"]
            .as_str()
            .map(|e| e.trim_start_matches('.').to_string());

        let resolved = self.sandbox.resolve(scope)?;
        // Refuse the shielded global `.env` as a direct scope (a symlink to it
        // canonicalizes here too). The recursive walk needs no per-entry check:
        // it skips dot-prefixed entries and never follows symlinks, and the
        // credentials live at `~/.omega-system/.env` — both the directory and
        // the file are dot-prefixed, so the walk can never reach them.
        if self.sandbox.is_protected_read(&resolved) {
            return Err(format!(
                "the agent's own global credentials are shielded from tools: {scope}"
            ));
        }
        if !resolved.is_dir() {
            return Err(format!("not a directory: {scope}"));
        }

        let mut search = Search {
            needle: if case_sensitive {
                query.to_string()
            } else {
                query.to_lowercase()
            },
            case_sensitive,
            extension,
            root: self.sandbox.root().map(Path::to_path_buf),
            matches: Vec::new(),
            hit_cap: false,
            skipped_large: 0,
        };
        search.walk(&resolved)?;

        let mut lines = search.matches;
        if search.hit_cap {
            lines.push(format!("[truncated: match cap of {MAX_MATCHES} reached]"));
        }
        if search.skipped_large > 0 {
            lines.push(format!(
                "[skipped {} file(s) over 1 MB]",
                search.skipped_large
            ));
        }
        if lines.is_empty() {
            return Ok("(no matches)".to_string());
        }
        Ok(truncate_response(lines.join("\n")))
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
    fn search_files_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        assert_eq!(tool.name(), "search_files");
        assert!(!tool.description().is_empty());

        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "query"));

        assert_eq!(tool.cost(), 1);
        assert!(!tool.requires_confirmation());
        assert!(!tool.side_effecting(&serde_json::json!({})));
    }

    #[test]
    fn search_matches_across_files_and_lines_deterministically() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("b.txt"), "no hit\nneedle here\n").unwrap();
        fs::write(
            dir.path().join("a.txt"),
            "needle first\nplain\nneedle again\n",
        )
        .unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub/c.txt"), "deep needle\n").unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(
            result,
            "a.txt:1: needle first\n\
             a.txt:3: needle again\n\
             b.txt:2: needle here\n\
             sub/c.txt:1: deep needle"
        );
    }

    #[test]
    fn search_no_match_reports_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "nothing to see\n").unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "absent"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "(no matches)");
    }

    #[test]
    fn search_is_case_sensitive_by_default() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "Hello World\n").unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "hello"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "(no matches)");
    }

    #[test]
    fn search_case_insensitive_when_requested() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "Hello World\n").unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(
                serde_json::json!({"query": "hello", "case_sensitive": false}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert_eq!(result, "a.txt:1: Hello World");
    }

    #[test]
    fn search_skips_binary_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut binary = b"needle".to_vec();
        binary.push(0); // null byte in the sniff window
        fs::write(dir.path().join("bin.dat"), &binary).unwrap();
        fs::write(dir.path().join("text.txt"), "needle\n").unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "text.txt:1: needle");
    }

    #[test]
    fn search_skips_hidden_files_and_directories() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join(".git")).unwrap();
        fs::write(dir.path().join(".git/config"), "needle\n").unwrap();
        fs::write(dir.path().join(".hidden.txt"), "needle\n").unwrap();
        fs::write(dir.path().join("seen.txt"), "needle\n").unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "seen.txt:1: needle");
    }

    #[test]
    fn search_scoped_to_subdirectory_keeps_root_relative_paths() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("outside.txt"), "needle\n").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub/inner.txt"), "needle\n").unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(
                serde_json::json!({"query": "needle", "path": "sub"}),
                &mut std::io::sink(),
            )
            .unwrap();
        // Paths stay relative to the sandbox root — the form read_file accepts.
        assert_eq!(result, "sub/inner.txt:1: needle");
    }

    #[test]
    fn search_extension_filter() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("code.rs"), "needle\n").unwrap();
        fs::write(dir.path().join("notes.txt"), "needle\n").unwrap();
        fs::write(dir.path().join("bare"), "needle\n").unwrap(); // no extension

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(
                serde_json::json!({"query": "needle", "extension": "rs"}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert_eq!(result, "code.rs:1: needle");

        // A leading dot is tolerated and normalized away.
        let dotted = tool
            .run(
                serde_json::json!({"query": "needle", "extension": ".rs"}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert_eq!(dotted, "code.rs:1: needle");
    }

    #[test]
    fn search_missing_query_field() {
        let dir = tempfile::tempdir().unwrap();
        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(serde_json::json!({}), &mut std::io::sink())
            .unwrap_err();
        assert!(err.contains("missing required field: query"));
    }

    #[test]
    fn search_rejects_empty_query() {
        let dir = tempfile::tempdir().unwrap();
        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(serde_json::json!({"query": ""}), &mut std::io::sink())
            .unwrap_err();
        assert!(err.contains("query must not be empty"));
    }

    #[test]
    fn search_rejects_absolute_escape() {
        let dir = tempfile::tempdir().unwrap();
        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"query": "x", "path": "/etc"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("escapes sandbox"));
    }

    #[test]
    fn search_rejects_dot_dot_escape() {
        let dir = tempfile::tempdir().unwrap();
        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"query": "x", "path": "../../.."}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("escapes sandbox") || err.contains("cannot resolve"));
    }

    #[test]
    fn search_rejects_nonexistent_scope() {
        let dir = tempfile::tempdir().unwrap();
        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"query": "x", "path": "no_such_dir"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("cannot resolve"));
    }

    #[test]
    fn search_rejects_file_scope() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("file.txt"), "needle\n").unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(
                serde_json::json!({"query": "needle", "path": "file.txt"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("not a directory"));
    }

    #[test]
    fn search_stops_at_match_cap_with_notice() {
        let dir = tempfile::tempdir().unwrap();
        let over = "needle\n".repeat(MAX_MATCHES + 5);
        fs::write(dir.path().join("a.txt"), &over).unwrap();
        // A second file after the cap-hitting one drives the walk's early return.
        fs::write(dir.path().join("z.txt"), "needle\n").unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap();
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), MAX_MATCHES + 1);
        assert_eq!(
            *lines.last().unwrap(),
            "[truncated: match cap of 200 reached]"
        );
        assert!(!result.contains("z.txt")); // walk ended at the cap
    }

    #[test]
    fn search_exactly_at_cap_has_no_notice() {
        let dir = tempfile::tempdir().unwrap();
        let exact = "needle\n".repeat(MAX_MATCHES);
        fs::write(dir.path().join("a.txt"), &exact).unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result.lines().count(), MAX_MATCHES);
        assert!(!result.contains("[truncated"));
    }

    #[test]
    fn search_skips_large_files_with_notice() {
        let dir = tempfile::tempdir().unwrap();
        let big = "needle ".repeat((MAX_FILE_BYTES as usize / 7) + 1);
        assert!(big.len() as u64 > MAX_FILE_BYTES);
        fs::write(dir.path().join("big.txt"), &big).unwrap();
        fs::write(dir.path().join("small.txt"), "needle\n").unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "small.txt:1: needle\n[skipped 1 file(s) over 1 MB]");
    }

    #[test]
    fn search_only_large_files_still_reports_the_skip() {
        // No matches at all, but a skipped file must never vanish silently.
        let dir = tempfile::tempdir().unwrap();
        let big = "x".repeat(MAX_FILE_BYTES as usize + 1);
        fs::write(dir.path().join("big.txt"), &big).unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "[skipped 1 file(s) over 1 MB]");
    }

    #[test]
    fn search_does_not_follow_symlinks() {
        let dir = tempfile::tempdir().unwrap();

        #[cfg(unix)]
        {
            let outside = tempfile::tempdir().unwrap();
            fs::write(outside.path().join("secret.txt"), "needle\n").unwrap();
            // A symlinked directory pointing outside the sandbox, and a
            // symlinked file inside it — neither may be followed.
            std::os::unix::fs::symlink(outside.path(), dir.path().join("linkdir")).unwrap();
            fs::write(dir.path().join("real.txt"), "needle\n").unwrap();
            std::os::unix::fs::symlink(
                dir.path().join("real.txt"),
                dir.path().join("linkfile.txt"),
            )
            .unwrap();

            let tool = SearchFilesTool::new(sandbox_in(dir.path()));
            let result = tool
                .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
                .unwrap();
            assert_eq!(result, "real.txt:1: needle");
        }
    }

    #[test]
    fn search_clips_long_matched_lines() {
        let dir = tempfile::tempdir().unwrap();
        let long = format!("{}needle{}", "a".repeat(MAX_LINE_CHARS), "b".repeat(100));
        fs::write(dir.path().join("long.txt"), &long).unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap();
        assert!(result.ends_with('…'));
        // Prefix (path:line: ) + exactly MAX_LINE_CHARS chars + the ellipsis.
        let line = result.lines().next().unwrap();
        let shown = line.strip_prefix("long.txt:1: ").unwrap();
        assert_eq!(shown.chars().count(), MAX_LINE_CHARS + 1);
    }

    #[test]
    fn clip_line_leaves_short_lines_untouched() {
        assert_eq!(clip_line("short"), "short");
    }

    #[test]
    fn search_decodes_invalid_utf8_lossily() {
        let dir = tempfile::tempdir().unwrap();
        // Invalid UTF-8 without null bytes — passes the binary sniff, decodes
        // lossily, and the surrounding text still matches.
        fs::write(dir.path().join("latin1.txt"), b"caf\xE9 needle\n").unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "latin1.txt:1: caf\u{FFFD} needle");
    }

    #[cfg(unix)]
    #[test]
    fn search_escapes_control_chars_in_displayed_paths() {
        // A newline in a filename would otherwise split the `path:line:` row in
        // two, letting a crafted file forge fake match rows. The control char
        // must survive as U+FFFD, keeping every match on a single line.
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("evil\nfake.txt"), "needle\n").unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "evil\u{FFFD}fake.txt:1: needle");
        assert_eq!(result.lines().count(), 1); // no spoofed second row
    }

    #[cfg(unix)]
    #[test]
    fn search_scrubs_invisible_format_chars_in_displayed_paths() {
        // A bidi override, zero-width space, or BOM in a filename renders as
        // nothing yet can visually reorder or hide the `path:line:` prefix — the
        // same spoofing vector as a raw control char. Each must survive as
        // U+FFFD in the displayed path component.
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a\u{202E}b.txt"), "needle\n").unwrap(); // bidi override
        fs::write(dir.path().join("c\u{200B}d.txt"), "needle\n").unwrap(); // zero-width space
        fs::write(dir.path().join("e\u{FEFF}f.txt"), "needle\n").unwrap(); // BOM / ZWNBSP

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(
            result,
            "a\u{FFFD}b.txt:1: needle\n\
             c\u{FFFD}d.txt:1: needle\n\
             e\u{FFFD}f.txt:1: needle"
        );
    }

    #[cfg(unix)]
    #[test]
    fn search_passes_plain_unicode_paths_through_untouched() {
        // Accents, CJK, and emoji are ordinary printable characters — the path
        // component must survive verbatim, only the invisible/control set is
        // scrubbed.
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("café_文件_🚀.txt"), "needle\n").unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let result = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "café_文件_🚀.txt:1: needle");
    }

    #[test]
    fn search_unbounded_sandbox_displays_absolute_paths() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "needle\n").unwrap();

        let tool = SearchFilesTool::new(Sandbox::unbounded());
        let result = tool
            .run(
                serde_json::json!({"query": "needle", "path": dir.path().to_str().unwrap()}),
                &mut std::io::sink(),
            )
            .unwrap();
        assert!(result.starts_with('/'));
        assert!(result.contains("a.txt:1: needle"));
    }

    #[cfg(unix)]
    #[test]
    fn search_unreadable_file_errors() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("secret.txt");
        fs::write(&file, "needle\n").unwrap();
        // Mode 000: stat succeeds but the read is denied — fail fast.
        fs::set_permissions(&file, fs::Permissions::from_mode(0o000)).unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap_err();
        assert!(err.contains("cannot read"));
    }

    #[cfg(unix)]
    #[test]
    fn search_unreadable_subdirectory_errors() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let locked = dir.path().join("locked");
        fs::create_dir(&locked).unwrap();
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();

        let tool = SearchFilesTool::new(sandbox_in(dir.path()));
        let err = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap_err();
        assert!(err.contains("cannot read directory"));

        // Restore permissions so the tempdir can clean itself up.
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn search_format_status() {
        let dir = tempfile::tempdir().unwrap();
        let tool = SearchFilesTool::new(sandbox_in(dir.path()));

        let input = serde_json::json!({"query": "fn main", "path": "src"});
        assert_eq!(
            tool.format_status(&input),
            Some("searching src for \"fn main\"".to_string())
        );

        let input_default = serde_json::json!({"query": "fn main"});
        assert_eq!(
            tool.format_status(&input_default),
            Some("searching . for \"fn main\"".to_string())
        );

        let input_missing = serde_json::json!({});
        assert_eq!(tool.format_status(&input_missing), None);
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
    fn search_never_returns_the_global_env_content() {
        // Walking from the sandbox root (the fake home) never scans the global
        // `.env`: `.omega-system` is a hidden directory, so the walk skips it
        // outright — a sibling file at the root still matches.
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_home_in(dir.path());
        fs::write(dir.path().join(".omega-system/.env"), "needle SECRET\n").unwrap();
        fs::write(dir.path().join("visible.txt"), "needle here\n").unwrap();

        let tool = SearchFilesTool::new(sandbox);
        let result = tool
            .run(serde_json::json!({"query": "needle"}), &mut std::io::sink())
            .unwrap();
        assert_eq!(result, "visible.txt:1: needle here");
        assert!(!result.contains("SECRET"));
    }

    #[test]
    fn search_refuses_the_global_env_as_a_direct_scope() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_home_in(dir.path());
        fs::write(dir.path().join(".omega-system/.env"), "needle\n").unwrap();

        let tool = SearchFilesTool::new(sandbox);
        let err = tool
            .run(
                serde_json::json!({"query": "needle", "path": ".omega-system/.env"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("shielded"), "got: {err}");
    }

    #[cfg(unix)]
    #[test]
    fn search_refuses_a_symlink_scope_to_the_global_env() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_home_in(dir.path());
        fs::write(dir.path().join(".omega-system/.env"), "needle\n").unwrap();
        std::os::unix::fs::symlink(
            dir.path().join(".omega-system/.env"),
            dir.path().join("link_env"),
        )
        .unwrap();

        let tool = SearchFilesTool::new(sandbox);
        let err = tool
            .run(
                serde_json::json!({"query": "needle", "path": "link_env"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("shielded"), "got: {err}");
    }

    #[test]
    fn error_text_helpers_format_reason() {
        let entry = entry_error(std::io::Error::other("io fault"));
        assert_eq!(entry, "error reading directory entry: io fault");
        let stat = stat_error(std::io::Error::other("race"));
        assert_eq!(stat, "cannot stat file: race");
    }
}
