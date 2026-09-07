//! Complete, bounded change previews for interactive file approvals.

use crate::display::escape_for_review;
use std::io::Read;
use std::path::Path;

// A prompt must remain reviewable on a terminal. Reject large changes instead
// of silently showing only a harmless prefix of what the operator approves.
const MAX_PREVIEW_BYTES: usize = 16 * 1024;
const MAX_PREVIEW_LINES: usize = 200;
const MAX_SOURCE_BYTES: u64 = 8 * 1024 * 1024;
const TOO_LARGE: &str = "change preview omitted because it exceeds the review limit; split the change into smaller edits before asking for approval";

/// Read existing text without allowing a device, pipe, or huge file to hang or
/// exhaust an interactive approval. Creation is the only missing-file case.
pub(super) fn read_existing(path: &Path) -> Result<Option<String>, String> {
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("cannot inspect file for preview: {e}")),
    };
    if !metadata.is_file() {
        return Err("cannot preview a non-regular file".to_string());
    }
    let mut file = std::fs::File::open(path).map_err(preview_read_error)?;
    read_source(&mut file).map(Some)
}

fn preview_read_error(error: std::io::Error) -> String {
    format!("cannot read file for preview: {error}")
}

fn read_source(reader: &mut dyn Read) -> Result<String, String> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_SOURCE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(preview_read_error)?;
    if bytes.len() as u64 > MAX_SOURCE_BYTES {
        return Err("change preview omitted: source exceeds the 8 MB review limit".to_string());
    }
    String::from_utf8(bytes).map_err(|e| format!("cannot preview non-UTF-8 text: {e}"))
}

/// Show the changed span with two unchanged context lines on either side.
/// A single contiguous hunk is intentional: even disjoint rewrites show every
/// intervening line rather than relying on a heuristic diff to hide text.
pub(super) fn render(path: &str, before: Option<&str>, after: &str) -> Result<String, String> {
    let old: Vec<_> = before.unwrap_or("").split_inclusive('\n').collect();
    let new: Vec<_> = after.split_inclusive('\n').collect();
    let prefix = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
    let suffix = old[prefix..]
        .iter()
        .rev()
        .zip(new[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let old_end = old.len() - suffix;
    let new_end = new.len() - suffix;
    let start = prefix.saturating_sub(2);
    let context_end = (old_end + 2).min(old.len());
    let shown_lines =
        (prefix - start) + (old_end - prefix) + (new_end - prefix) + (context_end - old_end);
    if shown_lines > MAX_PREVIEW_LINES {
        return Err(TOO_LARGE.to_string());
    }
    let path = escape_for_review(path);
    let action = if before.is_some() {
        "replace existing file"
    } else {
        "create new file"
    };
    let mut output = format!(
        "Change preview: {action}\n--- {path} (before)\n+++ {path} (after)\nControls use visible escapes; doubled backslashes represent one backslash.\n"
    );
    if before == Some(after) {
        output.push_str("[no content changes]\n");
    } else {
        if start > 0 {
            output.push_str(&format!("[{} unchanged leading lines omitted]\n", start));
        }
        for (marker, lines, first) in [
            (' ', &old[start..prefix], start),
            ('-', &old[prefix..old_end], prefix),
            ('+', &new[prefix..new_end], prefix),
            (' ', &old[old_end..context_end], new_end),
        ] {
            for (i, line) in lines.iter().enumerate() {
                // Check raw size before escaping so a huge single line never
                // causes an equally huge temporary preview allocation.
                if output.len().saturating_add(line.len()) > MAX_PREVIEW_BYTES {
                    return Err(TOO_LARGE.to_string());
                }
                let text = escape_for_review(line.strip_suffix('\n').unwrap_or(line));
                output.push_str(&format!("{marker} {:>5} | {text}\n", first + i + 1));
                if !line.ends_with('\n') {
                    output.push_str("\\ No newline at end of file\n");
                }
            }
        }
        if context_end < old.len() {
            output.push_str(&format!(
                "[{} unchanged trailing lines omitted]\n",
                old.len() - context_end
            ));
        }
    }
    if output.len() > MAX_PREVIEW_BYTES {
        return Err(TOO_LARGE.to_string());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation_and_replacement_show_all_changed_lines_and_context() {
        let created = render("new.rs", None, "one\ntwo").unwrap();
        assert!(created.contains("create new file"));
        assert!(created.contains("+     1 | one\n+     2 | two\n\\ No newline"));
        let replaced = render(
            "file",
            Some("a\nb\nc\nd\ne\nf\ng\n"),
            "a\nb\nc\nD\ne\nf\ng\n",
        )
        .unwrap();
        assert!(replaced.contains("1 unchanged leading lines omitted"));
        assert!(replaced.contains("-     4 | d\n+     4 | D\n"));
        assert!(replaced.contains("      2 | b"));
        assert!(replaced.contains("      5 | e"));
        assert!(replaced.contains("1 unchanged trailing lines omitted"));
        assert!(
            render("file", Some("same"), "same")
                .unwrap()
                .contains("no content changes")
        );
        assert!(
            render("file", Some("delete\n"), "")
                .unwrap()
                .contains("-     1 | delete")
        );
    }

    #[test]
    fn controls_and_literal_escape_sequences_remain_distinguishable() {
        let preview = render("a\x1b[2J\u{202e}", Some("\x1b"), "\\u{1b}\t\r\u{200b}").unwrap();
        assert!(preview.contains("a\\u{1b}[2J\\u{202e}"));
        assert!(preview.contains("-     1 | \\u{1b}"));
        assert!(preview.contains("+     1 | \\\\u{1b}\\t\\r\\u{200b}"));
        assert!(!preview.contains('\x1b'));
    }

    #[test]
    fn oversized_changes_are_rejected_with_split_guidance() {
        for text in [
            "x".repeat(MAX_PREVIEW_BYTES),
            "x\n".repeat(MAX_PREVIEW_LINES + 1),
            "\u{200b}".repeat(3000),
        ] {
            assert_eq!(render("file", None, &text).unwrap_err(), TOO_LARGE);
        }
        assert!(render(&"x".repeat(MAX_PREVIEW_BYTES), Some("same"), "same").is_err());
    }

    #[test]
    fn existing_source_is_bounded_and_read_errors_propagate() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("file");
        assert!(read_existing(&file).unwrap().is_none());
        std::fs::write(&file, "before").unwrap();
        assert_eq!(read_existing(&file).unwrap().as_deref(), Some("before"));
        assert!(
            read_existing(dir.path())
                .unwrap_err()
                .contains("non-regular")
        );
        assert!(
            read_source(&mut &b"\xff"[..])
                .unwrap_err()
                .contains("non-UTF-8")
        );
        assert!(
            read_source(&mut std::io::repeat(b'a'))
                .unwrap_err()
                .contains("8 MB")
        );
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("injected failure"))
            }
        }
        assert!(
            read_source(&mut Broken)
                .unwrap_err()
                .contains("injected failure")
        );
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_file_and_inaccessible_directory_fail_closed() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("file");
        std::fs::write(&file, "before").unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o0)).unwrap();
        assert!(
            read_existing(&file)
                .unwrap_err()
                .contains("cannot read file")
        );
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o0)).unwrap();
        let result = read_existing(&file);
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(result.unwrap_err().contains("cannot inspect file"));
    }
}
