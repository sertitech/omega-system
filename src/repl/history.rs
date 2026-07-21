//! The editor's recall history, with opt-in file persistence. The
//! in-memory `Vec` the editor core recalls over lives here, wrapped so a
//! configured `history_file` survives restarts: loaded at startup, rewritten
//! (atomically) after each submitted line by the interactive editor
//! ([`super::raw_live::TtyEditor`]).
//!
//! The disk format is plain lines — one entry per line, oldest first. An
//! entry is a submitted input line, which by construction contains no
//! newline, so there is nothing to escape and no format versioning ceremony
//! to carry (unlike [`crate::session`], whose nested structures earned JSON
//! and a version field). The write is atomic all the same (temp + rename,
//! the session-file precedent), so a crash mid-write never truncates the
//! history that was already on disk.

/// Ceiling on persisted entries: on save (and on load, for a file that grew
/// elsewhere), the oldest entries beyond this are trimmed. 1000 lines is a
/// few weeks of heavy interactive use yet keeps the file — rewritten whole
/// on every submitted line — trivially small; recall stepping 1000 entries
/// deep is already past any practical use. The in-memory store of a session
/// *without* a history file stays unbounded, as before (user-typed and
/// session-scoped).
const MAX_ENTRIES: usize = 1000;

/// The Up/Down recall store: the entries the editor core
/// ([`super::editor::edit_line`]) reads and appends, plus the optional file
/// they persist to. Constructed once per session — [`History::in_memory`]
/// without a configured `history_file` (today's behavior), or
/// [`History::load`] with one.
#[derive(Debug)]
pub struct History {
    entries: Vec<String>,
    /// The persistence target; `None` keeps the store memory-only.
    file: Option<String>,
}

impl History {
    /// A memory-only history — the default when no `history_file` is
    /// configured. [`History::save`] is a no-op.
    pub fn in_memory() -> Self {
        Self {
            entries: Vec::new(),
            file: None,
        }
    }

    /// A persistent history backed by `path`. A missing file is the first
    /// run — empty history, no error. Any other read failure (permissions,
    /// non-UTF-8 bytes) fails fast with a message naming the file, the
    /// `Session::load` policy: a history that exists but cannot be read is
    /// a real problem the operator must see, not entries to silently drop.
    /// Blank lines are skipped (the store never contains them), and a file
    /// grown past [`MAX_ENTRIES`] elsewhere is trimmed to the newest.
    pub fn load(path: &str) -> Result<Self, String> {
        let entries = match std::fs::read_to_string(path) {
            Ok(contents) => {
                let mut entries: Vec<String> = contents
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .map(str::to_string)
                    .collect();
                if entries.len() > MAX_ENTRIES {
                    entries.drain(..entries.len() - MAX_ENTRIES);
                }
                entries
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(format!("cannot read history file {path}: {e}")),
        };
        Ok(Self {
            entries,
            file: Some(path.to_string()),
        })
    }

    /// The entry store the editor core recalls over and appends to —
    /// [`super::editor::edit_line`]'s `history` parameter. Insert-time
    /// consecutive-dupe and blank suppression live there, in `remember`.
    pub fn entries_mut(&mut self) -> &mut Vec<String> {
        &mut self.entries
    }

    /// Current entry count — the caller's cheap grew-since check around an
    /// `edit_line` call, so [`History::save`] runs only when a line was
    /// actually stored.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Trim to [`MAX_ENTRIES`] (oldest first) and rewrite the file via
    /// [`crate::atomic_write`] — temp + rename, so a crash never truncates it.
    /// A no-op for a memory-only history. The temp name carries a pid+counter
    /// suffix, so two instances sharing a history file cannot clobber each
    /// other on one fixed `.tmp` path. Called after each submitted line; the
    /// file is small by the cap, so the whole-file rewrite is trivial.
    pub fn save(&mut self) -> Result<(), String> {
        let Some(path) = &self.file else {
            return Ok(());
        };
        if self.entries.len() > MAX_ENTRIES {
            self.entries.drain(..self.entries.len() - MAX_ENTRIES);
        }
        let mut contents = self.entries.join("\n");
        contents.push('\n');
        crate::atomic_write::atomic_write_text(std::path::Path::new(path), &contents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_in(dir: &tempfile::TempDir, name: &str) -> String {
        dir.path().join(name).to_str().unwrap().to_string()
    }

    #[test]
    fn in_memory_history_starts_empty_and_save_is_a_noop() {
        let mut history = History::in_memory();
        assert_eq!(history.entry_count(), 0);
        history.entries_mut().push("typed".to_string());
        history.save().unwrap();
        assert_eq!(history.entry_count(), 1);
    }

    #[test]
    fn in_memory_history_is_unbounded_across_saves() {
        // A session without a history file keeps its recall store unbounded.
        // The MAX_ENTRIES drain must stay below
        // the memory-only guard so `save()` never trims it.
        let mut history = History::in_memory();
        for i in 0..MAX_ENTRIES + 5 {
            history.entries_mut().push(format!("line {i}"));
            history.save().unwrap();
        }
        assert_eq!(history.entry_count(), MAX_ENTRIES + 5);
        assert_eq!(history.entries.first().unwrap(), "line 0");
    }

    #[test]
    fn load_missing_file_is_the_empty_first_run() {
        let dir = tempfile::tempdir().unwrap();
        let history = History::load(&path_in(&dir, "absent")).unwrap();
        assert_eq!(history.entry_count(), 0);
    }

    #[test]
    fn save_then_load_round_trips_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir, "history");
        let mut history = History::load(&path).unwrap();
        history.entries_mut().push("first".to_string());
        history.entries_mut().push("second".to_string());
        history.save().unwrap();

        let restored = History::load(&path).unwrap();
        assert_eq!(restored.entries, ["first", "second"]);
    }

    #[test]
    fn save_writes_plain_lines_with_a_trailing_newline() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir, "history");
        let mut history = History::load(&path).unwrap();
        history.entries_mut().push("a".to_string());
        history.entries_mut().push("b".to_string());
        history.save().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a\nb\n");
        // Only the target remains — the pid+counter temp file was renamed away.
        let entries: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn save_overwrites_the_previous_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir, "history");
        std::fs::write(&path, "stale\n").unwrap();
        let mut history = History::load(&path).unwrap();
        history.entries_mut().push("fresh".to_string());
        history.save().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "stale\nfresh\n");
    }

    #[test]
    fn load_skips_blank_lines() {
        // The store never contains blanks (insert-time suppression), so a
        // hand-edited or corrupted file's blank lines are dropped rather
        // than becoming un-recallable empty entries.
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir, "history");
        std::fs::write(&path, "one\n\n   \ntwo\n").unwrap();
        let history = History::load(&path).unwrap();
        assert_eq!(history.entries, ["one", "two"]);
    }

    #[test]
    fn load_unreadable_file_fails_fast_naming_the_path() {
        // Non-UTF-8 bytes: read_to_string reports InvalidData — a real
        // problem the operator must see, not entries to silently drop.
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir, "history");
        std::fs::write(&path, [0xff, 0xfe, 0x00]).unwrap();
        let err = History::load(&path).unwrap_err();
        assert!(err.contains("cannot read history file"), "got: {err}");
        assert!(err.contains("history"), "got: {err}");
    }

    #[test]
    fn save_trims_the_oldest_entries_beyond_the_cap() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir, "history");
        let mut history = History::load(&path).unwrap();
        for i in 0..MAX_ENTRIES + 3 {
            history.entries_mut().push(format!("line {i}"));
        }
        history.save().unwrap();
        // In memory and on disk alike: the newest MAX_ENTRIES survive.
        assert_eq!(history.entry_count(), MAX_ENTRIES);
        assert_eq!(history.entries.first().unwrap(), "line 3");
        let on_disk = History::load(&path).unwrap();
        assert_eq!(on_disk.entries, history.entries);
    }

    #[test]
    fn load_trims_a_file_grown_past_the_cap() {
        // A file appended to by other means still loads to the invariant.
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir, "history");
        let oversized: String = (0..MAX_ENTRIES + 2).fold(String::new(), |mut acc, i| {
            acc.push_str(&format!("line {i}\n"));
            acc
        });
        std::fs::write(&path, oversized).unwrap();
        let history = History::load(&path).unwrap();
        assert_eq!(history.entry_count(), MAX_ENTRIES);
        assert_eq!(history.entries.first().unwrap(), "line 2");
    }

    #[test]
    fn failed_write_reports_the_temp_path() {
        // The temp file cannot be created inside a path that is not a
        // directory.
        let dir = tempfile::tempdir().unwrap();
        let blocker = path_in(&dir, "not-a-dir");
        std::fs::write(&blocker, "file").unwrap();
        let inside = format!("{blocker}/history");
        let mut history = History {
            entries: vec!["x".to_string()],
            file: Some(inside),
        };
        let err = history.save().unwrap_err();
        assert!(err.contains("cannot write"), "got: {err}");
        assert!(err.contains("history"), "got: {err}");
        assert!(err.contains(".tmp"), "got: {err}");
    }

    #[test]
    fn failed_rename_reports_both_paths_and_keeps_the_old_file() {
        // Renaming a file over an existing *directory* fails after the temp
        // write succeeded — the pre-existing history must be untouched.
        let dir = tempfile::tempdir().unwrap();
        let target = path_in(&dir, "history");
        std::fs::create_dir(&target).unwrap();
        let mut history = History {
            entries: vec!["x".to_string()],
            file: Some(target.clone()),
        };
        let err = history.save().unwrap_err();
        assert!(err.contains("cannot rename"), "got: {err}");
        assert!(err.contains("history"), "got: {err}");
        assert!(err.contains(".tmp"), "got: {err}");
        assert!(std::path::Path::new(&target).is_dir());
    }

    #[test]
    fn save_survives_a_stale_fixed_tmp_from_another_process() {
        // F13 regression: the writer must not collide on a fixed `.tmp` name.
        // A stale `{path}.tmp` left by another instance — here a directory,
        // which the old fixed-name scheme would have failed the write against
        // — must not block this save, because the temp name now carries a
        // pid+counter suffix that no other process shares.
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir, "history");
        std::fs::create_dir(format!("{path}.tmp")).unwrap();
        let mut history = History::load(&path).unwrap();
        history.entries_mut().push("kept".to_string());
        history.save().unwrap();
        assert_eq!(History::load(&path).unwrap().entries, ["kept"]);
    }
}
