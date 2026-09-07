//! Named on-disk session store — the explicit `/save` / `/load` snapshots,
//! kept apart from the automatic `session_file` autosave.
//!
//! Saves live under `~/.omega-system/sessions/<project-key>/`, one directory
//! per project, so a `/save before-refactor` in one checkout never surfaces in
//! another. The `<project-key>` is the **canonical** project root — the same
//! canonicalized path the sandbox roots at — run through an **injective**
//! byte encoding (see [`project_key`]): two different roots can never collide
//! onto one key, and two symlinked checkouts of the same tree collapse onto
//! one. Plain string munging fails both directions at once — `/a/b`, `/a-b`,
//! and `/a b` would share a key, and a symlinked checkout would split from its
//! real path — which is exactly what the canonical-plus-injective pairing
//! rules out.

use crate::session::Session;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// A per-project directory of named session snapshots.
pub struct SessionStore {
    /// `~/.omega-system/sessions/<project-key>/`. Created lazily on the first
    /// save, so merely opening a project writes nothing.
    dir: PathBuf,
}

/// One saved session as it appears in a listing — metadata only, from the
/// directory stat, never the file body (a content preview
/// would mean deserializing a whole `Session` per row). `size` is the on-disk
/// byte count [`SessionStore::list`] already reads while stat-ing each entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionEntry {
    /// The save name — its filename stem, without the `.json` extension.
    pub name: String,
    /// The file's modification time, the basis for both the newest-first sort
    /// and the relative age in [`SessionEntry::describe`].
    pub modified: SystemTime,
    /// The file size in bytes.
    pub size: u64,
}

impl SessionEntry {
    /// A one-line, metadata-only summary — name, relative age, size — shared by
    /// the `--resume` picker and the `/sessions` / `/load` listings, so every
    /// listing renders the same way. Columns are space-padded so a run of rows
    /// lines up; a name wider than the column simply overflows it.
    pub fn describe(&self, now: SystemTime) -> String {
        format!(
            "{:<24}{:>8}   {}",
            self.name,
            format_age(self.modified, now),
            format_size(self.size),
        )
    }
}

impl SessionStore {
    /// Root a store at `<home>/.omega-system/sessions/<project-key>/`.
    ///
    /// `project_root` is canonicalized here so symlinked checkouts of one tree
    /// share a key; a root that cannot be canonicalized (it does not exist) is
    /// keyed verbatim, which is harmless — a project you cannot resolve has no
    /// saves to find under either spelling.
    pub fn new(home: &Path, project_root: &Path) -> SessionStore {
        let canonical =
            std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
        let dir = home
            .join(".omega-system")
            .join("sessions")
            .join(project_key(&canonical));
        SessionStore { dir }
    }

    /// Write `session` to `<name>.json`, creating the store directory if it is
    /// not there yet. The name is validated first (see [`validate_name`]); the
    /// write commits through the same temp-file+rename path as every other
    /// on-disk writer, so a crash mid-save leaves any prior snapshot intact.
    pub fn save(&self, name: &str, session: &Session) -> Result<(), String> {
        validate_name(name)?;
        self.ensure_dir()?;
        session.save_to(&self.dir.join(format!("{name}.json")))
    }

    /// Atomically choose and save under `base`, `base-2`, `base-3`, … without
    /// overwriting an automatic session created by another process. A
    /// create-new reservation serializes first-save allocation across Omega
    /// instances; the ordinary atomic writer remains the commit path.
    pub fn save_auto(&self, base: &str, session: &Session) -> Result<String, String> {
        self.save_auto_with(base, &|name| self.save(name, session))
    }

    /// The allocation half of [`SessionStore::save_auto`], seamed at the
    /// commit so failure and cleanup behavior stays hermetically testable.
    fn save_auto_with(
        &self,
        base: &str,
        commit: &dyn Fn(&str) -> Result<(), String>,
    ) -> Result<String, String> {
        self.ensure_dir()?;
        let mut suffix = None;
        loop {
            let candidate = suffix
                .map(|number| format!("{base}-{number}"))
                .unwrap_or_else(|| base.to_string());
            validate_name(&candidate)?;
            let reservation = self.dir.join(format!("{candidate}.reserve"));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&reservation)
            {
                Ok(file) => drop(file),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    suffix = Some(suffix.map_or(2, |number| number + 1));
                    continue;
                }
                Err(error) => {
                    return Err(format!(
                        "cannot reserve automatic session '{candidate}' at {}: {error}",
                        reservation.display()
                    ));
                }
            }

            let target = self.dir.join(format!("{candidate}.json"));
            if target.exists() {
                self.remove_reservation(&reservation)?;
                suffix = Some(suffix.map_or(2, |number| number + 1));
                continue;
            }

            let saved = commit(&candidate);
            let cleaned = self.remove_reservation(&reservation);
            return match (saved, cleaned) {
                (Ok(()), Ok(())) => Ok(candidate),
                (Err(error), Ok(())) => Err(error),
                (Ok(()), Err(cleanup)) => Err(cleanup),
                (Err(error), Err(cleanup)) => Err(format!("{error}; {cleanup}")),
            };
        }
    }

    fn remove_reservation(&self, path: &Path) -> Result<(), String> {
        std::fs::remove_file(path).map_err(|error| {
            format!(
                "cannot remove session reservation {}: {error}",
                path.display()
            )
        })
    }

    /// Read the snapshot saved as `name`. A name that was never saved fails
    /// with a clear "no saved session" error the REPL turns into a soft prompt
    /// listing what *is* available, rather than the raw file-not-found of the
    /// underlying read.
    pub fn load(&self, name: &str) -> Result<Session, String> {
        validate_name(name)?;
        let path = self.dir.join(format!("{name}.json"));
        if !path.is_file() {
            return Err(format!("no saved session named '{name}'"));
        }
        Session::load_from(&path)
    }

    /// The saved sessions as [`SessionEntry`] rows, newest first.
    ///
    /// Only regular files ending in `.json` count: the atomic writer can leave
    /// a `*.tmp` sibling behind after a crash, and a stray subdirectory could
    /// share the `.json` suffix — neither is a save, and neither may
    /// masquerade as one in the listing. A missing store directory (nothing
    /// saved yet) is simply an empty list.
    pub fn list(&self) -> Vec<SessionEntry> {
        let mut saves = Vec::new();
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return saves;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if let Ok(meta) = entry.metadata()
                && meta.is_file()
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                && let Ok(modified) = meta.modified()
            {
                saves.push(SessionEntry {
                    name: stem.to_string(),
                    modified,
                    size: meta.len(),
                });
            }
        }
        // Newest first — the order an operator scans for their most recent save.
        saves.sort_by(|a, b| b.modified.cmp(&a.modified));
        saves
    }

    /// Delete the save stored under `name`. The name is validated first, so a
    /// path-traversal argument cannot reach outside the store; a name that was
    /// never saved fails with the same clear "no saved session" miss [`load`]
    /// reports; otherwise the file is removed, surfacing any I/O error rather
    /// than swallowing it. The one pruning tool for the auto-saves that now
    /// accumulate one per session — deleting the live session's own auto file
    /// is harmless, since the next autosave recreates it through [`save`].
    ///
    /// [`load`]: SessionStore::load
    /// [`save`]: SessionStore::save
    pub fn delete(&self, name: &str) -> Result<(), String> {
        validate_name(name)?;
        let path = self.dir.join(format!("{name}.json"));
        if !path.is_file() {
            return Err(format!("no saved session named '{name}'"));
        }
        std::fs::remove_file(&path).map_err(|e| format!("cannot delete session '{name}': {e}"))
    }

    /// Create the store directory (and its parents) at mode `0700`. A saved
    /// session holds the full conversation — every message and tool result —
    /// so the directory must not be world- or group-readable. `recursive`
    /// makes a repeat save a no-op once the directory exists.
    fn ensure_dir(&self) -> Result<(), String> {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&self.dir)
            .map_err(|e| {
                format!(
                    "cannot create session directory {}: {e}",
                    self.dir.display()
                )
            })
    }
}

/// Encode a canonical path into an injective, filesystem-safe key.
///
/// ASCII alphanumerics pass through unchanged; every other byte becomes `-HH`
/// (two uppercase hex digits). The `-` prefix is itself a non-alphanumeric
/// byte and so encodes as `-2D`, which is what makes the scheme injective: no
/// literal `-` in the input can ever be confused with an escape marker, so
/// distinct paths always produce distinct keys. `/a/b` → `-2Fa-2Fb`,
/// `/a-b` → `-2Fa-2Db`, `/a b` → `-2Fa-20b` — three keys, no collision.
pub(crate) fn project_key(root: &Path) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut key = String::new();
    for &byte in root.as_os_str().as_bytes() {
        if byte.is_ascii_alphanumeric() {
            key.push(byte as char);
        } else {
            key.push('-');
            key.push(HEX[(byte >> 4) as usize] as char);
            key.push(HEX[(byte & 0x0f) as usize] as char);
        }
    }
    key
}

/// Reject any save name that could reach outside its store directory or hide.
///
/// Names must match `[A-Za-z0-9._-]+` with no leading dot: no `/` (path
/// separators), no `..` traversal (a leading `.` is refused outright, so
/// `..` never begins a name), and no dotfiles. The bounded character set also
/// keeps the `<name>.json` filename portable.
fn validate_name(name: &str) -> Result<(), String> {
    let valid = !name.is_empty()
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if valid {
        Ok(())
    } else {
        Err(format!(
            "invalid session name '{name}': use letters, digits, '.', '_', or '-', with no leading dot"
        ))
    }
}

/// Render the age of a save as a coarse `Ns`/`Nm`/`Nh`/`Nd ago` bucket — a
/// human glance-able time signal, not a precise timestamp. A `modified` time in
/// the future relative to `now` (clock skew, a copied-in file) reads as `just
/// now` rather than an underflowed or absurd age.
fn format_age(modified: SystemTime, now: SystemTime) -> String {
    let Ok(elapsed) = now.duration_since(modified) else {
        return "just now".to_string();
    };
    let secs = elapsed.as_secs();
    if secs < 60 {
        format!("{secs}s ago")
    } else if secs < 3_600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86_400 {
        format!("{}h ago", secs / 3_600)
    } else {
        format!("{}d ago", secs / 86_400)
    }
}

/// Render a byte count as `B`/`KB`/`MB`, one decimal above bytes. Session files
/// are small per-project JSON, so `MB` is the largest unit worth showing.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    if bytes < KB {
        format!("{bytes} B")
    } else if bytes < MB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SESSION_VERSION;
    use crate::turn::{Block, Role, TurnMessage};
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    /// A small but non-empty session to round-trip through the store.
    fn session() -> Session {
        Session {
            version: SESSION_VERSION,
            messages: vec![TurnMessage {
                role: Role::User,
                content: vec![Block::Text("hi".to_string())],
            }],
            compacted_summary: Some("earlier".to_string()),
            last_input_tokens: 42,
        }
    }

    fn session_with_text(text: &str) -> Session {
        let mut value = session();
        value.messages[0].content = vec![Block::Text(text.to_string())];
        value
    }

    // ── project-key encoding ──

    #[test]
    fn key_passes_alphanumerics_and_escapes_the_rest() {
        // Alphanumerics survive; every other byte is `-HH`. `/` → `-2F`.
        assert_eq!(project_key(Path::new("/aZ9")), "-2FaZ9");
    }

    #[test]
    fn key_is_injective_across_the_collision_triple() {
        // The three inputs plain munging would fold together stay distinct,
        // because `-` itself encodes (`-2D`) rather than passing through.
        let slash = project_key(Path::new("/a/b"));
        let dash = project_key(Path::new("/a-b"));
        let space = project_key(Path::new("/a b"));
        assert_eq!(slash, "-2Fa-2Fb");
        assert_eq!(dash, "-2Fa-2Db");
        assert_eq!(space, "-2Fa-20b");
        assert_ne!(slash, dash);
        assert_ne!(dash, space);
        assert_ne!(slash, space);
    }

    #[test]
    fn key_encodes_the_root_path() {
        // A single `/` — the shortest possible root.
        assert_eq!(project_key(Path::new("/")), "-2F");
    }

    #[test]
    fn key_encodes_unicode_byte_for_byte() {
        // Multi-byte UTF-8 encodes each byte independently: `π` is 0xCF 0x80.
        assert_eq!(project_key(Path::new("/π")), "-2F-CF-80");
    }

    #[test]
    fn new_canonicalizes_so_symlinked_checkouts_share_a_key() {
        // A real project directory reached two ways — directly and through a
        // symlink alias — must key to the same store directory.
        let home = tempfile::tempdir().unwrap();
        let real = tempfile::tempdir().unwrap();
        let alias = real.path().parent().unwrap().join("omega-alias-link");
        std::os::unix::fs::symlink(real.path(), &alias).unwrap();

        let direct = SessionStore::new(home.path(), real.path());
        let via_link = SessionStore::new(home.path(), &alias);
        assert_eq!(direct.dir, via_link.dir);

        std::fs::remove_file(&alias).unwrap();
    }

    #[test]
    fn new_keys_an_unresolvable_root_verbatim() {
        // A root that cannot be canonicalized falls back to its raw bytes —
        // covering the fallback path without a symlink.
        let home = tempfile::tempdir().unwrap();
        let store = SessionStore::new(home.path(), Path::new("/no/such/omega-project"));
        assert!(
            store
                .dir
                .ends_with(project_key(Path::new("/no/such/omega-project")))
        );
    }

    // ── name validation ──

    #[test]
    fn validate_accepts_the_allowed_shape() {
        // Alphanumerics, dot, underscore, hyphen — with no leading dot. A
        // `.tmp`-*suffixed* name is fine; only a leading dot is refused.
        for name in ["a", "before-refactor", "v1.2.3", "A_b-9", "snapshot.tmp"] {
            assert!(validate_name(name).is_ok(), "rejected {name}");
        }
    }

    #[test]
    fn validate_rejects_separators_traversal_and_dotfiles() {
        // Every rejected shape reports the same clear error. `.tmp` is refused
        // here as a leading-dot dotfile, not as an extension.
        for name in ["", "../x", "a/b", "a b", ".hidden", ".tmp", "café", "a\0b"] {
            let err = validate_name(name).unwrap_err();
            assert!(err.contains("invalid session name"), "for {name:?}: {err}");
        }
    }

    // ── save / load ──

    /// A store under a fresh home and project, ready to save into.
    fn fresh_store() -> (tempfile::TempDir, tempfile::TempDir, SessionStore) {
        let home = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        let store = SessionStore::new(home.path(), project.path());
        (home, project, store)
    }

    #[test]
    fn save_and_load_round_trip_through_the_store() {
        let (_home, _project, store) = fresh_store();
        let s = session();
        store.save("mine", &s).unwrap();
        assert_eq!(store.load("mine").unwrap(), s);
    }

    #[test]
    fn save_rejects_an_invalid_name() {
        let (_home, _project, store) = fresh_store();
        let err = store.save("../escape", &session()).unwrap_err();
        assert!(err.contains("invalid session name"), "got: {err}");
    }

    #[test]
    fn load_rejects_an_invalid_name() {
        let (_home, _project, store) = fresh_store();
        let err = store.load("a/b").unwrap_err();
        assert!(err.contains("invalid session name"), "got: {err}");
    }

    #[test]
    fn load_unknown_name_reports_it() {
        let (_home, _project, store) = fresh_store();
        let err = store.load("ghost").unwrap_err();
        assert!(err.contains("no saved session named 'ghost'"), "got: {err}");
    }

    #[test]
    fn save_surfaces_a_directory_creation_failure() {
        // A regular file sits where the `.omega-system` directory must be, so
        // the recursive create cannot make the store directory beneath it —
        // the error is surfaced, not swallowed.
        let home = tempfile::tempdir().unwrap();
        std::fs::write(home.path().join(".omega-system"), "x").unwrap();
        let project = tempfile::tempdir().unwrap();
        let store = SessionStore::new(home.path(), project.path());
        let err = store.save("s", &session()).unwrap_err();
        assert!(
            err.contains("cannot create session directory"),
            "got: {err}"
        );
    }

    #[test]
    fn store_directory_is_created_0700() {
        use std::os::unix::fs::PermissionsExt;
        let (_home, _project, store) = fresh_store();
        store.save("s", &session()).unwrap();
        let mode = std::fs::metadata(&store.dir).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700);
    }

    #[test]
    fn save_auto_skips_an_existing_target_without_overwriting_it() {
        let (_home, _project, store) = fresh_store();
        let original = session_with_text("original");
        let newer = session_with_text("newer");
        store.save("auto-fixed", &original).unwrap();

        let name = store.save_auto("auto-fixed", &newer).unwrap();

        assert_eq!(name, "auto-fixed-2");
        assert_eq!(store.load("auto-fixed").unwrap(), original);
        assert_eq!(store.load("auto-fixed-2").unwrap(), newer);
    }

    #[test]
    fn save_auto_skips_a_preexisting_reservation() {
        let (_home, _project, store) = fresh_store();
        store.ensure_dir().unwrap();
        std::fs::write(store.dir.join("auto-fixed.reserve"), "").unwrap();

        let name = store.save_auto("auto-fixed", &session()).unwrap();

        assert_eq!(name, "auto-fixed-2");
        assert!(store.dir.join("auto-fixed.reserve").exists());
        assert!(!store.dir.join("auto-fixed-2.reserve").exists());
    }

    #[test]
    fn save_auto_reservations_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let (_home, _project, store) = fresh_store();
        let reservation = store.dir.join("auto-fixed.reserve");

        let err = store
            .save_auto_with("auto-fixed", &|_| {
                let mode = std::fs::metadata(&reservation)
                    .unwrap()
                    .permissions()
                    .mode();
                assert_eq!(mode & 0o777, 0o600);
                Err("forced save failure".to_string())
            })
            .unwrap_err();

        assert_eq!(err, "forced save failure");
        assert!(!reservation.exists());
    }

    #[test]
    fn concurrent_save_auto_calls_preserve_every_unique_session() {
        const COUNT: usize = 32;
        let (home, project, store) = fresh_store();
        let home_path = home.path().to_path_buf();
        let project_path = project.path().to_path_buf();
        let barrier = Arc::new(Barrier::new(COUNT));

        let names = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..COUNT)
                .map(|index| {
                    let home_path = home_path.clone();
                    let project_path = project_path.clone();
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        let store = SessionStore::new(&home_path, &project_path);
                        let value = session_with_text(&format!("session-{index}"));
                        barrier.wait();
                        store.save_auto("auto-fixed", &value).unwrap()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        assert_eq!(names.iter().collect::<HashSet<_>>().len(), COUNT);
        assert_eq!(store.list().len(), COUNT);
        for index in 0..COUNT {
            let expected = session_with_text(&format!("session-{index}"));
            assert!(
                names
                    .iter()
                    .any(|name| store.load(name).unwrap() == expected)
            );
        }
    }

    #[test]
    fn save_auto_failure_releases_its_reservation() {
        let (_home, _project, store) = fresh_store();
        let reservation = store.dir.join("auto-fixed.reserve");

        let err = store
            .save_auto_with("auto-fixed", &|_| Err("save failed".to_string()))
            .unwrap_err();

        assert_eq!(err, "save failed");
        assert!(!reservation.exists());
    }

    #[test]
    fn save_auto_surfaces_reservation_creation_errors_with_the_path() {
        let (_home, _project, store) = fresh_store();
        let too_long = "a".repeat(300);

        let err = store.save_auto(&too_long, &session()).unwrap_err();

        assert!(
            err.contains("cannot reserve automatic session"),
            "got: {err}"
        );
        assert!(err.contains(".reserve"), "got: {err}");
    }

    #[test]
    fn save_auto_surfaces_cleanup_failure_after_a_successful_commit() {
        let (_home, _project, store) = fresh_store();
        let reservation = store.dir.join("auto-fixed.reserve");

        let err = store
            .save_auto_with("auto-fixed", &|name| {
                store.save(name, &session())?;
                std::fs::remove_file(&reservation).unwrap();
                std::fs::create_dir(&reservation).unwrap();
                Ok(())
            })
            .unwrap_err();

        assert!(
            err.contains("cannot remove session reservation"),
            "got: {err}"
        );
        assert!(store.dir.join("auto-fixed.json").is_file());
    }

    #[test]
    fn save_auto_appends_cleanup_failure_to_the_primary_save_error() {
        let (_home, _project, store) = fresh_store();
        let reservation = store.dir.join("auto-fixed.reserve");

        let err = store
            .save_auto_with("auto-fixed", &|_| {
                std::fs::remove_file(&reservation).unwrap();
                std::fs::create_dir(&reservation).unwrap();
                Err("save failed".to_string())
            })
            .unwrap_err();

        assert!(err.starts_with("save failed;"), "got: {err}");
        assert!(
            err.contains("cannot remove session reservation"),
            "got: {err}"
        );
    }

    #[test]
    fn save_auto_rejects_an_invalid_base() {
        let (_home, _project, store) = fresh_store();
        let err = store.save_auto("../escape", &session()).unwrap_err();
        assert!(err.contains("invalid session name"), "got: {err}");
    }

    // ── listing ──

    fn at(secs: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
    }

    fn set_mtime(path: &Path, when: SystemTime) {
        let f = std::fs::File::options().write(true).open(path).unwrap();
        f.set_times(std::fs::FileTimes::new().set_modified(when))
            .unwrap();
    }

    #[test]
    fn list_is_empty_when_nothing_saved() {
        // No store directory yet — the read_dir miss is an empty list, not an
        // error.
        let (_home, _project, store) = fresh_store();
        assert!(store.list().is_empty());
    }

    #[test]
    fn list_filters_non_json_and_sorts_newest_first() {
        let (_home, _project, store) = fresh_store();
        store.save("old", &session()).unwrap();
        store.save("new", &session()).unwrap();
        set_mtime(&store.dir.join("old.json"), at(100));
        set_mtime(&store.dir.join("new.json"), at(200));
        // Noise that must never appear as a save: a crash-left temp file, a
        // sibling with no extension, and a directory that shares the suffix.
        std::fs::write(store.dir.join("crash.tmp"), "x").unwrap();
        std::fs::write(store.dir.join("README"), "x").unwrap();
        std::fs::create_dir(store.dir.join("adir.json")).unwrap();

        let names: Vec<String> = store.list().into_iter().map(|e| e.name).collect();
        assert_eq!(names, ["new", "old"]);
    }

    #[test]
    fn list_carries_the_on_disk_size() {
        // The row's `size` is the real file length, so the picker can show it
        // without a second stat.
        let (_home, _project, store) = fresh_store();
        store.save("mine", &session()).unwrap();
        let saves = store.list();
        assert_eq!(saves.len(), 1);
        let bytes = std::fs::metadata(store.dir.join("mine.json"))
            .unwrap()
            .len();
        assert_eq!(saves[0].size, bytes);
        assert!(bytes > 0);
    }

    // ── delete ──

    #[test]
    fn delete_removes_an_existing_save() {
        let (_home, _project, store) = fresh_store();
        store.save("mine", &session()).unwrap();
        assert_eq!(store.list().len(), 1);
        store.delete("mine").unwrap();
        assert!(store.list().is_empty());
    }

    #[test]
    fn delete_rejects_an_invalid_name() {
        let (_home, _project, store) = fresh_store();
        let err = store.delete("../escape").unwrap_err();
        assert!(err.contains("invalid session name"), "got: {err}");
    }

    #[test]
    fn delete_reports_a_missing_save() {
        let (_home, _project, store) = fresh_store();
        let err = store.delete("ghost").unwrap_err();
        assert!(err.contains("no saved session named 'ghost'"), "got: {err}");
    }

    #[test]
    fn delete_surfaces_a_remove_failure() {
        use std::os::unix::fs::PermissionsExt;
        // The file exists, but its directory is read-only, so `remove_file`
        // fails with a permission error the store surfaces rather than
        // swallowing. Perms are restored so the tempdir can be cleaned up.
        let (_home, _project, store) = fresh_store();
        store.save("mine", &session()).unwrap();
        let ro = std::fs::Permissions::from_mode(0o500);
        std::fs::set_permissions(&store.dir, ro).unwrap();
        let err = store.delete("mine").unwrap_err();
        std::fs::set_permissions(&store.dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(err.contains("cannot delete session 'mine'"), "got: {err}");
    }

    #[test]
    fn deleting_the_live_auto_save_lets_the_next_autosave_recreate_it() {
        // Deleting the current session's file is harmless: saving under the
        // same name writes it back through the ordinary `save` path.
        let (_home, _project, store) = fresh_store();
        store.save("auto-19700101-000000", &session()).unwrap();
        store.delete("auto-19700101-000000").unwrap();
        assert!(store.list().is_empty());
        store.save("auto-19700101-000000", &session()).unwrap();
        assert_eq!(store.list().len(), 1);
    }

    // ── describe / age / size formatting ──

    #[test]
    fn format_age_buckets_at_each_boundary() {
        let base = at(1_000_000);
        let age = |secs: u64| format_age(base, base + Duration::from_secs(secs));
        assert_eq!(age(0), "0s ago");
        assert_eq!(age(59), "59s ago");
        assert_eq!(age(60), "1m ago");
        assert_eq!(age(3_599), "59m ago");
        assert_eq!(age(3_600), "1h ago");
        assert_eq!(age(86_399), "23h ago");
        assert_eq!(age(86_400), "1d ago");
    }

    #[test]
    fn format_age_treats_clock_skew_as_just_now() {
        // A modification time after `now` — clock skew or a copied file — must
        // not underflow; it reads as `just now`.
        let now = at(1_000);
        assert_eq!(format_age(now + Duration::from_secs(5), now), "just now");
    }

    #[test]
    fn format_size_across_the_unit_boundaries() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1023), "1023 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(12_698), "12.4 KB");
        assert_eq!(format_size(1_048_575), "1024.0 KB");
        assert_eq!(format_size(1_048_576), "1.0 MB");
        assert_eq!(format_size(3_355_443), "3.2 MB");
    }

    #[test]
    fn describe_renders_name_age_and_size() {
        let now = at(1_000_000);
        let entry = SessionEntry {
            name: "before-refactor".to_string(),
            modified: now - Duration::from_secs(180),
            size: 12_698,
        };
        let row = entry.describe(now);
        assert!(row.starts_with("before-refactor"));
        assert!(row.contains("3m ago"));
        assert!(row.contains("12.4 KB"));
    }
}
