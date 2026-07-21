//! The crate's one atomic text-write primitive, shared by every on-disk
//! writer: `write_file`, `edit_file`, [`crate::session::Session::save`], and
//! [`crate::repl::history::History::save`].
//!
//! A write must never leave a target half-overwritten when it fails partway
//! through, so every writer commits through a sibling temp file and renames it
//! over the target. The rename is the commit point: it is atomic within one
//! filesystem, so the target is only ever the old bytes or the new bytes.
//!
//! The temp name carries the pid and a per-process counter rather than a fixed
//! `.tmp` suffix, because two writers to the same directory must not race on
//! one temp path — whether that is the agent fanning concurrent tool calls out
//! to worker threads, or two Omega instances sharing a session/history file.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Distinguishes concurrent temp files within one process. Paired with the pid
/// it makes the temp name collision-resistant without a new dependency.
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write `content` to `path` atomically: to a sibling temp file first, then
/// rename over the target. A failure at either step leaves any existing target
/// intact and cleans up the temp file on a best-effort basis. Rename errors
/// name both paths, since the failure is about moving one over the other.
pub(crate) fn atomic_write_text(path: &Path, content: &str) -> Result<(), String> {
    let tmp = temp_sibling(path);

    if let Err(e) = write_private(&tmp, content) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("cannot write {}: {e}", tmp.display()));
    }

    if let Err(e) = copy_mode(path, &tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }

    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!(
            "cannot rename {} over {}: {e}",
            tmp.display(),
            path.display()
        )
    })
}

/// Write `content` to `tmp`, creating it owner-only (`0o600`) up front on unix
/// so the payload — which may be the full new secret — is never briefly
/// world-readable at the umask default before [`copy_mode`] narrows it. The mode
/// only ever tightens: `copy_mode` still widens an overwrite to the target's
/// mode, while a fresh create keeps the owner-only temp. On non-unix the window
/// is not a mode concern (Windows permissions are ACL-based), so the plain write
/// stands and the crate still builds everywhere.
#[cfg(unix)]
fn write_private(tmp: &Path, content: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(tmp)?;
    f.write_all(content.as_bytes())
}

#[cfg(not(unix))]
fn write_private(tmp: &Path, content: &str) -> std::io::Result<()> {
    std::fs::write(tmp, content)
}

/// Copy `target`'s permissions onto `tmp`, so the rename that commits `tmp`
/// over `target` does not reset them. The rename swaps the temp's inode in for
/// the target's, so without this an edit would strip a 0755 script's execute
/// bit or widen a 0600 file to the temp's default mode. A missing target is the
/// create case — leave `tmp` at its owner-only create mode (`write_private`);
/// any other stat error is surfaced rather than assumed absent, since it may
/// mean the target exists but is unreadable.
///
/// Split out from `atomic_write_text` so the apply-failure branch — unreachable
/// there, where `tmp` was just written and is owned by this process — is still
/// exercised by a direct unit test.
fn copy_mode(target: &Path, tmp: &Path) -> Result<(), String> {
    match std::fs::metadata(target) {
        Ok(meta) => std::fs::set_permissions(tmp, meta.permissions())
            .map_err(|e| format!("cannot set permissions on {}: {e}", tmp.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("cannot stat {}: {e}", target.display())),
    }
}

/// A temp path beside `path` in the same directory (so the rename stays within
/// one filesystem). The suffix is appended to the whole path, which keeps the
/// temp a sibling without splitting the path into parent/filename.
fn temp_sibling(path: &Path) -> PathBuf {
    let pid = std::process::id();
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut name = path.as_os_str().to_os_string();
    name.push(format!(".{pid}.{n}.tmp"));
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn creates_a_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("new.txt");

        atomic_write_text(&target, "hello").unwrap();

        assert_eq!(fs::read_to_string(&target).unwrap(), "hello");
    }

    #[test]
    fn overwrites_an_existing_file_leaving_no_temp() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("exist.txt");
        fs::write(&target, "old").unwrap();

        atomic_write_text(&target, "new").unwrap();

        assert_eq!(fs::read_to_string(&target).unwrap(), "new");
        // Only the target remains — the temp file was renamed away.
        let entries: Vec<_> = fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn distinct_temp_names_per_call() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("f.txt");
        let a = temp_sibling(&target);
        let b = temp_sibling(&target);
        assert_ne!(a, b);
        // The temp sits beside the target, in the same directory.
        assert_eq!(a.parent(), target.parent());
    }

    #[cfg(unix)]
    #[test]
    fn temp_write_failure_leaves_the_target_intact() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("ro");
        fs::create_dir(&sub).unwrap();
        let target = sub.join("keep.txt");
        fs::write(&target, "orig").unwrap();
        // Read-only directory: the temp file cannot be created, so the write
        // fails before any rename could touch the target.
        fs::set_permissions(&sub, fs::Permissions::from_mode(0o555)).unwrap();

        let err = atomic_write_text(&target, "new").unwrap_err();
        assert!(err.contains("cannot write"), "got: {err}");

        fs::set_permissions(&sub, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "orig");
    }

    #[cfg(unix)]
    #[test]
    fn overwrite_preserves_an_execute_bit() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("script.sh");
        fs::write(&target, "#!/bin/sh\necho old\n").unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o755)).unwrap();

        // The write_file / edit_file tools both commit through this one call,
        // so preserving the mode here preserves it for both.
        atomic_write_text(&target, "#!/bin/sh\necho new\n").unwrap();

        assert_eq!(
            fs::read_to_string(&target).unwrap(),
            "#!/bin/sh\necho new\n"
        );
        let mode = fs::metadata(&target).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o755);
    }

    #[cfg(unix)]
    #[test]
    fn overwrite_preserves_a_restrictive_mode() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("secret");
        fs::write(&target, "old").unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o600)).unwrap();

        atomic_write_text(&target, "new").unwrap();

        let mode = fs::metadata(&target).unwrap().permissions().mode();
        // Not widened to the temp file's default 0644.
        assert_eq!(mode & 0o777, 0o600);
    }

    #[cfg(unix)]
    #[test]
    fn new_file_lands_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("fresh.txt");

        // A missing target hits copy_mode's NotFound arm: no mode is copied, so
        // the file keeps write_private's owner-only create mode (0o600) rather
        // than the wider umask default — no transient world-readable window.
        atomic_write_text(&target, "hello").unwrap();

        let mode = fs::metadata(&target).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[cfg(unix)]
    #[test]
    fn write_private_creates_the_temp_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let tmp = dir.path().join("secret.tmp");

        write_private(&tmp, "payload").unwrap();

        assert_eq!(fs::read_to_string(&tmp).unwrap(), "payload");
        let mode = fs::metadata(&tmp).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[cfg(unix)]
    #[test]
    fn stat_failure_other_than_not_found_is_surfaced() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("loop");
        // A self-referential symlink: stat follows it and fails with a loop
        // error (not NotFound), so the write must abort rather than assume the
        // target is absent and silently drop its (unknowable) permissions.
        symlink(&target, &target).unwrap();

        let err = atomic_write_text(&target, "new").unwrap_err();
        assert!(err.contains("cannot stat"), "got: {err}");

        // The temp file was cleaned up; only the dangling symlink remains.
        let entries: Vec<_> = fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn copy_mode_surfaces_an_apply_failure() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("real");
        fs::write(&target, "x").unwrap();
        // `atomic_write_text` only ever applies the mode to a temp it just
        // wrote, so the apply cannot fail there; drive it directly with a
        // destination that does not exist to cover the surfaced error.
        let missing = dir.path().join("gone");

        let err = copy_mode(&target, &missing).unwrap_err();
        assert!(err.contains("cannot set permissions"), "got: {err}");
    }

    #[test]
    fn rename_failure_after_temp_write_reports_both_paths() {
        let dir = tempfile::tempdir().unwrap();
        // A directory at the target makes the temp write succeed and the
        // rename fail (a file cannot replace a directory) — the commit point
        // itself erroring, with the target left untouched.
        let target = dir.path().join("adir");
        fs::create_dir(&target).unwrap();

        let err = atomic_write_text(&target, "new").unwrap_err();
        assert!(err.contains("cannot rename"), "got: {err}");
        assert!(err.contains(target.to_str().unwrap()), "got: {err}");

        // The target is still the directory, and the temp file was cleaned up.
        assert!(target.is_dir());
        let entries: Vec<_> = fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(entries.len(), 1);
    }
}
