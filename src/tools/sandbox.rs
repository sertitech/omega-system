use std::path::{Path, PathBuf};

/// Filesystem sandbox that restricts tool access to a root directory.
///
/// When `root` is `Some`, all paths are canonicalized and checked against the
/// boundary before any read/write. When `root` is `None`, paths are used as-is
/// (unbounded mode).
#[derive(Debug, Clone)]
pub struct Sandbox {
    root: Option<PathBuf>,
    /// The canonical `~/.omega-system` directory whose credential files are
    /// shielded from the model's filesystem tools, or `None` when there is no
    /// home directory or it does not exist yet. Canonicalized at construction
    /// so the protected-path queries compare canonical parents (symlink aliases
    /// collapse); a `None` here makes every query inert.
    protected_home: Option<PathBuf>,
}

impl Sandbox {
    /// Create a sandbox rooted at the given directory. All resolved paths must
    /// fall within this root or the operation is rejected.
    ///
    /// The root is canonicalized at construction time so that boundary checks
    /// compare two canonical paths (avoiding symlink mismatches like macOS's
    /// `/var` → `/private/var`).
    ///
    /// Fails if the root directory does not exist or cannot be canonicalized.
    pub fn rooted(path: PathBuf) -> Result<Self, String> {
        let canonical = std::fs::canonicalize(&path)
            .map_err(|e| format!("cannot canonicalize sandbox root '{}': {e}", path.display()))?;
        Ok(Self {
            root: Some(canonical),
            protected_home: None,
        })
    }

    /// Create an unbounded sandbox — no path restrictions.
    pub fn unbounded() -> Self {
        Self {
            root: None,
            protected_home: None,
        }
    }

    /// The canonical sandbox root, or `None` when unbounded.
    pub fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    /// Resolve a user-provided path and verify it falls within the sandbox.
    ///
    /// - Relative paths are joined to the root (or CWD if unbounded).
    /// - Symlinks are fully resolved via `canonicalize` before boundary checks.
    /// - Returns the canonical absolute path on success.
    pub fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let root = match &self.root {
            Some(r) => r,
            None => return abs_or_cwd(path),
        };

        let candidate = if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            root.join(path)
        };

        let resolved = std::fs::canonicalize(&candidate)
            .map_err(|e| format!("cannot resolve path '{}': {e}", candidate.display()))?;

        if !resolved.starts_with(root) {
            return Err(format!("path escapes sandbox: {path}"));
        }

        Ok(resolved)
    }

    /// Resolve for writes — the target file may not exist yet, so we
    /// canonicalize the parent directory and append the filename.
    pub fn resolve_for_write(&self, path: &str) -> Result<PathBuf, String> {
        let root = match &self.root {
            Some(r) => r,
            None => return abs_or_cwd(path),
        };

        let candidate = if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            root.join(path)
        };

        let parent = candidate
            .parent()
            .ok_or_else(|| format!("path has no parent: {path}"))?;

        let resolved_parent = std::fs::canonicalize(parent).map_err(|e| {
            format!(
                "cannot resolve parent directory '{}': {e}",
                parent.display()
            )
        })?;

        if !resolved_parent.starts_with(root) {
            return Err(format!("path escapes sandbox: {path}"));
        }

        let filename = candidate
            .file_name()
            .ok_or_else(|| format!("path has no filename: {path}"))?;

        let final_path = resolved_parent.join(filename);

        // If the filename itself is a pre-existing symlink, verify it doesn't
        // escape the sandbox. read_link detects symlinks even if the target
        // doesn't exist (unlike exists() which follows the link).
        if std::fs::read_link(&final_path).is_ok() {
            match std::fs::canonicalize(&final_path) {
                Ok(resolved) if resolved.starts_with(root) => {}
                Ok(_) => return Err(format!("path escapes sandbox: {path}")),
                // Broken symlink — target doesn't exist, can't verify boundary.
                Err(_) => return Err(format!("path is a dangling symlink: {path}")),
            }
        }

        Ok(final_path)
    }

    /// Attach the protected home directory (`~/.omega-system`) whose credential
    /// files the ungated filesystem tools must refuse. `dir` is canonicalized
    /// here; `None`, a nonexistent directory, or one that cannot be
    /// canonicalized leaves the sandbox with no protection — which is safe,
    /// because [`Sandbox::resolve`] needs an existing file to resolve and
    /// [`Sandbox::resolve_for_write`] needs an existing parent, so a home
    /// directory that is not there yet has nothing to expose.
    pub fn with_protected_home(mut self, dir: Option<&Path>) -> Self {
        self.protected_home = dir.and_then(|d| std::fs::canonicalize(d).ok());
        self
    }

    /// Whether `resolved` (a canonical path from [`Sandbox::resolve`]) is the
    /// shielded global `.env`. The read shield covers only the credentials
    /// file — the global `config.json` carries no secrets.
    pub fn is_protected_read(&self, resolved: &Path) -> bool {
        self.protected_matches(resolved, &[".env"])
    }

    /// Whether `resolved` is a shielded global file for the automated-write
    /// floor: credentials, configuration, and the personal instruction and
    /// repository-consent files that seed the next session's trusted inputs.
    pub fn is_protected_write(&self, resolved: &Path) -> bool {
        self.protected_matches(
            resolved,
            &[
                ".env",
                "config.json",
                "AGENTS.md",
                "trusted.json",
                "trusted.lock",
            ],
        )
    }

    /// The shared key: `resolved`'s parent is the canonical protected home and
    /// its file name is one of `names`. Keying on `(parent dir, file name)`
    /// keeps the check canonical — symlink aliases already collapsed by
    /// `resolve` — and inert whenever no protected home is configured.
    fn protected_matches(&self, resolved: &Path, names: &[&str]) -> bool {
        let Some(home) = &self.protected_home else {
            return false;
        };
        if resolved.parent() != Some(home.as_path()) {
            return false;
        }
        resolved
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|name| names.contains(&name))
    }
}

/// For unbounded mode: make relative paths absolute via CWD.
fn abs_or_cwd(path: &str) -> Result<PathBuf, String> {
    let p = Path::new(path);
    if p.is_absolute() {
        Ok(PathBuf::from(path))
    } else {
        let cwd = std::env::current_dir().map_err(cwd_error)?;
        Ok(cwd.join(path))
    }
}

/// Error text when the process CWD cannot be read — only possible when the
/// CWD vanishes mid-process, which no hermetic test can stage. A named fn
/// (passed as a fn pointer) creates no closure for coverage to miss; the body
/// is covered by its own unit test. Shared with `shell_guardrails`, which has
/// the same un-drivable fallback.
pub(super) fn cwd_error(e: std::io::Error) -> String {
    format!("cannot determine working directory: {e}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // ── root accessor ──

    #[test]
    fn root_returns_canonical_root() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        assert_eq!(sb.root().unwrap(), fs::canonicalize(dir.path()).unwrap());
    }

    #[test]
    fn root_is_none_when_unbounded() {
        assert!(Sandbox::unbounded().root().is_none());
    }

    // ── rooted sandbox ──

    #[test]
    fn rooted_nonexistent_root_errors() {
        let err = Sandbox::rooted(PathBuf::from("/nonexistent/omega/root")).unwrap_err();
        assert!(err.contains("cannot canonicalize sandbox root"));
    }

    #[test]
    fn resolve_relative_path_within_root() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("hello.txt");
        fs::write(&file, "hi").unwrap();

        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let resolved = sb.resolve("hello.txt").unwrap();
        assert_eq!(resolved, fs::canonicalize(&file).unwrap());
    }

    #[test]
    fn resolve_absolute_path_within_root() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("data.txt");
        fs::write(&file, "ok").unwrap();

        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let resolved = sb.resolve(file.to_str().unwrap()).unwrap();
        assert_eq!(resolved, fs::canonicalize(&file).unwrap());
    }

    #[test]
    fn resolve_rejects_dot_dot_escape() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let err = sb.resolve("../../../etc/passwd").unwrap_err();
        assert!(err.contains("cannot resolve") || err.contains("escapes sandbox"));
    }

    #[test]
    fn resolve_rejects_absolute_outside_root() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let err = sb.resolve("/etc/hosts").unwrap_err();
        assert!(err.contains("escapes sandbox"));
    }

    #[test]
    fn resolve_rejects_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();
        let link = dir.path().join("sneaky");

        #[cfg(unix)]
        std::os::unix::fs::symlink("/etc", &link).unwrap();
        #[cfg(not(unix))]
        return; // symlink tests only run on Unix

        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let err = sb.resolve("sneaky/hosts").unwrap_err();
        assert!(err.contains("escapes sandbox"));
    }

    #[test]
    fn resolve_allows_symlink_within_root() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.txt");
        fs::write(&real, "content").unwrap();
        let link = dir.path().join("link.txt");

        #[cfg(unix)]
        std::os::unix::fs::symlink(&real, &link).unwrap();
        #[cfg(not(unix))]
        return;

        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let resolved = sb.resolve("link.txt").unwrap();
        assert_eq!(resolved, fs::canonicalize(&real).unwrap());
    }

    #[test]
    fn resolve_nonexistent_path_errors() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let err = sb.resolve("no_such_file.txt").unwrap_err();
        assert!(err.contains("cannot resolve"));
    }

    // ── resolve_for_write ──

    #[test]
    fn resolve_for_write_new_file_in_root() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let resolved = sb.resolve_for_write("new_file.txt").unwrap();
        assert!(resolved.starts_with(fs::canonicalize(dir.path()).unwrap()));
        assert_eq!(resolved.file_name().unwrap(), "new_file.txt");
    }

    #[test]
    fn resolve_for_write_rejects_escape() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let err = sb.resolve_for_write("/tmp/evil.txt").unwrap_err();
        assert!(err.contains("escapes sandbox"));
    }

    #[test]
    fn resolve_for_write_in_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();

        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let resolved = sb.resolve_for_write("sub/file.txt").unwrap();
        assert!(resolved.starts_with(fs::canonicalize(dir.path()).unwrap()));
        assert_eq!(resolved.file_name().unwrap(), "file.txt");
    }

    #[test]
    fn resolve_for_write_rejects_filename_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();

        #[cfg(unix)]
        {
            // Symlink to an existing path outside the sandbox.
            let link = dir.path().join("escape");
            std::os::unix::fs::symlink("/tmp", &link).unwrap();

            let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
            let err = sb.resolve_for_write("escape").unwrap_err();
            assert!(err.contains("escapes sandbox"));
        }
    }

    #[test]
    fn resolve_for_write_rejects_dangling_symlink() {
        let dir = tempfile::tempdir().unwrap();

        #[cfg(unix)]
        {
            // Broken symlink — target doesn't exist.
            let link = dir.path().join("dangling");
            std::os::unix::fs::symlink("/nonexistent/path", &link).unwrap();

            let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
            let err = sb.resolve_for_write("dangling").unwrap_err();
            assert!(err.contains("dangling symlink"));
        }
    }

    #[test]
    fn resolve_for_write_rejects_nonexistent_parent() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let err = sb.resolve_for_write("missing_dir/file.txt").unwrap_err();
        assert!(err.contains("cannot resolve parent directory"));
    }

    #[test]
    fn resolve_for_write_rejects_path_without_parent() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let err = sb.resolve_for_write("/").unwrap_err();
        assert!(err.contains("path has no parent"));
    }

    #[test]
    fn resolve_for_write_rejects_path_without_filename() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();

        // `sub/..` has a real parent but no final component to write to.
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let err = sb.resolve_for_write("sub/..").unwrap_err();
        assert!(err.contains("path has no filename"));
    }

    // ── unbounded sandbox ──

    #[test]
    fn unbounded_allows_any_absolute_path() {
        let sb = Sandbox::unbounded();
        let resolved = sb.resolve_for_write("/tmp/anything.txt").unwrap();
        assert_eq!(resolved, PathBuf::from("/tmp/anything.txt"));
    }

    #[test]
    fn unbounded_makes_relative_paths_absolute() {
        let sb = Sandbox::unbounded();
        let resolved = sb.resolve_for_write("some/file.txt").unwrap();
        assert!(resolved.is_absolute());
    }

    #[test]
    fn unbounded_resolve_absolute_path() {
        let sb = Sandbox::unbounded();
        let resolved = sb.resolve("/tmp/anything.txt").unwrap();
        assert_eq!(resolved, PathBuf::from("/tmp/anything.txt"));
    }

    #[test]
    fn unbounded_resolve_makes_relative_paths_absolute() {
        let sb = Sandbox::unbounded();
        let resolved = sb.resolve("some/file.txt").unwrap();
        assert!(resolved.is_absolute());
    }

    // ── protected home (global-credential shielding) ──

    /// A sandbox rooted at `dir` with `dir/.omega-system` (created here)
    /// attached as the protected home, plus the canonical home path.
    fn sandbox_with_home(dir: &std::path::Path) -> (Sandbox, PathBuf) {
        let home = dir.join(".omega-system");
        fs::create_dir(&home).unwrap();
        let sb = Sandbox::rooted(dir.to_path_buf())
            .unwrap()
            .with_protected_home(Some(&home));
        (sb, fs::canonicalize(&home).unwrap())
    }

    #[test]
    fn protected_read_and_write_hit_the_global_env() {
        let dir = tempfile::tempdir().unwrap();
        let (sb, home) = sandbox_with_home(dir.path());
        let env = home.join(".env");
        // `.env` is shielded from both reads and the write floor.
        assert!(sb.is_protected_read(&env));
        assert!(sb.is_protected_write(&env));
    }

    #[test]
    fn protected_write_covers_config_but_read_does_not() {
        let dir = tempfile::tempdir().unwrap();
        let (sb, home) = sandbox_with_home(dir.path());
        let config = home.join("config.json");
        // The global config.json holds no secrets — write-floored, not read-shielded.
        assert!(!sb.is_protected_read(&config));
        assert!(sb.is_protected_write(&config));
    }

    #[test]
    fn personal_instructions_and_repository_consent_require_the_write_floor() {
        let dir = tempfile::tempdir().unwrap();
        let (sb, home) = sandbox_with_home(dir.path());
        for name in ["AGENTS.md", "trusted.json", "trusted.lock"] {
            assert!(sb.is_protected_write(&home.join(name)));
            assert!(!sb.is_protected_read(&home.join(name)));
            assert!(!sb.is_protected_write(&dir.path().join(name)));
        }
    }

    #[test]
    fn protected_ignores_lookalike_directories_and_unrelated_files() {
        let dir = tempfile::tempdir().unwrap();
        let (sb, home) = sandbox_with_home(dir.path());
        // A sibling directory whose name merely starts the same is not the home.
        let evil = home.parent().unwrap().join(".omega-system.evil");
        assert!(!sb.is_protected_read(&evil.join(".env")));
        assert!(!sb.is_protected_write(&evil.join(".env")));
        // An unrelated `.env` in another directory (here the sandbox root).
        assert!(!sb.is_protected_read(&dir.path().join(".env")));
    }

    #[test]
    fn protected_home_nonexistent_is_inert() {
        let dir = tempfile::tempdir().unwrap();
        // A home that does not exist cannot be canonicalized → no protection.
        let sb = Sandbox::rooted(dir.path().to_path_buf())
            .unwrap()
            .with_protected_home(Some(&dir.path().join(".omega-system")));
        assert!(!sb.is_protected_read(&dir.path().join(".omega-system/.env")));
    }

    #[test]
    #[cfg(unix)]
    fn protected_home_symlink_alias_collapses_to_a_hit() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join(".omega-system");
        fs::create_dir(&home).unwrap();
        // Attach the home through a symlinked alias; canonicalization at
        // construction resolves it to the real directory, so a canonical path
        // under the real home still matches.
        let alias = dir.path().join("home-link");
        std::os::unix::fs::symlink(&home, &alias).unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf())
            .unwrap()
            .with_protected_home(Some(&alias));
        let env = fs::canonicalize(&home).unwrap().join(".env");
        assert!(sb.is_protected_read(&env));
    }

    #[test]
    fn no_protection_configured_is_inert() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        assert!(!sb.is_protected_read(&dir.path().join(".env")));
        assert!(!sb.is_protected_write(&dir.path().join("config.json")));
        // An unbounded sandbox is equally inert.
        assert!(!Sandbox::unbounded().is_protected_read(Path::new("/anything/.env")));
    }

    // ── error text helpers ──

    #[test]
    fn cwd_error_formats_reason() {
        let msg = cwd_error(std::io::Error::other("gone"));
        assert_eq!(msg, "cannot determine working directory: gone");
    }
}
