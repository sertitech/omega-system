pub mod agent;
pub mod anthropic;
pub(crate) mod atomic_write;
pub mod cli;
pub mod command;
pub mod config;
pub(crate) mod display;
pub mod markdown;
pub mod openai;
pub mod provider;
pub mod repl;
pub mod session;
pub mod session_store;
pub mod sse;
pub mod tools;
pub mod turn;

/// Shared test doubles (mock providers). `#[cfg(test)]` here keeps them out
/// of every non-test build; the module itself explains the layout.
#[cfg(test)]
pub(crate) mod testing;

/// The model id every test uses — one constant so the value can't drift across
/// the call sites that build requests or test configs. It points at a real,
/// current model so the `#[ignore]`'d live-API tests hit a valid endpoint. A
/// private item at the crate root is visible to every descendant module's test
/// code via `crate::TEST_MODEL`.
#[cfg(test)]
pub(crate) const TEST_MODEL: &str = "claude-sonnet-5";

/// The model id the OpenAI adapter's `#[ignore]`'d live tests hit. The shared
/// [`TEST_MODEL`] points at an Anthropic model, so the OpenAI live path needs
/// its own current, valid id.
/// Hermetic OpenAI tests reuse [`TEST_MODEL`] as an arbitrary string — its value
/// never reaches the network, so only the live path cares which provider owns it.
#[cfg(test)]
pub(crate) const TEST_MODEL_OPENAI: &str = "gpt-4o-mini";

/// A Responses-API-only id: Chat Completions rejected it with "Use the
/// v1/responses endpoint instead" (verified 2026-07-06). Pins the reach the
/// Responses adapter exists for — its live tests must drive an id no other
/// endpoint can serve. Update when the catalog moves on, like its siblings.
#[cfg(test)]
pub(crate) const TEST_MODEL_OPENAI_RESPONSES_ONLY: &str = "gpt-5.3-codex";

/// The user's home directory, read from `$HOME`. `None` when the variable is
/// unset or empty — the caller degrades to project-only behavior rather than
/// guessing a path. The global settings layer (`~/.omega-system/`) hangs off
/// this.
pub fn home_dir() -> Option<std::path::PathBuf> {
    home_dir_from(|name| std::env::var(name).ok())
}

/// [`home_dir`] with the environment lookup injected, so tests drive the
/// set/missing/empty cases without mutating the process environment — the test
/// binary runs them in parallel, and `set_var`/`remove_var` are process-wide.
fn home_dir_from(lookup: impl Fn(&str) -> Option<String>) -> Option<std::path::PathBuf> {
    match lookup("HOME") {
        Some(home) if !home.is_empty() => Some(std::path::PathBuf::from(home)),
        _ => None,
    }
}

/// Load an environment variable by name, best-effort: an unreadable `.env`
/// collapses to `None` (key not found). Resolution order: the process
/// environment, then `./.env`, then `~/.omega-system/.env`. The lazy seams
/// (mid-session `/model` switch, subagent spawn) call this so a broken `.env`
/// degrades a switch rather than aborting it; the eager startup credential uses
/// [`load_env_var_checked`] to fail fast instead.
pub fn load_env_var(name: &str) -> Option<String> {
    load_env_var_checked(name).ok().flatten()
}

/// Like [`load_env_var`] but fail-fast: an existing-but-unreadable `.env`
/// surfaces as `Err` instead of collapsing to "absent". Resolution order: the
/// process environment, then `./.env`, then `~/.omega-system/.env` — a key set
/// in the project's local file shadows the global one, and a key only in the
/// global file still resolves. A genuinely missing file is skipped (`Ok(None)`);
/// only a real read error is fatal.
pub fn load_env_var_checked(name: &str) -> Result<Option<String>, String> {
    let mut dotenvs = vec![std::path::PathBuf::from(".env")];
    if let Some(home) = home_dir() {
        dotenvs.push(home.join(".omega-system/.env"));
    }
    load_env_var_from(name, &dotenvs)
}

/// [`load_env_var_checked`] with the fallback files made explicit and ordered,
/// so tests can drive the `.env` parse deterministically instead of depending on
/// whatever `.env` happens to sit in the developer's working directory. Files are
/// tried in order; the first hit wins, a missing file falls through to the next,
/// and an unreadable file is fatal (propagated).
fn load_env_var_from(name: &str, dotenvs: &[std::path::PathBuf]) -> Result<Option<String>, String> {
    if let Ok(val) = std::env::var(name) {
        return Ok(Some(val));
    }
    for dotenv in dotenvs {
        if let Some(val) = dotenv_lookup(dotenv, name)? {
            return Ok(Some(val));
        }
    }
    Ok(None)
}

/// Parse a single `.env` file for `name`. A missing file returns `Ok(None)` (the
/// caller falls through to the next candidate); a present file is scanned line by
/// line, skipping blanks and `#` comments, trimming both key and value. Any read
/// error other than "not found" (permissions, non-UTF-8) is fatal (`Err`) rather
/// than collapsed into "absent" — an existing-but-unreadable `.env` is a real
/// problem the operator must see, not a missing key, matching the sibling loaders
/// `config::read_candidate` and `History::load`.
fn dotenv_lookup(dotenv: &std::path::Path, name: &str) -> Result<Option<String>, String> {
    let contents = match std::fs::read_to_string(dotenv) {
        Ok(contents) => contents,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("cannot read env file {}: {e}", dotenv.display())),
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=')
            && key.trim() == name
        {
            return Ok(Some(value.trim().to_string()));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_env_var_from_environment() {
        // SAFETY: test-only; no other test reads this variable.
        unsafe { std::env::set_var("_OMEGA_TEST_VAR", "hello") };
        assert_eq!(load_env_var("_OMEGA_TEST_VAR"), Some("hello".to_string()));
        unsafe { std::env::remove_var("_OMEGA_TEST_VAR") };
    }

    #[test]
    fn load_env_var_missing_returns_none() {
        // SAFETY: test-only; no other test reads this variable.
        unsafe { std::env::remove_var("_OMEGA_NONEXISTENT_VAR") };
        let result = load_env_var("_OMEGA_NONEXISTENT_VAR");
        assert!(result.is_none());
    }

    // ── .env file fallback ──
    //
    // Each test uses a `_OMEGA_DOTENV_*` name that never exists in the process
    // environment, so the lookup always falls through to the file(s).

    #[test]
    fn dotenv_key_hit_trims_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let dotenv = dir.path().join(".env");
        std::fs::write(
            &dotenv,
            "# a comment\n\nOTHER=nope\n  _OMEGA_DOTENV_HIT = padded value \n",
        )
        .unwrap();
        assert_eq!(
            load_env_var_from("_OMEGA_DOTENV_HIT", &[dotenv]),
            Ok(Some("padded value".to_string()))
        );
    }

    #[test]
    fn dotenv_key_miss_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let dotenv = dir.path().join(".env");
        std::fs::write(&dotenv, "# only comments\nOTHER=nope\n").unwrap();
        assert_eq!(load_env_var_from("_OMEGA_DOTENV_MISS", &[dotenv]), Ok(None));
    }

    #[test]
    fn dotenv_missing_file_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let dotenv = dir.path().join("no_such.env");
        assert_eq!(
            load_env_var_from("_OMEGA_DOTENV_NOFILE", &[dotenv]),
            Ok(None)
        );
    }

    #[test]
    fn dotenv_local_shadows_global() {
        // A key present in both files resolves from the earlier (local) one.
        let dir = tempfile::tempdir().unwrap();
        let local = dir.path().join(".env");
        let global = dir.path().join("global.env");
        std::fs::write(&local, "_OMEGA_DOTENV_SHADOW=local\n").unwrap();
        std::fs::write(&global, "_OMEGA_DOTENV_SHADOW=global\n").unwrap();
        assert_eq!(
            load_env_var_from("_OMEGA_DOTENV_SHADOW", &[local, global]),
            Ok(Some("local".to_string()))
        );
    }

    #[test]
    fn dotenv_falls_through_to_global() {
        // A key absent from the local file resolves from the global one.
        let dir = tempfile::tempdir().unwrap();
        let local = dir.path().join(".env");
        let global = dir.path().join("global.env");
        std::fs::write(&local, "OTHER=nope\n").unwrap();
        std::fs::write(&global, "_OMEGA_DOTENV_FALL=fromglobal\n").unwrap();
        assert_eq!(
            load_env_var_from("_OMEGA_DOTENV_FALL", &[local, global]),
            Ok(Some("fromglobal".to_string()))
        );
    }

    #[test]
    fn dotenv_skips_missing_files_in_the_list() {
        // A nonexistent earlier file does not abort the search.
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.env");
        let global = dir.path().join("global.env");
        std::fs::write(&global, "_OMEGA_DOTENV_SKIP=found\n").unwrap();
        assert_eq!(
            load_env_var_from("_OMEGA_DOTENV_SKIP", &[missing, global]),
            Ok(Some("found".to_string()))
        );
    }

    #[test]
    fn load_env_var_checked_reads_environment() {
        // SAFETY: test-only; no other test reads this variable.
        unsafe { std::env::set_var("_OMEGA_CHECKED_VAR", "hi") };
        assert_eq!(
            load_env_var_checked("_OMEGA_CHECKED_VAR"),
            Ok(Some("hi".to_string()))
        );
        unsafe { std::env::remove_var("_OMEGA_CHECKED_VAR") };
    }

    #[test]
    fn dotenv_non_utf8_is_fatal() {
        // An existing `.env` that isn't valid UTF-8 is a real problem, surfaced
        // as Err — not collapsed to "key not found". Driven through
        // load_env_var_from so the `?` propagation is exercised too.
        let dir = tempfile::tempdir().unwrap();
        let dotenv = dir.path().join(".env");
        std::fs::write(&dotenv, [0xff, 0xfe, 0x00]).unwrap();
        let err = load_env_var_from("_OMEGA_DOTENV_NONUTF8", &[dotenv]).unwrap_err();
        assert!(err.contains("cannot read env file"), "got: {err}");
    }

    #[cfg(unix)]
    #[test]
    fn dotenv_unreadable_file_is_fatal() {
        use std::os::unix::fs::PermissionsExt;
        // A present-but-unreadable `.env` (mode 0) is fatal, mirroring the config
        // sibling. Restore the mode before asserting so the tempdir cleans up.
        let dir = tempfile::tempdir().unwrap();
        let dotenv = dir.path().join(".env");
        std::fs::write(&dotenv, "_OMEGA_DOTENV_LOCKED=x\n").unwrap();
        std::fs::set_permissions(&dotenv, std::fs::Permissions::from_mode(0o000)).unwrap();

        let result = dotenv_lookup(&dotenv, "_OMEGA_DOTENV_LOCKED");

        std::fs::set_permissions(&dotenv, std::fs::Permissions::from_mode(0o644)).unwrap();
        let err = result.unwrap_err();
        assert!(err.contains("cannot read env file"), "got: {err}");
    }

    // ── home_dir ──

    #[test]
    fn home_dir_from_reads_the_lookup() {
        assert_eq!(
            home_dir_from(|_| Some("/home/omega".to_string())),
            Some(std::path::PathBuf::from("/home/omega"))
        );
    }

    #[test]
    fn home_dir_from_missing_is_none() {
        assert_eq!(home_dir_from(|_| None), None);
    }

    #[test]
    fn home_dir_from_empty_is_none() {
        // An empty `$HOME` is as good as unset — never join paths onto "".
        assert_eq!(home_dir_from(|_| Some(String::new())), None);
    }

    #[test]
    fn home_dir_mirrors_process_home() {
        // Read-only: both sides read the same `$HOME`, so this is deterministic
        // and parallel-safe without touching the environment.
        let expected = std::env::var("HOME")
            .ok()
            .filter(|h| !h.is_empty())
            .map(std::path::PathBuf::from);
        assert_eq!(home_dir(), expected);
    }
}
