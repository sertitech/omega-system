//! Omega's argv surface — currently a single flag, `--resume`, plus the
//! startup session-selection logic behind it.
//!
//! Both pieces live here, out of `main.rs`, on purpose: `main.rs` is excluded
//! from the coverage gate as thin process wiring, so the behavioral branch that
//! chooses which session to restore must sit in a covered module. `main.rs`
//! only collects argv, calls [`CliArgs::parse`] and
//! [`select_initial_session`], and applies the result.

use crate::session::Session;
use crate::session_store::{SessionEntry, SessionStore};
use std::io::{BufRead, Write};
use std::time::SystemTime;

/// The one-line usage string printed to stderr on any parse error, then pinned
/// by tests. `[name]` is optional; the outer `[--resume …]` is too — a bare
/// launch takes no arguments.
pub const USAGE: &str = "usage: omega [--resume [name]]";

/// Which session `--resume` selects, if the flag is present at all.
#[derive(Debug, Clone, PartialEq)]
pub enum Resume {
    /// Bare `--resume`: the newest save by mtime.
    Last,
    /// `--resume <name>`: the save stored under `<name>`.
    Named(String),
}

/// Parsed command-line arguments. `resume: None` is the ordinary no-flag
/// launch, which falls back to the configured `session_file` behavior.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CliArgs {
    pub resume: Option<Resume>,
}

impl CliArgs {
    /// Parse the argument list, which **excludes** `argv[0]` — `main` passes
    /// `std::env::args().skip(1)`. Understood exactly: nothing (`resume:
    /// None`), `--resume` (newest save), and `--resume <name>` (named save).
    /// Anything else — an unknown flag, trailing junk after the name, or a
    /// second `--resume` (which arrives as a flag-shaped "name") — is an `Err`
    /// carrying [`USAGE`].
    pub fn parse(args: &[String]) -> Result<CliArgs, String> {
        let mut iter = args.iter();
        match iter.next() {
            None => Ok(CliArgs { resume: None }),
            Some(flag) if flag == "--resume" => {
                let resume = match iter.next() {
                    None => Resume::Last,
                    // A flag-shaped token (a second `--resume`, any `-x`) is
                    // never a session name — reject rather than resume a save
                    // literally named after a flag.
                    Some(name) if name.starts_with('-') => return Err(USAGE.to_string()),
                    Some(name) => Resume::Named(name.clone()),
                };
                // Nothing may follow the optional name.
                if iter.next().is_some() {
                    return Err(USAGE.to_string());
                }
                Ok(CliArgs {
                    resume: Some(resume),
                })
            }
            Some(_) => Err(USAGE.to_string()),
        }
    }
}

/// Choose the session to restore at startup. `Ok(None)` means start fresh.
///
/// - `--resume <name>` → that save from `store`; a name that was never saved is
///   a startup error (the store's name-aware "no saved session named …").
/// - bare `--resume` → depends on how many saves exist ([`SessionStore::list`]
///   is newest-first): an empty store is a startup error, not a silent fresh
///   session; exactly one save loads directly with no prompt; several saves run
///   the [`pick_session`] picker when `interactive`, and otherwise (piped
///   stdin) fall back to the newest by mtime, consuming no picker input.
/// - either `--resume` form with no `store` (no `$HOME`) is a startup error —
///   there is nowhere saves could live.
/// - no `--resume` → the configured `session_file` behavior, unchanged: an
///   existing path loads fail-fast (a corrupt file is a real problem the
///   operator must see), a configured-but-absent path or no `session_file` at
///   all starts fresh.
///
/// `interactive`, `picker_in`, and `picker_out` drive only the several-saves
/// bare-`--resume` branch; every other arm ignores them and behaves exactly as
/// before. Precedence: `--resume` is explicit operator intent and wins over
/// `session_file` — the configured file is not read when the flag is present.
pub fn select_initial_session(
    args: &CliArgs,
    store: Option<&SessionStore>,
    session_file: Option<&str>,
    interactive: bool,
    picker_in: &mut impl BufRead,
    picker_out: &mut impl Write,
) -> Result<Option<Session>, String> {
    let Some(resume) = &args.resume else {
        // No `--resume`: the pre-existing opt-in `session_file` restore.
        return match session_file {
            Some(path) if std::path::Path::new(path).exists() => Session::load(path).map(Some),
            _ => Ok(None),
        };
    };
    let store = store.ok_or_else(|| {
        "--resume: no home directory ($HOME unset), so there are no saved sessions".to_string()
    })?;
    match resume {
        Resume::Named(name) => store.load(name).map(Some),
        Resume::Last => {
            let saves = store.list();
            match saves.as_slice() {
                [] => Err("--resume: no saved sessions to resume".to_string()),
                // A single save has nothing to disambiguate — load it directly.
                [only] => store.load(&only.name).map(Some),
                // Several saves: prompt when interactive, else take the newest
                // (index 0 of the newest-first list) with no input consumed.
                _ => {
                    let idx = if interactive {
                        pick_session(&saves, SystemTime::now(), picker_in, picker_out)
                    } else {
                        0
                    };
                    store.load(&saves[idx].name).map(Some)
                }
            }
        }
    }
}

/// Prompt for which saved session to resume, over an injected reader and writer
/// so it is fully hermetic — no `_live.rs` tail. Writes a newest-first numbered
/// list and a `Resume which? [1]: ` prompt, then reads one cooked line and
/// returns the **0-based** index of the chosen save:
///
/// - a valid `1..=N` selects that save;
/// - an empty line or EOF collapses to `0` — the newest, exactly what a plain
///   `--resume` would have picked, so every non-selection has the same effect;
/// - a non-numeric or out-of-range entry re-prints a one-line `(enter 1–N)`
///   hint and reads again.
///
/// It runs **before** `run_repl` (cooked mode, nothing typed ahead) and is
/// dropped before the loop's first read, so a transient `StdinLock` reader is
/// safe — the same sequencing the `LineSource` seam records.
pub fn pick_session(
    saves: &[SessionEntry],
    now: SystemTime,
    input: &mut impl BufRead,
    out: &mut impl Write,
) -> usize {
    for (i, entry) in saves.iter().enumerate() {
        let _ = writeln!(out, "  {}. {}", i + 1, entry.describe(now));
    }
    loop {
        let _ = write!(out, "Resume which? [1]: ");
        let _ = out.flush();
        let mut line = String::new();
        match input.read_line(&mut line) {
            // EOF or a read error: fall back to the newest, like an empty line.
            Ok(0) | Err(_) => return 0,
            Ok(_) => {}
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return 0;
        }
        match trimmed.parse::<usize>() {
            Ok(n) if (1..=saves.len()).contains(&n) => return n - 1,
            _ => {
                let _ = writeln!(out, "(enter 1–{})", saves.len());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SESSION_VERSION;
    use crate::turn::{Block, Role, TurnMessage};
    use std::fs::File;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, SystemTime};

    // ── CliArgs::parse table ──

    /// Parse from string literals, sparing every call site the `.to_string()`.
    fn parse(args: &[&str]) -> Result<CliArgs, String> {
        CliArgs::parse(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn parse_no_args_is_no_resume() {
        assert_eq!(parse(&[]).unwrap(), CliArgs { resume: None });
    }

    #[test]
    fn parse_bare_resume_is_last() {
        assert_eq!(
            parse(&["--resume"]).unwrap(),
            CliArgs {
                resume: Some(Resume::Last)
            }
        );
    }

    #[test]
    fn parse_resume_name_is_named() {
        assert_eq!(
            parse(&["--resume", "before-refactor"]).unwrap(),
            CliArgs {
                resume: Some(Resume::Named("before-refactor".to_string()))
            }
        );
    }

    #[test]
    fn parse_unknown_flag_is_usage_error() {
        assert_eq!(parse(&["--nope"]).unwrap_err(), USAGE);
    }

    #[test]
    fn parse_trailing_junk_after_name_is_usage_error() {
        assert_eq!(parse(&["--resume", "mine", "extra"]).unwrap_err(), USAGE);
    }

    #[test]
    fn parse_second_resume_is_usage_error() {
        // A second `--resume` arrives in the name position but is flag-shaped,
        // so it is rejected rather than taken as a save literally named
        // `--resume`.
        assert_eq!(parse(&["--resume", "--resume"]).unwrap_err(), USAGE);
    }

    #[test]
    fn usage_string_is_pinned() {
        assert_eq!(USAGE, "usage: omega [--resume [name]]");
    }

    // ── select_initial_session matrix ──

    /// A session tagged by its `last_input_tokens`, so a test can tell which
    /// save was restored.
    fn session(tag: u32) -> Session {
        Session {
            version: SESSION_VERSION,
            messages: vec![TurnMessage {
                role: Role::User,
                content: vec![Block::Text("hi".to_string())],
            }],
            compacted_summary: None,
            last_input_tokens: tag,
        }
    }

    /// A store under fresh home/project tempdirs; the tempdirs are returned so
    /// they outlive the store.
    fn fresh_store() -> (tempfile::TempDir, tempfile::TempDir, SessionStore) {
        let home = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        let store = SessionStore::new(home.path(), project.path());
        (home, project, store)
    }

    /// Locate a save file by name within the store's tree, without depending on
    /// the private project-key encoding — the sessions root holds exactly one
    /// project-key subdirectory, which is where every save under this home
    /// lives.
    fn save_file(home: &Path, name: &str) -> PathBuf {
        let sessions = home.join(".omega-system/sessions");
        let proj = std::fs::read_dir(&sessions)
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        proj.path().join(format!("{name}.json"))
    }

    fn set_mtime(path: &Path, secs: u64) {
        let when = SystemTime::UNIX_EPOCH + Duration::from_secs(secs);
        File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(when))
            .unwrap();
    }

    fn resume(r: Resume) -> CliArgs {
        CliArgs { resume: Some(r) }
    }

    /// Drive `select_initial_session` for the non-picker arms: not interactive,
    /// with picker I/O that must never be written to (asserted) and, since these
    /// arms never read, no input either.
    fn select(
        args: &CliArgs,
        store: Option<&SessionStore>,
        session_file: Option<&str>,
    ) -> Result<Option<Session>, String> {
        let mut input = std::io::Cursor::new(Vec::new());
        let mut out = Vec::new();
        let got = select_initial_session(args, store, session_file, false, &mut input, &mut out);
        assert!(
            out.is_empty(),
            "a non-picker arm wrote to the picker output"
        );
        got
    }

    #[test]
    fn named_hit_loads_the_save() {
        let (_home, _project, store) = fresh_store();
        store.save("mine", &session(7)).unwrap();
        let got = select(
            &resume(Resume::Named("mine".to_string())),
            Some(&store),
            None,
        )
        .unwrap();
        assert_eq!(got, Some(session(7)));
    }

    #[test]
    fn named_miss_is_a_startup_error() {
        let (_home, _project, store) = fresh_store();
        let err = select(
            &resume(Resume::Named("ghost".to_string())),
            Some(&store),
            None,
        )
        .unwrap_err();
        assert!(err.contains("no saved session named 'ghost'"), "got: {err}");
    }

    #[test]
    fn one_save_loads_directly_without_a_prompt() {
        // Exactly one save: no ambiguity, so it loads with no picker output
        // even when interactive.
        let (_home, _project, store) = fresh_store();
        store.save("only", &session(4)).unwrap();
        let mut input = std::io::Cursor::new(b"1\n".to_vec());
        let mut out = Vec::new();
        let got = select_initial_session(
            &resume(Resume::Last),
            Some(&store),
            None,
            true,
            &mut input,
            &mut out,
        )
        .unwrap();
        assert_eq!(got, Some(session(4)));
        assert_eq!(input.position(), 0, "no picker line consumed");
        assert!(out.is_empty(), "no prompt written for a single save");
    }

    #[test]
    fn several_saves_interactive_runs_the_picker() {
        let (home, _project, store) = fresh_store();
        store.save("a", &session(1)).unwrap();
        store.save("b", &session(2)).unwrap();
        store.save("c", &session(3)).unwrap();
        // Newest-first order is a, b, c; typing `2` picks the second row, `b`.
        set_mtime(&save_file(home.path(), "a"), 300);
        set_mtime(&save_file(home.path(), "b"), 200);
        set_mtime(&save_file(home.path(), "c"), 100);
        let mut input = std::io::Cursor::new(b"2\n".to_vec());
        let mut out = Vec::new();
        let got = select_initial_session(
            &resume(Resume::Last),
            Some(&store),
            None,
            true,
            &mut input,
            &mut out,
        )
        .unwrap();
        assert_eq!(got, Some(session(2)));
        assert!(String::from_utf8(out).unwrap().contains("Resume which?"));
    }

    #[test]
    fn several_saves_non_interactive_loads_newest_reading_nothing() {
        // Piped stdin with several saves: today's newest-by-mtime, no prompt,
        // and no scripted line consumed.
        let (home, _project, store) = fresh_store();
        store.save("old", &session(1)).unwrap();
        store.save("new", &session(2)).unwrap();
        set_mtime(&save_file(home.path(), "old"), 100);
        set_mtime(&save_file(home.path(), "new"), 200);
        let mut input = std::io::Cursor::new(b"1\n".to_vec());
        let mut out = Vec::new();
        let got = select_initial_session(
            &resume(Resume::Last),
            Some(&store),
            None,
            false,
            &mut input,
            &mut out,
        )
        .unwrap();
        assert_eq!(got, Some(session(2)));
        assert_eq!(input.position(), 0, "no picker line consumed");
        assert!(out.is_empty(), "no prompt written when non-interactive");
    }

    #[test]
    fn bare_resume_on_empty_store_is_a_startup_error() {
        let (_home, _project, store) = fresh_store();
        let err = select(&resume(Resume::Last), Some(&store), None).unwrap_err();
        assert!(err.contains("no saved sessions to resume"), "got: {err}");
    }

    #[test]
    fn resume_without_home_is_a_startup_error() {
        // No store (no `$HOME`) under `--resume` cannot be a silent fresh
        // session — it is an error naming the missing home.
        let err = select(&resume(Resume::Last), None, None).unwrap_err();
        assert!(err.contains("no home directory"), "got: {err}");
    }

    #[test]
    fn no_resume_loads_an_existing_session_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");
        session(9).save(path.to_str().unwrap()).unwrap();
        let got = select(&CliArgs::default(), None, Some(path.to_str().unwrap())).unwrap();
        assert_eq!(got, Some(session(9)));
    }

    #[test]
    fn no_resume_with_absent_session_file_starts_fresh() {
        let got = select(
            &CliArgs::default(),
            None,
            Some("/no/such/omega-session.json"),
        )
        .unwrap();
        assert_eq!(got, None);
    }

    #[test]
    fn no_resume_with_corrupt_session_file_fails_fast() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");
        std::fs::write(&path, "{not json").unwrap();
        let err = select(&CliArgs::default(), None, Some(path.to_str().unwrap())).unwrap_err();
        assert!(err.contains("invalid"), "got: {err}");
    }

    #[test]
    fn no_resume_without_session_file_starts_fresh() {
        let got = select(&CliArgs::default(), None, None).unwrap();
        assert_eq!(got, None);
    }

    #[test]
    fn resume_wins_over_session_file_which_is_not_read() {
        // Both apply: `--resume mine` and a `session_file` that would fail
        // fast if read. The save loads and the corrupt file is never touched,
        // proving `--resume` takes precedence.
        let (_home, _project, store) = fresh_store();
        store.save("mine", &session(5)).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let corrupt = dir.path().join("session.json");
        std::fs::write(&corrupt, "{not json").unwrap();
        let got = select(
            &resume(Resume::Named("mine".to_string())),
            Some(&store),
            Some(corrupt.to_str().unwrap()),
        )
        .unwrap();
        assert_eq!(got, Some(session(5)));
    }

    // ── pick_session: the covered picker over injected bytes ──

    /// A `SessionEntry` for the picker tests; the timestamp and size are inert
    /// (`describe`'s formatting is covered in `session_store`).
    fn entry(name: &str) -> SessionEntry {
        SessionEntry {
            name: name.to_string(),
            modified: SystemTime::UNIX_EPOCH,
            size: 100,
        }
    }

    /// Run the picker over `script`, returning the chosen index and what it
    /// wrote.
    fn pick(saves: &[SessionEntry], script: &[u8]) -> (usize, String) {
        let mut input = std::io::Cursor::new(script.to_vec());
        let mut out = Vec::new();
        let idx = pick_session(saves, SystemTime::UNIX_EPOCH, &mut input, &mut out);
        (idx, String::from_utf8(out).unwrap())
    }

    #[test]
    fn pick_first_try_valid_selection() {
        let saves = [entry("a"), entry("b"), entry("c")];
        let (idx, shown) = pick(&saves, b"2\n");
        assert_eq!(idx, 1);
        // The list is numbered and the prompt is shown.
        assert!(shown.contains("  1. a"), "got: {shown}");
        assert!(shown.contains("  3. c"), "got: {shown}");
        assert!(shown.contains("Resume which? [1]: "), "got: {shown}");
    }

    #[test]
    fn pick_empty_line_is_the_newest() {
        let saves = [entry("a"), entry("b")];
        let (idx, _) = pick(&saves, b"\n");
        assert_eq!(idx, 0);
    }

    #[test]
    fn pick_eof_is_the_newest() {
        let saves = [entry("a"), entry("b")];
        let (idx, _) = pick(&saves, b"");
        assert_eq!(idx, 0);
    }

    #[test]
    fn pick_non_numeric_then_valid_reprompts() {
        let saves = [entry("a"), entry("b"), entry("c")];
        let (idx, shown) = pick(&saves, b"foo\n3\n");
        assert_eq!(idx, 2);
        assert!(shown.contains("(enter 1–3)"), "got: {shown}");
    }

    #[test]
    fn pick_out_of_range_bounds_then_valid_reprompts() {
        // Both `0` and `N+1` are out of range and re-prompt; the third line
        // finally selects.
        let saves = [entry("a"), entry("b"), entry("c")];
        let (idx, shown) = pick(&saves, b"0\n4\n2\n");
        assert_eq!(idx, 1);
        assert_eq!(shown.matches("(enter 1–3)").count(), 2, "got: {shown}");
    }
}
