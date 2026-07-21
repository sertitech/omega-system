//! The interactive-terminal [`LineSource`] — raw-mode `termios` wiring over
//! the covered editor core ([`super::editor`]). Real-terminal syscalls
//! cannot run under the hermetic suite (there is no TTY in CI, and toggling
//! the developer's terminal mid-`cargo test` would corrupt it), so the
//! `_live.rs` suffix marks this file for wholesale exclusion from the
//! coverage gate. Nothing beyond the syscall wiring and its delegation
//! lives here — the editing, completion, and rendering logic it drives is
//! all in covered modules.
//!
//! `libc` is the sanctioned stdlib-parity exception this phase adds:
//! Python's stdlib ships `termios`, Rust's does not, and raw mode needs
//! `tcgetattr`/`tcsetattr` — the same rationale that admitted `ureq` for
//! TLS.

use super::completion::Completion;
use super::history::History;
use super::{LineSource, editor};

/// True when both stdin and stdout are terminals — the only setup where the
/// raw-mode editor makes sense. Piped input (`echo hi | omega`) must keep
/// plain cooked line reads, and a redirected transcript (`omega > log.txt`)
/// must never collect cursor-control escapes; `main` falls back to the
/// [`std::io::Stdin`] source for both.
pub fn stdin_stdout_are_ttys() -> bool {
    // SAFETY: isatty is a pure query on two fds the process always owns.
    unsafe { libc::isatty(libc::STDIN_FILENO) == 1 && libc::isatty(libc::STDOUT_FILENO) == 1 }
}

/// RAII raw-mode toggle: construction switches stdin to raw, dropping
/// restores the captured settings — on normal return and on unwind alike,
/// so a panic mid-edit cannot strand the terminal in raw mode.
struct RawMode {
    saved: libc::termios,
}

impl RawMode {
    /// Switch stdin to raw mode, capturing the settings to restore. `None`
    /// when termios is unavailable; the caller degrades to a cooked read.
    fn enable() -> Option<Self> {
        // SAFETY: tcgetattr/tcsetattr fill and read a plain termios struct
        // by pointer; a zeroed termios is a valid target for tcgetattr.
        unsafe {
            let mut raw: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut raw) != 0 {
                return None;
            }
            let saved = raw;
            // cfmakeraw: no echo (the editor renders), no canonical
            // buffering (bytes arrive per keystroke, VMIN=1), and no ^C
            // signal — the editor handles the byte as cancel-line; a SIGINT
            // would kill the process without running Drop and strand the
            // terminal raw.
            libc::cfmakeraw(&mut raw);
            if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &raw) != 0 {
                return None;
            }
            Some(RawMode { saved })
        }
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        // SAFETY: writes back the very settings tcgetattr captured. A failed
        // restore is unreportable from Drop — best-effort, like the REPL's
        // other terminal writes.
        unsafe {
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.saved);
        }
    }
}

/// The editor-backed [`LineSource`] `main` wires for interactive sessions.
/// Raw mode lasts exactly one read: the agent's streamed output, the shell
/// confirmation gate, and error reporting all run on a cooked terminal
/// between reads, keeping their plain `\n` writes and stdin reads correct.
/// It owns the Up/Down recall [`History`] — memory-only by default, backed
/// by the configured `history_file` when one is set — living here so
/// submitted lines survive across per-line [`editor::edit_line`] calls; the
/// recall logic and the persistence logic are both in covered modules
/// ([`editor`], [`super::history`]).
pub struct TtyEditor {
    history: History,
    // Did the previous per-line read finish on a bare CR? Persisted across
    // `edit_line` calls so the trailing LF of a CRLF pair — which lands in the
    // next read — is swallowed instead of submitting an empty line.
    after_cr: bool,
}

impl TtyEditor {
    pub fn new(history: History) -> Self {
        Self {
            history,
            after_cr: false,
        }
    }
}

impl LineSource for TtyEditor {
    fn read_line(
        &mut self,
        buf: &mut String,
        complete: &mut dyn FnMut(&str) -> Completion,
    ) -> std::io::Result<usize> {
        let before = self.history.entry_count();
        let result = match RawMode::enable() {
            // The Stdin handle locks per read, byte by byte, off the shared
            // process-wide buffer — the same no-read-ahead reasoning as the
            // cooked source in `stdin_live`. Echo writes relock the stdout
            // the REPL already holds; the lock is reentrant on one thread.
            Some(_raw) => editor::edit_line(
                &mut std::io::stdin(),
                &mut std::io::stdout(),
                buf,
                complete,
                self.history.entries_mut(),
                &mut self.after_cr,
            ),
            // isatty passed at startup but termios failed now — degrade to
            // a cooked read (no completion) rather than ending the session.
            None => std::io::Stdin::read_line(&std::io::stdin(), buf),
        };
        // Persist only when the edit stored a new entry, and only after
        // RawMode's drop restored the terminal — the warning below needs
        // cooked newlines. Fail soft like session autosave: a full disk
        // must not end the line read, but it is never swallowed silently.
        if self.history.entry_count() != before
            && let Err(e) = self.history.save()
        {
            eprintln!("warning: history not saved: {e}");
        }
        result
    }
}
