//! The live SIGINT tail of turn cancellation — real signal wiring that
//! cannot run under the hermetic test suite (installing a process-wide
//! handler and raising signals would race every other test). Nothing but
//! the syscall wiring lives here; the decision it acts on
//! ([`super::request_cancel`]) and every consumer of the flag (the agent's
//! seams, the shell tool's kill, the REPL's quiet notice) are in covered
//! modules. The `_live.rs` suffix marks the file for wholesale exclusion
//! from the coverage gate.
//!
//! Terminal-mode map (why this handler and the editor's `0x03` line-cancel
//! never collide): raw mode lasts exactly one `read_line` and switches ISIG
//! off, so Ctrl-C while *editing* arrives as byte `0x03` and stays the
//! editor's line-cancel — no SIGINT is ever raised there. During a turn the
//! terminal is cooked, so Ctrl-C raises SIGINT and lands here. A stray
//! SIGINT in the brief cooked gaps between turns is discarded by
//! `Agent::run`'s start-of-turn reset.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

/// The installed flag. A C signal handler cannot capture state, so the
/// `Arc` main shares with the agent and the shell tool is parked in a
/// process-wide static for the handler to find.
static FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// The handler: async-signal-safe operations only — an atomic swap and,
/// on a repeat Ctrl-C, `_exit`.
extern "C" fn handle_sigint(_signal: libc::c_int) {
    if let Some(flag) = FLAG.get()
        && super::request_cancel(flag)
    {
        // A second Ctrl-C with a cancellation already pending: the first
        // one isn't landing (a hung seam), so force-exit. 130 = 128 +
        // SIGINT, the shell convention. `_exit` skips destructors — raw
        // mode cannot be active here (ISIG is off in raw mode, so this
        // SIGINT came from a cooked terminal or an external kill).
        unsafe { libc::_exit(130) };
    }
}

/// Install the SIGINT → cancellation-flag handler for an interactive
/// session. Piped sessions never call this, keeping the default
/// kill-the-process disposition scripts expect. Installation is
/// best-effort: if `sigaction` fails, Ctrl-C simply keeps its default
/// meaning.
pub fn install_sigint_cancel(flag: Arc<AtomicBool>) {
    let _ = FLAG.set(flag);
    // SAFETY: fills a zeroed sigaction before use and installs a handler
    // that performs only async-signal-safe operations.
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = handle_sigint as *const () as libc::sighandler_t;
        libc::sigemptyset(&mut action.sa_mask);
        // Deliberately no SA_RESTART: a blocking read (the response
        // stream's socket, the confirmation gate's stdin) may return EINTR
        // instead of silently resuming, giving even a hung stream a chance
        // to observe the cancellation. Reads that retry EINTR (std's
        // buffered reads, the raw-mode editor) are simply unaffected.
        action.sa_flags = 0;
        libc::sigaction(libc::SIGINT, &action, std::ptr::null_mut());
    }
}
