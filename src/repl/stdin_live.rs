//! The live stdin [`LineSource`] — real-terminal wiring that cannot run
//! under the hermetic test suite (reading the process's stdin would block
//! `cargo test` on a real terminal). Nothing but this one delegation lives
//! here; the `_live.rs` suffix marks the file for wholesale exclusion from
//! the coverage gate — anything beyond this wiring belongs in a covered
//! module.

use super::{Completion, LineSource};

impl LineSource for std::io::Stdin {
    /// Delegates to `Stdin`'s inherent `read_line`: a per-call lock over the
    /// shared, process-wide stdin buffer. See [`LineSource`] for why a held
    /// `StdinLock` (deadlocks the confirmation gate) or a private
    /// `BufReader` (steals the gate's buffered input) would both be wrong.
    /// Cooked-mode reads deliver whole lines — there is no in-progress
    /// buffer to complete, so the completer goes unused.
    fn read_line(
        &mut self,
        buf: &mut String,
        _complete: &mut dyn FnMut(&str) -> Completion,
    ) -> std::io::Result<usize> {
        std::io::Stdin::read_line(self, buf)
    }
}
