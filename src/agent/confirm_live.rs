//! The live half of the confirmation gate: real stdin/stderr wiring.
//!
//! Reading the process's stdin cannot run under the hermetic test suite (it
//! would block `cargo test` on a real terminal), so this file holds nothing
//! but the thin wrapper over [`super::confirm_with`], which carries the
//! approve/deny logic and its coverage. The `_live.rs` suffix marks the file
//! for wholesale exclusion from the coverage gate — anything beyond this
//! wiring belongs in a covered module.

/// Default interactive confirmation: prompts on stderr and reads stdin.
/// The stdin lock is taken per call, never held across the prompt, so it
/// interleaves with the REPL's own (equally transient) stdin reads.
///
/// A plain `fn(&str) -> bool`, hence `Send + Sync`, so the `task` tool can
/// hold it as the base confirm an executor child prompts through — the child
/// wraps it to name the delegating profile, on the main thread where the
/// executor runs (see [`crate::tools::subagent::TaskTool`]).
pub fn interactive_confirm(summary: &str) -> bool {
    super::confirm_with(
        &mut std::io::stdin().lock(),
        &mut std::io::stderr(),
        summary,
    )
}
