//! Shell tool (phase 8j): execute a command via `/bin/sh -c` inside the
//! guardrails shipped in 8i — command blocklist, sandbox-rooted working
//! directory, allowlisted environment — with a hard timeout and capped output.
//!
//! Cost tier 0 (the default): local execution spends no API budget. Danger
//! is contained by the guardrails and the confirmation gate (8h), not the
//! budget system.

use super::sandbox::Sandbox;
use super::{MAX_RESPONSE_BYTES, ToolDef, shell_guardrails, truncate_response};
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_millis(25);

pub struct ShellTool {
    sandbox: Sandbox,
    timeout: Duration,
    /// The turn-cancellation flag, shared with the agent (see
    /// [`crate::agent::Agent::set_cancel_flag`]). The deadline loop polls it
    /// so a Ctrl-C mid-command kills the process group within one poll
    /// interval instead of riding out the full timeout — the child leads its
    /// own group, so the terminal's SIGINT never reaches it directly.
    cancel: Arc<AtomicBool>,
}

impl ShellTool {
    pub fn new(sandbox: Sandbox, cancel: Arc<AtomicBool>) -> Self {
        Self {
            sandbox,
            timeout: TIMEOUT,
            cancel,
        }
    }
}

/// Extract an optional string field, distinguishing absent from wrong type —
/// a non-string `working_dir` must be an error, not silently ignored.
fn optional_str<'a>(input: &'a serde_json::Value, field: &str) -> Result<Option<&'a str>, String> {
    match &input[field] {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::String(s) => Ok(Some(s)),
        _ => Err(format!("{field} must be a string")),
    }
}

/// Read a child's output stream on a dedicated thread, so the child never
/// blocks on a full pipe while we poll for exit. Capture is capped one byte
/// past the response limit (enough for `truncate_response` to detect
/// overflow); the remainder is drained and discarded.
fn drain(stream: impl Read + Send + 'static) -> JoinHandle<std::io::Result<Vec<u8>>> {
    std::thread::spawn(move || {
        let mut stream = stream;
        let mut buf = Vec::new();
        stream
            .by_ref()
            .take(MAX_RESPONSE_BYTES as u64 + 1)
            .read_to_end(&mut buf)?;
        std::io::copy(&mut stream, &mut std::io::sink())?;
        Ok(buf)
    })
}

fn collect(handle: JoinHandle<std::io::Result<Vec<u8>>>) -> Result<String, String> {
    let bytes = handle
        .join()
        .map_err(|_| "output reader thread panicked".to_string())?
        .map_err(|e| format!("failed to read command output: {e}"))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Kill the shell and everything in its process group. Killing only the
/// shell would orphan pipeline members (`sleep 99 | cat`), which keep the
/// output pipes open and keep running past the timeout. The child leads its
/// own group (spawned with `process_group(0)`), so its pid is the pgid.
///
/// We signal the group with `killpg(2)` directly. Shelling out to
/// `/bin/kill -9 -<pgid>` proved unreliable across platforms: BSD and
/// util-linux parse a leading-dash process-group operand differently, so on
/// some Linux hosts the group kill silently failed and the pipeline outlived
/// the timeout. `killpg` is the call behind Python's `os.killpg`; Rust's
/// stdlib doesn't expose it, so we declare the libc symbol directly rather
/// than take a dependency.
fn kill_group(child: &mut Child) {
    // SAFETY: `killpg` has no memory effects, and SIGKILL is 9 on every Unix.
    // A failure (e.g. the group already exited — a kill/exit race) is ignored.
    unsafe extern "C" {
        fn killpg(pgrp: i32, sig: i32) -> i32;
    }
    unsafe { killpg(child.id() as i32, 9) };
    // Fallback in case the group was never isolated; reap the child either way.
    let _ = child.kill();
    let _ = child.wait();
}

/// The slice of `Child` the deadline loop needs. A real child cannot make
/// `try_wait` fail, so the loop's error arm is only reachable through this
/// seam — tests script a failing waiter, production passes the `Child`.
trait Waitable {
    fn poll(&mut self) -> std::io::Result<Option<ExitStatus>>;
    fn kill(&mut self);
}

impl Waitable for Child {
    fn poll(&mut self) -> std::io::Result<Option<ExitStatus>> {
        self.try_wait()
    }
    fn kill(&mut self) {
        kill_group(self);
    }
}

/// How the deadline loop ended. The two kill outcomes stay distinct so the
/// caller reports the true cause — "timed out" and "cancelled by user" call
/// for different follow-ups from whoever reads the result.
#[derive(Debug)]
enum WaitOutcome {
    Exited(ExitStatus),
    TimedOut,
    Cancelled,
}

/// Poll for exit until `deadline`, killing the process group on timeout or on
/// a pending cancellation (both return `Ok` so the caller can report partial
/// output). A child that finished before the check reports its real exit
/// status even when a cancellation raced it. A wait failure also kills the
/// group (never leak a running child) and surfaces as `Err`.
fn wait_with_deadline(
    child: &mut dyn Waitable,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<WaitOutcome, String> {
    loop {
        match child.poll() {
            Ok(Some(status)) => return Ok(WaitOutcome::Exited(status)),
            Ok(None) if cancel.load(Ordering::Relaxed) => {
                child.kill();
                return Ok(WaitOutcome::Cancelled);
            }
            Ok(None) if Instant::now() >= deadline => {
                child.kill();
                return Ok(WaitOutcome::TimedOut);
            }
            Ok(None) => std::thread::sleep(POLL_INTERVAL),
            Err(e) => {
                child.kill();
                return Err(format!("failed to wait for command: {e}"));
            }
        }
    }
}

/// Error text for a failed `/bin/sh` spawn. The shell always exists on the
/// supported hosts, so no hermetic test can make `spawn` fail; a named fn
/// (passed as a fn pointer) creates no closure for coverage to miss, and the
/// body is covered by its own unit test.
fn spawn_error(e: std::io::Error) -> String {
    format!("failed to spawn shell: {e}")
}

/// Merge captured streams into one model-facing string. stderr is labeled so
/// the model can tell diagnostics apart from command output.
fn combine_output(stdout: &str, stderr: &str) -> String {
    match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => "(no output)".to_string(),
        (false, true) => stdout.to_string(),
        (true, false) => format!("[stderr]\n{stderr}"),
        (false, false) => format!("{stdout}\n[stderr]\n{stderr}"),
    }
}

impl ToolDef for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command via /bin/sh and return its combined stdout \
         and stderr. Commands run from the sandbox root with a minimal \
         environment and are killed after 30 seconds. Use for builds, tests, \
         version control, and other programs. Do NOT use to read files (use \
         read_file), write or modify files (use write_file / edit_file), list \
         directories (use list_directory), or fetch URLs (use web_fetch) — \
         the dedicated tools are safer and cheaper."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (run via /bin/sh -c)"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Directory to run in, relative to the sandbox root (defaults to the root)"
                }
            },
            "required": ["command"]
        })
    }

    fn requires_confirmation(&self) -> bool {
        true
    }

    fn side_effecting(&self, _input: &serde_json::Value) -> bool {
        true
    }

    fn validate(&self, input: &serde_json::Value) -> Result<(), String> {
        let Some(command) = input["command"].as_str() else {
            return Err("missing required field: command".to_string());
        };
        // Reject unknown fields — a decoy field (e.g. `path`) must never
        // accompany a command, where it could mislead anything that
        // summarizes this call. `input` is necessarily an object here:
        // indexing a non-object always yields `Null`, so `command` could
        // only have been extracted from an object.
        let obj = input.as_object().expect("input with a command field");
        for key in obj.keys() {
            if key != "command" && key != "working_dir" {
                return Err(format!("unknown field: {key}"));
            }
        }
        optional_str(input, "working_dir")?;
        shell_guardrails::check_command(command)
    }

    fn format_status(&self, input: &serde_json::Value) -> Option<String> {
        input["command"].as_str().map(|c| format!("$ {c}"))
    }

    fn run(
        &self,
        input: serde_json::Value,
        _out: &mut dyn std::io::Write,
    ) -> Result<String, String> {
        let command = input["command"]
            .as_str()
            .ok_or("missing required field: command")?;
        shell_guardrails::check_command(command)?;
        let working_dir = optional_str(&input, "working_dir")?;
        let cwd = shell_guardrails::resolve_working_dir(&self.sandbox, working_dir)?;

        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c")
            .arg(command)
            .current_dir(&cwd)
            .env_clear()
            .envs(shell_guardrails::sanitized_env())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Own process group, so the timeout can kill the whole pipeline.
            .process_group(0);

        let mut child = cmd.spawn().map_err(spawn_error)?;
        let stdout = drain(child.stdout.take().expect("stdout is piped"));
        let stderr = drain(child.stderr.take().expect("stderr is piped"));

        let deadline = Instant::now() + self.timeout;
        let outcome = wait_with_deadline(&mut child, deadline, &self.cancel)?;

        let output = combine_output(&collect(stdout)?, &collect(stderr)?);
        match outcome {
            WaitOutcome::TimedOut => Err(truncate_response(format!(
                "command timed out after {:?}\n{output}",
                self.timeout
            ))),
            WaitOutcome::Cancelled => Err(truncate_response(format!(
                "command cancelled by user\n{output}"
            ))),
            WaitOutcome::Exited(status) if status.success() => Ok(truncate_response(output)),
            WaitOutcome::Exited(status) => Err(truncate_response(format!(
                "command failed ({status})\n{output}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn sandbox_in(dir: &std::path::Path) -> Sandbox {
        Sandbox::rooted(dir.to_path_buf()).unwrap()
    }

    /// A flag nothing raises — the no-cancellation default for tests.
    fn no_cancel() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    fn tool_in(dir: &std::path::Path) -> ShellTool {
        ShellTool::new(sandbox_in(dir), no_cancel())
    }

    /// Tool with a short timeout so timeout tests don't take 30 seconds.
    fn quick_timeout_tool(dir: &std::path::Path) -> ShellTool {
        ShellTool {
            sandbox: sandbox_in(dir),
            timeout: Duration::from_millis(250),
            cancel: no_cancel(),
        }
    }

    fn run_cmd(tool: &ShellTool, command: &str) -> Result<String, String> {
        tool.run(
            serde_json::json!({ "command": command }),
            &mut std::io::sink(),
        )
    }

    // ── metadata ──

    #[test]
    fn shell_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool_in(dir.path());
        assert_eq!(tool.name(), "shell");
        assert!(!tool.description().is_empty());

        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "command"));
        assert!(!required.iter().any(|v| v == "working_dir"));
        assert!(schema["properties"]["working_dir"].is_object());
    }

    #[test]
    fn shell_format_status() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool_in(dir.path());
        assert_eq!(
            tool.format_status(&serde_json::json!({"command": "ls -la"})),
            Some("$ ls -la".to_string())
        );
        assert_eq!(tool.format_status(&serde_json::json!({})), None);
    }

    // ── validate ──

    #[test]
    fn validate_accepts_command() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool_in(dir.path());
        assert!(tool.validate(&serde_json::json!({"command": "ls"})).is_ok());
        assert!(
            tool.validate(&serde_json::json!({"command": "ls", "working_dir": "sub"}))
                .is_ok()
        );
    }

    #[test]
    fn validate_missing_command() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool_in(dir.path());
        let err = tool.validate(&serde_json::json!({})).unwrap_err();
        assert!(err.contains("missing required field: command"));
    }

    #[test]
    fn validate_rejects_blocked_command() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool_in(dir.path());
        let err = tool
            .validate(&serde_json::json!({"command": "sudo ls"}))
            .unwrap_err();
        assert!(err.contains("blocked command: sudo"));
    }

    #[test]
    fn validate_rejects_non_string_working_dir() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool_in(dir.path());
        let err = tool
            .validate(&serde_json::json!({"command": "ls", "working_dir": 42}))
            .unwrap_err();
        assert!(err.contains("working_dir must be a string"));
    }

    #[test]
    fn validate_accepts_null_working_dir() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool_in(dir.path());
        assert!(
            tool.validate(&serde_json::json!({"command": "ls", "working_dir": null}))
                .is_ok()
        );
    }

    #[test]
    fn validate_rejects_unknown_field() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool_in(dir.path());
        // A decoy `path` alongside the command must be rejected outright.
        let err = tool
            .validate(&serde_json::json!({"command": "rm -rf src", "path": "README.md"}))
            .unwrap_err();
        assert!(err.contains("unknown field: path"));
    }

    #[test]
    fn validate_rejects_non_object_input() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool_in(dir.path());
        // Indexing a non-object yields Null, so this fails the command check.
        let err = tool.validate(&serde_json::Value::Null).unwrap_err();
        assert!(err.contains("missing required field: command"));
    }

    // ── run: output capture ──

    #[test]
    fn run_captures_stdout() {
        let dir = tempfile::tempdir().unwrap();
        let output = run_cmd(&tool_in(dir.path()), "echo hello").unwrap();
        assert_eq!(output, "hello\n");
    }

    #[test]
    fn run_captures_stderr_labeled() {
        let dir = tempfile::tempdir().unwrap();
        let output = run_cmd(&tool_in(dir.path()), "echo oops 1>&2").unwrap();
        assert_eq!(output, "[stderr]\noops\n");
    }

    #[test]
    fn run_combines_streams() {
        let dir = tempfile::tempdir().unwrap();
        let output = run_cmd(&tool_in(dir.path()), "echo out; echo err 1>&2").unwrap();
        assert_eq!(output, "out\n\n[stderr]\nerr\n");
    }

    #[test]
    fn run_no_output_placeholder() {
        let dir = tempfile::tempdir().unwrap();
        let output = run_cmd(&tool_in(dir.path()), "true").unwrap();
        assert_eq!(output, "(no output)");
    }

    #[test]
    fn run_non_utf8_output_is_lossy() {
        let dir = tempfile::tempdir().unwrap();
        let output = run_cmd(&tool_in(dir.path()), r"printf '\377\376ok'").unwrap();
        assert!(output.contains("ok"));
        assert!(output.contains('\u{FFFD}'));
    }

    #[test]
    fn run_truncates_large_output() {
        let dir = tempfile::tempdir().unwrap();
        // 60,000 lines of "a\n" = 120,000 bytes — over the 100 KB cap.
        let output = run_cmd(&tool_in(dir.path()), "yes a | head -n 60000").unwrap();
        assert!(output.ends_with("[truncated at 100 KB]"));
        assert!(output.len() < MAX_RESPONSE_BYTES + 50);
    }

    // ── run: exit status ──

    #[test]
    fn run_nonzero_exit_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = run_cmd(&tool_in(dir.path()), "exit 3").unwrap_err();
        assert!(err.contains("command failed"));
        assert!(err.contains('3'));
    }

    #[test]
    fn run_error_includes_output() {
        let dir = tempfile::tempdir().unwrap();
        let err = run_cmd(&tool_in(dir.path()), "echo boom; exit 1").unwrap_err();
        assert!(err.contains("boom"));
    }

    // ── run: input rejection ──

    #[test]
    fn run_missing_command() {
        let dir = tempfile::tempdir().unwrap();
        let err = tool_in(dir.path())
            .run(serde_json::json!({}), &mut std::io::sink())
            .unwrap_err();
        assert!(err.contains("missing required field: command"));
    }

    #[test]
    fn run_rejects_blocked_command() {
        let dir = tempfile::tempdir().unwrap();
        let err = run_cmd(&tool_in(dir.path()), "sudo whoami").unwrap_err();
        assert!(err.contains("blocked command: sudo"));
    }

    #[test]
    fn run_rejects_background_execution() {
        let dir = tempfile::tempdir().unwrap();
        let err = run_cmd(&tool_in(dir.path()), "sleep 99 &").unwrap_err();
        assert!(err.contains("background"));
    }

    #[test]
    fn run_rejects_non_string_working_dir() {
        let dir = tempfile::tempdir().unwrap();
        let err = tool_in(dir.path())
            .run(
                serde_json::json!({"command": "pwd", "working_dir": ["sub"]}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("working_dir must be a string"));
    }

    // ── run: working directory ──

    #[test]
    fn run_defaults_to_sandbox_root() {
        let dir = tempfile::tempdir().unwrap();
        let output = run_cmd(&tool_in(dir.path()), "pwd").unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        assert_eq!(output.trim(), root.to_str().unwrap());
    }

    #[test]
    fn run_in_working_dir() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        let output = tool_in(dir.path())
            .run(
                serde_json::json!({"command": "pwd", "working_dir": "sub"}),
                &mut std::io::sink(),
            )
            .unwrap();
        let sub = fs::canonicalize(dir.path().join("sub")).unwrap();
        assert_eq!(output.trim(), sub.to_str().unwrap());
    }

    #[test]
    fn run_rejects_working_dir_escape() {
        let dir = tempfile::tempdir().unwrap();
        let err = tool_in(dir.path())
            .run(
                serde_json::json!({"command": "pwd", "working_dir": "/etc"}),
                &mut std::io::sink(),
            )
            .unwrap_err();
        assert!(err.contains("escapes sandbox"));
    }

    // ── run: environment ──

    #[test]
    fn run_strips_secret_env() {
        let dir = tempfile::tempdir().unwrap();
        // SAFETY: test-only; no other test reads this variable.
        unsafe { std::env::set_var("_OMEGA_8J_SECRET", "hunter2") };
        let output = run_cmd(&tool_in(dir.path()), "echo ${_OMEGA_8J_SECRET-unset}").unwrap();
        unsafe { std::env::remove_var("_OMEGA_8J_SECRET") };
        assert_eq!(output.trim(), "unset");
    }

    #[test]
    fn run_keeps_path_env() {
        let dir = tempfile::tempdir().unwrap();
        let output = run_cmd(&tool_in(dir.path()), "printenv PATH").unwrap();
        assert!(!output.trim().is_empty());
    }

    // ── run: timeout ──

    #[test]
    fn run_times_out() {
        let dir = tempfile::tempdir().unwrap();
        let err = run_cmd(&quick_timeout_tool(dir.path()), "sleep 5").unwrap_err();
        assert!(err.contains("timed out after 250ms"));
    }

    #[test]
    fn run_timeout_includes_partial_output() {
        let dir = tempfile::tempdir().unwrap();
        let err = run_cmd(&quick_timeout_tool(dir.path()), "echo first; sleep 5").unwrap_err();
        assert!(err.contains("timed out"));
        assert!(err.contains("first"));
    }

    #[test]
    fn run_timeout_kills_whole_pipeline() {
        let dir = tempfile::tempdir().unwrap();
        // `cat` inherits the output pipe; if only the shell were killed, cat
        // would survive, hold the pipe open, and stall collection for the
        // full 5 seconds. Group kill must finish promptly.
        let start = Instant::now();
        let err = run_cmd(&quick_timeout_tool(dir.path()), "sleep 5 | cat").unwrap_err();
        assert!(err.contains("timed out"));
        // Bound first, assert on one line: a message argument on its own line
        // only executes on failure, which coverage would read as a missed line.
        let elapsed = start.elapsed();
        let within_deadline = elapsed < Duration::from_secs(3);
        assert!(within_deadline, "group kill took {elapsed:?}");
    }

    // ── run: cancellation ──

    #[test]
    fn run_cancelled_kills_the_command_promptly() {
        // A pending cancellation with the full 30-second timeout in force:
        // the kill must come from the cancellation seam, fast, and the error
        // must say "cancelled", not "timed out" or "failed".
        let dir = tempfile::tempdir().unwrap();
        let tool = ShellTool {
            sandbox: sandbox_in(dir.path()),
            timeout: TIMEOUT,
            cancel: Arc::new(AtomicBool::new(true)),
        };
        let start = Instant::now();
        let err = run_cmd(&tool, "sleep 5").unwrap_err();
        assert!(err.contains("command cancelled by user"));
        let elapsed = start.elapsed();
        let within_deadline = elapsed < Duration::from_secs(3);
        assert!(within_deadline, "cancellation kill took {elapsed:?}");
    }

    // ── wait_with_deadline ──

    /// Scripted `Waitable` — drives every arm of the deadline loop, including
    /// the wait-failure arm a real `Child` can never produce.
    struct ScriptedWait {
        polls: Vec<std::io::Result<Option<ExitStatus>>>,
        killed: bool,
    }

    impl ScriptedWait {
        fn new(polls: Vec<std::io::Result<Option<ExitStatus>>>) -> Self {
            Self {
                polls,
                killed: false,
            }
        }
    }

    impl Waitable for ScriptedWait {
        fn poll(&mut self) -> std::io::Result<Option<ExitStatus>> {
            self.polls.remove(0)
        }
        fn kill(&mut self) {
            self.killed = true;
        }
    }

    /// A hermetic `ExitStatus` — raw wait status 0 is success on every Unix.
    fn exit_ok() -> ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(0)
    }

    #[test]
    fn wait_returns_immediate_exit() {
        let mut child = ScriptedWait::new(vec![Ok(Some(exit_ok()))]);
        let deadline = Instant::now() + Duration::from_secs(5);
        let outcome = wait_with_deadline(&mut child, deadline, &no_cancel()).unwrap();
        assert!(matches!(outcome, WaitOutcome::Exited(s) if s.success()));
        assert!(!child.killed);
    }

    #[test]
    fn wait_polls_until_exit() {
        // First poll finds the child still running — the loop sleeps and asks again.
        let mut child = ScriptedWait::new(vec![Ok(None), Ok(Some(exit_ok()))]);
        let deadline = Instant::now() + Duration::from_secs(5);
        let outcome = wait_with_deadline(&mut child, deadline, &no_cancel()).unwrap();
        assert!(matches!(outcome, WaitOutcome::Exited(s) if s.success()));
        assert!(!child.killed);
    }

    #[test]
    fn wait_kills_on_timeout() {
        let mut child = ScriptedWait::new(vec![Ok(None)]);
        let outcome = wait_with_deadline(&mut child, Instant::now(), &no_cancel()).unwrap();
        assert!(matches!(outcome, WaitOutcome::TimedOut));
        assert!(child.killed);
    }

    #[test]
    fn wait_kills_on_cancellation_before_the_deadline() {
        // A pending cancellation kills the still-running group with the
        // deadline nowhere near — and reports Cancelled, not TimedOut.
        let mut child = ScriptedWait::new(vec![Ok(None)]);
        let deadline = Instant::now() + Duration::from_secs(5);
        let cancel = AtomicBool::new(true);
        let outcome = wait_with_deadline(&mut child, deadline, &cancel).unwrap();
        assert!(matches!(outcome, WaitOutcome::Cancelled));
        assert!(child.killed);
    }

    #[test]
    fn wait_reports_the_real_exit_when_cancellation_races_a_finished_child() {
        // Exit wins over a pending cancellation: a command that completed
        // must report its true status, not read as killed.
        let mut child = ScriptedWait::new(vec![Ok(Some(exit_ok()))]);
        let deadline = Instant::now() + Duration::from_secs(5);
        let cancel = AtomicBool::new(true);
        let outcome = wait_with_deadline(&mut child, deadline, &cancel).unwrap();
        assert!(matches!(outcome, WaitOutcome::Exited(s) if s.success()));
        assert!(!child.killed);
    }

    #[test]
    fn wait_kills_on_wait_failure() {
        let mut child = ScriptedWait::new(vec![Err(std::io::Error::other("no such child"))]);
        let deadline = Instant::now() + Duration::from_secs(5);
        let err = wait_with_deadline(&mut child, deadline, &no_cancel()).unwrap_err();
        assert!(err.contains("failed to wait for command: no such child"));
        assert!(child.killed);
    }

    // ── collect ──

    #[test]
    fn collect_surfaces_reader_panic() {
        let handle = std::thread::spawn(|| -> std::io::Result<Vec<u8>> { panic!("boom") });
        let err = collect(handle).unwrap_err();
        assert_eq!(err, "output reader thread panicked");
    }

    #[test]
    fn collect_surfaces_reader_error() {
        let handle = std::thread::spawn(|| -> std::io::Result<Vec<u8>> {
            Err(std::io::Error::other("pipe burst"))
        });
        let err = collect(handle).unwrap_err();
        assert_eq!(err, "failed to read command output: pipe burst");
    }

    // ── error text helpers ──

    #[test]
    fn spawn_error_formats_reason() {
        let msg = spawn_error(std::io::Error::other("no shell"));
        assert_eq!(msg, "failed to spawn shell: no shell");
    }

    // ── combine_output ──

    #[test]
    fn combine_output_branches() {
        assert_eq!(combine_output("", ""), "(no output)");
        assert_eq!(combine_output("out", ""), "out");
        assert_eq!(combine_output("", "err"), "[stderr]\nerr");
        assert_eq!(combine_output("out", "err"), "out\n[stderr]\nerr");
    }
}
