//! Shell-specific guardrails (phase 8i), built before the shell tool ships.
//!
//! Three layers: a destructive-command blocklist, working-directory
//! resolution rooted in the sandbox, and environment sanitization via
//! allowlist. The blocklist is best-effort static analysis of the command
//! string — it does not parse shell quoting, so a literal `&` inside quotes
//! is rejected like real backgrounding (a fail-closed false positive).
//!
//! The user confirmation gate (`ToolDef::requires_confirmation`) is the
//! *permanent* backstop for that best-effort analysis, not a dev-phase
//! scaffold. Two blind spots are known and accepted rather than parsed
//! around: `split_segments` never tokenizes `$(…)`/backtick command
//! substitution, and `sh`/`bash`/`zsh` appear in neither `BLOCKED_COMMANDS`
//! nor `WRAPPER_COMMANDS` — so `echo $(sudo whoami)` and `sh -c 'dd …'`
//! both pass static analysis. They are safe only because the operator sees
//! and approves the full command before each run; sound shell parsing is
//! deliberately out of scope for a static blocklist. A refactor must not
//! make `shell` non-interactive without closing these holes.

use super::sandbox::{Sandbox, cwd_error};
use std::path::PathBuf;

/// Commands rejected outright, matched by basename after stripping any
/// directory prefix (`/bin/dd` → `dd`).
const BLOCKED_COMMANDS: &[&str] = &[
    "sudo", "doas", "su", // privilege escalation
    "shutdown", "reboot", "halt", "poweroff", // machine state
    "dd",       // raw device writes
    "nohup",    // outlives the tool's timeout
];

/// Transparent wrappers skipped to find the effective command, so
/// `env FOO=bar dd ...` is judged as `dd`.
const WRAPPER_COMMANDS: &[&str] = &["command", "exec", "env", "time", "nice", "xargs"];

/// Targets whose recursive removal is never legitimate for this agent.
const CRITICAL_RM_TARGETS: &[&str] = &["/", "/*", "~", "~/", "~/*", "$HOME", "$HOME/", "${HOME}"];

/// Environment variables passed through to child processes. Everything else —
/// including API keys — is stripped. Allowlisting (vs blocklisting known
/// secrets) means new secrets are protected by default.
const ENV_ALLOWLIST: &[&str] = &[
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "TZ", "TMPDIR",
];
const ENV_ALLOWLIST_PREFIXES: &[&str] = &["LC_"];

/// Validate a shell command against the destructive-command blocklist.
/// Returns `Err` with a reason suitable for sending back to the model.
pub fn check_command(command: &str) -> Result<(), String> {
    if command.trim().is_empty() {
        return Err("command must not be empty".to_string());
    }
    for segment in split_segments(command)? {
        check_segment(&segment)?;
    }
    Ok(())
}

/// Split a command into segments at `;`, `|`, `&&`, `||`, and newlines so
/// each sub-command is checked independently (`ls; dd ...` cannot hide the
/// `dd`). A lone `&` is rejected: background processes would outlive the
/// tool's timeout. `&&` (separator) and `>&` (fd duplication, e.g. `2>&1`) are
/// not backgrounding. `&>` *is* rejected: the tool runs `/bin/sh -c`, and under
/// POSIX sh (dash, the default `/bin/sh` on Debian/Ubuntu) `cmd &> file` parses
/// as `cmd &` (background) + `> file`, which backgrounds past the timeout,
/// hangs the drain, and leaks the child.
fn split_segments(command: &str) -> Result<Vec<String>, String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = command.chars().peekable();
    let mut prev = None;
    while let Some(c) = chars.next() {
        match c {
            ';' | '\n' => segments.push(std::mem::take(&mut current)),
            '|' => {
                if chars.peek() == Some(&'|') {
                    chars.next();
                }
                segments.push(std::mem::take(&mut current));
            }
            '&' => {
                if chars.peek() == Some(&'&') {
                    chars.next();
                    segments.push(std::mem::take(&mut current));
                } else if prev == Some('>') {
                    // `>&` fd-duplication (`2>&1`), not backgrounding. `&>` is
                    // NOT accepted: the tool runs `/bin/sh -c`, and under POSIX
                    // sh (dash) `cmd &> file` parses as `cmd &` (background) + `>
                    // file`, which backgrounds past the tool's timeout.
                    current.push('&');
                } else {
                    return Err("background execution (&) is not allowed".to_string());
                }
            }
            _ => current.push(c),
        }
        prev = Some(c);
    }
    segments.push(current);
    Ok(segments
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect())
}

fn check_segment(segment: &str) -> Result<(), String> {
    check_redirects(segment)?;

    let tokens: Vec<&str> = segment.split_whitespace().collect();

    // Find the effective command: skip env assignments (`FOO=bar cmd`),
    // transparent wrappers, and their flags. Once a wrapper is present, its
    // options may consume following operands, so conservatively treat every
    // later non-assignment token as a possible executable. This can reject a
    // blocked command name used only as a wrapper argument; failing closed is
    // preferable to letting an option operand hide the real command.
    let mut idx = 0;
    let mut wrapped_suffix = None;
    while idx < tokens.len() {
        let token = tokens[idx];
        if is_env_assignment(token) || token.starts_with('-') {
            idx += 1;
        } else if WRAPPER_COMMANDS.contains(&command_name(token)) {
            wrapped_suffix.get_or_insert(idx + 1);
            idx += 1;
        } else {
            break;
        }
    }

    if let Some(start) = wrapped_suffix {
        for possible_idx in start..tokens.len() {
            if is_env_assignment(tokens[possible_idx]) {
                continue;
            }
            check_possible_command(tokens[possible_idx], &tokens[possible_idx + 1..])?;
        }
        return Ok(());
    }

    let Some(&cmd_token) = tokens.get(idx) else {
        return Ok(());
    };
    check_possible_command(cmd_token, &tokens[idx + 1..])
}

fn check_possible_command(token: &str, remaining: &[&str]) -> Result<(), String> {
    let cmd = command_name(token);
    if BLOCKED_COMMANDS.contains(&cmd) {
        return Err(format!("blocked command: {cmd}"));
    }
    if cmd.starts_with("mkfs") {
        return Err(format!("blocked command: {cmd} (filesystem formatting)"));
    }
    if cmd == "rm" {
        check_rm(remaining)?;
    }
    Ok(())
}

/// Reject `rm` when a recursive flag is combined with a critical target.
fn check_rm(args: &[&str]) -> Result<(), String> {
    let mut recursive = false;
    let mut targets = Vec::new();
    for &arg in args {
        if arg == "--recursive" {
            recursive = true;
        } else if arg.starts_with("--") {
            // other long flag (--force, --verbose, ...)
        } else if let Some(flags) = arg.strip_prefix('-') {
            if flags.contains('r') || flags.contains('R') {
                recursive = true;
            }
        } else {
            targets.push(arg);
        }
    }
    if recursive {
        for target in targets {
            if CRITICAL_RM_TARGETS.contains(&target) {
                return Err(format!("blocked: recursive rm of critical path '{target}'"));
            }
        }
    }
    Ok(())
}

/// Reject redirects that write to device nodes (`> /dev/sda` destroys disks).
/// `/dev/null` is allowed — discarding output is routine.
fn check_redirects(segment: &str) -> Result<(), String> {
    let mut rest = segment;
    while let Some(pos) = rest.find('>') {
        rest = &rest[pos + 1..];
        let target: String = rest
            .trim_start_matches('>')
            .trim_start()
            .chars()
            .take_while(|c| !c.is_whitespace())
            .collect();
        if target.starts_with("/dev/") && target != "/dev/null" {
            return Err(format!("blocked: redirect to device '{target}'"));
        }
    }
    Ok(())
}

/// Basename of a command token: `/bin/rm` → `rm`, `./dd` → `dd`.
fn command_name(token: &str) -> &str {
    token.rsplit('/').next().unwrap_or(token)
}

/// `FOO=bar` and `FOO2=x` are env assignments; `if=/dev/zero` is not a valid
/// variable name but is harmless to skip — it can only precede the command.
fn is_env_assignment(token: &str) -> bool {
    match token.split_once('=') {
        Some((key, _)) => {
            !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        }
        None => false,
    }
}

/// Resolve the working directory for a shell command. `None` defaults to the
/// sandbox root (or the process CWD when unbounded); `Some` is resolved
/// through the sandbox so a command cannot run rooted outside the boundary.
pub fn resolve_working_dir(sandbox: &Sandbox, dir: Option<&str>) -> Result<PathBuf, String> {
    let resolved = match dir {
        Some(d) => sandbox.resolve(d)?,
        None => match sandbox.root() {
            Some(root) => root.to_path_buf(),
            None => std::env::current_dir().map_err(cwd_error)?,
        },
    };
    if !resolved.is_dir() {
        return Err(format!("not a directory: {}", resolved.display()));
    }
    Ok(resolved)
}

/// Environment for child processes: only allowlisted variables survive.
pub fn sanitized_env() -> Vec<(String, String)> {
    std::env::vars()
        .filter(|(name, _)| is_allowed_var(name))
        .collect()
}

fn is_allowed_var(name: &str) -> bool {
    ENV_ALLOWLIST.contains(&name)
        || ENV_ALLOWLIST_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // ── check_command: ordinary commands pass ──

    #[test]
    fn allows_simple_command() {
        assert!(check_command("ls -la").is_ok());
    }

    #[test]
    fn allows_chained_commands() {
        assert!(check_command("cargo build && cargo test").is_ok());
    }

    #[test]
    fn allows_pipes() {
        assert!(check_command("ls | grep foo | wc -l").is_ok());
    }

    #[test]
    fn allows_or_chain() {
        assert!(check_command("test -f x || echo missing").is_ok());
    }

    #[test]
    fn allows_env_assignment_prefix() {
        assert!(check_command("RUST_LOG=debug cargo test").is_ok());
    }

    #[test]
    fn allows_segment_with_no_effective_command() {
        // Every token is consumed as an env assignment or wrapper — nothing
        // left to judge, and a bare assignment is a harmless no-op.
        assert!(check_command("FOO=bar").is_ok());
        assert!(check_command("env").is_ok());
    }

    #[test]
    fn allows_fd_duplication() {
        assert!(check_command("cargo test > out.log 2>&1").is_ok());
    }

    #[test]
    fn rejects_ampersand_redirect_as_background() {
        // `&>` is bash/zsh redirect-both, but under POSIX sh (dash) it is
        // `cmd &` (background) + `> file` — which backgrounds past the tool's
        // timeout. The tool runs `/bin/sh -c`, so reject it.
        let err = check_command("cargo test &> out.log").unwrap_err();
        assert!(err.contains("background"));
    }

    #[test]
    fn allows_redirect_to_dev_null() {
        assert!(check_command("ls 2>/dev/null").is_ok());
        assert!(check_command("ls > /dev/null").is_ok());
    }

    #[test]
    fn allows_redirect_to_file() {
        assert!(check_command("echo hi > out.txt").is_ok());
        assert!(check_command("echo hi >> out.txt").is_ok());
    }

    // ── check_command: rejections ──

    #[test]
    fn rejects_empty_command() {
        let err = check_command("").unwrap_err();
        assert!(err.contains("must not be empty"));
    }

    #[test]
    fn rejects_whitespace_only_command() {
        let err = check_command("   \n  ").unwrap_err();
        assert!(err.contains("must not be empty"));
    }

    #[test]
    fn rejects_background_execution() {
        let err = check_command("sleep 100 &").unwrap_err();
        assert!(err.contains("background"));
    }

    #[test]
    fn rejects_fork_bomb_via_background() {
        let err = check_command(":(){ :|:& };:").unwrap_err();
        assert!(err.contains("background"));
    }

    #[test]
    fn rejects_sudo() {
        let err = check_command("sudo rm file").unwrap_err();
        assert!(err.contains("blocked command: sudo"));
    }

    #[test]
    fn rejects_doas_and_su() {
        assert!(check_command("doas ls").is_err());
        assert!(check_command("su root").is_err());
    }

    #[test]
    fn rejects_machine_state_commands() {
        assert!(check_command("shutdown -h now").is_err());
        assert!(check_command("reboot").is_err());
        assert!(check_command("halt").is_err());
        assert!(check_command("poweroff").is_err());
    }

    #[test]
    fn rejects_dd() {
        let err = check_command("dd if=/dev/zero of=/dev/sda").unwrap_err();
        assert!(err.contains("blocked command: dd"));
    }

    #[test]
    fn rejects_nohup() {
        let err = check_command("nohup ./server").unwrap_err();
        assert!(err.contains("blocked command: nohup"));
    }

    #[test]
    fn rejects_mkfs_variants() {
        assert!(check_command("mkfs /dev/sda1").is_err());
        let err = check_command("mkfs.ext4 /dev/sda1").unwrap_err();
        assert!(err.contains("filesystem formatting"));
    }

    #[test]
    fn rejects_blocked_command_by_absolute_path() {
        let err = check_command("/bin/dd if=x of=y").unwrap_err();
        assert!(err.contains("blocked command: dd"));
    }

    #[test]
    fn rejects_blocked_command_by_relative_path() {
        let err = check_command("./sudo whatever").unwrap_err();
        assert!(err.contains("blocked command: sudo"));
    }

    #[test]
    fn rejects_blocked_command_in_later_segment() {
        let err = check_command("ls; dd if=x of=y").unwrap_err();
        assert!(err.contains("blocked command: dd"));
    }

    #[test]
    fn rejects_blocked_command_after_pipe() {
        let err = check_command("echo y | sudo tee /etc/hosts").unwrap_err();
        assert!(err.contains("blocked command: sudo"));
    }

    #[test]
    fn rejects_blocked_command_behind_wrapper() {
        let err = check_command("env FOO=bar dd if=x").unwrap_err();
        assert!(err.contains("blocked command: dd"));
    }

    #[test]
    fn rejects_blocked_commands_hidden_by_wrapper_option_operands() {
        for command in [
            "nice -n 10 dd if=/dev/zero of=/dev/null",
            "env -u DEBUG sudo true",
            "time -f FORMAT nohup ./server",
            "xargs -n 1 mkfs.ext4 /dev/sda1",
        ] {
            assert!(check_command(command).is_err(), "accepted {command:?}");
        }
    }

    #[test]
    fn rejects_wrapped_blocked_commands_and_recursive_rm() {
        for command in [
            "nice sudo true",
            "env nohup ./server",
            "time mkfs.ext4 /dev/sda1",
            "nice -n 10 rm -rf /",
        ] {
            assert!(check_command(command).is_err(), "accepted {command:?}");
        }
    }

    #[test]
    fn allows_harmless_wrapped_commands_with_option_operands() {
        assert!(check_command("nice -n 10 cargo test").is_ok());
        assert!(check_command("env -u DEBUG cargo test").is_ok());
    }

    #[test]
    fn conservatively_rejects_blocked_name_used_as_wrapper_option_operand() {
        let err = check_command("env -u sudo cargo test").unwrap_err();
        assert!(err.contains("blocked command: sudo"));
    }

    #[test]
    fn rejects_blocked_command_behind_env_assignment() {
        let err = check_command("FOO=bar dd if=x").unwrap_err();
        assert!(err.contains("blocked command: dd"));
    }

    #[test]
    fn rejects_rm_via_xargs() {
        let err = check_command("xargs rm -rf /").unwrap_err();
        assert!(err.contains("critical path"));
    }

    #[test]
    fn rejects_redirect_to_device() {
        let err = check_command("echo x > /dev/sda").unwrap_err();
        assert!(err.contains("redirect to device"));
        assert!(check_command("echo x >/dev/sda").is_err());
        assert!(check_command("echo x >> /dev/sda").is_err());
    }

    // ── rm rules ──

    #[test]
    fn rejects_rm_rf_root() {
        let err = check_command("rm -rf /").unwrap_err();
        assert!(err.contains("critical path '/'"));
    }

    #[test]
    fn rejects_rm_recursive_variants() {
        assert!(check_command("rm -fr /").is_err());
        assert!(check_command("rm -r /").is_err());
        assert!(check_command("rm -R /").is_err());
        assert!(check_command("rm --recursive --force /").is_err());
    }

    #[test]
    fn rejects_rm_rf_critical_targets() {
        assert!(check_command("rm -rf /*").is_err());
        assert!(check_command("rm -rf ~").is_err());
        assert!(check_command("rm -rf ~/").is_err());
        assert!(check_command("rm -rf $HOME").is_err());
    }

    #[test]
    fn allows_rm_non_recursive() {
        assert!(check_command("rm file.txt").is_ok());
    }

    #[test]
    fn allows_rm_recursive_of_local_dir() {
        assert!(check_command("rm -rf target").is_ok());
        assert!(check_command("rm -rf ./build").is_ok());
    }

    // ── helpers ──

    #[test]
    fn command_name_strips_path() {
        assert_eq!(command_name("/usr/bin/rm"), "rm");
        assert_eq!(command_name("./dd"), "dd");
        assert_eq!(command_name("ls"), "ls");
    }

    #[test]
    fn is_env_assignment_detection() {
        assert!(is_env_assignment("FOO=bar"));
        assert!(is_env_assignment("RUST_LOG=debug"));
        assert!(!is_env_assignment("ls"));
        assert!(!is_env_assignment("=bar"));
        assert!(!is_env_assignment("a b=c"));
    }

    // ── resolve_working_dir ──

    #[test]
    fn working_dir_defaults_to_sandbox_root() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let resolved = resolve_working_dir(&sb, None).unwrap();
        assert_eq!(resolved, fs::canonicalize(dir.path()).unwrap());
    }

    #[test]
    fn working_dir_resolves_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let resolved = resolve_working_dir(&sb, Some("sub")).unwrap();
        assert_eq!(resolved, fs::canonicalize(dir.path().join("sub")).unwrap());
    }

    #[test]
    fn working_dir_rejects_escape() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let err = resolve_working_dir(&sb, Some("/etc")).unwrap_err();
        assert!(err.contains("escapes sandbox"));
    }

    #[test]
    fn working_dir_rejects_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("f.txt"), "x").unwrap();
        let sb = Sandbox::rooted(dir.path().to_path_buf()).unwrap();
        let err = resolve_working_dir(&sb, Some("f.txt")).unwrap_err();
        assert!(err.contains("not a directory"));
    }

    #[test]
    fn working_dir_unbounded_defaults_to_cwd() {
        let sb = Sandbox::unbounded();
        let resolved = resolve_working_dir(&sb, None).unwrap();
        assert_eq!(resolved, std::env::current_dir().unwrap());
    }

    // ── sanitized_env ──

    #[test]
    fn sanitized_env_strips_secrets() {
        // SAFETY: test-only; no other test reads this variable.
        unsafe { std::env::set_var("_OMEGA_8I_SECRET_KEY", "hunter2") };
        let env = sanitized_env();
        assert!(!env.iter().any(|(k, _)| k == "_OMEGA_8I_SECRET_KEY"));
        unsafe { std::env::remove_var("_OMEGA_8I_SECRET_KEY") };
    }

    #[test]
    fn sanitized_env_keeps_path() {
        let env = sanitized_env();
        assert!(env.iter().any(|(k, _)| k == "PATH"));
    }

    #[test]
    fn sanitized_env_keeps_lc_prefix() {
        // SAFETY: test-only; no other test reads this variable.
        unsafe { std::env::set_var("LC_OMEGA_TEST", "x") };
        let env = sanitized_env();
        assert!(env.iter().any(|(k, _)| k == "LC_OMEGA_TEST"));
        unsafe { std::env::remove_var("LC_OMEGA_TEST") };
    }

    #[test]
    fn is_allowed_var_rules() {
        assert!(is_allowed_var("PATH"));
        assert!(is_allowed_var("HOME"));
        assert!(is_allowed_var("LC_ALL"));
        assert!(!is_allowed_var("ANTHROPIC_API_KEY"));
        assert!(!is_allowed_var("FIRECRAWL_API_KEY"));
        assert!(!is_allowed_var("LCX"));
    }
}
