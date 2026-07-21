//! Runtime configuration, loaded from `config.json`. This is the single source
//! of truth for which provider the agent talks to, the model id, and the
//! generation limits — the factory in [`crate::provider`] turns the `provider`
//! field into a live [`crate::provider::Provider`].

use crate::provider::ProviderKind;
use std::path::Path;

/// The parsed contents of `config.json`. `provider` selects the adapter and
/// (via [`ProviderKind::api_key_env`]) the env var holding its key; `model` and
/// `max_tokens` flow into every request; `system` is the optional system prompt;
/// `context_token_limit` sizes the context-compaction guard; `session_file`
/// opts into on-disk session persistence; `history_file` opts into a
/// persistent editor history.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub provider: ProviderKind,
    pub model: String,
    pub max_tokens: u32,
    #[serde(default)]
    pub system: Option<String>,
    /// The model's context-window size in tokens. The agent compacts its
    /// conversation history once the measured prompt size crosses a fixed
    /// fraction of this limit. Defaulted — `deny_unknown_fields` makes every
    /// non-defaulted field mandatory, which would break existing configs — and
    /// deliberately not derived from `model`: there is no model registry to
    /// look it up, so it is the operator's knob.
    #[serde(default = "default_context_token_limit")]
    pub context_token_limit: u32,
    /// Maximum number of provider round-trips the agent's tool loop runs before
    /// it stops with [`crate::agent::AgentError::TurnLimitExceeded`]. Defaulted
    /// to [`crate::agent::MAX_TURNS`] — the `context_token_limit` precedent
    /// exactly, so existing configs are unchanged — and a spend knob, not a
    /// safety one: ten rounds suit a scoped executor, but a repo survey needs
    /// more. A per-profile
    /// [`AgentProfile::max_turns`] overrides it for a delegated child. Rejected
    /// when zero by [`validate`].
    #[serde(default = "default_max_turns")]
    pub max_turns: u32,
    /// Optional path of the session-persistence file. Absent (the default)
    /// keeps conversations in memory only — today's behavior, so existing
    /// configs need no change. Set, the agent restores from the file at
    /// startup and autosaves after every turn (see [`crate::session`]).
    /// Defaulted for the same `deny_unknown_fields` reason as
    /// `context_token_limit`.
    #[serde(default)]
    pub session_file: Option<String>,
    /// Optional path of the editor-history file — the `session_file`
    /// precedent exactly. Absent (the default) keeps Up/Down recall
    /// in-memory and session-scoped, today's behavior; set, the interactive
    /// editor loads it at startup and appends each submitted line (see
    /// [`crate::repl::History`]). Only interactive sessions consult it —
    /// piped runs have no editor.
    #[serde(default)]
    pub history_file: Option<String>,
    /// Optional reasoning-effort level applied to conversation turns — the
    /// `session_file` precedent exactly. Absent (the default) sends no effort
    /// field, keeping every request byte-identical to today's; set, it flows
    /// into each turn and is validated by the provider at request time (there
    /// is no client-side enum or registry, like `model`). Recoverable and
    /// visible at runtime through the REPL's `/effort` command.
    #[serde(default)]
    pub effort: Option<String>,
    /// Named subagent profiles the `task` tool can delegate to.
    /// Absent (the default) is an empty list — no `task` tool is registered
    /// and the agent behaves exactly as before. Each profile names a provider,
    /// model, and (optionally) effort, system prompt, and tool allowlist;
    /// [`validate_agents`] fails the load on a malformed set.
    #[serde(default)]
    pub agents: Vec<AgentProfile>,
    /// The session confirmation policy. Absent (the default) is
    /// [`ConfirmMode::Ask`] — the interactive prompt, so existing configs are
    /// unchanged. `allow` and `judge` enable unattended runs; a `judge` mode
    /// must name a profile from `agents`, checked at load so a typo fails
    /// here rather than as a silent deny on the first dangerous call.
    #[serde(default)]
    pub confirm: ConfirmMode,
}

/// One confirmation mode — the session-level setting in `config.json`'s
/// `confirm` field, a per-profile override in [`AgentProfile::confirm`], and
/// the `/confirm` command's argument. Serde's external tagging yields the wire
/// shape directly: `"ask"`, `"allow"`, or `{"judge": "<profile>"}`.
///
/// The modes are dispatch labels only; the behavior — what `allow` still
/// guards, how a judge verdict is parsed — lives in
/// [`crate::agent::ConfirmPolicy`].
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConfirmMode {
    /// The interactive TTY prompt — today's behavior and the default.
    #[default]
    Ask,
    /// Auto-approve the human prompt only; every deterministic guard still
    /// runs. For unattended runs where the operator trusts the toolset.
    Allow,
    /// Adjudicate each dangerous call with a one-shot request to the named
    /// profile. Fail-closed: anything but a well-formed ALLOW is a deny.
    Judge(String),
}

impl std::fmt::Display for ConfirmMode {
    /// Renders the `/confirm` argument spelling (`ask`, `allow`,
    /// `judge <profile>`), so the report line echoes what the operator would
    /// type to select the mode.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfirmMode::Ask => f.write_str("ask"),
            ConfirmMode::Allow => f.write_str("allow"),
            ConfirmMode::Judge(profile) => write!(f, "judge {profile}"),
        }
    }
}

/// One named subagent profile from `config.json`'s `agents` array — a role the
/// `task` tool can spawn (e.g. a `gpt-5.6-sol` reviewer). `provider`/`model`
/// pick the child's backend; `effort`/`system` tune it; `tools` is its
/// allowlist (omitted ⇒ the read-only default set). The child transcript is
/// ephemeral — only its final text returns to the parent.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentProfile {
    /// The name the model passes as the `task` tool's `profile` argument.
    /// Non-empty and unique across the set — enforced by [`validate_agents`].
    pub name: String,
    /// The provider the child talks to. Deserialized as [`ProviderKind`], so
    /// an unknown name fails the parse with `unknown variant` before
    /// validation runs — the config's "known provider" guard.
    pub provider: ProviderKind,
    /// The child's model id — opaque and validated provider-side, like the
    /// top-level `model`. Non-empty, enforced by [`validate_agents`].
    pub model: String,
    /// Optional per-profile reasoning effort. A per-call `task` `effort`
    /// argument overrides it; absent both, the child runs at the provider's
    /// default.
    #[serde(default)]
    pub effort: Option<String>,
    /// Optional system prompt for the child. `None` gives it no persona beyond
    /// the shared tool-selection policy.
    #[serde(default)]
    pub system: Option<String>,
    /// Optional tool allowlist. `None` ⇒ the read-only default set
    /// ([`crate::tools::subagent::default_child_tool_names`]). Every listed
    /// name must be a known read-only child tool; `task` is rejected (no
    /// recursion) — both enforced by [`validate_agents`].
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Optional confirmation-mode override for children spawned from this
    /// profile, taking precedence over the session setting — e.g. an executor
    /// profile that always goes through the judge even when the session runs
    /// `allow`. `None` (the default) inherits the session policy.
    #[serde(default)]
    pub confirm: Option<ConfirmMode>,
    /// Optional per-profile turn-limit override. `None` (the default) inherits
    /// the session's top-level [`Config::max_turns`]; a value lets a
    /// survey-shaped profile run more rounds than a scoped executor — resolved
    /// as `max_turns.unwrap_or(top_level)` when the `task` tool spawns the
    /// child. Rejected when zero by [`validate`], like the top-level field.
    #[serde(default)]
    pub max_turns: Option<u32>,
}

/// Fail-fast validation of the `agents` set, run at config load. Guards: names
/// are non-empty and unique, models are non-empty, and every tool in a
/// profile's allowlist is a known read-only child tool that is not `task`
/// (recursion). The provider is already validated by serde (an unknown name is
/// an `unknown variant` parse error), so it needs no check here.
fn validate_agents(agents: &[AgentProfile]) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    for agent in agents {
        if agent.name.is_empty() {
            return Err("agent profile has an empty name".to_string());
        }
        if !seen.insert(agent.name.as_str()) {
            return Err(format!("duplicate agent profile name: {}", agent.name));
        }
        if agent.model.is_empty() {
            return Err(format!("agent profile '{}' has an empty model", agent.name));
        }
        if let Some(tools) = &agent.tools {
            for tool in tools {
                if tool == "task" {
                    return Err(format!(
                        "agent profile '{}' may not grant the task tool (no recursion)",
                        agent.name
                    ));
                }
                if !crate::tools::subagent::is_valid_child_tool(tool) {
                    return Err(format!(
                        "agent profile '{}' lists unknown tool: {tool}",
                        agent.name
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Fail-fast check that a `judge` mode names a configured profile, run at load
/// for the session `confirm` and every profile override. A typo would
/// otherwise surface as a silent deny on the first dangerous call of an
/// unattended run — the one place nobody is watching for it. `owner` names the
/// field in the error (`confirm` or the profile carrying the override).
fn validate_confirm(
    mode: &ConfirmMode,
    agents: &[AgentProfile],
    owner: &str,
) -> Result<(), String> {
    if let ConfirmMode::Judge(profile) = mode
        && !agents.iter().any(|a| a.name == *profile)
    {
        return Err(format!("{owner} names unknown judge profile: {profile}"));
    }
    Ok(())
}

/// 200k tokens — the context window of current Anthropic models and a safe
/// floor for the models the two adapters target.
fn default_context_token_limit() -> u32 {
    200_000
}

/// The turn-limit default: [`crate::agent::MAX_TURNS`], so a config that never
/// mentions `max_turns` runs the historical ten rounds byte-identically.
fn default_max_turns() -> u32 {
    crate::agent::MAX_TURNS
}

/// Fail-fast validation shared by the single-file and layered load paths, so a
/// merged config runs the identical checks (a global `confirm: {"judge": ...}`
/// resolves against the merged profile table). Runs after any confirm clamps.
fn validate(config: &Config) -> Result<(), String> {
    // A zero limit would arm the compaction guard permanently — every turn
    // would summarize history away. Reject the typo at load time.
    if config.context_token_limit == 0 {
        return Err("context_token_limit must be greater than 0".to_string());
    }
    // A zero turn limit would stop the loop before its first round-trip — the
    // agent could never answer. Reject the typo at load time, and every
    // per-profile override too (each resolves to a live limit for a child).
    if config.max_turns == 0 {
        return Err("max_turns must be greater than 0".to_string());
    }
    for agent in &config.agents {
        if agent.max_turns == Some(0) {
            return Err(format!(
                "agent profile '{}' max_turns must be greater than 0",
                agent.name
            ));
        }
    }
    validate_agents(&config.agents)?;
    validate_confirm(&config.confirm, &config.agents, "confirm")?;
    for agent in &config.agents {
        if let Some(mode) = &agent.confirm {
            let owner = format!("agent profile '{}' confirm", agent.name);
            validate_confirm(mode, &config.agents, &owner)?;
        }
    }
    Ok(())
}

impl Config {
    /// Parse configuration from a JSON string. The error is the raw serde
    /// message; [`Config::load`] adds the file-path context.
    pub fn from_json(contents: &str) -> Result<Config, String> {
        let config: Config = serde_json::from_str(contents).map_err(|e| e.to_string())?;
        validate(&config)?;
        Ok(config)
    }

    /// Read and parse the config file at `path`, failing fast with a message
    /// that names the file for both the read and the parse error.
    pub fn load(path: &str) -> Result<Config, String> {
        Config::load_path(std::path::Path::new(path))
    }

    /// [`Config::load`] over a [`Path`], so the layered loader can reuse the
    /// exact single-file behavior — same `cannot read`/`invalid` messages — when
    /// the global layer is absent.
    fn load_path(path: &std::path::Path) -> Result<Config, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Config::from_json(&contents).map_err(|e| format!("invalid {}: {e}", path.display()))
    }
}

// ── Layered configuration ─────────────────────────────────────────────────
//
// Omega gains a global settings layer (`~/.omega-system/config.json`) that a
// project's `config.json` overrides field by field. A plain `Option<T>` cannot
// tell an absent key from an explicit `null`, but four fields are already
// nullable and the merge must distinguish "inherit the other layer" (absent)
// from "clear the other layer" (null) — so each layer parses into a mirror of
// `Config` whose every field is a three-state [`LayerField`].

/// One config field's state within a single layer: absent from the file
/// (`Missing`), present as JSON `null` (`Null`), or present with a value
/// (`Value`). The distinction drives the merge — project `Missing` inherits the
/// global's state, project `Null` clears it, project `Value` wins.
#[derive(Debug)]
enum LayerField<T> {
    Missing,
    Null,
    Value(T),
}

/// `#[serde(default)]` supplies `Missing` for keys absent from the file; a
/// present key is deserialized by the impl below.
impl<T> Default for LayerField<T> {
    fn default() -> Self {
        LayerField::Missing
    }
}

impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for LayerField<T> {
    /// A present key deserializes as `Option<T>`: JSON `null` → `None` → `Null`,
    /// any other value → `Some` → `Value`. Absent keys never reach here —
    /// `#[serde(default)]` fills them with `Missing`.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(match Option::<T>::deserialize(deserializer)? {
            Some(v) => LayerField::Value(v),
            None => LayerField::Null,
        })
    }
}

/// One layer of configuration — the global file or the project file — parsed
/// strictly. `deny_unknown_fields` keeps both layers strict (tolerant parsing is
/// only for managed layers, which Omega has none of), and every field being a
/// [`LayerField`] means each layer fully typechecks on its own: a type error in
/// one layer is fatal even when the other would have overridden the field.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigLayer {
    #[serde(default)]
    provider: LayerField<ProviderKind>,
    #[serde(default)]
    model: LayerField<String>,
    #[serde(default)]
    max_tokens: LayerField<u32>,
    #[serde(default)]
    system: LayerField<String>,
    #[serde(default)]
    context_token_limit: LayerField<u32>,
    #[serde(default)]
    max_turns: LayerField<u32>,
    #[serde(default)]
    session_file: LayerField<String>,
    #[serde(default)]
    history_file: LayerField<String>,
    #[serde(default)]
    effort: LayerField<String>,
    #[serde(default)]
    agents: LayerField<Vec<AgentProfile>>,
    #[serde(default)]
    confirm: LayerField<ConfirmMode>,
}

/// Restrictiveness rank for the more-restrictive-wins `confirm` merge:
/// `Ask` (interactive, nothing auto-approved) is strongest, `Allow` (every human
/// prompt auto-approved) weakest, `Judge` between. A global floor can only raise
/// a project's mode, never lower it.
fn confirm_rank(mode: &ConfirmMode) -> u8 {
    match mode {
        ConfirmMode::Allow => 0,
        ConfirmMode::Judge(_) => 1,
        ConfirmMode::Ask => 2,
    }
}

/// Merge a plain field: the project layer wins unless it is `Missing`, in which
/// case the global layer's state carries through.
fn merge_plain<T>(project: LayerField<T>, global: LayerField<T>) -> LayerField<T> {
    match project {
        LayerField::Missing => global,
        project => project,
    }
}

/// Collapse a merged field to `Some` only when it carries a value; `Missing` and
/// `Null` both mean "unset" at finalization (serde-default / `None`).
fn into_option<T>(field: LayerField<T>) -> Option<T> {
    match field {
        LayerField::Value(v) => Some(v),
        _ => None,
    }
}

/// A required field (`provider`, `model`, `max_tokens`) after merging: an error
/// names the field and the layer paths consulted so a half-populated pair of
/// files points the operator at both.
fn require<T>(field: LayerField<T>, name: &str, consulted: &str) -> Result<T, String> {
    match field {
        LayerField::Value(v) => Ok(v),
        _ => Err(format!(
            "missing required field '{name}' (consulted {consulted})"
        )),
    }
}

/// `session_file` and `history_file` are project-only by layering policy: they
/// are CWD-anchored, so a global default would resolve against whatever
/// directory Omega starts in. Present in the global layer — value *or* explicit
/// null — is a fatal misconfiguration, named so the operator can relocate it.
fn reject_global_only<T>(
    field: &LayerField<T>,
    name: &str,
    global_path: &Path,
) -> Result<(), String> {
    match field {
        LayerField::Missing => Ok(()),
        _ => Err(format!(
            "{name} is project-only and may not appear in the global config {} \
             (project-only by layering policy)",
            global_path.display()
        )),
    }
}

/// Merge the `confirm` field more-restrictive-wins. The special case fires only
/// when *both* layers set a `Value`; a stronger global mode then overrides the
/// project's and pushes a notice. Equal rank keeps the project's value (its
/// judge profile, when both are `Judge`). A project `Null` clears to the default
/// `Ask` — the strongest mode, so no weakening is possible and no notice is due.
fn merge_confirm(
    project: LayerField<ConfirmMode>,
    global: LayerField<ConfirmMode>,
    notices: &mut Vec<String>,
) -> Option<ConfirmMode> {
    match (project, global) {
        (LayerField::Value(p), LayerField::Value(g)) => {
            if confirm_rank(&g) > confirm_rank(&p) {
                notices.push(format!(
                    "confirm: project '{p}' overridden by global '{g}' (more-restrictive wins)"
                ));
                Some(g)
            } else {
                Some(p)
            }
        }
        (LayerField::Value(p), _) => Some(p),
        (LayerField::Null, _) => None,
        (LayerField::Missing, LayerField::Value(g)) => Some(g),
        (LayerField::Missing, _) => None,
    }
}

impl Config {
    /// Load configuration from the global layer (`~/.omega-system/config.json`,
    /// or `None` when there is no `$HOME`) overlaid by the project layer. Returns
    /// the finalized config plus pre-built startup notices (confirm overrides and
    /// profile clamps) that the caller only prints — the covered side owns all
    /// file I/O and message construction.
    ///
    /// When the global layer is absent (no path, or `NotFound`) this delegates to
    /// the exact single-file [`Config::load`] behavior, so existing setups parse
    /// byte-identically. Every other read failure (permissions, invalid UTF-8, …)
    /// is fatal with the offending path.
    pub fn load_layers(
        global_path: Option<&Path>,
        project_path: &Path,
    ) -> Result<(Config, Vec<String>), String> {
        let Some(global_path) = global_path else {
            return Config::load_path(project_path).map(|config| (config, Vec::new()));
        };
        let Some(global_contents) = read_candidate(global_path)? else {
            return Config::load_path(project_path).map(|config| (config, Vec::new()));
        };
        let global = parse_layer(&global_contents, global_path)?;
        let (project, project_present) = match read_candidate(project_path)? {
            Some(contents) => (parse_layer(&contents, project_path)?, true),
            None => (ConfigLayer::default(), false),
        };
        merge_layers(project, global, global_path, project_path, project_present)
    }
}

/// Read a candidate layer file. `Ok(Some(_))` when it exists, `Ok(None)` only
/// for `NotFound` (the layer is absent), and `Err` for every other failure —
/// permissions, invalid UTF-8, … — naming the path so a real I/O problem fails
/// fast rather than masquerading as an absent layer.
fn read_candidate(path: &Path) -> Result<Option<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(contents) => Ok(Some(contents)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("cannot read {}: {e}", path.display())),
    }
}

/// Parse one strict layer, naming the file on a parse error to match the
/// single-file `invalid <path>` shape.
fn parse_layer(contents: &str, path: &Path) -> Result<ConfigLayer, String> {
    serde_json::from_str(contents).map_err(|e| format!("invalid {}: {e}", path.display()))
}

/// Merge the two parsed layers into a finalized [`Config`], running the shared
/// [`validate`] on the result. `project_present` distinguishes a genuinely
/// missing project file (only the global path is consulted for required-field
/// errors) from a present one.
fn merge_layers(
    project: ConfigLayer,
    global: ConfigLayer,
    global_path: &Path,
    project_path: &Path,
    project_present: bool,
) -> Result<(Config, Vec<String>), String> {
    let mut notices = Vec::new();

    reject_global_only(&global.session_file, "session_file", global_path)?;
    reject_global_only(&global.history_file, "history_file", global_path)?;

    let consulted = if project_present {
        format!("{} and {}", global_path.display(), project_path.display())
    } else {
        global_path.display().to_string()
    };

    let provider = require(
        merge_plain(project.provider, global.provider),
        "provider",
        &consulted,
    )?;
    let model = require(
        merge_plain(project.model, global.model),
        "model",
        &consulted,
    )?;
    let max_tokens = require(
        merge_plain(project.max_tokens, global.max_tokens),
        "max_tokens",
        &consulted,
    )?;
    let system = into_option(merge_plain(project.system, global.system));
    let context_token_limit = into_option(merge_plain(
        project.context_token_limit,
        global.context_token_limit,
    ))
    .unwrap_or_else(default_context_token_limit);
    let max_turns = into_option(merge_plain(project.max_turns, global.max_turns))
        .unwrap_or_else(default_max_turns);
    let session_file = into_option(merge_plain(project.session_file, global.session_file));
    let history_file = into_option(merge_plain(project.history_file, global.history_file));
    let effort = into_option(merge_plain(project.effort, global.effort));

    // `agents` merges as a whole value; provenance is simply which layer supplied
    // the list, and only a project-supplied list is subject to the confirm floor.
    let project_supplied_agents = matches!(project.agents, LayerField::Value(_));
    let mut agents = into_option(merge_plain(project.agents, global.agents)).unwrap_or_default();

    // The global top-level `confirm` value is the floor for project profile
    // overrides; capture it before the merge consumes the layer.
    let global_floor = match &global.confirm {
        LayerField::Value(g) => Some(g.clone()),
        _ => None,
    };
    let confirm = merge_confirm(project.confirm, global.confirm, &mut notices).unwrap_or_default();

    // Profile-override floor: a project-supplied profile whose `confirm` override
    // is strictly weaker than the global top-level mode is clamped up to it, so a
    // repo cannot walk around the session floor by pinning a permissive executor
    // (profile overrides beat the session policy). Global-supplied profiles are
    // the user's own and are never clamped.
    if project_supplied_agents && let Some(floor) = &global_floor {
        for agent in &mut agents {
            if let Some(over) = &agent.confirm
                && confirm_rank(floor) > confirm_rank(over)
            {
                notices.push(format!(
                    "confirm: agent profile '{}' '{over}' overridden by global '{floor}' (more-restrictive wins)",
                    agent.name
                ));
                agent.confirm = Some(floor.clone());
            }
        }
    }

    let config = Config {
        provider,
        model,
        max_tokens,
        system,
        context_token_limit,
        max_turns,
        session_file,
        history_file,
        effort,
        agents,
        confirm,
    };
    validate(&config)?;
    Ok((config, notices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_json_parses_full_config() {
        let cfg = Config::from_json(
            r#"{
                "provider": "anthropic",
                "model": "some-model",
                "max_tokens": 4096,
                "system": "be terse",
                "context_token_limit": 100000
            }"#,
        )
        .unwrap();
        assert_eq!(cfg.provider, ProviderKind::Anthropic);
        assert_eq!(cfg.model, "some-model");
        assert_eq!(cfg.max_tokens, 4096);
        assert_eq!(cfg.system.as_deref(), Some("be terse"));
        assert_eq!(cfg.context_token_limit, 100_000);
    }

    #[test]
    fn from_json_system_defaults_to_none() {
        let cfg = Config::from_json(r#"{"provider": "anthropic", "model": "m", "max_tokens": 1}"#)
            .unwrap();
        assert!(cfg.system.is_none());
    }

    #[test]
    fn from_json_session_file_defaults_to_none() {
        // Absent from existing configs — persistence stays opt-in, and the
        // serde default must fill it in rather than failing the parse.
        let cfg = Config::from_json(r#"{"provider": "anthropic", "model": "m", "max_tokens": 1}"#)
            .unwrap();
        assert!(cfg.session_file.is_none());
    }

    #[test]
    fn from_json_parses_session_file() {
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1, "session_file": "session.json"}"#,
        )
        .unwrap();
        assert_eq!(cfg.session_file.as_deref(), Some("session.json"));
    }

    #[test]
    fn from_json_history_file_defaults_to_none() {
        // Absent from existing configs — persistent history stays opt-in,
        // and the serde default must fill it in rather than failing the
        // parse.
        let cfg = Config::from_json(r#"{"provider": "anthropic", "model": "m", "max_tokens": 1}"#)
            .unwrap();
        assert!(cfg.history_file.is_none());
    }

    #[test]
    fn from_json_parses_history_file() {
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1, "history_file": ".omega_history"}"#,
        )
        .unwrap();
        assert_eq!(cfg.history_file.as_deref(), Some(".omega_history"));
    }

    #[test]
    fn from_json_effort_defaults_to_none() {
        // Absent from existing configs — effort stays opt-in, and the serde
        // default must fill it in rather than failing the parse.
        let cfg = Config::from_json(r#"{"provider": "anthropic", "model": "m", "max_tokens": 1}"#)
            .unwrap();
        assert!(cfg.effort.is_none());
    }

    #[test]
    fn from_json_parses_effort() {
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1, "effort": "high"}"#,
        )
        .unwrap();
        assert_eq!(cfg.effort.as_deref(), Some("high"));
    }

    // ── agent profiles ──

    #[test]
    fn from_json_agents_default_to_empty() {
        // Absent from existing configs — no profiles, no `task` tool, and the
        // serde default fills the field rather than failing the parse.
        let cfg = Config::from_json(r#"{"provider": "anthropic", "model": "m", "max_tokens": 1}"#)
            .unwrap();
        assert!(cfg.agents.is_empty());
    }

    #[test]
    fn from_json_parses_a_full_profile() {
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "reviewer", "provider": "openai", "model": "gpt-x",
                            "effort": "high", "system": "review", "tools": ["read_file"]}]}"#,
        )
        .unwrap();
        assert_eq!(cfg.agents.len(), 1);
        let p = &cfg.agents[0];
        assert_eq!(p.name, "reviewer");
        assert_eq!(p.provider, ProviderKind::Openai);
        assert_eq!(p.model, "gpt-x");
        assert_eq!(p.effort.as_deref(), Some("high"));
        assert_eq!(p.system.as_deref(), Some("review"));
        assert_eq!(p.tools.as_deref(), Some(&["read_file".to_string()][..]));
    }

    #[test]
    fn from_json_profile_optionals_default() {
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "r", "provider": "anthropic", "model": "m"}]}"#,
        )
        .unwrap();
        let p = &cfg.agents[0];
        assert!(p.effort.is_none() && p.system.is_none() && p.tools.is_none());
    }

    #[test]
    fn from_json_rejects_duplicate_profile_names() {
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "dup", "provider": "anthropic", "model": "a"},
                           {"name": "dup", "provider": "anthropic", "model": "b"}]}"#,
        )
        .unwrap_err();
        assert!(
            err.contains("duplicate agent profile name: dup"),
            "got: {err}"
        );
    }

    #[test]
    fn from_json_rejects_empty_profile_name() {
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "", "provider": "anthropic", "model": "m"}]}"#,
        )
        .unwrap_err();
        assert!(err.contains("empty name"), "got: {err}");
    }

    #[test]
    fn from_json_rejects_empty_profile_model() {
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "r", "provider": "anthropic", "model": ""}]}"#,
        )
        .unwrap_err();
        assert!(err.contains("empty model"), "got: {err}");
    }

    #[test]
    fn from_json_rejects_unknown_profile_provider() {
        // The provider is deserialized as ProviderKind, so an unknown name is
        // an `unknown variant` parse error before validation runs.
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "r", "provider": "gemini", "model": "m"}]}"#,
        )
        .unwrap_err();
        assert!(err.contains("unknown variant"), "got: {err}");
    }

    #[test]
    fn from_json_rejects_unknown_profile_tool() {
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "r", "provider": "anthropic", "model": "m",
                            "tools": ["read_file", "frobnicate"]}]}"#,
        )
        .unwrap_err();
        assert!(err.contains("unknown tool: frobnicate"), "got: {err}");
    }

    #[test]
    fn from_json_rejects_task_in_a_child_allowlist() {
        // No recursion: a child profile may not be granted the task tool.
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "r", "provider": "anthropic", "model": "m",
                            "tools": ["task"]}]}"#,
        )
        .unwrap_err();
        assert!(err.contains("no recursion"), "got: {err}");
    }

    #[test]
    fn from_json_accepts_an_empty_tool_allowlist() {
        // A pure-reasoning child with no tools is valid.
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "r", "provider": "anthropic", "model": "m", "tools": []}]}"#,
        )
        .unwrap();
        assert_eq!(cfg.agents[0].tools.as_deref(), Some(&[][..]));
    }

    #[test]
    fn from_json_rejects_unknown_profile_field() {
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "r", "provider": "anthropic", "model": "m", "oops": 1}]}"#,
        )
        .unwrap_err();
        assert!(err.contains("unknown field"), "got: {err}");
    }

    // ── confirmation policy ──

    #[test]
    fn from_json_confirm_defaults_to_ask() {
        // Absent from existing configs — the serde default must fill in the
        // interactive mode rather than failing the parse.
        let cfg = Config::from_json(r#"{"provider": "anthropic", "model": "m", "max_tokens": 1}"#)
            .unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Ask);
    }

    #[test]
    fn from_json_parses_confirm_ask_and_allow() {
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1, "confirm": "ask"}"#,
        )
        .unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Ask);
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1, "confirm": "allow"}"#,
        )
        .unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Allow);
    }

    #[test]
    fn from_json_parses_confirm_judge_naming_a_profile() {
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "sentinel", "provider": "anthropic", "model": "m"}],
                "confirm": {"judge": "sentinel"}}"#,
        )
        .unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Judge("sentinel".to_string()));
    }

    #[test]
    fn from_json_rejects_confirm_judge_with_unknown_profile() {
        // A typo'd judge profile would otherwise become a silent deny on the
        // first dangerous call of an unattended run — fail the load instead.
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "confirm": {"judge": "ghost"}}"#,
        )
        .unwrap_err();
        assert!(
            err.contains("confirm names unknown judge profile: ghost"),
            "got: {err}"
        );
    }

    #[test]
    fn from_json_rejects_unknown_confirm_shape() {
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1, "confirm": "yolo"}"#,
        )
        .unwrap_err();
        assert!(err.contains("unknown variant"), "got: {err}");
    }

    #[test]
    fn from_json_profile_confirm_defaults_to_none() {
        // No override — the child inherits the session policy.
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "r", "provider": "anthropic", "model": "m"}]}"#,
        )
        .unwrap();
        assert!(cfg.agents[0].confirm.is_none());
    }

    #[test]
    fn from_json_parses_profile_confirm_override() {
        // An executor pinned to the judge even when the session runs allow —
        // the precedence case. Self-reference is fine: the
        // judge makes one-shot requests and never consults a policy itself.
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "executor", "provider": "anthropic", "model": "m",
                            "confirm": {"judge": "executor"}},
                           {"name": "trusted", "provider": "anthropic", "model": "m",
                            "confirm": "allow"}]}"#,
        )
        .unwrap();
        assert_eq!(
            cfg.agents[0].confirm,
            Some(ConfirmMode::Judge("executor".to_string()))
        );
        assert_eq!(cfg.agents[1].confirm, Some(ConfirmMode::Allow));
    }

    #[test]
    fn from_json_rejects_profile_confirm_judge_with_unknown_profile() {
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "r", "provider": "anthropic", "model": "m",
                            "confirm": {"judge": "ghost"}}]}"#,
        )
        .unwrap_err();
        assert!(
            err.contains("agent profile 'r' confirm names unknown judge profile: ghost"),
            "got: {err}"
        );
    }

    #[test]
    fn confirm_mode_displays_the_command_spelling() {
        // The report line echoes what the operator would type at /confirm.
        assert_eq!(ConfirmMode::Ask.to_string(), "ask");
        assert_eq!(ConfirmMode::Allow.to_string(), "allow");
        assert_eq!(
            ConfirmMode::Judge("sentinel".to_string()).to_string(),
            "judge sentinel"
        );
    }

    #[test]
    fn from_json_rejects_zero_context_token_limit() {
        // A zero limit would make the compaction guard fire on every turn,
        // silently summarizing history away — fail the parse instead.
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1, "context_token_limit": 0}"#,
        )
        .unwrap_err();
        assert!(err.contains("context_token_limit"), "got: {err}");
    }

    #[test]
    fn from_json_context_token_limit_defaults_to_200k() {
        // Absent from existing configs — the serde default must fill it in
        // rather than failing the parse.
        let cfg = Config::from_json(r#"{"provider": "anthropic", "model": "m", "max_tokens": 1}"#)
            .unwrap();
        assert_eq!(cfg.context_token_limit, 200_000);
    }

    #[test]
    fn from_json_max_turns_defaults_to_the_loop_constant() {
        // Absent from existing configs — the serde default fills in the
        // historical round budget, preserving existing behavior.
        let cfg = Config::from_json(r#"{"provider": "anthropic", "model": "m", "max_tokens": 1}"#)
            .unwrap();
        assert_eq!(cfg.max_turns, crate::agent::MAX_TURNS);
    }

    #[test]
    fn from_json_parses_max_turns() {
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1, "max_turns": 40}"#,
        )
        .unwrap();
        assert_eq!(cfg.max_turns, 40);
    }

    #[test]
    fn from_json_rejects_zero_max_turns() {
        // A zero limit would stop the loop before its first round-trip — fail
        // the parse instead, naming the field.
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1, "max_turns": 0}"#,
        )
        .unwrap_err();
        assert!(
            err.contains("max_turns must be greater than 0"),
            "got: {err}"
        );
    }

    #[test]
    fn from_json_profile_max_turns_defaults_to_none() {
        // No override — the child inherits the session's top-level limit.
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "r", "provider": "anthropic", "model": "m"}]}"#,
        )
        .unwrap();
        assert!(cfg.agents[0].max_turns.is_none());
    }

    #[test]
    fn from_json_parses_profile_max_turns() {
        let cfg = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "surveyor", "provider": "anthropic", "model": "m",
                            "max_turns": 40}]}"#,
        )
        .unwrap();
        assert_eq!(cfg.agents[0].max_turns, Some(40));
    }

    #[test]
    fn from_json_rejects_zero_profile_max_turns() {
        // A zero per-profile override is as broken as a zero top-level one — a
        // child that could never run. Reject it, naming the profile.
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1,
                "agents": [{"name": "surveyor", "provider": "anthropic", "model": "m",
                            "max_turns": 0}]}"#,
        )
        .unwrap_err();
        assert!(
            err.contains("agent profile 'surveyor' max_turns must be greater than 0"),
            "got: {err}"
        );
    }

    #[test]
    fn from_json_rejects_unknown_provider() {
        // `gemini` is not a known ProviderKind variant (anthropic/openai are).
        let err = Config::from_json(r#"{"provider": "gemini", "model": "m", "max_tokens": 1}"#)
            .unwrap_err();
        assert!(err.contains("unknown variant"), "got: {err}");
    }

    #[test]
    fn from_json_rejects_unknown_field() {
        let err = Config::from_json(
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 1, "oops": true}"#,
        )
        .unwrap_err();
        assert!(err.contains("unknown field"), "got: {err}");
    }

    #[test]
    fn from_json_rejects_missing_required_field() {
        // `model` is absent.
        let err = Config::from_json(r#"{"provider": "anthropic", "max_tokens": 1}"#).unwrap_err();
        assert!(err.contains("missing field"), "got: {err}");
    }

    #[test]
    fn from_json_rejects_malformed_json() {
        let err = Config::from_json("{not json").unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn load_reads_and_parses_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(
            &path,
            r#"{"provider": "anthropic", "model": "m", "max_tokens": 8}"#,
        )
        .unwrap();
        let cfg = Config::load(path.to_str().unwrap()).unwrap();
        assert_eq!(cfg.model, "m");
        assert_eq!(cfg.max_tokens, 8);
    }

    #[test]
    fn load_missing_file_reports_path() {
        let err = Config::load("/no/such/config.json").unwrap_err();
        assert!(
            err.contains("cannot read /no/such/config.json"),
            "got: {err}"
        );
    }

    #[test]
    fn load_invalid_contents_reports_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, "{bad").unwrap();
        let err = Config::load(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("invalid"), "got: {err}");
    }

    // ── layered configuration ──
    //
    // Each test writes real global/project files under a tempdir and drives
    // `Config::load_layers`, the covered entry point that owns all file I/O and
    // notice construction.

    /// Write `contents` to `<dir>/<name>` and return the path.
    fn write(dir: &std::path::Path, name: &str, contents: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    const MINIMAL: &str = r#"{"provider": "anthropic", "model": "m", "max_tokens": 1}"#;

    #[test]
    fn load_layers_no_global_path_delegates_to_single_file() {
        // No `$HOME` → global layer absent → today's exact single-file load.
        let dir = tempfile::tempdir().unwrap();
        let project = write(dir.path(), "config.json", MINIMAL);
        let (cfg, notices) = Config::load_layers(None, &project).unwrap();
        assert_eq!(cfg.model, "m");
        assert!(notices.is_empty());
    }

    #[test]
    fn load_layers_global_notfound_delegates_to_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let global = dir.path().join("no_global.json");
        let project = write(dir.path(), "config.json", MINIMAL);
        let (cfg, notices) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.model, "m");
        assert!(notices.is_empty());
    }

    #[test]
    fn load_layers_no_global_and_no_project_is_fatal() {
        // Both layers absent stays today's fatal `cannot read` on the project.
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("config.json");
        let err = Config::load_layers(None, &project).unwrap_err();
        assert!(err.contains("cannot read"), "got: {err}");
    }

    #[test]
    fn load_layers_merges_required_from_global() {
        // Global supplies the required trio; project adds only an optional.
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "openai", "model": "g", "max_tokens": 7}"#,
        );
        let project = write(dir.path(), "config.json", r#"{"system": "be terse"}"#);
        let (cfg, notices) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.provider, ProviderKind::Openai);
        assert_eq!(cfg.model, "g");
        assert_eq!(cfg.max_tokens, 7);
        assert_eq!(cfg.system.as_deref(), Some("be terse"));
        assert!(notices.is_empty());
    }

    #[test]
    fn load_layers_project_value_overrides_global() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7}"#,
        );
        let project = write(dir.path(), "config.json", r#"{"model": "p"}"#);
        let (cfg, _) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.model, "p");
    }

    #[test]
    fn load_layers_project_null_clears_global_optional() {
        // Global sets `system`; project's explicit null clears it to None.
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7, "system": "persona"}"#,
        );
        let project = write(dir.path(), "config.json", r#"{"system": null}"#);
        let (cfg, _) = Config::load_layers(Some(&global), &project).unwrap();
        assert!(cfg.system.is_none());
    }

    #[test]
    fn load_layers_project_null_required_field_fails_finalization() {
        // Explicit null on a required field clears the global value, so
        // finalization reports it missing and names both consulted paths.
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7}"#,
        );
        let project = write(dir.path(), "config.json", r#"{"max_tokens": null}"#);
        let err = Config::load_layers(Some(&global), &project).unwrap_err();
        assert!(
            err.contains("missing required field 'max_tokens'"),
            "got: {err}"
        );
        assert!(
            err.contains("global.json") && err.contains("config.json"),
            "got: {err}"
        );
    }

    #[test]
    fn load_layers_missing_required_names_field_and_both_paths() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "max_tokens": 7}"#,
        );
        let project = write(dir.path(), "config.json", r#"{"max_tokens": 9}"#);
        let err = Config::load_layers(Some(&global), &project).unwrap_err();
        assert!(err.contains("missing required field 'model'"), "got: {err}");
        assert!(
            err.contains("global.json") && err.contains("config.json"),
            "got: {err}"
        );
    }

    #[test]
    fn load_layers_missing_provider_is_an_error_too() {
        // `provider` is the first required field checked — its own error path,
        // not just `model`'s, must fire.
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "global.json", r#"{"model": "m"}"#);
        let project = write(dir.path(), "config.json", r#"{"max_tokens": 9}"#);
        let err = Config::load_layers(Some(&global), &project).unwrap_err();
        assert!(
            err.contains("missing required field 'provider'"),
            "got: {err}"
        );
    }

    #[test]
    fn load_layers_project_notfound_uses_global_only() {
        // A missing project file is no longer fatal when the global supplies
        // the required fields.
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "global.json", MINIMAL);
        let project = dir.path().join("config.json");
        let (cfg, notices) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.model, "m");
        assert!(notices.is_empty());
    }

    #[test]
    fn load_layers_project_absent_missing_required_names_only_global() {
        // With no project file, only the global path is consulted.
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "max_tokens": 7}"#,
        );
        let project = dir.path().join("config.json");
        let err = Config::load_layers(Some(&global), &project).unwrap_err();
        assert!(err.contains("missing required field 'model'"), "got: {err}");
        assert!(err.contains("global.json"), "got: {err}");
        assert!(!err.contains("config.json"), "got: {err}");
    }

    #[test]
    fn load_layers_rejects_unknown_field_in_global() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7, "oops": 1}"#,
        );
        let project = write(dir.path(), "config.json", "{}");
        let err = Config::load_layers(Some(&global), &project).unwrap_err();
        assert!(
            err.contains("invalid") && err.contains("unknown field"),
            "got: {err}"
        );
    }

    #[test]
    fn load_layers_rejects_unknown_field_in_project() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "global.json", MINIMAL);
        let project = write(dir.path(), "config.json", r#"{"oops": 1}"#);
        let err = Config::load_layers(Some(&global), &project).unwrap_err();
        assert!(err.contains("unknown field"), "got: {err}");
    }

    #[test]
    fn load_layers_rejects_type_error_even_when_other_layer_overrides() {
        // The project's bad `max_tokens` type is fatal at parse even though the
        // global would have supplied a valid value — each layer typechecks alone.
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "global.json", MINIMAL);
        let project = write(dir.path(), "config.json", r#"{"max_tokens": "nope"}"#);
        let err = Config::load_layers(Some(&global), &project).unwrap_err();
        assert!(err.contains("invalid"), "got: {err}");
    }

    #[test]
    fn load_layers_rejects_session_file_in_global_value() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7, "session_file": "s.json"}"#,
        );
        let project = write(dir.path(), "config.json", "{}");
        let err = Config::load_layers(Some(&global), &project).unwrap_err();
        assert!(err.contains("session_file is project-only"), "got: {err}");
        assert!(err.contains("global.json"), "got: {err}");
    }

    #[test]
    fn load_layers_rejects_session_file_in_global_null() {
        // Even an explicit null is "present" and rejected — the three-state
        // parse makes that detectable.
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7, "session_file": null}"#,
        );
        let project = write(dir.path(), "config.json", "{}");
        let err = Config::load_layers(Some(&global), &project).unwrap_err();
        assert!(err.contains("session_file is project-only"), "got: {err}");
    }

    #[test]
    fn load_layers_rejects_history_file_in_global() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7, "history_file": "h"}"#,
        );
        let project = write(dir.path(), "config.json", "{}");
        let err = Config::load_layers(Some(&global), &project).unwrap_err();
        assert!(err.contains("history_file is project-only"), "got: {err}");
    }

    #[test]
    fn load_layers_project_only_fields_allowed_in_project() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "global.json", MINIMAL);
        let project = write(
            dir.path(),
            "config.json",
            r#"{"session_file": "s.json", "history_file": "h"}"#,
        );
        let (cfg, _) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.session_file.as_deref(), Some("s.json"));
        assert_eq!(cfg.history_file.as_deref(), Some("h"));
    }

    // ── confirm restrictiveness merge ──

    /// Build a global file whose `confirm` is `confirm_json` and a project file
    /// whose top-level body is `project_body`, then load the pair.
    fn load_confirm(
        dir: &std::path::Path,
        global_confirm: &str,
        project_body: &str,
    ) -> Result<(Config, Vec<String>), String> {
        let global = write(
            dir,
            "global.json",
            &format!(
                r#"{{"provider": "anthropic", "model": "g", "max_tokens": 7, "confirm": {global_confirm}}}"#
            ),
        );
        let project = write(dir, "config.json", project_body);
        Config::load_layers(Some(&global), &project)
    }

    #[test]
    fn confirm_global_stronger_overrides_project_with_notice() {
        // project allow < global judge → global wins, notice pinned.
        let dir = tempfile::tempdir().unwrap();
        let (cfg, notices) = load_confirm(
            dir.path(),
            r#"{"judge": "sentinel"}"#,
            r#"{"agents": [{"name": "sentinel", "provider": "anthropic", "model": "m"}], "confirm": "allow"}"#,
        )
        .unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Judge("sentinel".to_string()));
        assert_eq!(
            notices,
            vec![
                "confirm: project 'allow' overridden by global 'judge sentinel' (more-restrictive wins)"
                    .to_string()
            ]
        );
    }

    #[test]
    fn confirm_project_stronger_keeps_project_no_notice() {
        // project ask > global allow → project wins, no notice.
        let dir = tempfile::tempdir().unwrap();
        let (cfg, notices) =
            load_confirm(dir.path(), r#""allow""#, r#"{"confirm": "ask"}"#).unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Ask);
        assert!(notices.is_empty());
    }

    #[test]
    fn confirm_global_ask_overrides_project_allow() {
        // project allow < global ask → global wins, notice pinned.
        let dir = tempfile::tempdir().unwrap();
        let (cfg, notices) =
            load_confirm(dir.path(), r#""ask""#, r#"{"confirm": "allow"}"#).unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Ask);
        assert_eq!(
            notices,
            vec![
                "confirm: project 'allow' overridden by global 'ask' (more-restrictive wins)"
                    .to_string()
            ]
        );
    }

    #[test]
    fn confirm_both_judge_keeps_project_profile() {
        // Equal rank: the project's judge profile wins, no notice.
        let dir = tempfile::tempdir().unwrap();
        let (cfg, notices) = load_confirm(
            dir.path(),
            r#"{"judge": "gsentinel"}"#,
            r#"{"agents": [{"name": "psentinel", "provider": "anthropic", "model": "m"}], "confirm": {"judge": "psentinel"}}"#,
        )
        .unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Judge("psentinel".to_string()));
        assert!(notices.is_empty());
    }

    #[test]
    fn confirm_both_allow_keeps_project_no_notice() {
        // Equal rank, non-judge: project value carries, no notice.
        let dir = tempfile::tempdir().unwrap();
        let (cfg, notices) =
            load_confirm(dir.path(), r#""allow""#, r#"{"confirm": "allow"}"#).unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Allow);
        assert!(notices.is_empty());
    }

    #[test]
    fn confirm_project_only_set_merges_plain() {
        // Global unset, project allow → project value, no notice.
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "global.json", MINIMAL);
        let project = write(dir.path(), "config.json", r#"{"confirm": "allow"}"#);
        let (cfg, notices) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Allow);
        assert!(notices.is_empty());
    }

    #[test]
    fn confirm_global_only_set_merges_plain() {
        // Project unset, global allow → global value, no notice.
        let dir = tempfile::tempdir().unwrap();
        let (cfg, notices) = load_confirm(dir.path(), r#""allow""#, "{}").unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Allow);
        assert!(notices.is_empty());
    }

    #[test]
    fn confirm_project_null_clears_to_ask() {
        // Explicit null clears the global value to the default Ask, no notice.
        let dir = tempfile::tempdir().unwrap();
        let (cfg, notices) =
            load_confirm(dir.path(), r#""allow""#, r#"{"confirm": null}"#).unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Ask);
        assert!(notices.is_empty());
    }

    #[test]
    fn confirm_neither_set_defaults_to_ask() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "global.json", MINIMAL);
        let project = write(dir.path(), "config.json", "{}");
        let (cfg, _) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Ask);
    }

    // ── profile-override confirm floor ──

    #[test]
    fn profile_override_weaker_than_global_floor_is_clamped() {
        // Project supplies an executor pinned `allow`; the global top-level
        // judge floors it up to the global mode, with a notice. Project confirm
        // is `ask` so the top-level merge keeps Ask (floor still applies).
        let dir = tempfile::tempdir().unwrap();
        let (cfg, notices) = load_confirm(
            dir.path(),
            r#"{"judge": "sentinel"}"#,
            r#"{"confirm": "ask",
                "agents": [{"name": "sentinel", "provider": "anthropic", "model": "m"},
                           {"name": "executor", "provider": "anthropic", "model": "m",
                            "confirm": "allow"}]}"#,
        )
        .unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Ask);
        assert_eq!(
            cfg.agents[1].confirm,
            Some(ConfirmMode::Judge("sentinel".to_string()))
        );
        assert_eq!(
            notices,
            vec![
                "confirm: agent profile 'executor' 'allow' overridden by global 'judge sentinel' (more-restrictive wins)"
                    .to_string()
            ]
        );
    }

    #[test]
    fn profile_clamp_applies_when_project_omits_top_level_confirm() {
        // The walk-around the floor exists to stop: a repo that says nothing at
        // the top level (inheriting the global judge) but replaces `agents`
        // with a permissive executor. The clamp keys on the raw global value,
        // not on what the project's own `confirm` merged to.
        let dir = tempfile::tempdir().unwrap();
        let (cfg, notices) = load_confirm(
            dir.path(),
            r#"{"judge": "sentinel"}"#,
            r#"{"agents": [{"name": "sentinel", "provider": "anthropic", "model": "m"},
                           {"name": "executor", "provider": "anthropic", "model": "m",
                            "confirm": "allow"}]}"#,
        )
        .unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Judge("sentinel".to_string()));
        assert_eq!(
            cfg.agents[1].confirm,
            Some(ConfirmMode::Judge("sentinel".to_string()))
        );
        assert_eq!(
            notices,
            vec![
                "confirm: agent profile 'executor' 'allow' overridden by global 'judge sentinel' (more-restrictive wins)"
                    .to_string()
            ]
        );
    }

    #[test]
    fn profile_override_equal_rank_judge_is_kept() {
        // A project profile's own judge (equal rank to the global floor) is not
        // weaker, so it is kept untouched — no clamp, no notice.
        let dir = tempfile::tempdir().unwrap();
        let (cfg, notices) = load_confirm(
            dir.path(),
            r#"{"judge": "gj"}"#,
            r#"{"confirm": "ask",
                "agents": [{"name": "pj", "provider": "anthropic", "model": "m"},
                           {"name": "exec", "provider": "anthropic", "model": "m",
                            "confirm": {"judge": "pj"}}]}"#,
        )
        .unwrap();
        assert_eq!(
            cfg.agents[1].confirm,
            Some(ConfirmMode::Judge("pj".to_string()))
        );
        assert!(notices.is_empty());
    }

    #[test]
    fn global_supplied_agents_are_not_clamped() {
        // The agents list comes from the global layer (project supplies none),
        // so a weak profile override stands even under the judge floor.
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7,
                "confirm": {"judge": "j"},
                "agents": [{"name": "j", "provider": "anthropic", "model": "m"},
                           {"name": "weak", "provider": "anthropic", "model": "m",
                            "confirm": "allow"}]}"#,
        );
        let project = write(dir.path(), "config.json", "{}");
        let (cfg, notices) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.agents[1].confirm, Some(ConfirmMode::Allow));
        assert!(notices.is_empty());
    }

    #[test]
    fn profile_clamp_to_unknown_judge_fails_merged_validation() {
        // Clamping the executor up to the global `judge sentinel` yields a
        // profile naming a judge absent from the project's replaced agents —
        // merged validation then fails fast. Top-level stays a valid Ask.
        let dir = tempfile::tempdir().unwrap();
        let err = load_confirm(
            dir.path(),
            r#"{"judge": "sentinel"}"#,
            r#"{"confirm": "ask",
                "agents": [{"name": "executor", "provider": "anthropic", "model": "m",
                            "confirm": "allow"}]}"#,
        )
        .unwrap_err();
        assert!(
            err.contains("agent profile 'executor' confirm names unknown judge profile: sentinel"),
            "got: {err}"
        );
    }

    // ── agents whole-value merge ──

    #[test]
    fn agents_project_value_replaces_global_list() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7,
                "agents": [{"name": "a", "provider": "anthropic", "model": "m"},
                           {"name": "b", "provider": "anthropic", "model": "m"}]}"#,
        );
        let project = write(
            dir.path(),
            "config.json",
            r#"{"agents": [{"name": "c", "provider": "anthropic", "model": "m"}]}"#,
        );
        let (cfg, _) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.agents.len(), 1);
        assert_eq!(cfg.agents[0].name, "c");
    }

    #[test]
    fn agents_project_null_clears_to_empty() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7,
                "agents": [{"name": "a", "provider": "anthropic", "model": "m"}]}"#,
        );
        let project = write(dir.path(), "config.json", r#"{"agents": null}"#);
        let (cfg, _) = Config::load_layers(Some(&global), &project).unwrap();
        assert!(cfg.agents.is_empty());
    }

    #[test]
    fn agents_inherited_from_global_when_project_missing() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7,
                "agents": [{"name": "a", "provider": "anthropic", "model": "m"}]}"#,
        );
        let project = write(dir.path(), "config.json", "{}");
        let (cfg, _) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.agents.len(), 1);
        assert_eq!(cfg.agents[0].name, "a");
    }

    // ── merged validation ──

    #[test]
    fn merged_global_judge_resolves_against_project_agents() {
        // The global `confirm` names a profile that only the project supplies —
        // merged validation resolves it against the merged table.
        let dir = tempfile::tempdir().unwrap();
        let (cfg, _) = load_confirm(
            dir.path(),
            r#"{"judge": "psentinel"}"#,
            r#"{"agents": [{"name": "psentinel", "provider": "anthropic", "model": "m"}]}"#,
        )
        .unwrap();
        assert_eq!(cfg.confirm, ConfirmMode::Judge("psentinel".to_string()));
    }

    #[test]
    fn merged_global_judge_unknown_profile_fails() {
        let dir = tempfile::tempdir().unwrap();
        let err = load_confirm(dir.path(), r#"{"judge": "ghost"}"#, "{}").unwrap_err();
        assert!(
            err.contains("confirm names unknown judge profile: ghost"),
            "got: {err}"
        );
    }

    #[test]
    fn merged_zero_context_token_limit_fails() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "global.json", MINIMAL);
        let project = write(dir.path(), "config.json", r#"{"context_token_limit": 0}"#);
        let err = Config::load_layers(Some(&global), &project).unwrap_err();
        assert!(err.contains("context_token_limit"), "got: {err}");
    }

    #[test]
    fn load_layers_context_token_limit_defaults_when_unset() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "global.json", MINIMAL);
        let project = write(dir.path(), "config.json", "{}");
        let (cfg, _) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.context_token_limit, 200_000);
    }

    #[test]
    fn merged_zero_max_turns_fails() {
        // The zero-rejection runs on the merged config too, so a project that
        // sets `max_turns: 0` fails the layered path the same way.
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "global.json", MINIMAL);
        let project = write(dir.path(), "config.json", r#"{"max_turns": 0}"#);
        let err = Config::load_layers(Some(&global), &project).unwrap_err();
        assert!(
            err.contains("max_turns must be greater than 0"),
            "got: {err}"
        );
    }

    #[test]
    fn merged_zero_profile_max_turns_fails() {
        // A project profile pinning `max_turns: 0` fails merged validation,
        // naming the profile.
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "global.json", MINIMAL);
        let project = write(
            dir.path(),
            "config.json",
            r#"{"agents": [{"name": "surveyor", "provider": "anthropic", "model": "m",
                           "max_turns": 0}]}"#,
        );
        let err = Config::load_layers(Some(&global), &project).unwrap_err();
        assert!(
            err.contains("agent profile 'surveyor' max_turns must be greater than 0"),
            "got: {err}"
        );
    }

    #[test]
    fn load_layers_max_turns_defaults_when_unset() {
        // Neither layer sets it → the loop constant, preserving existing behavior.
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "global.json", MINIMAL);
        let project = write(dir.path(), "config.json", "{}");
        let (cfg, _) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.max_turns, crate::agent::MAX_TURNS);
    }

    #[test]
    fn load_layers_allows_max_turns_in_global() {
        // Unlike `session_file`, `max_turns` is a legitimate global default —
        // present in the global layer, it flows through and the project
        // inherits it (no `reject_global_only`).
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7, "max_turns": 25}"#,
        );
        let project = write(dir.path(), "config.json", "{}");
        let (cfg, _) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.max_turns, 25);
    }

    #[test]
    fn load_layers_project_max_turns_overrides_global() {
        // Value in both layers → the project's plain override wins (no
        // restrictiveness floor — an economics knob, not a confirmation mode).
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7, "max_turns": 25}"#,
        );
        let project = write(dir.path(), "config.json", r#"{"max_turns": 40}"#);
        let (cfg, _) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.max_turns, 40);
    }

    #[test]
    fn load_layers_project_null_max_turns_clears_to_default() {
        // Global sets it; the project's explicit null clears it, so the
        // finalized value falls back to the default — the Null merge arm.
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7, "max_turns": 25}"#,
        );
        let project = write(dir.path(), "config.json", r#"{"max_turns": null}"#);
        let (cfg, _) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.max_turns, crate::agent::MAX_TURNS);
    }

    #[test]
    fn load_layers_merges_context_token_limit_and_effort() {
        // Cover the remaining plain-field merges end to end.
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.json",
            r#"{"provider": "anthropic", "model": "g", "max_tokens": 7,
                "context_token_limit": 4096, "effort": "high"}"#,
        );
        let project = write(dir.path(), "config.json", "{}");
        let (cfg, _) = Config::load_layers(Some(&global), &project).unwrap();
        assert_eq!(cfg.context_token_limit, 4096);
        assert_eq!(cfg.effort.as_deref(), Some("high"));
    }

    #[test]
    #[cfg(unix)]
    fn load_layers_global_read_error_is_fatal() {
        // A non-NotFound read failure (permission denied) is fatal with the
        // path, never silently treated as an absent layer.
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "global.json", MINIMAL);
        std::fs::set_permissions(&global, std::fs::Permissions::from_mode(0o000)).unwrap();
        let project = write(dir.path(), "config.json", MINIMAL);
        let result = Config::load_layers(Some(&global), &project);
        // Restore permissions so the tempdir can be cleaned up.
        std::fs::set_permissions(&global, std::fs::Permissions::from_mode(0o644)).unwrap();
        let err = result.unwrap_err();
        assert!(
            err.contains("cannot read") && err.contains("global.json"),
            "got: {err}"
        );
    }
}
