//! REPL input classification.
//!
//! The REPL loop ([`crate::repl::run_repl`]) is a thin shell: it reads a
//! line, hands it to [`Command::parse`], and matches on the result. Keeping
//! the classification here — rather than as a chain of string comparisons
//! inside the loop — lets it carry full unit coverage on its own.

use crate::config::ConfirmMode;
use crate::provider::ProviderKind;

/// A single line of REPL input, classified into the action the loop takes.
///
/// Surrounding whitespace is trimmed before classification, so a blank or
/// whitespace-only line is [`Command::Empty`]. The recognized commands are
/// `/quit` (also `/exit`), `/clear`, `/confirm` (optionally with a mode),
/// `/effort` (optionally with a value), and `/model` (optionally with a
/// [`ModelSpec`] argument). Anything else — including unrecognized slash input
/// like `/foo`, or a known command word followed by an unexpected argument
/// like `/quit now` — is a [`Command::Prompt`] sent verbatim to the model.
///
/// `/agents` is also recognized, bare only — a report of the configured
/// subagent profiles. `/save <name>` and `/load <name>` take a name argument
/// (bare forms report usage); `/sessions` is bare only, listing the project's
/// saves.
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    /// A blank line — re-prompt without doing anything.
    Empty,
    /// `/quit` (or its alias `/exit`) — exit the REPL.
    Quit,
    /// `/clear` — reset the conversation history.
    Clear,
    /// `/agents` — list the configured subagent profiles. Bare only; with an
    /// argument it falls through to [`Command::Prompt`] like the other
    /// no-argument commands. A pure report of operator-facing configuration —
    /// the profile names, backends, tool allowlists, and safety flags the
    /// model-facing `task` description does not surface.
    Agents,
    /// `/save <name>` — write the current conversation to the named session
    /// store. `None` (bare `/save`) reports usage; `Some(name)` names the save.
    /// The name is unvalidated here — the store rejects a bad one (path
    /// separators, a leading dot) when the write is attempted.
    Save(Option<String>),
    /// `/load <name>` — restore a named session, replacing the live
    /// conversation. `None` (bare `/load`) reports usage; `Some(name)` names
    /// the save. An unknown name fails soft at the REPL, listing what exists.
    Load(Option<String>),
    /// `/sessions` — list the project's saved sessions, newest first. Bare;
    /// only the `delete` sub-word is an argument (see [`Command::SessionsDelete`]),
    /// anything else falls through to [`Command::Prompt`].
    Sessions,
    /// `/sessions delete <name>` — remove a named save. `None` (bare `/sessions
    /// delete`) reports usage, like bare `/save`; `Some(name)` names the save
    /// to delete. The name is unvalidated here — the store rejects a bad one
    /// when the delete is attempted, and an unknown name fails soft at the REPL.
    SessionsDelete(Option<String>),
    /// `/model` views or switches the active model — and, with a
    /// `<provider>:` prefix, the provider. `None` (bare `/model`) reports the
    /// current provider + model; `Some(spec)` switches for subsequent turns.
    Model(Option<ModelSpec>),
    /// `/model <provider>:` — a known provider name with nothing after the
    /// colon. Lists that provider's models, so the operator can see what a
    /// switch could target before committing to one.
    Models(ProviderKind),
    /// `/effort` views or sets the reasoning effort. `None` (bare `/effort`)
    /// reports the current value; `Some(value)` sets it for subsequent turns.
    /// The value is opaque and unvalidated here, like a model id — the
    /// provider rejects a bad one at request time.
    Effort(Option<String>),
    /// `/confirm` views or switches the confirmation mode. `None` (bare
    /// `/confirm`) reports the current mode; `Some(mode)` switches it for
    /// subsequent turns. Unlike an effort value, the modes are a closed set,
    /// so only `ask`, `allow`, and `judge <profile>` classify as the command
    /// — anything else falls through to a prompt like any unrecognized
    /// argument. The judge's profile name is validated by the REPL against
    /// the configured table, not here (parsing stays table-free).
    Confirm(Option<ConfirmMode>),
    /// Any other non-empty input — a prompt for the model. Carries the trimmed
    /// line.
    Prompt(String),
}

/// Every command word [`Command::parse`] recognizes, sorted, as the REPL's
/// Tab-completion candidates. Kept in step with the `parse` arms by hand —
/// the sync test below parses each entry and asserts it classifies as a
/// command, not a prompt.
pub const COMMANDS: &[&str] = &[
    "/agents",
    "/clear",
    "/confirm",
    "/effort",
    "/exit",
    "/load",
    "/model",
    "/quit",
    "/save",
    "/sessions",
];

/// The argument of a `/model <arg>` switch: a model id, optionally prefixed
/// with the provider to switch to (`/model openai:gpt-4o`).
#[derive(Debug, PartialEq, Eq)]
pub struct ModelSpec {
    /// `Some` only when the argument had a `<name>:` prefix naming a known
    /// provider with a model after the colon. Every other shape — no colon,
    /// an unknown or misspelled prefix (`opneai:gpt-4o`), a colon-bearing
    /// model id (`ft:gpt-4o-mini-…:org:…`) — leaves this `None` and the
    /// whole argument in `model`. (A known provider with a *trailing* colon
    /// never reaches here: `Command::parse` classifies it as
    /// [`Command::Models`] first.)
    pub provider: Option<ProviderKind>,
    /// The model id to switch to. Unvalidated, like `/model` has always
    /// been: a bad id fails at request time with the provider's own error.
    pub model: String,
}

impl ModelSpec {
    /// Split a non-empty `/model` argument into an optional provider prefix
    /// and a model id. The prefix is a provider switch only when it names a
    /// known provider ([`ProviderKind::from_name`]) *and* a model follows the
    /// colon; anything else is a bare model id, taken verbatim. Deliberately
    /// no "unknown provider" error: model ids can legitimately carry colons
    /// (OpenAI fine-tune ids), so a non-provider prefix must stay part of the
    /// id — a typo'd provider then fails at request time with the full string
    /// in the provider's error, the same fail-fast contract as any bad id.
    fn parse(arg: &str) -> ModelSpec {
        if let Some((prefix, model)) = arg.split_once(':')
            && !model.is_empty()
            && let Some(provider) = ProviderKind::from_name(prefix)
        {
            return ModelSpec {
                provider: Some(provider),
                model: model.to_string(),
            };
        }
        ModelSpec {
            provider: None,
            model: arg.to_string(),
        }
    }
}

impl Command {
    /// Classify a raw input line, trimming surrounding whitespace first.
    ///
    /// Input is split once on the first run of whitespace into a command word
    /// and the rest (itself trimmed), so a command and its argument are matched
    /// uniformly regardless of how they are spaced. A known command word with an
    /// argument it does not accept (e.g. `/quit now`) falls through to
    /// [`Command::Prompt`] rather than being silently treated as the command.
    pub fn parse(line: &str) -> Command {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Command::Empty;
        }
        let (head, arg) = match trimmed.split_once(char::is_whitespace) {
            Some((head, rest)) => (head, rest.trim()),
            None => (trimmed, ""),
        };
        match (head, arg) {
            ("/quit", "") | ("/exit", "") => Command::Quit,
            ("/clear", "") => Command::Clear,
            ("/agents", "") => Command::Agents,
            ("/sessions", "") => Command::Sessions,
            // Bare `/sessions delete` reports usage, like bare `/save`; with a
            // name it deletes. Anything else after `/sessions` (`/sessions
            // all`) falls through to a prompt like the other bare commands.
            ("/sessions", "delete") => Command::SessionsDelete(None),
            ("/sessions", arg) => match arg.split_once(char::is_whitespace) {
                Some(("delete", name)) => Command::SessionsDelete(Some(name.trim().to_string())),
                _ => Command::Prompt(trimmed.to_string()),
            },
            ("/save", "") => Command::Save(None),
            ("/save", name) => Command::Save(Some(name.to_string())),
            ("/load", "") => Command::Load(None),
            ("/load", name) => Command::Load(Some(name.to_string())),
            ("/effort", "") => Command::Effort(None),
            ("/effort", arg) => Command::Effort(Some(arg.to_string())),
            ("/confirm", "") => Command::Confirm(None),
            ("/confirm", "ask") => Command::Confirm(Some(ConfirmMode::Ask)),
            ("/confirm", "allow") => Command::Confirm(Some(ConfirmMode::Allow)),
            // `judge` needs a profile name — the rest of the argument,
            // verbatim (profile names are free-form, like an effort value).
            // A bare `/confirm judge` or an unknown mode falls through to a
            // prompt, the same shape as any command with an argument it does
            // not accept.
            ("/confirm", arg) => match arg.split_once(char::is_whitespace) {
                Some(("judge", profile)) => {
                    Command::Confirm(Some(ConfirmMode::Judge(profile.trim().to_string())))
                }
                _ => Command::Prompt(trimmed.to_string()),
            },
            ("/model", "") => Command::Model(None),
            // A known provider name with a trailing colon and no model id is
            // a listing request, not a switch — the shape an operator types
            // when they know the vendor but not its model ids. An unknown
            // prefix keeps the bare-model-id fallthrough (`ft:` ids).
            ("/model", arg) => match arg.strip_suffix(':').and_then(ProviderKind::from_name) {
                Some(kind) => Command::Models(kind),
                None => Command::Model(Some(ModelSpec::parse(arg))),
            },
            _ => Command::Prompt(trimmed.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_line() {
        assert_eq!(Command::parse(""), Command::Empty);
    }

    #[test]
    fn parse_whitespace_only_is_empty() {
        assert_eq!(Command::parse("   \t  "), Command::Empty);
        assert_eq!(Command::parse("\n"), Command::Empty);
    }

    #[test]
    fn parse_quit() {
        assert_eq!(Command::parse("/quit"), Command::Quit);
    }

    #[test]
    fn parse_exit() {
        // `/exit` is an alias for `/quit` — both classify as Command::Quit.
        assert_eq!(Command::parse("/exit"), Command::Quit);
    }

    #[test]
    fn parse_clear() {
        assert_eq!(Command::parse("/clear"), Command::Clear);
    }

    #[test]
    fn parse_agents() {
        assert_eq!(Command::parse("/agents"), Command::Agents);
    }

    #[test]
    fn parse_agents_with_arg_is_a_prompt() {
        // Bare only — an argument falls through to a prompt like the other
        // no-argument commands.
        assert_eq!(
            Command::parse("/agents reviewer"),
            Command::Prompt("/agents reviewer".to_string())
        );
    }

    #[test]
    fn parse_save_with_name() {
        assert_eq!(
            Command::parse("/save before-refactor"),
            Command::Save(Some("before-refactor".to_string()))
        );
        // Split on the first whitespace run and trimmed, like every argument.
        assert_eq!(
            Command::parse("/save   snap  "),
            Command::Save(Some("snap".to_string()))
        );
    }

    #[test]
    fn parse_bare_save_reports_usage() {
        // Bare `/save` carries no name — the None variant the REPL answers with
        // a usage line rather than saving.
        assert_eq!(Command::parse("/save"), Command::Save(None));
    }

    #[test]
    fn parse_load_with_name() {
        assert_eq!(
            Command::parse("/load before-refactor"),
            Command::Load(Some("before-refactor".to_string()))
        );
    }

    #[test]
    fn parse_bare_load_reports_usage() {
        assert_eq!(Command::parse("/load"), Command::Load(None));
    }

    #[test]
    fn parse_sessions() {
        assert_eq!(Command::parse("/sessions"), Command::Sessions);
    }

    #[test]
    fn parse_sessions_with_arg_is_a_prompt() {
        // Only the `delete` sub-word is an argument — any other falls through
        // to a prompt like `/agents`.
        assert_eq!(
            Command::parse("/sessions all"),
            Command::Prompt("/sessions all".to_string())
        );
    }

    #[test]
    fn parse_sessions_delete_with_name() {
        assert_eq!(
            Command::parse("/sessions delete old-experiment"),
            Command::SessionsDelete(Some("old-experiment".to_string()))
        );
        // Split on the first whitespace run and trimmed, like every argument.
        assert_eq!(
            Command::parse("/sessions   delete   snap  "),
            Command::SessionsDelete(Some("snap".to_string()))
        );
    }

    #[test]
    fn parse_bare_sessions_delete_reports_usage() {
        // Bare `/sessions delete` carries no name — the None variant the REPL
        // answers with a usage line, like bare `/save`. Trailing whitespace is
        // trimmed off first, so `delete   ` is still bare.
        assert_eq!(
            Command::parse("/sessions delete"),
            Command::SessionsDelete(None)
        );
        assert_eq!(
            Command::parse("/sessions delete   "),
            Command::SessionsDelete(None)
        );
    }

    #[test]
    fn parse_sessions_unknown_subcommand_is_a_prompt() {
        // A non-`delete` sub-word is not a command — it falls through verbatim.
        assert_eq!(
            Command::parse("/sessions purge old"),
            Command::Prompt("/sessions purge old".to_string())
        );
    }

    #[test]
    fn parse_model_no_arg_views() {
        assert_eq!(Command::parse("/model"), Command::Model(None));
    }

    /// Shorthand for the expected `/model <arg>` classification.
    fn model(provider: Option<ProviderKind>, model: &str) -> Command {
        Command::Model(Some(ModelSpec {
            provider,
            model: model.to_string(),
        }))
    }

    #[test]
    fn parse_model_with_arg_switches() {
        assert_eq!(Command::parse("/model gpt-4o"), model(None, "gpt-4o"));
    }

    #[test]
    fn parse_model_arg_is_trimmed_and_whitespace_agnostic() {
        // The argument is split off the first whitespace run and trimmed, so any
        // spacing — extra spaces or a tab — yields the same model id.
        assert_eq!(
            Command::parse("/model   claude-sonnet-4-6  "),
            model(None, "claude-sonnet-4-6")
        );
        assert_eq!(
            Command::parse("/model\tclaude-sonnet-4-6"),
            model(None, "claude-sonnet-4-6")
        );
    }

    #[test]
    fn parse_model_with_provider_prefix_switches_provider() {
        assert_eq!(
            Command::parse("/model openai:gpt-4o"),
            model(Some(ProviderKind::Openai), "gpt-4o")
        );
        assert_eq!(
            Command::parse("/model anthropic:claude-sonnet-4-6"),
            model(Some(ProviderKind::Anthropic), "claude-sonnet-4-6")
        );
    }

    #[test]
    fn parse_model_splits_on_the_first_colon_only() {
        // Only the first colon can end a provider prefix; the model id keeps
        // any later ones.
        assert_eq!(
            Command::parse("/model anthropic:claude:latest"),
            model(Some(ProviderKind::Anthropic), "claude:latest")
        );
    }

    #[test]
    fn parse_model_unknown_prefix_is_part_of_the_model_id() {
        // Colon-bearing model ids exist (OpenAI fine-tune ids), so a prefix
        // that is not a known provider is not an error — the whole argument
        // is the model id, and a genuinely bad one fails at request time.
        assert_eq!(
            Command::parse("/model ft:gpt-4o-mini-2024-07-18:org:ckpt"),
            model(None, "ft:gpt-4o-mini-2024-07-18:org:ckpt")
        );
        // A typo'd provider takes the same path: fail-fast at request time
        // with the full string in the provider's error.
        assert_eq!(
            Command::parse("/model opneai:gpt-4o"),
            model(None, "opneai:gpt-4o")
        );
        // Case-sensitive: only the config.json spelling names a provider.
        assert_eq!(
            Command::parse("/model Openai:gpt-4o"),
            model(None, "Openai:gpt-4o")
        );
    }

    #[test]
    fn parse_model_provider_with_trailing_colon_lists_its_models() {
        // The shape an operator types when they know the vendor but not its
        // ids — a listing request, never a literal model id (which used to
        // 404 at request time).
        assert_eq!(
            Command::parse("/model openai:"),
            Command::Models(ProviderKind::Openai)
        );
        assert_eq!(
            Command::parse("/model anthropic:"),
            Command::Models(ProviderKind::Anthropic)
        );
    }

    #[test]
    fn parse_model_degenerate_colons_are_a_bare_model_id() {
        // A leading colon has no provider before it, a lone colon has
        // neither half, and a trailing colon after a non-provider is not a
        // listing — none is a switch, all pass through verbatim to fail at
        // request time. A trailing colon *inside* a switch stays part of the
        // model id (only the whole argument can be a listing request).
        assert_eq!(Command::parse("/model :gpt-4o"), model(None, ":gpt-4o"));
        assert_eq!(Command::parse("/model :"), model(None, ":"));
        assert_eq!(Command::parse("/model opneai:"), model(None, "opneai:"));
        assert_eq!(
            Command::parse("/model anthropic:claude:"),
            model(Some(ProviderKind::Anthropic), "claude:")
        );
    }

    #[test]
    fn parse_effort_no_arg_views() {
        // Bare `/effort` reports the current value rather than setting one.
        assert_eq!(Command::parse("/effort"), Command::Effort(None));
    }

    #[test]
    fn parse_effort_with_arg_sets() {
        assert_eq!(
            Command::parse("/effort high"),
            Command::Effort(Some("high".to_string()))
        );
    }

    #[test]
    fn parse_effort_arg_is_trimmed_and_whitespace_agnostic() {
        // The value is split off the first whitespace run and trimmed, exactly
        // like a `/model` argument — the value stays opaque, unvalidated here.
        assert_eq!(
            Command::parse("/effort   xhigh  "),
            Command::Effort(Some("xhigh".to_string()))
        );
        assert_eq!(
            Command::parse("/effort\tnone"),
            Command::Effort(Some("none".to_string()))
        );
    }

    #[test]
    fn parse_effort_keeps_a_multi_word_value_verbatim() {
        // Only the first whitespace run splits head from arg; the rest — even
        // with internal spaces — is the opaque value, taken as-is.
        assert_eq!(
            Command::parse("/effort a b"),
            Command::Effort(Some("a b".to_string()))
        );
    }

    #[test]
    fn parse_confirm_no_arg_views() {
        // Bare `/confirm` reports the current mode rather than setting one.
        assert_eq!(Command::parse("/confirm"), Command::Confirm(None));
    }

    #[test]
    fn parse_confirm_ask_and_allow_switch() {
        assert_eq!(
            Command::parse("/confirm ask"),
            Command::Confirm(Some(ConfirmMode::Ask))
        );
        assert_eq!(
            Command::parse("/confirm allow"),
            Command::Confirm(Some(ConfirmMode::Allow))
        );
    }

    #[test]
    fn parse_confirm_judge_takes_the_profile_verbatim() {
        // The profile name is the rest of the argument, trimmed — free-form
        // like an effort value, validated by the REPL against the table.
        assert_eq!(
            Command::parse("/confirm judge sentinel"),
            Command::Confirm(Some(ConfirmMode::Judge("sentinel".to_string())))
        );
        assert_eq!(
            Command::parse("/confirm judge   spaced name  "),
            Command::Confirm(Some(ConfirmMode::Judge("spaced name".to_string())))
        );
    }

    #[test]
    fn parse_confirm_unknown_or_bare_judge_is_a_prompt() {
        // The modes are a closed set: an unknown mode, a bare `judge` with no
        // profile, and a trailing argument on ask/allow all fall through to a
        // prompt rather than being silently treated as a switch.
        assert_eq!(
            Command::parse("/confirm yolo"),
            Command::Prompt("/confirm yolo".to_string())
        );
        assert_eq!(
            Command::parse("/confirm judge"),
            Command::Prompt("/confirm judge".to_string())
        );
        assert_eq!(
            Command::parse("/confirm ask now"),
            Command::Prompt("/confirm ask now".to_string())
        );
    }

    #[test]
    fn parse_trims_before_matching_commands() {
        assert_eq!(Command::parse("  /quit  "), Command::Quit);
        assert_eq!(Command::parse("\t/clear\n"), Command::Clear);
        assert_eq!(Command::parse("  /model  "), Command::Model(None));
    }

    #[test]
    fn parse_plain_text_is_a_prompt() {
        assert_eq!(
            Command::parse("hello there"),
            Command::Prompt("hello there".to_string())
        );
    }

    #[test]
    fn parse_prompt_is_trimmed() {
        assert_eq!(
            Command::parse("  hello world  "),
            Command::Prompt("hello world".to_string())
        );
    }

    #[test]
    fn parse_unrecognized_slash_input_is_a_prompt() {
        // Only the known command words are commands; anything else is sent to
        // the model verbatim (preserving the ability to prompt with a slash).
        assert_eq!(Command::parse("/foo"), Command::Prompt("/foo".to_string()));
        assert_eq!(
            Command::parse("/foo bar"),
            Command::Prompt("/foo bar".to_string())
        );
    }

    #[test]
    fn commands_const_stays_in_step_with_parse() {
        // Every completion candidate must classify as a real command — a word
        // listed here but not matched in `parse` would Tab-complete into a
        // line that gets sent to the model as a prompt.
        for word in COMMANDS {
            let parsed = Command::parse(word);
            assert!(
                !matches!(parsed, Command::Prompt(_) | Command::Empty),
                "{word} completes but parses to {parsed:?}"
            );
        }
        // Sorted: completion lists candidates in this order.
        assert!(COMMANDS.is_sorted());
    }

    #[test]
    fn parse_known_command_with_unexpected_arg_is_a_prompt() {
        // A no-argument command followed by text is not that command — it falls
        // through to a prompt rather than swallowing the extra words.
        assert_eq!(
            Command::parse("/quit now"),
            Command::Prompt("/quit now".to_string())
        );
        assert_eq!(
            Command::parse("/exit now"),
            Command::Prompt("/exit now".to_string())
        );
        assert_eq!(
            Command::parse("/clear all"),
            Command::Prompt("/clear all".to_string())
        );
    }
}
