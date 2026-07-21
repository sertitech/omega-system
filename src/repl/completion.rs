//! Tab-completion candidates for an in-progress REPL line.
//!
//! Pure classification over the buffer text — the one dynamic input, a
//! provider's model ids, comes through an injected source so the engine
//! stays hermetic and the lazy fetch fires only when a model id can
//! actually be completed. The sources, closed sets first, live last:
//! command words ([`crate::command::COMMANDS`]), provider names
//! ([`ProviderKind::ALL`], as `<name>:` prefixes for `/model`), and the
//! per-provider model ids.

use crate::command::COMMANDS;
use crate::provider::ProviderKind;

/// The candidates for one Tab press: every valid replacement for
/// `line[start..]`, in listing order. Empty when nothing completes.
#[derive(Debug, PartialEq, Eq)]
pub struct Completion {
    /// Byte index into the queried line where the completed token starts.
    pub start: usize,
    /// The full replacement tokens. One candidate is a completion; several
    /// are listed for the operator to disambiguate by typing more.
    pub candidates: Vec<String>,
}

impl Completion {
    /// No candidates. `start` points at the line's end so a (never occurring)
    /// replacement would be a no-op append.
    fn none(line: &str) -> Completion {
        Completion {
            start: line.len(),
            candidates: Vec::new(),
        }
    }
}

/// Compute the Tab candidates for `line`, the buffer up to the cursor (the
/// editor keeps the cursor at the end of the line, so this is the whole
/// buffer). Only slash-command lines complete — free prompt text, a
/// non-`/model` argument, or a second `/model` argument yields nothing.
/// Unlike [`crate::command::Command::parse`], no leading whitespace is
/// tolerated: completion describes the line as typed, not as it will parse.
///
/// `models` supplies the queried provider's cached ids (the REPL's
/// per-provider cache over [6b's
/// `Agent::list_models_cached`](crate::agent::Agent::list_models_cached))
/// and is invoked only when the token being completed can be a model id, so
/// completing a command word never triggers the lazy fetch. A bare token
/// draws from the *active* provider (`provider`); a `<name>:` prefix draws
/// from the named one, whichever it is.
pub fn complete(
    line: &str,
    provider: ProviderKind,
    models: &mut dyn FnMut(ProviderKind) -> Vec<String>,
) -> Completion {
    if !line.starts_with('/') {
        return Completion::none(line);
    }
    // No whitespace yet: the command word itself is being completed.
    let Some(word_end) = line.find(char::is_whitespace) else {
        let candidates = COMMANDS
            .iter()
            .filter(|word| word.starts_with(line))
            .map(|word| word.to_string())
            .collect();
        return Completion {
            start: 0,
            candidates,
        };
    };
    // Only /model takes a completable argument, and only one of it.
    let arg_start = line.len() - line[word_end..].trim_start().len();
    let arg = &line[arg_start..];
    if &line[..word_end] != "/model" || arg.contains(char::is_whitespace) {
        return Completion::none(line);
    }
    // A known-provider prefix scopes completion to that provider's model ids
    // — any provider's, not just the active one; the source fetches and
    // caches per kind. An unknown prefix is not a provider at all (it may be
    // a colon-bearing model id, mirroring `ModelSpec::parse`) and falls
    // through to the bare-token pool.
    if let Some((prefix, partial)) = arg.split_once(':')
        && let Some(kind) = ProviderKind::from_name(prefix)
    {
        return Completion {
            start: arg_start + prefix.len() + 1,
            candidates: matching(models(kind), partial),
        };
    }
    // A bare token: either a provider prefix ("anthropic:") or one of the
    // active provider's model ids can continue it.
    let mut candidates: Vec<String> = ProviderKind::ALL
        .iter()
        .map(|kind| format!("{kind}:"))
        .filter(|name| name.starts_with(arg))
        .collect();
    candidates.extend(matching(models(provider), arg));
    Completion {
        start: arg_start,
        candidates,
    }
}

/// The ids that start with the typed partial, in the provider's listing order.
fn matching(ids: Vec<String>, partial: &str) -> Vec<String> {
    ids.into_iter()
        .filter(|id| id.starts_with(partial))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A models source that counts its calls: command-word completion must
    /// never trigger it (the fetch is lazy and can hit the network),
    /// argument completion must. Per-provider ids make cross-provider
    /// queries observable: the candidates reveal which kind was fetched.
    fn counted_models(count: &Cell<usize>) -> impl FnMut(ProviderKind) -> Vec<String> + '_ {
        move |kind| {
            count.set(count.get() + 1);
            match kind {
                ProviderKind::Anthropic => ["claude-sonnet-4-6", "claude-opus-4-1"],
                ProviderKind::Openai => ["gpt-4o", "gpt-4o-mini"],
            }
            .map(String::from)
            .to_vec()
        }
    }

    /// Run [`complete`] with the counted test ids and an Anthropic session,
    /// returning the result and how often the model source fired.
    fn complete_counting(line: &str) -> (Completion, usize) {
        complete_counting_as(line, ProviderKind::Anthropic)
    }

    fn complete_counting_as(line: &str, provider: ProviderKind) -> (Completion, usize) {
        let count = Cell::new(0);
        let completion = complete(line, provider, &mut counted_models(&count));
        (completion, count.get())
    }

    /// Shorthand for the expected candidates.
    fn candidates(start: usize, candidates: &[&str]) -> Completion {
        Completion {
            start,
            candidates: candidates.iter().map(|c| c.to_string()).collect(),
        }
    }

    #[test]
    fn free_prompt_text_has_no_candidates() {
        // Prompt text is sent to the model verbatim — nothing to complete.
        // The empty line is the same case: Tab before typing does nothing.
        for line in ["hello there", "", "model cl", " /mo"] {
            let (completion, calls) = complete_counting(line);
            assert!(completion.candidates.is_empty(), "{line:?} completed");
            assert_eq!(calls, 0, "{line:?} fetched models");
        }
    }

    #[test]
    fn command_words_complete_from_the_commands_list() {
        let (completion, calls) = complete_counting("/q");
        assert_eq!(completion, candidates(0, &["/quit"]));
        // The lazy model fetch must not fire for a command word.
        assert_eq!(calls, 0);
        let (completion, _) = complete_counting("/");
        assert_eq!(
            completion,
            candidates(
                0,
                &[
                    "/agents",
                    "/clear",
                    "/confirm",
                    "/effort",
                    "/exit",
                    "/load",
                    "/model",
                    "/quit",
                    "/save",
                    "/sessions"
                ]
            )
        );
        // An exact word is its own single candidate — completing it is a no-op.
        let (completion, _) = complete_counting("/model");
        assert_eq!(completion, candidates(0, &["/model"]));
        // An unknown word has no candidates.
        let (completion, _) = complete_counting("/x");
        assert_eq!(completion, candidates(0, &[]));
    }

    #[test]
    fn model_bare_argument_offers_providers_and_model_ids() {
        // Both continuations are valid: a provider prefix for a switch, or
        // one of the active provider's ids. Providers list first.
        let (completion, calls) = complete_counting("/model ");
        assert_eq!(
            completion,
            candidates(
                7,
                &[
                    "anthropic:",
                    "openai:",
                    "claude-sonnet-4-6",
                    "claude-opus-4-1"
                ]
            )
        );
        assert_eq!(calls, 1);
    }

    #[test]
    fn model_bare_argument_filters_by_the_typed_prefix() {
        let (completion, _) = complete_counting("/model an");
        assert_eq!(completion, candidates(7, &["anthropic:"]));
        let (completion, _) = complete_counting("/model cl");
        assert_eq!(
            completion,
            candidates(7, &["claude-sonnet-4-6", "claude-opus-4-1"])
        );
        let (completion, _) = complete_counting("/model zzz");
        assert_eq!(completion, candidates(7, &[]));
    }

    #[test]
    fn model_bare_argument_draws_ids_from_the_active_provider() {
        // The same bare token completes differently under an OpenAI session:
        // the id pool follows the active provider.
        let (completion, _) = complete_counting_as("/model gp", ProviderKind::Openai);
        assert_eq!(completion, candidates(7, &["gpt-4o", "gpt-4o-mini"]));
    }

    #[test]
    fn model_argument_start_survives_extra_spacing() {
        // The token starts after the whole whitespace run, wherever it ends.
        let (completion, _) = complete_counting("/model   an");
        assert_eq!(completion, candidates(9, &["anthropic:"]));
    }

    #[test]
    fn active_provider_prefix_completes_its_model_ids() {
        // Completion continues after the colon: candidates replace only the
        // partial id, so a listing shows bare ids, not repeated prefixes.
        let (completion, _) = complete_counting("/model anthropic:cl");
        assert_eq!(
            completion,
            candidates(17, &["claude-sonnet-4-6", "claude-opus-4-1"])
        );
        // The flagship flow: Tab right after the colon lists everything.
        let (completion, _) = complete_counting("/model anthropic:");
        assert_eq!(
            completion,
            candidates(17, &["claude-sonnet-4-6", "claude-opus-4-1"])
        );
    }

    #[test]
    fn other_provider_prefix_completes_that_providers_ids() {
        // The prefix names the pool: an `openai:` token under an Anthropic
        // session completes OpenAI's ids — the source is queried for the
        // named kind, not the active one.
        let (completion, calls) = complete_counting("/model openai:g");
        assert_eq!(completion, candidates(14, &["gpt-4o", "gpt-4o-mini"]));
        assert_eq!(calls, 1);
        // And symmetrically from an OpenAI session.
        let (completion, _) = complete_counting_as("/model anthropic:cl", ProviderKind::Openai);
        assert_eq!(
            completion,
            candidates(17, &["claude-sonnet-4-6", "claude-opus-4-1"])
        );
    }

    #[test]
    fn unknown_colon_prefix_is_a_bare_model_id() {
        // Mirrors ModelSpec::parse: colon-bearing ids (OpenAI fine-tunes)
        // are not provider prefixes. The whole token filters the bare pool.
        let (completion, _) = complete_counting("/model ft:gpt");
        assert_eq!(completion, candidates(7, &[]));
    }

    #[test]
    fn non_model_arguments_have_no_candidates() {
        // /quit takes no argument; /foo is not a command; a second /model
        // argument is beyond what the command accepts.
        for line in ["/quit n", "/foo bar", "/model claude x", "/model a "] {
            let (completion, calls) = complete_counting(line);
            assert!(completion.candidates.is_empty(), "{line:?} completed");
            assert_eq!(calls, 0, "{line:?} fetched models");
        }
    }
}
