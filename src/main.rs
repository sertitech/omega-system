use omega_system::agent::{
    Agent, AgentConfig, BudgetLedger, Concurrency, ConfirmPolicy, interactive_confirm,
};
use omega_system::cli::{CliArgs, USAGE, select_initial_session};
use omega_system::config::Config;
use omega_system::provider;
use omega_system::repl::{
    History, TtyEditor, install_sigint_cancel, persist_sink, run_repl, stdin_stdout_are_ttys,
};
use omega_system::session_store::SessionStore;
use omega_system::tools::default_tools;
use omega_system::tools::sandbox::Sandbox;
use omega_system::tools::subagent::TaskTool;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const CONFIG_PATH: &str = "config.json";

/// Thin process wiring only: load config, resolve the key, build the
/// provider/sandbox/tools, then hand the loop to [`run_repl`]. Everything
/// interactive lives behind that seam, where it carries hermetic coverage —
/// nothing here should grow logic.
fn main() {
    // Omega's only argv surface: `--resume [name]`. Parsed first so a typo'd
    // flag fails fast with usage before any startup work; the behavioral
    // selection it drives lives in the covered `cli` module below. `args_os`
    // (not `args`) so a non-UTF-8 argument folds into the same clean usage error
    // instead of panicking mid-iteration.
    let args = match std::env::args_os()
        .skip(1)
        .map(|a| a.into_string())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| USAGE.to_string())
        .and_then(|raw| CliArgs::parse(&raw))
    {
        Ok(args) => args,
        Err(usage) => {
            eprintln!("{usage}");
            std::process::exit(1);
        }
    };

    // The global settings layer lives at `~/.omega-system/`; computed once and
    // reused for both the config merge and the credential shield below.
    let home = omega_system::home_dir();
    let global_config = home.as_ref().map(|h| h.join(".omega-system/config.json"));
    let (config, notices) =
        match Config::load_layers(global_config.as_deref(), Path::new(CONFIG_PATH)) {
            Ok(loaded) => loaded,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
    // Startup notices (confirm overrides, profile clamps) go to stderr so a
    // piped stdout transcript stays clean.
    for notice in &notices {
        eprintln!("{notice}");
    }

    let key_env = config.provider.api_key_env();
    let api_key = match omega_system::load_env_var_checked(key_env) {
        Ok(Some(key)) => key,
        Ok(None) => {
            eprintln!("error: {key_env} not found in environment or .env file");
            std::process::exit(1);
        }
        Err(e) => {
            // An existing-but-unreadable `.env` is a real problem — surface it
            // like the adjacent Config::load_layers / History::load handlers, not
            // as a misleading "key not found".
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    // The shared turn-cancellation flag: the SIGINT handler (interactive
    // sessions only, installed below) raises it; the agent's seams, the shell
    // tool's deadline loop, and the providers' retry backoff poll it.
    let cancel = Arc::new(AtomicBool::new(false));

    // The provider-construction seam: builds providers with the shared cancel
    // flag threaded in, and resolves their keys lazily. Used here for the
    // initial provider and, inside `run_repl`, for every `/model` switch.
    let providers = provider::ProviderFactory::new(Arc::clone(&cancel), omega_system::load_env_var);
    let provider = providers.build(config.provider, api_key);

    let cwd = match std::env::current_dir() {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("error: cannot determine working directory: {e}");
            std::process::exit(1);
        }
    };
    let sandbox = match Sandbox::rooted(cwd) {
        Ok(sb) => sb,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    // Shield the global credentials from the ungated filesystem tools: running
    // Omega from `$HOME` would otherwise place `~/.omega-system/.env` inside
    // the sandbox. Inert when there is no home directory (the builder decides).
    let protected_home = home.as_ref().map(|h| h.join(".omega-system"));
    let sandbox = sandbox.with_protected_home(protected_home.as_deref());

    // The named session store (`/save` / `/load` / `/sessions`), rooted under
    // `~/.omega-system/sessions/<project-key>/`. Keyed off the sandbox's
    // canonical root so symlinked checkouts share saves; `None` without a home
    // directory, which the three commands report as unavailable.
    let session_store = home
        .as_ref()
        .zip(sandbox.root())
        .map(|(h, root)| SessionStore::new(h, root));

    let firecrawl_key = omega_system::load_env_var("FIRECRAWL_API_KEY");
    let mut tools = default_tools(sandbox.clone(), firecrawl_key.clone(), Arc::clone(&cancel));

    // The process-wide tool-call budget ledger, shared by the top-level agent
    // and every subagent the `task` tool spawns, so children draw against the
    // same per-tier ceilings as their parent.
    let budget = BudgetLedger::new();

    // The process-wide fan-out concurrency pool, shared the same way: the
    // top-level agent's tool fan-out and every child's fan-out draw permits
    // from one ceiling, so a wide or deep delegation tree cannot outgrow it.
    let concurrency = Concurrency::new();

    // The session confirmation policy: mode from `config.json`,
    // `ask` prompts through the interactive TTY tail, judge providers built
    // through the same factory as every child spawn, the floor anchored at
    // the sandbox root. One object, cloned to the agent and the `task` tool —
    // the shared mode cell is what lets `/confirm` reach both.
    let confirm = ConfirmPolicy::new(
        config.confirm,
        interactive_confirm,
        providers.clone(),
        config.agents.clone(),
        sandbox.clone(),
    );

    // The `/agents` REPL report reads the profile table after `config.agents`
    // is moved into the `task` tool below — clone it once more for the loop.
    let agent_profiles = config.agents.clone();

    // The `task` tool is registered only when profiles are configured — with
    // no `agents`, the tool list the model sees stays byte-identical to before.
    // It holds the provider-construction seam (lazy key resolution per spawn),
    // the sandbox and firecrawl key for child toolsets, the shared cancel flag
    // and budget ledger, the session confirmation policy for children, and the
    // parent's token budgets for child requests.
    if !config.agents.is_empty() {
        tools.push(Box::new(TaskTool::new(
            providers.clone(),
            sandbox,
            firecrawl_key,
            Arc::clone(&cancel),
            budget.clone(),
            concurrency.clone(),
            confirm.clone(),
            config.agents,
            config.max_tokens,
            config.context_token_limit,
            config.max_turns,
        )));
    }

    let agent_config = AgentConfig {
        provider_kind: config.provider,
        model: config.model,
        max_tokens: config.max_tokens,
        system: config.system,
        context_token_limit: config.context_token_limit,
        effort: config.effort,
        max_turns: config.max_turns,
    };
    let mut agent = Agent::new(provider, agent_config, tools);
    // Hand the agent the shared ledger, replacing the fresh one `Agent::new`
    // built, so the top-level agent and the `task` tool draw from one budget —
    // and the session confirmation policy, replacing the bare interactive
    // default with the config-selected mode over the real seams.
    agent.set_budget_ledger(budget);
    agent.set_concurrency(concurrency);
    agent.set_confirm_policy(confirm);

    // Computed once, before the session selection: an interactive session
    // (stdin and stdout both TTYs) gets the raw-mode editor below, and a bare
    // `--resume` with several saves gets the picker; a piped run gets neither.
    let interactive = stdin_stdout_are_ttys();

    // Startup session selection (the covered branch in `cli`): `--resume`
    // restores a named or newest save from the store — and, for a bare
    // `--resume` with several saves in an interactive session, prompts through
    // the picker (rendered to stderr, over a transient stdin lock dropped
    // before the loop); otherwise the pre-existing opt-in `session_file`
    // restore applies (an existing file loads **fail-fast** — a corrupt or
    // version-mismatched session is a real problem the operator must see, like
    // `Config::load` above — and a configured-but-absent path is the first-run
    // case that starts fresh). The save side (fail soft) lives in
    // `repl::autosave`.
    match select_initial_session(
        &args,
        session_store.as_ref(),
        config.session_file.as_deref(),
        interactive,
        &mut std::io::stdin().lock(),
        &mut std::io::stderr().lock(),
    ) {
        Ok(Some(session)) => agent.restore(session),
        Ok(None) => {}
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
    // The autosave sink run_repl calls after every turn, /clear, and /load.
    // The covered factory selects the target: the project `session_file` when
    // set (byte-identical to before), else a per-session auto-named file under
    // the store, else a no-op without either. It borrows the store — passed on
    // to run_repl below — since SessionStore is not Clone.
    let persist = persist_sink(config.session_file, session_store.as_ref());

    // An interactive session (stdin and stdout both TTYs) gets the raw-mode
    // editor with Tab completion and ANSI-dimmed meta lines; piped input or
    // a redirected transcript keeps plain cooked line reads and plain text —
    // the stdin *handle*, not a held StdinLock: the confirmation gate reads
    // the same stdin mid-turn, so the REPL's reads must lock per call (see
    // `repl::LineSource`). stdout/stderr locks are reentrant and safe to
    // hold. The trailing arguments are the `/model <provider>:<model>` switch
    // seam — the `providers` factory built above — and the persistence sink.
    agent.set_styled(interactive);
    agent.set_cancel_flag(Arc::clone(&cancel));
    if interactive {
        // Ctrl-C cancels the in-flight turn (twice force-exits). Installed
        // only for interactive sessions: a piped run keeps SIGINT's default
        // kill-the-process disposition, which scripts expect.
        install_sigint_cancel(cancel);
        // Opt-in persistent editor history. **Load fails fast** like the
        // session file above: a history file that exists but cannot be
        // read is a real problem, not entries to silently drop; absent
        // config or a missing file starts empty.
        let history = match &config.history_file {
            Some(path) => match History::load(path) {
                Ok(history) => history,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            },
            None => History::in_memory(),
        };
        run_repl(
            TtyEditor::new(history),
            std::io::stdout().lock(),
            std::io::stderr().lock(),
            &mut agent,
            &providers,
            &agent_profiles,
            session_store.as_ref(),
            persist,
        );
    } else {
        run_repl(
            std::io::stdin(),
            std::io::stdout().lock(),
            std::io::stderr().lock(),
            &mut agent,
            &providers,
            &agent_profiles,
            session_store.as_ref(),
            persist,
        );
    }
}
