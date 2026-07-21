# Omega System

Omega System is a personal Rust CLI agent built from scratch to make the
runtime understandable end to end. It streams responses from Anthropic's
Messages API or OpenAI's Responses API, translates both into one
provider-neutral turn model, executes model-requested tools, and drives the
conversation through an interactive terminal.

There is no agent framework underneath it. The provider adapters, SSE parser,
tool loop, confirmation policy, terminal editor, session persistence, and
subagent orchestration are implemented here. The runtime dependency surface is
deliberately small: Serde/`serde_json`, `ureq`, and `libc`.

## Highlights

- **Two provider adapters, one agent core.** Anthropic Messages and OpenAI
  Responses map into the same normalized messages, streaming deltas, tool
  calls, and usage accounting. Providers and models can be switched without
  discarding the conversation.
- **A real tool loop.** Omega can read, search, create, and edit files; fetch
  public web pages; optionally search through Firecrawl; and run shell
  commands. Independent read-only calls can fan out concurrently, while
  mutating calls remain ordered.
- **Terminal-first interaction.** The raw-mode editor supports cursor movement,
  history, command and model completion, streamed Markdown rendering, Ctrl-C
  cancellation, and resumable sessions.
- **Configurable delegation.** Named subagent profiles can select different
  providers, models, reasoning effort, tool allowlists, and confirmation modes
  while sharing process-wide tool budgets and concurrency limits.
- **Explicit safety boundaries.** Filesystem tools reject path and symlink
  escapes from the project root, dangerous tools pass through validation and
  confirmation, shell children receive a minimal environment and a timeout,
  and web fetching rejects non-public network destinations.
- **Scoped 100% coverage gate.** All hermetically testable lines must be
  covered. Only thin process wiring and live network/TTY tails are excluded.

## Requirements

- macOS or Linux
- Rust 1.94.1
- An Anthropic or OpenAI API key
- Optional: a Firecrawl API key for `web_search`

## Install and configure

```sh
git clone https://github.com/sertitech/omega-system.git
cd omega-system
cargo install --path . --locked

mkdir -p ~/.omega-system
cp config.example.json ~/.omega-system/config.json
cp .env.example ~/.omega-system/.env
```

Edit `~/.omega-system/config.json` and replace the model placeholder with a
model served by the selected provider. Then add the matching credential to
`~/.omega-system/.env`.

Configuration is layered: `~/.omega-system/config.json` supplies global
defaults, and a `config.json` in the working directory can override individual
fields. Credentials are resolved from the process environment, then `./.env`,
then `~/.omega-system/.env`; API keys never belong in JSON configuration.

Launch Omega from the project it should work in:

```sh
cd /path/to/project
omega-system
```

The launch directory becomes the root used by Omega's dedicated filesystem
tools. Useful commands include `/model`, `/effort`, `/agents`, `/save`,
`/load`, `/sessions`, `/clear`, and `/quit`. Start with
`omega-system --resume` to resume a saved session.

## Security model

Omega is a local coding agent, not a containment system. Its dedicated
filesystem tools canonicalize paths and refuse access outside the launch
directory, but the shell tool runs as the current operating-system user and
can access anything that user can access. The filesystem sandbox therefore
does not confine an approved shell command.

`confirm: "ask"` is the default and prompts before file writes, edits, or shell
execution. Shell guardrails block several destructive command shapes, but they
are best-effort static analysis rather than a complete shell parser. Review
every proposed command before approving it, and run Omega only in projects and
environments where you accept that trust boundary.

## Quality gate

Install the pinned coverage tool once:

```sh
cargo install cargo-llvm-cov --locked --version 0.8.7
```

Then run the complete local gate:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
cargo llvm-cov --summary-only \
  --ignore-filename-regex '(^|/)src/main\.rs$|_live\.rs$' \
  --fail-under-lines 100
```

The coverage scope excludes only thin process wiring and live API/TTY tails;
validation, parsing, state transitions, retries, tool behavior, and interactive
dispatch remain on the covered side.

## License

Omega System is available under the [MIT License](LICENSE).
