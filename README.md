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

Omega loads personal instructions from `~/.omega-system/AGENTS.md`, followed
by `AGENTS.md` at the project launch directory's root. Repository instructions require
one-time interactive consent, defaulting to no; the decision is stored per
canonical project path in `~/.omega-system/trusted.json`. Piped sessions load
only previously trusted projects. Delete a project's entry to reconsider its
decision. Trust covers the project, so changing its instructions does not
prompt again. Keep `~/.omega-system/trusted.lock` in place; it coordinates
consent updates between running Omega processes.

Each instruction file must be a regular UTF-8 file of at most 32 KiB;
symlinks are rejected. Omega does not search parent or nested directories.
Instructions supplement the configured system prompt, cannot change tool
confirmation policy, and are not automatically inherited by subagents.

Launch Omega from the project it should work in:

```sh
cd /path/to/project
omega-system
```

The launch directory becomes the root used by Omega's dedicated filesystem
tools. Useful commands include `/model`, `/effort`, `/agents`, `/save`,
`/load`, `/sessions`, `/clear`, and `/quit`. Start with
`omega-system --resume` to resume a saved session.

## Working with larger files and conversations

The `read_file` tool accepts an optional 1-based `offset` and a positive `limit`
in lines. For example, `{"path":"src/agent/mod.rs","offset":101,"limit":80}`
reads lines 101–180. Ranged results report the next offset or EOF; the 100 KB
output cap still applies and identifies an incomplete final line when reached.

Before every provider request, Omega checks a conservative UTF-8 byte estimate
of the messages, system prompt, and tool definitions against the configured
context limit and reply budget. It can shorten tool results in the outgoing
request while preserving the full results in the session. Older conversation
groups can be summarized before a new turn; a summary is committed only if the
resulting request fits. If the prompt itself cannot fit, Omega reports a local
error: use a shorter prompt, smaller file ranges, or `/clear`. This estimate is
not an exact provider token count.

## Security model

Omega is a local coding agent, not a containment system. Its dedicated
filesystem tools canonicalize paths and refuse access outside the launch
directory, but the shell tool runs as the current operating-system user and
can access anything that user can access. The filesystem sandbox therefore
does not confine an approved shell command.

`confirm: "ask"` is the default and prompts before file writes, edits, or shell
execution. File approvals show every changed line with line numbers and visible
escapes for terminal controls. A change that exceeds the 16 KiB / 200-line
preview limit is rejected with a request for smaller edits; existing source
files larger than 8 MiB cannot be previewed. These review limits apply to
interactive approval.

Shell deadlines and cancellation cover both process execution and output
collection, including pipes left open by descendants. Omega cleans up the shell
process group when collection finishes or fails. Shell guardrails block several
destructive command shapes, but they
are best-effort static analysis rather than a complete shell parser. Review
every proposed command before approving it, and run Omega only in projects and
environments where you accept that trust boundary.

## Quality gate

Install the pinned coverage tool once:

```sh
cargo install cargo-llvm-cov --locked --version 0.8.7
```

Pull requests and pushes to `main` run the gate on macOS and Linux. To run the
same gate locally:

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
