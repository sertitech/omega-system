# Git hooks

Versioned, repo-wide git hooks. Unlike `.git/hooks` (per-checkout, never
committed), this directory is tracked and shared across every worktree via
`core.hooksPath`.

## Activation (once per clone)

`core.hooksPath` is a local git config value, so it is **not** set automatically
by checking out the repo. Run this once after cloning:

```sh
git config core.hooksPath hooks
```

The path is relative, so it resolves correctly in every worktree.

## Hooks

- **`pre-push`** — refuses any push whose target is `main` (all changes land via
  PR), then runs `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  and `cargo test --locked` before the push reaches the network. Run the slower scoped
  100%-coverage gate separately. Those cargo runs use a tree-local
  `CARGO_TARGET_DIR` (`target-push/`, `.gitignore`'d) rather than the shared
  cache below: cargo's freshness check can serve a test binary built from
  another worktree's sources,
  and the push gate must always compile and test the tree being pushed. The first
  push per tree cold-builds there; that is the accepted cost.
- **`post-checkout`** — in linked worktrees only (`git worktree add` fires it),
  generates an un-versioned, `.gitignore`'d `.cargo/config.toml` pointing
  `build.target-dir` at the main checkout's `target/`, so worktree builds reuse
  the warm dependency cache instead of rebuilding it in every worktree. A no-op in
  the main checkout, and it never clobbers a `.cargo/config.toml` it did not
  generate. Best-effort by design: its exit status propagates to the checkout
  that fired it, so every fallible step warns and exits clean rather than
  failing a successful checkout — including on git < 2.31, where the
  `--path-format=absolute` it needs is unavailable and the cache is simply not
  wired. Trade-offs (accepted): concurrent builds across worktrees serialize
  on cargo's build-directory lock — serialized-warm beats parallel-cold;
  `cargo clean` from any worktree cleans the shared cache; and cargo's freshness
  check may run a test binary built from another tree's sources, which is why
  `pre-push` builds into its own `target-push/` instead.

These are hand-rolled POSIX `sh` — no `pre-commit` framework, no Node toolchain —
per the project's no-external-dependencies principle.
