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
  PR), then runs [`./check`](../check) — the same fmt, clippy, test, and scoped
  100%-coverage gate CI runs — before the push reaches the network.

These are hand-rolled POSIX `sh` — no `pre-commit` framework, no Node toolchain —
per the project's no-external-dependencies principle.
