# Contributing to `bytesandbrains`

Thanks for your interest. During the v0.x stabilization period
external pull requests land by invitation; the fastest way to
influence the framework is an issue describing your use case and
the surface you need.

## Code of Conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md).
Participation in any project channel — issues, pull requests,
discussions — implies agreement.

## Filing issues

- Search existing issues first.
- For bugs: include a minimal reproduction, the crate version, your
  Rust toolchain (`rustc --version`), and your platform.
- For features: describe the use case, not the implementation. We
  may have a different shape in mind that solves the same problem.
- For security issues: do NOT open a public issue. See
  [SECURITY.md](SECURITY.md).

## Building from source

Requirements:

- Rust toolchain pinned in `rust-toolchain.toml` (currently 1.86,
  matching the workspace MSRV).
- `protoc` on `PATH` (the build script invokes `prost-build`).

```bash
git clone https://github.com/Bytes-Brains/bytesandbrains
cd bytesandbrains
cargo build --workspace
```

## Workspace layout

Seven publishable crates:

| Crate            | Layer       |
|------------------|-------------|
| `bytesandbrains` | facade      |
| `bb-ir`          | foundation  |
| `bb-ops`         | concrete components |
| `bb-dsl`         | authoring   |
| `bb-compiler`    | compilation pipeline |
| `bb-runtime`     | engine      |
| `bb-derive`      | proc-macros |

End users depend only on `bytesandbrains`. Library authors targeting
a specific layer (e.g. a new compiler pass) can depend on the
relevant member crate directly.

## Pre-PR check

Every PR must pass the same gate that CI enforces:

```bash
cargo fmt --all --check
cargo test    --workspace --features test-components --locked
cargo clippy  --workspace --all-targets --features test-components --locked -- -D warnings
cargo doc     --workspace --no-deps --features test-components --locked
cargo check   --workspace --no-default-features --locked
cargo deny    check
```

Do not use `--no-verify` or `--no-gpg-sign`. If a hook fails,
diagnose the underlying issue and create a new commit.

## Coding conventions

- **No dead code.** A new placeholder ships in the same commit as
  its consumer. No `_Reserved` variants, no `#[allow(dead_code)]`
  without a named downstream consumer, no `pub(crate) fn` without
  callers.
- **Test on every regression.** Bug-fix commits include a
  regression test that failed before and passes after.
- **Tests live in siblings.** Use `#[cfg(test)] #[path = "name_tests.rs"]
  mod tests;` — three lines, sibling file. Never inline
  `#[cfg(test)] mod tests { ... }` in `src/*.rs`.
- **No `tokio::` in `src/`.** The Node is sans-IO; transport
  adapters land in a separate crate.
- **Async by `CommandId`.** Every long-running role primitive
  returns synchronously and emits a typed completion event keyed by
  `CommandId`.
- **The bus is the only cross-Component channel.** No globals, no
  shared handles, no inter-Component method calls.
- **The IR is ONNX.** `bb_ir::proto::onnx::ModelProto` is the
  program. No parallel custom IR types.
- **Three-phase construction.** Author records via
  `Module::build() → ModelProto`; the compiler binds via
  `Compiler::new().bind_<role>::<T>("slot").compile(model)`; the
  host installs via `bb::install(peer_id, addr, compiled, target,
  Config::new())`. No `NodeBuilder`, no `Node::with_*` chain.
- **Inventory is the registration mechanism.** Concretes and ops
  submit via `inventory::submit!` (the `#[derive(bb::Concrete)]` /
  `#[derive(bb::<Role>)]` derives + `register_op!` /
  `register_protocol!` macros emit the submission).
- **Declarative present tense in all prose.** No "previously", "is
  now", "no longer", "after the refactor", "Phase N", "Concern N",
  "Stage N". Comments and docs describe current behavior.

## Commit messages

Format: `<type>(<scope>): <subject>`

Types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`.

Body: explain the _why_, not the _what_. The diff covers the what.

No `Co-Authored-By: Claude` lines.

## Pull requests

- One logical change per PR.
- Update `CHANGELOG.md` under `[Unreleased]` if the change is
  user-visible.
- Update the relevant `docs/*.md` file if the change shifts
  architecture.
- Mark draft until CI is green.

## License

By contributing you agree your changes are dual-licensed under the
project's [AGPL-3.0-or-later](LICENSE) and the project's commercial
license (see [README.md](README.md#license)).
