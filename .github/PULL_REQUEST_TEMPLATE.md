<!--
  bytesandbrains pull request template.
  Fill every section. Delete the HTML comments before submitting.
-->

## Summary

<!-- 1-3 bullets describing what changes and why. Focus on intent, not diff. -->

-
-

## Per-commit gate

All six commands from `CLAUDE.md` pass locally on the tip commit. No
`--no-verify`, no `--no-gpg-sign`.

- [ ] `cargo fmt --all --check`
- [ ] `cargo test    --workspace --features test-components --locked`
- [ ] `cargo clippy  --workspace --all-targets --features test-components --locked -- -D warnings`
- [ ] `cargo doc     --workspace --no-deps --features test-components --locked`
- [ ] `cargo check   --workspace --no-default-features --locked`
- [ ] `cargo deny    check`

## Doc impact

Check every item that applies to this PR's surface; strike through the rest.

- [ ] `README.md` updated (user-visible API, install, or framing shifted)
- [ ] `CHANGELOG.md` entry added (user-visible behavior change)
- [ ] `docs/COMPILER.md` updated (compilation pipeline / per-pass invariants moved)
- [ ] `docs/IR_AND_DSL.md` updated (DSL surface or ONNX IR shape moved)
- [ ] `docs/ENGINE.md` updated (Engine state machine / poll loop moved)
- [ ] `docs/ROLES.md` / `docs/CONTRACT_DISPATCH.md` updated (role or contract surface moved)
- [ ] `docs/ADDRESSING.md` updated (multiaddr / wire-envelope routing moved)
- [ ] No doc impact — diff is internal-only and none of the above surfaces shifted

## Audit residue

Confirm the prose touched by this PR (docs, comments, commit bodies,
CHANGELOG, README) is free of retired-vocabulary references. Grep the
diff before checking the box.

- [ ] No `net::send` references in prose for this PR's surface
- [ ] No `net::recv` references in prose for this PR's surface
- [ ] No `g.wire(` references in prose for this PR's surface
- [ ] No `BackendRuntime` references in prose for this PR's surface
- [ ] No `TypeMeta` references in prose for this PR's surface
