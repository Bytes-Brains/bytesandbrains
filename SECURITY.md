# Security Policy

## Supported Versions

We release security updates for the most recent published version
on the v0.x line. Older v0.x patch versions are not maintained;
upgrade to the latest minor before reporting.

| Version | Supported |
|---------|-----------|
| 0.3.x   | yes       |
| < 0.3   | no        |

## Reporting a Vulnerability

Report security issues privately to **security@bytesandbrains.com**.

Please include:

- A description of the issue and its impact.
- Reproduction steps or a minimal proof-of-concept.
- The crate(s) and version(s) affected.
- Whether the issue is already public and, if so, where.

Do not file public GitHub issues for suspected vulnerabilities.

### Response timeline

| Phase            | Target            |
|------------------|-------------------|
| Acknowledgement  | within 3 business days  |
| Initial triage   | within 7 business days  |
| Coordinated fix  | within 90 days of triage |

Critical issues with active exploitation will be expedited.

## Scope

The `bytesandbrains` framework is sans-IO Rust code; the engine's
attack surface is defined by:

- The `bb_runtime::engine::Engine` state machine and its ingress
  queue (`bb_runtime::ingress`).
- The wire envelope codec (`bb_runtime::envelope::EnvelopeCodec`)
  and the prost-decoded `WireEnvelope` proto.
- The compilation passport check (`bytesandbrains::install`).
- The inventory registry (`bb_runtime::registry`) — including any
  Component's `restore_fn` and `serialize_fn`.

Out of scope (report upstream instead):

- Vulnerabilities in `prost`, `inventory`, `multihash`, `bincode`,
  `serde`, `instant-distance`, or any other listed dependency.
- Issues in transport adapters (libp2p, gRPC, etc.) that are not
  part of this workspace.

## Disclosure

Once a fix is published we will:

1. Tag and publish patched releases of every affected crate.
2. Update `CHANGELOG.md` with the security-relevant entry.
3. File a RustSec advisory.
4. Credit the reporter (with permission).
