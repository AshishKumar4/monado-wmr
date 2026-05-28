# Unified Associator Lean Scaffold

This directory is a standalone Lean 4 proof-model scaffold for the unified
associator audit. It is intentionally disjoint from the C/C++ tracker
implementation and is not wired into the repository build.

## Scope

- Finite per-frame blob keys and LED keys.
- Finite candidate hypotheses, each carrying a blob-to-LED assignment.
- Compatibility obligations for assigned blob/LED pairs.
- A joint cost model where reprojection and prior costs are combined before
  taking the argmin.
- State decisions that reject, accept, or reset only through the joint-argmin
  contract.
- Telemetry obligations for frame summaries, hypothesis scores, compatibility
  gates, decisions, and accepted/reset residuals.

## Non-Scope

- Numeric geometry, PnP/P3P solving, quaternion algebra, Kalman filtering, and
  Mahalanobis covariance gates.
- Proof that hypothesis enumeration is complete for the production engine.
- Extraction, code generation, or equivalence proof against C/C++ tracker code.
- Main CMake/Gradle integration.

## Layout

- `lakefile.lean` defines a minimal Lake package with no third-party
  dependencies.
- `UnifiedAssociator/Keys.lean` defines finite key domains and key types.
- `UnifiedAssociator/ProofModel.lean` defines assignments, compatibility,
  joint argmin, state decisions, telemetry, and bundled proof obligations.

If Lean 4 and Lake are available locally, run:

```sh
cd formal
lake build
```

No dependency installation was performed while creating this scaffold.
