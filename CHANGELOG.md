# Changelog

## Version 2

- Replaced the misleading `sign_identifiable_rate`.
- Added `certified_nonidentifiable`.
- Added `sign_criterion_applicable`.
- Added nullable `generic_identifiable_candidate`.
- Zero-bias initialization groups now report no generic identifiability
  rate when every sample lies on a sign boundary.
- Added regression tests for sign-boundary behavior.
- Added VS Code recommendations and `AGENTS.md` for Codex.

## Version 3

- Removed zero-bias initialization experiments.
- Retained only small random bias initialization.
- Simplified the empirical summary and notebook explanation.
- Removed the sign-boundary demonstration from the main notebook.

## Version 4

- Kept exact zero-valued parameters fully supported while limiting completeness
  and exact fiber-dimension claims to the paper's generic regime.
- Added translation-only equivalence certificates for weak hiding patterns with
  zero non-hidden coefficients.
- Made hiding translations validate their exact parameter domain.
- Generalized width-one sign flips to arbitrary nonzero coefficient magnitudes
  by conjugating the normalized action with positive scalings.
- Report generic-identifiability and exact generic-fiber dimensions as
  undetermined on detected sign boundaries, together with explicit symmetry
  lower bounds.
- Corrected the generic layerwise orientation-family counts.
- Added exact zero-boundary and transformation-domain regression tests.

## Version 5

- Installed and registered the dedicated `Python (relu-bottleneck)` Jupyter
  kernel.
- Moved all environment/setup guidance out of the notebook.
- Reorganized the notebook into implementation, editable inputs, exact
  certificates, fiber dimensions, and initialization results.
- Replaced wide raw dictionaries with compact styled result tables.
- Saved a fresh, fully executed set of notebook outputs.
- Declared `nbconvert` explicitly and made `setup.sh` install from the shared
  requirements file.
