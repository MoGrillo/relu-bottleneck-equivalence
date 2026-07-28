# Changelog

## Version 2

- Replaced the misleading `sign_identifiable_rate`.
- Added `certified_nonidentifiable`.
- Added `sign_criterion_applicable`.
- Added nullable `generic_identifiable_candidate`.
- Zero-bias initialization groups now report no generic identifiability
  rate when every sample lies on a sign boundary.
- Added regression tests for sign-boundary behavior.
- Added VS Code recommendations.

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

## Version 6

- Added reproducible sweeps over input dimension and both hidden-layer widths.
- Added exact aggregation of generic-identifiable candidate, certified
  non-identifiable, and undetermined rates with a shared denominator.
- Added an exact iid symmetric-sign baseline for separating width effects from
  Monte Carlo variation across input dimensions.
- Made non-bottleneck samples explicitly theorem-out-of-scope rather than
  generic-identifiable candidates.
- Added compact architecture tables and a width-by-width candidate-rate matrix
  to the notebook.
- Expanded the saved sweep to 50 architectures with `d,n1` up to 16 and `n2`
  up to 32, using 2,500 exact-classification samples.
- Added a batch mode that skips unused first-layer warning diagnostics while
  preserving every mathematical classification and dimension result.
- Added regression coverage for sweep reproducibility, rate accounting, and
  non-bottleneck scope.

## Version 7

- Made the expensive exact `W1.rank()` genericity diagnostic opt-in through
  `check_first_layer_rank=True`.
- Kept inexpensive zero-row and coincident-hyperplane warnings enabled in
  ordinary one-parameter reports.
- Added an executed notebook example analyzing an exact 493-parameter network.
- Added measured exact-rational scaling results through 1,083,001 parameters.
- Added 12,609-parameter dead-neuron, linear-neuron, and hiding examples,
  together with sound equivalence certificates for explicit symmetry actions.
