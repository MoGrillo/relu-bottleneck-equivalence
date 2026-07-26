# Instructions for Codex

This repository implements exact-arithmetic certificates for three-layer
bottleneck ReLU networks.

## Non-negotiable mathematical guarantees

- `equivalence_test(theta, eta).accepted == True` must remain sound for
  arbitrary exact inputs.
- Completeness is claimed only under the paper's generic bottleneck
  assumptions.
- Never interpret a sign-boundary sample as generically identifiable.
- `certified_nonidentifiable` is unconditional when an explicit dead,
  linear, or hiding symmetry is found.
- `generic_identifiable_candidate` must be `None` whenever the strict
  sign criterion is not applicable.

## Implementation conventions

- Preserve exact SymPy rational arithmetic.
- Do not replace exact sign or equality tests by floating-point tolerances.
- Keep the Python module and the notebook's embedded implementation in sync.
- Run `python run_tests.py` after every change.
- Execute the full notebook after changes affecting its API or examples.

## Main files

- `relu_bottleneck_tools.py`: reusable implementation.
- `relu_bottleneck_symmetry_workbench.ipynb`: self-contained workbench.
- `run_tests.py`: exact smoke and regression tests.
