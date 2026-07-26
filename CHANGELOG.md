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
