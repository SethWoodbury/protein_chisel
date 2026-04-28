# tests/

`pytest` tests. Run with `pytest` from the repo root (after `pip install -e .[dev]`).

Tests for tools that need GPU / containers should be marked `@pytest.mark.cluster` and skipped by default; CI on the cluster runs them via slurm.

Tiny test fixtures (PDBs etc.) live in `tests/data/`. `.gitignore` excludes large binaries by default — override with a per-file allowlist if a small fixture must be committed.
