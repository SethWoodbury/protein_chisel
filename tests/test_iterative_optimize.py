"""Host tests for the iterative_optimize pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from protein_chisel.pipelines.iterative_optimize import (
    IterativeOptimizeConfig,
    iterative_optimize,
)
from protein_chisel.sampling.plm_fusion import AA_ORDER as PLM_AA_ORDER


def _peaked_log_probs(L: int, target_aa: str) -> np.ndarray:
    """Each row: 1.0 prob on target_aa, ~0 elsewhere."""
    p = np.full((L, 20), 1e-6)
    j = PLM_AA_ORDER.index(target_aa)
    p[:, j] = 1.0 - 19 * 1e-6
    return np.log(p / p.sum(axis=-1, keepdims=True))


def test_constrained_local_search_converges_to_target():
    """If we propose only A and accept anything, walk converges to all-A."""
    seq = "MGGGGGGGG"
    lp = _peaked_log_probs(len(seq), "A")
    cfg = IterativeOptimizeConfig(
        mode="constrained_local_search",
        n_iterations=200,
        sample_temperature=0.1,
        convergence_window=50,
        seed=42,
    )
    result = iterative_optimize(
        seq, lp, fixed_positions={0},  # keep M frozen
        accept_fn=lambda s: True, config=cfg,
    )
    final = result.final_sequences[0]
    # First position kept (M); rest should be A
    assert final[0] == "M"
    assert final[1:] == "A" * (len(seq) - 1)


def test_fixed_positions_never_mutated():
    seq = "MGGGGG"
    lp = _peaked_log_probs(len(seq), "A")
    cfg = IterativeOptimizeConfig(mode="constrained_local_search", n_iterations=100, seed=0)
    result = iterative_optimize(
        seq, lp, fixed_positions={0, 2, 4},
        accept_fn=lambda s: True, config=cfg,
    )
    final = result.final_sequences[0]
    assert final[0] == seq[0]
    assert final[2] == seq[2]
    assert final[4] == seq[4]


def test_walk_log_has_required_columns():
    seq = "MGGG"
    lp = _peaked_log_probs(len(seq), "A")
    cfg = IterativeOptimizeConfig(mode="constrained_local_search", n_iterations=20, seed=0)
    result = iterative_optimize(
        seq, lp, fixed_positions={0},
        accept_fn=lambda s: True, config=cfg,
    )
    df = result.walk_log
    for col in ("chain", "iter", "position", "old_aa", "new_aa", "accepted"):
        assert col in df.columns


def test_constrained_local_search_rejects_per_filter():
    """If the accept_fn always rejects, no candidates are kept."""
    seq = "MGGG"
    lp = _peaked_log_probs(len(seq), "A")
    cfg = IterativeOptimizeConfig(
        mode="constrained_local_search", n_iterations=20, seed=0,
        convergence_window=10,
    )
    result = iterative_optimize(
        seq, lp, fixed_positions={0},
        accept_fn=lambda s: False, config=cfg,
    )
    # No accepted moves → final sequence equals starting
    assert result.final_sequences[0] == seq
    assert len(result.candidate_set.df) == 0


def test_mh_decreases_energy_on_average():
    """Synthetic energy = number of non-A residues. MH should reduce it.

    Uses a NEAR-UNIFORM proposal distribution: with a heavily-peaked
    proposal, the q-correction term in MH (q(s|s') / q(s'|s)) dominates
    the energy term at most temperatures and rejects the move. That's
    correct MH behavior; this test verifies MH can move when given a
    sensible proposal.
    """
    seq = "MGGGGGGGGG"
    # Uniform-ish proposal: equal log-prob across all 20 AAs
    lp = np.full((len(seq), 20), np.log(1.0 / 20))

    def energy(s: str) -> float:
        return float(sum(1 for c in s if c != "A"))

    cfg = IterativeOptimizeConfig(
        mode="mh", n_iterations=400,
        initial_mh_temperature=2.0, final_mh_temperature=0.05,
        sample_temperature=1.0,
        convergence_window=400, seed=0,
    )
    result = iterative_optimize(
        seq, lp, fixed_positions={0}, energy_fn=energy, config=cfg,
    )
    e0 = energy(seq)
    e_final = energy(result.final_sequences[0])
    assert e_final < e0, f"MH should reduce energy; e0={e0}, e_final={e_final}"


def test_multiple_chains_run_independently():
    seq = "MGGGGGG"
    lp = _peaked_log_probs(len(seq), "A")
    cfg = IterativeOptimizeConfig(
        mode="constrained_local_search", n_iterations=30, n_chains=3, seed=0,
        convergence_window=20,
    )
    result = iterative_optimize(
        seq, lp, fixed_positions={0},
        accept_fn=lambda s: True, config=cfg,
    )
    assert len(result.final_sequences) == 3
    assert len(result.final_scores) == 3
    # Walk log contains all 3 chains
    assert set(result.walk_log["chain"].unique()) == {0, 1, 2}


def test_writes_outputs_to_disk(tmp_path: Path):
    seq = "MGG"
    lp = _peaked_log_probs(len(seq), "A")
    cfg = IterativeOptimizeConfig(mode="constrained_local_search", n_iterations=10, seed=0)
    iterative_optimize(
        seq, lp, fixed_positions={0},
        accept_fn=lambda s: True, config=cfg,
        out_dir=tmp_path,
    )
    assert (tmp_path / "walk_log.tsv").exists()
    # Candidates may or may not exist depending on accepted moves; if any,
    # check the FASTA was written.
    if (tmp_path / "candidates.fasta").exists():
        assert (tmp_path / "candidates.fasta").read_text().startswith(">")
