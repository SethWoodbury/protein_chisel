"""Tests for ESM-C and SaProt logits + score wrappers.

Cluster tests; run inside esmc.sif. Mark gpu where useful — small models
work on CPU and we run the smoke tests there for cheap.

Uses the smallest checkpoints (esmc_300m, saprot_35m) for speed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cluster


TEST_DIR = Path("/home/woodbuse/testing_space/align_seth_test")
DESIGN_PDB = TEST_DIR / "design.pdb"


# ---- ESM-C ----------------------------------------------------------------


def test_esmc_logits_shape_and_normalization():
    from protein_chisel.tools.esmc import esmc_logits

    seq = "MEEVEEYARLVIEAIEKHRDLIREAIEEEIRIYRETGEETHAKR"
    # Use unmasked path here for speed (masked is L forward passes); we
    # exercise the masked path explicitly below on a shorter sequence.
    res = esmc_logits(seq, model_name="esmc_300m", device="cpu", masked=False)
    L = len(seq)
    assert res.log_probs.shape == (L, 20)
    sums = np.exp(res.log_probs).sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-3)


def test_esmc_logits_masked_differs_from_unmasked():
    """Masked-LM marginals must differ from unmasked single-pass logits.

    Regression test for the bug where esmc_logits used a single unmasked
    forward pass (so the model saw the true AA at each position) — that
    is NOT a masked-LM marginal and was wrong for fusion.
    """
    from protein_chisel.tools.esmc import esmc_logits

    seq = "MGGGAA"  # short enough to keep CPU runtime modest
    res_unmasked = esmc_logits(seq, model_name="esmc_300m", device="cpu", masked=False)
    res_masked = esmc_logits(seq, model_name="esmc_300m", device="cpu", masked=True)
    assert res_unmasked.log_probs.shape == res_masked.log_probs.shape
    diff = np.abs(res_unmasked.log_probs - res_masked.log_probs).max()
    assert diff > 0.1, (
        f"masked and unmasked logits should differ; max abs diff {diff:.4f}"
    )


def test_esmc_score_pseudo_perplexity_finite():
    from protein_chisel.tools.esmc import esmc_score

    seq = "MEEVEEYARLVIEAIEKHRDLIREAIEEEIRIYRETGEET"  # 40 aa for speed
    res = esmc_score(seq, model_name="esmc_300m", device="cpu")
    assert np.isfinite(res.pseudo_perplexity)
    # Reasonable ESM-C perplexity is typically 1-15 for designed proteins;
    # use a permissive upper bound.
    assert 1.0 < res.pseudo_perplexity < 30.0
    assert res.per_position_loglik.shape == (len(seq),)


def test_esmc_score_to_dict():
    from protein_chisel.tools.esmc import esmc_score

    res = esmc_score("MEEVEEYARLVIE", model_name="esmc_300m", device="cpu")
    d = res.to_dict()
    assert "esmc__pseudo_perplexity" in d
    assert d["esmc__pseudo_perplexity"] == res.pseudo_perplexity


# ---- SaProt ---------------------------------------------------------------


def test_saprot_logits_shape_and_normalization():
    from protein_chisel.tools.saprot import saprot_logits

    res = saprot_logits(DESIGN_PDB, model_name="saprot_35m", device="cpu")
    # design has 208 protein residues
    assert res.log_probs.shape == (208, 20)
    sums = np.exp(res.log_probs).sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-3)


def test_saprot_score_pseudo_perplexity():
    from protein_chisel.tools.saprot import saprot_score

    # Smallest model + CPU is fine for ~200 aa
    res = saprot_score(DESIGN_PDB, model_name="saprot_35m", device="cpu", batch_size=16)
    assert np.isfinite(res.pseudo_perplexity)
    assert 1.0 < res.pseudo_perplexity < 30.0
    d = res.to_dict()
    assert "saprot__pseudo_perplexity" in d
