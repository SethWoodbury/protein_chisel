"""Tests for the pluggable expert registry + base caching (no models loaded)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from protein_chisel.experts import (
    ExpertContext,
    available_experts,
    get_expert,
    resolve_experts,
)
from protein_chisel.experts.base import Expert
from protein_chisel.experts.esmc import ESMCExpert
from protein_chisel.experts.saprot import SaProtExpert


def test_importing_experts_does_not_pull_torch():
    # Lazy-import contract: importing the package must not load torch/esm.
    # Run in a CLEAN subprocess so prior test imports don't pollute sys.modules.
    import subprocess
    import sys
    code = ("import sys, protein_chisel.experts as _; "
            "assert 'torch' not in sys.modules, 'torch imported'; "
            "assert 'esm' not in sys.modules, 'esm imported'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_registry_lists_defaults():
    names = available_experts()
    assert "esmc" in names and "saprot" in names


def test_get_expert_types_and_versions():
    e = get_expert("esmc", model_name="esmc_600m")
    s = get_expert("saprot", model_name="saprot_1.3b")
    assert isinstance(e, ESMCExpert) and e.version == "esmc:esmc_600m"
    assert isinstance(s, SaProtExpert) and s.version == "saprot:saprot_1.3b"
    assert e.modality == "sequence" and s.modality == "structure"
    assert e.cache_filename == "esmc_log_probs.npy"
    assert s.cache_filename == "saprot_log_probs.npy"


def test_get_expert_defaults_match_legacy():
    assert get_expert("esmc").model_name == "esmc_300m"
    assert get_expert("saprot").model_name == "saprot_35m"


def test_unknown_expert_raises():
    with pytest.raises(KeyError):
        get_expert("nope")


def test_resolve_experts_csv_and_models():
    experts = resolve_experts("esmc,saprot",
                              model_names={"esmc": "esmc_600m", "saprot": "saprot_1.3b"})
    assert [x.name for x in experts] == ["esmc", "saprot"]
    assert experts[0].version == "esmc:esmc_600m"


def test_resolve_experts_empty_raises():
    with pytest.raises(ValueError):
        resolve_experts("")


def test_esmc_expert_calls_tool(monkeypatch):
    arr = np.zeros((5, 20))
    import protein_chisel.tools.esmc as esmc_mod
    monkeypatch.setattr(esmc_mod, "esmc_logits",
                        lambda seq, **kw: SimpleNamespace(log_probs=arr), raising=False)
    ctx = ExpertContext(seq="ACDEF", pdb_path="x.pdb", chain="A")
    out = get_expert("esmc").log_probs(ctx, use_cache=False)
    assert out.shape == (5, 20)


def test_saprot_expert_calls_tool(monkeypatch):
    arr = np.ones((7, 20))
    import protein_chisel.tools.saprot as saprot_mod
    monkeypatch.setattr(saprot_mod, "saprot_logits",
                        lambda pdb, **kw: SimpleNamespace(log_probs=arr), raising=False)
    ctx = ExpertContext(seq="A" * 7, pdb_path="x.pdb", chain="A")
    out = get_expert("saprot").log_probs(ctx, use_cache=False)
    assert out.shape == (7, 20)


class _FakeExpert(Expert):
    name = "fake"
    modality = "sequence"

    def __init__(self):
        self.calls = 0

    def compute_log_probs(self, ctx):
        self.calls += 1
        return np.arange(20 * 4, dtype=float).reshape(4, 20)


def test_cache_roundtrip(tmp_path):
    exp = _FakeExpert()
    ctx = ExpertContext(seq="ACDE", pdb_path="x.pdb", chain="A", out_dir=tmp_path)
    a = exp.log_probs(ctx)
    assert exp.calls == 1
    assert (tmp_path / "fake_log_probs.npy").exists()
    b = exp.log_probs(ctx)            # second call hits cache, no recompute
    assert exp.calls == 1
    assert np.array_equal(a, b)


def test_bad_shape_raises():
    class _BadExpert(_FakeExpert):
        name = "bad"
        def compute_log_probs(self, ctx):
            return np.zeros((4, 21))   # wrong width
    with pytest.raises(ValueError):
        _BadExpert().log_probs(ExpertContext(seq="ACDE", pdb_path="x", chain="A"),
                               use_cache=False)
