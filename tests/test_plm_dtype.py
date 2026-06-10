"""Byte-identity guard for the opt-in --plm_dtype: every PLM entry point must
default to fp32 (so the default pipeline is unchanged). Pure signature inspection —
no torch / model loading."""
from __future__ import annotations

import inspect

from protein_chisel.experts.base import ExpertContext


def _default(func, param):
    return inspect.signature(func).parameters[param].default


def test_tool_dtype_defaults_fp32():
    from protein_chisel.tools import esmc, saprot
    assert _default(esmc.esmc_logits, "dtype") == "fp32"
    assert _default(esmc._load_esmc, "dtype") == "fp32"
    assert _default(saprot.saprot_logits, "dtype") == "fp32"
    assert _default(saprot._saprot_logits_from_sa_string, "dtype") == "fp32"
    assert _default(saprot._load_saprot, "dtype") == "fp32"


def test_expert_context_dtype_default_fp32():
    assert ExpertContext(seq="A", pdb_path="x").plm_dtype == "fp32"


def test_dtype_maps_cover_fp16_bf16():
    from protein_chisel.tools import esmc, saprot
    for d in ("fp16", "bf16"):
        assert d in esmc._TORCH_DTYPE
        assert d in saprot._TORCH_DTYPE
    # saprot maps fp32 too (it passes torch_dtype explicitly); esmc only casts non-fp32
    assert saprot._TORCH_DTYPE["fp32"] == "float32"
