"""Tests for protein_chisel.provenance (run version-control record)."""
from __future__ import annotations

import json

from protein_chisel import __version__
from protein_chisel.provenance import RunProvenance


def _prov(**kw):
    base = dict(experts=["esmc", "saprot"],
                expert_versions={"esmc": "esmc:esmc_600m", "saprot": "saprot:saprot_1.3b"},
                fusion_version="fusion-v1")
    base.update(kw)
    return RunProvenance(**base)


def test_defaults_record_addons_off():
    p = _prov()
    d = p.to_manifest_dict()
    assert d["chisel_version"] == __version__
    assert d["mpnn_backend"] == "bias"
    assert d["plm_refresh_rounds"] == 0
    assert d["conserve_hbonds"] is False
    assert d["conserve_depth"] == 1
    assert d["conserve_interaction_types"] == ["hbond"]
    assert d["hermes_model_version"] is None
    # metric/filter registry selection defaults to the full set
    assert d["metrics_selection"] == "all"
    assert d["filters_selection"] == "all"
    assert d["active_metrics"] == []


def test_metric_selection_recorded():
    p = _prov(metrics_selection="fitness,fpocket",
              filters_selection="instability",
              active_metrics=["fitness", "fpocket"])
    d = p.to_manifest_dict()
    assert d["metrics_selection"] == "fitness,fpocket"
    assert d["filters_selection"] == "instability"
    assert d["active_metrics"] == ["fitness", "fpocket"]


def test_manifest_dict_is_json_serializable():
    json.dumps(_prov().to_manifest_dict())   # must not raise


def test_remark_line_format_and_parse():
    p = _prov(conserve_hbonds=True, conserve_depth=2, conserve_seed_base=123)
    line = p.to_remark_line()
    assert line.startswith("REMARK PROVENANCE ") and line.endswith("\n")
    body = line[len("REMARK PROVENANCE "):].strip()
    parsed = json.loads(body)
    assert parsed["conserve_depth"] == 2 and parsed["conserve_seed_base"] == 123
    assert parsed["experts"] == ["esmc", "saprot"]


def test_write_json_roundtrip(tmp_path):
    p = _prov(mpnn_backend="poe", plm_refresh_rounds=2)
    out = tmp_path / "provenance.json"
    p.write_json(out)
    loaded = json.loads(out.read_text())
    assert loaded["mpnn_backend"] == "poe" and loaded["plm_refresh_rounds"] == 2
    assert loaded["expert_versions"]["esmc"] == "esmc:esmc_600m"
