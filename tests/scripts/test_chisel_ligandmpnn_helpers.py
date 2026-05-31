"""Tests for pure-python helpers in scripts/chisel_ligandMPNN.py."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import chisel_ligandMPNN as w  # noqa: E402


def test_parse_probability_decimal():
    assert w.parse_probability(0.8) == 0.8
    assert w.parse_probability(0.0) == 0.0
    assert w.parse_probability(1.0) == 1.0


def test_parse_probability_percentage(caplog):
    assert w.parse_probability(80) == pytest.approx(0.8)
    assert w.parse_probability(100) == pytest.approx(1.0)


def test_parse_probability_errors():
    with pytest.raises(SystemExit):
        w.parse_probability(150)
    with pytest.raises(SystemExit):
        w.parse_probability(-0.1)


def test_roll_conserved_bounds():
    labels = ["A10", "A20", "A30"]
    assert w._roll_conserved(labels, 1.0, random.Random(0)) == set(labels)
    assert w._roll_conserved(labels, 0.0, random.Random(0)) == set()


def test_roll_conserved_seed_reproducible():
    labels = ["A10", "A20", "A30", "A40", "A50"]
    a = w._roll_conserved(labels, 0.5, random.Random("seed:1"))
    b = w._roll_conserved(labels, 0.5, random.Random("seed:1"))
    assert a == b                      # same seed -> identical
    assert a.issubset(set(labels))


def test_fixed_residues_json(tmp_path):
    out = tmp_path / "fx.json"
    w._fixed_residues_json("/abs/x.pdb", ["A54", "A11", "A54"], out)
    data = json.loads(out.read_text())
    assert list(data.keys()) == [str(Path("/abs/x.pdb").resolve())]
    # de-duped and sorted by (chain, resno)
    assert data[str(Path("/abs/x.pdb").resolve())] == ["A11", "A54"]


def test_label_sort_key():
    assert sorted(["A100", "A9", "A11"], key=w._label_sort_key) == ["A9", "A11", "A100"]
