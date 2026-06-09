"""Tests for protein_chisel.scoring.dfi — focus on deterministic per-class column
ordering (the dfi__mean__<class> / dfi__std__<class> columns must not depend on
PYTHONHASHSEED set-iteration order, or the design TSV schema is nondeterministic)."""
from __future__ import annotations

import numpy as np

from protein_chisel.scoring.dfi import compute_dfi


def _write_ca_pdb(path, n=8, chain="A"):
    """Minimal CA-only PDB: n residues on a ~3.8 Å straight backbone trace."""
    lines = []
    for i in range(n):
        x = 3.8 * i
        lines.append(
            f"ATOM  {i+1:5d}  CA  ALA {chain}{i+1:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_per_class_keys_are_sorted(tmp_path):
    pdb = _write_ca_pdb(tmp_path / "trace.pdb", n=8)
    # classes deliberately given in a non-sorted, interleaved order
    classes = ["primary_sphere", "distal_buried", "secondary_sphere",
               "distal_buried", "primary_sphere", "secondary_sphere",
               "distal_buried", "primary_sphere"]
    res = compute_dfi(pdb, classes=classes)
    keys = list(res.per_class_mean.keys())
    assert keys == sorted(set(classes)), f"per-class keys not sorted: {keys}"
    assert list(res.per_class_std.keys()) == keys
    # to_dict emits the per-class columns in the same sorted order
    d = res.to_dict()
    mean_cols = [k for k in d if k.startswith("dfi__mean__")]
    assert mean_cols == [f"dfi__mean__{c}" for c in sorted(set(classes))]


def test_per_class_order_independent_of_input_class_order(tmp_path):
    """Same class assignment in different list order -> identical column order."""
    pdb = _write_ca_pdb(tmp_path / "trace.pdb", n=6)
    a = ["primary_sphere", "distal_buried", "secondary_sphere",
         "primary_sphere", "distal_buried", "secondary_sphere"]
    # a permutation that keeps each resno's class the same is not possible without
    # changing assignments; instead verify the KEY order is sorted regardless of
    # which class happens to appear first in the row list.
    b = ["secondary_sphere", "secondary_sphere", "primary_sphere",
         "distal_buried", "primary_sphere", "distal_buried"]
    ka = list(compute_dfi(pdb, classes=a).per_class_mean.keys())
    kb = list(compute_dfi(pdb, classes=b).per_class_mean.keys())
    assert ka == sorted(ka) and kb == sorted(kb)
    assert ka == kb == ["distal_buried", "primary_sphere", "secondary_sphere"]


def test_no_classes_means_no_per_class(tmp_path):
    pdb = _write_ca_pdb(tmp_path / "trace.pdb", n=5)
    res = compute_dfi(pdb)
    assert res.per_class_mean is None
    d = res.to_dict()
    assert not any(k.startswith("dfi__mean__") for k in d)
    # overall columns still present + finite
    for k in ("dfi__mean", "dfi__std", "dfi__max", "dfi__min"):
        assert k in d and np.isfinite(d[k])
