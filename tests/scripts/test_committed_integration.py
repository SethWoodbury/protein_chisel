"""Host-runnable integration tests for the solubility-steering workstreams.

These guard already-committed work on ``feat/solubility-steering`` (see git log:
WS-A hard solubility veto, the shared ``scoring/sap.py`` + opt-in corrected SAP,
the configurable ``CHISEL_SUFFIX`` / ``--design_token`` naming, and the codex-P1
8-element ``_struct_filter_worker`` tuple). Everything here runs on the HOST with
NO freesasa / GPU / apptainer / cluster / network: synthetic data, tiny temp PDBs,
a mocked ``freesasa`` module, and ``subprocess`` for CLI/shell smokes.

Import idiom mirrors ``tests/scripts/test_iterative_design_helpers.py``: the script
is imported as a module (its sibling imports are defensive ``sys.path`` inserts, so
host pytest works as long as ``src/`` is importable).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import math

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import iterative_design as idz  # noqa: E402


# A minimal but parseable single-atom PDB line (chain A, ALA 1, CA). The writer
# only ``shutil.copy2``s these, and the worker short-circuits before reading them,
# so the content just needs to be a real file on disk.
_TINY_PDB = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    "END\n"
)


# ----------------------------------------------------------------------------
# 1. P1 REGRESSION: worker tuple arity + empty-row schema gating.
# ----------------------------------------------------------------------------
#
# Guards commit 5770f87 "fix(sap): thread sap_corrected to the 2nd worker-tuple
# producer (codex P1 crash)". The worker now unpacks an 8-element tuple
# ``(cid, pdb, cat_his, fixed, clash_severe_distance, sap_max_threshold,
#   seed_dfi_metrics, sap_corrected)``. Passing the old 7-tuple is the exact crash
# class the fix addressed, and the missing-PDB short-circuit must emit a
# schema-consistent empty row WITHOUT touching freesasa.


def test_worker_8tuple_pdb_none_short_circuits_no_raise():
    """8-tuple + pdb=None: returns (cid, row, [], [reason]); sap_* are NaN.

    The missing-PDB branch runs no freesasa / no GPU, so it is host-safe, and it
    must not raise (catches the 7-vs-8 arity crash class)."""
    cid, row, hbonds, reasons = idz._struct_filter_worker((
        "cid_a", None, [50], [50], 1.5, 1.35, None, False,
    ))
    assert cid == "cid_a"
    assert hbonds == []
    assert len(reasons) == 1 and reasons[0].startswith("pdb_missing")
    assert math.isnan(row["sap_max"])
    assert math.isnan(row["sap_mean"])
    assert math.isnan(row["sap_p95"])
    # Empty row carries the full schema so a missing-PDB row never NaN-leaks.
    assert row["_passed"] is False
    assert row["clash__detail"] == ""
    assert "ligand_int__n_total" in row


def test_worker_empty_row_sap_corr_gated_by_flag():
    """sap_corr_* keys are ABSENT when sap_corrected=False, PRESENT (NaN) when True.

    This is the byte-identity contract: the corrected columns only materialize
    under the opt-in flag, on the missing-PDB path too."""
    _, row_off, _, _ = idz._struct_filter_worker((
        "cid_off", None, [50], [50], 1.5, 1.35, None, False,
    ))
    assert "sap_corr_max" not in row_off
    assert "sap_corr_mean" not in row_off
    assert "sap_corr_p95" not in row_off

    _, row_on, _, _ = idz._struct_filter_worker((
        "cid_on", None, [50], [50], 1.5, 1.35, None, True,
    ))
    for k in ("sap_corr_max", "sap_corr_mean", "sap_corr_p95"):
        assert k in row_on
        assert math.isnan(row_on[k])


def test_worker_7tuple_raises_valueerror_documents_contract():
    """A legacy 7-tuple must raise ValueError (not silently mis-bind).

    Documents the worker's arity contract: the producer and consumer must agree
    on 8 elements (the codex-P1 bug was a 7-element producer)."""
    with pytest.raises(ValueError):
        idz._struct_filter_worker((
            "cid_x", None, [50], [50], 1.5, 1.35, None,  # only 7 elements
        ))


# ----------------------------------------------------------------------------
# 2. Veto -> writer END-TO-END.
# ----------------------------------------------------------------------------
#
# Guards commits a8234d7 / 1788c8f (opt-in hard solubility veto) + the pure
# ``_write_final_topk_artifacts`` writer. With the veto ENABLED, an out-of-band
# (GRAVY=1.05) design must never ship; with it DISABLED, the frame is returned
# unchanged (byte-identical) and all rows ship.


def _build_top_and_pdb_map(tmp: Path):
    """3 rows (2 in-band, 1 GRAVY=1.05) + a pdb_map keyed by source_id.

    ``_write_final_topk_artifacts`` resolves PDBs via ``_row_source_id`` (which
    prefers ``source_id`` over ``id``) and copies to ``{id}.pdb``."""
    specs = [
        ("d0", "s_ok1", -0.5, -10.0),   # in band
        ("d1", "s_ok2", 0.10, -8.0),    # in band
        ("d2", "s_bad", 1.05, -10.0),   # GRAVY out of band -> must be vetoed
    ]
    rows = []
    pdb_map: dict[str, Path] = {}
    for rid, sid, gravy, charge in specs:
        p = tmp / f"{sid}.pdb"
        p.write_text(_TINY_PDB)
        pdb_map[sid] = p
        rows.append({
            "id": rid, "source_id": sid, "sequence": "ACDE",
            "gravy": gravy, "net_charge_full_HH": charge,
        })
    return pd.DataFrame(rows), pdb_map


def test_veto_enabled_drops_outofband_then_writer_ships_only_inband(tmp_path):
    """Veto ON: only the 2 in-band rows reach topk.{tsv,fasta}/topk_pdbs; the
    GRAVY=1.05 id is gone; selection__solubility_passed present + all True."""
    top, pdb_map = _build_top_and_pdb_map(tmp_path)
    top2 = idz._apply_solubility_veto(
        top, enabled=True, gravy_min=-0.8, gravy_max=0.3,
        net_charge_min=-18.0, net_charge_max=-4.0,
    )
    assert len(top2) == 2
    assert "d2" not in set(top2["id"])
    assert "selection__solubility_passed" in top2.columns
    assert bool(top2["selection__solubility_passed"].all())

    final_dir = tmp_path / "final"
    final_dir.mkdir()
    tsv, mat = idz._write_final_topk_artifacts(
        top=top2, final_dir=final_dir, pdb_map=pdb_map, seed_pdb=None,
    )
    # Only the 2 in-band designs ship, across every artifact.
    assert len(mat) == 2
    assert set(mat["id"]) == {"d0", "d1"}
    written = pd.read_csv(tsv, sep="\t")
    assert set(written["id"].astype(str)) == {"d0", "d1"}
    pdbs = sorted((final_dir / "topk_pdbs").glob("*.pdb"))
    assert {p.stem for p in pdbs} == {"d0", "d1"}
    fasta = (final_dir / "topk.fasta").read_text()
    assert fasta.count(">") == 2
    assert ">d2" not in fasta


def test_veto_disabled_is_byte_identical_passthrough(tmp_path):
    """Veto OFF (default): returns the frame UNCHANGED (same object, no new
    column) and the writer ships all 3 rows -> byte-identity guarantee."""
    top, pdb_map = _build_top_and_pdb_map(tmp_path)
    top3 = idz._apply_solubility_veto(
        top, enabled=False, gravy_min=None, gravy_max=None,
        net_charge_min=None, net_charge_max=None,
    )
    # Unchanged: same object, no solubility column added.
    assert top3 is top
    assert "selection__solubility_passed" not in top3.columns

    final_dir = tmp_path / "final"
    final_dir.mkdir()
    _tsv, mat = idz._write_final_topk_artifacts(
        top=top3, final_dir=final_dir, pdb_map=pdb_map, seed_pdb=None,
    )
    assert len(mat) == 3
    assert set(mat["id"]) == {"d0", "d1", "d2"}
    assert "selection__solubility_passed" not in mat.columns


def test_veto_enabled_missing_band_bound_raises(tmp_path):
    """enabled=True but a band bound is None -> ValueError (caller must pass all
    four). Guards the explicit contract check in _apply_solubility_veto."""
    top, _ = _build_top_and_pdb_map(tmp_path)
    with pytest.raises(ValueError):
        idz._apply_solubility_veto(
            top, enabled=True, gravy_min=-0.8, gravy_max=0.3,
            net_charge_min=None, net_charge_max=-4.0,
        )


# ----------------------------------------------------------------------------
# 3. freesasa-MOCKED _compute_sap_proxy.
# ----------------------------------------------------------------------------
#
# Guards commit 7a9b812 "feat(sap): shared scoring/sap.py + opt-in corrected
# sap_corr_*". A fake ``freesasa`` lets us exercise the real proxy on the host.
# The scenario (an exposed ASP between exposed ALA/LEU) demonstrates the corrected
# weight's whole point: the signed-KD legacy proxy lets the polar D (KD -3.5)
# CANCEL its hydrophobic neighbours, while the centered/zero-clamped corrected
# weight does not -> sap_corr_max > sap_max.


class _FakeFreesasaStructure:
    """Models ~3 exposed residues (chain A): ALA, ASP, LEU, each as a single CA
    atom. Exposes exactly the freesasa.Structure surface _compute_sap_proxy uses:
    nAtoms / chainLabel / residueNumber / residueName / atomName / coord."""

    # CAs spaced 3 Å apart so all three are within the 10 Å SAP neighbourhood.
    _ATOMS = [
        {"chain": "A", "resno": 1, "resname": "ALA", "atom": "CA",
         "xyz": (0.0, 0.0, 0.0), "area": 110.0},
        {"chain": "A", "resno": 2, "resname": "ASP", "atom": "CA",
         "xyz": (3.0, 0.0, 0.0), "area": 170.0},
        {"chain": "A", "resno": 3, "resname": "LEU", "atom": "CA",
         "xyz": (6.0, 0.0, 0.0), "area": 175.0},
        # A residue on a DIFFERENT chain that must be filtered out (CHAIN == "A").
        {"chain": "B", "resno": 1, "resname": "PHE", "atom": "CA",
         "xyz": (0.0, 0.0, 0.0), "area": 200.0},
    ]

    def __init__(self, path):  # freesasa.Structure(str(pdb_path))
        self._path = path

    def nAtoms(self):
        return len(self._ATOMS)

    def chainLabel(self, i):
        return self._ATOMS[i]["chain"]

    def residueNumber(self, i):
        # freesasa returns a (sometimes space-padded) string; proxy .strip()s it.
        return f" {self._ATOMS[i]['resno']} "

    def residueName(self, i):
        return self._ATOMS[i]["resname"]

    def atomName(self, i):
        return f" {self._ATOMS[i]['atom']} "

    def coord(self, i):
        return self._ATOMS[i]["xyz"]


class _FakeFreesasaResult:
    def __init__(self, struct):
        self._struct = struct

    def atomArea(self, i):
        return self._struct._ATOMS[i]["area"]


def _install_fake_freesasa(monkeypatch):
    fake = types.ModuleType("freesasa")
    fake.silent = 0
    fake.setVerbosity = lambda _v: None
    fake.Structure = _FakeFreesasaStructure
    fake.calc = lambda struct: _FakeFreesasaResult(struct)
    monkeypatch.setitem(sys.modules, "freesasa", fake)


def test_sap_proxy_raw_only_when_uncorrected(monkeypatch, tmp_path):
    """corrected=False returns ONLY sap_max/mean/p95 (no sap_corr_*)."""
    _install_fake_freesasa(monkeypatch)
    p = tmp_path / "design.pdb"
    p.write_text(_TINY_PDB)
    out = idz._compute_sap_proxy(p, corrected=False)
    assert out is not None
    assert set(out.keys()) == {"sap_max", "sap_mean", "sap_p95"}
    assert not any(k.startswith("sap_corr") for k in out)


def test_sap_proxy_corrected_removes_polar_cancellation(monkeypatch, tmp_path):
    """corrected=True adds sap_corr_* and sap_corr_max > sap_max.

    The exposed ASP cancels hydrophobic neighbours under the signed-KD legacy
    weight but contributes 0 under the centered corrected weight, so the corrected
    surface SAP is strictly higher -- the audit's whole motivation."""
    _install_fake_freesasa(monkeypatch)
    p = tmp_path / "design.pdb"
    p.write_text(_TINY_PDB)
    out = idz._compute_sap_proxy(p, corrected=True)
    assert out is not None
    for k in ("sap_corr_max", "sap_corr_mean", "sap_corr_p95"):
        assert k in out
    assert out["sap_corr_max"] > out["sap_max"]


# ----------------------------------------------------------------------------
# 4. finalize CLI subprocess (WS-H: configurable CHISEL_SUFFIX / --design_token).
# ----------------------------------------------------------------------------
#
# Guards commit 01adb48 "feat(naming): configurable CHISEL_SUFFIX / --design_token".
# Runs the real CLI in a plain python subprocess (no container) on a tiny
# published run dir and asserts the rank-ordered rename token. We use 11 designs
# so the zero-padded width is 2 (``len(str(n-1)) == 2``), giving 2-digit indices.


def _build_published_run(tmp: Path, n_designs: int = 11):
    """A minimal published run dir: N ``<stem>_chisel_<i>.pdb`` design files + a
    chiseled_design_metrics.tsv with id / pdb_path / is_input. Plus one is_input
    row that must NOT be renamed."""
    designs = tmp
    rows = []
    # The input/reference row (is_input=true) — never renamed.
    seed = designs / "myseed_input.pdb"
    seed.write_text(_TINY_PDB)
    rows.append({"id": "myseed_input", "pdb_path": str(seed), "is_input": "true"})
    # N design rows, each a real file named with the legacy chisel token.
    for i in range(n_designs):
        name = f"myseed_chisel_{i}"
        p = designs / f"{name}.pdb"
        p.write_text(_TINY_PDB)
        rows.append({"id": name, "pdb_path": str(p), "is_input": "false"})
    tsv = designs / "chiseled_design_metrics.tsv"
    pd.DataFrame(rows).to_csv(tsv, sep="\t", index=False)
    return designs, tsv, n_designs


def _run_finalize(final_root: Path, suffix: str | None):
    env = dict(os.environ)
    env.pop("CHISEL_SUFFIX", None)
    if suffix is not None:
        env["CHISEL_SUFFIX"] = suffix
    return subprocess.run(
        [sys.executable, "scripts/finalize_design_names.py",
         "--final_root", str(final_root)],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120,
    )


def test_finalize_cli_custom_chisel_suffix(tmp_path):
    """CHISEL_SUFFIX=chiselX -> exit 0 and PDBs renamed <stem>_chiselX_NN.pdb."""
    designs, _tsv, n = _build_published_run(tmp_path)
    proc = _run_finalize(designs, suffix="chiselX")
    assert proc.returncode == 0, proc.stderr
    renamed = sorted(p.name for p in designs.glob("myseed_chiselX_*.pdb"))
    assert len(renamed) == n
    # Width 2 for 11 designs (indices 00..10).
    assert "myseed_chiselX_00.pdb" in renamed
    assert "myseed_chiselX_10.pdb" in renamed
    # The is_input reference is untouched (no chiselX rename).
    assert (designs / "myseed_input.pdb").is_file()
    # No legacy-token files survive (all swapped to the custom token).
    assert not list(designs.glob("myseed_chisel_*.pdb"))


def test_finalize_cli_default_suffix_is_chisel(tmp_path):
    """No CHISEL_SUFFIX -> default token 'chisel' -> <stem>_chisel_NN.pdb
    (byte-identical to legacy naming)."""
    designs, _tsv, n = _build_published_run(tmp_path)
    proc = _run_finalize(designs, suffix=None)
    assert proc.returncode == 0, proc.stderr
    renamed = sorted(p.name for p in designs.glob("myseed_chisel_*.pdb"))
    assert len(renamed) == n
    assert "myseed_chisel_00.pdb" in renamed
    assert "myseed_chisel_10.pdb" in renamed


# ----------------------------------------------------------------------------
# 5. argparse + shell wiring smokes.
# ----------------------------------------------------------------------------


def test_iterative_design_help_advertises_new_flags():
    """`iterative_design.py --help` exits 0 on the host and its usage mentions
    --ship_solubility_veto and --sap_corrected (the opt-in feature flags)."""
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--ship_solubility_veto" in proc.stdout
    assert "--sap_corrected" in proc.stdout


# The exact truthy regex extracted verbatim from run_chisel_design.sh
# (lines ~482 / 488): the env var enables the flag ONLY when it matches
# 1/true/yes/on (case-insensitive); everything else (0/false/off/empty) disables.
_SHELL_TRUTHY_RE = r"^([Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn]|1)$"


@pytest.mark.parametrize(
    "value, expect_flag",
    [
        ("0", False), ("false", False), ("False", False), ("off", False),
        ("OFF", False), ("", False), ("no", False),
        ("1", True), ("true", True), ("TRUE", True), ("yes", True),
        ("YES", True), ("on", True), ("On", True),
    ],
)
def test_shell_ship_veto_truthiness(value, expect_flag):
    """The run_chisel_design.sh truthiness snippet maps SHIP_SOLUBILITY_VETO in
    {0,false,off,...} -> disabled and {1,true,yes,on} -> --ship_solubility_veto.

    Runs the EXACT regex from the shell file in a real bash subprocess, so this
    stays pinned to the shipped wiring (not a Python re-implementation)."""
    script = (
        'SHIP_SOLUBILITY_VETO_CLI=()\n'
        f'[[ "${{SHIP_SOLUBILITY_VETO:-0}}" =~ {_SHELL_TRUTHY_RE} ]] '
        '&& SHIP_SOLUBILITY_VETO_CLI+=( --ship_solubility_veto )\n'
        'echo "${SHIP_SOLUBILITY_VETO_CLI[@]}"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "SHIP_SOLUBILITY_VETO": value},
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    got_flag = "--ship_solubility_veto" in proc.stdout
    assert got_flag is expect_flag


def test_shell_regex_matches_committed_source():
    """Pin _SHELL_TRUTHY_RE to the actual run_chisel_design.sh source so this test
    fails loudly if the shell wiring's regex ever drifts from what we assert."""
    sh = (REPO / "scripts" / "run_chisel_design.sh").read_text()
    assert _SHELL_TRUTHY_RE in sh
    # Both feature flags use the identical truthy guard.
    assert sh.count(_SHELL_TRUTHY_RE) >= 2
