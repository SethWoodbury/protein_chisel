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

import numpy as np
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
    # The feature flags use the identical truthy guard (veto, sap, +WS-C, +CF-5).
    assert sh.count(_SHELL_TRUTHY_RE) >= 2


# ----------------------------------------------------------------------------
# CF-5: verbose controller trace (opt-in --controller_verbose).
# ----------------------------------------------------------------------------
#
# --help advertises the flag (and must do so WITHOUT PYTHONPATH, since --help
# reaches argparse before any protein_chisel import); the shell CONTROLLER_VERBOSE
# env var maps onto --controller_verbose with the same truthy guard as the other
# opt-in flags. Default OFF => byte-identical (no trace, no new import at runtime).


def test_iterative_design_help_advertises_controller_verbose():
    """`iterative_design.py --help` mentions --controller_verbose (CF-5)."""
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--controller_verbose" in proc.stdout


def test_controller_verbose_help_works_without_pythonpath():
    """CF-5 constraint: --help must exit 0 and show --controller_verbose even with
    PYTHONPATH unset — i.e. the trace import stays inside the opt-in branch, not at
    module/parse time (so a default `--help` never needs the package on the path)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--controller_verbose" in proc.stdout


@pytest.mark.parametrize(
    "value, expect_flag",
    [
        ("0", False), ("false", False), ("off", False), ("", False),
        ("no", False),
        ("1", True), ("true", True), ("YES", True), ("On", True),
    ],
)
def test_shell_controller_verbose_truthiness(value, expect_flag):
    """The run_chisel_design.sh CONTROLLER_VERBOSE snippet maps {1,true,yes,on} ->
    --controller_verbose and everything else -> disabled. Runs the EXACT regex from
    the shell file in real bash so it stays pinned to the shipped wiring."""
    script = (
        'ADAPTIVE_BIAS_CLI=()\n'
        f'[[ "${{CONTROLLER_VERBOSE:-0}}" =~ {_SHELL_TRUTHY_RE} ]] '
        '&& ADAPTIVE_BIAS_CLI+=( --controller_verbose )\n'
        'echo "${ADAPTIVE_BIAS_CLI[@]}"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "CONTROLLER_VERBOSE": value},
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert ("--controller_verbose" in proc.stdout) is expect_flag


def test_shell_controller_verbose_wired_in_source():
    """Pin the CONTROLLER_VERBOSE -> --controller_verbose wiring to the committed
    run_chisel_design.sh so the passthrough can't silently disappear."""
    sh = (REPO / "scripts" / "run_chisel_design.sh").read_text()
    assert "CONTROLLER_VERBOSE" in sh
    assert "--controller_verbose" in sh


# ----------------------------------------------------------------------------
# 6. WS-C composition control: fraction-cap omit builder + flag wiring.
# ----------------------------------------------------------------------------
#
# Guards the opt-in --aa_fraction_cap / --composition_soft_bias /
# --composition_suppress_all_overrep wiring. The fraction-cap omit builder is a
# pure host helper; the flags are smoke-checked via --help and the shell truthy
# guard (mirrors the veto/sap pattern above). All default to a NO-OP so a
# default run is byte-identical.


def test_fraction_cap_omit_builds_nonfixed_designable_omit():
    """26% A / 20% L / 20% G pool with cap 0.15 -> omit {A,G,L} at every
    NON-FIXED designable position; the fixed (catalytic) resno is excluded."""
    pool = "A" * 52 + "L" * 40 + "G" * 40 + "S" * 24 + "T" * 16 + "E" * 14 + "D" * 14
    omit = idz._build_fraction_cap_omit(
        pool, 0.15, protein_resnos=[10, 11, 12, 13], fixed_resnos=[12],
        chain="A", exclude_aas="",
    )
    assert set(omit.keys()) == {"A10", "A11", "A13"}      # A12 fixed -> excluded
    assert all(set(v) == set("AGL") for v in omit.values())


def test_fraction_cap_omit_none_cap_is_empty_noop():
    """cap=None -> {} (the byte-identical default path)."""
    assert idz._build_fraction_cap_omit(
        "AAAA", None, protein_resnos=[1, 2], fixed_resnos=[],
    ) == {}


def test_fraction_cap_omit_nothing_over_cap_is_empty():
    """Uniform 5%-each pool -> nothing over a 15% cap -> {}."""
    pool = "ACDEFGHIKLMNPQRSTVWY" * 5
    assert idz._build_fraction_cap_omit(
        pool, 0.15, protein_resnos=[1, 2], fixed_resnos=[],
    ) == {}


def test_fraction_cap_omit_respects_exclude_aas():
    """An already-omitted AA (C) is never re-capped even at 30% of the pool."""
    pool = "C" * 60 + "A" * 140                            # 30% C, 70% A
    omit = idz._build_fraction_cap_omit(
        pool, 0.15, protein_resnos=[1], fixed_resnos=[], exclude_aas="C",
    )
    assert omit == {f"{idz.CHAIN}1": "A"}


def test_fraction_cap_omit_skips_when_cap_too_low_for_pool():
    """SAFETY (codex/subagent P1): a valid-but-too-low cap that would omit nearly
    every AA must NOT produce an all/most-AA omit (which fused MPNN encodes as
    equal -1e8 -> silently samples from 'forbidden' AAs). The builder skips +
    leaves the omit empty rather than over-constrain the sampler."""
    uniform = "ACDEFGHIKLMNPQRSTVWY" * 10                  # 5% each, 20 AAs
    omit = idz._build_fraction_cap_omit(
        uniform, 0.04, protein_resnos=[1, 2, 3], fixed_resnos=[],
    )
    assert omit == {}                                      # degenerate cap -> no-op


def test_fraction_arg_rejects_out_of_range_and_nonfinite():
    """--aa_fraction_cap validator accepts a finite fraction in (0, 1] and rejects
    0, negatives, >1, nan, inf, and non-numbers."""
    import argparse
    for bad in ("0", "0.0", "-0.1", "1.5", "2", "nan", "inf", "-inf", "abc"):
        with pytest.raises(argparse.ArgumentTypeError):
            idz._fraction_arg(bad)
    assert idz._fraction_arg("0.15") == pytest.approx(0.15)
    assert idz._fraction_arg("1.0") == pytest.approx(1.0)


def test_soft_bias_nats_arg_rejects_nonfinite_and_negative():
    """--composition_soft_bias_nats validator accepts a finite >=0 magnitude and
    rejects nan/inf/negatives (a NaN bias would poison the sampling softmax)."""
    import argparse
    for bad in ("-0.1", "nan", "inf", "abc"):
        with pytest.raises(argparse.ArgumentTypeError):
            idz._nonneg_finite_arg(bad)
    assert idz._nonneg_finite_arg("0") == pytest.approx(0.0)
    assert idz._nonneg_finite_arg("0.5") == pytest.approx(0.5)


def test_nonneg_finite_arg_enforces_max_value():
    """With max_value, an absurd-but-finite magnitude (e.g. 1e100, which casts to
    -inf in the float32 sampler bias) is rejected; the bound is inclusive."""
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        idz._nonneg_finite_arg("1e100", max_value=20.0)
    with pytest.raises(argparse.ArgumentTypeError):
        idz._nonneg_finite_arg("25", max_value=20.0)
    assert idz._nonneg_finite_arg("20", max_value=20.0) == pytest.approx(20.0)
    assert idz._nonneg_finite_arg("0.5", max_value=20.0) == pytest.approx(0.5)


def test_enforce_min_sampleable_reverts_post_merge_overomit():
    """SAFETY (codex re-review P1): the cap omit MERGED with structural omits can
    still leave a position with < N sampleable AAs even though the cap-set alone
    passed. The post-merge guard reverts the offending position to its pre-cap
    (structural-only) omit."""
    canon = "ACDEFGHIKLMNPQRSTVWY"
    cap_aas = canon.replace("C", "").replace("D", "").replace("E", "")  # 17 AAs
    assert len(cap_aas) == 17
    base = {"A5": "DE"}                                  # structural omit (2 AAs)
    merged = {"A5": "".join(sorted(set(cap_aas) | set("DE")))}   # 19 canon omitted
    # global omit "CX" also forbids C -> all 20 canonical omitted at A5 -> degenerate.
    out = idz._enforce_min_sampleable_after_cap(merged, base, "CX", 3)
    assert out["A5"] == "DE"                             # reverted to the structural omit


def test_enforce_min_sampleable_noop_when_safe():
    """When the merged omit leaves >= min_keep AAs, the guard is a pure no-op."""
    base = {"A5": "KR"}
    merged = {"A5": "AKR"}                               # 3 omitted +C global = 4; 16 left
    out = idz._enforce_min_sampleable_after_cap(merged, base, "CX", 3)
    assert out == merged


def test_enforce_min_sampleable_flags_structural_overomit(caplog):
    """If reverting the cap is NOT enough (the structural base itself + global omit
    already leave < min_keep), the guard still reverts the cap AND surfaces the
    pre-existing structural over-omit (it must not silently claim safety it lacks)."""
    import logging
    canon = "ACDEFGHIKLMNPQRSTVWY"
    base_aas = canon.replace("A", "").replace("V", "")  # 18 AAs omitted (leaves A,V)
    base = {"A5": base_aas}
    merged = {"A5": canon.replace("C", "")}             # cap omits 19
    with caplog.at_level(logging.ERROR):
        out = idz._enforce_min_sampleable_after_cap(merged, base, "CX", 3)
    assert out["A5"] == base_aas                        # cap reverted to base
    # base(18) | C(global) = 19 omitted -> 1 left < 3 -> a DISTINCT structural
    # warning (unique phrase) is surfaced, separate from the generic revert log.
    assert "even without the cap" in caplog.text.lower()


def test_enforce_min_sampleable_no_structural_warning_when_base_is_safe(caplog):
    """The structural warning fires ONLY when base+global is itself degenerate —
    a normal cap revert (safe base) must not emit it (guards the wrong-reason pass)."""
    import logging
    canon = "ACDEFGHIKLMNPQRSTVWY"
    cap_aas = canon.replace("C", "").replace("D", "").replace("E", "")  # 17 AAs
    base = {"A5": "DE"}                                  # safe base (2 AAs)
    merged = {"A5": "".join(sorted(set(cap_aas) | set("DE")))}
    with caplog.at_level(logging.ERROR):
        out = idz._enforce_min_sampleable_after_cap(merged, base, "CX", 3)
    assert out["A5"] == "DE"
    assert "even without the cap" not in caplog.text.lower()


def test_cli_rejects_out_of_range_fraction_and_huge_nats():
    """End-to-end argparse rejection: a >1 cap and an absurd nats both exit != 0."""
    for args in (["--aa_fraction_cap", "2"],
                 ["--aa_fraction_cap", "0"],
                 ["--composition_soft_bias_nats", "1e100"]):
        proc = subprocess.run(
            [sys.executable, "scripts/iterative_design.py", "--seed_pdb", "x.pdb",
             *args],
            cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode != 0, f"{args} should be rejected"
        assert "aa_fraction_cap" in proc.stderr or "composition_soft_bias_nats" in proc.stderr


def test_iterative_design_help_advertises_ws_c_flags():
    """`--help` exits 0 and advertises every WS-C opt-in flag."""
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in (
        "--composition_suppress_all_overrep",
        "--aa_fraction_cap",
        "--composition_soft_bias",
        "--composition_soft_bias_nats",
    ):
        assert flag in proc.stdout, flag


@pytest.mark.parametrize(
    "env_var, cli_flag",
    [
        ("COMPOSITION_SUPPRESS_ALL_OVERREP", "--composition_suppress_all_overrep"),
        ("COMPOSITION_SOFT_BIAS", "--composition_soft_bias"),
    ],
)
@pytest.mark.parametrize(
    "value, expect_flag",
    [("0", False), ("false", False), ("", False),
     ("1", True), ("true", True), ("on", True)],
)
def test_shell_ws_c_boolean_truthiness(env_var, cli_flag, value, expect_flag):
    """The WS-C boolean env vars use the same shipped truthy guard as the veto."""
    script = (
        'COMPOSITION_CLI=()\n'
        f'[[ "${{{env_var}:-0}}" =~ {_SHELL_TRUTHY_RE} ]] '
        f'&& COMPOSITION_CLI+=( {cli_flag} )\n'
        'echo "${COMPOSITION_CLI[@]}"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, env_var: value},
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert (cli_flag in proc.stdout) is expect_flag


# ----------------------------------------------------------------------------
# 8. WS-G omit-tunnel-lining helpers.
# ----------------------------------------------------------------------------


def test_build_tunnel_lining_omit_forbids_bulky_at_nonfixed_lining():
    """--omit_tunnel_lining hard-omits bulky/hydrophobic AAs at tunnel-lining
    positions (default WFYHMLIV), skipping fixed/catalytic resnos; Ala is NOT in
    the default set (it's small — it can't constrict the channel)."""
    omit = idz._build_tunnel_lining_omit([50, 51, 52], "A", "WFYHMLIV",
                                         fixed_resnos=[52])
    assert set(omit.keys()) == {"A50", "A51"}            # 52 fixed -> excluded
    assert all(v == "".join(sorted(set("WFYHMLIV"))) for v in omit.values())
    assert "A" not in omit["A50"]


def test_build_tunnel_lining_omit_empty_is_noop():
    assert idz._build_tunnel_lining_omit([], "A", "WFYHMLIV") == {}
    assert idz._build_tunnel_lining_omit([50], "A", "") == {}        # empty aas
    assert idz._build_tunnel_lining_omit([50], "A", "XZ-") == {}     # non-canonical


def test_build_tunnel_lining_omit_rejects_degenerate_set():
    """codex: a too-large omit set would forbid nearly every AA at a lining
    position -> fused-MPNN samples uniformly from the 'forbidden' set. Reject it."""
    with pytest.raises(ValueError):
        idz._build_tunnel_lining_omit([50], "A", "ACDEFGHIKLMNPQRSTVWY")   # all 20
    # A reasonable bulky set is fine.
    assert idz._build_tunnel_lining_omit([50], "A", "FWY")


def test_read_seed_tunnel_lining_from_tsv(tmp_path):
    p = tmp_path / "seed_tunnel_residues.tsv"
    pd.DataFrame({"resno": [10, 11, 12],
                  "is_tunnel_lining": [True, False, True],
                  "min_dist_to_alpha_sphere": [1.0, 9.0, 2.0]}).to_csv(
        p, sep="\t", index=False)
    assert idz._read_seed_tunnel_lining(p) == {10, 12}


def test_read_seed_tunnel_lining_missing_or_empty_is_empty_set(tmp_path):
    assert idz._read_seed_tunnel_lining(tmp_path / "nope.tsv") == set()
    empty = tmp_path / "empty.tsv"
    pd.DataFrame({"resno": [], "is_tunnel_lining": []}).to_csv(
        empty, sep="\t", index=False)
    assert idz._read_seed_tunnel_lining(empty) == set()


def test_clash_bulky_set_includes_lysine_symmetric_with_arg():
    """The always-on graded-clash bias must treat Lys and Arg symmetrically: both
    are long (Cb->NZ ~5.5/6.0 A) and can clash with fixed catalytic atoms. A prior
    call site passed 'YFWHMR' (dropping K while keeping R); the shared
    _CLASH_BULKY_AAS constant is now used by BOTH the function default and the call
    site so they cannot diverge again."""
    import inspect
    assert idz._CLASH_BULKY_AAS == "YFWHMRK"
    assert "K" in idz._CLASH_BULKY_AAS and "R" in idz._CLASH_BULKY_AAS
    # function default uses the shared constant
    assert inspect.signature(
        idz.compute_graded_clash_bias).parameters["bulky_aas"].default == idz._CLASH_BULKY_AAS
    # regression: the K-dropping literal must not reappear anywhere in the driver
    src = (REPO / "scripts" / "iterative_design.py").read_text()
    assert '"YFWHMR"' not in src and "'YFWHMR'" not in src


def test_iterative_design_help_advertises_seed_triage_flags():
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in ("--plm_autoskip_bad_input", "--plm_autoskip_gravy",
                 "--plm_autoskip_max_aa_frac", "--plm_autoskip_hydrophobic_frac"):
        assert flag in proc.stdout


@pytest.mark.parametrize("value, on", [("0", False), ("false", False), ("", False),
                                       ("1", True), ("on", True)])
def test_shell_plm_autoskip_truthiness(value, on):
    script = (
        'PLM_AUTOSKIP_CLI=()\n'
        f'if [[ "${{PLM_AUTOSKIP_BAD_INPUT:-0}}" =~ {_SHELL_TRUTHY_RE} ]]; then\n'
        '  PLM_AUTOSKIP_CLI+=( --plm_autoskip_bad_input )\n'
        'fi\n'
        'echo "${PLM_AUTOSKIP_CLI[@]}"\n'
    )
    proc = subprocess.run(["bash", "-c", script],
                          env={**os.environ, "PLM_AUTOSKIP_BAD_INPUT": value},
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert ("--plm_autoskip_bad_input" in proc.stdout) is on


def test_ws_g_default_constant_matches_throat_bulky_set():
    """The driver's tunnel-lining omit default is a guard-tested literal (NOT an
    import-time call), so arg-parsing never imports protein_chisel — but it MUST
    stay equal to the throat's bulky-blocker set (single source of truth)."""
    from protein_chisel.tools.tunnel_metrics import bulky_blocker_aas
    assert idz._OMIT_TUNNEL_LINING_DEFAULT == bulky_blocker_aas(0.70)
    assert idz._OMIT_TUNNEL_LINING_DEFAULT == "FHKRWY"


def test_help_works_without_protein_chisel_importable():
    """--help must not require protein_chisel on sys.path (arg-parsing is pure
    argparse). Regression guard: a tunnel_metrics import at parse time broke this."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--omit_tunnel_lining" in proc.stdout


def test_iterative_design_help_advertises_ws_g_flag():
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--omit_tunnel_lining" in proc.stdout
    # WS-G default = the throat's bulky-blocker set (_BLOCKER_WEIGHT >= 0.70):
    # aromatics W/F/Y/H plus the long charged R/K. Lysine MUST be in it (Cb->NZ
    # ~5.5 A — a genuine channel constrictor), same as the throat-feedback bias.
    from protein_chisel.tools.tunnel_metrics import bulky_blocker_aas
    default_set = bulky_blocker_aas(0.70)
    assert default_set == "FHKRWY"
    assert "K" in default_set and "R" in default_set
    assert default_set in proc.stdout                 # rendered via %(default)s


@pytest.mark.parametrize("value, on", [("0", False), ("false", False), ("", False),
                                       ("1", True), ("on", True)])
def test_shell_omit_tunnel_lining_truthiness(value, on):
    script = (
        'WS_E_CLI=()\n'
        f'if [[ "${{OMIT_TUNNEL_LINING:-0}}" =~ {_SHELL_TRUTHY_RE} ]]; then\n'
        '  WS_E_CLI+=( --omit_tunnel_lining )\n'
        '  [[ -n "${OMIT_TUNNEL_LINING_AAS:-}" ]] && WS_E_CLI+=( --omit_tunnel_lining_aas "$OMIT_TUNNEL_LINING_AAS" )\n'
        'fi\n'
        'echo "${WS_E_CLI[@]}"\n'
    )
    proc = subprocess.run(["bash", "-c", script],
                          env={**os.environ, "OMIT_TUNNEL_LINING": value},
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert ("--omit_tunnel_lining" in proc.stdout) is on


# ----------------------------------------------------------------------------
# 7. WS-E sampling-core safety helpers.
# ----------------------------------------------------------------------------


def test_clamp_bias_total_clips_summed_bias():
    """--bias_total_clamp bounds the summed per-(pos,AA) bias_k (the uncapped
    consensus+PLM stack) to ±N nats (bias_AA_vec=None => bias_k only)."""
    bias = np.array([[5.0, -4.0, 0.1], [1.0, -1.0, 0.0]], dtype=np.float32)
    out = idz._clamp_bias_total(bias, 2.0)
    assert float(out.max()) == 2.0 and float(out.min()) == -2.0
    assert float(out[1, 0]) == 1.0                 # within-band cells untouched
    assert out.dtype == np.float32                 # dtype preserved


def test_clamp_bias_total_bounds_the_EFFECTIVE_bias_with_global_bias_AA():
    """codex: LigandMPNN adds bias_per_residue (bias_k) AND the global bias_AA
    separately, so the clamp must bound the EFFECTIVE sum bias_k+bias_AA. After
    clamping, bias_k+g must lie within ±N (and a cell already in-band is unchanged)."""
    bias_k = np.array([[2.0, 0.0, -2.0]], dtype=np.float32)
    g = np.array([2.5, 0.1, -2.5], dtype=np.float32)        # global per-AA bias
    out = idz._clamp_bias_total(bias_k, 3.0, bias_AA_vec=g)
    eff = out + g
    assert np.all(eff <= 3.0 + 1e-6) and np.all(eff >= -3.0 - 1e-6)
    assert eff[0, 0] == pytest.approx(3.0)         # 2.0+2.5=4.5 -> clamped to 3.0
    assert out[0, 1] == pytest.approx(0.0)         # 0.0+0.1 in band -> bias_k unchanged


def test_clamp_bias_total_none_is_identity_noop():
    """clamp=None returns the SAME array object (byte-identical default path)."""
    bias = np.array([[5.0, -4.0]], dtype=np.float32)
    assert idz._clamp_bias_total(bias, None) is bias
    assert idz._clamp_bias_total(bias, None, bias_AA_vec=np.zeros(2)) is bias


def test_iterative_design_help_advertises_ws_e_flags():
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in ("--bias_total_clamp", "--sampling_temperature_floor",
                 "--plm_class_strength"):
        assert flag in proc.stdout, flag


def test_cli_rejects_nonfinite_plm_strength():
    """--plm_strength nan/inf is rejected at startup (codex: the <0/>5 checks let
    NaN through)."""
    for bad in ("nan", "inf"):
        proc = subprocess.run(
            [sys.executable, "scripts/iterative_design.py", "--seed_pdb", "x.pdb",
             "--plm_strength", bad],
            cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode != 0
        assert "plm_strength" in proc.stderr


def test_parse_plm_class_strength_parses_and_validates():
    """--plm_class_strength k=v,...: absolute per-class overrides; reject unknown
    class names, NaN/inf, and negatives (codex)."""
    assert idz._parse_plm_class_strength("") == {}
    assert idz._parse_plm_class_strength("distal_surface=0.3,primary_sphere=0.0") == {
        "distal_surface": 0.3, "primary_sphere": 0.0}
    for bad in ("distal_surface=-0.1", "distal_surface=nan", "distal_surface=inf",
                "bogus_class=0.3", "distal_surface", "distal_surface=x",
                "surface=0.3", "buried=0.3"):     # legacy keys -> silent no-op -> reject
        with pytest.raises(ValueError):
            idz._parse_plm_class_strength(bad)


def test_parse_charge_band_arg_requires_exactly_two_floats():
    """--adaptive_charge_band must be exactly 'LO,HI' floats (codex: a 3-field value
    was silently truncated)."""
    assert idz._parse_charge_band_arg("-15,-5") == (-15.0, -5.0)
    for bad in ("-15,-5,0", "-15", "", "a,b", "-15,"):
        with pytest.raises(ValueError):
            idz._parse_charge_band_arg(bad)


def test_iterative_design_help_advertises_ws_d_flags():
    """`--help` exits 0 and advertises every WS-D controller-expansion flag."""
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in ("--adaptive_surface_sasa_gate", "--adaptive_charge_band",
                 "--adaptive_bias_axes"):
        assert flag in proc.stdout, flag


def test_shell_ws_d_value_passthrough():
    """The WS-D env vars emit their CLI flags only when set AND ADAPTIVE_BIAS!=0;
    unset emits nothing (byte-identical default)."""
    snippet = (
        'ADAPTIVE_BIAS_CLI=()\n'
        'if [[ "${ADAPTIVE_BIAS:-0}" != "0" ]]; then\n'
        '  [[ -n "${ADAPTIVE_SURFACE_SASA_GATE:-}" ]] && ADAPTIVE_BIAS_CLI+=( --adaptive_surface_sasa_gate "$ADAPTIVE_SURFACE_SASA_GATE" )\n'
        '  [[ -n "${ADAPTIVE_CHARGE_BAND:-}" ]] && ADAPTIVE_BIAS_CLI+=( --adaptive_charge_band "$ADAPTIVE_CHARGE_BAND" )\n'
        '  [[ -n "${ADAPTIVE_BIAS_AXES:-}" ]] && ADAPTIVE_BIAS_CLI+=( --adaptive_bias_axes "$ADAPTIVE_BIAS_AXES" )\n'
        'fi\n'
        'echo "${ADAPTIVE_BIAS_CLI[@]}"\n'
    )
    # Pin the snippet to the shipped shell file.
    sh = (REPO / "scripts" / "run_chisel_design.sh").read_text()
    assert '[[ -n "${ADAPTIVE_SURFACE_SASA_GATE:-}" ]]' in sh
    assert '[[ -n "${ADAPTIVE_CHARGE_BAND:-}"       ]]' in sh
    set_env = {**os.environ, "ADAPTIVE_BIAS": "1", "ADAPTIVE_SURFACE_SASA_GATE": "0.20",
               "ADAPTIVE_CHARGE_BAND": "-15,-5", "ADAPTIVE_BIAS_AXES": "charge"}
    got = subprocess.run(["bash", "-c", snippet], env=set_env,
                         capture_output=True, text=True, timeout=30)
    assert got.stdout.strip() == (
        "--adaptive_surface_sasa_gate 0.20 --adaptive_charge_band -15,-5 "
        "--adaptive_bias_axes charge")
    # ADAPTIVE_BIAS unset => the whole block is skipped (no WS-D flags).
    off_env = {k: v for k, v in os.environ.items()
               if k not in ("ADAPTIVE_BIAS", "ADAPTIVE_SURFACE_SASA_GATE",
                            "ADAPTIVE_CHARGE_BAND", "ADAPTIVE_BIAS_AXES")}
    off_env["ADAPTIVE_SURFACE_SASA_GATE"] = "0.20"      # set but ADAPTIVE_BIAS off
    off = subprocess.run(["bash", "-c", snippet], env=off_env,
                        capture_output=True, text=True, timeout=30)
    assert off.stdout.strip() == ""


def test_shell_aa_fraction_cap_value_passthrough():
    """AA_FRACTION_CAP=<frac> emits `--aa_fraction_cap <frac>`; unset emits
    nothing (byte-identical default)."""
    snippet = (
        'COMPOSITION_CLI=()\n'
        '[[ -n "${AA_FRACTION_CAP:-}" ]] '
        '&& COMPOSITION_CLI+=( --aa_fraction_cap "$AA_FRACTION_CAP" )\n'
        'echo "${COMPOSITION_CLI[@]}"\n'
    )
    # Verify this exact snippet is present in the shipped shell file.
    assert '[[ -n "${AA_FRACTION_CAP:-}" ]]' in (
        REPO / "scripts" / "run_chisel_design.sh"
    ).read_text()
    set_proc = subprocess.run(
        ["bash", "-c", snippet],
        env={**os.environ, "AA_FRACTION_CAP": "0.15"},
        capture_output=True, text=True, timeout=30,
    )
    assert set_proc.stdout.strip() == "--aa_fraction_cap 0.15"
    env_unset = {k: v for k, v in os.environ.items() if k != "AA_FRACTION_CAP"}
    unset_proc = subprocess.run(
        ["bash", "-c", snippet], env=env_unset,
        capture_output=True, text=True, timeout=30,
    )
    assert unset_proc.stdout.strip() == ""
