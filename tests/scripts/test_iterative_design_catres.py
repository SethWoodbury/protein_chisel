"""Tests for the generalized catalytic-residue resolution in
``scripts/iterative_design.py``.

The driver auto-derives catalytic resnos from the seed PDB's REMARK 666. When
REMARK 666 is ABSENT it falls back to the hard-coded PTE_i1 defaults
(``DEFAULT_CATRES`` / ``CATALYTIC_HIS_RESNOS``) — correct only for that one
scaffold. These tests pin the opt-in override (``--catalytic_resnos``), the loud
warning on the silent-PTE-fallback path, and the ``--pi_min`` help-text fix.

Import idiom mirrors ``tests/scripts/test_iterative_design_helpers.py``: the
script is imported as a module (its sibling-script imports are defensive sys.path
inserts, so this works on host pytest as long as PYTHONPATH includes src/).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import iterative_design as v2   # noqa: E402


# The hard-coded PTE_i1 fallback values. Pinned here so the byte-identity
# guarantee (resolver returns EXACTLY these when nothing overrides) is asserted
# against a literal, not against whatever the module currently holds.
_PTE_CATRES = (60, 64, 128, 131, 132, 157)
_PTE_HIS = (60, 64, 128, 132)


# ----------------------------------------------------------------------
# REMARK 666 fixtures
# ----------------------------------------------------------------------


def _write_remark666_pdb(path: Path, motifs) -> Path:
    """Write a minimal PDB whose REMARK 666 block declares ``motifs`` =
    iterable of (motif_resname, motif_resno). One coordinate line so the
    parser has a coordinate block to stop at."""
    lines = []
    for i, (resname, resno) in enumerate(motifs, start=1):
        lines.append(
            f"REMARK 666 MATCH TEMPLATE B YYE  209 MATCH MOTIF A "
            f"{resname:<3s} {resno:>3d}  {i}  1\n"
        )
    lines.append(
        "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00  0.00           N\n"
    )
    path.write_text("".join(lines))
    return path


def _write_apo_pdb(path: Path) -> Path:
    """A PDB with NO REMARK 666 (e.g. an apo backbone / non-PTE scaffold)."""
    path.write_text(
        "REMARK   1 just a normal remark, not a 666 motif line\n"
        "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00  0.00           N\n"
    )
    return path


# ----------------------------------------------------------------------
# Argument parser for --catalytic_resnos
# ----------------------------------------------------------------------


def test_parse_catalytic_resnos_arg_basic():
    assert v2._parse_catalytic_resnos_arg("10,20,30") == (10, 20, 30)


def test_parse_catalytic_resnos_arg_whitespace_and_sort():
    # tolerant of spaces; sorted + de-duplicated ascending.
    assert v2._parse_catalytic_resnos_arg(" 30, 10 ,20,10") == (10, 20, 30)


def test_parse_catalytic_resnos_arg_rejects_non_int():
    with pytest.raises(Exception):
        v2._parse_catalytic_resnos_arg("10,foo,30")


def test_parse_catalytic_resnos_arg_rejects_nonpositive():
    with pytest.raises(Exception):
        v2._parse_catalytic_resnos_arg("0,5")


# ----------------------------------------------------------------------
# Resolver: override > REMARK 666 > builtin PTE defaults
# ----------------------------------------------------------------------


def test_resolver_override_beats_builtin(tmp_path, caplog):
    """--catalytic_resnos 10,20,30 is honored over the builtin PTE defaults,
    even when the seed has NO REMARK 666 (so the builtin would otherwise win)."""
    pdb = _write_apo_pdb(tmp_path / "apo.pdb")
    with caplog.at_level(logging.WARNING, logger=v2.LOGGER.name):
        all_resnos, his_resnos, source = v2._resolve_catalytic_resnos(
            override=(10, 20, 30), seed_pdb=pdb,
        )
    assert all_resnos == (10, 20, 30)
    assert source == "override"
    # The override result must NOT be the PTE builtin.
    assert all_resnos != _PTE_CATRES
    # Overriding silences the PTE-fallback warning.
    assert "almost certainly wrong" not in caplog.text


def test_resolver_override_beats_remark666(tmp_path):
    """An explicit override wins even when REMARK 666 IS present (the user is
    deliberately pinning a different set)."""
    pdb = _write_remark666_pdb(
        tmp_path / "seed.pdb", [("HIS", 41), ("LYS", 64), ("ASP", 200)],
    )
    all_resnos, his_resnos, source = v2._resolve_catalytic_resnos(
        override=(10, 20, 30), seed_pdb=pdb,
    )
    assert all_resnos == (10, 20, 30)
    assert source == "override"


def test_resolver_uses_remark666_when_present(tmp_path, caplog):
    """REMARK 666 present, no override -> derive from the seed; HIS subset is the
    HIS-resname motifs only; NO PTE-fallback warning."""
    pdb = _write_remark666_pdb(
        tmp_path / "seed.pdb",
        [("HIS", 41), ("LYS", 64), ("ASP", 200), ("HIS", 187)],
    )
    with caplog.at_level(logging.WARNING, logger=v2.LOGGER.name):
        all_resnos, his_resnos, source = v2._resolve_catalytic_resnos(
            override=None, seed_pdb=pdb,
        )
    assert all_resnos == (41, 64, 187, 200)
    assert his_resnos == (41, 187)
    assert source == "remark_666"
    # The whole point: a real scaffold-derived set must NOT be the PTE builtin,
    # and no scary warning fires for the supported (REMARK-666-present) path.
    assert all_resnos != _PTE_CATRES
    assert "almost certainly wrong" not in caplog.text


def test_resolver_builtin_fallback_is_byte_identical(tmp_path, caplog):
    """No REMARK 666 and no override -> the resolver returns the EXACT PTE_i1
    builtin values (byte-identical: same residues, just warned)."""
    pdb = _write_apo_pdb(tmp_path / "apo.pdb")
    with caplog.at_level(logging.WARNING, logger=v2.LOGGER.name):
        all_resnos, his_resnos, source = v2._resolve_catalytic_resnos(
            override=None, seed_pdb=pdb,
        )
    assert all_resnos == _PTE_CATRES
    assert his_resnos == _PTE_HIS
    assert source == "builtin"


def test_resolver_builtin_fallback_warns_loudly(tmp_path, caplog):
    """The silent-wrong-residue trap is now LOUD: the PTE-fallback path must warn,
    name the PTE-specific residues, flag them as wrong for non-PTE scaffolds, and
    say how to override."""
    pdb = _write_apo_pdb(tmp_path / "apo.pdb")
    with caplog.at_level(logging.WARNING, logger=v2.LOGGER.name):
        v2._resolve_catalytic_resnos(override=None, seed_pdb=pdb)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "expected a WARNING on the PTE builtin-fallback path"
    text = caplog.text
    # Names the PTE-specific residues (so the user can see exactly what got pinned).
    assert "60" in text and "157" in text
    # Flags them as almost certainly wrong off-PTE.
    assert "almost certainly wrong" in text.lower() or "wrong" in text.lower()
    # Tells the user how to fix it.
    assert "--catalytic_resnos" in text


# ----------------------------------------------------------------------
# --help (host, no container) advertises the flag + fixes the help bug
# ----------------------------------------------------------------------


def test_help_advertises_catalytic_resnos():
    """`iterative_design.py --help` exits 0 and mentions --catalytic_resnos."""
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--catalytic_resnos" in proc.stdout


def test_help_works_without_pythonpath():
    """--help must exit 0 and show --catalytic_resnos even with PYTHONPATH unset
    (argparse is reached before any protein_chisel import)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--catalytic_resnos" in proc.stdout


def test_pi_min_help_no_longer_claims_net_charge_has_no_flag():
    """The --pi_min help text used to claim the net-charge band "comes from the
    per-cycle strategy schedule, not a CLI flag". Net-charge bands are now first-
    class CLI flags (--net_charge_min/--net_charge_max), so the help must NOT make
    that stale "not a CLI flag" claim, and the real flag must be advertised."""
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "not a CLI flag" not in proc.stdout
    # the net-charge band is now a genuine, advertised CLI flag
    assert "--net_charge_max" in proc.stdout
    assert "--net_charge_min" in proc.stdout


# ----------------------------------------------------------------------
# run_chisel_design.sh: CATALYTIC_RESNOS -> --catalytic_resnos passthrough
# ----------------------------------------------------------------------


def test_shell_wires_catalytic_resnos_passthrough():
    """run_chisel_design.sh forwards CATALYTIC_RESNOS to --catalytic_resnos."""
    sh = (REPO / "scripts" / "run_chisel_design.sh").read_text()
    assert "CATALYTIC_RESNOS" in sh
    assert "--catalytic_resnos" in sh


@pytest.mark.parametrize(
    "value, expect_flag",
    [
        ("", False),            # unset/empty -> default (byte-identical, no flag)
        ("10,20,30", True),     # set -> forwarded
        ("60,64,128,131,132,157", True),
    ],
)
def test_shell_catalytic_resnos_only_when_set(value, expect_flag):
    """The CATALYTIC_RESNOS snippet appends --catalytic_resnos ONLY when the env
    var is non-empty, so the default invocation is byte-identical. Runs the EXACT
    idiom from the shell file in real bash so it stays pinned to shipped wiring."""
    sh = (REPO / "scripts" / "run_chisel_design.sh").read_text()
    # Extract the committed one-liner that builds CATALYTIC_RESNOS_CLI so the test
    # exercises the real source, not a re-implementation.
    snippet_lines = [
        ln.strip() for ln in sh.splitlines()
        if "CATALYTIC_RESNOS_CLI" in ln and "+=(" in ln
    ]
    assert snippet_lines, "expected a CATALYTIC_RESNOS_CLI builder line in the shell"
    script = (
        'CATALYTIC_RESNOS_CLI=()\n'
        + "\n".join(snippet_lines) + "\n"
        + 'echo "${CATALYTIC_RESNOS_CLI[@]}"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "CATALYTIC_RESNOS": value},
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    got_flag = "--catalytic_resnos" in proc.stdout
    assert got_flag is expect_flag
    if expect_flag:
        assert value in proc.stdout
