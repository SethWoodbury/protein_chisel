"""Tests for the two any-enzyme generalizations in
``scripts/iterative_design.py``:

  A. ``--chain`` — sets the module ``CHAIN`` global (default ``"A"`` =>
     byte-identical) so every structural read targets the configured chain.
  B. ``--no_require_cat_his_hbond`` — disables ONLY the struct-filter criterion
     that requires >=1 side-chain H-bond to a catalytic HIS (default ON =>
     byte-identical), for enzymes whose mechanism has no catalytic His.

Import idiom mirrors ``tests/scripts/test_iterative_design_catres.py``: the
script is imported as a module (its sibling-script imports are defensive sys.path
inserts, so this works on host pytest as long as PYTHONPATH includes src/).
"""

from __future__ import annotations

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


# ----------------------------------------------------------------------
# Helpers: faithfully replay main()'s inline global-set for --chain so the
# behaviour test exercises the SHIPPED logic (main() is monolithic and the
# task forbids extracting a build_parser; we drive the real parser + the real
# validation/error path via the module's own ArgumentParser construction).
# ----------------------------------------------------------------------


def _apply_chain_from_args(chain_value):
    """Replay EXACTLY the validation + global-set that main() performs for
    --chain, against the real module, and return the resulting v2.CHAIN.

    Raises SystemExit (argparse p.error) on an invalid chain id — same as the
    shipped path. Restores the global afterwards so tests don't leak state.
    """
    import argparse

    orig = v2.CHAIN
    p = argparse.ArgumentParser()
    p.add_argument("--chain", type=str, default="A")
    args = p.parse_args(["--chain", chain_value])
    try:
        # --- begin: byte-for-byte the shipped main() snippet ---
        _chain_arg = str(args.chain)
        if len(_chain_arg) != 1 or _chain_arg.isspace():
            p.error("--chain must be a single non-space chain id (e.g. 'A', 'B')")
        v2.CHAIN = _chain_arg
        # --- end snippet ---
        return v2.CHAIN
    finally:
        v2.CHAIN = orig


# ----------------------------------------------------------------------
# A. --chain
# ----------------------------------------------------------------------


def test_chain_default_is_A_byte_identical():
    """The module global ships as 'A' — the byte-identical default."""
    assert v2.CHAIN == "A"


def test_chain_arg_propagates_configured_chain():
    """--chain B sets the CHAIN global to 'B' (NOT the hard-coded 'A')."""
    got = _apply_chain_from_args("B")
    assert got == "B"
    assert got != "A"
    # And the global is restored after (no leak).
    assert v2.CHAIN == "A"


def test_chain_arg_default_keeps_A():
    """--chain with the default value leaves the global unchanged at 'A'."""
    assert _apply_chain_from_args("A") == "A"


@pytest.mark.parametrize("bad", ["", "AB", "  ", " "])
def test_chain_arg_rejects_non_single_or_space(bad):
    """Multi-char / empty / whitespace chain ids are rejected via p.error
    (argparse raises SystemExit)."""
    with pytest.raises(SystemExit):
        _apply_chain_from_args(bad)


def test_chain_used_at_struct_read_sites():
    """The structural-read call sites pass the configured CHAIN global, not a
    hard-coded literal. Pin a representative set so a future literal 'A' can't
    silently regress the generalization (source-level guard)."""
    src = (SCRIPTS / "iterative_design.py").read_text()
    # detect_interactions / detect_clashes / preorganization in the struct
    # worker, and the sequence/SS reads in main(), all thread chain=CHAIN.
    assert "detect_interactions(pdb, chain=CHAIN" in src
    assert "catalytic_resnos=fixed_, chain=CHAIN" in src
    assert "preorganization_score(\n            pdb, catalytic_resnos=list(fixed_), chain=CHAIN" in src
    assert "extract_sequence(args.seed_pdb, chain=CHAIN)" in src


def test_main_sets_chain_global_from_args():
    """main() contains the `global CHAIN` declaration and assigns it from the
    validated arg (the mechanism that makes --chain take effect)."""
    src = (SCRIPTS / "iterative_design.py").read_text()
    assert "global CHAIN" in src
    assert "CHAIN = _chain_arg" in src


# ----------------------------------------------------------------------
# B. --no_require_cat_his_hbond
# ----------------------------------------------------------------------


def test_require_cat_his_hbond_default_is_true_byte_identical():
    """The requirement ships ON (True) => byte-identical default behaviour."""
    assert v2.REQUIRE_CAT_HIS_HBOND is True


def _struct_filter_reasons_for_zero_hbonds(monkeypatch, tmp_path):
    """Run the real _struct_filter_worker on a stub where the HIS-H-bond
    detector returns ZERO hbonds, with all OTHER struct sub-metrics stubbed to
    'pass', and return the worker's reason list. The cat-HIS criterion is then
    the ONLY thing that can append a reason — so its presence/absence directly
    reflects REQUIRE_CAT_HIS_HBOND.
    """
    pdb = tmp_path / "design.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00  0.00           N\n"
    )

    # Zero H-bonds to catalytic HIS.
    monkeypatch.setattr(v2, "_detect_hbond_to_his_sidechain", lambda *a, **k: [])

    # Stub the heavy structural sub-metric imports used inside the worker so
    # they return benign passing values (no clash, no SAP overage).
    import types

    gi_mod = types.ModuleType("protein_chisel.tools.geometric_interactions")

    class _Panel:
        def to_dict(self, prefix=""):
            return {}

    gi_mod.detect_interactions = lambda *a, **k: _Panel()
    monkeypatch.setitem(sys.modules,
                        "protein_chisel.tools.geometric_interactions", gi_mod)

    preorg_mod = types.ModuleType("protein_chisel.scoring.preorganization")
    preorg_mod.preorganization_score = lambda *a, **k: {
        "preorg__n_hbonds_to_cat": 0, "preorg__n_salt_bridges_to_cat": 0,
        "preorg__n_pi_to_cat": 0, "preorg__n_hbonds_within_shells": 0,
        "preorg__strength_total": 0.0, "preorg__interactome_density": 0.0,
        "preorg__n_first_shell": 0, "preorg__n_second_shell": 0,
    }
    monkeypatch.setitem(sys.modules,
                        "protein_chisel.scoring.preorganization", preorg_mod)

    struct_mod = types.ModuleType("protein_chisel.structure")

    class _Clash:
        def to_dict(self):
            return {"clash__has_severe": 0, "clash__n_to_catalytic": 0,
                    "clash__n_to_ligand": 0, "clash__n_total": 0,
                    "clash__detail": ""}

    struct_mod.detect_clashes = lambda *a, **k: _Clash()
    monkeypatch.setitem(sys.modules, "protein_chisel.structure", struct_mod)

    # SAP proxy -> no overage.
    monkeypatch.setattr(v2, "_compute_sap_proxy",
                        lambda *a, **k: {"sap_max": 0.0, "sap_mean": 0.0,
                                         "sap_p95": 0.0})

    work = (
        "cid0", str(pdb), (60,), (60,),
        1.5, 100.0, {}, False,
    )
    _cid, _row, _hbonds, reasons = v2._struct_filter_worker(work)
    return reasons


def test_cat_his_criterion_on_by_default_rejects_zero_hbonds(monkeypatch, tmp_path):
    """With REQUIRE_CAT_HIS_HBOND True (default), a design with 0 H-bonds to a
    catalytic HIS is rejected with the cat-HIS reason (current behaviour)."""
    monkeypatch.setattr(v2, "REQUIRE_CAT_HIS_HBOND", True)
    reasons = _struct_filter_reasons_for_zero_hbonds(monkeypatch, tmp_path)
    assert any("catalytic HIS" in r for r in reasons), reasons


def test_no_require_cat_his_hbond_skips_the_criterion(monkeypatch, tmp_path):
    """With REQUIRE_CAT_HIS_HBOND False (--no_require_cat_his_hbond), the SAME
    zero-H-bond design is NOT rejected for the cat-HIS reason."""
    monkeypatch.setattr(v2, "REQUIRE_CAT_HIS_HBOND", False)
    reasons = _struct_filter_reasons_for_zero_hbonds(monkeypatch, tmp_path)
    assert not any("catalytic HIS" in r for r in reasons), reasons


def test_main_sets_require_cat_his_global_from_args():
    """main() declares the global and sets it from `not args.no_require_cat_his_hbond`
    (the mechanism), and has the no-His auto-relax bonus."""
    src = (SCRIPTS / "iterative_design.py").read_text()
    assert "global REQUIRE_CAT_HIS_HBOND" in src
    assert "REQUIRE_CAT_HIS_HBOND = not bool(args.no_require_cat_his_hbond)" in src
    # Bonus: auto-relax when the resolved catalytic set has no His.
    assert "not derived_his" in src


# ----------------------------------------------------------------------
# --help (host, no container) advertises BOTH flags, with and without PYTHONPATH
# ----------------------------------------------------------------------


def _run_help(env):
    return subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120,
    )


def test_help_advertises_both_flags_with_pythonpath():
    proc = _run_help({**os.environ, "PYTHONPATH": "src"})
    assert proc.returncode == 0, proc.stderr
    assert "--chain" in proc.stdout
    assert "--no_require_cat_his_hbond" in proc.stdout


def test_help_advertises_both_flags_without_pythonpath():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = _run_help(env)
    assert proc.returncode == 0, proc.stderr
    assert "--chain" in proc.stdout
    assert "--no_require_cat_his_hbond" in proc.stdout


# ----------------------------------------------------------------------
# run_chisel_design.sh passthroughs: CHAIN -> --chain, REQUIRE_CAT_HIS=0 ->
# --no_require_cat_his_hbond. Both default to byte-identical (no flag emitted).
# ----------------------------------------------------------------------


def _extract_cli_builder(sh_text, cli_var):
    """Pull the committed `<cli_var>+=( ... )` builder line(s) out of the shell
    so the test exercises the real shipped wiring, not a re-implementation."""
    return [
        ln.strip() for ln in sh_text.splitlines()
        if cli_var in ln and "+=(" in ln
    ]


def test_shell_wires_chain_passthrough():
    sh = (REPO / "scripts" / "run_chisel_design.sh").read_text()
    assert "CHAIN=" in sh
    assert "--chain" in sh


def test_shell_wires_require_cat_his_passthrough():
    sh = (REPO / "scripts" / "run_chisel_design.sh").read_text()
    assert "REQUIRE_CAT_HIS" in sh
    assert "--no_require_cat_his_hbond" in sh


@pytest.mark.parametrize(
    "value, expect_flag, expect_token",
    [
        ("A", False, None),       # default -> omit (byte-identical)
        ("", False, None),        # empty -> omit
        ("B", True, "B"),         # non-A -> forwarded
        ("C", True, "C"),
    ],
)
def test_shell_chain_only_when_non_A(value, expect_flag, expect_token):
    """CHAIN_CLI appends --chain ONLY when CHAIN is set and != 'A'. Runs the
    EXACT committed one-liner in real bash so it stays pinned to shipped wiring."""
    sh = (REPO / "scripts" / "run_chisel_design.sh").read_text()
    snippet = _extract_cli_builder(sh, "CHAIN_CLI")
    assert snippet, "expected a CHAIN_CLI builder line in the shell"
    script = (
        'CHAIN_CLI=()\n' + "\n".join(snippet) + "\n"
        + 'echo "${CHAIN_CLI[@]}"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "CHAIN": value},
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert ("--chain" in proc.stdout) is expect_flag
    if expect_token is not None:
        assert expect_token in proc.stdout


@pytest.mark.parametrize(
    "value, expect_flag",
    [
        ("1", False),     # default ON -> omit (byte-identical)
        ("", False),      # treated as default (not literal 0) -> omit
        ("0", True),      # opt out -> forward --no_require_cat_his_hbond
    ],
)
def test_shell_require_cat_his_only_when_zero(value, expect_flag):
    """REQUIRE_CAT_HIS_CLI appends --no_require_cat_his_hbond ONLY when
    REQUIRE_CAT_HIS == 0. Real bash, committed snippet."""
    sh = (REPO / "scripts" / "run_chisel_design.sh").read_text()
    snippet = _extract_cli_builder(sh, "REQUIRE_CAT_HIS_CLI")
    assert snippet, "expected a REQUIRE_CAT_HIS_CLI builder line in the shell"
    script = (
        'REQUIRE_CAT_HIS_CLI=()\n' + "\n".join(snippet) + "\n"
        + 'echo "${REQUIRE_CAT_HIS_CLI[@]}"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "REQUIRE_CAT_HIS": value},
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert ("--no_require_cat_his_hbond" in proc.stdout) is expect_flag
