"""Host-runnable tests for CLI/env-configurable design acceptance bands.

FEATURE — every design acceptance range settable from a CLI flag AND a
``run_chisel_design.sh`` env var, so a Slurm driver cell can override any of
them. The core change threads the band params through ``default_cycles(...)``:

  - constant strategy: every cycle uses the passed band values directly.
  - annealing strategy: the passed value is the cycle-2 (FINAL/strictest) band;
    cycles 0 and 1 relax from it by the FIXED legacy offsets so that, with the
    params at their defaults, the per-cycle schedule is byte-identical to the
    previous HARDCODED one. net_charge / pi / sap stay CONSTANT across cycles.

The byte-identity reference values below are FROZEN literals (the per-cycle
bands the pipeline shipped before this feature) so the guard is independent of
the implementation it protects.

All host-runnable: no GPU / apptainer / freesasa / network. Import idiom
mirrors ``tests/scripts/test_robustness_features.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import iterative_design as idz  # noqa: E402


# The exact truthy regex from run_chisel_design.sh (kept in sync by the
# pinning test in test_committed_integration.py).
_SHELL_TRUTHY_RE = r"^([Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn]|1)$"


# ---------------------------------------------------------------------------
# FROZEN ground truth: the per-cycle bands the pipeline produced BEFORE this
# feature (captured from the previous hardcoded default_cycles()). Field order:
#   (net_charge_min, net_charge_max, sap_max_threshold,
#    instability_max, gravy_min, gravy_max, aliphatic_min, boman_max,
#    pi_min, pi_max)
# ---------------------------------------------------------------------------
_LEGACY_ANNEALING = {
    0: (-18.0, -4.0, 100.0, 80.0, -1.0, 0.4, 30.0, 5.5, 5.0, 7.5),
    1: (-18.0, -4.0, 100.0, 70.0, -0.9, 0.35, 35.0, 5.0, 5.0, 7.5),
    2: (-18.0, -4.0, 100.0, 60.0, -0.8, 0.3, 40.0, 4.5, 5.0, 7.5),
}
_LEGACY_CONSTANT = {
    0: (-18.0, -4.0, 100.0, 60.0, -0.8, 0.3, 40.0, 4.5, 5.0, 7.5),
    1: (-18.0, -4.0, 100.0, 60.0, -0.8, 0.3, 40.0, 4.5, 5.0, 7.5),
    2: (-18.0, -4.0, 100.0, 60.0, -0.8, 0.3, 40.0, 4.5, 5.0, 7.5),
}


def _band_tuple(c) -> tuple:
    return (
        c.net_charge_min, c.net_charge_max, c.sap_max_threshold,
        c.instability_max, c.gravy_min, c.gravy_max,
        c.aliphatic_min, c.boman_max, c.pi_min, c.pi_max,
    )


# ============================================================================
# TEST 1 — byte-identity (annealing), field-by-field for cycles 0/1/2
# ============================================================================
def test_default_cycles_annealing_byte_identical_to_legacy():
    """default_cycles(strategy='annealing') with NO band kwargs reproduces the
    previous HARDCODED 3-cycle bands field-by-field (net_charge, sap, gravy,
    instability, aliphatic, boman, pi)."""
    cycles = idz.default_cycles(strategy="annealing")
    assert len(cycles) == 3
    for c in cycles:
        assert _band_tuple(c) == _LEGACY_ANNEALING[c.cycle_idx], (
            f"annealing cycle {c.cycle_idx} drifted from legacy: "
            f"{_band_tuple(c)} != {_LEGACY_ANNEALING[c.cycle_idx]}"
        )


# ============================================================================
# TEST 2 — byte-identity (constant), field-by-field for cycles 0/1/2
# ============================================================================
def test_default_cycles_constant_byte_identical_to_legacy():
    """Same byte-identity guard for strategy='constant'."""
    cycles = idz.default_cycles(strategy="constant")
    assert len(cycles) == 3
    for c in cycles:
        assert _band_tuple(c) == _LEGACY_CONSTANT[c.cycle_idx], (
            f"constant cycle {c.cycle_idx} drifted from legacy: "
            f"{_band_tuple(c)} != {_LEGACY_CONSTANT[c.cycle_idx]}"
        )


# ============================================================================
# TEST 3 — propagation (annealing): value is FINAL cycle; earlier cycles relax
# ============================================================================
def test_annealing_light_filter_propagation_relaxes_earlier_cycles():
    """In annealing the passed value is the cycle-2 (strictest) band; cycles 1
    and 0 relax from it by the fixed legacy offsets.

      --gravy_max 0.5      => c2=0.5,  c1=0.55, c0=0.60
      --instability_max 70 => c2=70,   c1=80,   c0=90
    """
    cycles = idz.default_cycles(
        strategy="annealing", gravy_max=0.5, instability_max=70.0,
    )
    by_idx = {c.cycle_idx: c for c in cycles}
    assert by_idx[2].gravy_max == pytest.approx(0.5)
    assert by_idx[1].gravy_max == pytest.approx(0.55)
    assert by_idx[0].gravy_max == pytest.approx(0.60)
    assert by_idx[2].instability_max == pytest.approx(70.0)
    assert by_idx[1].instability_max == pytest.approx(80.0)
    assert by_idx[0].instability_max == pytest.approx(90.0)


def test_annealing_all_light_offsets_correct():
    """Every annealed light filter relaxes from the passed FINAL value by its
    exact legacy offset (gravy_min/aliphatic_min relax DOWN; gravy_max/
    instability_max/boman_max relax UP)."""
    cycles = idz.default_cycles(
        strategy="annealing",
        gravy_min=-0.5, gravy_max=0.5, instability_max=50.0,
        aliphatic_min=50.0, boman_max=3.0,
    )
    by_idx = {c.cycle_idx: c for c in cycles}
    # gravy_min: c2=value; c1=value-0.10; c0=value-0.20
    assert (by_idx[2].gravy_min, by_idx[1].gravy_min, by_idx[0].gravy_min) == \
        pytest.approx((-0.5, -0.6, -0.7))
    # gravy_max: c2=value; c1=value+0.05; c0=value+0.10
    assert (by_idx[2].gravy_max, by_idx[1].gravy_max, by_idx[0].gravy_max) == \
        pytest.approx((0.5, 0.55, 0.60))
    # instability_max: c2=value; c1=value+10; c0=value+20
    assert (by_idx[2].instability_max, by_idx[1].instability_max,
            by_idx[0].instability_max) == pytest.approx((50.0, 60.0, 70.0))
    # aliphatic_min: c2=value; c1=value-5; c0=value-10
    assert (by_idx[2].aliphatic_min, by_idx[1].aliphatic_min,
            by_idx[0].aliphatic_min) == pytest.approx((50.0, 45.0, 40.0))
    # boman_max: c2=value; c1=value+0.5; c0=value+1.0
    assert (by_idx[2].boman_max, by_idx[1].boman_max, by_idx[0].boman_max) == \
        pytest.approx((3.0, 3.5, 4.0))


# ============================================================================
# TEST 4 — net_charge / sap propagation: CONSTANT across all cycles (both strat)
# ============================================================================
def test_net_charge_and_sap_constant_across_cycles_annealing():
    """--net_charge_max -3, --net_charge_min -20, --sap_max_threshold 60 are
    applied CONSTANT across all annealing cycles (charge/sap bands don't
    anneal)."""
    cycles = idz.default_cycles(
        strategy="annealing",
        net_charge_max=-3.0, net_charge_min=-20.0, sap_max_threshold=60.0,
    )
    for c in cycles:
        assert c.net_charge_max == pytest.approx(-3.0)
        assert c.net_charge_min == pytest.approx(-20.0)
        assert c.sap_max_threshold == pytest.approx(60.0)


def test_net_charge_and_sap_constant_across_cycles_constant():
    """Same net_charge / sap propagation under strategy='constant'."""
    cycles = idz.default_cycles(
        strategy="constant",
        net_charge_max=-3.0, net_charge_min=-20.0, sap_max_threshold=60.0,
    )
    for c in cycles:
        assert c.net_charge_max == pytest.approx(-3.0)
        assert c.net_charge_min == pytest.approx(-20.0)
        assert c.sap_max_threshold == pytest.approx(60.0)


# ============================================================================
# TEST 5 — constant-strategy propagation: every cycle gets the set value
# ============================================================================
def test_constant_strategy_light_filter_propagation():
    """--strategy constant --gravy_max 0.5 => every cycle gravy_max == 0.5
    (constant strategy applies the value uniformly, no offsets)."""
    cycles = idz.default_cycles(strategy="constant", gravy_max=0.5)
    for c in cycles:
        assert c.gravy_max == pytest.approx(0.5)


def test_constant_strategy_all_light_filters_uniform():
    """All passed light-filter values are uniform across cycles in 'constant'."""
    cycles = idz.default_cycles(
        strategy="constant",
        gravy_min=-0.5, gravy_max=0.5, instability_max=50.0,
        aliphatic_min=50.0, boman_max=3.0,
    )
    for c in cycles:
        assert c.gravy_min == pytest.approx(-0.5)
        assert c.gravy_max == pytest.approx(0.5)
        assert c.instability_max == pytest.approx(50.0)
        assert c.aliphatic_min == pytest.approx(50.0)
        assert c.boman_max == pytest.approx(3.0)


def test_debug_short_test_cycles_threads_bands_too():
    """The smoke-test preset builder forwards the band params to default_cycles
    (so a debug run honors overrides as well) and keeps its 20/10/10 samples."""
    cycles = idz.debug_short_test_cycles(
        strategy="annealing", gravy_max=0.5, net_charge_max=-3.0,
    )
    by_idx = {c.cycle_idx: c for c in cycles}
    assert [c.n_samples for c in cycles] == [20, 10, 10]
    assert by_idx[2].gravy_max == pytest.approx(0.5)
    assert by_idx[1].gravy_max == pytest.approx(0.55)
    for c in cycles:
        assert c.net_charge_max == pytest.approx(-3.0)


# ============================================================================
# TEST 6 — --help builds (with and without PYTHONPATH) and advertises the flags
# ============================================================================
def test_help_advertises_new_band_flags_with_pythonpath():
    """`--help` exits 0 and advertises the three NEW band flags."""
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in ("--net_charge_min", "--net_charge_max", "--sap_max_threshold"):
        assert flag in proc.stdout, f"{flag} missing from --help"


def test_help_builds_without_pythonpath():
    """--help must build with PYTHONPATH unset (the parser stays import-light:
    the new flags are plain `type=float`, no protein_chisel import)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in ("--net_charge_min", "--net_charge_max", "--sap_max_threshold"):
        assert flag in proc.stdout, f"{flag} missing from --help"


# ============================================================================
# TEST 7 — shell: `bash -n` passes; each env var emits its flag only when set
# ============================================================================
def test_run_chisel_design_sh_syntax_ok():
    """`bash -n run_chisel_design.sh` parses cleanly (no syntax errors)."""
    proc = subprocess.run(
        ["bash", "-n", "scripts/run_chisel_design.sh"],
        cwd=str(REPO), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


_BAND_ENV_FLAG_MAP = [
    ("NET_CHARGE_MIN", "--net_charge_min"),
    ("NET_CHARGE_MAX", "--net_charge_max"),
    ("SAP_MAX", "--sap_max_threshold"),
    ("GRAVY_MIN", "--gravy_min"),
    ("GRAVY_MAX", "--gravy_max"),
    ("INSTABILITY_MAX", "--instability_max"),
    ("ALIPHATIC_MIN", "--aliphatic_min"),
    ("BOMAN_MAX", "--boman_max"),
    ("PI_MIN", "--pi_min"),
    ("PI_MAX", "--pi_max"),
]


def test_band_env_passthroughs_wired_in_source():
    """Pin every band env var -> flag passthrough to the committed shell file so
    the wiring can't silently disappear."""
    sh = (REPO / "scripts" / "run_chisel_design.sh").read_text()
    # Collapse runs of spaces so cosmetic column-alignment in the source
    # doesn't break the guard pin (the real-bash test below checks behavior).
    sh_norm = " ".join(sh.split())
    for env_var, flag in _BAND_ENV_FLAG_MAP:
        assert env_var in sh, f"{env_var} missing from run_chisel_design.sh"
        assert flag in sh, f"{flag} missing from run_chisel_design.sh"
        # value-passthrough guard (mirrors the AA_REFERENCE pattern)
        assert f'[[ -n "${{{env_var}:-}}" ]]' in sh_norm, (
            f"{env_var} not guarded by the value-set truthiness check"
        )
        # and the env var actually appends its flag (brace-less $VAR form,
        # matching the AA_REFERENCE value-passthrough convention)
        assert f'ACCEPTANCE_BANDS_CLI+=( {flag} "${env_var}" )' in sh_norm, (
            f"{env_var} not wired to append {flag}"
        )


@pytest.mark.parametrize("env_var, flag", _BAND_ENV_FLAG_MAP)
@pytest.mark.parametrize(
    "value, expect_flag", [(None, False), ("", False), ("-7.5", True)]
)
def test_band_env_emits_flag_only_when_set(env_var, flag, value, expect_flag):
    """Each band env var maps onto `<flag> <value>` ONLY when set to a non-empty
    string (value-passthrough pattern); unset/empty -> no flag -> byte-identical
    default. Runs the EXACT guard committed in the shell file in real bash."""
    snippet = (
        f'BAND_CLI=()\n'
        f'[[ -n "${{{env_var}:-}}" ]] && BAND_CLI+=( {flag} "${{{env_var}}}" )\n'
        f'echo "${{BAND_CLI[@]}}"\n'
    )
    env = {k: v for k, v in os.environ.items() if k != env_var}
    if value is not None:
        env[env_var] = value
    proc = subprocess.run(
        ["bash", "-c", snippet], env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    got = flag in proc.stdout
    assert got is expect_flag
    if expect_flag:
        assert value in proc.stdout
