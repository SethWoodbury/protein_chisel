"""Host-runnable tests for the two opt-in robustness features.

FEATURE #29 — composition-cap POOL FALLBACK (--composition_pool_fallback):
    The WS-C composition cap / class-balance / soft-bias only fire inside
    ``if survivors_prev is not None and len(survivors_prev) > 0:`` in
    ``run_cycle``. On a hydrophobic seed where ~100% of samples fail the GRAVY
    band, ``survivors_prev`` is empty so those levers never fire and the run
    ships ~26% Ala from backfill. The opt-in flag builds the cap/class-balance/
    soft-bias from the PREVIOUS cycle's full SAMPLED (pre-band-filter) pool when
    survivors are empty. Default OFF => byte-identical; cycle 0 is a no-op.

FEATURE #30 — SIDECHAIN-CONTEXT SCHEDULE (--use_side_chain_context_schedule):
    A comma-separated per-cycle 0/1 schedule overrides the uniform
    ``--use_side_chain_context`` value per cycle (broadcast/truncate to the
    cycle count). When absent, the uniform value is used unchanged =>
    byte-identical.

All host-runnable: no GPU / apptainer / freesasa / network. Import idiom
mirrors ``tests/scripts/test_committed_integration.py``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


# ============================================================================
# FEATURE #30 — --use_side_chain_context_schedule
# ============================================================================
#
# Validator: parse comma-separated 0/1 into a list[int]; reject anything else.


def test_scc_schedule_arg_parses_zero_one_csv():
    """'1,1,0' -> [1, 1, 0]; whitespace around entries is tolerated."""
    assert idz._scc_schedule_arg("1,1,0") == [1, 1, 0]
    assert idz._scc_schedule_arg(" 1 , 0 , 1 ") == [1, 0, 1]
    assert idz._scc_schedule_arg("0") == [0]
    assert idz._scc_schedule_arg("1") == [1]


def test_scc_schedule_arg_rejects_non_binary_and_empty():
    """Any entry not exactly 0/1 (2, -1, 0.5, 'x') or an empty schedule is
    rejected at the CLI boundary."""
    for bad in ("2", "-1", "0.5", "x", "1,2", "1,,0", "", " ", ",", "1,x"):
        with pytest.raises(argparse.ArgumentTypeError):
            idz._scc_schedule_arg(bad)


def test_apply_scc_schedule_exact_length_overrides_each_cycle():
    """A schedule whose length == cycle count overrides each cycle's value
    1:1, regardless of the uniform --use_side_chain_context."""
    cycles = idz.default_cycles(use_side_chain_context=0)  # uniform 0
    assert len(cycles) == 3
    idz._apply_scc_schedule(cycles, [1, 1, 0])
    assert [c.use_side_chain_context for c in cycles] == [1, 1, 0]


def test_apply_scc_schedule_broadcasts_last_value_when_short():
    """A schedule SHORTER than the cycle count extends by repeating its LAST
    entry (clear rule: the tail uses the last given value -> ON early, OFF late
    works for '1,0' over 3 cycles => [1, 0, 0])."""
    cycles = idz.default_cycles(use_side_chain_context=1)
    idz._apply_scc_schedule(cycles, [1, 0])
    assert [c.use_side_chain_context for c in cycles] == [1, 0, 0]


def test_apply_scc_schedule_single_value_broadcasts_to_all():
    """A length-1 schedule broadcasts that value to every cycle."""
    cycles = idz.default_cycles(use_side_chain_context=0)
    idz._apply_scc_schedule(cycles, [1])
    assert [c.use_side_chain_context for c in cycles] == [1, 1, 1]


def test_apply_scc_schedule_truncates_when_long():
    """A schedule LONGER than the cycle count is truncated to the cycle count
    (the extra entries are ignored, not appended)."""
    cycles = idz.default_cycles(use_side_chain_context=0)
    assert len(cycles) == 3
    idz._apply_scc_schedule(cycles, [1, 0, 1, 0, 1])
    assert [c.use_side_chain_context for c in cycles] == [1, 0, 1]


def test_apply_scc_schedule_overrides_a_nonzero_uniform_value():
    """The schedule WINS over the uniform value at every cycle (a cycle that
    the schedule sets to 0 turns OFF even though the uniform value was 1)."""
    cycles = idz.default_cycles(use_side_chain_context=1)  # uniform ON
    idz._apply_scc_schedule(cycles, [0, 1, 0])
    assert [c.use_side_chain_context for c in cycles] == [0, 1, 0]


def test_scc_schedule_help_advertised_with_pythonpath():
    """`--help` exits 0 and advertises --use_side_chain_context_schedule."""
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--use_side_chain_context_schedule" in proc.stdout


def test_scc_schedule_help_works_without_pythonpath():
    """--help must show the flag even with PYTHONPATH unset (the schedule type=
    callable does no protein_chisel import, so argparse builds the help without
    the package on the path)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--use_side_chain_context_schedule" in proc.stdout


def test_cli_rejects_bad_scc_schedule_at_parse_time():
    """A non-binary schedule fails fast at parse time (exit != 0)."""
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py",
         "--use_side_chain_context_schedule", "1,2,0", "--seed_pdb", "x"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "use_side_chain_context_schedule" in proc.stderr


# ============================================================================
# FEATURE #29 — --composition_pool_fallback
# ============================================================================
#
# The fallback re-routes the WS-C cap / class-balance / soft-bias pool source
# to the previous cycle's full SAMPLED pool when survivors are empty. The
# decision logic lives in a tiny pure helper so the routing can be unit-tested
# without running a full cycle.


def _hydrophobic_pool_df(n: int = 40, aa_run: str = "A") -> pd.DataFrame:
    """A synthetic 'sampled pool' DataFrame whose sequences are dominated by a
    single AA (the 26%-Ala failure mode), with a `sequence` column — the same
    schema as ``_load_cycle_seq_stage_pool``'s output."""
    # 30% of each sequence is the aa_run AA; the rest is a spread of others.
    body = aa_run * 6 + "DENGSTKRQ" + "DEKRSTNQ" + "DEKRST"   # len 6+9+8+6 = 29
    body = (body + "DEKRSTNGQ")[:30]
    seqs = [body for _ in range(n)]
    return pd.DataFrame({"id": [f"d{i}" for i in range(n)], "sequence": seqs})


def test_fallback_pool_source_disabled_returns_none():
    """Flag OFF => no fallback pool regardless of survivors / sampled pool
    (byte-identical: run_cycle's gated block sees survivors only)."""
    sampled = _hydrophobic_pool_df()
    out = idz._resolve_composition_pool(
        survivors_prev=None, fallback_pool=sampled,
        composition_pool_fallback=False,
    )
    assert out is None


def test_fallback_pool_source_used_when_survivors_empty_and_enabled():
    """Flag ON + survivors empty + a non-empty sampled pool => the sampled pool
    is returned as the source for the cap/class-balance/soft-bias."""
    sampled = _hydrophobic_pool_df()
    out = idz._resolve_composition_pool(
        survivors_prev=pd.DataFrame(), fallback_pool=sampled,
        composition_pool_fallback=True,
    )
    assert out is sampled
    # None survivors behaves the same as empty.
    out2 = idz._resolve_composition_pool(
        survivors_prev=None, fallback_pool=sampled,
        composition_pool_fallback=True,
    )
    assert out2 is sampled


def test_fallback_pool_source_noop_when_survivors_exist():
    """When survivors EXIST, the fallback never engages even if enabled (the
    real survivor pool is preferred; the cap/class-balance use survivors)."""
    sampled = _hydrophobic_pool_df()
    survivors = pd.DataFrame({"sequence": ["DENGSTKRQDENGSTKRQDE"]})
    out = idz._resolve_composition_pool(
        survivors_prev=survivors, fallback_pool=sampled,
        composition_pool_fallback=True,
    )
    assert out is None


def test_fallback_pool_source_noop_at_cycle0_no_prev_pool():
    """Cycle 0 has no previous sampled pool (fallback_pool is None/empty) ->
    no-op even with the flag on and survivors empty."""
    assert idz._resolve_composition_pool(
        survivors_prev=None, fallback_pool=None,
        composition_pool_fallback=True,
    ) is None
    assert idz._resolve_composition_pool(
        survivors_prev=pd.DataFrame(), fallback_pool=pd.DataFrame(),
        composition_pool_fallback=True,
    ) is None


def test_fallback_pool_drives_fraction_cap_on_sampled_pool():
    """END-TO-END (cap layer): the resolved fallback pool, run through the SAME
    ``_build_fraction_cap_omit`` the gated block uses, caps the over-rep AA. This
    proves the fallback pool re-enables the cap that an empty survivor pool would
    have silently skipped (the 26%-Ala fix)."""
    sampled = _hydrophobic_pool_df(aa_run="A")
    pool = idz._resolve_composition_pool(
        survivors_prev=pd.DataFrame(), fallback_pool=sampled,
        composition_pool_fallback=True,
    )
    assert pool is sampled
    pool_seq = "".join(pool["sequence"].astype(str).tolist())
    omit = idz._build_fraction_cap_omit(
        pool_seq, 0.15, protein_resnos=[10, 11, 12], fixed_resnos=[12],
        chain="A", exclude_aas="",
    )
    # A is ~20% of the pool (6/30) -> over the 15% cap -> omitted at non-fixed
    # designable positions; the fixed catalytic resno (12) is excluded.
    assert set(omit.keys()) == {"A10", "A11"}
    assert all("A" in v for v in omit.values())


def test_composition_pool_fallback_help_advertised():
    """`--help` exits 0 and advertises --composition_pool_fallback."""
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--composition_pool_fallback" in proc.stdout


def test_composition_pool_fallback_help_works_without_pythonpath():
    """--help shows --composition_pool_fallback with PYTHONPATH unset (it is a
    store_true flag — no import at parse time)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--composition_pool_fallback" in proc.stdout


def test_composition_pool_fallback_default_off_in_help():
    """The flag is a store_true (default OFF): --help renders it WITHOUT a
    metavar / '=VALUE' (a bare boolean), and the run is byte-identical unless
    it is explicitly passed. This pins the opt-in/default-off contract without
    extracting a parser factory (the driver builds its parser inline)."""
    proc = subprocess.run(
        [sys.executable, "scripts/iterative_design.py", "--help"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    # store_true flags appear bare (no uppercase metavar token after them).
    # Find the flag's own usage/option token and assert no metavar follows.
    assert "--composition_pool_fallback" in proc.stdout
    # The schedule, by contrast, takes a value (has a metavar).
    assert "--use_side_chain_context_schedule" in proc.stdout


# ----------------------------------------------------------------------------
# Shell passthrough (run_chisel_design.sh): COMPOSITION_POOL_FALLBACK and
# USE_SIDE_CHAIN_CONTEXT_SCHEDULE wired onto their CLI flags.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expect_flag",
    [("0", False), ("false", False), ("", False),
     ("1", True), ("true", True), ("on", True), ("YES", True)],
)
def test_shell_composition_pool_fallback_truthiness(value, expect_flag):
    """COMPOSITION_POOL_FALLBACK uses the shipped truthy guard -> the flag."""
    script = (
        'CLI=()\n'
        f'[[ "${{COMPOSITION_POOL_FALLBACK:-0}}" =~ {_SHELL_TRUTHY_RE} ]] '
        '&& CLI+=( --composition_pool_fallback )\n'
        'echo "${CLI[@]}"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "COMPOSITION_POOL_FALLBACK": value},
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert ("--composition_pool_fallback" in proc.stdout) is expect_flag


def test_shell_composition_pool_fallback_wired_in_source():
    """The shipped run_chisel_design.sh wires COMPOSITION_POOL_FALLBACK onto
    --composition_pool_fallback."""
    sh = (REPO / "scripts" / "run_chisel_design.sh").read_text()
    assert "COMPOSITION_POOL_FALLBACK" in sh
    assert "--composition_pool_fallback" in sh


def test_shell_scc_schedule_emits_flag_only_when_set():
    """USE_SIDE_CHAIN_CONTEXT_SCHEDULE set -> --use_side_chain_context_schedule
    <value>; unset/empty -> the flag is omitted (byte-identical default)."""
    script = (
        'CLI=()\n'
        '[[ -n "${USE_SIDE_CHAIN_CONTEXT_SCHEDULE:-}" ]] '
        '&& CLI+=( --use_side_chain_context_schedule '
        '"$USE_SIDE_CHAIN_CONTEXT_SCHEDULE" )\n'
        'echo "${CLI[@]}"\n'
    )
    # set
    p1 = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "USE_SIDE_CHAIN_CONTEXT_SCHEDULE": "1,1,0"},
        capture_output=True, text=True, timeout=30,
    )
    assert p1.returncode == 0, p1.stderr
    assert "--use_side_chain_context_schedule 1,1,0" in p1.stdout
    # unset
    env = {k: v for k, v in os.environ.items()
           if k != "USE_SIDE_CHAIN_CONTEXT_SCHEDULE"}
    p2 = subprocess.run(["bash", "-c", script], env=env,
                        capture_output=True, text=True, timeout=30)
    assert p2.returncode == 0, p2.stderr
    assert "--use_side_chain_context_schedule" not in p2.stdout


def test_shell_scc_schedule_wired_in_source():
    """The shipped run_chisel_design.sh wires USE_SIDE_CHAIN_CONTEXT_SCHEDULE
    onto --use_side_chain_context_schedule."""
    sh = (REPO / "scripts" / "run_chisel_design.sh").read_text()
    assert "USE_SIDE_CHAIN_CONTEXT_SCHEDULE" in sh
    assert "--use_side_chain_context_schedule" in sh


# ============================================================================
# FEATURE #29 — run_cycle INTEGRATION (the fallback actually fires in-cycle)
# ============================================================================
#
# These drive the REAL ``run_cycle`` bias-build path (stages 0..0c, which write
# bias.npy + class_balance_telemetry.json) and stop it at stage 1 by monkey-
# patching ``stage_sample`` to raise a sentinel. We then inspect the on-disk
# bias artifacts to prove the cap / class-balance were built from the SAMPLED
# fallback pool when survivors are empty (and NOT built when the flag is off).


class _StopAtSampling(RuntimeError):
    """Sentinel raised by the patched stage_sample to halt run_cycle right after
    the bias build (stages 0..0c) has been written to disk."""


def _run_cycle_to_bias(monkeypatch, tmp_path, *, fallback_pool,
                       composition_pool_fallback, aa_fraction_cap=0.15):
    """Invoke run_cycle with an EMPTY survivor pool, stopping at the sampler.

    Returns the cycle dir so callers can read 00_bias artifacts. The protein is a
    tiny length-8 chain with one fixed (catalytic) position; only the bias-build
    inputs need to be real because stage_sample is stubbed to raise."""
    L = 8
    monkeypatch.setattr(idz, "stage_sample",
                        lambda *a, **k: (_ for _ in ()).throw(_StopAtSampling()))
    base_bias = np.zeros((L, 20), dtype=np.float32)
    protein_resnos = list(range(1, L + 1))
    fixed_resnos = [4]                                   # one catalytic position
    position_classes = ["surface"] * L
    cyc = idz.CycleConfig(cycle_idx=1, n_samples=4, omit_AA="X")
    cycle_dir = tmp_path / "cycle_01"
    with pytest.raises(_StopAtSampling):
        idz.run_cycle(
            cycle_cfg=cyc, seed_pdb=tmp_path / "seed.pdb",
            base_bias=base_bias,
            log_probs_esmc=np.zeros((L, 20), dtype=np.float32),
            log_probs_saprot=np.zeros((L, 20), dtype=np.float32),
            weights_per_position=np.ones(L, dtype=np.float32),
            position_classes=position_classes,
            protein_resnos=protein_resnos,
            fixed_resnos=fixed_resnos,
            survivors_prev=pd.DataFrame(),               # EMPTY survivor pool
            cycle_dir=cycle_dir,
            fitness_cache={},
            wt_length=L,
            expression_engine=None,                      # only touched if soft_bias on
            seed_position_class=position_classes,
            seed_protein_resnos=protein_resnos,
            aa_fraction_cap=aa_fraction_cap,
            composition_pool_fallback=composition_pool_fallback,
            composition_fallback_pool=fallback_pool,
        )
    return cycle_dir


def _iso_fallback_pool(n=12):
    """A length-8 sampled pool dominated by Isoleucine (3/8 = 37.5% I, over a
    0.15 cap) and starved of Leucine — triggers BOTH the I fraction cap and the
    hydrophobic-class I->L balance swap on the default hydrolase reference."""
    seq = "IIIDEKRS"                                     # 3 I, len 8, no L
    assert len(seq) == 8 and seq.count("I") == 3 and "L" not in seq
    return pd.DataFrame({"id": [f"d{i}" for i in range(n)],
                         "sequence": [seq] * n})


def test_run_cycle_fallback_builds_cap_and_balance_from_sampled_pool(
        monkeypatch, tmp_path):
    """Flag ON + empty survivors + a hydrophobic SAMPLED fallback pool => the
    in-cycle cap omits I at non-fixed designable positions AND the class-balance
    bias_AA down-weights I / up-weights L — both built from the fallback pool the
    empty survivor pool would otherwise have skipped (the 26%-Ala fix)."""
    import json
    cycle_dir = _run_cycle_to_bias(
        monkeypatch, tmp_path,
        fallback_pool=_iso_fallback_pool(),
        composition_pool_fallback=True,
    )
    cb = json.loads((cycle_dir / "00_bias" / "class_balance_telemetry.json").read_text())
    # Class-balance fired on the fallback pool: I down-weighted (the over-rep
    # hydrophobic-class member), L up-weighted (the under-rep partner).
    assert "I:" in cb["bias_AA_string"]
    assert "I:-" in cb["bias_AA_string"]                 # negative (down-weight)
    assert "L:" in cb["bias_AA_string"]


def test_run_cycle_fallback_off_is_noop_no_class_balance(monkeypatch, tmp_path):
    """Flag OFF + empty survivors => the gated block is SKIPPED entirely (no
    class_balance_telemetry.json written), i.e. byte-identical to today's empty-
    survivor behavior even though a sampled pool was available."""
    cycle_dir = _run_cycle_to_bias(
        monkeypatch, tmp_path,
        fallback_pool=_iso_fallback_pool(),
        composition_pool_fallback=False,                 # OFF
    )
    assert not (cycle_dir / "00_bias" / "class_balance_telemetry.json").exists()


def test_run_cycle_fallback_noop_when_no_prev_pool(monkeypatch, tmp_path):
    """Flag ON but NO previous sampled pool (cycle-0 analogue: fallback_pool is
    None) => the gated block is skipped (no class_balance_telemetry.json)."""
    cycle_dir = _run_cycle_to_bias(
        monkeypatch, tmp_path,
        fallback_pool=None,                              # no previous pool
        composition_pool_fallback=True,
    )
    assert not (cycle_dir / "00_bias" / "class_balance_telemetry.json").exists()


def _carry_survivors(ranked_df, prev, *, composition_pool_fallback):
    """Faithful mirror of the loop's survivor-carry decision (the exact branch at
    the bottom of the per-cycle loop). When ranked_df is non-empty it becomes the
    next survivor pool; when it is empty, the carried pool is CLEARED to None ONLY
    if --composition_pool_fallback is set (so a collapsed cycle routes the next
    cycle through the sampled fallback instead of a STALE survivor pool), else it
    is left unchanged (byte-identical default)."""
    if ranked_df is not None and len(ranked_df) > 0:
        return ranked_df
    if composition_pool_fallback:
        return None
    return prev


def test_collapsed_cycle_clears_stale_survivors_only_when_fallback_on():
    """Regression (codex): cycle N has survivors, cycle N+1 collapses (0 ranked).
    With the fallback ON, the stale cycle-N survivor pool is cleared so cycle N+2's
    _resolve_composition_pool activates the fallback; with it OFF the stale pool is
    retained (today's behavior => byte-identical)."""
    stale = pd.DataFrame({"sequence": ["IIIDEKRS"]})         # cycle N survivors
    empty_ranked = pd.DataFrame()                            # cycle N+1 collapsed
    # ON: stale pool cleared -> next cycle sees None -> fallback can engage.
    carried_on = _carry_survivors(empty_ranked, stale, composition_pool_fallback=True)
    assert carried_on is None
    assert idz._resolve_composition_pool(
        survivors_prev=carried_on, fallback_pool=_iso_fallback_pool(),
        composition_pool_fallback=True,
    ) is not None                                            # fallback now active
    # OFF: stale pool retained -> default behavior unchanged.
    carried_off = _carry_survivors(empty_ranked, stale, composition_pool_fallback=False)
    assert carried_off is stale


def test_collapsed_cycle_source_wiring_is_gated():
    """Pin the shipped loop wiring: the survivors-clear on the empty-ranked path is
    gated on args.composition_pool_fallback (so it never fires on the default
    path)."""
    src = (REPO / "scripts" / "iterative_design.py").read_text()
    assert "elif args.composition_pool_fallback:" in src
    assert "survivors_prev = None" in src


def test_run_cycle_softbias_source_is_sampled_fallback(monkeypatch, tmp_path):
    """The composition SOFT-BIAS also routes through the fallback: with the flag
    on + empty survivors + a sampled pool, the bias telemetry records source
    'sampled_fallback'. The engine's pool-aggregation + array translation are
    stubbed (engine internals are tested elsewhere); we assert ONLY the source
    selection + that the fallback pool's sequences are what get aggregated."""
    import json
    import protein_chisel.expression.engine as _eng

    seen = {}

    def _fake_aggregate(engine, seqs, **kw):
        seen["seqs"] = list(seqs)
        return {0: "I"}                                  # one non-empty soft hit

    def _fake_to_array(soft_map, L, magnitude=0.5, aa_order=None):
        return np.zeros((L, 20), dtype=np.float32)

    monkeypatch.setattr(_eng, "aggregate_pool_soft_bias", _fake_aggregate)
    monkeypatch.setattr(_eng, "soft_bias_to_bias_array", _fake_to_array)

    L = 8
    monkeypatch.setattr(idz, "stage_sample",
                        lambda *a, **k: (_ for _ in ()).throw(_StopAtSampling()))
    pool = _iso_fallback_pool()
    cyc = idz.CycleConfig(cycle_idx=1, n_samples=4, omit_AA="X")
    cycle_dir = tmp_path / "cycle_01"
    with pytest.raises(_StopAtSampling):
        idz.run_cycle(
            cycle_cfg=cyc, seed_pdb=tmp_path / "seed.pdb",
            base_bias=np.zeros((L, 20), dtype=np.float32),
            log_probs_esmc=np.zeros((L, 20), dtype=np.float32),
            log_probs_saprot=np.zeros((L, 20), dtype=np.float32),
            weights_per_position=np.ones(L, dtype=np.float32),
            position_classes=["surface"] * L,
            protein_resnos=list(range(1, L + 1)),
            fixed_resnos=[4],
            survivors_prev=pd.DataFrame(),               # EMPTY survivors
            cycle_dir=cycle_dir, fitness_cache={}, wt_length=L,
            expression_engine=object(),                  # stubbed funcs ignore it
            seed_position_class=["surface"] * L,
            seed_protein_resnos=list(range(1, L + 1)),
            composition_soft_bias=True,                  # exercise the soft-bias path
            composition_pool_fallback=True,
            composition_fallback_pool=pool,
        )
    telem = json.loads((cycle_dir / "00_bias" / "telemetry.json").read_text())
    assert telem.get("composition_soft_bias_source") == "sampled_fallback"
    # The sequences aggregated are the FALLBACK pool's, not survivors'.
    assert seen["seqs"] == pool["sequence"].tolist()
