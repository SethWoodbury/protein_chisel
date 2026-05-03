"""Unit tests for scoring/{metrics,cache,tier}.py.

CPU-only; all metrics are dummies that produce deterministic values
from sequence/structure hashes. The whole foundation is exercised
without invoking any apptainer image.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from protein_chisel.scoring.cache import (
    InMemoryCache, JsonlCache, call_metric_cached, make_cache_key,
)
from protein_chisel.scoring.metrics import (
    Candidate, MetricRegistry, MetricResult, MetricSpec, call_metric,
    from_tool_result, hash_params, hash_sequence,
)
from protein_chisel.scoring.tier import (
    TierConstraint, TierPlan, evaluate_tiered, write_tier_log,
)


# ----------------------------------------------------------------------
# dummy metrics for testing
# ----------------------------------------------------------------------


def _len_metric(c: Candidate, params: dict) -> MetricResult:
    """Sequence length, exposed as the canonical len__value column."""
    L = len(c.sequence or "")
    return MetricResult(metric_name="len", values={"len__value": L})


def _ord_sum_metric(c: Candidate, params: dict) -> MetricResult:
    """Sum of ord(aa) -- a deterministic 'cheap energy'."""
    s = sum(ord(ch) for ch in (c.sequence or "")) % 100
    return MetricResult(metric_name="ord_sum", values={"ord_sum__value": float(s)})


def _slow_metric(c: Candidate, params: dict) -> MetricResult:
    """Pretends to be expensive; just the length / 10."""
    L = len(c.sequence or "")
    return MetricResult(metric_name="slow", values={"slow__value": float(L) / 10})


def _flaky_metric(c: Candidate, params: dict) -> MetricResult:
    """Raises on candidates whose id contains 'broken'."""
    if "broken" in c.candidate_id:
        raise RuntimeError("simulated broken upstream tool")
    return MetricResult(metric_name="flaky", values={"flaky__value": 1.0})


def _make_specs():
    return {
        "len": MetricSpec(name="len", fn=_len_metric, kind="seq", cost_seconds=0.0),
        "ord_sum": MetricSpec(name="ord_sum", fn=_ord_sum_metric, kind="seq", cost_seconds=0.0),
        "slow": MetricSpec(name="slow", fn=_slow_metric, kind="seq", cost_seconds=10.0, needs_gpu=True),
        "flaky": MetricSpec(name="flaky", fn=_flaky_metric, kind="seq", cost_seconds=0.0),
    }


# ----------------------------------------------------------------------
# MetricSpec / Registry
# ----------------------------------------------------------------------


def test_metricspec_default_prefix_uses_name_with_double_underscore():
    spec = MetricSpec(name="foo", fn=_len_metric, kind="seq", cost_seconds=0.0)
    assert spec.prefix == "foo__"


def test_metricspec_with_params_is_immutable_and_merges():
    spec = MetricSpec(
        name="len", fn=_len_metric, kind="seq", cost_seconds=0.0,
        default_params={"a": 1, "b": 2},
    )
    new = spec.with_params(b=5, c=3)
    assert new.default_params == {"a": 1, "b": 5, "c": 3}
    assert spec.default_params == {"a": 1, "b": 2}  # original not touched


def test_registry_register_and_lookup():
    r = MetricRegistry()
    spec = MetricSpec(name="foo", fn=_len_metric, kind="seq", cost_seconds=0.0)
    r.register(spec)
    assert "foo" in r
    assert r["foo"] is spec
    with pytest.raises(ValueError):
        r.register(spec)  # double-register


def test_registry_unknown_key_lists_available():
    r = MetricRegistry()
    r.register(MetricSpec(name="a", fn=_len_metric, kind="seq", cost_seconds=0))
    r.register(MetricSpec(name="b", fn=_len_metric, kind="seq", cost_seconds=0))
    with pytest.raises(KeyError) as exc:
        r["missing"]
    assert "available: a, b" in str(exc.value)


# ----------------------------------------------------------------------
# call_metric: timing + soft fail
# ----------------------------------------------------------------------


def test_call_metric_records_runtime():
    spec = _make_specs()["len"]
    cand = Candidate(candidate_id="c1", sequence="ACDEFG")
    res = call_metric(spec, cand)
    # call_metric injects the universal failed=0 sentinel on success.
    assert res.values == {"len__value": 6, "len__failed": 0}
    assert res.runtime_seconds >= 0


def test_call_metric_soft_fail_emits_failed_sentinel():
    spec = _make_specs()["flaky"]
    cand = Candidate(candidate_id="x_broken", sequence="ACD")
    res = call_metric(spec, cand)
    assert res.is_failed()
    assert res.values == {"flaky__failed": 1}


def test_call_metric_hard_fail_raises_when_soft_fail_disabled():
    spec = MetricSpec(
        name="flaky", fn=_flaky_metric, kind="seq",
        cost_seconds=0.0, soft_fail=False,
    )
    cand = Candidate(candidate_id="x_broken", sequence="ACD")
    with pytest.raises(RuntimeError):
        call_metric(spec, cand)


# ----------------------------------------------------------------------
# from_tool_result adapter
# ----------------------------------------------------------------------


class _ToolResult:
    def to_dict(self, prefix: str = "x__") -> dict:
        return {f"{prefix}a": 1, f"{prefix}b": 2.5}


def test_from_tool_result_appends_failed_sentinel():
    res = from_tool_result("rotamer_score", _ToolResult())
    assert res.values["rotamer_score__failed"] == 0
    assert res.values["rotamer_score__a"] == 1


def test_from_tool_result_with_error_overrides_values():
    res = from_tool_result("rotamer_score", _ToolResult(), error="boom")
    assert res.is_failed()
    assert res.values == {"rotamer_score__failed": 1}


# ----------------------------------------------------------------------
# Hashing helpers
# ----------------------------------------------------------------------


def test_hash_sequence_is_case_insensitive_and_stable():
    a = hash_sequence("ACDEFG")
    b = hash_sequence("acdefg")
    assert a == b
    assert len(a) == 16


def test_hash_params_canonical_key_order():
    a = hash_params({"x": 1, "y": 2})
    b = hash_params({"y": 2, "x": 1})
    assert a == b
    assert hash_params({}) == "noparams"


def test_hash_params_rejects_non_json_native_values():
    """Path / numpy scalars / dataclasses must be canonicalised by the
    caller, not silently stringified into a collision-prone hash."""
    from pathlib import Path as _P
    with pytest.raises(TypeError, match="JSON-native"):
        hash_params({"weights": _P("/tmp/x.pt")})
    # plain str is fine
    assert hash_params({"weights": "/tmp/x.pt"})


def test_hash_params_distinguishes_string_vs_int():
    """{"a": 1} and {"a": "1"} must hash distinctly (regression for
    default=str silent collision)."""
    a = hash_params({"a": 1})
    b = hash_params({"a": "1"})
    assert a != b


# ----------------------------------------------------------------------
# Cache: get/put + key derivation + JSONL persistence
# ----------------------------------------------------------------------


def test_inmemory_cache_get_put_roundtrip():
    cache = InMemoryCache()
    spec = _make_specs()["len"]
    cand = Candidate(candidate_id="c1", sequence="ACDEFG")
    key = make_cache_key(spec, cand, {})
    assert cache.get(key) is None
    res = call_metric(spec, cand)
    cache.put(key, res)
    got = cache.get(key)
    assert got is not None
    assert got.values["len__value"] == 6


def test_call_metric_cached_avoids_recompute():
    cache = InMemoryCache()
    spec = _make_specs()["len"]
    cand = Candidate(candidate_id="c1", sequence="ACDEFG")
    res1 = call_metric_cached(spec, cand, cache)
    res2 = call_metric_cached(spec, cand, cache)
    # Second call hits the cache; runtime gets restored from the original
    # call's recorded time, so values must be identical.
    assert res1.values == res2.values


def test_cache_key_changes_with_params():
    cache = InMemoryCache()
    spec = _make_specs()["len"]
    cand = Candidate(candidate_id="c1", sequence="ACDEFG")
    call_metric_cached(spec, cand, cache, params={"a": 1})
    call_metric_cached(spec, cand, cache, params={"a": 2})
    # Two different param dicts -> two cache entries.
    assert len(cache) == 2


def test_jsonl_cache_persists_across_instances(tmp_path: Path):
    p = tmp_path / "cache.jsonl"
    cache = JsonlCache(p)
    spec = _make_specs()["len"]
    cand = Candidate(candidate_id="c1", sequence="ACDEFG")
    call_metric_cached(spec, cand, cache)
    assert p.is_file()
    # Spin up a fresh instance and check the entry is found.
    cache2 = JsonlCache(p)
    key = make_cache_key(spec, cand, {})
    got = cache2.get(key)
    assert got is not None
    assert got.values["len__value"] == 6


def test_jsonl_cache_does_not_cache_per_residue(tmp_path: Path):
    p = tmp_path / "cache.jsonl"
    cache = JsonlCache(p)
    spec = MetricSpec(
        name="big", fn=lambda c, _: MetricResult(
            metric_name="big",
            values={"big__value": 1.0},
            per_residue=pd.DataFrame({"resno": [1, 2, 3]}),
        ),
        kind="seq", cost_seconds=0.0,
    )
    cand = Candidate(candidate_id="c1", sequence="A")
    res = call_metric_cached(spec, cand, cache)
    assert res.values["big__value"] == 1.0
    # round-trip
    cache2 = JsonlCache(p)
    res2 = cache2.get(make_cache_key(spec, cand, {}))
    assert res2 is not None
    assert res2.per_residue is None        # explicitly NOT serialized


def test_failure_caching_default_is_on():
    cache = InMemoryCache()
    spec = _make_specs()["flaky"]
    cand = Candidate(candidate_id="x_broken", sequence="A")
    call_metric_cached(spec, cand, cache)
    assert len(cache) == 1


def test_failure_caching_can_be_disabled():
    cache = InMemoryCache()
    spec = _make_specs()["flaky"]
    cand = Candidate(candidate_id="x_broken", sequence="A")
    call_metric_cached(spec, cand, cache, cache_failures=False)
    assert len(cache) == 0


# ----------------------------------------------------------------------
# Tier scheduler: ordering, gating, survivor_topk, log
# ----------------------------------------------------------------------


def _candidates(seqs):
    return [Candidate(candidate_id=f"c{i}", sequence=s) for i, s in enumerate(seqs)]


def test_evaluate_tiered_runs_all_metrics_when_no_constraints():
    specs = _make_specs()
    plan = TierPlan(tiers=[[specs["len"]], [specs["ord_sum"]]])
    cands = _candidates(["ACDE", "GHIK"])
    res = evaluate_tiered(cands, plan)
    df = res.metrics_df
    assert set(df["candidate_id"]) == {"c0", "c1"}
    for col in ("len__value", "ord_sum__value"):
        assert col in df.columns
    assert len(res.survivors) == 2
    assert len(res.tier_log) == 2


def test_evaluate_tiered_drops_candidates_failing_constraint():
    specs = _make_specs()
    plan = TierPlan(
        tiers=[[specs["len"]]],
        constraints_per_tier=[[
            TierConstraint("len__value", ">=", 5, description="long enough"),
        ]],
    )
    res = evaluate_tiered(_candidates(["ACDE", "GHIKL"]), plan)
    # only the longer one survives
    assert res.survivors == ["c1"]
    # both are still in the metrics_df (so consumers can see why c0 was dropped)
    assert set(res.metrics_df["candidate_id"]) == {"c0", "c1"}
    fails = res.constraint_failures
    assert len(fails) == 1
    assert fails.iloc[0]["candidate_id"] == "c0"
    assert fails.iloc[0]["constraint_column"] == "len__value"


def test_evaluate_tiered_topk_picks_best_by_first_metric_col():
    specs = _make_specs()
    plan = TierPlan(
        tiers=[[specs["len"]]],
        survivor_topk_per_tier=[1],
        rank_ascending_per_tier=[False],   # higher = better
    )
    res = evaluate_tiered(_candidates(["AC", "ACDEFG", "ACDE"]), plan)
    # The longest survives (ACDEFG = c1, length 6).
    assert res.survivors == ["c1"]


def test_evaluate_tiered_does_not_crash_when_all_drop_out():
    specs = _make_specs()
    plan = TierPlan(
        tiers=[[specs["len"]], [specs["ord_sum"]]],
        constraints_per_tier=[
            [TierConstraint("len__value", ">=", 100)],   # nothing passes
            [],
        ],
    )
    res = evaluate_tiered(_candidates(["AC", "GH"]), plan)
    assert res.survivors == []
    # tier 1 (after the empty survivor set) still appears in the log
    assert len(res.tier_log) == 2
    assert res.tier_log[1].n_candidates_in == 0


def test_evaluate_tiered_caches_repeated_calls():
    specs = _make_specs()
    plan = TierPlan(tiers=[[specs["len"]]])
    cache = InMemoryCache()
    cands = _candidates(["ACDE", "ACDE"])  # duplicate sequence, different ids
    res = evaluate_tiered(cands, plan, cache=cache)
    # Same sequence -> 1 cache key; first call writes, second hits.
    assert res.tier_log[0].cache_hits == 1
    assert res.tier_log[0].cache_misses == 1


def test_evaluate_tiered_orders_cpu_before_gpu_within_tier():
    """For batching: CPU specs should run before GPU specs within a tier
    so all GPU calls happen back-to-back."""
    calls: list[str] = []

    def cpu_fn(c, _):
        calls.append("cpu")
        return MetricResult(metric_name="cpu", values={"cpu__v": 1})

    def gpu_fn(c, _):
        calls.append("gpu")
        return MetricResult(metric_name="gpu", values={"gpu__v": 1})

    cpu = MetricSpec(name="cpu", fn=cpu_fn, kind="seq", cost_seconds=0)
    gpu = MetricSpec(name="gpu", fn=gpu_fn, kind="seq", cost_seconds=0, needs_gpu=True)

    plan = TierPlan(tiers=[[gpu, cpu]])  # intentionally GPU first in the spec list
    evaluate_tiered(_candidates(["AC", "GH"]), plan)
    # the scheduler reorders so all CPU runs before all GPU
    assert calls == ["cpu", "cpu", "gpu", "gpu"]


def test_evaluate_tiered_constraint_violation_is_logged():
    specs = _make_specs()
    plan = TierPlan(
        tiers=[[specs["len"]]],
        constraints_per_tier=[[TierConstraint("len__value", ">=", 100, description="length floor")]],
    )
    res = evaluate_tiered(_candidates(["AC", "GHIJ"]), plan)
    assert set(res.constraint_failures["candidate_id"]) == {"c0", "c1"}
    assert all(res.constraint_failures["constraint_column"] == "len__value")


def test_write_tier_log_persists_tsv(tmp_path: Path):
    specs = _make_specs()
    plan = TierPlan(tiers=[[specs["len"]]])
    res = evaluate_tiered(_candidates(["AC"]), plan)
    out_path = write_tier_log(tmp_path, res.tier_log)
    assert out_path.is_file()
    df = pd.read_csv(out_path, sep="\t")
    assert "tier_idx" in df.columns
    assert df.iloc[0]["n_candidates_in"] == 1


def test_evaluate_tiered_handles_soft_failure_without_dropping_row():
    specs = _make_specs()
    plan = TierPlan(tiers=[[specs["flaky"]]])
    res = evaluate_tiered(_candidates(["AC", "GH"]), plan)
    # both survive (no constraints); broken one has flaky__failed=1
    assert len(res.survivors) == 2
    # dummy candidates have ids 'c0' / 'c1' -- no 'broken' so flaky never fails.
    # Now force one with 'broken' in its id:
    cands = [Candidate(candidate_id="c0_broken", sequence="AC"),
             Candidate(candidate_id="c1", sequence="GH")]
    res = evaluate_tiered(cands, plan)
    df = res.metrics_df
    flaky_failed = dict(zip(df["candidate_id"], df["flaky__failed"]))
    assert flaky_failed["c0_broken"] == 1
    assert flaky_failed["c1"] == 0


# ----------------------------------------------------------------------
# Regression tests for cross-review findings
# ----------------------------------------------------------------------


def test_registry_rejects_exact_prefix_collision():
    """Two metrics that emit columns under the same prefix would
    silently overwrite each other -- the registry must refuse.
    """
    r = MetricRegistry()
    r.register(MetricSpec(name="a", fn=_len_metric, kind="seq", cost_seconds=0.0, prefix="ester__"))
    with pytest.raises(ValueError, match="prefix"):
        r.register(MetricSpec(name="b", fn=_len_metric, kind="seq", cost_seconds=0.0, prefix="ester__"))


def test_registry_rejects_string_prefix_overlap():
    """Custom prefixes that are string-prefix-related must be rejected
    so column "ester__deep__x" can't ambiguously belong to two metrics.
    """
    r = MetricRegistry()
    r.register(MetricSpec(name="a", fn=_len_metric, kind="seq", cost_seconds=0.0, prefix="ester__"))
    with pytest.raises(ValueError, match="string-prefix"):
        r.register(MetricSpec(name="b", fn=_len_metric, kind="seq", cost_seconds=0.0, prefix="ester__deep__"))


def test_call_metric_rejects_columns_that_dont_start_with_prefix():
    """A metric implementation that emits a key NOT starting with its
    declared prefix could silently overwrite another metric's column.
    """
    def _bad_metric(c, _):
        return MetricResult(metric_name="x", values={"y__leaked": 1.0})  # wrong prefix

    spec = MetricSpec(name="x", fn=_bad_metric, kind="seq", cost_seconds=0.0)
    cand = Candidate(candidate_id="c1", sequence="A")
    res = call_metric(spec, cand)
    # The bad column must be flagged + scrubbed; the metric is treated as soft-failed.
    assert res.is_failed()
    assert "y__leaked" not in res.values


def test_evaluate_tiered_rejects_duplicate_candidate_ids():
    """Two candidates with the same id silently merged metric rows
    before the fix; now we raise."""
    specs = _make_specs()
    plan = TierPlan(tiers=[[specs["len"]]])
    cands = [
        Candidate(candidate_id="dup", sequence="AC"),
        Candidate(candidate_id="dup", sequence="GHIK"),  # same id, diff seq
    ]
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        evaluate_tiered(cands, plan)


def test_topk_does_not_pick_failed_candidate():
    """A soft-failed candidate must NOT win top-K regardless of sort
    direction (regression: NaN -> inf used to outrank real scores when
    sorting descending).
    """
    def _good_metric(c, _):
        # ord("A") == 65, ord("Z") == 90 -- bigger seq -> bigger score
        s = sum(ord(x) for x in c.sequence)
        return MetricResult(metric_name="g", values={"g__score": float(s)})

    def _broken_metric(c, _):
        if "broken" in c.candidate_id:
            raise RuntimeError("upstream toy fail")
        s = sum(ord(x) for x in c.sequence)
        return MetricResult(metric_name="g", values={"g__score": float(s)})

    spec = MetricSpec(name="g", fn=_broken_metric, kind="seq", cost_seconds=0.0)
    plan = TierPlan(
        tiers=[[spec]],
        survivor_topk_per_tier=[1],
        rank_score_col_per_tier=["g__score"],
        rank_ascending_per_tier=[False],   # higher = better
    )
    cands = [
        Candidate(candidate_id="ok_short", sequence="AA"),
        Candidate(candidate_id="ok_long_broken", sequence="ZZZZZZZZZZ"),
    ]
    res = evaluate_tiered(cands, plan)
    # Despite "ok_long_broken" having the longer sequence, it failed and
    # must not win top-K=1.
    assert res.survivors == ["ok_short"]


def test_constraint_failures_classifies_missing_vs_nan_vs_violation():
    """failure_reason categorises why each constraint failed."""
    specs = _make_specs()

    def _nan_metric(c, _):
        return MetricResult(metric_name="x", values={"x__score": float("nan")})

    nan_spec = MetricSpec(name="x", fn=_nan_metric, kind="seq", cost_seconds=0.0)

    plan = TierPlan(
        tiers=[[specs["len"], nan_spec]],
        constraints_per_tier=[[
            TierConstraint("len__value", ">=", 100),    # numeric violation
            TierConstraint("missing__col", ">=", 0),    # missing col
            TierConstraint("x__score", ">=", 0),         # nan
        ]],
    )
    res = evaluate_tiered(_candidates(["AC"]), plan)
    failures = res.constraint_failures
    reasons = dict(zip(failures["constraint_column"], failures["failure_reason"]))
    assert reasons["len__value"] == "violation"
    assert reasons["missing__col"] == "missing"
    assert reasons["x__score"] == "nan"


def test_constraint_failures_serializes_nan_as_string():
    """observed_value of NaN must round-trip as the string 'nan' so
    TSV / JSON consumers don't choke."""
    def _nan_metric(c, _):
        return MetricResult(metric_name="x", values={"x__score": float("nan")})

    spec = MetricSpec(name="x", fn=_nan_metric, kind="seq", cost_seconds=0.0)
    plan = TierPlan(
        tiers=[[spec]],
        constraints_per_tier=[[TierConstraint("x__score", ">=", 0)]],
    )
    res = evaluate_tiered(_candidates(["AC"]), plan)
    assert res.constraint_failures.iloc[0]["observed_value"] == "nan"


def test_cache_key_includes_metric_version():
    """Bumping MetricSpec.cache_version busts that metric's entries
    without affecting other metrics."""
    specs = _make_specs()
    spec_v1 = specs["len"]
    spec_v2 = MetricSpec(
        name="len_v2", fn=_len_metric, kind="seq", cost_seconds=0.0,
        cache_version=2,
    )
    cand = Candidate(candidate_id="c1", sequence="AC")
    k1 = make_cache_key(spec_v1, cand, {})
    k2 = make_cache_key(spec_v2, cand, {})
    assert k1.metric_version == 1
    assert k2.metric_version == 2
    assert k1.as_str() != k2.as_str()


def test_cache_key_includes_resolved_provenance():
    """cache_provenance() result becomes part of the params hash so
    e.g. a weights mtime change busts the entry."""
    specs_v1 = MetricSpec(
        name="w", fn=_len_metric, kind="seq", cost_seconds=0.0,
        cache_provenance=lambda: {"weights_mtime": 100},
    )
    specs_v2 = MetricSpec(
        name="w", fn=_len_metric, kind="seq", cost_seconds=0.0,
        cache_provenance=lambda: {"weights_mtime": 200},
    )
    cand = Candidate(candidate_id="c1", sequence="AC")
    k1 = make_cache_key(specs_v1, cand, {})
    k2 = make_cache_key(specs_v2, cand, {})
    assert k1.params_hash != k2.params_hash


def test_cache_key_distinguishes_seq_vs_struct_with_nul_delimiter():
    """seq+structure hash must use a delimiter that can't appear in
    either component, so seq='AA' struct_hash='BB_CC' doesn't alias
    seq='AA_BB' struct_hash='CC'.
    """
    # We test indirectly: same seq, different structures should never collide.
    spec = MetricSpec(name="ss", fn=_len_metric, kind="seq+structure", cost_seconds=0.0)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as fh1:
        fh1.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
        p1 = Path(fh1.name)
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as fh2:
        fh2.write("ATOM      1  N   ALA A   1      10.000   0.000   0.000  1.00  0.00           N\n")
        p2 = Path(fh2.name)
    try:
        c1 = Candidate(candidate_id="c1", sequence="A", structure_path=p1)
        c2 = Candidate(candidate_id="c2", sequence="A", structure_path=p2)
        k1 = make_cache_key(spec, c1, {})
        k2 = make_cache_key(spec, c2, {})
        assert k1.input_hash != k2.input_hash
    finally:
        p1.unlink(missing_ok=True)
        p2.unlink(missing_ok=True)


def test_jsonl_cache_sees_sibling_writes_after_mtime_change(tmp_path: Path):
    """Two JsonlCache instances pointing at the same file: the second
    must see writes from the first (mirror-coherence regression).
    """
    p = tmp_path / "shared.jsonl"
    cache_a = JsonlCache(p)
    cache_b = JsonlCache(p)

    spec = _make_specs()["len"]
    cand = Candidate(candidate_id="c1", sequence="ACDEFG")
    key = make_cache_key(spec, cand, {})

    # writer A puts an entry
    res = call_metric(spec, cand)
    cache_a.put(key, res)

    # reader B should see it on next get (mtime-based reload)
    got = cache_b.get(key)
    assert got is not None
    assert got.values["len__value"] == 6


def test_jsonl_cache_lock_timeout_fails_closed(tmp_path: Path):
    """If the lock can't be acquired in time, put() must raise rather
    than write unlocked (correctness > availability)."""
    import fcntl
    p = tmp_path / "cache.jsonl"
    cache = JsonlCache(p, lock_timeout_s=1.0)

    # Hold the lock from outside the cache so the put() can never grab it.
    cache._lockfile.touch()
    holder = open(cache._lockfile, "rb+")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    try:
        spec = _make_specs()["len"]
        cand = Candidate(candidate_id="c1", sequence="AC")
        with pytest.raises(TimeoutError):
            call_metric_cached(spec, cand, cache)
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()


def test_hash_structure_distinguishes_element_column(tmp_path: Path):
    """Regression: hash_structure used to truncate at byte 66, dropping
    PDB v3.30+ element column (76-77). Different elements with same
    coordinates must produce different hashes.
    """
    from protein_chisel.scoring.metrics import hash_structure
    base = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           "
    p_n = tmp_path / "n.pdb"
    p_c = tmp_path / "c.pdb"
    p_n.write_text(base + "N\n")
    p_c.write_text(base + "C\n")
    h1 = hash_structure(p_n)
    h2 = hash_structure(p_c)
    assert h1 != h2


def test_cache_key_includes_ligand_params_path(tmp_path: Path):
    """structure+ligand metrics must distinguish two structures with
    different ligand .params files."""
    p = tmp_path / "test.pdb"
    p.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
    p_lig1 = tmp_path / "lig1.params"
    p_lig2 = tmp_path / "lig2.params"
    p_lig1.write_text("NAME LIG1\n")
    p_lig2.write_text("NAME LIG2\n")

    spec = MetricSpec(name="sl", fn=_len_metric, kind="structure+ligand", cost_seconds=0.0)
    c1 = Candidate(candidate_id="c1", sequence=None, structure_path=p, ligand_params_path=p_lig1)
    c2 = Candidate(candidate_id="c2", sequence=None, structure_path=p, ligand_params_path=p_lig2)
    k1 = make_cache_key(spec, c1, {})
    k2 = make_cache_key(spec, c2, {})
    assert k1.input_hash != k2.input_hash


def test_cache_key_includes_catalytic_resnos(tmp_path: Path):
    """Different catalytic_resnos must produce different cache keys
    even with identical sequence/structure."""
    p = tmp_path / "test.pdb"
    p.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
    spec = MetricSpec(name="rstest", fn=_len_metric, kind="structure", cost_seconds=0.0)
    c1 = Candidate(candidate_id="c1", structure_path=p, catalytic_resnos=(1, 2, 3))
    c2 = Candidate(candidate_id="c2", structure_path=p, catalytic_resnos=(1, 2, 4))
    k1 = make_cache_key(spec, c1, {})
    k2 = make_cache_key(spec, c2, {})
    assert k1.input_hash != k2.input_hash


def test_first_metric_col_skips_failed_sentinel():
    """Default ranking column heuristic should NOT pick `<prefix>failed`."""
    from protein_chisel.scoring.tier import _first_metric_col_in
    spec = MetricSpec(name="t", fn=_len_metric, kind="seq", cost_seconds=0.0)
    rows = {
        "c1": {
            "candidate_id": "c1",
            "t__failed": 0,
            "t__score": 4.2,
        },
    }
    col = _first_metric_col_in([spec], rows)
    assert col == "t__score"
