"""Tests for pipelines/tier_filter.py.

Two flavours:

- Unit tests (default): build a TierPlan from synthetic in-process
  metrics so we exercise the orchestration layer without touching any
  apptainer image. Fast (< 1 s).

- Cluster integration test (cluster-marked): end-to-end against the
  real metric_specs adapters, on the design.pdb test fixture. This
  catches actual sif startup + tool invocation issues that the unit
  tests can't.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from protein_chisel.pipelines.tier_filter import (
    CandidateInput,
    TierFilterConfig,
    TierFilterResult,
    default_enzyme_tier_plan,
    tier_filter,
)
from protein_chisel.scoring.metrics import (
    Candidate, MetricRegistry, MetricResult, MetricSpec, get_registry,
)
from protein_chisel.scoring.tier import TierConstraint, TierPlan


# ----------------------------------------------------------------------
# Synthetic-metric helpers
# ----------------------------------------------------------------------


def _make_dummy_pdbs(tmp_path: Path, n: int, lengths: list[int]) -> list[Path]:
    """Write `n` minimal PDB files with `lengths[i]` ATOM lines each."""
    paths = []
    for i in range(n):
        p = tmp_path / f"design_{i}.pdb"
        with open(p, "w") as fh:
            for j in range(lengths[i]):
                fh.write(
                    f"ATOM  {j+1:>5d}  CA  ALA A{j+1:>4d}    "
                    f"{i*10.0:8.3f}{j*1.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
                )
            fh.write("END\n")
        paths.append(p)
    return paths


def _len_metric(c: Candidate, params: dict) -> MetricResult:
    """Score = number of CA atoms in the PDB (deterministic)."""
    n_ca = sum(
        1 for line in Path(c.structure_path).read_text().splitlines()
        if line.startswith("ATOM") and line[12:16].strip() == "CA"
    )
    return MetricResult(
        metric_name="dummy_len",
        values={"dummy_len__n_ca": n_ca, "dummy_len__score": float(n_ca)},
    )


def _slow_metric(c: Candidate, params: dict) -> MetricResult:
    """Pretends to be expensive; computes a different deterministic score."""
    n_ca = sum(
        1 for line in Path(c.structure_path).read_text().splitlines()
        if line.startswith("ATOM") and line[12:16].strip() == "CA"
    )
    return MetricResult(
        metric_name="dummy_slow",
        values={"dummy_slow__expensive_score": float(n_ca) / 10},
    )


@pytest.fixture
def synthetic_specs() -> dict:
    """Two synthetic specs we can compose into TierPlans without touching
    any sif. Each test gets a fresh dict so prefix-collision rules don't
    leak between tests via the global registry.
    """
    return {
        "len": MetricSpec(
            name="dummy_len", fn=_len_metric, kind="structure",
            cost_seconds=0.0, needs_gpu=False,
        ),
        "slow": MetricSpec(
            name="dummy_slow", fn=_slow_metric, kind="structure",
            cost_seconds=10.0, needs_gpu=True,
        ),
    }


# ----------------------------------------------------------------------
# Unit tests (no apptainer)
# ----------------------------------------------------------------------


def test_tier_filter_validates_missing_structure(tmp_path: Path, synthetic_specs):
    plan = TierPlan(tiers=[[synthetic_specs["len"]]])
    cfg = TierFilterConfig(out_dir=tmp_path, write_outputs=False)
    cands = [CandidateInput(candidate_id="c0", structure_path=tmp_path / "missing.pdb")]
    with pytest.raises(FileNotFoundError, match="missing.pdb"):
        tier_filter(cands, plan, cfg)


def test_tier_filter_validates_missing_ligand_params(tmp_path: Path, synthetic_specs):
    pdbs = _make_dummy_pdbs(tmp_path, 1, [3])
    plan = TierPlan(tiers=[[synthetic_specs["len"]]])
    cfg = TierFilterConfig(out_dir=tmp_path, write_outputs=False)
    cands = [CandidateInput(
        candidate_id="c0",
        structure_path=pdbs[0],
        ligand_params_path=tmp_path / "no_such.params",
    )]
    with pytest.raises(FileNotFoundError, match="no_such.params"):
        tier_filter(cands, plan, cfg)


def test_tier_filter_rejects_empty_input(tmp_path: Path, synthetic_specs):
    plan = TierPlan(tiers=[[synthetic_specs["len"]]])
    cfg = TierFilterConfig(out_dir=tmp_path, write_outputs=False)
    with pytest.raises(ValueError, match="empty candidate list"):
        tier_filter([], plan, cfg)


def test_tier_filter_runs_two_tiers_persists_outputs(tmp_path: Path, synthetic_specs):
    pdbs = _make_dummy_pdbs(tmp_path, 4, [3, 5, 8, 12])
    plan = TierPlan(
        tiers=[[synthetic_specs["len"]], [synthetic_specs["slow"]]],
        constraints_per_tier=[
            [TierConstraint("dummy_len__n_ca", ">=", 5, description="cheap floor")],
            [],
        ],
        rank_score_col_per_tier=["dummy_len__score", "dummy_slow__expensive_score"],
        rank_ascending_per_tier=[False, False],
    )
    cands = [
        CandidateInput(candidate_id=f"c{i}", structure_path=p)
        for i, p in enumerate(pdbs)
    ]
    cfg = TierFilterConfig(out_dir=tmp_path / "run")
    res = tier_filter(cands, plan, cfg)

    # c0 has 3 CAs -> should be dropped at tier 0; c1/c2/c3 survive.
    assert len(res.evaluation.survivors) == 3
    assert "c0" not in res.evaluation.survivors

    # On disk
    out_dir = tmp_path / "run"
    assert (out_dir / "metrics.tsv").is_file()
    assert (out_dir / "survivors.tsv").is_file()
    assert (out_dir / "tier_log.tsv").is_file()
    assert (out_dir / "constraint_failures.tsv").is_file()
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "metric_cache.jsonl").is_file()

    # Survivors sorted by the LAST tier's ranking column (descending here).
    survivors = pd.read_csv(out_dir / "survivors.tsv", sep="\t")
    expensive = survivors["dummy_slow__expensive_score"].tolist()
    assert expensive == sorted(expensive, reverse=True)

    # Manifest sanity
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["pipeline"] == "tier_filter"
    assert manifest["n_candidates_in"] == 4
    assert manifest["n_survivors"] == 3
    assert len(manifest["tier_plan"]) == 2
    assert manifest["tier_plan"][0]["metrics"][0]["name"] == "dummy_len"


def test_tier_filter_reuses_cache_across_calls(tmp_path: Path, synthetic_specs):
    """Second call to tier_filter on the same candidates should hit the cache
    and return immediately.
    """
    pdbs = _make_dummy_pdbs(tmp_path, 2, [5, 7])
    plan = TierPlan(tiers=[[synthetic_specs["len"]]])
    cands = [
        CandidateInput(candidate_id=f"c{i}", structure_path=p)
        for i, p in enumerate(pdbs)
    ]
    cfg_a = TierFilterConfig(out_dir=tmp_path / "run")
    cfg_b = TierFilterConfig(out_dir=tmp_path / "run")  # same dir -> shared cache

    res_a = tier_filter(cands, plan, cfg_a)
    assert res_a.evaluation.tier_log[0].cache_hits == 0
    assert res_a.evaluation.tier_log[0].cache_misses == 2

    res_b = tier_filter(cands, plan, cfg_b)
    assert res_b.evaluation.tier_log[0].cache_hits == 2
    assert res_b.evaluation.tier_log[0].cache_misses == 0


def test_tier_filter_constraint_failures_persisted(tmp_path: Path, synthetic_specs):
    pdbs = _make_dummy_pdbs(tmp_path, 2, [3, 8])
    plan = TierPlan(
        tiers=[[synthetic_specs["len"]]],
        constraints_per_tier=[[
            TierConstraint("dummy_len__n_ca", ">=", 100, description="impossibly tall"),
        ]],
    )
    cands = [
        CandidateInput(candidate_id=f"c{i}", structure_path=p)
        for i, p in enumerate(pdbs)
    ]
    cfg = TierFilterConfig(out_dir=tmp_path / "run")
    res = tier_filter(cands, plan, cfg)

    assert res.evaluation.survivors == []
    failures = pd.read_csv(tmp_path / "run" / "constraint_failures.tsv", sep="\t")
    assert len(failures) == 2
    assert set(failures["constraint_column"]) == {"dummy_len__n_ca"}
    # All should be classified as "violation" (numeric < 100), not "missing"/"nan"
    assert set(failures["failure_reason"]) == {"violation"}


def test_default_enzyme_tier_plan_has_4_tiers():
    """Smoke test: the canonical plan factory returns the structure
    documented in the architectural review."""
    # We don't actually run it, just check shape -- the metric_specs
    # registry import shouldn't crash even without the wrappers' deps.
    fresh_registry = MetricRegistry()
    plan = default_enzyme_tier_plan(registry=fresh_registry, enable_attnpacker=False)
    assert len(plan.tiers) == 4
    assert [t for t in plan.tier_names] == [
        "cheap_cpu", "cheap_kde", "pocket_geom", "learned_likelihood",
    ]
    # Tier 3 has 2 metrics by default (pippack + flowpacker).
    assert len(plan.tiers[3]) == 2


def test_default_enzyme_tier_plan_respects_disable_flags():
    fresh_registry = MetricRegistry()
    plan = default_enzyme_tier_plan(
        registry=fresh_registry,
        enable_pippack=False,
        enable_flowpacker=True,
        enable_attnpacker=True,
    )
    tier3_names = [s.name for s in plan.tiers[3]]
    assert "pippack" not in tier3_names
    assert "flowpacker" in tier3_names
    assert "attnpacker" in tier3_names


def test_default_enzyme_tier_plan_constraints_match_kwargs():
    fresh_registry = MetricRegistry()
    plan = default_enzyme_tier_plan(
        registry=fresh_registry,
        fa_dun_max_mean=4.2,
        rotalyze_max_outlier_frac=0.07,
        fpocket_min_druggability=0.3,
    )
    # Constraint values flow through.
    fa_dun_constraint = plan.constraints_per_tier[0][0]
    assert fa_dun_constraint.value == 4.2
    rotalyze_constraint = plan.constraints_per_tier[1][0]
    assert rotalyze_constraint.value == 0.07
    fpocket_constraint = plan.constraints_per_tier[2][0]
    assert fpocket_constraint.value == 0.3


# ----------------------------------------------------------------------
# Regression tests for second-round review findings
# ----------------------------------------------------------------------


def test_tier_filter_rejects_duplicate_candidate_ids(tmp_path: Path, synthetic_specs):
    pdbs = _make_dummy_pdbs(tmp_path, 2, [3, 5])
    plan = TierPlan(tiers=[[synthetic_specs["len"]]])
    cands = [
        CandidateInput(candidate_id="dup", structure_path=pdbs[0]),
        CandidateInput(candidate_id="dup", structure_path=pdbs[1]),  # same id
    ]
    cfg = TierFilterConfig(out_dir=tmp_path, write_outputs=False)
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        tier_filter(cands, plan, cfg)


def test_tier_filter_sort_uses_correct_tier_ascending(tmp_path: Path, synthetic_specs):
    """When the LAST tier has rank_score_col=None, we should fall back
    to an EARLIER tier's ranking column AND its matching ascending flag,
    not the last tier's ascending (which was unrelated)."""
    pdbs = _make_dummy_pdbs(tmp_path, 3, [3, 6, 9])
    plan = TierPlan(
        tiers=[[synthetic_specs["len"]], [synthetic_specs["slow"]]],
        rank_score_col_per_tier=["dummy_len__score", None],
        rank_ascending_per_tier=[False, True],   # tier 0 desc, tier 1 asc
    )
    cands = [
        CandidateInput(candidate_id=f"c{i}", structure_path=p)
        for i, p in enumerate(pdbs)
    ]
    cfg = TierFilterConfig(out_dir=tmp_path / "run")
    res = tier_filter(cands, plan, cfg)
    # Last non-None rank col is tier 0's "dummy_len__score" with ascending=False.
    # Survivors should be sorted descending by that column.
    scores = res.survivors_df["dummy_len__score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_tier_filter_constraint_failures_tsv_always_written(tmp_path: Path, synthetic_specs):
    """Even when zero failures occur, constraint_failures.tsv must be
    written (with headers only) so a previous run's stale failures
    don't leak into this run's output dir."""
    pdbs = _make_dummy_pdbs(tmp_path, 2, [5, 7])
    plan = TierPlan(
        tiers=[[synthetic_specs["len"]]],
        constraints_per_tier=[
            [TierConstraint("dummy_len__n_ca", ">=", 1)],   # everyone passes
        ],
    )
    cands = [
        CandidateInput(candidate_id=f"c{i}", structure_path=p)
        for i, p in enumerate(pdbs)
    ]
    cfg = TierFilterConfig(out_dir=tmp_path / "run")
    res = tier_filter(cands, plan, cfg)
    # All passed, but the file still exists with headers (or empty).
    failures_path = tmp_path / "run" / "constraint_failures.tsv"
    assert failures_path.is_file()
    # And no orphan rows.
    df = pd.read_csv(failures_path, sep="\t") if failures_path.stat().st_size > 0 else pd.DataFrame()
    assert len(df) == 0


def test_tier_filter_manifest_has_schema_version_and_artifacts(tmp_path: Path, synthetic_specs):
    pdbs = _make_dummy_pdbs(tmp_path, 1, [5])
    plan = TierPlan(tiers=[[synthetic_specs["len"]]])
    cands = [CandidateInput(candidate_id="c0", structure_path=pdbs[0])]
    cfg = TierFilterConfig(out_dir=tmp_path / "run")
    tier_filter(cands, plan, cfg)
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["schema_version"] >= 1
    assert "artifacts" in manifest
    for k in ("metrics_tsv", "survivors_tsv", "constraint_failures_tsv"):
        assert k in manifest["artifacts"]
        assert "sha256" in manifest["artifacts"][k]


def test_tier_filter_atomic_writes_leave_no_tmp_files(tmp_path: Path, synthetic_specs):
    """After a successful run, no .tmp files should be left behind."""
    pdbs = _make_dummy_pdbs(tmp_path, 1, [5])
    plan = TierPlan(tiers=[[synthetic_specs["len"]]])
    cands = [CandidateInput(candidate_id="c0", structure_path=pdbs[0])]
    cfg = TierFilterConfig(out_dir=tmp_path / "run")
    tier_filter(cands, plan, cfg)
    leftovers = list((tmp_path / "run").glob("*.tmp"))
    assert leftovers == []


def test_tier_filter_explicit_cache_logs_when_shadowing(tmp_path: Path, synthetic_specs, caplog):
    """Passing an explicit cache when a metric_cache.jsonl exists in
    out_dir should log a warning so the surprise is visible."""
    import logging
    from protein_chisel.scoring.cache import InMemoryCache

    out_dir = tmp_path / "run"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Pre-populate the auto-cache path so the shadow log fires.
    (out_dir / "metric_cache.jsonl").write_text("")

    pdbs = _make_dummy_pdbs(tmp_path, 1, [5])
    plan = TierPlan(tiers=[[synthetic_specs["len"]]])
    cands = [CandidateInput(candidate_id="c0", structure_path=pdbs[0])]
    cfg = TierFilterConfig(out_dir=out_dir, cache=InMemoryCache())

    with caplog.at_level(logging.INFO, logger="protein_chisel.tier_filter"):
        tier_filter(cands, plan, cfg)

    msgs = [r.message for r in caplog.records]
    assert any("shadows existing" in m for m in msgs)


def test_default_enzyme_tier_plan_warns_about_pyrosetta_dep_in_docs():
    """The default plan documents that tier 0 needs pyrosetta on the
    host. Until we add a sif-invoking adapter, callers using
    default_enzyme_tier_plan from a host without pyrosetta will see
    everyone soft-fail tier 0. Explicit test asserts the docstring
    flags this so future readers know it's a known limitation."""
    doc = default_enzyme_tier_plan.__doc__ or ""
    # We don't enforce particular wording, just that the limitation
    # is mentioned somewhere in the docstring.
    assert "tier 0" in doc.lower() or "fa_dun" in doc.lower()


# ----------------------------------------------------------------------
# Cluster integration (real metric specs, design.pdb)
# ----------------------------------------------------------------------


@pytest.mark.cluster
def test_tier_filter_real_metric_specs_on_design_pdb(tmp_path: Path):
    """End-to-end: real rotalyze + fpocket on design.pdb.

    Drops fa_dun/rotamer_score because that adapter currently requires
    pyrosetta to be importable in the host process; on the head node
    the metric soft-fails. (Follow-up: build a sif-invoking adapter
    that calls pyrosetta.sif via subprocess.)

    Drops the GPU tier (pippack/flowpacker/attnpacker) so this stays a
    cluster test rather than a slow test.
    """
    design_pdb = Path("/home/woodbuse/testing_space/align_seth_test/design.pdb")
    if not design_pdb.is_file():
        pytest.skip(f"design.pdb fixture missing at {design_pdb}")

    from protein_chisel.tools.sidechain_packing_and_scoring.metric_specs import (
        FPOCKET_SPEC, ROTALYZE_SPEC,
    )

    plan = TierPlan(
        tiers=[[ROTALYZE_SPEC], [FPOCKET_SPEC]],
        constraints_per_tier=[
            [TierConstraint("rotalyze__frac_outliers", "<=", 0.5)],
            [],
        ],
        rank_score_col_per_tier=[
            "rotalyze__frac_outliers",
            "fpocket__most_druggable_score",
        ],
        rank_ascending_per_tier=[True, False],
        tier_names=["rotalyze", "fpocket"],
    )
    cands = [CandidateInput(candidate_id="design", structure_path=design_pdb)]
    cfg = TierFilterConfig(out_dir=tmp_path / "design_run", verbose=True)
    res = tier_filter(cands, plan, cfg)

    metrics = pd.read_csv(tmp_path / "design_run" / "metrics.tsv", sep="\t")
    expected = [
        "rotalyze__frac_outliers",
        "fpocket__most_druggable_score",
    ]
    for col in expected:
        assert col in metrics.columns, (
            f"expected {col} in metrics columns: {metrics.columns.tolist()}"
        )

    # Survivors should include design (a real protein PDB passes both).
    assert res.evaluation.survivors == ["design"]
