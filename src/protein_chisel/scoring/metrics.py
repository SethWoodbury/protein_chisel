"""Generic metric protocol + registry.

A *metric* in protein_chisel is any deterministic function that takes a
candidate (sequence, structure, or both) plus a parameter bundle and
returns a small bag of float / int / bool values + provenance metadata.
The registry lets pipelines compose metrics by name without having to
import individual tool modules, which keeps the tier-scheduler / cache
generic.

Why this layer exists
=====================
Most pipelines today (``pipelines/comprehensive_metrics.py``) hard-code
each tool's import + invocation. That makes:
  - tier scheduling impossible (we can't say "run cheap metrics first,
    then expensive ones on the survivors" without knowing each tool's
    cost),
  - caching impossible (we can't key by metric name + params if there
    isn't a metric name + params),
  - and config-driven objective changes painful (changing the fa_dun
    threshold requires editing pipeline code).

By forcing every tool through ``MetricSpec`` + ``MetricResult``, the
pipeline layer can be agnostic to which tool produces which value.

Contract
========
Every protein_chisel tool that wants to be schedulable should expose a
function::

    def score(<candidate>, params: dict) -> MetricResult: ...

where ``<candidate>`` is one of the canonical input kinds
(:class:`Candidate`) and ``params`` is a JSON-serialisable dict (so it
can participate in the cache key).  Use :func:`as_metric_spec` to wrap
the function into a ``MetricSpec`` that the registry can keep.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Protocol


LOGGER = logging.getLogger("protein_chisel.scoring.metrics")


# ----------------------------------------------------------------------
# Candidate kinds
# ----------------------------------------------------------------------
# A candidate is the thing a metric scores. We support 4 input kinds.
# Most metrics take a structure or a (sequence, structure) pair; a few
# (PLM marginals, sequence-level filters) take a sequence only.
# ----------------------------------------------------------------------

CandidateKind = Literal["seq", "structure", "structure+ligand", "seq+structure"]


@dataclass(frozen=True)
class Candidate:
    """The thing a metric scores.

    Exactly one of :attr:`sequence` / :attr:`structure_path` is required.
    Some metrics need both; some need a ligand-bound structure.
    """
    candidate_id: str                          # stable identifier (used as cache key)
    sequence: Optional[str] = None             # 1-letter, length L
    structure_path: Optional[Path] = None      # PDB file
    ligand_params_path: Optional[Path] = None  # Rosetta .params for the ligand
    catalytic_resnos: tuple[int, ...] = ()     # resseq, sorted; flagged in metric output
    parent_design_id: Optional[str] = None     # provenance


# ----------------------------------------------------------------------
# MetricResult
# ----------------------------------------------------------------------


@dataclass
class MetricResult:
    """The output of a single metric call.

    Attributes:
        metric_name: the registry name of the producing metric.
        values: scalar key/value pairs to be flat-merged into the
            per-candidate metric table. Keys should be prefixed with the
            metric name (e.g. ``fa_dun__mean``); use :meth:`from_to_dict`
            on existing tool result objects for free compliance.
        per_residue: optional per-residue DataFrame for downstream
            inspection (mean(fa_dun) is a scalar; the per-resno
            breakdown lives here). Not cached -- only re-created at
            evaluate time.
        runtime_seconds: wall-clock cost of the call (filled by the
            evaluator wrapper, not by the metric itself).
        provenance: free-form dict for sif image, model weights path,
            pinned-commit hashes etc. Persisted with the cache entry.
        error: filled when the metric failed soft (e.g. ProLIF on a
            non-canonical ligand). The ``values`` dict will contain
            ``<metric>__failed = 1`` and downstream filters should treat
            those rows as no-info, not as outliers.
    """
    metric_name: str
    values: dict[str, float | int | bool | str] = field(default_factory=dict)
    per_residue: object = None             # pd.DataFrame | None; lazy-typed for envs without pandas
    runtime_seconds: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def is_failed(self) -> bool:
        return self.error is not None


# ----------------------------------------------------------------------
# MetricSpec + Registry
# ----------------------------------------------------------------------


class MetricFn(Protocol):
    """Callable signature every metric must satisfy.

    Note: ``params`` is a JSON-serialisable dict so it can be hashed for
    the cache key. Metrics that need richer config (e.g. RotamerScoreConfig)
    should accept a dict and rebuild the dataclass internally.
    """
    def __call__(self, candidate: Candidate, params: dict) -> MetricResult: ...


@dataclass
class MetricSpec:
    """Static description of a metric.

    Used by the scheduler to decide what to call, in what order, with
    what parameters, and at what cost.

    Attributes:
        name: stable registry key. Convention: ``<tool>__<variant>``,
            e.g. ``rotamer_score`` (the canonical fa_dun aggregate) or
            ``pippack_score__chi_only`` (a variant that only emits the
            chi NLL without running assess_packing).
        fn: the actual scorer.
        kind: which Candidate fields the metric consumes -- the
            evaluator validates this so we don't silently feed a
            sequence-only metric into a structure-only metric.
        cost_seconds: nominal wall-clock per candidate, used to order
            tiers and to decide batching strategy. Conservative (real
            wall-clock can be higher under contention).
        needs_gpu: whether the metric requires a GPU. The evaluator
            uses this + ``cost_seconds`` to batch GPU calls (one sif
            launch per tier rather than per candidate).
        default_params: invoked if the caller doesn't override.
        prefix: namespace under which ``values`` keys live; used for
            metric-table column naming. Defaults to ``name + "__"``.
        description: human-readable one-liner; surfaced in verbose logs.
        soft_fail: if True, exceptions in ``fn`` produce a MetricResult
            with ``error=...`` rather than aborting the whole tier.
        cache_version: per-metric integer. Bumping it busts cache
            entries for THIS metric only -- updating one tool's
            output schema doesn't invalidate the other 9. Folded into
            the cache key alongside the global SCHEMA_VERSION so a
            metric author can opt-in to fine-grained invalidation.
        cache_provenance: optional callable returning a JSON-serialisable
            dict whose contents contribute to the cache key. Use it to
            include weights-file mtime/size/hash, container-image
            digest, etc -- anything that should bust the cache when it
            changes underneath. The callable runs ONCE per process at
            registration time-ish (lazily, on first hash) and is
            cached. Default: no extra provenance.
    """
    name: str
    fn: MetricFn
    kind: CandidateKind
    cost_seconds: float
    needs_gpu: bool = False
    default_params: dict = field(default_factory=dict)
    prefix: str = ""
    description: str = ""
    soft_fail: bool = True
    cache_version: int = 1
    cache_provenance: Optional[Callable[[], dict]] = None

    def __post_init__(self) -> None:
        if not self.prefix:
            object.__setattr__(self, "prefix", f"{self.name}__")

    def with_params(self, **overrides) -> "MetricSpec":
        """Return a copy with merged params (immutable)."""
        merged = dict(self.default_params)
        merged.update(overrides)
        return replace(self, default_params=merged)

    def resolved_cache_provenance(self) -> dict:
        """Resolve the (cached) cache_provenance dict for this metric.

        Computed once and memoised on the instance. If the callable
        raises, returns ``{}`` -- we never want a metric to be unreachable
        because its provenance probe failed.
        """
        if self.cache_provenance is None:
            return {}
        cached = getattr(self, "_resolved_cache_provenance_cache", None)
        if cached is not None:
            return cached
        try:
            d = dict(self.cache_provenance() or {})
        except Exception as exc:  # pragma: no cover -- defensive
            LOGGER.warning(
                "cache_provenance() raised for metric %s: %s -- using {}",
                self.name, exc,
            )
            d = {}
        # Cache on the instance even though MetricSpec is a dataclass --
        # we use object.__setattr__ to dodge the (frozen=False) default
        # but leave the public field unmutated.
        object.__setattr__(self, "_resolved_cache_provenance_cache", d)
        return d


class MetricRegistry:
    """Process-local registry of named metrics.

    Pipelines look up metrics by name; tools register on import (or
    via explicit calls in a config-loading entry point). The registry
    is intentionally simple: no scoping, no autoload-on-attribute,
    just a dict + lookup helpers.

    Pattern::

        registry = MetricRegistry()
        registry.register(MetricSpec(name="fa_dun", ...))
        plan = TierPlan(tiers=[[registry["fa_dun"]], [registry["pippack_score"]]])
    """

    def __init__(self) -> None:
        self._specs: dict[str, MetricSpec] = {}

    def register(self, spec: MetricSpec) -> MetricSpec:
        if spec.name in self._specs:
            raise ValueError(f"metric {spec.name!r} already registered")
        # Prefix-collision check: rejecting prefix string-prefix relations
        # so column "foo__failed" can't accidentally come from two
        # different metrics. Equality already covered by name check.
        for existing in self._specs.values():
            if spec.prefix == existing.prefix:
                raise ValueError(
                    f"metric {spec.name!r} has prefix {spec.prefix!r} which "
                    f"already used by {existing.name!r}"
                )
            if spec.prefix.startswith(existing.prefix) or existing.prefix.startswith(spec.prefix):
                raise ValueError(
                    f"metric {spec.name!r} has prefix {spec.prefix!r} which "
                    f"is a string-prefix of (or contains) existing prefix "
                    f"{existing.prefix!r} from {existing.name!r}; this "
                    "would cause column-name collisions in the metric table"
                )
        self._specs[spec.name] = spec
        return spec

    def __getitem__(self, name: str) -> MetricSpec:
        if name not in self._specs:
            available = ", ".join(sorted(self._specs))
            raise KeyError(
                f"metric {name!r} not registered; available: {available}"
            )
        return self._specs[name]

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> list[str]:
        return sorted(self._specs)

    def all_specs(self) -> list[MetricSpec]:
        return list(self._specs.values())

    def __iter__(self):
        return iter(self._specs.values())


# ----------------------------------------------------------------------
# Built-in registry singleton
# ----------------------------------------------------------------------
# Tools register themselves into this registry. Use :func:`get_registry`
# rather than reaching in directly so tests can inject a fresh registry.

_GLOBAL_REGISTRY = MetricRegistry()


def get_registry() -> MetricRegistry:
    return _GLOBAL_REGISTRY


def register(spec: MetricSpec) -> MetricSpec:
    """Convenience -- register into the global registry."""
    return _GLOBAL_REGISTRY.register(spec)


# ----------------------------------------------------------------------
# Adapter: wrap an existing tool's result.to_dict() into a MetricResult
# ----------------------------------------------------------------------


def from_tool_result(
    metric_name: str,
    result_obj: Any,
    *,
    prefix: Optional[str] = None,
    per_residue: Any = None,
    provenance: Optional[dict] = None,
    error: Optional[str] = None,
) -> MetricResult:
    """Wrap a tool's existing ``result_obj.to_dict(prefix=...)`` into a
    MetricResult.

    Existing tools (rotamer_score, rotalyze_score, faspr_pack, ...)
    already expose ``to_dict(prefix='X__')``. This adapter lets us drop
    them into the new pipeline without modifying the tool modules.

    Args:
        metric_name: registry name to attach.
        result_obj: the tool's result dataclass. Must implement
            ``.to_dict(prefix=...)``.
        prefix: forwarded to ``to_dict``; defaults to ``metric_name + "__"``.
        per_residue: per-residue DataFrame if the tool exposes one.
        provenance: extra provenance fields.
        error: soft-failure message.
    """
    pref = prefix if prefix is not None else f"{metric_name}__"
    if error is not None:
        # Soft-fail: emit a sentinel + nothing else, so the metric table
        # has a column tracking failure.
        values = {f"{pref}failed": 1}
    else:
        values = dict(result_obj.to_dict(prefix=pref))
        values[f"{pref}failed"] = 0
    return MetricResult(
        metric_name=metric_name,
        values=values,
        per_residue=per_residue,
        provenance=provenance or {},
        error=error,
    )


# ----------------------------------------------------------------------
# Hashing helpers (used by the cache and provenance)
# ----------------------------------------------------------------------


# Hash truncation length: 16 hex chars = 64 bits. Birthday-collision-safe
# at ~4 billion unique inputs (2^32) which is far beyond our scale (10k
# candidates x 10 metrics x 1000 iterations = 100M lookups but only ~1M
# unique sequences). Bump to 32 chars (128 bits) if you go to billions.
_HASH_LEN = 16


def hash_sequence(sequence: str) -> str:
    """SHA-256 of the upper-case 1-letter sequence; first 16 hex chars."""
    return hashlib.sha256(sequence.upper().encode()).hexdigest()[:_HASH_LEN]


def hash_structure(pdb_path: Path, *, include_hetatm: bool = False) -> str:
    """SHA-256 of canonicalised ATOM (and optionally HETATM) records.

    Hashes columns 0-80 of each ATOM/HETATM record so PDB v3.30
    element symbol (cols 76-77) and formal charge (cols 78-79) are
    included -- different element annotations on the same coordinates
    must produce different hashes.

    REMARK / HEADER / CONECT / TER / END / etc are intentionally not
    hashed; cosmetic edits that don't touch atomic records won't bust
    the cache. By default HETATM is also dropped for cleanliness; pass
    ``include_hetatm=True`` for ligand-aware metrics (or use
    :func:`hash_structure_full`).
    """
    h = hashlib.sha256()
    with open(pdb_path, "rb") as fh:
        for raw in fh:
            if raw.startswith(b"ATOM  ") or (include_hetatm and raw.startswith(b"HETATM")):
                # cols 0-80 covers element + charge in PDB v3.30+.
                h.update(raw[:80])
                h.update(b"\n")
    return h.hexdigest()[:_HASH_LEN]


def hash_structure_full(pdb_path: Path) -> str:
    """Full-content hash (every byte). Use when HETATM / CONECT matter."""
    h = hashlib.sha256()
    with open(pdb_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:_HASH_LEN]


def hash_file(path: Path) -> str:
    """Stable hash of an arbitrary file's bytes (e.g. ligand .params)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:_HASH_LEN]


# Types we accept as JSON-canonicalisable for params hashing.
_JSON_ALLOWED = (str, int, float, bool, type(None))


def _validate_json_native(obj, path: str = "params") -> None:
    """Reject non-JSON-native values to prevent cache key collisions.

    ``json.dumps(..., default=str)`` would happily stringify
    ``Path("/tmp/x")``, ``np.float64(0.5)``, dataclasses, or sets --
    making semantically distinct values collide on hash. We forbid
    them at hash time so callers must explicitly canonicalise.
    """
    if isinstance(obj, _JSON_ALLOWED):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"{path}: dict key {k!r} is not a string"
                )
            _validate_json_native(v, f"{path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _validate_json_native(v, f"{path}[{i}]")
        return
    raise TypeError(
        f"{path}: value {obj!r} of type {type(obj).__name__} is not "
        "JSON-native; convert Path -> str, np scalars -> float/int, "
        "dataclasses -> dict before passing to hash_params"
    )


def hash_params(params: dict) -> str:
    """SHA-256 of a JSON-canonicalized params dict; first 16 hex chars.

    Sorted keys + separators so semantically-equivalent params
    (different key orderings) map to the same hash. Raises ``TypeError``
    on non-JSON-native values to prevent collisions like
    ``Path("/x") == "/x"``.
    """
    if not params:
        return "noparams"
    _validate_json_native(params)
    canon = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:_HASH_LEN]


# ----------------------------------------------------------------------
# Evaluator: time + soft-fail wrapper around a single metric call
# ----------------------------------------------------------------------


def call_metric(
    spec: MetricSpec,
    candidate: Candidate,
    params: Optional[dict] = None,
) -> MetricResult:
    """Invoke ``spec.fn`` with timing + (optional) soft-fail handling.

    The evaluator uses this; tests can call it directly. Does NOT
    consult the cache (the cache layer wraps this).

    Every result -- success or failure -- gets a ``<prefix>failed`` flag
    (0 / 1) so downstream filters can do ``df[df['<prefix>failed']==0]``
    without worrying about which metric returned which set of columns.
    """
    use_params = dict(spec.default_params)
    if params:
        use_params.update(params)
    t0 = time.perf_counter()
    try:
        res = spec.fn(candidate, use_params)
    except Exception as exc:
        if not spec.soft_fail:
            raise
        LOGGER.warning(
            "metric %s soft-failed on %s: %s",
            spec.name, candidate.candidate_id, exc,
        )
        res = MetricResult(
            metric_name=spec.name,
            values={f"{spec.prefix}failed": 1},
            error=f"{type(exc).__name__}: {exc}",
        )
    res.runtime_seconds = time.perf_counter() - t0
    # Validate every emitted column key starts with the metric's prefix.
    # If a metric implementation typo'd its prefix (or copy-pasted from
    # another metric), columns could silently leak into another metric's
    # namespace. Treat as a soft-failure with a clear error message.
    if not res.is_failed():
        bad = [k for k in res.values if not k.startswith(spec.prefix)]
        if bad:
            LOGGER.error(
                "metric %s emitted columns without the %s prefix: %s -- "
                "treating as soft-failure to prevent column collisions",
                spec.name, spec.prefix, bad,
            )
            res = MetricResult(
                metric_name=spec.name,
                values={f"{spec.prefix}failed": 1},
                error=f"prefix violation: emitted {bad!r} outside {spec.prefix!r}",
                runtime_seconds=res.runtime_seconds,
            )
    # Universal failed flag. We override only on hard fail (error set);
    # on success we ensure the flag exists with value 0.
    failed_key = f"{spec.prefix}failed"
    if res.is_failed():
        res.values[failed_key] = 1
    else:
        res.values.setdefault(failed_key, 0)
    return res


__all__ = [
    "Candidate",
    "CandidateKind",
    "MetricFn",
    "MetricRegistry",
    "MetricResult",
    "MetricSpec",
    "call_metric",
    "from_tool_result",
    "get_registry",
    "hash_params",
    "hash_sequence",
    "hash_structure",
    "hash_structure_full",
    "register",
]
