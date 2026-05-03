"""Persistent metric cache.

Antipattern #1 from the architectural review: don't re-run cheap metrics
every MH step. The cache wraps a metric call with a content-addressed
key so the same (candidate, metric, params) triple resolves on the
second hit instead of re-paying the runtime.

Key derivation
==============
The cache key is a 4-tuple::

    (input_hash, metric_name, params_hash, schema_version)

- ``input_hash`` is sequence- or structure-derived depending on what
  the metric consumes. The metric's ``MetricSpec.kind`` tells us which.
- ``metric_name`` namespaces variants (``rotalyze``, ``rotalyze__strict``).
- ``params_hash`` covers the params dict at call time.
- ``schema_version`` lets us bust the entire cache if the MetricResult
  schema changes (e.g. a new key is added that older entries don't have).

Persistence
===========
Two backends:

- :class:`InMemoryCache`  -- transient, for tests and single-process runs.
- :class:`JsonlCache`     -- one entry per line in a single ``.jsonl`` file
                             plus a ``.lock`` directory for cross-process
                             writes; durable across pipeline restarts.

The default cache for production pipelines is JsonlCache, located at
``<pipeline_out_dir>/metric_cache.jsonl`` so each pipeline run has its
own (sharable) cache. Multiple chains writing concurrently use a
mkdir-based lock; the operations are append-only so concurrent reads
during writes still see a consistent prefix.

Soft failures (``MetricResult.error is not None``) are cached with
``cache_failures=True`` (the default) so a crashing metric doesn't
get re-tried on every MH step. Pass ``cache_failures=False`` if you
want failures to retry.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

from protein_chisel.scoring.metrics import (
    Candidate,
    MetricResult,
    MetricSpec,
    call_metric,
    hash_file,
    hash_params,
    hash_sequence,
    hash_structure,
    hash_structure_full,
)


LOGGER = logging.getLogger("protein_chisel.scoring.cache")


# Bump when the on-disk JSONL schema becomes incompatible.
SCHEMA_VERSION = 1


# ----------------------------------------------------------------------
# Cache key
# ----------------------------------------------------------------------


# Delimiter used inside composite hash strings. Chosen to never appear
# in hex digests or in any of our other hashed components.
_HASH_DELIM = "\x00"


@dataclass(frozen=True)
class CacheKey:
    """Content-addressed identifier for a single metric call.

    Composition: ``{input_hash}|{metric_name}|{params_hash}|m{metric_version}|v{schema_version}``

    ``metric_version`` is the per-spec :attr:`MetricSpec.cache_version`;
    bumping it busts only that metric's cache entries (not the global
    schema). ``schema_version`` covers the on-disk MetricResult schema.
    """
    input_hash: str        # sequence or structure hash
    metric_name: str
    params_hash: str
    metric_version: int = 1
    schema_version: int = SCHEMA_VERSION

    def as_str(self) -> str:
        return (
            f"{self.input_hash}|{self.metric_name}|"
            f"{self.params_hash}|m{self.metric_version}|v{self.schema_version}"
        )


def derive_input_hash(spec: MetricSpec, candidate: Candidate) -> str:
    """Compute the content-addressed input hash for a candidate+metric pair.

    The choice of hash depends on ``spec.kind``:
      - "seq"               -> hash of candidate.sequence
      - "structure"         -> ATOM-only hash of candidate.structure_path,
                               + catalytic_resnos when non-empty
      - "structure+ligand"  -> ATOM+HETATM hash + ligand_params_path content,
                               + catalytic_resnos when non-empty
      - "seq+structure"     -> seq + structure, joined with NUL delimiter

    Hashing is stable across pipeline runs as long as the file contents
    don't change.
    """
    parts: list[str] = []

    if spec.kind == "seq":
        if candidate.sequence is None:
            raise ValueError(
                f"metric {spec.name!r} has kind='seq' but candidate "
                f"{candidate.candidate_id!r} has no sequence"
            )
        parts.append(hash_sequence(candidate.sequence))

    elif spec.kind == "structure":
        if candidate.structure_path is None:
            raise ValueError(
                f"metric {spec.name!r} has kind='structure' but candidate "
                f"{candidate.candidate_id!r} has no structure_path"
            )
        parts.append(hash_structure(candidate.structure_path))

    elif spec.kind == "structure+ligand":
        if candidate.structure_path is None:
            raise ValueError(
                f"metric {spec.name!r} kind='structure+ligand' needs "
                f"structure_path"
            )
        # Include HETATM (ligand atoms) in the structure hash so two
        # protein-identical poses with different ligand placements
        # produce different keys.
        parts.append(hash_structure(candidate.structure_path, include_hetatm=True))
        # And include ligand .params content if provided -- changing the
        # ligand topology (charges, atom types) must bust the cache.
        if candidate.ligand_params_path is not None:
            parts.append(hash_file(Path(candidate.ligand_params_path)))

    elif spec.kind == "seq+structure":
        if candidate.sequence is None or candidate.structure_path is None:
            raise ValueError(
                f"metric {spec.name!r} kind='seq+structure' needs both "
                f"sequence and structure_path"
            )
        parts.append(hash_sequence(candidate.sequence))
        parts.append(hash_structure(candidate.structure_path))

    else:
        raise ValueError(f"unsupported metric kind: {spec.kind!r}")

    # Catalytic resnos are part of the "input" -- a metric that flags
    # catalytic outliers would silently return the wrong rows if we
    # changed the catalytic set without busting the cache.
    if candidate.catalytic_resnos:
        cats = ",".join(str(r) for r in sorted(candidate.catalytic_resnos))
        parts.append(f"cat[{cats}]")

    return _HASH_DELIM.join(parts)


def make_cache_key(
    spec: MetricSpec, candidate: Candidate, params: dict,
) -> CacheKey:
    """Compose a CacheKey for the (spec, candidate, params) triple.

    ``params`` is merged with ``spec.default_params`` and ``spec.resolved_cache_provenance()``
    before hashing, so model-weights mtime / container digest / etc.
    propagate into the key. The metric author is responsible for
    putting these into ``cache_provenance``; the foundation just uses
    them.
    """
    full_params = dict(spec.default_params)
    full_params.update(params)
    # Include the provenance dict (e.g. {"weights_mtime": 1234, "weights_size": 567})
    # under a reserved prefixed key so user params can't accidentally collide.
    prov = spec.resolved_cache_provenance()
    if prov:
        full_params["__cache_provenance__"] = prov
    return CacheKey(
        input_hash=derive_input_hash(spec, candidate),
        metric_name=spec.name,
        params_hash=hash_params(full_params),
        metric_version=spec.cache_version,
        schema_version=SCHEMA_VERSION,
    )


# ----------------------------------------------------------------------
# Cache backends
# ----------------------------------------------------------------------


class MetricCache(Protocol):
    """Cache backend protocol.

    A cache is keyed by :class:`CacheKey` and stores serialised
    :class:`MetricResult` payloads. ``per_residue`` is NOT cached --
    the column-oriented per-residue dataframe can be re-derived on
    demand if a downstream consumer needs it; caching it would balloon
    the on-disk size without adding much value.
    """
    def get(self, key: CacheKey) -> Optional[MetricResult]: ...
    def put(self, key: CacheKey, result: MetricResult) -> None: ...
    def __contains__(self, key: CacheKey) -> bool: ...


class InMemoryCache(MetricCache):
    """Process-local cache. Lost when the process exits."""

    def __init__(self) -> None:
        self._d: dict[str, MetricResult] = {}

    def get(self, key: CacheKey) -> Optional[MetricResult]:
        return self._d.get(key.as_str())

    def put(self, key: CacheKey, result: MetricResult) -> None:
        # Strip per_residue (not part of cache) before storing.
        self._d[key.as_str()] = replace(result, per_residue=None)

    def __contains__(self, key: CacheKey) -> bool:
        return key.as_str() in self._d

    def __len__(self) -> int:
        return len(self._d)


class JsonlCache(MetricCache):
    """Append-only JSONL cache, durable across pipeline restarts.

    On-disk format: one JSON object per line::

        {"key": "<key.as_str()>", "values": {...}, "runtime_seconds": ...,
         "provenance": {...}, "error": null}

    Concurrency model
    -----------------
    JsonlCache is safe for *moderate* multi-process contention (e.g. a
    handful of sbatch workers writing to the same cache). For heavy
    distributed scale we'd want SQLite WAL or LMDB; this layer is the
    pragmatic middle ground.

    - Writes acquire an exclusive ``fcntl.flock`` on a sidecar ``.lock``
      file. The lock is held only during the actual append. fcntl
      locks are auto-released on process death (no stale-lock spinning),
      which is the main upgrade over the mkdir-based lock the early
      version of this class used.

    - On every ``get()`` miss, we ``stat`` the JSONL file and reload
      any new lines appended by sibling processes since our last read.
      This keeps the in-memory mirror coherent across distributed
      writers without forcing a full re-read on every call.

    - If the lock cannot be acquired within ``lock_timeout_s``, the
      write FAILS rather than appending unlocked -- silent data loss
      from interleaved appends is worse than a clean error the caller
      can choose to retry.
    """

    def __init__(
        self, path: str | Path, *, lock_timeout_s: float = 60.0,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lockfile = self.path.with_suffix(self.path.suffix + ".lock")
        self._lock_timeout_s = float(lock_timeout_s)
        self._mirror: dict[str, MetricResult] = {}
        self._mtime_ns: int = -1     # last seen mtime; -1 forces full reload
        self._byte_offset: int = 0   # last byte we read up to
        self._loaded = False

    def _stat_mtime_ns(self) -> int:
        try:
            return self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return -1

    def _load_if_needed(self) -> None:
        """Initial full-file load + tail-reload for sibling-process appends.

        On first call: read the whole file, record mtime + byte offset.
        On every subsequent call: if mtime changed, read from
        ``self._byte_offset`` to EOF and merge into the mirror.
        """
        cur_mtime_ns = self._stat_mtime_ns()
        if self._loaded and cur_mtime_ns == self._mtime_ns:
            return  # mirror is up to date

        if not self.path.is_file():
            self._loaded = True
            self._mtime_ns = cur_mtime_ns
            return

        n_loaded = 0
        n_skipped = 0
        # On full reload, start at 0; on incremental, resume from last offset.
        start = 0 if not self._loaded else self._byte_offset
        with open(self.path, "rb") as fh:
            fh.seek(start)
            for raw in fh:
                line = raw.decode("utf-8", errors="strict").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    n_skipped += 1
                    continue
                key = obj.get("key", "")
                if not key.endswith(f"|v{SCHEMA_VERSION}"):
                    n_skipped += 1  # stale schema
                    continue
                self._mirror[key] = MetricResult(
                    metric_name=obj.get("metric_name", ""),
                    values=obj.get("values", {}),
                    runtime_seconds=float(obj.get("runtime_seconds", 0.0)),
                    provenance=obj.get("provenance", {}),
                    error=obj.get("error"),
                )
                n_loaded += 1
            self._byte_offset = fh.tell()

        self._loaded = True
        self._mtime_ns = cur_mtime_ns
        if n_loaded:
            LOGGER.debug(
                "JsonlCache loaded %d new entries (skipped %d stale) from %s",
                n_loaded, n_skipped, self.path,
            )

    def get(self, key: CacheKey) -> Optional[MetricResult]:
        self._load_if_needed()
        ks = key.as_str()
        if ks in self._mirror:
            return self._mirror[ks]
        # Mirror miss -- but a sibling process may have written since
        # our last reload. Force a tail-reload check before declaring miss.
        cur = self._stat_mtime_ns()
        if cur != self._mtime_ns:
            self._load_if_needed()
            return self._mirror.get(ks)
        return None

    def put(self, key: CacheKey, result: MetricResult) -> None:
        clean = replace(result, per_residue=None)
        # Write under lock first; only update mirror after successful
        # durable append (so a failed write doesn't pollute the mirror
        # with a value disk doesn't have).
        self._with_lock(self._append_line, key, clean)
        self._mirror[key.as_str()] = clean
        # The write changed the file; refresh our offset so we don't
        # re-read our own line on next reload.
        self._mtime_ns = self._stat_mtime_ns()
        try:
            self._byte_offset = self.path.stat().st_size
        except FileNotFoundError:
            pass

    def __contains__(self, key: CacheKey) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        self._load_if_needed()
        return len(self._mirror)

    # -- locking ----------------------------------------------------

    def _with_lock(self, fn, *args, **kw):
        """Acquire an fcntl exclusive lock on the sidecar file.

        Auto-released on process death (kernel-managed); raises
        TimeoutError on contention beyond ``lock_timeout_s``.
        """
        import fcntl
        # Touch the lock file so flock has something to grab.
        self._lockfile.touch()
        deadline = time.time() + self._lock_timeout_s
        with open(self._lockfile, "rb+") as lf:
            while True:
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() >= deadline:
                        raise TimeoutError(
                            f"JsonlCache lock {self._lockfile} held > "
                            f"{self._lock_timeout_s}s -- another writer is "
                            "stuck. Investigate before retrying."
                        )
                    time.sleep(0.05)
            try:
                return fn(*args, **kw)
            finally:
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass

    def _append_line(self, key: CacheKey, result: MetricResult) -> None:
        line = json.dumps({
            "key": key.as_str(),
            "metric_name": result.metric_name,
            "values": _make_jsonable(result.values),
            "runtime_seconds": float(result.runtime_seconds),
            "provenance": _make_jsonable(result.provenance),
            "error": result.error,
        }, separators=(",", ":"), default=str)
        with open(self.path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
            try:
                # fsync best-effort: ensures the line is durable before
                # we release the lock. On NFS this is the difference
                # between "another worker can see this entry" and "I
                # guess it'll show up eventually."
                import os as _os
                _os.fsync(fh.fileno())
            except OSError:
                pass


def _make_jsonable(v):
    """Recursively convert numpy scalars / Path / etc to plain python."""
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except ImportError:
        pass
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {k: _make_jsonable(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_make_jsonable(vv) for vv in v]
    return v


# ----------------------------------------------------------------------
# call_metric_cached: the wrapper pipelines should use
# ----------------------------------------------------------------------


def call_metric_cached(
    spec: MetricSpec,
    candidate: Candidate,
    cache: MetricCache,
    params: Optional[dict] = None,
    *,
    cache_failures: bool = True,
) -> MetricResult:
    """Look up (spec, candidate, params) in ``cache``; on miss, run + store.

    Returns the cached or freshly-computed MetricResult. The returned
    object's ``runtime_seconds`` reflects ORIGINAL compute time (not the
    cache lookup time) when served from cache; this lets aggregate
    "wall-clock spent" reports stay honest about real cost.

    Args:
        spec: registered metric to call.
        candidate: the candidate to score.
        cache: backend to consult / populate.
        params: per-call params; merged on top of ``spec.default_params``.
        cache_failures: when True (default), MetricResult instances with
            ``error != None`` are also cached, so the same broken
            candidate doesn't retry on every step. Set False if you've
            fixed an upstream bug and want to retry stale failures.
    """
    full_params = dict(spec.default_params)
    if params:
        full_params.update(params)
    key = make_cache_key(spec, candidate, full_params)

    cached = cache.get(key)
    if cached is not None:
        LOGGER.debug(
            "cache HIT  %s  %s", spec.name, candidate.candidate_id,
        )
        return cached

    LOGGER.debug("cache MISS %s  %s", spec.name, candidate.candidate_id)
    result = call_metric(spec, candidate, full_params)
    if cache_failures or not result.is_failed():
        cache.put(key, result)
    return result


__all__ = [
    "CacheKey",
    "InMemoryCache",
    "JsonlCache",
    "MetricCache",
    "SCHEMA_VERSION",
    "call_metric_cached",
    "derive_input_hash",
    "make_cache_key",
]
