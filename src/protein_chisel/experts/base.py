"""Expert interface for pluggable per-position design priors.

Each :class:`Expert` computes a ``(L, 20)`` masked-LM-style log-probability array
(AA order = ``plm_fusion.AA_ORDER``) for one scaffold. Experts differ in
*modality*: sequence-based (ESM-C consumes the sequence), structure-based (SaProt
consumes the PDB+chain; HERMES consumes the local atomic environment). The
:class:`ExpertContext` carries everything any expert might need so the registry
can treat them uniformly.

Design borrowed from Sebastian (sebols) / Joe Mi's ``fused_mpnn_poe`` expert
abstraction; adapted to emit a static calibrated ``(L,20)`` for our fusion.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ExpertContext:
    """Inputs available to every expert for one scaffold.

    Attributes:
        seq: One-letter protein sequence of the designed chain (length L).
        pdb_path: Path to the scaffold PDB (structure-based experts).
        chain: Chain id of the designed protein (default "A").
        device: "auto" | "cpu" | "gpu" — passed through to the model loader.
        out_dir: Optional directory for per-expert ``.npy`` caching; when set, an
            expert may read/write ``<out_dir>/<cache_filename>`` to skip recompute.
        plm_dtype: model inference precision — "fp32" (default; byte-identical) |
            "fp16" | "bf16". Half precision (opt-in) roughly halves PLM memory but
            changes the logits, so it is keyed into the cache filename (see
            :meth:`Expert.cache_filename_for`) — an fp16 run never reuses an fp32
            cache or vice-versa.
    """
    seq: str
    pdb_path: str | Path
    chain: str = "A"
    device: str = "auto"
    out_dir: Optional[Path] = None
    plm_dtype: str = "fp32"


class Expert(abc.ABC):
    """A per-position amino-acid log-probability provider.

    Subclasses set :attr:`name` (registry key + cache prefix), :attr:`modality`
    ("sequence" | "structure"), and implement :meth:`compute_log_probs`.
    """

    #: Short registry key, also the cache-file prefix (``<name>_log_probs.npy``).
    name: str = "expert"
    #: "sequence" | "structure" — documents what the expert conditions on.
    modality: str = "sequence"

    @property
    def version(self) -> str:
        """Version string recorded in run provenance (model id + variant)."""
        return self.name

    @property
    def cache_filename(self) -> str:
        """The legacy (fp32) per-artifact cache filename. Kept byte-for-byte for the
        default experts (precompute's manifest/outputs reference this name)."""
        return f"{self.name}_log_probs.npy"

    def cache_filename_for(self, plm_dtype: str = "fp32") -> str:
        """Dtype-aware cache filename. ``fp32`` returns the legacy
        ``<name>_log_probs.npy`` (byte-identical default); any other dtype returns
        ``<name>_log_probs.<dtype>.npy`` so half-precision artifacts never collide
        with — or get silently reused as — the fp32 ones."""
        if plm_dtype == "fp32":
            return self.cache_filename
        return f"{self.name}_log_probs.{plm_dtype}.npy"

    @abc.abstractmethod
    def compute_log_probs(self, ctx: ExpertContext) -> np.ndarray:
        """Return the raw ``(L, 20)`` masked-LM log-probabilities for ``ctx``."""

    def log_probs(self, ctx: ExpertContext, *, use_cache: bool = True) -> np.ndarray:
        """Cached wrapper around :meth:`compute_log_probs`.

        When ``ctx.out_dir`` is set and ``use_cache`` is True, read/write
        ``<out_dir>/<cache_filename>`` so re-runs are free. Models are loaded
        inside :meth:`compute_log_probs` and freed when it returns, so experts are
        computed one at a time (memory discipline — never hold several resident).
        """
        import logging

        logger = logging.getLogger(f"protein_chisel.experts.{self.name}")
        cache = None
        if use_cache and ctx.out_dir is not None:
            cache = Path(ctx.out_dir) / self.cache_filename_for(ctx.plm_dtype)
            if cache.exists():
                logger.info("%s cache hit -> %s", self.name, cache)
                return np.load(cache)
        lp = self.compute_log_probs(ctx)
        if lp.ndim != 2 or lp.shape[1] != 20:
            raise ValueError(
                f"{self.name}.compute_log_probs returned shape {lp.shape}, "
                "expected (L, 20)"
            )
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, lp)
            logger.info("%s -> %s shape=%s", self.name, cache, lp.shape)
        return lp
