"""Run provenance / version control.

Captures which experts (+ model versions), fusion math, sampling backend, and
conserved-interaction-network settings produced a design run, so any output is
traceable to the exact code + model set. Written as ``provenance.json`` in the run
dir (additive metadata) and, optionally, stamped as a ``REMARK PROVENANCE`` line on
the shipped PDBs.

Forward-compatible: fields for the not-yet-default add-ons (PoE backend, PLM
refresh, HERMES) carry inert defaults so a run records them as off.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from protein_chisel import __version__ as CHISEL_VERSION


@dataclass
class RunProvenance:
    """Reproducibility record for one design run."""

    experts: list[str]
    expert_versions: dict[str, str]
    fusion_version: str
    chisel_version: str = CHISEL_VERSION
    # Sampling backend (Phase: PoE). "bias" = our static calibrated fusion.
    mpnn_backend: str = "bias"
    # PLM-bias refresh rounds (Phase: refresh). 0 = static seed bias.
    plm_refresh_rounds: int = 0
    # Conserved-interaction network (add-on #6).
    conserve_hbonds: bool = False
    conserve_depth: int = 1
    conserve_interaction_types: list[str] = field(default_factory=lambda: ["hbond"])
    conserve_grow_network: bool = False
    conserve_seed_base: Optional[int] = None
    # HERMES expert (Phase: HERMES).
    hermes_model_version: Optional[str] = None
    # Metric/filter registry selection (add-on #7). "all" = today's full set;
    # active_metrics is the resolved catalog name list actually in effect.
    metrics_selection: str = "all"
    filters_selection: str = "all"
    active_metrics: list[str] = field(default_factory=list)
    # Any extra run-specific tags.
    extra: dict = field(default_factory=dict)

    def to_manifest_dict(self) -> dict:
        """JSON-serializable dict for manifests / provenance.json."""
        return asdict(self)

    def to_remark_line(self) -> str:
        """Single ``REMARK PROVENANCE <json>`` line (newline-terminated).

        Compact, sorted keys so it's stable/greppable. Parsed into the ``_misc``
        REMARK group by ``tools.remarks`` (preserved through reorg)."""
        body = json.dumps(self.to_manifest_dict(), sort_keys=True,
                          separators=(",", ":"))
        return f"REMARK PROVENANCE {body}\n"

    def write_json(self, path) -> None:
        """Write ``provenance.json`` to ``path``."""
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_manifest_dict(), indent=2))
