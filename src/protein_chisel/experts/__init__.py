"""Pluggable per-position design *experts* for protein_chisel.

An **expert** produces a calibrated ``(L, 20)`` per-position amino-acid
log-probability array for a scaffold, which the static fusion
(``sampling/plm_fusion.fuse_experts``) combines into the LigandMPNN bias. This
package makes the fusion *plug-and-play*: add a new model by implementing
:class:`Expert` and registering it in :mod:`registry`, then select it by name
(``--experts esmc,saprot,hermes``) — no edits to the fusion math.

The registry + multi-expert design is borrowed from Sebastian (sebols) and
Joe Mi's ``fused_mpnn_poe`` (decode-time product-of-experts); here the experts
feed our *static, calibrated* fusion rather than a decode-time PoE.

Default experts (``esmc``, ``saprot``) reproduce today's pipeline exactly.
"""
from __future__ import annotations

from protein_chisel.experts.base import Expert, ExpertContext
from protein_chisel.experts.registry import (
    available_experts,
    get_expert,
    resolve_experts,
)

__all__ = [
    "Expert",
    "ExpertContext",
    "get_expert",
    "resolve_experts",
    "available_experts",
]
