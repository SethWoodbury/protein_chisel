"""SaProt structure-aware expert — wrapper over ``tools.saprot.saprot_logits``.

Structure-based masked-LM marginals (consumes the PDB + chain via foldseek 3Di
tokens). Heavy deps imported lazily inside :meth:`compute_log_probs`.
"""
from __future__ import annotations

import numpy as np

from protein_chisel.experts.base import Expert, ExpertContext


class SaProtExpert(Expert):
    name = "saprot"
    modality = "structure"

    def __init__(self, model_name: str = "saprot_35m"):
        self.model_name = model_name

    @property
    def version(self) -> str:
        return f"saprot:{self.model_name}"

    def compute_log_probs(self, ctx: ExpertContext) -> np.ndarray:
        from protein_chisel.tools.saprot import saprot_logits

        return saprot_logits(
            ctx.pdb_path, chain=ctx.chain, model_name=self.model_name,
            device=ctx.device, masked=True,
        ).log_probs
