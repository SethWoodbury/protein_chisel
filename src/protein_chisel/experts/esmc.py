"""ESM-C sequence expert — thin wrapper over ``tools.esmc.esmc_logits``.

Sequence-based masked-LM marginals. Heavy deps (torch/esm) are imported lazily
inside :meth:`compute_log_probs` so importing the registry stays cheap.
"""
from __future__ import annotations

import numpy as np

from protein_chisel.experts.base import Expert, ExpertContext


class ESMCExpert(Expert):
    name = "esmc"
    modality = "sequence"

    def __init__(self, model_name: str = "esmc_300m"):
        self.model_name = model_name

    @property
    def version(self) -> str:
        return f"esmc:{self.model_name}"

    def compute_log_probs(self, ctx: ExpertContext) -> np.ndarray:
        # Lazy import: only pay the torch/esm import when actually running ESM-C.
        from protein_chisel.tools.esmc import esmc_logits

        return esmc_logits(
            ctx.seq, model_name=self.model_name, device=ctx.device, masked=True,
            dtype=ctx.plm_dtype,
        ).log_probs
