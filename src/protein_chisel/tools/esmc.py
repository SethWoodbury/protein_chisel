"""ESM-C wrapper: per-position logits + pseudo-perplexity scoring.

Two complementary functions:

- ``esmc_logits(seq, model_name)`` — per-position log-probabilities over the
  ESM-C vocabulary (64 tokens including special tokens). Returns a dense
  numpy array of shape (L+2, 64): rows correspond to sequence positions
  + 2 special tokens (BOS/EOS or similar). Position log-probs are taken
  as model.logits.sequence (the raw prediction from a forward pass with
  no masking).

- ``esmc_score(seq)`` — pseudo-perplexity, computed by masking each
  position one at a time. Returns a scalar (lower is more "natural"
  per the model).

These wrap the evolutionaryscale `esm` package's ESMC model. Run inside
esmc.sif. GPU recommended for sequences > a few hundred residues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# Standard 20 AAs in ESM-C / ESM-2 ordering.
ESM_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class ESMCLogitsResult:
    sequence: str
    log_probs: np.ndarray  # shape (L, 20) — log p over the 20 AAs at each pos
    raw_logits: np.ndarray  # shape (L+2, vocab) — full vocabulary logits incl. specials
    aa_token_ids: np.ndarray  # shape (20,) — vocab indices for canonical AAs
    model_name: str


def _get_aa_token_ids(model) -> np.ndarray:
    """Return token ids for the 20 canonical AAs in ESM_AA_ALPHABET order."""
    # The new ESM tokenizer: model has a tokenizer attribute under model.tokenizer
    # Each AA single-letter is its own token. We look them up.
    tok = model.tokenizer
    ids: list[int] = []
    for aa in ESM_AA_ALPHABET:
        # encode returns a list-like with the BOS/EOS plus the token; take the
        # middle one. Different versions of esm may differ — try both.
        toks = tok.encode(aa, add_special_tokens=False)
        if hasattr(toks, "tolist"):
            toks = toks.tolist()
        if isinstance(toks, list):
            ids.append(int(toks[0]))
        else:
            ids.append(int(toks))
    return np.array(ids, dtype=np.int64)


def _load_esmc(model_name: str = "esmc_300m", device: str = "auto"):
    """Load an ESMC model (cached in HF_HOME)."""
    import torch
    from esm.models.esmc import ESMC

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESMC.from_pretrained(model_name).to(device).eval()
    return model, device


def esmc_logits(
    sequence: str,
    model_name: str = "esmc_300m",
    device: str = "auto",
) -> ESMCLogitsResult:
    """Per-position log-probs over the 20-AA alphabet for one sequence.

    Args:
        sequence: 1-letter AA string.
        model_name: "esmc_300m" or "esmc_600m".
        device: "auto" / "cuda" / "cpu".

    Returns ESMCLogitsResult — `log_probs` has shape (L, 20).
    """
    import torch
    from esm.sdk.api import ESMProtein, LogitsConfig

    model, dev = _load_esmc(model_name=model_name, device=device)
    aa_ids = _get_aa_token_ids(model)

    protein = ESMProtein(sequence=sequence)
    p_t = model.encode(protein)
    with torch.no_grad():
        out = model.logits(p_t, LogitsConfig(sequence=True, return_embeddings=False))
    raw = out.logits.sequence.squeeze(0).float().cpu().numpy()  # (L+2, vocab)

    # Skip the first and last special tokens; index AA columns to (L, 20)
    body = raw[1:-1]
    body_aa = body[:, aa_ids]  # (L, 20)
    log_p = body_aa - _logsumexp(body_aa, axis=-1, keepdims=True)
    return ESMCLogitsResult(
        sequence=sequence,
        log_probs=log_p,
        raw_logits=raw,
        aa_token_ids=aa_ids,
        model_name=model_name,
    )


def _logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (np.log(np.exp(x - m).sum(axis=axis, keepdims=keepdims)) + (m if keepdims else m.squeeze(axis)))


@dataclass
class ESMCScoreResult:
    sequence: str
    pseudo_perplexity: float
    per_position_loglik: np.ndarray  # shape (L,)
    model_name: str

    def to_dict(self, prefix: str = "esmc__") -> dict[str, float]:
        return {
            f"{prefix}pseudo_perplexity": self.pseudo_perplexity,
            f"{prefix}mean_loglik": float(self.per_position_loglik.mean()),
            f"{prefix}min_loglik": float(self.per_position_loglik.min()),
        }


def esmc_score(
    sequence: str,
    model_name: str = "esmc_300m",
    device: str = "auto",
    batch_size: int = 32,
) -> ESMCScoreResult:
    """Pseudo-perplexity by masking each position one at a time.

    Pseudo-perplexity = exp(-mean(log p(true_aa | seq_minus_i, mask_at_i))).

    Lower = more "natural" per the model.

    Implementation: build a (L, L+2) batch of masked sequences (one mask
    per position), forward in chunks of batch_size, gather log p(true)
    at each mask position, sum, exp(-mean).
    """
    import torch
    from esm.sdk.api import ESMProtein, LogitsConfig

    model, dev = _load_esmc(model_name=model_name, device=device)
    aa_ids_for_alphabet = _get_aa_token_ids(model)
    # Map AA letter -> token id
    aa_to_id = dict(zip(ESM_AA_ALPHABET, aa_ids_for_alphabet))

    protein = ESMProtein(sequence=sequence)
    base_t = model.encode(protein)
    base_tokens = base_t.sequence  # 1-D tensor of length L+2 with BOS/EOS

    # The mask token id — get it via the model's tokenizer
    tok = model.tokenizer
    mask_id = int(tok.mask_token_id)

    L = len(sequence)
    # Build a batch of L sequences, each with one position masked.
    full = base_tokens.unsqueeze(0).repeat(L, 1).to(dev)  # (L, L+2)
    eye_mask = torch.eye(L, dtype=torch.bool, device=dev)
    # Place masks at body positions (1..L) so column index in `full` is 1..L
    # i.e., full[i, i+1] = mask_id for i in range(L)
    body_idx = torch.arange(1, L + 1, device=dev)
    rows = torch.arange(L, device=dev)
    full[rows, body_idx] = mask_id

    per_pos_loglik = np.zeros(L, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, L, batch_size):
            end = min(start + batch_size, L)
            chunk = full[start:end]
            # Wrap each row as ESMProteinTensor and forward; or use logits directly.
            # The cleanest API: model.forward equivalent — use model.logits via
            # ESMProteinTensor inputs.
            from esm.sdk.api import ESMProteinTensor
            chunk_results: list[np.ndarray] = []
            for r in chunk:
                ept = ESMProteinTensor(sequence=r)
                logits = model.logits(ept, LogitsConfig(sequence=True)).logits.sequence
                chunk_results.append(logits.squeeze(0).float().cpu().numpy())
            for i, raw in enumerate(chunk_results):
                pos = start + i
                # log p over vocab at masked position (pos+1 in raw)
                row = raw[pos + 1]
                row_norm = row - row.max()
                log_p = row_norm - np.log(np.exp(row_norm).sum())
                true_aa = sequence[pos]
                if true_aa in aa_to_id:
                    per_pos_loglik[pos] = float(log_p[aa_to_id[true_aa]])
                else:
                    per_pos_loglik[pos] = float("nan")

    valid = ~np.isnan(per_pos_loglik)
    pseudo_perp = float(np.exp(-per_pos_loglik[valid].mean()))
    return ESMCScoreResult(
        sequence=sequence,
        pseudo_perplexity=pseudo_perp,
        per_position_loglik=per_pos_loglik,
        model_name=model_name,
    )


__all__ = [
    "ESM_AA_ALPHABET",
    "ESMCLogitsResult",
    "ESMCScoreResult",
    "esmc_logits",
    "esmc_score",
]
