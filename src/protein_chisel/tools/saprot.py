"""SaProt wrapper: per-position logits + naturalness scoring.

SaProt's input is a "SA token" string: per residue we concatenate the
amino-acid letter (uppercase) and the foldseek 3Di letter (lowercase),
e.g. ``MdQpIv...``. Foldseek extracts 3Di tokens from a PDB; the helper
``saprot_utils.sa_tokens_from_pdb`` (kept compat with our existing
~/special_scripts/ESM/saprot_utils.py) builds the token string.

Two functions parallel the ESM-C ones:

- ``saprot_logits(pdb_path, chain, model_name)`` — per-position log-probs
  over the 20-AA marginal, computed by summing across the 21 3Di columns
  per AA (so the output is a fair (L, 20) table compatible with our
  fusion code).
- ``saprot_score(pdb_path, chain, model_name)`` — pseudo-perplexity.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


SAPROT_3DI_ALPHABET = "acdefghiklmnpqrstvwy"  # 20 lowercase 3Di letters
SAPROT_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# 3Di extraction (foldseek)
# ---------------------------------------------------------------------------


def run_foldseek_3di(
    pdb_path: str | Path,
    chain: Optional[str] = None,
    foldseek_bin: str = None,
) -> tuple[str, str]:
    """Return ``(aa_seq, di_seq)`` for the requested chain (or first chain).

    Uses ``foldseek structureto3didescriptor``. The bundled binary inside
    esmc.sif is at ``/usr/local/bin/foldseek``; on other systems the
    caller can pass `foldseek_bin`.
    """
    bin_path = foldseek_bin or os.environ.get("FOLDSEEK_BIN", "foldseek")
    pdb_abs = str(Path(pdb_path).resolve())
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out.tsv")
        subprocess.run(
            [bin_path, "structureto3didescriptor", pdb_abs, out],
            check=True,
            capture_output=True,
        )
        with open(out) as fh:
            rows = [line.rstrip("\n").split("\t") for line in fh if line.strip()]
    if not rows:
        raise RuntimeError(f"foldseek produced no output for {pdb_abs}")
    if chain is not None:
        rows = [r for r in rows if r[0].endswith(f"_{chain}")]
        if not rows:
            raise ValueError(f"chain {chain!r} not found in {pdb_abs}")
    aa_seq = rows[0][1]
    di_seq = rows[0][2]
    if len(aa_seq) != len(di_seq):
        raise RuntimeError(
            f"AA/3Di length mismatch for {pdb_abs}: {len(aa_seq)} vs {len(di_seq)}"
        )
    return aa_seq, di_seq


def make_sa_tokens(aa_seq: str, di_seq: str) -> str:
    """Build the SaProt SA-token string by interleaving AA + lowercased 3Di."""
    if len(aa_seq) != len(di_seq):
        raise ValueError("aa_seq and di_seq must have the same length")
    out: list[str] = []
    for aa, di in zip(aa_seq, di_seq):
        di_lc = di.lower()
        if di_lc not in SAPROT_3DI_ALPHABET:
            di_lc = "#"  # 'no structure' token
        out.append(f"{aa}{di_lc}")
    return "".join(out)


def sa_tokens_from_pdb(pdb_path: str | Path, chain: Optional[str] = None) -> str:
    aa, di = run_foldseek_3di(pdb_path, chain=chain)
    return make_sa_tokens(aa, di)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


SAPROT_MODELS = {
    "saprot_35m": "westlake-repl/SaProt_35M_AF2",
    "saprot_650m": "westlake-repl/SaProt_650M_PDB",
    "saprot_1.3b": "westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI",
}


def _load_saprot(model_name: str = "saprot_35m", device: str = "auto"):
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    repo = SAPROT_MODELS.get(model_name, model_name)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = EsmForMaskedLM.from_pretrained(repo, torch_dtype=torch.float32).to(device).eval()
    return tokenizer, model, device


# ---------------------------------------------------------------------------
# Logits
# ---------------------------------------------------------------------------


@dataclass
class SaProtLogitsResult:
    sa_string: str
    log_probs: np.ndarray   # shape (L, 20) — per-position log-probs over 20 AAs
    raw_logits: np.ndarray  # shape (L+2, 446) — full vocabulary
    model_name: str


def saprot_logits(
    pdb_path: str | Path,
    chain: Optional[str] = None,
    model_name: str = "saprot_35m",
    device: str = "auto",
) -> SaProtLogitsResult:
    """Per-position log-probs over the 20-AA marginal.

    SaProt's vocabulary is 20 AAs × 21 3Di tokens (= 420) plus 5 special
    tokens, so the raw logits are over 446 columns. We marginalize: the
    log-prob of an AA at a position is logsumexp of the 21 ``Aa, Ac, ..., A#``
    rows for that AA.
    """
    import torch
    sa_str = sa_tokens_from_pdb(pdb_path, chain=chain)
    return _saprot_logits_from_sa_string(sa_str, model_name=model_name, device=device)


def _saprot_logits_from_sa_string(
    sa_str: str,
    model_name: str = "saprot_35m",
    device: str = "auto",
) -> SaProtLogitsResult:
    import torch
    tokenizer, model, dev = _load_saprot(model_name=model_name, device=device)
    spaced = " ".join(sa_str[i:i + 2] for i in range(0, len(sa_str), 2))
    inputs = tokenizer(spaced, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits.squeeze(0).float().cpu().numpy()  # (L+2, 446)

    # Build map AA -> list of token ids whose vocab token is "Aa", "Ac", ...
    vocab = tokenizer.get_vocab()
    aa_to_ids: dict[str, list[int]] = {}
    for tok, idx in vocab.items():
        if len(tok) != 2:
            continue
        aa, di = tok[0], tok[1]
        if aa not in SAPROT_AA_ALPHABET:
            continue
        # 3Di can be a..t letter or '#' for unknown
        if di not in (SAPROT_3DI_ALPHABET + "#"):
            continue
        aa_to_ids.setdefault(aa, []).append(idx)

    # Body positions are logits[1:-1] (skip BOS/EOS or [CLS]/[SEP])
    body = logits[1:-1]
    L = body.shape[0]
    log_p_aa = np.zeros((L, 20), dtype=np.float64)
    for j, aa in enumerate(SAPROT_AA_ALPHABET):
        ids = aa_to_ids.get(aa, [])
        if not ids:
            log_p_aa[:, j] = -1e9
            continue
        sub = body[:, ids]  # (L, 21ish)
        # Marginalize over 3Di: log sum exp over those columns
        m = sub.max(axis=-1, keepdims=True)
        log_p_aa[:, j] = (np.log(np.exp(sub - m).sum(axis=-1)) + m.squeeze(-1))

    # Normalize across 20 AAs so rows are proper log-probabilities
    m = log_p_aa.max(axis=-1, keepdims=True)
    log_p_aa = log_p_aa - (m + np.log(np.exp(log_p_aa - m).sum(axis=-1, keepdims=True)))

    return SaProtLogitsResult(
        sa_string=sa_str,
        log_probs=log_p_aa,
        raw_logits=logits,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Score (pseudo-perplexity)
# ---------------------------------------------------------------------------


@dataclass
class SaProtScoreResult:
    sa_string: str
    pseudo_perplexity: float
    per_position_loglik: np.ndarray
    model_name: str

    def to_dict(self, prefix: str = "saprot__") -> dict[str, float]:
        return {
            f"{prefix}pseudo_perplexity": self.pseudo_perplexity,
            f"{prefix}mean_loglik": float(self.per_position_loglik.mean()),
            f"{prefix}min_loglik": float(self.per_position_loglik.min()),
        }


def saprot_score(
    pdb_path: str | Path,
    chain: Optional[str] = None,
    model_name: str = "saprot_35m",
    device: str = "auto",
    batch_size: int = 32,
) -> SaProtScoreResult:
    """Pseudo-perplexity over a structured sequence.

    Mask each AA position (replacing the AA letter only — 3Di unchanged)
    and compute log p(true_aa | masked_seq). pseudo_perplexity =
    exp(-mean(log p)).
    """
    import torch

    sa_str = sa_tokens_from_pdb(pdb_path, chain=chain)
    if len(sa_str) % 2 != 0:
        raise ValueError(
            f"SaProt SA-token string has odd length {len(sa_str)}; expected "
            "even (each residue contributes one AA letter + one 3Di letter)"
        )
    tokenizer, model, dev = _load_saprot(model_name=model_name, device=device)

    L = len(sa_str) // 2
    aa_seq = sa_str[::2]
    di_seq = sa_str[1::2]

    # The mask token in SaProt is '<mask>'; inputs are tokenized as space-separated
    # SA tokens. To "mask AA only", replace the AA letter with '#' and use the
    # corresponding "#<3di>" token if it exists, OR replace the whole SA token
    # with the mask token.
    # Simplest: replace the whole SA token with <mask>, accept slight loss of
    # 3Di info at that one position.
    mask_token = tokenizer.mask_token  # "<mask>"

    sa_tokens = [sa_str[i:i + 2] for i in range(0, len(sa_str), 2)]
    base_ids = tokenizer(" ".join(sa_tokens), return_tensors="pt").input_ids[0].tolist()
    # base_ids has prefix CLS + L tokens + SEP

    per_pos_loglik = np.zeros(L, dtype=np.float64)
    aa_to_ids = _build_saprot_aa_token_map(tokenizer)

    with torch.no_grad():
        for start in range(0, L, batch_size):
            end = min(start + batch_size, L)
            batch_ids = []
            for i in range(start, end):
                # Build a copy with position i replaced by mask
                tokens_i = list(sa_tokens)
                tokens_i[i] = mask_token
                ids = tokenizer(" ".join(tokens_i), return_tensors="pt").input_ids[0]
                batch_ids.append(ids)
            # All same length so we can stack
            stacked = torch.stack(batch_ids, dim=0).to(dev)
            out = model(input_ids=stacked).logits  # (B, L+2, 446)
            for k, i in enumerate(range(start, end)):
                row = out[k, i + 1].float().cpu().numpy()  # logits at masked pos
                # Marginalize over 3Di for the true AA
                true_aa = aa_seq[i]
                ids = aa_to_ids.get(true_aa, [])
                if not ids:
                    per_pos_loglik[i] = float("nan")
                    continue
                m = row.max()
                log_z = m + np.log(np.exp(row - m).sum())
                m2 = row[ids].max()
                log_aa = m2 + np.log(np.exp(row[ids] - m2).sum())
                per_pos_loglik[i] = float(log_aa - log_z)

    valid = ~np.isnan(per_pos_loglik)
    pseudo_perp = float(np.exp(-per_pos_loglik[valid].mean()))
    return SaProtScoreResult(
        sa_string=sa_str,
        pseudo_perplexity=pseudo_perp,
        per_position_loglik=per_pos_loglik,
        model_name=model_name,
    )


def _build_saprot_aa_token_map(tokenizer) -> dict[str, list[int]]:
    """AA -> list of token ids for that AA across all 3Di states."""
    out: dict[str, list[int]] = {}
    for tok, idx in tokenizer.get_vocab().items():
        if len(tok) != 2:
            continue
        aa, di = tok[0], tok[1]
        if aa not in SAPROT_AA_ALPHABET:
            continue
        if di not in (SAPROT_3DI_ALPHABET + "#"):
            continue
        out.setdefault(aa, []).append(idx)
    return out


__all__ = [
    "SAPROT_AA_ALPHABET",
    "SAPROT_3DI_ALPHABET",
    "SAPROT_MODELS",
    "SaProtLogitsResult",
    "SaProtScoreResult",
    "make_sa_tokens",
    "run_foldseek_3di",
    "sa_tokens_from_pdb",
    "saprot_logits",
    "saprot_score",
]
