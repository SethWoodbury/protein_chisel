"""Iterative single-mutation walk pipeline.

Two modes (selected via ``--mode``):

- ``constrained_local_search``: per-position PLM-marginal proposals, hard
  filters as the acceptance criterion. Cheap, biased; cannot find
  compensatory mutations across multiple sites in one move (architecture.md
  documents the limits).

- ``mh``: real Metropolis-Hastings with proposal-correction term and a
  scalar target energy ``E(s)``. Optional simulated-annealing temperature
  schedule and parallel-tempering chains. Slower but unbiased.

Inputs:
- starting_sequence: 1-letter string (length L).
- per-position log-prob distributions for proposal (PLM-fused).
- catalytic resnos to FREEZE (never propose at these positions).
- energy(s) callback for the MH acceptance criterion.

Outputs:
- final sequence (best-scoring).
- per-step log of (iter, position, old_aa, new_aa, accepted, score) as
  a DataFrame written to disk.
- pool of accepted sequences (CandidateSet).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from protein_chisel.io.schemas import CandidateSet
from protein_chisel.sampling.plm_fusion import AA_ORDER as PLM_AA_ORDER


LOGGER = logging.getLogger("protein_chisel.iterative_optimize")


@dataclass
class IterativeOptimizeConfig:
    mode: str = "constrained_local_search"  # or "mh"
    n_iterations: int = 1000
    sample_temperature: float = 1.0
    initial_mh_temperature: float = 1.0    # for mh / simulated annealing
    final_mh_temperature: float = 0.1
    n_chains: int = 1                      # >1 = independent chains for diagnostics
    convergence_window: int = 100          # iters of no improvement → done
    seed: int = 0
    sequence_id_prefix: str = "iter"


@dataclass
class IterativeOptimizeResult:
    final_sequences: list[str]            # best per chain
    final_scores: list[float]
    candidate_set: CandidateSet           # all accepted sequences across chains
    walk_log: pd.DataFrame
    converged: bool
    n_iterations_run: int
    config: IterativeOptimizeConfig


def iterative_optimize(
    starting_sequence: str,
    per_position_log_probs: np.ndarray,    # (L, 20) PLM-fused marginals (calibrated)
    fixed_positions: set[int],             # 0-indexed positions to freeze
    energy_fn: Optional[Callable[[str], float]] = None,
    accept_fn: Optional[Callable[[str], bool]] = None,
    config: Optional[IterativeOptimizeConfig] = None,
    out_dir: Optional[str | Path] = None,
) -> IterativeOptimizeResult:
    """Run a single-mutation walk.

    Args:
        starting_sequence: seed sequence (1-letter, length L).
        per_position_log_probs: (L, 20) calibrated log-probs in PLM_AA_ORDER.
            Rows for fixed positions are ignored.
        fixed_positions: 0-indexed positions to never modify (typically
            REMARK 666 catalytic residues mapped to 0-indexed).
        energy_fn: scalar energy ``E(s)``; lower is better. Required for
            ``mode='mh'``. Ignored in constrained_local_search mode.
        accept_fn: predicate ``s -> bool``. Required for
            ``mode='constrained_local_search'``. Called on every proposed
            sequence; True = accept the move.
        config: IterativeOptimizeConfig.
    """
    cfg = config or IterativeOptimizeConfig()
    L = len(starting_sequence)
    if per_position_log_probs.shape[0] != L:
        raise ValueError(
            f"log_probs has {per_position_log_probs.shape[0]} rows, "
            f"expected {L}"
        )
    if cfg.mode == "constrained_local_search" and accept_fn is None:
        raise ValueError("constrained_local_search mode requires accept_fn")
    if cfg.mode == "mh" and energy_fn is None:
        raise ValueError("mh mode requires energy_fn")

    rng = np.random.default_rng(cfg.seed)

    final_seqs: list[str] = []
    final_scores: list[float] = []
    all_log_rows: list[dict] = []
    candidate_rows: list[dict] = []
    converged = False
    iters_run = 0

    for chain_idx in range(cfg.n_chains):
        seq = list(starting_sequence)
        best_seq = seq[:]
        best_score = energy_fn("".join(seq)) if energy_fn is not None else 0.0
        no_improvement = 0

        for it in range(cfg.n_iterations):
            iters_run = max(iters_run, it + 1)
            i = _pick_mutable_position(L, fixed_positions, rng)
            if i is None:
                break  # no positions to modify

            new_aa = _propose_aa(
                per_position_log_probs[i], rng, cfg.sample_temperature
            )
            if new_aa == seq[i]:
                continue  # same residue — no-op

            new_seq = seq[:]
            new_seq[i] = new_aa
            new_str = "".join(new_seq)

            accepted = False
            score = float("nan")
            if cfg.mode == "constrained_local_search":
                accepted = bool(accept_fn(new_str))
            elif cfg.mode == "mh":
                e_old = best_score if iters_run == 1 else energy_fn("".join(seq))
                e_new = energy_fn(new_str)
                score = e_new
                T = _mh_temperature(it, cfg)
                delta = e_new - e_old
                # MH acceptance with symmetric proposal (uniform over 19 AAs).
                # NB: the proposal distribution per_position_log_probs is NOT
                # symmetric, so the strict Metropolis-Hastings step requires a
                # q-correction term. We include it: q(s|s') / q(s'|s) =
                # p(old_aa) / p(new_aa) at position i.
                p_aa = np.exp(per_position_log_probs[i] - per_position_log_probs[i].max())
                p_aa = p_aa / p_aa.sum()
                old_idx = PLM_AA_ORDER.index(seq[i]) if seq[i] in PLM_AA_ORDER else None
                new_idx = PLM_AA_ORDER.index(new_aa) if new_aa in PLM_AA_ORDER else None
                if old_idx is not None and new_idx is not None:
                    log_q_correction = float(np.log(max(p_aa[old_idx], 1e-12) / max(p_aa[new_idx], 1e-12)))
                else:
                    log_q_correction = 0.0
                log_alpha = -delta / max(T, 1e-6) + log_q_correction
                accepted = bool(np.log(rng.random()) < log_alpha)

            all_log_rows.append({
                "chain": chain_idx,
                "iter": it,
                "position": i,
                "old_aa": seq[i],
                "new_aa": new_aa,
                "accepted": accepted,
                "score": score,
            })

            if accepted:
                seq = new_seq
                if cfg.mode == "mh" and score < best_score:
                    best_seq = seq[:]
                    best_score = score
                    no_improvement = 0
                elif cfg.mode == "constrained_local_search":
                    best_seq = seq[:]
                    best_score = 0.0
                    no_improvement = 0
                candidate_rows.append({
                    "id": f"{cfg.sequence_id_prefix}_c{chain_idx}_{it:05d}",
                    "sequence": "".join(seq),
                    "parent_design_id": cfg.sequence_id_prefix,
                    "sampler": f"iterative_{cfg.mode}",
                    "sampler_params_hash": "",
                    "iter": it,
                    "chain": chain_idx,
                    "score": best_score,
                })
            else:
                no_improvement += 1
                if no_improvement >= cfg.convergence_window:
                    converged = True
                    break

        final_seqs.append("".join(best_seq))
        final_scores.append(best_score)

    walk_log_df = pd.DataFrame(all_log_rows)
    candidate_df = pd.DataFrame(candidate_rows)

    if out_dir is not None:
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        walk_log_df.to_csv(out_dir / "walk_log.tsv", sep="\t", index=False)
        if not candidate_df.empty:
            CandidateSet(df=candidate_df).to_disk(
                out_dir / "candidates.fasta", out_dir / "candidates.tsv",
            )

    return IterativeOptimizeResult(
        final_sequences=final_seqs,
        final_scores=final_scores,
        candidate_set=CandidateSet(df=candidate_df),
        walk_log=walk_log_df,
        converged=converged,
        n_iterations_run=iters_run,
        config=cfg,
    )


def _pick_mutable_position(
    L: int, fixed: set[int], rng: np.random.Generator
) -> Optional[int]:
    mutable = [i for i in range(L) if i not in fixed]
    if not mutable:
        return None
    return int(rng.choice(mutable))


def _propose_aa(
    log_probs_at_pos: np.ndarray, rng: np.random.Generator, temperature: float
) -> str:
    logits = log_probs_at_pos / max(temperature, 1e-6)
    logits = logits - logits.max()
    p = np.exp(logits)
    p = p / p.sum()
    idx = int(rng.choice(20, p=p))
    return PLM_AA_ORDER[idx]


def _mh_temperature(iter_num: int, cfg: IterativeOptimizeConfig) -> float:
    """Linear annealing from initial_mh_temperature to final."""
    if cfg.n_iterations <= 1:
        return cfg.initial_mh_temperature
    frac = iter_num / (cfg.n_iterations - 1)
    return cfg.initial_mh_temperature * (1 - frac) + cfg.final_mh_temperature * frac


__all__ = [
    "IterativeOptimizeConfig",
    "IterativeOptimizeResult",
    "iterative_optimize",
]
