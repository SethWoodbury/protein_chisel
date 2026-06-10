"""Precompute PLM artifacts for the iterative_design driver.

Runs INSIDE esmc.sif. Inputs:
- seed PDB
- a PositionTable parquet/tsv produced by classify_positions in pyrosetta.sif

Outputs to ``<out_dir>/``:
    esmc_log_probs.npy            (L, 20)   masked-LM marginals
    saprot_log_probs.npy          (L, 20)   masked-LM marginals
    fusion_bias.npy               (L, 20)   calibrated cycle-0 bias
    fusion_log_odds_esmc.npy      (L, 20)   calibrated log-odds
    fusion_log_odds_saprot.npy    (L, 20)   calibrated log-odds
    fusion_weights.npy            (L, 2)    per-position β, γ
    manifest.json

Each artifact is keyed by ``input_seq + esmc/saprot model + chain``;
re-running with the same inputs is idempotent (won't recompute).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger("precompute_plm_artifacts")


def _file_hash(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed_pdb", type=Path, required=True)
    p.add_argument("--position_table", type=Path, required=True,
                   help="parquet/tsv from classify_positions")
    p.add_argument("--chain", default="A")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument(
        "--esmc_model", default="esmc_300m",
        choices=["esmc_300m", "esmc_600m"],
        help="ESM-C model variant. esmc_300m (default, ~46s on H100, "
             "fits in ~2 GB VRAM) | esmc_600m (~2× slower, ~4 GB VRAM, "
             "marginally better masked-LM probs). Cached at "
             "/net/databases/huggingface/esmc/hub.",
    )
    p.add_argument(
        "--saprot_model", default="saprot_35m",
        choices=["saprot_35m", "saprot_650m", "saprot_650m_af2", "saprot_1.3b"],
        help="SaProt model variant. From westlake-repl on HF, all four "
             "now cached at /net/databases/huggingface/saprot/hub: "
             "saprot_35m = SaProt_35M_AF2 (35M params, 40M AF2 structures; "
             "default, ~10s on H100). "
             "saprot_650m = SaProt_650M_PDB (650M, 40M AF2 phase1 + 60K "
             "PDB phase2; ~3× slower). "
             "saprot_650m_af2 = SaProt_650M_AF2 (650M, AF2-only pretrain; "
             "added 2026-05-04 because the upstream README documents "
             "experimental benchmarking against this checkpoint; not yet "
             "validated in our pipeline). "
             "saprot_1.3b = SaProt_1.3B_AFDB_OMG_NCBI (1.3B, 40M AF2 + "
             "200M OMG_prot50 + 150M NCBI 70%% identity-filtered; ~10× "
             "slower, ~6 GB VRAM, broadest pretraining).",
    )
    p.add_argument("--device", default="auto",
                   help="'auto' picks cuda if available else cpu. Pass "
                        "'cpu' to force CPU even on a GPU node.")
    p.add_argument("--plm_dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="PLM inference precision. 'fp32' (default) is byte-identical; "
                        "'fp16'/'bf16' ~halve PLM memory but change the logits, so "
                        "their artifacts are written to dtype-suffixed cache files "
                        "(e.g. saprot_log_probs.fp16.npy) and never reuse the fp32 cache.")
    p.add_argument(
        "--experts", default="esmc,saprot",
        help="Comma list of fusion experts (registry names). Default "
             "'esmc,saprot' reproduces the legacy two-PLM artifacts exactly. "
             "Add e.g. 'esmc,saprot,hermes' to fuse more experts. Per-expert "
             "model variants come from --esmc_model/--saprot_model.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Memory warning BEFORE the heavy PLM imports/loads, so it lands early in the
    # Stage-2 log if the configured models risk exceeding the --mem budget.
    # Logging-only — never changes behavior.
    try:
        from protein_chisel.utils.resources import warn_if_plm_mem_tight
        # --device cpu forces host-RAM placement (the GPU factor would underestimate);
        # "auto"/"cuda" -> None lets the estimate auto-detect a visible GPU.
        _on_gpu = False if args.device == "cpu" else None
        warn_if_plm_mem_tight(args.esmc_model, args.saprot_model,
                              getattr(args, "plm_dtype", "fp32"), on_gpu=_on_gpu)
    except Exception as _e:        # never let a memory probe break the run
        LOGGER.debug("PLM memory warn skipped: %s", _e)

    # Lazy imports — only available inside esmc.sif
    import gc
    import time
    from protein_chisel.io.pdb import extract_sequence
    from protein_chisel.io.schemas import PositionTable
    from protein_chisel.experts import ExpertContext, resolve_experts
    from protein_chisel.sampling.plm_fusion import FusionConfig, fuse_experts

    seq = extract_sequence(args.seed_pdb, chain=args.chain)
    L = len(seq)
    LOGGER.info("seed sequence: chain=%s, L=%d", args.chain, L)

    pt = PositionTable.from_parquet(args.position_table)
    protein_rows = pt.df[pt.df["is_protein"]].sort_values("resno")
    if len(protein_rows) != L:
        raise RuntimeError(
            f"PositionTable protein rows={len(protein_rows)} != seq length {L}; "
            "classify_positions must agree with extract_sequence on chain"
        )
    pos_classes = protein_rows["class"].tolist()
    LOGGER.info("position classes (counts): %s",
                 protein_rows["class"].value_counts().to_dict())

    # ---- Per-expert masked-LM marginals (registry-driven) ----------------
    # Default experts ["esmc","saprot"] reproduce the legacy artifacts exactly:
    # each Expert writes <name>_log_probs.npy (esmc_log_probs.npy /
    # saprot_log_probs.npy) and the fusion delegates to the legacy 2-PLM path.
    # Experts are computed one at a time (load model -> compute -> free) so we
    # never hold multiple large models resident.
    experts = resolve_experts(
        args.experts,
        model_names={"esmc": args.esmc_model, "saprot": args.saprot_model},
    )
    expert_names = [e.name for e in experts]
    LOGGER.info("experts: %s", [e.version for e in experts])
    ctx = ExpertContext(seq=seq, pdb_path=args.seed_pdb, chain=args.chain,
                        device=args.device, out_dir=args.out_dir,
                        plm_dtype=args.plm_dtype)
    expert_lps = []
    for exp in experts:
        t0 = time.perf_counter()
        lp = exp.log_probs(ctx)   # cached <name>_log_probs.npy via out_dir
        if lp.shape[0] != L:
            raise RuntimeError(
                f"{exp.name} log-probs length {lp.shape[0]} != seq length {L}")
        LOGGER.info("expert %s -> shape=%s (%.1fs)", exp.name, lp.shape,
                    time.perf_counter() - t0)
        expert_lps.append(lp)
        # Memory discipline: the model lives inside compute_log_probs and is already
        # out of scope, but reclaim its allocations NOW (don't wait for lazy GC) so
        # the next expert's model never overlaps the previous one's footprint. Frees
        # only the model/cache — the small (L,20) `lp` is kept in expert_lps.
        gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
    shapes = {tuple(lp.shape) for lp in expert_lps}
    if len(shapes) != 1:
        raise RuntimeError(f"expert log-prob shapes disagree: {shapes}")

    # ---- Calibrated fusion -----------------------------------------------
    # Dtype-suffix ALL fusion artifacts too (not just per-expert), so the
    # existence-based "fusion cache hit" below is dtype-aware: re-running a
    # different --plm_dtype in the same out_dir never reuses another dtype's bias.
    # fp32 keeps the legacy names (fusion_bias.npy ...) => byte-identical.
    _dtsuf = "" if args.plm_dtype == "fp32" else f".{args.plm_dtype}"
    bias_path = args.out_dir / f"fusion_bias{_dtsuf}.npy"
    log_odds_esmc_path = args.out_dir / f"fusion_log_odds_esmc{_dtsuf}.npy"
    log_odds_saprot_path = args.out_dir / f"fusion_log_odds_saprot{_dtsuf}.npy"
    weights_path = args.out_dir / f"fusion_weights{_dtsuf}.npy"
    fusion_cfg = FusionConfig()
    if bias_path.exists():
        LOGGER.info("fusion cache hit -> %s", bias_path)
        bias = np.load(bias_path)
    else:
        t0 = time.perf_counter()
        LOGGER.info("fusing %d expert log-prob array(s) -> bias matrix",
                     len(experts))
        # For the default ["esmc","saprot"] (no per-expert knobs) fuse_experts
        # takes its N=2 fast-path and DELEGATES to the legacy fuse_plm_logits, so
        # the bias + legacy artifacts are byte-identical (proven by
        # tests/sampling/test_fuse_experts.py::test_default_two_expert_byte_identical).
        result = fuse_experts(
            expert_lps, pos_classes, config=fusion_cfg, expert_names=expert_names,
        )
        np.save(bias_path, result.bias)
        # Legacy 2-expert artifacts: kept whenever the result populated them (the
        # N=2 case) so the iterative_design loader + fitness scorer are unchanged.
        if (result.log_odds_esmc is not None
                and result.log_odds_saprot is not None
                and result.weights_per_position is not None):
            np.save(log_odds_esmc_path, result.log_odds_esmc)
            np.save(log_odds_saprot_path, result.log_odds_saprot)
            np.save(weights_path, result.weights_per_position)
        # Generic per-expert artifacts only for non-default expert sets.
        if result.log_odds is not None and len(experts) != 2:
            for nm, lo in zip(expert_names, result.log_odds):
                np.save(args.out_dir / f"fusion_log_odds_{nm}{_dtsuf}.npy", lo)
            if result.weights_per_expert is not None:
                np.save(args.out_dir / f"fusion_weights_per_expert{_dtsuf}.npy",
                        result.weights_per_expert)
        bias = result.bias
        LOGGER.info("fusion bias shape=%s, mean_abs=%.4f (%.2fs)",
                     bias.shape, float(np.abs(bias).mean()),
                     time.perf_counter() - t0)

    # ---- Manifest --------------------------------------------------------
    outputs = {
        # dtype-aware: fp32 -> legacy <name>_log_probs.npy; fp16/bf16 -> suffixed.
        f"{e.name}_log_probs": str(args.out_dir / e.cache_filename_for(args.plm_dtype))
        for e in experts
    }
    outputs["fusion_bias"] = str(bias_path)
    if len(experts) == 2:
        outputs["fusion_log_odds_esmc"] = str(log_odds_esmc_path)
        outputs["fusion_log_odds_saprot"] = str(log_odds_saprot_path)
        outputs["fusion_weights"] = str(weights_path)
    manifest = {
        "tool": "precompute_plm_artifacts",
        "seed_pdb": str(args.seed_pdb),
        "seed_pdb_sha16": _file_hash(args.seed_pdb),
        "chain": args.chain,
        "wt_length": L,
        "esmc_model": args.esmc_model,
        "saprot_model": args.saprot_model,
        "plm_dtype": args.plm_dtype,
        # Provenance: which experts (+ versions) and fusion math produced the bias.
        "experts": expert_names,
        "expert_versions": {e.name: e.version for e in experts},
        "fusion_version": fusion_cfg.version,
        "fusion_config": {
            "entropy_match": fusion_cfg.entropy_match,
            "shrink_disagreement": fusion_cfg.shrink_disagreement,
            "shrink_threshold": fusion_cfg.shrink_threshold,
            "class_weights": dict(fusion_cfg.class_weights),
        },
        "outputs": outputs,
    }
    with open(args.out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    LOGGER.info("DONE -> %s", args.out_dir)


if __name__ == "__main__":
    main()
