"""Precompute PLM artifacts for the iterative_design_v2 driver.

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
        choices=["saprot_35m", "saprot_650m", "saprot_1.3b"],
        help="SaProt model variant. saprot_35m (default, ~10s on H100; "
             "AF2-trained) | saprot_650m (PDB-trained, ~3× slower, "
             "structure-aware sharper) | saprot_1.3b (AFDB+OMG+NCBI, "
             "~10× slower, ~6 GB VRAM, broadest pretraining). All "
             "cached at /net/databases/huggingface/saprot/hub.",
    )
    p.add_argument("--device", default="auto",
                   help="'auto' picks cuda if available else cpu. Pass "
                        "'cpu' to force CPU even on a GPU node.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports — only available inside esmc.sif
    from protein_chisel.io.pdb import extract_sequence
    from protein_chisel.io.schemas import PositionTable
    from protein_chisel.tools.esmc import esmc_logits
    from protein_chisel.tools.saprot import saprot_logits
    from protein_chisel.sampling.plm_fusion import (
        FusionConfig, fuse_plm_logits,
    )

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

    # ---- ESM-C masked-LM marginals ---------------------------------------
    esmc_path = args.out_dir / "esmc_log_probs.npy"
    if esmc_path.exists():
        LOGGER.info("esmc cache hit -> %s", esmc_path)
        esmc_lp = np.load(esmc_path)
    else:
        LOGGER.info("running ESM-C (%s) masked-LM (L=%d forward passes)",
                     args.esmc_model, L)
        esmc_lp = esmc_logits(
            seq, model_name=args.esmc_model, device=args.device, masked=True,
        ).log_probs
        np.save(esmc_path, esmc_lp)
        LOGGER.info("esmc -> %s shape=%s", esmc_path, esmc_lp.shape)

    # ---- SaProt masked-LM marginals --------------------------------------
    saprot_path = args.out_dir / "saprot_log_probs.npy"
    if saprot_path.exists():
        LOGGER.info("saprot cache hit -> %s", saprot_path)
        saprot_lp = np.load(saprot_path)
    else:
        LOGGER.info("running SaProt (%s) masked-LM", args.saprot_model)
        saprot_lp = saprot_logits(
            args.seed_pdb, chain=args.chain,
            model_name=args.saprot_model, device=args.device, masked=True,
        ).log_probs
        np.save(saprot_path, saprot_lp)
        LOGGER.info("saprot -> %s shape=%s", saprot_path, saprot_lp.shape)

    if esmc_lp.shape != saprot_lp.shape:
        raise RuntimeError(
            f"ESM-C shape {esmc_lp.shape} != SaProt {saprot_lp.shape}"
        )
    if esmc_lp.shape[0] != L:
        raise RuntimeError(
            f"PLM log-probs length {esmc_lp.shape[0]} != seq length {L}"
        )

    # ---- Calibrated fusion -----------------------------------------------
    bias_path = args.out_dir / "fusion_bias.npy"
    log_odds_esmc_path = args.out_dir / "fusion_log_odds_esmc.npy"
    log_odds_saprot_path = args.out_dir / "fusion_log_odds_saprot.npy"
    weights_path = args.out_dir / "fusion_weights.npy"
    fusion_cfg = FusionConfig()
    if bias_path.exists():
        LOGGER.info("fusion cache hit -> %s", bias_path)
        bias = np.load(bias_path)
    else:
        LOGGER.info("fusing PLM log-probs -> bias matrix")
        result = fuse_plm_logits(
            log_probs_esmc=esmc_lp,
            log_probs_saprot=saprot_lp,
            position_classes=pos_classes,
            config=fusion_cfg,
        )
        np.save(bias_path, result.bias)
        np.save(log_odds_esmc_path, result.log_odds_esmc)
        np.save(log_odds_saprot_path, result.log_odds_saprot)
        np.save(weights_path, result.weights_per_position)
        bias = result.bias
        LOGGER.info("fusion bias shape=%s, mean_abs=%.4f",
                     bias.shape, float(np.abs(bias).mean()))

    # ---- Manifest --------------------------------------------------------
    manifest = {
        "tool": "precompute_plm_artifacts",
        "seed_pdb": str(args.seed_pdb),
        "seed_pdb_sha16": _file_hash(args.seed_pdb),
        "chain": args.chain,
        "wt_length": L,
        "esmc_model": args.esmc_model,
        "saprot_model": args.saprot_model,
        "fusion_config": {
            "entropy_match": fusion_cfg.entropy_match,
            "shrink_disagreement": fusion_cfg.shrink_disagreement,
            "shrink_threshold": fusion_cfg.shrink_threshold,
            "class_weights": dict(fusion_cfg.class_weights),
        },
        "outputs": {
            "esmc_log_probs": str(esmc_path),
            "saprot_log_probs": str(saprot_path),
            "fusion_bias": str(bias_path),
            "fusion_log_odds_esmc": str(log_odds_esmc_path),
            "fusion_log_odds_saprot": str(log_odds_saprot_path),
            "fusion_weights": str(weights_path),
        },
    }
    with open(args.out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    LOGGER.info("DONE -> %s", args.out_dir)


if __name__ == "__main__":
    main()
