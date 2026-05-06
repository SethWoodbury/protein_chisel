"""Run classify_positions on the PTE_i1 seed PDB.

Runs INSIDE pyrosetta.sif. Writes the PositionTable to a parquet/tsv
that ``precompute_plm_artifacts.py`` (esmc.sif) consumes downstream.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


LOGGER = logging.getLogger("classify_positions_pte_i1")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed_pdb", type=Path, required=True)
    p.add_argument("--ligand_params", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--pose_id", default="PTE_i1")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = args.out_dir / "positions.tsv"
    if out_tsv.exists():
        LOGGER.info("positions cache hit -> %s", out_tsv)
        return

    from protein_chisel.tools.classify_positions import classify_positions

    pt = classify_positions(
        args.seed_pdb,
        pose_id=args.pose_id,
        params=[args.ligand_params],
    )
    pt.to_parquet(out_tsv)
    LOGGER.info("classify -> %s rows=%d", out_tsv, len(pt.df))


if __name__ == "__main__":
    main()
