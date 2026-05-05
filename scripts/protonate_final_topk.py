"""CLI driver: post-design protonation cleanup on a directory of design PDBs.

Wraps ``protein_chisel.tools.protonate_final.protonate_final_topk``. Run
this INSIDE pyrosetta.sif, after the iterative_design_v2 pipeline finishes.

Inputs:
    --topk_dir         directory of .pdb files (typically final_topk/topk_pdbs/)
    --seed_pdb         original seed PDB (source of REMARK 666 + ligand atoms)
    --ligand_params    one or more Rosetta .params files for the ligand
    --out_dir          where to write the cleaned PDBs (default: alongside)
    --ligand_resname   optional 3-letter ligand code (auto-detected if missing)
    --keep_intermediate keep the .rosetta.pdb intermediates for inspection

Output: one ``<stem>.protonated.pdb`` per input PDB. Each one is a
standard, downstream-clean PDB with:
  - all residue hydrogens added by PyRosetta
  - ligand HETATM block (incl. hydrogens) forced from the seed
  - REMARK 666 catalytic-motif lines preserved verbatim
  - REMARK 668 block documenting tautomer/protonation state of every
    catalytic residue, paired by index with REMARK 666
  - standard 3-character residue names (HIS, not HIS_D)
  - clean serial numbering, single TER between protein and ligand
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topk_dir", type=Path, required=True)
    p.add_argument("--seed_pdb", type=Path, required=True)
    p.add_argument("--ligand_params", type=Path, nargs="+", required=True)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--ligand_resname", default=None,
                   help="Optional 3-letter ligand code; auto-detected if missing")
    p.add_argument("--keep_intermediate", action="store_true",
                   help="Keep .rosetta.pdb intermediates for inspection")
    p.add_argument("--summary_json", type=Path, default=None,
                   help="Write run summary stats to this JSON path")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # protein_chisel must be importable
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from protein_chisel.tools.protonate_final import protonate_final_topk

    summary = protonate_final_topk(
        topk_dir=args.topk_dir,
        seed_pdb=args.seed_pdb,
        ligand_params=args.ligand_params,
        out_dir=args.out_dir,
        ligand_resname=args.ligand_resname,
        keep_intermediate=args.keep_intermediate,
    )

    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    if summary.get("failures"):
        print("FAILURES:")
        for name, err in summary["failures"]:
            print(f"  {name}: {err}")

    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2, default=str))

    return 0 if summary["pdbs_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
