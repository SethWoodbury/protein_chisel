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


def _parse_bool_arg(value: str | bool) -> bool:
    """Parse a CLI boolean from common true/false spellings."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected boolean true/false for this flag, got {value!r}",
    )


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
    p.add_argument("--shipping_layout", action="store_true",
                   help="After protonation, REORGANIZE the parent run_dir "
                        "into a clean shipping layout: run_dir/designs/*.pdb "
                        "(renamed from .protonated.pdb), run_dir/designs.tsv "
                        "(top-K with pdb_path), run_dir/designs.fasta, and "
                        "removes heavy cycle_NN/ subtrees + the dual "
                        "final_topk wrapper. Pair with --keep_intermediates "
                        "if you want intermediates preserved.")
    p.add_argument("--no_strip_intermediates", action="store_true",
                   help="With --shipping_layout, also keep cycle_NN/ etc. "
                        "Useful for diagnostics. Default behavior with "
                        "--shipping_layout strips them.")
    p.add_argument("--minimal_layout", action="store_true",
                   help="With --shipping_layout, collapse the run_dir to "
                        "ONLY {designs/, designs.tsv}. Embeds manifest + "
                        "cycle_metrics + throat_blocker_telemetry + "
                        "protonation_summary as a single-line "
                        "'# RUN_META: <json>' comment at the top of "
                        "designs.tsv (load with "
                        "pd.read_csv(path, sep='\\t', comment='#')). "
                        "Drops designs.fasta (sequence column is in the "
                        "TSV) and all aux JSON files. Use for /net/scratch "
                        "space efficiency on production sweeps.")
    p.add_argument("--copy-input-structure-into-out-dir",
                   "--copy_input_structure_into_out_dir",
                   dest="copy_input_structure_into_out_dir",
                   type=_parse_bool_arg, default=True,
                   metavar="{true,false}",
                   help="Default true. Copy the original input PDB into the "
                        "final shipped output directory and append its metrics "
                        "row to chiseled_design_metrics.tsv.")
    p.add_argument("--ptm", type=str, default=None,
                   help="Comma-separated PTM declarations recorded in the "
                        "output PDB's REMARK 668 block. ANNOTATION ONLY: "
                        "the residue is left as its unmodified form (LYS, "
                        "not KCX) for Rosetta / sequence reading / "
                        "protonation; the PTM column is metadata for "
                        "downstream tools. "
                        "Two spec formats: "
                        "(a) motif-index form 'A/LYS/3:KCX' = chain A, "
                        "expected resname LYS, REMARK 666 motif index 3, "
                        "PTM code KCX (preferred for catalytic residues — "
                        "stable across a design campaign even when the "
                        "sequence position varies between scaffolds); "
                        "(b) explicit-residue form 'A:157=KCX' = chain A "
                        "residue 157 -> KCX. "
                        "Use '-' as code to FORCE no-PTM (overrides "
                        "auto-detect from seed atom inventory).")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
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
        ptm_map=args.ptm,
    )

    if args.shipping_layout:
        from protein_chisel.tools.protonate_final import reorganize_for_shipping
        # The run_dir is the parent of topk_dir's parent (final_topk/topk_pdbs)
        run_dir = args.topk_dir.parent.parent
        reorg_stats = reorganize_for_shipping(
            run_dir=run_dir,
            strip_intermediates=not args.no_strip_intermediates,
            minimal=args.minimal_layout,
            seed_pdb=args.seed_pdb,
            copy_input_structure=args.copy_input_structure_into_out_dir,
        )
        summary["shipping_layout"] = reorg_stats

    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    if summary.get("failures"):
        print("FAILURES:")
        for name, err in summary["failures"]:
            print(f"  {name}: {err}")

    if args.summary_json and not args.minimal_layout:
        # In minimal mode the summary is captured inside the embedded
        # RUN_META block in chiseled_design_metrics.tsv, so the separate
        # JSON file would just be noise in the otherwise-flat run_dir.
        args.summary_json.write_text(json.dumps(summary, indent=2, default=str))

    return 0 if summary["pdbs_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
