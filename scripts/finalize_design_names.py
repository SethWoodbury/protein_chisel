"""CLI: final rank-order rename + DESIGN_PATH collapse on a published run dir.

Invoked as the LAST step of run_chisel_design.sh, in plain python3 (no container),
on the final published directory. See protein_chisel.tools.finalize_names.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--final_root", type=Path, required=True,
                   help="Published run directory containing "
                        "chiseled_design_metrics.tsv + the shipped design PDBs.")
    p.add_argument("--minimal", default=None,
                   help="Layout hint (true/false); ignored — the designs dir is "
                        "derived from the TSV pdb_path. Accepted for shell parity.")
    p.add_argument("--keep_intermediate", action="store_true",
                   help="Keep the intermediate iterative_design + protonate_topk "
                        "DESIGN_PATH lines (default: drop them, leaving one "
                        "chisel_iterative_design line with the final path).")
    p.add_argument("--design_token",
                   default=(os.environ.get("CHISEL_SUFFIX") or "chisel"),
                   help="Name component in the shipped filename "
                        "<stem>_<design_token>_<NNN>.pdb (default 'chisel', or "
                        "the CHISEL_SUFFIX env var). Alphanumeric only.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from protein_chisel.tools.finalize_names import finalize_design_names

    summary = finalize_design_names(
        args.final_root, keep_intermediate=args.keep_intermediate,
        design_token=args.design_token)
    logging.getLogger("finalize_design_names").info("DONE: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
