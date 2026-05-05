"""Empirical memory probe: run ESM-C 600m + SaProt 1.3B masked-LM
on a synthetic 275-residue protein on CPU and report peak RSS.

Strategy:
- Generate 275-aa alpha helix PDB (real backbone CA/N/C/O coords are
  needed for SaProt's foldseek-3Di tokenizer to produce sensible
  tokens; the sequence content is irrelevant for the memory test).
- Call esmc_logits (sequence-only, length-275) and saprot_logits
  (structure-aware) directly.
- After each, print rss/maxrss via resource.getrusage.

Run inside esmc.sif container with the protein_chisel package on
PYTHONPATH.
"""
from __future__ import annotations

import argparse
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np


def _ru_mb() -> tuple[float, float]:
    """Return (current_rss_mb, peak_rss_mb_so_far)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # Linux ru_maxrss is KB
    peak_kb = ru.ru_maxrss
    # Read /proc for current
    try:
        with open(f"/proc/{os.getpid()}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    cur_kb = int(line.split()[1])
                    return cur_kb / 1024.0, peak_kb / 1024.0
    except Exception:
        pass
    return -1.0, peak_kb / 1024.0


def _log(tag: str) -> None:
    cur, peak = _ru_mb()
    print(f"[mem] {tag:<32s} cur_rss={cur:9.1f} MB  peak_rss={peak:9.1f} MB",
          flush=True)


def write_helix_pdb(L: int, path: Path) -> str:
    """Write an idealized alpha-helix PDB of length L. All residues = ALA.

    Pure-alpha geometry: 5.4 A pitch, ~3.6 res/turn, 1.5 A radius for CA.
    Backbone atoms N, CA, C, O placed at standard offsets.
    """
    seq = "A" * L
    rise_per_res = 1.5  # angstroms along z
    twist_per_res = 100.0 * math.pi / 180.0  # radians
    radius = 2.3

    rows = []
    atom_idx = 1
    for i in range(L):
        theta = i * twist_per_res
        z = i * rise_per_res
        # CA position
        ca_x = radius * math.cos(theta)
        ca_y = radius * math.sin(theta)
        ca_z = z
        # Approximate backbone offsets (idealized — good enough for tokenizer)
        n_x = ca_x + 0.5 * math.cos(theta + 1.0)
        n_y = ca_y + 0.5 * math.sin(theta + 1.0)
        n_z = ca_z - 0.6
        c_x = ca_x + 0.7 * math.cos(theta + 0.5)
        c_y = ca_y + 0.7 * math.sin(theta + 0.5)
        c_z = ca_z + 0.6
        o_x = c_x + 0.8 * math.cos(theta + 0.7)
        o_y = c_y + 0.8 * math.sin(theta + 0.7)
        o_z = c_z + 0.4

        for atom_name, (x, y, z_) in [
            ("N", (n_x, n_y, n_z)),
            ("CA", (ca_x, ca_y, ca_z)),
            ("C", (c_x, c_y, c_z)),
            ("O", (o_x, o_y, o_z)),
        ]:
            rows.append(
                f"ATOM  {atom_idx:5d}  {atom_name:<3s} ALA A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z_:8.3f}  1.00  0.00           "
                f"{atom_name[0]:>2s}"
            )
            atom_idx += 1

    rows.append("TER")
    rows.append("END")
    path.write_text("\n".join(rows) + "\n")
    return seq


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=int, default=275)
    p.add_argument("--esmc_model", default="esmc_600m")
    p.add_argument("--saprot_model", default="saprot_1.3b")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", type=Path,
                   default=Path("/net/scratch/woodbuse/probe_275aa"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = args.out_dir / f"helix_L{args.length}.pdb"
    seq = write_helix_pdb(args.length, pdb_path)
    print(f"[setup] wrote synthetic helix PDB L={args.length} -> {pdb_path}",
          flush=True)
    _log("baseline (post import numpy)")

    print(f"[setup] esmc_model={args.esmc_model}  saprot_model={args.saprot_model}  device={args.device}",
          flush=True)

    # Import inside esmc.sif
    from protein_chisel.tools.esmc import esmc_logits
    from protein_chisel.tools.saprot import saprot_logits
    _log("after protein_chisel import")

    # ----- ESM-C masked-LM
    t0 = time.time()
    esmc_lp = esmc_logits(
        seq, model_name=args.esmc_model, device=args.device, masked=True,
    ).log_probs
    print(f"[time] esmc_logits L={args.length} {args.esmc_model}: {time.time()-t0:.1f}s",
          flush=True)
    _log(f"after ESM-C ({args.esmc_model})")
    print(f"[shape] esmc_lp.shape = {esmc_lp.shape}", flush=True)

    # ----- SaProt masked-LM
    t0 = time.time()
    saprot_lp = saprot_logits(
        pdb_path, chain=None,  # synthetic single-chain PDB; foldseek
                               # emits basename without _A suffix
        model_name=args.saprot_model, device=args.device, masked=True,
    ).log_probs
    print(f"[time] saprot_logits L={args.length} {args.saprot_model}: {time.time()-t0:.1f}s",
          flush=True)
    _log(f"after SaProt ({args.saprot_model})")
    print(f"[shape] saprot_lp.shape = {saprot_lp.shape}", flush=True)

    print(f"[done] L={args.length} both PLMs run on CPU", flush=True)


if __name__ == "__main__":
    main()
