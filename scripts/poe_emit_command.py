#!/usr/bin/env python3
"""Print the host ``apptainer exec poe_mpnn.sif python run.py …`` command for the
one-shot PoE sampling stage, NUL-separated (one argv token per record).

Run inside the stage-3 container (which has the ``protein_chisel`` package) so the
command is built by the single source of truth
(:func:`protein_chisel.sampling.mpnn_backends.build_poe_command`); the SHELL then
execs the printed command at the HOST level (nested apptainer is blocked). NUL
separation lets the shell read it into a bash array safely regardless of paths/spaces.

Used by ``run_chisel_design.sh`` when ``MPNN_BACKEND=poe``; not for direct use.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protein_chisel.sampling.mpnn_backends import (  # noqa: E402
    DEFAULT_POE_LIGAND_CHECKPOINT, build_poe_command, validate_expert_lambdas,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", required=True, help="input PDB")
    ap.add_argument("--out", required=True, help="PoE out_folder (gets seqs/ + packed/)")
    ap.add_argument("--experts", required=True)
    ap.add_argument("--lambdas", required=True)
    ap.add_argument("--bias_json", default=None)
    ap.add_argument("--omit_json", default=None)
    ap.add_argument("--fixed_json", default=None)
    ap.add_argument("--checkpoint", default=DEFAULT_POE_LIGAND_CHECKPOINT)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--number_of_batches", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--seed_int", type=int, default=0)
    ap.add_argument("--use_atom_context", type=int, default=1)
    ap.add_argument("--use_side_chain_context", type=int, default=0)
    ap.add_argument("--omit_AA", default="", help="global omit AAs, e.g. 'CX'")
    ap.add_argument("--hermes_probs", default=None)
    a = ap.parse_args()

    # Validate here too (fail fast at command-build time with a clear message).
    experts, lambdas = validate_expert_lambdas(a.experts, a.lambdas)
    cmd = build_poe_command(
        pdb_path=a.seed, out_folder=a.out, experts=experts, lambdas=lambdas,
        bias_json=a.bias_json, omit_json=a.omit_json, fixed_json=a.fixed_json,
        checkpoint=a.checkpoint, batch_size=a.batch_size,
        number_of_batches=a.number_of_batches, temperature=a.temperature,
        seed=a.seed_int, use_atom_context=a.use_atom_context,
        use_side_chain_context=a.use_side_chain_context, omit_AA=a.omit_AA,
        hermes_probs=a.hermes_probs,
    )
    # NUL-terminate each token so the shell can `mapfile -d ''` it safely
    # regardless of spaces/special chars in paths.
    sys.stdout.write("".join(tok + "\0" for tok in cmd))


if __name__ == "__main__":
    main()
