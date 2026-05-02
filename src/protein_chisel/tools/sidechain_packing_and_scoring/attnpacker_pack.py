"""AttnPacker wrapper -- SE(3)-equivariant transformer side-chain packer.

AttnPacker (McPartlon & Xu 2023, *PNAS*) is one of the most accurate
neural side-chain packers. **License caveat**: the upstream GitHub repo
has NO LICENSE file -- treat as academic-use-only and do NOT redistribute.

The repo is vendored as a git submodule at ``external/attnpacker``;
runtime lives inside ``esmc.sif`` at ``/opt/attnpacker``. Unlike the
other packers, AttnPacker has no standalone inference CLI -- it exposes
a ``Inference`` Python class that takes ``RESOURCE_ROOT=<unzipped weights
dir>`` and a ``infer(pdb_path=..., chunk_size=500, format=True)`` call.
We drive it via a small inline script through ``apptainer exec``.

Pretrained weights are at:
- ``/net/databases/lab/attnpacker/AttnPackerPTM_V2.zip`` (6.9 GB,
  Zenodo 7713779) -- the standard full-atom packer.
- (optional) ``AttnPackerPlusRotPTM.zip`` (4.6 GB, Zenodo 7843977) --
  the design variant supporting partial sequence + rotamer conditioning.

We auto-unzip the weights to ``AttnPackerPTM_V2/`` next to the .zip on
first use, and pass that as RESOURCE_ROOT.

Single entry point :func:`attnpacker_pack`. There's no chi-log-likelihood
output from AttnPacker -- per-atom confidence ("uncertainty") is the
closest scoring signal but isn't exposed as a calibrated likelihood.
For deviation-from-prediction scoring, use :func:`attnpacker_score` which
runs pack and compares to the input as the "native" reference, same
pattern as :mod:`pippack_score`.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger("protein_chisel.attnpacker_pack")


@dataclass
class AttnPackerPackResult:
    """Result of an AttnPacker repack run."""
    out_pdb_path: Path
    runtime_seconds: float = 0.0
    resource_root: Optional[Path] = None    # which weights dir we used


@dataclass
class AttnPackerScoreResult:
    """Result of pack-and-compare scoring with AttnPacker.

    Same per-residue chi MAE / rotamer recovery / sidechain RMSD layout
    as :class:`pippack_score.PIPPackScoreResult`, but the metrics are
    derived from RMSD comparison to the input (no native rotamer
    library lookup, just XYZ-vs-XYZ).
    """
    per_residue_df: object = None  # pandas.DataFrame
    n_residues_scored: int = 0
    mean_chi_mae: float = 0.0
    chi1_mae: float = 0.0
    chi2_mae: float = 0.0
    chi3_mae: float = 0.0
    chi4_mae: float = 0.0
    rotamer_recovery: float = 0.0
    mean_sidechain_rmsd: float = 0.0
    runtime_seconds: float = 0.0

    def to_dict(self, prefix: str = "attnpacker__") -> dict[str, float | int]:
        return {
            f"{prefix}n_residues_scored": int(self.n_residues_scored),
            f"{prefix}mean_chi_mae": float(self.mean_chi_mae),
            f"{prefix}chi1_mae": float(self.chi1_mae),
            f"{prefix}chi2_mae": float(self.chi2_mae),
            f"{prefix}chi3_mae": float(self.chi3_mae),
            f"{prefix}chi4_mae": float(self.chi4_mae),
            f"{prefix}rotamer_recovery": float(self.rotamer_recovery),
            f"{prefix}mean_sidechain_rmsd": float(self.mean_sidechain_rmsd),
        }


def _resolve_resource_root(weights_dir: Path) -> Path:
    """Find or create the unzipped AttnPacker weights directory.

    Prefers an existing ``AttnPackerPTM_V2/`` next to the zip. If only
    the zip exists, unzip it on first use.
    """
    unzipped = weights_dir / "AttnPackerPTM_V2"
    if unzipped.is_dir() and any(unzipped.iterdir()):
        return unzipped
    zipped = weights_dir / "AttnPackerPTM_V2.zip"
    if not zipped.is_file():
        raise FileNotFoundError(
            f"AttnPacker weights not found at {weights_dir}. "
            f"Expected either {unzipped} (unzipped) or {zipped} (zip from "
            "Zenodo 7713779)."
        )
    LOGGER.info("Unzipping %s -> %s (one-time, ~7 GB)", zipped, weights_dir)
    with zipfile.ZipFile(zipped, "r") as zf:
        zf.extractall(weights_dir)
    if not unzipped.is_dir():
        raise RuntimeError(
            f"Expected {unzipped} after unzip; got {list(weights_dir.iterdir())}"
        )
    return unzipped


# Inline driver script run inside the sif. Mirrors the
# `protein_learning/examples/Inference.ipynb` flow: instantiate
# Inference(model_n_config_root, ...), call infer(pdb_path, ...), then
# build the predicted Protein via make_predicted_protein and write
# with .to_pdb(beta=pLDDT).
_ATTNPACKER_PACK_SCRIPT = r"""
import os
import sys
import time

resource_root = sys.argv[1]
in_pdb = sys.argv[2]
out_pdb = sys.argv[3]
chunk_size = int(sys.argv[4]) if len(sys.argv) > 4 else 500

# AttnPacker expects to be importable from /opt/attnpacker (top-level
# `protein_learning` package).
sys.path.insert(0, "/opt/attnpacker")

from protein_learning.models.inference_utils import (
    Inference, make_predicted_protein,
)

t0 = time.perf_counter()
# Inference's first positional arg is the resource root (named
# `model_n_config_root` in the source). The path layout expected is
# <root>/models/<name>.tar + <root>/params/<name>.npy.
runner = Inference(resource_root, use_design_variant=False)
prediction = runner.infer(
    pdb_path=in_pdb,
    chunk_size=chunk_size,
    format=True,
)
predicted_protein = make_predicted_protein(
    model_out=prediction["model_out"],
    seq=prediction["seq"],
)
predicted_protein.to_pdb(out_pdb, beta=prediction["pred_plddt"].squeeze())
elapsed = time.perf_counter() - t0
print(f"ATTNPACKER_DONE seconds={elapsed:.2f} out={out_pdb}")
"""


def attnpacker_pack(
    pdb_path: str | Path,
    out_pdb_path: Optional[str | Path] = None,
    weights_dir: Optional[Path] = None,
    chunk_size: int = 500,
    timeout: float = 1800.0,
) -> AttnPackerPackResult:
    """Repack side chains with AttnPacker.

    Args:
        pdb_path: input PDB.
        out_pdb_path: output PDB. Defaults to ``<input_stem>_attnpacker.pdb``.
        weights_dir: directory containing ``AttnPackerPTM_V2.zip`` (or
            already-unzipped ``AttnPackerPTM_V2/``). Defaults to
            ``/net/databases/lab/attnpacker``.
        chunk_size: AttnPacker's tile size (residues processed at once).
            500 is a reasonable default for a4000/a6000.
        timeout: subprocess timeout (s).
    """
    import time

    from protein_chisel.paths import (
        ATTNPACKER_GUEST_SOURCE_DIR,
        ATTNPACKER_WEIGHTS_DIR,
    )
    from protein_chisel.utils.apptainer import esmc_call

    pdb_path = Path(pdb_path).resolve()
    if out_pdb_path is None:
        out_pdb_path = pdb_path.with_name(f"{pdb_path.stem}_attnpacker.pdb")
    out_pdb_path = Path(out_pdb_path).resolve()
    out_pdb_path.parent.mkdir(parents=True, exist_ok=True)

    resource_root = _resolve_resource_root(
        Path(weights_dir) if weights_dir else ATTNPACKER_WEIGHTS_DIR
    )

    # AttnPacker's SE(3) transformer caches Clebsch-Gordan basis pickles
    # at /opt/attnpacker/protein_learning/models/.basis_cache/<rxN>/ and
    # acquires a `mutex` filelock on first access. /opt is read-only inside
    # the sif, so we copy the cache to a writable scratch dir and bind
    # it over the in-sif path.
    cache_scratch = Path(tempfile.mkdtemp(prefix="chisel_attnpacker_cache_"))
    src_cache = ATTNPACKER_GUEST_SOURCE_DIR / "protein_learning" / "models" / ".basis_cache"
    # The host source dir mirrors /opt/attnpacker (vendored as submodule).
    from protein_chisel.paths import ATTNPACKER_SOURCE_DIR
    host_cache = ATTNPACKER_SOURCE_DIR / "protein_learning" / "models" / ".basis_cache"
    if host_cache.is_dir():
        shutil.copytree(host_cache, cache_scratch / ".basis_cache")

    call = (
        esmc_call(nv=True)
        .with_bind(str(pdb_path.parent))
        .with_bind(str(out_pdb_path.parent))
        .with_bind(str(resource_root.parent))
        .with_bind(str(cache_scratch / ".basis_cache"), str(src_cache))
    )

    argv = [
        "python", "-c", _ATTNPACKER_PACK_SCRIPT,
        str(resource_root),
        str(pdb_path),
        str(out_pdb_path),
        str(chunk_size),
    ]
    LOGGER.info("running AttnPacker on %s", pdb_path.name)
    try:
        t0 = time.perf_counter()
        result = call.run(argv, timeout=timeout, check=True)
        runtime = time.perf_counter() - t0

        if not out_pdb_path.is_file():
            raise RuntimeError(
                f"AttnPacker exited 0 but didn't write {out_pdb_path}. "
                f"stdout tail:\n{result.stdout[-2000:]}"
            )

        return AttnPackerPackResult(
            out_pdb_path=out_pdb_path,
            runtime_seconds=runtime,
            resource_root=resource_root,
        )
    finally:
        shutil.rmtree(cache_scratch, ignore_errors=True)


def attnpacker_score(
    pdb_path: str | Path,
    weights_dir: Optional[Path] = None,
    timeout: float = 1800.0,
) -> AttnPackerScoreResult:
    """Score rotamer plausibility = deviation between observed and AttnPacker-repacked.

    Pipeline (same as :func:`pippack_score.pippack_score`):
        1. AttnPack-repack the input.
        2. Compare per-residue chi angles + side-chain RMSD between the
           input and the repacked output via PIPPack's ``assess_packing.py``
           (we re-use that script -- it's structure-only and PDB-format
           agnostic).

    High chi MAE / low recovery = AttnPacker's environment-aware model
    disagrees with the design's chosen rotamer.
    """
    import time

    pdb_path = Path(pdb_path).resolve()
    workdir = Path(tempfile.mkdtemp(prefix="chisel_attnpacker_score_"))
    try:
        # Repack into a side-by-side dir layout that assess_packing.py
        # can consume: native/<pdb> + decoy/<pdb> with matching names.
        native_dir = workdir / "native"
        decoy_dir = workdir / "decoy"
        native_dir.mkdir()
        decoy_dir.mkdir()
        shutil.copy2(pdb_path, native_dir / pdb_path.name)
        decoy_path = decoy_dir / pdb_path.name

        t0 = time.perf_counter()
        attnpacker_pack(
            pdb_path,
            out_pdb_path=decoy_path,
            weights_dir=weights_dir,
            timeout=timeout,
        )
        # Reuse PIPPack's assess_packing.py for the per-residue comparison.
        from protein_chisel.tools.sidechain_packing_and_scoring.pippack_score import (
            _summarize_assess_payload,
        )
        from protein_chisel.paths import PIPPACK_GUEST_SOURCE_DIR
        from protein_chisel.utils.apptainer import esmc_call

        out_pkl = workdir / "assess.pkl"
        call = esmc_call(nv=False).with_bind(str(workdir))
        argv = [
            "python", str(PIPPACK_GUEST_SOURCE_DIR / "assess_packing.py"),
            str(native_dir), str(decoy_dir),
            "--out_filename", str(out_pkl.with_suffix("")),
            "--per_aatype",
        ]
        call.run(argv, timeout=timeout, check=True)
        runtime = time.perf_counter() - t0

        if not out_pkl.is_file():
            pkls = list(workdir.glob("assess*"))
            if not pkls:
                raise RuntimeError(f"assess_packing wrote no pkl under {workdir}")
            out_pkl = pkls[0]
        import pickle
        with open(out_pkl, "rb") as fh:
            payload = pickle.load(fh)

        # Re-use PIPPack's summarizer (the pickle layout is the same
        # regardless of which packer produced the decoy).
        pip_res = _summarize_assess_payload(payload, runtime_seconds=runtime)

        return AttnPackerScoreResult(
            per_residue_df=pip_res.per_residue_df,
            n_residues_scored=pip_res.n_residues_scored,
            mean_chi_mae=pip_res.mean_chi_mae,
            chi1_mae=pip_res.chi1_mae,
            chi2_mae=pip_res.chi2_mae,
            chi3_mae=pip_res.chi3_mae,
            chi4_mae=pip_res.chi4_mae,
            rotamer_recovery=pip_res.rotamer_recovery,
            mean_sidechain_rmsd=pip_res.mean_sidechain_rmsd,
            runtime_seconds=runtime,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


__all__ = [
    "AttnPackerPackResult",
    "AttnPackerScoreResult",
    "attnpacker_pack",
    "attnpacker_score",
]
