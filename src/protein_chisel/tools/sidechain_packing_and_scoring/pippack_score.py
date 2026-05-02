"""PIPPack wrapper -- side-chain repacking and rotamer scoring.

PIPPack (Randolph & Kuhlman 2024, MIT) is a graph-NN side-chain packer
based on Invariant Point Message Passing. Source vendored as a git
submodule under ``external/pippack/``; runtime lives inside ``esmc.sif``
at ``/opt/pippack``. Trained weights (~100 MB, 3 ensembled models) live
at ``/net/databases/lab/pippack/model_weights/``.

Two entry points:

- :func:`pippack_pack` runs ``inference.py`` to repack a PDB. Returns
  the path to a new PDB with repacked side chains.
- :func:`pippack_score` repacks then runs ``assess_packing.py`` against
  the input as the "native" reference, returning per-residue chi MAE,
  rotamer recovery, and side-chain RMSD. Use it as a "deviation from
  prediction" rotamer plausibility score: residues with high chi MAE
  or low recovery are ones where PIPPack's environment-aware model
  thinks the rotamer should be different from what the design picked.

Usage::

    from protein_chisel.tools.sidechain_packing_and_scoring.pippack_score \
        import pippack_pack, pippack_score

    repacked = pippack_pack("design.pdb", out_dir="/tmp/run")
    res = pippack_score("design.pdb")
    res.mean_chi_mae        # mean per-residue chi MAE in degrees
    res.rotamer_recovery    # fraction of residues PIPPack got "right" (<20deg chi MAE)
    res.to_dict()
"""

from __future__ import annotations

import logging
import pickle
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger("protein_chisel.pippack_score")


@dataclass
class PIPPackPackResult:
    """Result of a PIPPack repack run."""
    out_pdb_path: Path
    runtime_seconds: float = 0.0
    model_name: str = "pippack_model_1"


@dataclass
class PIPPackScoreResult:
    """Result of running PIPPack and comparing to the input ("native") PDB.

    Per-residue df columns (when available): resseq, resname,
    chi1_mae, chi2_mae, chi3_mae, chi4_mae, sidechain_rmsd, recovered.
    The aggregates are the headline numbers for filter wiring.
    """
    per_residue_df: object = None  # pandas.DataFrame; lazy import for non-pandas envs
    n_residues_scored: int = 0
    mean_chi_mae: float = 0.0          # mean over chi1-4 across residues, degrees
    chi1_mae: float = 0.0
    chi2_mae: float = 0.0
    chi3_mae: float = 0.0
    chi4_mae: float = 0.0
    rotamer_recovery: float = 0.0      # frac residues where all chi MAEs < 20deg
    mean_sidechain_rmsd: float = 0.0   # angstroms
    n_clashes: int = 0
    runtime_seconds: float = 0.0

    def to_dict(self, prefix: str = "pippack__") -> dict[str, float | int]:
        return {
            f"{prefix}n_residues_scored": int(self.n_residues_scored),
            f"{prefix}mean_chi_mae": float(self.mean_chi_mae),
            f"{prefix}chi1_mae": float(self.chi1_mae),
            f"{prefix}chi2_mae": float(self.chi2_mae),
            f"{prefix}chi3_mae": float(self.chi3_mae),
            f"{prefix}chi4_mae": float(self.chi4_mae),
            f"{prefix}rotamer_recovery": float(self.rotamer_recovery),
            f"{prefix}mean_sidechain_rmsd": float(self.mean_sidechain_rmsd),
            f"{prefix}n_clashes": int(self.n_clashes),
        }


def pippack_pack(
    pdb_path: str | Path,
    out_dir: str | Path,
    model_name: str = "pippack_model_1",
    weights_dir: Optional[Path] = None,
    n_recycle: int = 3,
    temperature: float = 0.0,
    use_resample: bool = False,
    seed: int = 42,
    timeout: float = 600.0,
) -> PIPPackPackResult:
    """Repack side chains on a PDB with PIPPack.

    Args:
        pdb_path: input PDB.
        out_dir: directory for the repacked output PDB. Will be created.
        model_name: which trained checkpoint to use. Three are bundled:
            ``pippack_model_1``, ``..._2``, ``..._3``. Use
            ``ensembled_inference.py`` separately for ensemble averaging.
        weights_dir: where to find ``<model_name>_ckpt.pt``. Defaults to
            ``/net/databases/lab/pippack/model_weights``.
        n_recycle: number of recycle iterations (PIPPack default 3).
        temperature: 0 = greedy; >0 enables sampling.
        use_resample: whether to enable PIPPack's resample loop.
        seed: rng seed.

    Returns:
        PIPPackPackResult with the path to the repacked PDB.
    """
    import time

    from protein_chisel.paths import (
        PIPPACK_GUEST_SOURCE_DIR,
        PIPPACK_WEIGHTS_DIR,
    )
    from protein_chisel.utils.apptainer import esmc_call

    pdb_path = Path(pdb_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = Path(weights_dir) if weights_dir else PIPPACK_WEIGHTS_DIR

    # PIPPack's inference.py uses Hydra; CLI args are dotted overrides.
    # `pdb_path` is a folder, not a file -- PIPPack scans it for .pdb files.
    # Stage the input in a temp folder so we have a clean invocation.
    workdir = Path(tempfile.mkdtemp(prefix="chisel_pippack_pack_"))
    try:
        staged = workdir / "input"
        staged.mkdir()
        shutil.copy2(pdb_path, staged / pdb_path.name)
        outdir_in_workdir = workdir / "out"

        call = (
            esmc_call(nv=True)
            .with_bind(str(workdir))
            .with_bind(str(weights))
        )

        # PIPPack uses package-relative imports (`from data.protein import
        # ...`), so we must run with cwd=/opt/pippack inside the sif.
        # We accomplish that by using `python -c` and prepending sys.path.
        runner_argv = [
            "python", "-c",
            (
                "import sys, os; "
                f"sys.path.insert(0, '{PIPPACK_GUEST_SOURCE_DIR}'); "
                f"os.chdir('{PIPPACK_GUEST_SOURCE_DIR}'); "
                "import inference;"  # registers hydra config
            ),
        ]
        # Hmm -- actually inference.py is hydra-decorated; calling it as
        # an `import` won't do anything. The clean way is to call its
        # python entry as a script. We do that with a small driver:
        runner_argv = [
            "python", str(PIPPACK_GUEST_SOURCE_DIR / "inference.py"),
            f"inference.weights_path={weights}",
            f"inference.pdb_path={staged}",
            f"inference.output_dir={outdir_in_workdir}",
            f"inference.model_name={model_name}",
            f"inference.n_recycle={n_recycle}",
            f"inference.temperature={temperature}",
            f"inference.use_resample={str(use_resample).lower()}",
            f"inference.seed={seed}",
        ]

        # PIPPack's hydra config expects cwd to be /opt/pippack so that
        # config_path resolves. We invoke python with the absolute script
        # path; hydra's config_path is relative to that script.
        LOGGER.info("running PIPPack inference: %s", " ".join(runner_argv))
        t0 = time.perf_counter()
        result = call.run(runner_argv, timeout=timeout, check=True)
        runtime = time.perf_counter() - t0

        # Output PDB path: PIPPack writes <output_dir>/<input_stem>.pdb
        produced = outdir_in_workdir / pdb_path.name
        if not produced.is_file():
            # Fallback: scan the output dir for any .pdb
            pdbs = list(outdir_in_workdir.rglob("*.pdb"))
            if not pdbs:
                raise RuntimeError(
                    f"PIPPack ran (exit {result.returncode}) but no PDB was "
                    f"written under {outdir_in_workdir}. "
                    f"stdout tail:\n{result.stdout[-2000:]}"
                )
            produced = pdbs[0]

        final = out_dir / pdb_path.with_suffix(".pippack.pdb").name
        shutil.copy2(produced, final)
        return PIPPackPackResult(
            out_pdb_path=final,
            runtime_seconds=runtime,
            model_name=model_name,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def pippack_score(
    pdb_path: str | Path,
    model_name: str = "pippack_model_1",
    weights_dir: Optional[Path] = None,
    timeout: float = 600.0,
) -> PIPPackScoreResult:
    """Score rotamer plausibility = deviation between observed and PIPPack-repacked.

    Pipeline:
        1. Run :func:`pippack_pack` on ``pdb_path`` -> repacked PDB.
        2. Run PIPPack's ``assess_packing.py`` with native=input,
           decoy=repacked. This script emits per-residue chi MAE,
           rotamer recovery (boolean per-residue 'within 20 deg'), and
           side-chain RMSD.
        3. Aggregate into headline metrics.

    Residues with high chi MAE or no recovery flag are ones where the
    learned packer disagrees with the observed rotamer.
    """
    import time

    from protein_chisel.paths import PIPPACK_GUEST_SOURCE_DIR
    from protein_chisel.utils.apptainer import esmc_call

    pdb_path = Path(pdb_path).resolve()
    workdir = Path(tempfile.mkdtemp(prefix="chisel_pippack_score_"))
    try:
        # Stage the native PDB and repack.
        native_dir = workdir / "native"
        decoy_dir = workdir / "decoy"
        native_dir.mkdir()
        decoy_dir.mkdir()
        shutil.copy2(pdb_path, native_dir / pdb_path.name)

        pack_res = pippack_pack(
            pdb_path,
            out_dir=decoy_dir,
            model_name=model_name,
            weights_dir=weights_dir,
            timeout=timeout,
        )
        # assess_packing.py expects matching filenames in native/ and
        # decoy/. Rename the repacked PDB to match the native filename.
        decoy_pdb = decoy_dir / pdb_path.name
        if pack_res.out_pdb_path != decoy_pdb:
            shutil.move(pack_res.out_pdb_path, decoy_pdb)

        # assess_packing.py ALWAYS writes its output pickle into
        # `decoy_dir/<out_filename>.pkl` (positional) -- the path we pass
        # via `--out_filename` is just the basename, not a full path.
        # Use a basename here so `decoy_dir/packing_stats.pkl` is the
        # expected output location.
        out_basename = "chisel_assess"
        call = (
            esmc_call(nv=False)  # assess_packing is CPU-only
            .with_bind(str(workdir))
        )
        argv = [
            "python", str(PIPPACK_GUEST_SOURCE_DIR / "assess_packing.py"),
            str(native_dir), str(decoy_dir),
            "--out_filename", out_basename,
        ]
        LOGGER.info("running assess_packing: %s", " ".join(argv))
        t0 = time.perf_counter()
        call.run(argv, timeout=timeout, check=True)
        runtime = time.perf_counter() - t0 + pack_res.runtime_seconds

        out_pkl = decoy_dir / f"{out_basename}.pkl"
        if not out_pkl.is_file():
            pkls = list(decoy_dir.glob("*.pkl"))
            if not pkls:
                raise RuntimeError(
                    f"assess_packing wrote no pkl under {decoy_dir}"
                )
            out_pkl = pkls[0]

        with open(out_pkl, "rb") as fh:
            payload = pickle.load(fh)

        return _summarize_assess_payload(payload, runtime_seconds=runtime)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _summarize_assess_payload(
    payload, runtime_seconds: float = 0.0,
) -> PIPPackScoreResult:
    """Convert assess_packing.py's pickle into a PIPPackScoreResult.

    Without `--per_aatype`, assess_packing.py's `summarize()` produces::

        {
          "all": {           # whole-protein bucket
              "chi_mae":      np.array([4]),    # per-chi, in degrees
              "mean_rr":      float,            # rotamer recovery (frac)
              "mean_rmsd":    float,            # sidechain RMSD (Angstrom)
              "num_residues": int,
              "num_rotamers": int,
              "num_chi":      np.array([4]),
              "num_sc":       int,
          },
          "core":    {... same shape, residues with high centrality ...},
          "surface": {... same shape, residues with low centrality ...},
          "clash_info": {tol: {"num_clashes": float, "loss_avg": float}},
          "unclosed_pro_pct": float,
        }

    We surface the "all" bucket as the headline, plus clash counts.
    """
    import numpy as np
    import pandas as pd

    if not isinstance(payload, dict):
        return PIPPackScoreResult(
            per_residue_df=pd.DataFrame(), runtime_seconds=runtime_seconds,
        )

    # The "all" key is always present (per_aatype=False puts it at top
    # level; per_aatype=True puts it inside each aatype dict, but we
    # don't pass --per_aatype so it's at top level).
    all_bucket = payload.get("all")
    if not isinstance(all_bucket, dict):
        # Not the layout we expect; bail with empty result.
        return PIPPackScoreResult(
            per_residue_df=pd.DataFrame(), runtime_seconds=runtime_seconds,
        )

    chi_mae_per_chi = np.asarray(all_bucket.get("chi_mae", []), dtype=float)
    if chi_mae_per_chi.size != 4:
        chi_mae_per_chi = np.full(4, float("nan"))

    n_residues = int(all_bucket.get("num_residues", 0) or 0)
    rotamer_recovery = float(all_bucket.get("mean_rr", 0.0) or 0.0)
    mean_rmsd = float(all_bucket.get("mean_rmsd", 0.0) or 0.0)

    # Clash totals across all tolerances. Take the median tolerance (0.9)
    # for the headline n_clashes; loss_avg is also useful but not surfaced.
    clash_info = payload.get("clash_info", {}) or {}
    if 0.9 in clash_info:
        n_clashes = int(clash_info[0.9].get("num_clashes", 0) or 0)
    elif clash_info:
        # Pick the first entry deterministically.
        first = next(iter(clash_info.values()))
        n_clashes = int(first.get("num_clashes", 0) or 0)
    else:
        n_clashes = 0

    def _safe(v):
        v = float(v) if v is not None else 0.0
        return 0.0 if (v != v) else v  # NaN -> 0

    return PIPPackScoreResult(
        # No per-residue df from assess_packing's aggregated summary.
        per_residue_df=pd.DataFrame(),
        n_residues_scored=n_residues,
        mean_chi_mae=_safe(np.nanmean(chi_mae_per_chi)),
        chi1_mae=_safe(chi_mae_per_chi[0]),
        chi2_mae=_safe(chi_mae_per_chi[1]),
        chi3_mae=_safe(chi_mae_per_chi[2]),
        chi4_mae=_safe(chi_mae_per_chi[3]),
        rotamer_recovery=rotamer_recovery,
        mean_sidechain_rmsd=mean_rmsd,
        n_clashes=n_clashes,
        runtime_seconds=runtime_seconds,
    )


__all__ = [
    "PIPPackPackResult",
    "PIPPackScoreResult",
    "pippack_pack",
    "pippack_score",
]
