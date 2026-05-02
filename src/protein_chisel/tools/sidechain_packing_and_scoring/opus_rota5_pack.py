"""OPUS-Rota5 wrapper -- ligand-aware side-chain packing.

OPUS-Rota5 (Xu 2024, GPL-3) uses a 3D-Unet voxel encoder (which captures
the local environment INCLUDING any bound ligand) feeding a RotaFormer
transformer that predicts chi1-4. Three ensembled models per stage
(``unet3d/models/model_{1,2,3}.h5`` and ``Rota5/models/rota5_{1,2,3}.h5``).

License caveat: GPL-3 propagates downstream, so do NOT redistribute a
container that includes the OPUS-Rota5 source as part of a closed-source
binary distribution. For internal research use it's fine.

Source vendored as a git submodule under ``external/opus_rota5/``.
Runtime lives inside ``esmc.sif`` at ``/opt/opus_rota5``. The vendored
``Rota5/mkdssp/mkdssp`` ELF (built against boost 1.53 -- unusable on
Ubuntu 24.04) is dropped in the sif; the wrapper passes the cluster's
modern ``/net/software/utils/mkdssp`` via the ``$OPUS_ROTA5_MKDSSP`` env
var. Trained weights (~1.8 GB total: 3 unet + 3 rotaformer .h5 files)
live at ``/net/databases/lab/opus_rota5/opus_rota5/Rota5/`` and are
bind-mounted into the sif at runtime.

Single entry point :func:`opus_rota5_pack`. Returns the path to a
repacked PDB (full-atom, side chains rebuilt by ``mkpdb.toPDB`` from
the predicted chi angles + the input backbone).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger("protein_chisel.opus_rota5_pack")


@dataclass
class OpusRota5PackResult:
    """Result of an OPUS-Rota5 repack."""
    out_pdb_path: Path
    rota5_file: Optional[Path] = None     # raw .rota5 chi-angle output
    runtime_seconds: float = 0.0
    n_residues: int = 0


def opus_rota5_pack(
    pdb_path: str | Path,
    out_pdb_path: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    num_cpu: Optional[int] = None,
    timeout: float = 1800.0,
) -> OpusRota5PackResult:
    """Repack side chains with OPUS-Rota5.

    Args:
        pdb_path: input backbone-only or full-atom PDB. OPUS-Rota5 will
            use only the backbone atoms; side chains are rebuilt from
            the predicted chi angles.
        out_pdb_path: where to write the repacked PDB. Defaults to
            ``<input_stem>_opus_rota5.pdb`` next to the input.
        out_dir: optional dir to keep the raw ``.rota5`` chi-angle file
            next to the output PDB. If None, the chi file is deleted.
        num_cpu: how many CPU workers to use for the prep stage (DSSP +
            feature computation). Defaults to ``min(4, cpu_count)``;
            the upstream default of 56 is meant for server hosts.
        timeout: subprocess timeout (s).
    """
    import os
    import time

    from protein_chisel.paths import (
        MKDSSP_BIN,
        MKDSSP_FOR_OPUS_ROTA5,
        MOLPROBITY_LIBCIFPP_DIR,
        OPUS_ROTA5_GUEST_SOURCE_DIR,
        OPUS_ROTA5_ROTAFORMER_WEIGHTS,
        OPUS_ROTA5_UNET3D_LIBRARY_PKL,
        OPUS_ROTA5_UNET3D_WEIGHTS,
        OPUS_ROTA5_WEIGHTS_DIR,
    )
    from protein_chisel.utils.apptainer import esmc_call

    pdb_path = Path(pdb_path).resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")
    if out_pdb_path is None:
        out_pdb_path = pdb_path.with_name(f"{pdb_path.stem}_opus_rota5.pdb")
    out_pdb_path = Path(out_pdb_path).resolve()
    out_pdb_path.parent.mkdir(parents=True, exist_ok=True)

    if num_cpu is None:
        num_cpu = min(4, os.cpu_count() or 1)

    workdir = Path(tempfile.mkdtemp(prefix="chisel_opus_rota5_"))
    try:
        # OPUS-Rota5 reads .bb files (backbone-only PDB-format). mkdssp
        # 4.5.5 chokes on non-canonical HETATM ligands (e.g. YYE) when
        # the libcifpp components.cif isn't available, so we strip
        # HETATM records and any non-standard residues before staging
        # -- OPUS-Rota5 only needs the protein backbone anyway.
        input_dir = workdir / "input"
        tmp_dir = workdir / "tmp_files"
        out_pred_dir = workdir / "predictions"
        for d in (input_dir, tmp_dir, out_pred_dir):
            d.mkdir()

        # mkdssp 4.5.5 autodetects input format from filename extension --
        # ".bb" sends it down the mmCIF path, which fails. Use ".pdb" to
        # force PDB-format input parsing. The downstream OPUS-Rota5 code
        # uses splitext-style stem naming, so ".pdb" is safe.
        bb_path = input_dir / f"{pdb_path.stem}.pdb"
        _write_protein_only_pdb(pdb_path, bb_path)

        # bb_list: one absolute path per line.
        bb_list = workdir / "bb_list"
        bb_list.write_text(f"{bb_path}\n")

        # Bind weights over the empty model dirs in /opt/opus_rota5, plus
        # the cluster mkdssp + workdir + input.
        call = (
            esmc_call(nv=True)
            .with_bind(str(workdir))
            .with_bind(str(input_dir))
            # mkdssp 4.5.5 (modern, dynamically linked to system libs)
            .with_bind(str(MKDSSP_BIN.parent))
            # mkdssp_for_rota5.sh wrapper (reformats mkdssp 4.5.5 output
            # to the older DSSP layout that OPUS-Rota5's mk_ss parser
            # expects).
            .with_bind(str(MKDSSP_FOR_OPUS_ROTA5.parent))
            # 3D-UNet weights (3 models)
            .with_bind(
                str(OPUS_ROTA5_UNET3D_WEIGHTS),
                str(OPUS_ROTA5_GUEST_SOURCE_DIR / "Rota5" / "unet3d" / "models"),
            )
            # library.pkl is only in the standalone (not in the github source);
            # bind as a single-file overlay so unet.py:_load_library finds it.
            .with_bind(
                str(OPUS_ROTA5_UNET3D_LIBRARY_PKL),
                str(OPUS_ROTA5_GUEST_SOURCE_DIR / "Rota5" / "unet3d" / "library.pkl"),
            )
            # RotaFormer weights (3 models)
            .with_bind(
                str(OPUS_ROTA5_ROTAFORMER_WEIGHTS),
                str(OPUS_ROTA5_GUEST_SOURCE_DIR / "Rota5" / "models"),
            )
            # libcifpp data dir (needed by mkdssp 4.5.5 to validate
            # residue compounds against the CCD components.cif).
            .with_bind(str(MOLPROBITY_LIBCIFPP_DIR))
            .with_env(
                # Use the mkdssp_for_rota5.sh wrapper -- mk_ss expects
                # the older DSSP layout (3-token rows with SS code at
                # index [2]); the wrapper post-processes mkdssp 4.5.5
                # output to fit that contract.
                OPUS_ROTA5_MKDSSP=str(MKDSSP_FOR_OPUS_ROTA5),
                OPUS_ROTA5_BB_LIST=str(bb_list),
                OPUS_ROTA5_NUM_CPU=str(num_cpu),
                OPUS_ROTA5_TMP_DIR=str(tmp_dir),
                OPUS_ROTA5_OUT_DIR=str(out_pred_dir),
                # TF 2.16+ defaults to Keras 3; the upstream .h5 weights
                # need legacy Keras to load.
                TF_USE_LEGACY_KERAS="1",
                TF_CPP_MIN_LOG_LEVEL="2",  # silence INFO + WARN; show ERROR
                LIBCIFPP_DATA_DIR=str(MOLPROBITY_LIBCIFPP_DIR),
            )
        )

        # Run the patched run_opus_rota5.py. Two things to set up:
        #   1. cd /opt/opus_rota5  -- the script uses relative paths like
        #      ./Rota5/unet3d/models/model_1.h5 to load weights.
        #   2. Stage libdevice.10.bc into the layout XLA expects
        #      (<dir>/nvvm/libdevice/libdevice.10.bc). The cuda:12.8.1-runtime
        #      base doesn't ship the CUDA toolkit, but Triton does -- we
        #      symlink from there into a /tmp staging dir and tell XLA via
        #      XLA_FLAGS=--xla_gpu_cuda_data_dir.
        triton_libdevice = (
            "/opt/esmc/lib/python3.12/site-packages/triton/backends/nvidia"
            "/lib/libdevice.10.bc"
        )
        bash = (
            f"set -e && "
            f"mkdir -p /tmp/xla_cuda/nvvm/libdevice && "
            f"ln -sf {triton_libdevice} /tmp/xla_cuda/nvvm/libdevice/libdevice.10.bc && "
            f"export XLA_FLAGS='--xla_gpu_cuda_data_dir=/tmp/xla_cuda' && "
            f"cd {OPUS_ROTA5_GUEST_SOURCE_DIR} && "
            f"python run_opus_rota5.py"
        )
        argv = ["bash", "-c", bash]
        LOGGER.info("running OPUS-Rota5 on %s", pdb_path.name)
        t0 = time.perf_counter()
        result = call.run(argv, timeout=timeout, check=True)
        runtime = time.perf_counter() - t0

        # Find the output PDB. run_opus_rota5.py writes <stem>.pdb (or
        # <stem>_rota5.pdb) into out_pred_dir, plus a <stem>.rota5 file
        # with the per-residue chi predictions.
        produced_pdbs = sorted(out_pred_dir.glob("*.pdb"))
        if not produced_pdbs:
            raise RuntimeError(
                f"OPUS-Rota5 ran (exit {result.returncode}) but produced no PDB "
                f"under {out_pred_dir}. stdout tail:\n{result.stdout[-2000:]}"
            )
        produced = produced_pdbs[0]
        shutil.copy2(produced, out_pdb_path)

        rota5_path: Optional[Path] = None
        if out_dir is not None:
            out_dir = Path(out_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            for r5 in out_pred_dir.glob("*.rota5"):
                staged = out_dir / r5.name
                shutil.copy2(r5, staged)
                rota5_path = staged

        # n_residues from the output PDB (CA atoms)
        n_residues = sum(
            1 for line in out_pdb_path.read_text().splitlines()
            if line.startswith("ATOM  ") and line[12:16].strip() == "CA"
        )
        return OpusRota5PackResult(
            out_pdb_path=out_pdb_path,
            rota5_file=rota5_path,
            runtime_seconds=runtime,
            n_residues=n_residues,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


_STANDARD_AA_3 = {
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
}


def _write_protein_only_pdb(src: Path, dst: Path) -> None:
    """Strip HETATMs + non-standard residues from a PDB.

    mkdssp 4.5.5 (the modern dynamically-linked binary at
    /net/software/utils/mkdssp) consults a CCD components.cif at parse
    time to validate non-standard residues like 'YYE' and bombs out
    when it can't find them. OPUS-Rota5 only needs the protein backbone,
    so we drop everything except canonical amino acid ATOM records.

    We also prepend a synthetic ``HEADER`` line so mkdssp's libcifpp
    auto-format-detection picks PDB rather than mmCIF (it sniffs file
    contents, not just the .pdb extension).
    """
    with open(src, "r") as fh_in, open(dst, "w") as fh_out:
        fh_out.write(
            "HEADER    PROTEIN                                 "
            "01-JAN-00   XXXX              \n"
        )
        for line in fh_in:
            if line.startswith("ATOM  "):
                resname = line[17:20].strip()
                if resname in _STANDARD_AA_3:
                    fh_out.write(line)
            elif line.startswith("TER ") or line.startswith("TER\n"):
                fh_out.write(line)
            elif line.startswith("END"):
                fh_out.write(line)
            # Drop HETATM, REMARK, MODEL, etc. -- mkdssp doesn't need them
            # and CONECT/REMARK lines can confuse the parser.


__all__ = ["OpusRota5PackResult", "opus_rota5_pack"]
