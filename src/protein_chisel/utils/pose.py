"""PyRosetta common operations.

This module imports PyRosetta. Code that calls these helpers must run
inside `pyrosetta.sif` (or have PyRosetta available some other way).
The package itself is importable on the host without PyRosetta — the
import of PyRosetta here is lazy.

Patterns:
- ``init_pyrosetta`` is idempotent and accepts a list of ligand `.params`
  files. Call it once per process before constructing a pose.
- ``fix_scorefxn`` is the canonical bcov hbond decomposition pattern;
  required wherever you read per-residue energies.
- ``getSASA`` is the Coventry recipe — supports per-pose, per-residue,
  and per-atom SASA with optional sidechain ignore.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Iterable, Optional, Sequence

# PyRosetta gets imported on first use; stays a module-level alias for
# downstream code (and avoids per-call imports).
_pyr = None
_rosetta = None
_INITIALIZED = False


def _ensure_pyrosetta_imported() -> None:
    global _pyr, _rosetta
    if _pyr is not None:
        return
    import pyrosetta as _pr  # noqa: F401
    import pyrosetta.rosetta as _ros  # noqa: F401

    _pyr = _pr
    _rosetta = _ros


def init_pyrosetta(
    params: Iterable[str | Path] = (),
    extra_flags: str = "",
    mute: bool = True,
    use_beta: bool = True,
    dalphaball: Optional[str] = "/net/software/lab/scripts/enzyme_design/DAlphaBall.gcc",
) -> None:
    """Idempotent PyRosetta init.

    Args:
        params: ligand .params files to load. Accepts file paths or globbed
            directories — `init_pyrosetta(params=['/some/dir'])` will glob
            `*.params` from each directory.
        extra_flags: appended to the init flag string verbatim.
        mute: silence routine Rosetta logging.
        use_beta: use beta_nov16 score function (the lab default; what
            most filters were calibrated against).
        dalphaball: path to DAlphaBall binary if available; used by holes
            and cavity scoring. Pass None to skip.
    """
    global _INITIALIZED
    _ensure_pyrosetta_imported()
    if _INITIALIZED:
        return

    flags: list[str] = []
    if mute:
        flags.append("-mute all")
    if use_beta:
        flags.append("-corrections:beta_nov16")

    params_files: list[str] = []
    for p in params:
        p = Path(p)
        if p.is_dir():
            params_files.extend(sorted(glob(str(p / "*.params"))))
        elif p.is_file():
            params_files.append(str(p))
    if params_files:
        flags.append("-extra_res_fa " + " ".join(params_files))

    if dalphaball and Path(dalphaball).exists():
        flags.append(f"-holes:dalphaball {dalphaball}")
        flags.append(f"-dalphaball {dalphaball}")

    if extra_flags:
        flags.append(extra_flags)

    _pyr.init(" ".join(flags))
    _INITIALIZED = True


def pose_from_file(pdb_path: str | Path):
    """Load a PDB into a PyRosetta Pose. init_pyrosetta() must have been called."""
    _ensure_pyrosetta_imported()
    return _pyr.pose_from_file(str(pdb_path))


# ---------------------------------------------------------------------------
# Scorefunction helpers
# ---------------------------------------------------------------------------


def get_default_scorefxn(name: str = "beta_nov16"):
    """Get a canonical scorefunction with the bcov hbond pattern applied.

    Pattern: decompose_bb_hb_into_pair_energies + (allow_double_bb=True
    by default since per-residue summations should not double-count
    bb-bb hbonds).
    """
    _ensure_pyrosetta_imported()
    sfxn = _rosetta.core.scoring.ScoreFunctionFactory.create_score_function(name)
    fix_scorefxn(sfxn, allow_double_bb=True)
    return sfxn


def fix_scorefxn(sfxn, allow_double_bb: bool = False) -> None:
    """The canonical bcov scorefxn fix-up.

    Without this, ``pose.energies().residue_total_energies()`` mixes bb
    hbond energies between paired residues in a way that makes
    per-residue summations incorrect.

    From /home/bcov/util/dump_hbset.py and many bcov scripts.

    Args:
        sfxn: a ScoreFunction.
        allow_double_bb: if True, do NOT exclude the bb-donor-acceptor
            "double bb" check — i.e., let bb hbonds count between paired
            residues. Use True when the goal is per-residue energy
            decomposition; False (the default) for canonical scoring.
    """
    _ensure_pyrosetta_imported()
    opts = sfxn.energy_method_options()
    opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    opts.hbond_options().bb_donor_acceptor_check(not allow_double_bb)
    sfxn.set_energy_method_options(opts)


# ---------------------------------------------------------------------------
# Ligand helpers
# ---------------------------------------------------------------------------


def find_ligand_seqpos(pose, exclude_virtual: bool = True) -> Optional[int]:
    """Return the seqpos of the first ligand residue, or None if none."""
    for res in pose.residues:
        if not res.is_ligand():
            continue
        if exclude_virtual and res.is_virtual_residue():
            continue
        return res.seqpos()
    return None


def get_ligand_seqposes(pose) -> list[int]:
    """All ligand residue seqposes."""
    return [
        r.seqpos() for r in pose.residues if r.is_ligand() and not r.is_virtual_residue()
    ]


# ---------------------------------------------------------------------------
# SASA — Coventry recipe (supports per-pose / per-residue / per-atom)
# ---------------------------------------------------------------------------


def getSASA(
    pose,
    resno: Optional[int | list[int]] = None,
    SASA_atoms: Optional[list[int]] = None,
    ignore_sc: bool = False,
    probe_radius: float = 1.4,
):
    """Compute SASA. Adapted from process_diffusion3 / bcov.

    Args:
        pose: PyRosetta pose.
        resno: if None, returns the surf_vol object (per-pose). If int,
            returns SASA of that residue. If list, returns sum over those
            residues. SASA_atoms restricts to specific atom indices.
        SASA_atoms: atom indices (1-based) within the residue(s).
        ignore_sc: if True, only count backbone atoms.
        probe_radius: Å; 1.4 for solvent, 2.8 for contact-region calcs
            (the latter is bcov's per_atom_sasa.py default).

    Returns:
        - surf_vol object if resno is None
        - float if resno is an int or list
    """
    _ensure_pyrosetta_imported()

    atoms = _rosetta.core.id.AtomID_Map_bool_t()
    atoms.resize(pose.size())

    for i, res in enumerate(pose.residues):
        if res.is_ligand():
            atoms.resize(i + 1, res.natoms(), True)
        else:
            atoms.resize(i + 1, res.natoms(), not ignore_sc)
            if ignore_sc:
                for n in range(1, res.natoms() + 1):
                    if res.atom_is_backbone(n) and not res.atom_is_hydrogen(n):
                        atoms[i + 1][n] = True

    surf_vol = _rosetta.core.scoring.packing.get_surf_vol(pose, atoms, probe_radius)

    if resno is None:
        return surf_vol

    def one_res_sasa(rno: int) -> float:
        s = 0.0
        natom = pose.residue(rno).natoms()
        for a in range(1, natom + 1):
            if SASA_atoms is not None and a not in SASA_atoms:
                continue
            s += surf_vol.surf(rno, a)
        return s

    if isinstance(resno, int):
        return one_res_sasa(resno)
    return sum(one_res_sasa(r) for r in resno)


def get_per_atom_sasa(pose, probe_radius: float = 2.8):
    """Per-atom SASA with bcov's 2.8 Å probe (vs 1.4 for solvent).

    Returns the surf_vol object; access values via ``surf_vol.surf(resno, atomno)``.
    """
    return getSASA(pose, resno=None, probe_radius=probe_radius)


def get_per_residue_sasa(pose, probe_radius: float = 1.4) -> dict[int, float]:
    """Per-residue SASA (heavy + sidechain), keyed by seqpos."""
    surf_vol = getSASA(pose, resno=None, probe_radius=probe_radius)
    out: dict[int, float] = {}
    for res in pose.residues:
        rno = res.seqpos()
        s = 0.0
        for a in range(1, res.natoms() + 1):
            s += surf_vol.surf(rno, a)
        out[rno] = s
    return out


# ---------------------------------------------------------------------------
# Mutation / threading
# ---------------------------------------------------------------------------


_AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def mutate_residue(pose, seqpos: int, new_aa: str) -> None:
    """Mutate a single residue in-place (single-letter or 3-letter AA)."""
    _ensure_pyrosetta_imported()
    name3 = _AA1_TO_3.get(new_aa.upper(), new_aa.upper())
    mut = _rosetta.protocols.simple_moves.MutateResidue()
    mut.set_target(seqpos)
    mut.set_res_name(name3)
    mut.apply(pose)


def thread_sequence(pose, sequence: str, skip_seqposes: Optional[Sequence[int]] = None):
    """Thread a sequence onto a pose (clone first; original untouched).

    Skips ligand residues automatically. Iterates 1-indexed seqposes; the
    Nth character of `sequence` goes to seqpos N+1.

    Returns the threaded pose.
    """
    _ensure_pyrosetta_imported()
    skip = set(skip_seqposes or ())
    out = pose.clone()
    for i, aa in enumerate(sequence):
        seqpos = i + 1
        if seqpos in skip:
            continue
        if seqpos > out.size():
            break
        if out.residue(seqpos).is_ligand():
            continue
        mutate_residue(out, seqpos, aa)
    return out


# ---------------------------------------------------------------------------
# Hbonds (canonical bcov pattern)
# ---------------------------------------------------------------------------


def get_hbond_set(pose, scorefxn=None):
    """Return an HBondSet, scoring the pose first if a scorefxn is given."""
    _ensure_pyrosetta_imported()
    if scorefxn is None:
        scorefxn = get_default_scorefxn()
    scorefxn(pose)
    hbset = _rosetta.core.scoring.hbonds.HBondSet()
    _rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, hbset)
    return hbset


def hbonds_as_dicts(pose, hbset=None) -> list[dict]:
    """Return a list of dicts describing each hbond.

    Each dict: donor_res, donor_name3, donor_atom, acceptor_res,
    acceptor_name3, acceptor_atom, energy.
    """
    _ensure_pyrosetta_imported()
    if hbset is None:
        hbset = get_hbond_set(pose)
    out: list[dict] = []
    for i in range(1, hbset.nhbonds() + 1):
        hb = hbset.hbond(i)
        d_res = hb.don_res()
        a_res = hb.acc_res()
        out.append({
            "donor_res": d_res,
            "donor_name3": pose.residue(d_res).name3(),
            "donor_atom": pose.residue(d_res).atom_name(hb.don_hatm()).strip(),
            "acceptor_res": a_res,
            "acceptor_name3": pose.residue(a_res).name3(),
            "acceptor_atom": pose.residue(a_res).atom_name(hb.acc_atm()).strip(),
            "energy": hb.energy(),
        })
    return out
