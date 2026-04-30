"""pdbe-arpeggio wrapper — exhaustive per-atom-pair contact analysis.

pdbe-arpeggio (https://github.com/PDBeurope/arpeggio) labels every atom-
pair contact within an interaction radius with a multi-label flag set:
``hbond``, ``halogen``, ``ionic``, ``aromatic``, ``hydrophobic``, ``carbonyl``,
``polar``, ``weak_polar``, ``vdw``, ``vdw_clash``, ``proximal``, plus
type-of-contact flags (intra/inter, atom-atom).

This is the most exhaustive open-source contact analyser available.
Slow on large structures; reserve for late-stage characterization on the
top hits, NOT inner loops.

Run inside esmc.sif (where pdbe-arpeggio is installed via pip).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger("protein_chisel.arpeggio_interactions")


# Standard arpeggio contact flag set (multi-label). Each contact may have any
# subset of these.
ARPEGGIO_CONTACT_FLAGS = (
    "covalent", "vdw_clash", "vdw", "proximal", "hbond", "weak_hbond",
    "xbond", "ionic", "metal_complex", "aromatic", "hydrophobic",
    "carbonyl", "polar", "weak_polar",
)


@dataclass
class ArpeggioResult:
    contacts: list[dict] = field(default_factory=list)
    n_contacts_by_type: dict[str, int] = field(default_factory=dict)
    n_total_contacts: int = 0
    raw_json_path: Optional[str] = None

    def to_dict(self, prefix: str = "arpeggio__") -> dict[str, float | int]:
        out: dict[str, float | int] = {
            f"{prefix}n_total_contacts": self.n_total_contacts,
        }
        for typ, n in self.n_contacts_by_type.items():
            out[f"{prefix}{typ}__count"] = n
        return out


def find_arpeggio_executable() -> str:
    """Find the pdbe-arpeggio CLI binary."""
    found = shutil.which("pdbe-arpeggio")
    if found:
        return found
    raise RuntimeError(
        "pdbe-arpeggio CLI not found on PATH. Install via "
        "`pip install pdbe-arpeggio` or rebuild the sif."
    )


def arpeggio_interactions(
    cif_path: str | Path,
    selection: Optional[str] = None,
    out_dir: Optional[str | Path] = None,
    keep_outputs: bool = False,
    timeout: float = 1200.0,
) -> ArpeggioResult:
    """Run pdbe-arpeggio and return the parsed contact list.

    Args:
        cif_path: input mmCIF file. **pdbe-arpeggio requires mmCIF**, not
            PDB. Convert with PyRosetta or biotite if you only have PDBs:
                from biotite.structure.io import pdb, pdbx
                pdb_struct = pdb.PDBFile.read(pdb_path).get_structure()
                pdbx.CIFFile.from_structure(pdb_struct).write(cif_path)
        selection: optional arpeggio selection string (e.g. "/A/64/").
            When None, the full structure is analyzed.
        out_dir: workspace directory; defaults to a tempdir.
        keep_outputs: if True, the workspace is preserved.
    """
    cif_path = Path(cif_path).resolve()
    if cif_path.suffix.lower() not in (".cif", ".mmcif"):
        raise ValueError(
            f"pdbe-arpeggio requires mmCIF input; got {cif_path.suffix}. "
            "Convert your PDB to mmCIF first (see docstring)."
        )
    exe = find_arpeggio_executable()

    workspace = Path(out_dir).resolve() if out_dir else Path(tempfile.mkdtemp(prefix="chisel_arpeggio_"))
    workspace.mkdir(parents=True, exist_ok=True)
    local_cif = workspace / cif_path.name
    local_cif.write_bytes(cif_path.read_bytes())

    cmd = [exe, str(local_cif), "-o", str(workspace)]
    if selection:
        cmd += ["-s", selection]
    LOGGER.info("running arpeggio: %s", " ".join(cmd))

    proc = subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pdbe-arpeggio failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout[-1000:]}\nstderr: {proc.stderr[-1000:]}"
        )

    # Outputs land in workspace as <stem>.json
    out_json = workspace / f"{local_cif.stem}.json"
    contacts: list[dict] = []
    if out_json.exists():
        contacts = _parse_arpeggio_json(out_json)
    else:
        # Some versions emit a contacts file with a different extension.
        candidates = list(workspace.glob(f"{local_cif.stem}*.json"))
        if candidates:
            contacts = _parse_arpeggio_json(candidates[0])
            out_json = candidates[0]
        else:
            LOGGER.warning("arpeggio produced no JSON in %s", workspace)

    # PDBe-Arpeggio's per-record schema: ``contact`` is the LIST of interaction
    # flags (the interaction set we actually want); ``type`` is geometry
    # metadata ("atom-atom", "atom-plane", ...) — NOT an interaction flag.
    # Don't fall back to "type".
    by_type: dict[str, int] = {}
    for c in contacts:
        types = c.get("contact") or []
        if isinstance(types, str):
            types = [types]
        for t in types:
            by_type[t] = by_type.get(t, 0) + 1

    saved_path = str(out_json) if out_json.exists() else None
    if not keep_outputs and not out_dir:
        shutil.rmtree(workspace, ignore_errors=True)
        saved_path = None

    return ArpeggioResult(
        contacts=contacts,
        n_contacts_by_type=by_type,
        n_total_contacts=len(contacts),
        raw_json_path=saved_path,
    )


def _parse_arpeggio_json(path: Path) -> list[dict]:
    """Parse arpeggio's JSON output into a flat list of contact dicts."""
    text = path.read_text()
    if not text.strip():
        return []
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Some versions wrap the list under a key like "contacts".
        for key in ("contacts", "interactions", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


__all__ = ["ARPEGGIO_CONTACT_FLAGS", "ArpeggioResult", "arpeggio_interactions"]
