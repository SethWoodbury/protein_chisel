"""MolProbity rotalyze wrapper -- per-residue Top8000 KDE rotamer scorer.

This is the modern statistical complement to Rosetta's `fa_dun`:
- `fa_dun` scores -log p(rotamer | phi,psi) under the Shapovalov-Dunbrack
  2011 backbone-dependent rotamer library.
- `rotalyze` scores per-residue percentile in chi-space under the
  Top8000 reference dataset (Hintze et al. 2016) and classifies each
  residue as Favored / Allowed / Outlier.

The two reference datasets are independent, so the signals are
complementary -- a residue that's an outlier under both is a stronger
signal than one outlier in either alone.

`rotalyze` lives inside cctbx-base (mmtbx.validation.rotalyze), which
we install into the esmc.sif at /opt/esmc/lib/python3.12/site-packages/.
We invoke it via `apptainer exec esmc.sif python -c "..."` and parse
the stdout JSON.

Usage::

    from protein_chisel.tools.sidechain_packing_and_scoring.rotalyze_score \
        import rotalyze_score
    res = rotalyze_score("design.pdb")
    res.frac_outliers     # 0.05 == 5% of residues are rotamer outliers
    res.per_residue_df    # full per-residue table
    res.to_dict()         # filter-friendly metric dict
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("protein_chisel.rotalyze_score")


@dataclass
class RotalyzeResult:
    """Result of a MolProbity rotalyze run.

    Per-residue df columns:
        chain_id (str), resseq (int), icode (str), resname (str),
        score (float, 0-100 percent), evaluation (str: Favored/Allowed/OUTLIER),
        rotamer (str, named rotamer like 'mt-85' or '' if outlier),
        chi1, chi2, chi3, chi4 (floats; nan when undefined for residue type),
        is_outlier (bool), is_catalytic (bool, only set if catalytic_resnos passed)
    """
    per_residue_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_residues_scored: int = 0
    n_outliers: int = 0
    n_allowed: int = 0
    n_favored: int = 0
    frac_outliers: float = 0.0
    frac_favored: float = 0.0
    mean_score: float = 0.0      # mean percent score across scored residues
    outlier_resnos: list[int] = field(default_factory=list)

    def to_dict(self, prefix: str = "rotalyze__") -> dict[str, float | int]:
        return {
            f"{prefix}n_residues_scored": int(self.n_residues_scored),
            f"{prefix}n_outliers": int(self.n_outliers),
            f"{prefix}n_allowed": int(self.n_allowed),
            f"{prefix}n_favored": int(self.n_favored),
            f"{prefix}frac_outliers": float(self.frac_outliers),
            f"{prefix}frac_favored": float(self.frac_favored),
            f"{prefix}mean_score": float(self.mean_score),
        }


# The script we run inside esmc.sif. Reads PDB path from argv[1], writes
# JSON to stdout. Runs cctbx's rotalyze on the structure and emits a
# per-residue list. Single string so we can `python -c '<this>' <pdb>`.
_ROTALYZE_INNER_SCRIPT = r"""
import json
import sys
from iotbx import pdb as iotbx_pdb
from mmtbx.validation.rotalyze import rotalyze

pdb_path = sys.argv[1]
hierarchy = iotbx_pdb.input(file_name=pdb_path).construct_hierarchy()
res = rotalyze(hierarchy, outliers_only=False, quiet=True)

rows = []
for r in res.results:
    chis = list(r.chi_angles or ())
    chis = chis + [None] * (4 - len(chis))
    rows.append({
        "chain_id": (r.chain_id or "").strip(),
        "resseq": int((r.resseq or "0").strip() or 0),
        "icode": (r.icode or "").strip(),
        "resname": (r.resname or "").strip(),
        "score": float(r.score) if r.score is not None else None,
        "evaluation": r.evaluation or "",
        "rotamer": getattr(r, "rotamer_name", "") or "",
        "chi1": chis[0],
        "chi2": chis[1],
        "chi3": chis[2],
        "chi4": chis[3],
    })

print(json.dumps({"results": rows}))
"""


def rotalyze_score(
    pdb_path: str | Path,
    catalytic_resnos: Optional[set[int]] = None,
    sif: Optional[Path] = None,
    timeout: float = 120.0,
) -> RotalyzeResult:
    """Score every residue's rotamer under MolProbity Top8000 KDE.

    Args:
        pdb_path: input PDB.
        catalytic_resnos: residue numbers (resseq) to flag in the
            per-residue df via `is_catalytic=True`. Aggregates do NOT
            currently exclude catalytic by default -- the caller can
            filter the df themselves.
        sif: optional explicit sif path. Defaults to esmc.sif which
            ships cctbx-base.
        timeout: subprocess timeout (s).

    Returns:
        RotalyzeResult with per-residue df + aggregates.
    """
    from protein_chisel.paths import (
        MOLPROBITY_CHEM_DATA_DIR,
        MOLPROBITY_CHEM_DATA_GUEST,
        MOLPROBITY_MONOMERS_DIR,
    )
    from protein_chisel.utils.apptainer import esmc_call

    pdb_path = Path(pdb_path).resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    # rotalyze needs the CCP4 monomer library + the Top8000 rotarama cache.
    # We mount monomers at the same path inside as outside (so CLIBD_MON
    # is portable), and mount chem_data over the in-sif guest path so
    # libtbx.find_in_repositories('chem_data/rotarama_data') resolves.
    call = (
        esmc_call(nv=False)
        .with_bind(str(pdb_path.parent))
        .with_bind(str(MOLPROBITY_MONOMERS_DIR))
        .with_bind(str(MOLPROBITY_CHEM_DATA_DIR), str(MOLPROBITY_CHEM_DATA_GUEST))
        .with_env(CLIBD_MON=f"{MOLPROBITY_MONOMERS_DIR}/")
    )
    if sif is not None:
        # Caller wants a different sif; replace the .sif on the call.
        from dataclasses import replace
        call = replace(call, sif=Path(sif))

    LOGGER.info("rotalyze on %s", pdb_path)
    result = call.run(
        ["python", "-c", _ROTALYZE_INNER_SCRIPT, str(pdb_path)],
        capture_output=True, timeout=timeout, check=True,
    )

    # Stdout may contain warnings before the JSON line; take the last
    # JSON-looking line.
    json_line = ""
    for line in result.stdout.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_line = line
            break
    if not json_line:
        raise RuntimeError(
            f"rotalyze returned no JSON output. stdout tail:\n"
            f"{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-2000:]}"
        )
    payload = json.loads(json_line)
    rows = payload.get("results", [])

    catalytic_set = set(catalytic_resnos or ())

    cooked: list[dict] = []
    for r in rows:
        evaluation = r.get("evaluation") or ""
        is_outlier = evaluation.upper().startswith("OUTLIER")
        cooked.append({
            "chain_id": r.get("chain_id", ""),
            "resseq": int(r.get("resseq", 0)),
            "icode": r.get("icode", ""),
            "resname": r.get("resname", ""),
            "score": (
                float(r["score"]) if r.get("score") is not None else float("nan")
            ),
            "evaluation": evaluation,
            "rotamer": r.get("rotamer", ""),
            "chi1": _to_float(r.get("chi1")),
            "chi2": _to_float(r.get("chi2")),
            "chi3": _to_float(r.get("chi3")),
            "chi4": _to_float(r.get("chi4")),
            "is_outlier": is_outlier,
            "is_catalytic": int(r.get("resseq", 0)) in catalytic_set,
        })

    df = pd.DataFrame(cooked)
    if df.empty:
        return RotalyzeResult(per_residue_df=df)

    n_total = int(len(df))
    n_outliers = int(df["is_outlier"].sum())
    eval_upper = df["evaluation"].str.upper()
    n_favored = int((eval_upper == "FAVORED").sum())
    n_allowed = int((eval_upper == "ALLOWED").sum())
    scores = df["score"].dropna().to_numpy(dtype=float)

    return RotalyzeResult(
        per_residue_df=df,
        n_residues_scored=n_total,
        n_outliers=n_outliers,
        n_allowed=n_allowed,
        n_favored=n_favored,
        frac_outliers=n_outliers / n_total if n_total else 0.0,
        frac_favored=n_favored / n_total if n_total else 0.0,
        mean_score=float(scores.mean()) if scores.size else 0.0,
        outlier_resnos=df.loc[df["is_outlier"], "resseq"].astype(int).tolist(),
    )


def _to_float(v) -> float:
    """Convert a JSON value (None or number) to a Python float; nan when None."""
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


__all__ = ["RotalyzeResult", "rotalyze_score"]
