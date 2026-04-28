"""DSSP per-residue secondary structure + summary metrics.

`secondary_structure` returns the per-residue DSSP labels (full and reduced
alphabet). `ss_summary` rolls those up into scalar metrics like
loop_frac, longest_helix, longest_sheet, helix_count, sheet_count, and
loop_at_motif (catalytic residue in a loop region).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SecondaryStructureResult:
    ss_full: dict[int, str]      # full DSSP alphabet H E L T S B G I -
    ss_reduced: dict[int, str]   # reduced alphabet H | E | L


def secondary_structure(
    pdb_path: str | Path,
    params: list[str | Path] = (),
) -> SecondaryStructureResult:
    """Per-residue DSSP labels."""
    from protein_chisel.utils.pose import init_pyrosetta, pose_from_file
    from protein_chisel.utils.geometry import dssp

    init_pyrosetta(params=list(params))
    pose = pose_from_file(pdb_path)

    ss_full = dssp(pose, reduced=False)
    ss_red = dssp(pose, reduced=True)
    return SecondaryStructureResult(ss_full=ss_full, ss_reduced=ss_red)


@dataclass
class SSSummaryResult:
    n_protein: int
    helix_frac: float
    sheet_frac: float
    loop_frac: float
    longest_helix: int
    longest_sheet: int
    longest_loop: int
    helix_count: int
    sheet_count: int
    loop_at_motif: bool      # any catalytic residue in a loop
    catalytic_in_helix: int  # count of catalytic residues in helix
    catalytic_in_sheet: int
    catalytic_in_loop: int

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "ss__n_protein": self.n_protein,
            "ss__helix_frac": self.helix_frac,
            "ss__sheet_frac": self.sheet_frac,
            "ss__loop_frac": self.loop_frac,
            "ss__longest_helix": self.longest_helix,
            "ss__longest_sheet": self.longest_sheet,
            "ss__longest_loop": self.longest_loop,
            "ss__helix_count": self.helix_count,
            "ss__sheet_count": self.sheet_count,
            "ss__loop_at_motif": self.loop_at_motif,
            "ss__catalytic_in_helix": self.catalytic_in_helix,
            "ss__catalytic_in_sheet": self.catalytic_in_sheet,
            "ss__catalytic_in_loop": self.catalytic_in_loop,
        }


def ss_summary(
    pdb_path: str | Path,
    params: list[str | Path] = (),
    catalytic_resnos: Optional[set[int]] = None,
) -> SSSummaryResult:
    """Summary stats over DSSP labels."""
    res = secondary_structure(pdb_path=pdb_path, params=params)
    ss_red = res.ss_reduced
    if not ss_red:
        return SSSummaryResult(0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, False, 0, 0, 0)

    seqposes_sorted = sorted(ss_red.keys())
    string = "".join(ss_red[s] for s in seqposes_sorted)
    n = len(string)

    helix_frac = string.count("H") / n
    sheet_frac = string.count("E") / n
    loop_frac = string.count("L") / n

    helix_count, helix_longest = _count_runs(string, "H")
    sheet_count, sheet_longest = _count_runs(string, "E")
    _, loop_longest = _count_runs(string, "L")

    cat_in_helix = cat_in_sheet = cat_in_loop = 0
    if catalytic_resnos:
        for r in catalytic_resnos:
            label = ss_red.get(r)
            if label == "H":
                cat_in_helix += 1
            elif label == "E":
                cat_in_sheet += 1
            elif label == "L":
                cat_in_loop += 1

    return SSSummaryResult(
        n_protein=n,
        helix_frac=helix_frac,
        sheet_frac=sheet_frac,
        loop_frac=loop_frac,
        longest_helix=helix_longest,
        longest_sheet=sheet_longest,
        longest_loop=loop_longest,
        helix_count=helix_count,
        sheet_count=sheet_count,
        loop_at_motif=cat_in_loop > 0,
        catalytic_in_helix=cat_in_helix,
        catalytic_in_sheet=cat_in_sheet,
        catalytic_in_loop=cat_in_loop,
    )


def _count_runs(s: str, ch: str) -> tuple[int, int]:
    """Return (count_of_runs, longest_run_length) for a single character."""
    count = 0
    longest = 0
    cur = 0
    for c in s:
        if c == ch:
            cur += 1
            if cur == 1:
                count += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    return count, longest


__all__ = ["SecondaryStructureResult", "SSSummaryResult", "secondary_structure", "ss_summary"]
