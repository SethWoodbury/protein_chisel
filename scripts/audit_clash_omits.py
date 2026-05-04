"""Audit `compute_clash_prone_first_shell_omits` decisions.

For each clash-prone position picked by the v2 driver:
  1. Cb -> nearest-fixed-atom DIRECTION test (acute angle to Cb-Cα).
     Cb pointing AWAY (>90 deg) -> low clash risk.
  2. Per-AA approximate rotamer-feasibility scan: place a stub
     sidechain at each canonical chi1/chi2 and count fraction of
     rotamers with min-distance < 2.0 A to fixed sidechains.

Output: a refined omit dict + a TSV report.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from protein_chisel.structure.clash_check import SIDECHAIN_ATOM_NAMES, _read_atoms

CHAIN = "A"
DEFAULT_CATRES = (60, 64, 128, 131, 132, 157)
SEED_PDB = Path(
    "/net/scratch/aruder2/projects/PTE_i1/af3_out/filtered_i1/ref_pdbs/"
    "ZAPP_p1D1_rotP_1_ORI_11_C7_i_20_model_1__eV2_T0_20__8_1_FS269.pdb"
)
POS_TABLE = Path(
    "/net/scratch/woodbuse/iterative_design_v2_PTE_i1_20260503-212554/classify/positions.tsv"
)

# Approx per-AA reach: (chi1 grid, chi2 grid, longest stub length in A)
# CB-CG ~1.53, CG-CD ~1.5; Y/F/W/H reach ~5-6 A from Cb at far ring carbon.
# We model each AA as a single "tip" atom at the canonical reach distance,
# moved in a sphere defined by chi1 (around Cα-Cβ axis), chi2 (around Cβ-Cγ),
# with idealized 109.5 deg geometry. This is a coarse but conservative
# proxy: if the tip clashes, the heavier sidechain almost certainly does too.
AA_GEOM = {
    "Y": {"reach": 6.5, "chi1_grid": (-60, 60, 180),
          "chi2_grid": (-90, 90), "label": "TYR"},
    "F": {"reach": 5.7, "chi1_grid": (-60, 60, 180),
          "chi2_grid": (-90, 90), "label": "PHE"},
    "W": {"reach": 6.8, "chi1_grid": (-60, 60, 180),
          "chi2_grid": (-90, 90), "label": "TRP"},
    "H": {"reach": 5.0, "chi1_grid": (-60, 60, 180),
          "chi2_grid": (-90, 90), "label": "HIS"},
    "M": {"reach": 5.6, "chi1_grid": (-60, 60, 180),
          "chi2_grid": (-60, 60, 180), "label": "MET"},
    "R": {"reach": 7.0, "chi1_grid": (-60, 60, 180),
          "chi2_grid": (-60, 60, 180), "label": "ARG"},
    "K": {"reach": 6.5, "chi1_grid": (-60, 60, 180),
          "chi2_grid": (-60, 60, 180), "label": "LYS"},
    "L": {"reach": 4.0, "chi1_grid": (-60, 60, 180),
          "chi2_grid": (-60, 60, 180), "label": "LEU"},
    "I": {"reach": 4.0, "chi1_grid": (-60, 60, 180),
          "chi2_grid": (-60, 60, 180), "label": "ILE"},
}

CLASH_DIST = 2.0   # A; stricter than 1.5 to allow for atom radius


def get_n_ca_cb(atoms, chain, resno):
    """Return (N, Cα, Cβ) coordinates."""
    out = {}
    for a in atoms:
        if a["chain_id"] == chain and a["res_seq"] == resno and a["record"] == "ATOM":
            if a["atom_name"] in ("N", "CA", "CB"):
                out[a["atom_name"]] = np.array([a["x"], a["y"], a["z"]])
    return out.get("N"), out.get("CA"), out.get("CB")


def place_tip(N, CA, CB, reach, chi1_deg, chi2_deg=None):
    """Place an approximate side-chain tip atom.

    Builds a local frame at CB (z = CB-CA, y in N-CA-CB plane), rotates
    the bond vector by chi1 (around CA-CB) and chi2 (perpendicular) and
    extends by `reach`. Coarse but monotone in clash potential.
    """
    z = CB - CA
    z = z / np.linalg.norm(z)
    # x perpendicular to z, in N-CA-CB plane
    n_dir = N - CA
    x = n_dir - np.dot(n_dir, z) * z
    x = x / max(np.linalg.norm(x), 1e-9)
    y = np.cross(z, x)

    chi1 = np.radians(chi1_deg)
    chi2 = np.radians(chi2_deg) if chi2_deg is not None else 0.0
    # First bond direction (CB->CG) at ~109.5 deg from CA-CB axis,
    # rotated by chi1 around z.
    tetra = np.radians(180.0 - 109.5)
    cg_dir = (np.cos(tetra) * z
              + np.sin(tetra) * (np.cos(chi1) * x + np.sin(chi1) * y))
    cg = CB + 1.53 * cg_dir
    if chi2_deg is None:
        return cg + (reach - 1.53) * cg_dir
    # Second segment: CG->tip, rotated by chi2 around CG-CB axis.
    bond_axis = cg_dir
    # Build orthonormal frame at CG with z' = bond_axis.
    if abs(bond_axis[0]) < 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    else:
        helper = np.array([0.0, 1.0, 0.0])
    x2 = helper - np.dot(helper, bond_axis) * bond_axis
    x2 = x2 / np.linalg.norm(x2)
    y2 = np.cross(bond_axis, x2)
    tip_dir = (np.cos(tetra) * bond_axis
               + np.sin(tetra) * (np.cos(chi2) * x2 + np.sin(chi2) * y2))
    return cg + (reach - 1.53) * tip_dir


def main():
    atoms = _read_atoms(SEED_PDB)
    fixed_set = set(DEFAULT_CATRES)
    fixed_sc_atoms = []
    for a in atoms:
        if (a["chain_id"] == CHAIN and a["record"] == "ATOM"
                and a["res_seq"] in fixed_set):
            sc_names = SIDECHAIN_ATOM_NAMES.get(a["res_name"], set())
            if a["atom_name"] in sc_names:
                fixed_sc_atoms.append(np.array([a["x"], a["y"], a["z"]]))
    fixed_arr = np.array(fixed_sc_atoms)

    # Auto-detect clash-prone positions identical to the driver.
    import pandas as pd
    df = pd.read_csv(POS_TABLE, sep="\t")
    eligible_classes = {"first_shell", "buried"}
    prot = df[df["is_protein"] & (df["chain"] == CHAIN)].sort_values("resno")
    eligible = prot[prot["class"].isin(eligible_classes)
                    & ~prot["resno"].astype(int).isin(fixed_set)]
    cb_clash_prone = []
    for _, row in eligible.iterrows():
        resno = int(row["resno"])
        _, _, CB = get_n_ca_cb(atoms, CHAIN, resno)
        if CB is None:
            continue
        d = np.linalg.norm(fixed_arr - CB, axis=1)
        if d.min() < 5.0:
            cb_clash_prone.append((resno, row["name1"], row["class"],
                                   float(d.min())))
    print(f"\n=== Clash-prone positions ({len(cb_clash_prone)}): ===")
    for r, n, c, d in cb_clash_prone:
        print(f"  A{r} {n} ({c}) Cb-min={d:.2f}A")

    # 1. Cb DIRECTION test
    print("\n=== Cb direction test (angle Cb->fixed-atom-vec, Cb->Cα-vec) ===")
    direction_safe = []
    rows = []
    for resno, name1, cls, dmin in cb_clash_prone:
        N, CA, CB = get_n_ca_cb(atoms, CHAIN, resno)
        v_cb_out = CB - CA   # Cα→Cβ (sidechain points this way)
        v_cb_out /= np.linalg.norm(v_cb_out)
        # Vector from Cβ to nearest fixed atom
        d = np.linalg.norm(fixed_arr - CB, axis=1)
        nearest = fixed_arr[d.argmin()]
        v_to_fixed = nearest - CB
        v_to_fixed /= np.linalg.norm(v_to_fixed)
        cosang = float(np.dot(v_cb_out, v_to_fixed))
        ang_deg = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
        # If sidechain points AWAY (angle > 90 deg between Cα→Cβ and Cβ→fixed),
        # then sidechain extension goes AWAY from the fixed atom.
        # Equivalent: angle between sidechain growth direction and fixed-direction > 90
        sidechain_pointing_away = ang_deg > 90.0
        rows.append(dict(resno=resno, name1=name1, cls=cls, cb_min=dmin,
                         angle_cb_growth_to_fixed_deg=ang_deg,
                         pointing_away=sidechain_pointing_away))
        if sidechain_pointing_away:
            direction_safe.append(resno)
        print(f"  A{resno} {name1}: angle(Cα→Cβ , Cβ→fixed)={ang_deg:6.1f}°  "
              f"{'(POINTING AWAY -- safer)' if sidechain_pointing_away else ''}")

    # 2. Per-AA rotamer feasibility scan
    print("\n=== Rotamer-feasibility scan: %clashing rotamers per AA per pos ===")
    print(f"  (clash threshold = {CLASH_DIST} A to any fixed sidechain atom)")
    aa_results = {}
    for resno, name1, cls, dmin in cb_clash_prone:
        N, CA, CB = get_n_ca_cb(atoms, CHAIN, resno)
        per_aa = {}
        for aa, geom in AA_GEOM.items():
            n_total, n_clash = 0, 0
            chi2_grid = geom["chi2_grid"]
            for chi1 in geom["chi1_grid"]:
                for chi2 in chi2_grid:
                    tip = place_tip(N, CA, CB, geom["reach"], chi1, chi2)
                    d = np.linalg.norm(fixed_arr - tip, axis=1).min()
                    n_total += 1
                    if d < CLASH_DIST:
                        n_clash += 1
            pct = 100.0 * n_clash / n_total
            per_aa[aa] = pct
        aa_results[resno] = per_aa
        compact = " ".join(f"{aa}={pct:5.1f}%" for aa, pct in per_aa.items())
        print(f"  A{resno} {name1}: {compact}")

    # 3. Refined omit dict: forbid AA only when >=80% of rotamers clash.
    print("\n=== Refined omit dict (forbid AA only when >=80% rotamers clash) ===")
    THRESH = 80.0
    refined = {}
    for resno, per_aa in aa_results.items():
        omit = "".join(aa for aa, pct in per_aa.items() if pct >= THRESH)
        if omit:
            refined[f"{CHAIN}{resno}"] = omit
    print(json.dumps(refined, indent=2))

    # 4. Compare to current driver output.
    current_omits = {f"A{r}": "YFWHM" for r, *_ in cb_clash_prone}
    print("\n=== Comparison to current driver (forbids YFWHM at all 8) ===")
    print(f"  current: {len(current_omits)} positions x 5 AAs = "
          f"{sum(len(v) for v in current_omits.values())} (pos,AA) constraints")
    print(f"  refined: {len(refined)} positions, "
          f"{sum(len(v) for v in refined.values())} (pos,AA) constraints")

    # 5. Write to TSV
    import pandas as pd
    out_tsv = Path("/tmp/clash_audit_report.tsv")
    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
    print(f"\nReport written to {out_tsv}")
    refined_path = Path("/tmp/refined_omit.json")
    refined_path.write_text(json.dumps(refined, indent=2))
    print(f"Refined omit dict at {refined_path}")


if __name__ == "__main__":
    main()
