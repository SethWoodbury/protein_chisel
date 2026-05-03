"""Comprehensive RosettaScripts metrics + ligand h-bonds.

Modernized port of ``metrics_and_hbond_rosetta_seth_no_RELAX_V2.py``
(no relax — preserves the input pose). Drives a single XML protocol
through ``rosetta_scripts.SingleoutputRosettaScriptsTask`` and pulls
the resulting filter/simple-metric values off the packed_pose. After
the XML, computes ligand-atom SASA via the Coventry recipe and
enumerates h-bonds involving the ligand from ``pose.get_hbonds()``.

Spawns ``pyrosetta.sif`` from the host via :func:`pyrosetta_call`. The
in-container python builds the XML, runs it, and emits a JSON blob to
stdout that this wrapper deserializes into :class:`RosettaMetricsResult`.

The simple metric ``SAP_score`` here is the Lauer SapScoreMetric (the
real Rosetta SAP), distinct from the freesasa proxy used in the iterative
driver.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


LOGGER = logging.getLogger("protein_chisel.rosetta_metrics")


# Sentinel that the inner script wraps around the JSON payload so we can
# pluck it cleanly out of mixed stdout (Rosetta is chatty).
_JSON_BEGIN = "<<<ROSETTA_METRICS_JSON_BEGIN>>>"
_JSON_END = "<<<ROSETTA_METRICS_JSON_END>>>"


@dataclass
class RosettaMetricsConfig:
    """Knobs for the metrics XML + post-XML python.

    ``apbs_path`` is bound into the container at ``/usr/local/bin/apbs``
    when ``include_ec`` is True (ElectrostaticComplementarityMetric
    shells out to ``apbs``). The default points at PyMOL-2's bundled
    APBS, which is the canonical lab cluster path.
    """
    scorefxn: str = "beta"
    dalphaball: str = "/net/software/lab/scripts/enzyme_design/DAlphaBall.gcc"
    ligand_chain: str = "B"
    include_sap: bool = True
    include_ec: bool = True
    apbs_path: str = "/net/software/pymol-2/bin/apbs"
    timeout: float = 1800.0


@dataclass
class RosettaMetricsResult:
    """Flat metrics returned by the XML + post-XML python.

    Numeric filter / simple-metric fields are populated from the score
    map of the packed_pose. ``per_atom_hbonds`` is a per-key-atom dict
    of SimpleHbondsToAtomFilter counts. ``ligand_hbonds_table`` is the
    list of dicts (donor_res, donor_atom, acceptor_res, acceptor_atom,
    energy) for h-bonds involving the ligand seqpos.
    """
    # XML filter scalars
    contact_molecular_surface: float = float("nan")
    ddg: float = float("nan")
    ligand_interface_energy: float = float("nan")
    total_residues_in_design_plus_ligand: float = float("nan")
    hydrophobic_residues_in_design: float = float("nan")
    aliphatic_residues_in_design: float = float("nan")
    net_charge_in_design_NOT_w_HIS: float = float("nan")
    dSasa_fraction: float = float("nan")
    number_DSSP_helices_in_design: float = float("nan")
    number_DSSP_sheets_in_design: float = float("nan")
    number_DSSP_loops_in_design: float = float("nan")
    holes_in_design_lower_is_better: float = float("nan")
    interface_holes_at_ligand: float = float("nan")
    num_residues_at_ligand_interface: float = float("nan")
    shape_complementarity_interface_area: float = float("nan")
    shape_complementarity_median_distance_at_interface: float = float("nan")
    hydrophobic_exposure_sasa_in_design: float = float("nan")
    sasa_ligand_interface: float = float("nan")
    total_pose_sasa: float = float("nan")
    bad_torsion_preproline: float = float("nan")
    longest_cont_polar_seg: float = float("nan")
    longest_cont_apolar_seg: float = float("nan")

    # Simple metrics (XML <SIMPLE_METRICS>)
    total_rosetta_energy_metric: float = float("nan")
    secondary_structure: str = ""
    secondary_structure_DSSP_reduced_alphabet: str = ""
    SAP_score: float = float("nan")
    electrostatic_complementarity: float = float("nan")

    # Per-key-atom h-bond counts via SimpleHbondsToAtomFilter
    per_atom_hbonds: dict[str, float] = field(default_factory=dict)

    # Custom python-side metrics
    ligand_seqpos: int = 0
    ligand_exposed_atoms_sasa: float = float("nan")
    n_hbonds_to_ligand: int = 0
    ligand_hbonds_table: list[dict] = field(default_factory=list)

    def to_dict(self, prefix: str = "rosetta__") -> dict[str, Any]:
        out: dict[str, Any] = {
            f"{prefix}contact_molecular_surface": self.contact_molecular_surface,
            f"{prefix}ddg": self.ddg,
            f"{prefix}ligand_interface_energy": self.ligand_interface_energy,
            f"{prefix}total_residues_in_design_plus_ligand": self.total_residues_in_design_plus_ligand,
            f"{prefix}hydrophobic_residues_in_design": self.hydrophobic_residues_in_design,
            f"{prefix}aliphatic_residues_in_design": self.aliphatic_residues_in_design,
            f"{prefix}net_charge_in_design_NOT_w_HIS": self.net_charge_in_design_NOT_w_HIS,
            f"{prefix}dSasa_fraction": self.dSasa_fraction,
            f"{prefix}number_DSSP_helices_in_design": self.number_DSSP_helices_in_design,
            f"{prefix}number_DSSP_sheets_in_design": self.number_DSSP_sheets_in_design,
            f"{prefix}number_DSSP_loops_in_design": self.number_DSSP_loops_in_design,
            f"{prefix}holes_in_design_lower_is_better": self.holes_in_design_lower_is_better,
            f"{prefix}interface_holes_at_ligand": self.interface_holes_at_ligand,
            f"{prefix}num_residues_at_ligand_interface": self.num_residues_at_ligand_interface,
            f"{prefix}shape_complementarity_interface_area": self.shape_complementarity_interface_area,
            f"{prefix}shape_complementarity_median_distance_at_interface": self.shape_complementarity_median_distance_at_interface,
            f"{prefix}hydrophobic_exposure_sasa_in_design": self.hydrophobic_exposure_sasa_in_design,
            f"{prefix}sasa_ligand_interface": self.sasa_ligand_interface,
            f"{prefix}total_pose_sasa": self.total_pose_sasa,
            f"{prefix}bad_torsion_preproline": self.bad_torsion_preproline,
            f"{prefix}longest_cont_polar_seg": self.longest_cont_polar_seg,
            f"{prefix}longest_cont_apolar_seg": self.longest_cont_apolar_seg,
            f"{prefix}total_rosetta_energy_metric": self.total_rosetta_energy_metric,
            f"{prefix}secondary_structure": self.secondary_structure,
            f"{prefix}secondary_structure_DSSP_reduced_alphabet": self.secondary_structure_DSSP_reduced_alphabet,
            f"{prefix}SAP_score": self.SAP_score,
            f"{prefix}electrostatic_complementarity": self.electrostatic_complementarity,
            f"{prefix}ligand_seqpos": self.ligand_seqpos,
            f"{prefix}ligand_exposed_atoms_sasa": self.ligand_exposed_atoms_sasa,
            f"{prefix}n_hbonds_to_ligand": self.n_hbonds_to_ligand,
        }
        for atom, n in self.per_atom_hbonds.items():
            out[f"{prefix}{atom}_hbond"] = n
        return out


# ---------------------------------------------------------------------------
# Inner script: runs inside pyrosetta.sif. Reads a single JSON arg from argv
# describing inputs + config; emits a JSON blob between sentinel markers on
# stdout. Self-contained (no protein_chisel imports) so it survives even if
# the in-container PYTHONPATH is misconfigured.
# ---------------------------------------------------------------------------
_INNER_SCRIPT = r'''
import json
import sys

_BEGIN = "<<<ROSETTA_METRICS_JSON_BEGIN>>>"
_END = "<<<ROSETTA_METRICS_JSON_END>>>"

cfg = json.loads(sys.argv[1])

import pyrosetta
import pyrosetta.rosetta as ros

dalphaball = cfg["dalphaball"]
extra_res_fa = cfg["ligand_params"]
init_flags = (
    f"-mute all -beta -in:file:extra_res_fa {extra_res_fa} "
    f"-dalphaball {dalphaball} -holes:dalphaball {dalphaball}"
)
pyrosetta.init(init_flags)

pose = pyrosetta.pose_from_file(cfg["pdb_path"])

# Find ligand seqpos (first non-virtual ligand residue)
ligand_seqpos = 0
for r in pose.residues:
    if r.is_ligand() and not r.is_virtual_residue():
        ligand_seqpos = int(r.seqpos())
        break
if ligand_seqpos == 0:
    raise RuntimeError("no ligand residue found in pose")

key_atoms = list(cfg["key_atoms"])
ligand_chain = cfg["ligand_chain"]
include_sap = bool(cfg["include_sap"])
include_ec = bool(cfg["include_ec"])
scorefxn_weights = cfg["scorefxn"]

# Build per-key-atom SimpleHbondsToAtomFilter blocks
filters_txt_lines = []
protocols_txt_lines = []
for atom in key_atoms:
    fname = f"{atom}_hbond"
    filters_txt_lines.append(
        f'<SimpleHbondsToAtomFilter name="{fname}" n_partners="1" '
        f'hb_e_cutoff="-0.1" target_atom_name="{atom}" confidence="0" '
        f'res_num="{ligand_seqpos}" scorefxn="sfxn_design"/>'
    )
    protocols_txt_lines.append(f'<Add filter_name="{fname}" />')

filters_txt = "\n          ".join(filters_txt_lines)
protocols_txt = "\n         ".join(protocols_txt_lines)

# Optional simple metrics + the protocol Add line for them.
sm_lines = ['<TotalEnergyMetric name="total_energy" scorefxn="sfxn_design" />',
            '<SecondaryStructureMetric name="secondary_structure" dssp_reduced="false"/>',
            '<SecondaryStructureMetric name="secondary_structure_reduced" dssp_reduced="true"/>']
metrics_names = ["total_energy", "secondary_structure", "secondary_structure_reduced"]
metrics_labels = ["total_rosetta_energy_metric", "secondary_structure",
                  "secondary_structure_DSSP_reduced_alphabet"]
if include_sap:
    sm_lines.append('<SapScoreMetric name="spatial_aggregation_propensity_score"/>')
    metrics_names.append("spatial_aggregation_propensity_score")
    metrics_labels.append("SAP_score")
if include_ec:
    sm_lines.append(
        '<ElectrostaticComplementarityMetric name="electrostatic_complementarity" '
        'ignore_radius="-1" interface_trim_radius="0" partially_solvated="1" '
        'jump="1" report_all_ec="0" />'
    )
    metrics_names.append("electrostatic_complementarity")
    metrics_labels.append("electrostatic_complementarity")
sm_block = "\n          ".join(sm_lines)

xml_script = f"""
<ROSETTASCRIPTS>
  <SCOREFXNS>
      <ScoreFunction name="sfxn_design" weights="{scorefxn_weights}">
          <Reweight scoretype="arg_cation_pi" weight="3"/>
          <Reweight scoretype="angle_constraint" weight="1.0"/>
          <Reweight scoretype="coordinate_constraint" weight="1.0"/>
          <Reweight scoretype="dihedral_constraint" weight="1.0"/>
      </ScoreFunction>
      <ScoreFunction name="sfxn" weights="{scorefxn_weights}" />
  </SCOREFXNS>
  <RESIDUE_SELECTORS>
      <Chain name="chainA" chains="A"/>
      <Chain name="chainB" chains="{ligand_chain}"/>
  </RESIDUE_SELECTORS>
  <SIMPLE_METRICS>
      {sm_block}
  </SIMPLE_METRICS>
  <FILTERS>
      {filters_txt}
      <ContactMolecularSurface name="contact_molecular_surface" use_rosetta_radii="true" distance_weight="0.5" target_selector="chainB" binder_selector="chainA" confidence="0"/>
      <Ddg name="ddg_norepack" threshold="0" jump="1" repeats="1" repack="0" confidence="0" scorefxn="sfxn_design"/>
      <Report name="ddg" filter="ddg_norepack"/>
      <LigInterfaceEnergy name="ligand_interface_energy" scorefxn="sfxn_design" include_cstE="1" confidence="0"/>
      <ResidueCount name="total_residues_in_design_plus_ligand" max_residue_count="99999" min_residue_count="0" count_as_percentage="0" confidence="0"/>
      <ResidueCount name="hydrophobic_residues_in_design" include_property="HYDROPHOBIC" max_residue_count="99999" min_residue_count="0" count_as_percentage="0" confidence="0"/>
      <ResidueCount name="aliphatic_residues_in_design" include_property="ALIPHATIC" max_residue_count="99999" min_residue_count="0" count_as_percentage="0" confidence="0"/>
      <NetCharge name="net_charge_in_design_NOT_w_HIS" chain="1" confidence="0"/>
      <DSasa name="dSasa_fraction" lower_threshold="0.0" upper_threshold="1.0" confidence="0"/>
      <SecondaryStructureCount name="number_DSSP_helices_in_design" num_helix_sheet="0" num_helix="1" num_sheet="0" num_loop="0" filter_helix_sheet="0" filter_helix="1" filter_sheet="0" filter_loop="0" min_helix_length="3" max_helix_length="9999" min_sheet_length="3" max_sheet_length="9999" min_loop_length="1" max_loop_length="9999" return_total="true" confidence="0"/>
      <SecondaryStructureCount name="number_DSSP_sheets_in_design" num_helix_sheet="0" num_helix="0" num_sheet="1" num_loop="0" filter_helix_sheet="0" filter_helix="0" filter_sheet="1" filter_loop="0" min_helix_length="3" max_helix_length="9999" min_sheet_length="3" max_sheet_length="9999" min_loop_length="1" max_loop_length="9999" return_total="true" confidence="0"/>
      <SecondaryStructureCount name="number_DSSP_loops_in_design" num_helix_sheet="0" num_helix="0" num_sheet="0" num_loop="1" filter_helix_sheet="0" filter_helix="0" filter_sheet="0" filter_loop="1" min_helix_length="3" max_helix_length="9999" min_sheet_length="3" max_sheet_length="9999" min_loop_length="1" max_loop_length="9999" return_total="true" confidence="0"/>
      <Holes name="holes_in_design_lower_is_better" threshold="2" normalize_per_residue="false" exclude_bb_atoms="false" confidence="0"/>
      <InterfaceHoles name="interface_holes_at_ligand" jump="1" threshold="200" confidence="0"/>
      <ResInInterface name="num_residues_at_ligand_interface" residues="20" jump_number="1" confidence="0"/>
      <ShapeComplementarity name="shape_complementarity_interface_area" min_sc="0.5" min_interface="1" verbose="0" quick="0" jump="1" write_int_area="1" write_median_dist="0" max_median_dist="1000" residue_selector1="chainA" residue_selector2="chainB" confidence="0"/>
      <ShapeComplementarity name="shape_complementarity_median_distance_at_interface" min_sc="0.5" min_interface="1" verbose="0" quick="0" jump="1" write_int_area="0" write_median_dist="1" max_median_dist="1000" residue_selector1="chainA" residue_selector2="chainB" confidence="0"/>
      <ExposedHydrophobics name="hydrophobic_exposure_sasa_in_design" sasa_cutoff="20" threshold="-1" confidence="0"/>
      <Sasa name="sasa_ligand_interface" threshold="800" upper_threshold="1000000000000000" hydrophobic="0" polar="0" jump="1" confidence="0"/>
      <TotalSasa name="total_pose_sasa" threshold="800" upper_threshold="1000000000000000" hydrophobic="0" polar="0" confidence="0"/>
      <PreProline name="bad_torsion_preproline" use_statistical_potential="0" confidence="0"/>
      <LongestContinuousPolarSegment name="longest_cont_polar_seg" exclude_chain_termini="false" count_gly_as_polar="false" filter_out_high="false" cutoff="5" confidence="0"/>
      <LongestContinuousApolarSegment name="longest_cont_apolar_seg" exclude_chain_termini="false" filter_out_high="false" cutoff="5" confidence="0"/>
  </FILTERS>
  <PROTOCOLS>
      {protocols_txt}
      <Add filter="contact_molecular_surface"/>
      <Add filter="ddg"/>
      <Add filter="ligand_interface_energy"/>
      <Add filter="total_residues_in_design_plus_ligand"/>
      <Add filter="hydrophobic_residues_in_design"/>
      <Add filter="aliphatic_residues_in_design"/>
      <Add filter="net_charge_in_design_NOT_w_HIS"/>
      <Add filter="dSasa_fraction"/>
      <Add filter="number_DSSP_helices_in_design"/>
      <Add filter="number_DSSP_sheets_in_design"/>
      <Add filter="number_DSSP_loops_in_design"/>
      <Add filter="holes_in_design_lower_is_better"/>
      <Add filter="interface_holes_at_ligand"/>
      <Add filter="num_residues_at_ligand_interface"/>
      <Add filter="shape_complementarity_interface_area"/>
      <Add filter="shape_complementarity_median_distance_at_interface"/>
      <Add filter="hydrophobic_exposure_sasa_in_design"/>
      <Add filter="sasa_ligand_interface"/>
      <Add filter="total_pose_sasa"/>
      <Add filter="bad_torsion_preproline"/>
      <Add filter="longest_cont_polar_seg"/>
      <Add filter="longest_cont_apolar_seg"/>
      <Add metrics="{','.join(metrics_names)}" labels="{','.join(metrics_labels)}"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
"""

# Apply via XmlObjects (non-distributed path, doesn't require the
# `--serialization` build of PyRosetta). The XML's <PROTOCOLS> block
# is exposed as the "ParsedProtocol" mover.
xo = ros.protocols.rosetta_scripts.XmlObjects.create_from_string(xml_script)
parsed = xo.get_mover("ParsedProtocol")
parsed.apply(pose)

# After ParsedProtocol.apply(), filter values + simple-metric labels
# live in pose.scores.
score_map = dict(pose.scores)


def _scal(name, default=float("nan")):
    v = score_map.get(name, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _str(name, default=""):
    v = score_map.get(name, default)
    return v if isinstance(v, str) else (str(v) if v is not None else default)


# Key-atom hbond counts
per_atom_hbonds = {atom: _scal(f"{atom}_hbond", 0.0) for atom in key_atoms}

# Filter scalars
filter_keys = [
    "contact_molecular_surface", "ddg", "ligand_interface_energy",
    "total_residues_in_design_plus_ligand", "hydrophobic_residues_in_design",
    "aliphatic_residues_in_design", "net_charge_in_design_NOT_w_HIS",
    "dSasa_fraction", "number_DSSP_helices_in_design",
    "number_DSSP_sheets_in_design", "number_DSSP_loops_in_design",
    "holes_in_design_lower_is_better", "interface_holes_at_ligand",
    "num_residues_at_ligand_interface", "shape_complementarity_interface_area",
    "shape_complementarity_median_distance_at_interface",
    "hydrophobic_exposure_sasa_in_design", "sasa_ligand_interface",
    "total_pose_sasa", "bad_torsion_preproline", "longest_cont_polar_seg",
    "longest_cont_apolar_seg",
]
filter_vals = {k: _scal(k) for k in filter_keys}

# Simple-metric scalars (under their labels)
total_rosetta_energy_metric = _scal("total_rosetta_energy_metric")
secondary_structure = _str("secondary_structure")
secondary_structure_DSSP_reduced_alphabet = _str("secondary_structure_DSSP_reduced_alphabet")
sap_score = _scal("SAP_score") if include_sap else float("nan")
# EC writes three keys with the label-suffix _avg / _p / _s. The aggregate
# average is the canonical EC score; we expose that.
electrostatic_complementarity = (
    _scal("electrostatic_complementarity_avg") if include_ec else float("nan")
)


# getSASA helper, ported from Indrek's script (Coventry recipe)
def getSASA(pose, resno=None, SASA_atoms=None, ignore_sc=False):
    atoms = ros.core.id.AtomID_Map_bool_t()
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
    surf_vol = ros.core.scoring.packing.get_surf_vol(pose, atoms, 1.4)
    if resno is None:
        return surf_vol
    res_surf = 0.0
    for i in range(1, pose.residue(resno).natoms() + 1):
        if SASA_atoms is not None and i not in SASA_atoms:
            continue
        res_surf += surf_vol.surf(resno, i)
    return res_surf


# Custom metric: ligand_exposed_atoms_sasa
ligand_exposed_atoms = list(cfg.get("ligand_exposed_atoms") or [])
ligand_exposed_atoms_sasa = float("nan")
if ligand_exposed_atoms:
    lig_residue = pose.residue(ligand_seqpos)
    sasa_atoms = [
        lig_residue.atom_index(a) for a in ligand_exposed_atoms if lig_residue.has(a)
    ]
    if sasa_atoms:
        ligand_exposed_atoms_sasa = float(getSASA(pose, resno=ligand_seqpos, SASA_atoms=sasa_atoms))

# H-bonds involving the ligand. Use the post-XML pose (which the XML
# scored) so the hbond set is fresh.
hbonds = pose.get_hbonds()
ligand_hbonds_table = []
for hb in hbonds.hbonds():
    d_res = int(hb.don_res())
    a_res = int(hb.acc_res())
    if d_res != ligand_seqpos and a_res != ligand_seqpos:
        continue
    donor_atom = pose.residue(d_res).atom_name(hb.don_hatm()).strip()
    acceptor_atom = pose.residue(a_res).atom_name(hb.acc_atm()).strip()
    ligand_hbonds_table.append({
        "donor_res": d_res,
        "donor_atom": donor_atom,
        "acceptor_res": a_res,
        "acceptor_atom": acceptor_atom,
        "energy": float(hb.energy()),
    })

payload = {
    "ligand_seqpos": ligand_seqpos,
    "filters": filter_vals,
    "per_atom_hbonds": per_atom_hbonds,
    "total_rosetta_energy_metric": total_rosetta_energy_metric,
    "secondary_structure": secondary_structure,
    "secondary_structure_DSSP_reduced_alphabet": secondary_structure_DSSP_reduced_alphabet,
    "SAP_score": sap_score,
    "electrostatic_complementarity": electrostatic_complementarity,
    "ligand_exposed_atoms_sasa": ligand_exposed_atoms_sasa,
    "n_hbonds_to_ligand": len(ligand_hbonds_table),
    "ligand_hbonds_table": ligand_hbonds_table,
}

print(_BEGIN)
print(json.dumps(payload))
print(_END)
'''


def compute_rosetta_metrics(
    pdb_path: str | Path,
    ligand_params: str | Path,
    key_atoms: Iterable[str],
    ligand_exposed_atoms: Optional[Iterable[str]] = None,
    config: Optional[RosettaMetricsConfig] = None,
) -> RosettaMetricsResult:
    """Run the Rosetta metrics XML against a (designed PDB + ligand .params).

    Args:
        pdb_path: input PDB (chain A protein, chain B ligand by default).
        ligand_params: Rosetta ``.params`` file for the ligand.
        key_atoms: ligand atom names to wire into per-atom h-bond filters
            (one ``SimpleHbondsToAtomFilter`` per atom). E.g. ``["S1","O5","O6","N2"]``.
        ligand_exposed_atoms: optional list of ligand atom names to sum
            SASA over (the "ligand_exposed_atoms_sasa" custom metric).
        config: :class:`RosettaMetricsConfig`. ``ligand_chain`` controls
            which chain the chainB ResidueSelector targets (default "B").

    Returns:
        :class:`RosettaMetricsResult` with all metrics flattened.
    """
    from protein_chisel.utils.apptainer import pyrosetta_call

    cfg = config or RosettaMetricsConfig()

    pdb = Path(pdb_path).resolve()
    if not pdb.is_file():
        raise FileNotFoundError(f"PDB not found: {pdb}")
    params = Path(ligand_params).resolve()
    if not params.is_file():
        raise FileNotFoundError(f"params not found: {params}")

    key_atoms_list = [str(a).strip() for a in key_atoms if str(a).strip()]
    exposed_list = (
        [str(a).strip() for a in (ligand_exposed_atoms or []) if str(a).strip()]
    )

    inner_cfg = {
        "pdb_path": str(pdb),
        "ligand_params": str(params),
        "key_atoms": key_atoms_list,
        "ligand_exposed_atoms": exposed_list,
        "ligand_chain": cfg.ligand_chain,
        "include_sap": cfg.include_sap,
        "include_ec": cfg.include_ec,
        "scorefxn": cfg.scorefxn,
        "dalphaball": cfg.dalphaball,
    }

    # Bind the dirs holding the PDB and params so the in-container python
    # can read them at the host paths. If EC is enabled, bind APBS into the
    # sif at /usr/local/bin/apbs (Rosetta's ElectrostaticComplementarityMetric
    # shells out to the `apbs` binary on PATH).
    call = (
        pyrosetta_call()
        .with_bind(str(pdb.parent))
        .with_bind(str(params.parent))
    )
    if cfg.include_ec:
        apbs = Path(cfg.apbs_path)
        if not apbs.is_file():
            raise FileNotFoundError(
                f"include_ec=True but apbs binary not found at {apbs}. "
                "Pass config.apbs_path or set include_ec=False."
            )
        call = call.with_bind(str(apbs), "/usr/local/bin/apbs")

    LOGGER.info("running rosetta_metrics on %s (params=%s)", pdb, params.name)
    result = call.run(
        ["python", "-c", _INNER_SCRIPT, json.dumps(inner_cfg)],
        capture_output=True, timeout=cfg.timeout, check=True,
    )

    # Pluck the JSON payload between the sentinels.
    stdout = result.stdout
    if _JSON_BEGIN not in stdout or _JSON_END not in stdout:
        raise RuntimeError(
            "rosetta_metrics inner script did not emit a JSON payload.\n"
            f"stdout tail:\n{stdout[-4000:]}\nstderr tail:\n{result.stderr[-4000:]}"
        )
    json_blob = stdout.split(_JSON_BEGIN, 1)[1].split(_JSON_END, 1)[0].strip()
    payload = json.loads(json_blob)

    out = RosettaMetricsResult(
        ligand_seqpos=int(payload["ligand_seqpos"]),
        per_atom_hbonds={k: float(v) for k, v in payload["per_atom_hbonds"].items()},
        total_rosetta_energy_metric=_to_float(payload.get("total_rosetta_energy_metric")),
        secondary_structure=str(payload.get("secondary_structure", "") or ""),
        secondary_structure_DSSP_reduced_alphabet=str(
            payload.get("secondary_structure_DSSP_reduced_alphabet", "") or ""
        ),
        SAP_score=_to_float(payload.get("SAP_score")),
        electrostatic_complementarity=_to_float(payload.get("electrostatic_complementarity")),
        ligand_exposed_atoms_sasa=_to_float(payload.get("ligand_exposed_atoms_sasa")),
        n_hbonds_to_ligand=int(payload.get("n_hbonds_to_ligand", 0)),
        ligand_hbonds_table=list(payload.get("ligand_hbonds_table", [])),
    )
    for fk, fv in payload["filters"].items():
        if hasattr(out, fk):
            setattr(out, fk, _to_float(fv))
    return out


def _to_float(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# MetricSpec adapter (for the tier filter)
# ---------------------------------------------------------------------------


def _rosetta_metrics_metric(c, params: dict):
    from protein_chisel.scoring.metrics import from_tool_result

    cfg_keys = set(RosettaMetricsConfig.__dataclass_fields__)
    cfg_kwargs = {k: v for k, v in params.items() if k in cfg_keys}
    cfg = RosettaMetricsConfig(**cfg_kwargs)

    res = compute_rosetta_metrics(
        c.structure_path,
        ligand_params=params["ligand_params"],
        key_atoms=params["key_atoms"],
        ligand_exposed_atoms=params.get("ligand_exposed_atoms"),
        config=cfg,
    )
    return from_tool_result("rosetta_metrics", res, prefix="rosetta__")


# Lazy MetricSpec construction: import scoring.metrics on demand to avoid
# pulling its deps at module import time. Build via a function so consumers
# can `from protein_chisel.tools.rosetta_metrics import ROSETTA_METRICS_SPEC`
# and the import resolves.
def _build_spec():
    from protein_chisel.scoring.metrics import MetricSpec
    return MetricSpec(
        name="rosetta_metrics",
        fn=_rosetta_metrics_metric,
        kind="structure+ligand",
        cost_seconds=180.0,
        needs_gpu=False,
        prefix="rosetta__",
        description=(
            "Comprehensive RosettaScripts metrics (no relax): contact MS, "
            "ddg, ligand interface energy, secondary-structure counts, holes, "
            "shape complementarity, SAP, EC, and per-key-atom h-bond counts."
        ),
        cache_version=1,
    )


try:
    ROSETTA_METRICS_SPEC = _build_spec()
except Exception:  # pragma: no cover -- only triggers in stripped environments
    ROSETTA_METRICS_SPEC = None


__all__ = [
    "ROSETTA_METRICS_SPEC",
    "RosettaMetricsConfig",
    "RosettaMetricsResult",
    "compute_rosetta_metrics",
]
