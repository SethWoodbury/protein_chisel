#!/usr/bin/env python
"""chisel_ligandMPNN.py — wrapper around the fused_mpnn LigandMPNN ``run.py``.

Runs MPNN exactly like a raw ``apptainer exec ... run.py ...`` command (every
LigandMPNN flag is passed straight through) while optionally layering on two
toggleable steps and an optional multi-combo sweep:

  PRE  (default ON; ``--no_fix_remark666_catres`` to disable):
       if you did NOT supply ``--fixed_residues`` / ``--fixed_residues_multi``,
       parse the input PDB's REMARK 666 catalytic-residue lines and inject them
       as ``--fixed_residues_multi`` so the catalytic site stays fixed. If the
       PDB has no REMARK 666 lines we WARN and run a full (unfixed) redesign.

  POST (default ON; ``--no_protonate`` to disable; auto-skips if no
       ``--ligand_params``): protonate each run's output PDBs (PyRosetta, via
       ``protonate_final_topk.py`` inside pyrosetta.sif) and OVERWRITE them in
       place.

  SWEEP (optional): give an explicit list of hyper-parameter combos via
       repeatable ``--run`` or ``--sweep_config <json>``. Each combo overrides
       only the keys it names (``temperature``, ``number_of_batches``,
       ``batch_size``, ``enhance``, ``omit_AA``, ``bias_AA``); everything else
       inherits your universal/base value. Each combo gets a minimal unique
       ``--packed_suffix`` so packed PDBs never collide.

OUTPUT layout: the final ``--out_folder`` is a single flat layer — just the
packed PDBs (named ``<stem><suffix>_<i>_<c>.pdb``) plus a copy of the input
structure under its original name. run.py's ``seqs/``, ``backbones/``, and
per-combo staging subdirs are removed. ``--keep_intermediates`` additionally
keeps the sequence FASTAs (tag-qualified) and the REMARK 666 JSON, still flat.
``--no_copy_input_structure`` suppresses the input copy.

This is a thin host-side orchestrator: it builds ``apptainer exec`` commands
(reusing protein_chisel.utils.apptainer) and shells out. CPU = no ``--nv``;
MPNN runs fine on CPU. Nothing here submits SLURM jobs.

Examples
--------
Reproduce a raw command verbatim (no pre/post)::

    python scripts/chisel_ligandMPNN.py --no_fix_remark666_catres --no_protonate \\
        --model_type ligand_mpnn --pdb_path in.pdb --out_folder out \\
        --temperature 0.2 --number_of_batches 15 --batch_size 1 \\
        --pack_side_chains 1 --omit_AA CX

Auto-fix catalytic residues + protonate, sweeping three combos::

    python scripts/chisel_ligandMPNN.py --device cpu \\
        --ligand_params lig.params \\
        --model_type ligand_mpnn --pdb_path in.pdb --out_folder out \\
        --pack_side_chains 1 --omit_AA CX --bias_AA 'K:-0.5,R:-0.75' \\
        --run 't=0.1;n=15' --run 't=0.2;n=15;bs=2' --run 't=0.3;n=10;bias=K:-1.0'

Note: within a single ``--run`` string, fields are separated by ``;`` (not
``,``) so values like ``bias=K:-0.5,R:-0.75`` keep their commas. For rich/
notebook use, prefer ``--sweep_config combos.json`` (a JSON list of objects).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Bootstrap: make `import protein_chisel` work on the host (PYTHONPATH=src
# convention) before importing anything from the package.
# ---------------------------------------------------------------------------
def _bootstrap_sys_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap_sys_path()

from protein_chisel.io.pdb import parse_remark_666  # noqa: E402
from protein_chisel.paths import FUSED_MPNN_RUN, UNIVERSAL_SIF  # noqa: E402
from protein_chisel.tools import remarks  # noqa: E402
from protein_chisel.utils.apptainer import ApptainerCall, pyrosetta_call  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROTONATE_SCRIPT = REPO / "scripts" / "protonate_final_topk.py"
STANDARD_ROOTS = ("/net/scratch", "/net/software", "/net/databases")
# universal.sif ships cifutils (and friends) on PYTHONPATH at /cifutils/src.
# ApptainerCall.build_command() forces PYTHONPATH=/code/src, which would clobber
# that, so we explicitly preserve it — exactly as run_chisel_design.sh sets
# `PYTHONPATH=/code/src:/cifutils/src` for the design/MPNN stage. Without this,
# run.py fails with `ModuleNotFoundError: No module named 'cifutils'`.
_UNIVERSAL_PYTHONPATH_KEEPERS = ("/cifutils/src",)

LOGGER = logging.getLogger("chisel_ligandMPNN")
_QUIET = False


def say(msg: str = "") -> None:
    """Verbose narrative print, suppressed under --quiet."""
    if not _QUIET:
        print(msg, flush=True)


def banner(title: str) -> None:
    say("")
    say("=" * 72)
    say(title)
    say("=" * 72)


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------
def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        prog="chisel_ligandMPNN.py",
        description=(
            "Wrapper around fused_mpnn LigandMPNN run.py. ALL unrecognized args "
            "are passed straight through to run.py, so you can drive MPNN exactly "
            "as you would raw."
        ),
        epilog=(
            "Within a --run string, separate fields with ';' (e.g. "
            "'t=0.2;n=15;bias=K:-0.5,R:-0.75'). Keys: t/temperature, "
            "n/number_of_batches, bs/batch_size, enh/enhance (or none), "
            "omit/omit_AA, bias/bias_AA, tag."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--no_fix_remark666_catres", action="store_true",
                   help="Disable REMARK 666 catalytic-residue auto-fixing.")
    p.add_argument("--no_protonate", action="store_true",
                   help="Disable the post-MPNN protonation step entirely.")
    p.add_argument("--ligand_params", nargs="+", default=None,
                   help="Rosetta .params for the ligand. With it, full holo "
                        "protonation. Without it, the apo fallback runs (protonate "
                        "protein, ligand re-added from input) unless "
                        "--no_apo_protonate_fallback.")
    p.add_argument("--no_apo_protonate_fallback", action="store_true",
                   help="When no --ligand_params is given, do NOT run the apo "
                        "protonation fallback; leave the MPNN PDBs raw instead.")
    p.add_argument("--no_restore_catalytic", action="store_true",
                   help="Do NOT restore REMARK 666 catalytic residue states (HIS "
                        "tautomer / KCX / catalytic H) from the input before "
                        "protonation. Default: restore them, so the catalytic HIS "
                        "protonation matches the input (LigandMPNN emits plain HIS, "
                        "which would otherwise protonate to the Rosetta default).")
    p.add_argument("--seed_pdb", default=None,
                   help="Protonation reference PDB (REMARK 666 + ligand). "
                        "Defaults to --pdb_path.")
    p.add_argument("--protonate_subdir", default=None,
                   help="Which run.py output subdir holds the structures to keep + "
                        "protonate (default: auto — 'packed' unless pack_side_chains=0, "
                        "then 'backbones'). These are flattened into --out_folder.")
    p.add_argument("--mpnn_run_script", default=None,
                   help=f"Override the run.py path (default {FUSED_MPNN_RUN}).")
    p.add_argument("--sif", default=None,
                   help="Override the MPNN container (default: universal.sif).")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                   help="gpu => --nv; cpu => no --nv; auto => probe nvidia-smi.")
    p.add_argument("--ptm", default=None,
                   help="PTM declaration(s) annotated into the REMARK 668 lines of "
                        "the protonated outputs (annotation only; coords unchanged). "
                        "Motif-index form (preferred): 'A/LYS/3:KCX'; or explicit "
                        "'A:157=KCX' / '157=KCX'; comma/space-separated for several. "
                        "Forwarded to the protonate step's --ptm.")
    p.add_argument("--quiet", action="store_true",
                   help="Reduce verbose narration.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print assembled commands without executing.")
    p.add_argument("--run", action="append", default=None, dest="runs",
                   metavar="'t=..;n=..'",
                   help="One sweep combo (repeatable). See epilog for keys.")
    p.add_argument("--sweep_config", default=None,
                   help="JSON file: a list of combo objects (alternative to --run).")
    p.add_argument("--suffix_base", default="",
                   help="Prepended to the auto run-tag in each --packed_suffix.")
    p.add_argument("--keep_intermediates", action="store_true",
                   help="Keep the sequence FASTAs and the REMARK 666 fixed-residues "
                        "JSON (flattened into --out_folder, one layer). Default: only "
                        "the packed PDBs (+ input copy) are kept.")
    p.add_argument("--no_copy_input_structure", action="store_true",
                   help="Do NOT copy the input PDB into --out_folder. Default: the "
                        "input structure is copied in under its original name.")
    p.add_argument("--no_transfer_remarks", action="store_true",
                   help="Do NOT transfer REMARK lines (REMARK 666 etc.) from the input "
                        "PDB onto each output PDB. Default: transfer them.")
    p.add_argument("--no_design_path_remark", action="store_true",
                   help="Do NOT add a 'REMARK DESIGN_PATH chisel_ligandmpnn output "
                        "<path>' line to each output PDB. Default: add it.")
    p.add_argument("--no_omit_nterm_met", action="store_true",
                   help="Allow Met at the N-terminal residue. Default: omit 'M' at "
                        "position 1 (unless that residue is fixed) since a Met tag is "
                        "added during expression. This is additive to omit_AA.")
    # --- H-bond sidechain conservation (opt-in) ---
    p.add_argument("--conserve_hbonds", action="store_true",
                   help="Detect designable-residue SIDECHAIN H-bonds to the ligand / "
                        "fixed residues and probabilistically pin those residues into the "
                        "fixed list (rolled independently per LigandMPNN run).")
    p.add_argument("--conserve_hbond_prob", type=float, default=0.8,
                   help="Per-residue fix probability (default 0.8). 0-1, or a percentage "
                        ">1 (e.g. 80) which is auto-converted with a warning.")
    p.add_argument("--conserve_hbond_all_or_none", action="store_true",
                   help="Roll each candidate ONCE and apply to all runs (default: roll "
                        "independently per run).")
    p.add_argument("--conserve_seed", type=int, default=None,
                   help="Seed the conservation rolls for reproducibility (default: random).")
    p.add_argument("--conserve_anchors", default="ligand,catalytic,user_fixed",
                   help="Comma list of anchor groups a sidechain must H-bond to "
                        "(subset of ligand,catalytic,user_fixed; default all).")
    p.add_argument("--conserve_hbond_max_dist", type=float, default=3.9,
                   help="Heavy-atom donor···acceptor distance cutoff (default 3.9 A).")
    p.add_argument("--conserve_hbond_max_angle", type=float, default=90.0,
                   help="Antecedent-D-A angle gate; larger = more permissive "
                        "(default 90; stock detector uses 70).")
    p.add_argument("--conserve_keep_clashing", action="store_true",
                   help="Keep conserved candidates that clash with fixed backbone / ligand "
                        "(default: exclude them with a warning).")
    return p.parse_known_args(argv)


def _get_flag_value(extras: list[str], name: str) -> Optional[str]:
    """Last value of `--name value` or `--name=value` in extras (read-only)."""
    val = None
    i = 0
    while i < len(extras):
        tok = extras[i]
        if tok == name:
            if i + 1 < len(extras):
                val = extras[i + 1]
            i += 2
            continue
        if tok.startswith(name + "="):
            val = tok.split("=", 1)[1]
        i += 1
    return val


def _has_flag(extras: list[str], name: str) -> bool:
    return any(t == name or t.startswith(name + "=") for t in extras)


def _strip_flags(extras: list[str], names: list[str]) -> list[str]:
    """Remove `--flag value` / `--flag=value` for each name; preserve the rest."""
    nameset = set(names)
    out: list[str] = []
    i = 0
    while i < len(extras):
        tok = extras[i]
        key = tok.split("=", 1)[0]
        if key in nameset:
            i += 1 if "=" in tok else 2
            continue
        out.append(tok)
        i += 1
    return out


@dataclass
class ScannedArgs:
    pdb_path: Optional[str]
    out_folder: Optional[str]
    pack_side_chains: Optional[str]
    packed_suffix: Optional[str]
    enhance: Optional[str]
    temperature: Optional[str]
    number_of_batches: Optional[str]
    batch_size: Optional[str]
    omit_AA: Optional[str]
    bias_AA: Optional[str]
    has_fixed: bool
    has_pdb_path_multi: bool


def scan_mpnn_args(extras: list[str]) -> ScannedArgs:
    return ScannedArgs(
        pdb_path=_get_flag_value(extras, "--pdb_path"),
        out_folder=_get_flag_value(extras, "--out_folder"),
        pack_side_chains=_get_flag_value(extras, "--pack_side_chains"),
        packed_suffix=_get_flag_value(extras, "--packed_suffix"),
        enhance=_get_flag_value(extras, "--enhance"),
        temperature=_get_flag_value(extras, "--temperature"),
        number_of_batches=_get_flag_value(extras, "--number_of_batches"),
        batch_size=_get_flag_value(extras, "--batch_size"),
        omit_AA=_get_flag_value(extras, "--omit_AA"),
        bias_AA=_get_flag_value(extras, "--bias_AA"),
        has_fixed=_has_flag(extras, "--fixed_residues")
        or _has_flag(extras, "--fixed_residues_multi"),
        has_pdb_path_multi=_has_flag(extras, "--pdb_path_multi"),
    )


def resolve_device(device: str) -> bool:
    """Return True if --nv should be used."""
    if device == "gpu":
        nv = True
    elif device == "cpu":
        nv = False
    else:  # auto
        nv = False
        if shutil.which("nvidia-smi"):
            try:
                nv = subprocess.run(
                    ["nvidia-smi"], capture_output=True
                ).returncode == 0
            except Exception:
                nv = False
    LOGGER.info("device=%s -> apptainer %s", device, "--nv (GPU)" if nv else "(CPU)")
    return nv


# ---------------------------------------------------------------------------
# Sweep / combo expansion
# ---------------------------------------------------------------------------
_KEYMAP = {
    "t": "temperature", "temp": "temperature", "temperature": "temperature",
    "n": "number_of_batches", "nbatches": "number_of_batches",
    "number_of_batches": "number_of_batches",
    "bs": "batch_size", "batch_size": "batch_size",
    "enh": "enhance", "enhance": "enhance",
    "omit": "omit_AA", "omit_AA": "omit_AA",
    "bias": "bias_AA", "bias_AA": "bias_AA",
    "tag": "tag",
}


@dataclass
class RunSpec:
    temperature: Optional[str]
    number_of_batches: Optional[str]
    batch_size: Optional[str]
    enhance: Optional[str]
    omit_AA: Optional[str]
    bias_AA: Optional[str]
    run_tag: Optional[str]
    packed_suffix: Optional[str]
    out_folder: Optional[str]


def _temp_tag(t: Optional[str]) -> str:
    if t is None:
        return ""
    f = f"{float(t):.2f}".rstrip("0").rstrip(".")
    return f.replace(".", "_")


def _parse_run_string(s: str) -> dict:
    combo: dict[str, str] = {}
    for field in s.split(";"):
        field = field.strip()
        if not field:
            continue
        if "=" not in field:
            raise SystemExit(f"--run: bad field {field!r} (expected key=value)")
        k, v = field.split("=", 1)
        k, v = k.strip(), v.strip()
        if k not in _KEYMAP:
            raise SystemExit(f"--run: unknown key {k!r} in {s!r}")
        combo[_KEYMAP[k]] = v
    return combo


def parse_sweep(args: argparse.Namespace, scanned: ScannedArgs) -> tuple[list[RunSpec], bool]:
    if args.runs and args.sweep_config:
        raise SystemExit("use either --run or --sweep_config, not both")

    combos: list[dict] = []
    if args.runs:
        combos = [_parse_run_string(s) for s in args.runs]
    elif args.sweep_config:
        data = json.loads(Path(args.sweep_config).read_text())
        if not isinstance(data, list):
            raise SystemExit("--sweep_config must be a JSON list of combo objects")
        for d in data:
            combo: dict[str, str] = {}
            for k, v in d.items():
                if k not in _KEYMAP:
                    raise SystemExit(f"--sweep_config: unknown key {k!r}")
                combo[_KEYMAP[k]] = v if isinstance(v, str) else str(v)
            combos.append(combo)

    if not combos:
        # Single run from the user's base/passthrough values, verbatim.
        return [RunSpec(
            temperature=scanned.temperature,
            number_of_batches=scanned.number_of_batches,
            batch_size=scanned.batch_size,
            enhance=scanned.enhance,
            omit_AA=scanned.omit_AA,
            bias_AA=scanned.bias_AA,
            run_tag=None,
            packed_suffix=scanned.packed_suffix,
            out_folder=scanned.out_folder,
        )], False

    # Resolve each combo against the base (override-else-inherit).
    resolved: list[dict] = []
    for combo in combos:
        if "enhance" in combo:
            ev = combo["enhance"]
            enhance = None if ev.lower() in ("none", "stock", "") else ev
        else:
            enhance = scanned.enhance
        resolved.append({
            "temperature": combo.get("temperature", scanned.temperature),
            "number_of_batches": combo.get("number_of_batches", scanned.number_of_batches),
            "batch_size": combo.get("batch_size", scanned.batch_size),
            "enhance": enhance,
            "omit_AA": combo.get("omit_AA", scanned.omit_AA),
            "bias_AA": combo.get("bias_AA", scanned.bias_AA),
            "tag": combo.get("tag"),
        })

    # Short, stable enhance tokens (only used in tags when enhance varies).
    enh_varies = len({r["enhance"] for r in resolved}) > 1
    enh_token: dict[Optional[str], str] = {}
    idx = 0
    for r in resolved:
        e = r["enhance"]
        if e not in enh_token:
            enh_token[e] = "stock" if e is None else f"e{idx}"
            if e is not None:
                idx += 1

    sb = args.suffix_base.strip()
    if sb and not sb.startswith("_"):
        sb = "_" + sb
    sb = sb.rstrip("_")

    specs: list[RunSpec] = []
    used: set[str] = set()
    for i, r in enumerate(resolved):
        if r["tag"]:
            tag = r["tag"]
        else:
            parts = []
            if enh_varies:
                parts.append(enh_token[r["enhance"]])
            parts.append("T" + _temp_tag(r["temperature"]))
            tag = "_".join(p for p in parts if p)
        if tag in used:
            tag = f"{tag}_r{i}"
        used.add(tag)

        packed_suffix = f"{sb}_{tag}" if sb else f"_{tag}"

        # Each combo runs in its own staging subdir <out_folder>/<tag>/ to keep
        # run.py's seqs/backbones/packed isolated; finalize_outputs() flattens
        # the packed PDBs up into <out_folder> and removes the staging dirs.
        base_out = scanned.out_folder
        out_folder = f"{base_out.rstrip('/')}/{tag}" if base_out else base_out

        specs.append(RunSpec(
            temperature=r["temperature"],
            number_of_batches=r["number_of_batches"],
            batch_size=r["batch_size"],
            enhance=r["enhance"],
            omit_AA=r["omit_AA"],
            bias_AA=r["bias_AA"],
            run_tag=tag,
            packed_suffix=packed_suffix,
            out_folder=out_folder,
        ))
    return specs, True


# ---------------------------------------------------------------------------
# Pre-process: REMARK 666 -> fixed_residues_multi JSON
# ---------------------------------------------------------------------------
def build_fixed_residues_from_remark666(catres: dict, pdb_path: str, out_json: Path) -> Optional[Path]:
    if not catres:
        LOGGER.warning(
            "no REMARK 666 catalytic residues found in %s -> running FULL "
            "(unfixed) redesign; NOTHING will be held fixed.", pdb_path,
        )
        return None
    residues = sorted(catres.values(), key=lambda c: (c.chain, c.resno))
    say("")
    say(f"  REMARK 666: found {len(residues)} catalytic residue(s) -> "
        f"these WILL BE FIXED (not redesigned):")
    say(f"      {'cst#':<5}{'chain':<7}{'resname':<9}{'resno':<7}ligand (target)")
    say(f"      {'-' * 4:<5}{'-' * 5:<7}{'-' * 7:<9}{'-' * 5:<7}{'-' * 22}")
    labels: list[str] = []
    for cr in residues:
        ligand = f"{cr.target_name3}/{cr.target_resno} (chain {cr.target_chain})"
        say(f"      {cr.cst_no:<5}{cr.chain:<7}{cr.name3:<9}{cr.resno:<7}{ligand}")
        lab = f"{cr.chain}{cr.resno}"
        if lab not in labels:
            labels.append(lab)
    payload = {str(Path(pdb_path).resolve()): labels}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    say(f"      fixed_residues tokens : {', '.join(labels)}")
    say(f"      fixed-residues JSON   : {out_json}")
    return out_json


# ---------------------------------------------------------------------------
# Command assembly
# ---------------------------------------------------------------------------
_SWEPT_FLAGS = [
    "--temperature", "--number_of_batches", "--batch_size", "--enhance",
    "--omit_AA", "--bias_AA", "--packed_suffix", "--out_folder",
]


def build_run_extras(
    extras: list[str], run: RunSpec, injected: list[str], sweeping: bool
) -> list[str]:
    if sweeping:
        result = _strip_flags(extras, _SWEPT_FLAGS)

        def add(flag: str, val: Optional[str]) -> None:
            if val is not None and str(val) != "":
                result.extend([flag, str(val)])

        add("--temperature", run.temperature)
        add("--number_of_batches", run.number_of_batches)
        add("--batch_size", run.batch_size)
        if run.enhance:  # None/empty => no --enhance for this run
            result.extend(["--enhance", run.enhance])
        add("--omit_AA", run.omit_AA)
        add("--bias_AA", run.bias_AA)
        add("--packed_suffix", run.packed_suffix)
        add("--out_folder", run.out_folder)
    else:
        result = list(extras)
    result += list(injected)  # e.g. --fixed_residues_multi / --omit_AA_per_residue_multi
    return result


def _bind_dirs(files: list[Optional[str]], dir_paths: list[Optional[str]]) -> list[str]:
    dirs: list[str] = []

    def add(d: str) -> None:
        d = str(Path(d).resolve())
        if d not in dirs and Path(d).is_dir():
            dirs.append(d)

    for f in files:
        if f:
            add(str(Path(f).resolve().parent))
    for d in dir_paths:
        if d:
            add(d)
    for r in STANDARD_ROOTS:
        if Path(r).is_dir():
            add(r)
    return dirs


def assemble_apptainer_command(
    inner_argv: list[str], bind_dirs: list[str], sif: Optional[str], nv: bool
) -> list[str]:
    # Build the universal-sif call directly so we can preserve /cifutils/src on
    # PYTHONPATH (build_command would otherwise overwrite it with /code/src).
    call = ApptainerCall(
        sif=Path(sif) if sif else UNIVERSAL_SIF,
        nv=nv,
        container_pythonpath_keepers=_UNIVERSAL_PYTHONPATH_KEEPERS,
    )
    for d in bind_dirs:
        call = call.with_bind(d)
    return call.build_command(inner_argv)


def run_streaming(cmd: list[str], dry_run: bool, label: str) -> int:
    say(f"  $ {shlex.join(cmd)}")
    if dry_run:
        return 0
    t0 = time.time()
    proc = subprocess.run(cmd)
    dt = time.time() - t0
    level = LOGGER.info if proc.returncode == 0 else LOGGER.error
    level("  %s -> exit %d in %.1fs", label, proc.returncode, dt)
    return proc.returncode


# ---------------------------------------------------------------------------
# Post-process: protonate + overwrite in place
# ---------------------------------------------------------------------------
def resolve_protonate_subdir(scanned: ScannedArgs, args: argparse.Namespace) -> str:
    if args.protonate_subdir:
        return args.protonate_subdir
    psc = scanned.pack_side_chains
    return "backbones" if (psc is not None and str(psc) == "0") else "packed"


def _is_intermediate(name: str) -> bool:
    return name.endswith((".protonated.pdb", ".rosetta.pdb", ".norm.pdb"))


def _designs(run: RunSpec) -> int:
    """Best-effort packed-PDB count for one run (run.py defaults n=bs=1)."""
    try:
        n = int(run.number_of_batches) if run.number_of_batches is not None else 1
        bs = int(run.batch_size) if run.batch_size is not None else 1
        return n * bs
    except (TypeError, ValueError):
        return 0


def finalize_outputs(
    base_dir: str, runs: list[RunSpec], sweeping: bool, subdir: str, keep: bool,
) -> int:
    """Flatten each run's `<subdir>/*.pdb` up into base_dir (one layer) and
    delete the seqs/backbones/packed/stats staging dirs. With keep=True also
    pull the sequence FASTAs (tag-qualified) and the _chisel_pre/ JSONs up
    flat; otherwise they're removed."""
    base = Path(base_dir)
    moved, kept_fa = 0, 0
    for run in runs:
        rdir = Path(run.out_folder)
        src = rdir / subdir
        if src.is_dir():
            for pdb in sorted(src.glob("*.pdb")):
                os.replace(pdb, base / pdb.name)
                moved += 1
        if keep:
            seqs = rdir / "seqs"
            if seqs.is_dir():
                for fa in sorted(seqs.glob("*")):
                    if not fa.is_file():
                        continue
                    if sweeping and run.run_tag:
                        stem, ext = os.path.splitext(fa.name)
                        name = f"{stem}__{run.run_tag}{ext}"
                    else:
                        name = fa.name
                    os.replace(fa, base / name)
                    kept_fa += 1
        if rdir.resolve() != base.resolve():
            shutil.rmtree(rdir, ignore_errors=True)        # whole staging subdir
        else:
            for d in ("seqs", "backbones", "packed", "stats"):
                shutil.rmtree(base / d, ignore_errors=True)
    pre = base / "_chisel_pre"
    if pre.is_dir():
        if keep:
            for f in sorted(pre.glob("*")):
                if f.is_file():
                    os.replace(f, base / f.name)
        shutil.rmtree(pre, ignore_errors=True)
    say(f"  flattened {moved} packed PDB(s) into {base}")
    say("  removed staging dirs (seqs/ backbones/ packed/)"
        + (f"; kept {kept_fa} sequence FASTA(s) + _chisel_pre JSONs" if keep else
           " and the _chisel_pre helper JSONs (use --keep_intermediates to keep them)"))
    return moved


def copy_input_structure(pdb_path: str, base_dir: str) -> Optional[str]:
    src = Path(pdb_path)
    dest = Path(base_dir) / src.name
    if dest.resolve() == src.resolve():
        return None
    shutil.copy2(src, dest)
    say(f"  copied input structure -> {dest}")
    return str(dest)


def protonate_dir(
    topk_dir: str, seed_pdb: str, ligand_params: list[str], args: argparse.Namespace,
) -> dict:
    """Protonate every design PDB in topk_dir (flat) and OVERWRITE in place."""
    topk = Path(topk_dir)
    inner = [
        "python", str(PROTONATE_SCRIPT),
        "--topk_dir", str(topk),
        "--seed_pdb", str(Path(seed_pdb).resolve()),
    ]
    if ligand_params:
        inner += ["--ligand_params", *[str(Path(p).resolve()) for p in ligand_params]]
    # else: apo fallback — PyRosetta ignores the unparametrized ligand on load and
    # protonate_final_topk re-adds it verbatim from --seed_pdb.
    if args.ptm:
        inner += ["--ptm", args.ptm]
    binds = _bind_dirs(files=[seed_pdb, *ligand_params], dir_paths=[str(topk)])
    cmd = pyrosetta_call()
    for d in binds:
        cmd = cmd.with_bind(d)
    full = cmd.build_command(inner)

    if args.dry_run:
        say(f"  [protonate would run in-place on {topk}]")
        say(f"  $ {shlex.join(full)}")
        return {"dry_run": True}

    if not topk.is_dir():
        LOGGER.warning("protonate: %s missing; skipping", topk)
        return {"skipped": "no_dir"}
    pdbs = [p for p in sorted(topk.glob("*.pdb")) if not _is_intermediate(p.name)]
    if not pdbs:
        LOGGER.warning("protonate: no PDBs in %s; skipping", topk)
        return {"skipped": "empty"}

    say(f"  protonating {len(pdbs)} PDB(s) in {topk} (PyRosetta, pyrosetta.sif) ...")
    rc = run_streaming(full, dry_run=False, label="protonate")
    if rc != 0:
        LOGGER.error("protonate: failed (exit %d); outputs left raw", rc)
        return {"failed": rc}

    overwritten = 0
    for p in pdbs:
        prot = topk / f"{p.stem}.protonated.pdb"
        if prot.exists():
            os.replace(prot, p)  # atomic same-FS overwrite-in-place
            overwritten += 1
    for pattern in ("*.rosetta.pdb", "*.norm.pdb", "*.protonated.pdb"):
        for stray in topk.glob(pattern):
            try:
                stray.unlink()
            except FileNotFoundError:
                pass
    say(f"  protonated + overwrote {overwritten}/{len(pdbs)} PDB(s) in place")
    return {"overwritten": overwritten, "total": len(pdbs)}


def restore_catalytic_states_in_dir(
    base_dir: str, seed_pdb: str, catalytic_resnos: list[int], chain: str,
) -> int:
    """Relabel each design PDB's REMARK 666 catalytic residues to the input's
    HIS tautomer / KCX state and copy catalytic H from the input, IN PLACE.

    LigandMPNN emits plain heavy-atom `HIS` with no tautomer, so without this
    PyRosetta would protonate every catalytic HIS to its default tautomer. This
    is the same restore the production pipeline runs before final protonation
    (`protein_chisel.tools.pdb_restoration.restore_pdb_features`)."""
    from protein_chisel.tools.pdb_restoration import restore_pdb_features

    base = Path(base_dir)
    pdbs = [p for p in sorted(base.glob("*.pdb")) if not _is_intermediate(p.name)]
    total_relabeled = 0
    for p in pdbs:
        tmp = p.with_name(p.name + ".rst")  # non-.pdb so it isn't re-globbed
        stats = restore_pdb_features(
            mpnn_pdb=str(p), ref_pdb=str(seed_pdb), out_pdb=str(tmp),
            his_tautomers=True, kcx=True, catalytic_hydrogens=True,
            catalytic_resnos=list(catalytic_resnos), chain=chain,
            drop_mpnn_remarks=True,
        )
        os.replace(tmp, p)
        total_relabeled += stats.get("his_relabeled", 0)
    say(f"  restored catalytic states from input on {len(pdbs)} PDB(s) "
        f"(HIS tautomer relabels total: {total_relabeled}; ref={Path(seed_pdb).name})")
    return len(pdbs)


# ---------------------------------------------------------------------------
# Post-process: REMARK transfer + REMARK 668 annotation.
# The transfer/merge + DESIGN_PATH logic lives in protein_chisel.tools.remarks
# (shared with the classic pipeline); here we just apply the wrapper's policy.
# ---------------------------------------------------------------------------
def annotate_input_copy_remark668(copy_path: str, ptm_spec: Optional[str]) -> bool:
    """(Re)build the REMARK 667/668 protonation-state block on the copied input.

    Uses the input's own REMARK 666 + catalytic H atoms to detect states and
    applies any --ptm overrides (e.g. KCX). No-op if the input has no REMARK 666
    (build returns no lines)."""
    from protein_chisel.tools.protonate_final import build_remark_668_block

    block = build_remark_668_block(copy_path, copy_path, ptm_map=ptm_spec)
    if not remarks.replace_remark_block(copy_path, block):
        return False
    say(f"  annotated REMARK 668 on the input copy ({Path(copy_path).name})")
    return True


def _first_protein_residue(pdb_path: str) -> Optional[tuple[str, int]]:
    """(chain, resno) of the first ATOM record = the N-terminal residue."""
    try:
        with open(pdb_path, "r") as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    chain = line[21].strip() or "A"
                    try:
                        return (chain, int(line[22:26]))
                    except ValueError:
                        return None
    except OSError:
        return None
    return None


def _parse_user_fixed_labels(extras: list[str]) -> set[str]:
    """Collect fixed-residue labels the user supplied via --fixed_residues /
    --fixed_residues_multi (used to decide whether the N-term is already fixed)."""
    labels: set[str] = set()
    fr = _get_flag_value(extras, "--fixed_residues")
    if fr:
        labels.update(fr.split())
    frm = _get_flag_value(extras, "--fixed_residues_multi")
    if frm and Path(frm).is_file():
        try:
            data = json.loads(Path(frm).read_text())
            for v in data.values():
                labels.update(v if isinstance(v, list) else str(v).split())
        except (OSError, ValueError):
            pass
    return labels


def build_omit_nterm_met_json(pdb_path: str, label: str, out_json: Path) -> Path:
    payload = {str(Path(pdb_path).resolve()): {label: "M"}}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    return out_json


# ---------------------------------------------------------------------------
# H-bond sidechain conservation helpers
# ---------------------------------------------------------------------------
def parse_probability(value: float) -> float:
    """A 0-1 probability, or a percentage >1 (auto-converted with a warning)."""
    v = float(value)
    if v < 0:
        raise SystemExit(f"--conserve_hbond_prob must be >= 0 (got {v})")
    if v <= 1:
        return v
    if v <= 100:
        LOGGER.warning("--conserve_hbond_prob=%g > 1; interpreting as a percentage -> %g",
                       v, v / 100.0)
        return v / 100.0
    raise SystemExit(f"--conserve_hbond_prob must be 0-1 or a percentage <=100; got {v}")


def _protein_resnos(pdb_path: str, chain: str) -> set[int]:
    out: set[int] = set()
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM  ") and line[21] == chain:
                try:
                    out.add(int(line[22:26]))
                except ValueError:
                    pass
    return out


def _label_sort_key(lab: str) -> tuple:
    return (lab[0], int(lab[1:]) if lab[1:].isdigit() else 0)


def _fixed_residues_json(pdb_path: str, labels, out_json: Path) -> Path:
    payload = {str(Path(pdb_path).resolve()): sorted(set(labels), key=_label_sort_key)}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    return out_json


def _roll_conserved(labels, prob: float, rng: random.Random) -> set:
    return {lab for lab in labels if rng.random() < prob}


def _fmt_dur(seconds: float) -> str:
    """Human-readable wall-clock duration (e.g. ``1h02m03s`` / ``2m05s`` / ``12.3s``)."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{seconds:.1f}s"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    global _QUIET
    args, extras = parse_args(argv if argv is not None else sys.argv[1:])
    _QUIET = args.quiet
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    scanned = scan_mpnn_args(extras)
    run_script = str(args.mpnn_run_script) if args.mpnn_run_script else str(FUSED_MPNN_RUN)

    banner("chisel_ligandMPNN")
    nv = resolve_device(args.device)

    # Feature guards.
    multi = scanned.has_pdb_path_multi
    if multi:
        LOGGER.warning(
            "--pdb_path_multi detected: REMARK 666 auto-fix and protonation are "
            "disabled (single-seed post-processing can't map multiple seeds); "
            "running MPNN passthrough only."
        )
    # ---- PRE (computed once; reused across all sweep runs) ----------------
    pre_base = scanned.out_folder or (
        str(Path(scanned.pdb_path).resolve().parent) if scanned.pdb_path else "."
    )
    pre_dir = Path(pre_base) / "_chisel_pre"

    # Parse REMARK 666 once (used for the fixed-residues JSON, the catalytic
    # table, and the N-term Met-guard "is it fixed?" check).
    catres = (parse_remark_666(scanned.pdb_path, key_by="chain_resno")
              if (scanned.pdb_path and not multi) else {})
    catres_labels: list[str] = []
    for cr in sorted(catres.values(), key=lambda c: (c.chain, c.resno)):
        lab = f"{cr.chain}{cr.resno}"
        if lab not in catres_labels:
            catres_labels.append(lab)

    do_fix = (
        not args.no_fix_remark666_catres
        and not scanned.has_fixed
        and bool(scanned.pdb_path)
        and not multi
    )
    if scanned.has_fixed and not args.no_fix_remark666_catres and not multi:
        say("  fixed residues supplied by user -> skipping REMARK 666 auto-fix")

    # `injected` = flags appended to EVERY run (per-residue omit, etc.).
    # The fixed-residues source is tracked separately because it becomes
    # PER-COMBO when --conserve_hbonds is active (rolled independently per run).
    injected: list[str] = []
    fixed_residues_json_shared: Optional[Path] = None
    if do_fix:
        fixed_residues_json_shared = build_fixed_residues_from_remark666(
            catres, scanned.pdb_path, pre_dir / "fixed_residues_remark666.json")

    # N-terminal Met guard (additive per-residue omit of 'M' at position 1).
    if (not args.no_omit_nterm_met) and scanned.pdb_path and not multi:
        if _has_flag(extras, "--omit_AA_per_residue") or _has_flag(extras, "--omit_AA_per_residue_multi"):
            LOGGER.warning("--omit_AA_per_residue[_multi] supplied by user; skipping "
                           "the automatic N-term Met guard.")
        else:
            nterm = _first_protein_residue(scanned.pdb_path)
            fixed_labels = (set(catres_labels) if do_fix
                            else _parse_user_fixed_labels(extras) if scanned.has_fixed
                            else set())
            if not nterm:
                LOGGER.warning("could not determine the N-terminal residue; skipping "
                               "N-term Met guard.")
            elif f"{nterm[0]}{nterm[1]}" in fixed_labels:
                say(f"  N-term {nterm[0]}{nterm[1]} is fixed -> no Met guard needed")
            else:
                nterm_label = f"{nterm[0]}{nterm[1]}"
                omit_json = build_omit_nterm_met_json(
                    scanned.pdb_path, nterm_label, pre_dir / "omit_AA_per_residue.json")
                injected += ["--omit_AA_per_residue_multi", str(omit_json)]
                say(f"  N-term Met guard: omitting 'M' at {nterm_label} "
                    f"(additive to omit_AA; --no_omit_nterm_met to disable)")

    # ---- H-bond sidechain conservation (opt-in; per-combo fixed residues) ----
    conserve_active = False
    conserve_candidates: list[str] = []
    conserve_base: set[str] = set()
    conserve_prob = 0.8
    conserve_once: set[str] = set()
    conserve_seed_base = args.conserve_seed
    if args.conserve_hbonds and scanned.pdb_path and not multi:
        from protein_chisel.tools.conserved_hbonds import find_conservable_sidechain_hbonds
        conserve_prob = parse_probability(args.conserve_hbond_prob)
        chain = next(iter(catres.values())).chain if catres else "A"
        anchors = {a.strip() for a in (args.conserve_anchors or "").split(",") if a.strip()}
        user_fixed_labels = _parse_user_fixed_labels(extras)
        user_fixed_resnos = {int(l[1:]) for l in user_fixed_labels if l[1:].isdigit()}
        cat_resnos = {cr.resno for cr in catres.values()}
        designable = sorted(_protein_resnos(scanned.pdb_path, chain)
                            - cat_resnos - user_fixed_resnos)
        recs = find_conservable_sidechain_hbonds(
            scanned.pdb_path,
            designable_resnos=designable,
            catalytic_resnos=(cat_resnos if "catalytic" in anchors else set()),
            user_fixed_resnos=(user_fixed_resnos if "user_fixed" in anchors else set()),
            include_ligand=("ligand" in anchors),
            chain=chain,
            max_dist=args.conserve_hbond_max_dist,
            max_angle_deg=args.conserve_hbond_max_angle,
        )
        banner("H-BOND SIDECHAIN CONSERVATION")
        for r in sorted(recs, key=lambda r: (r.resno, r.distance)):
            ptag = f"{r.partner_resname}{r.partner_resno if r.partner_resno > 0 else ''}"
            flag = f"  [CLASH {r.clash_with}]" if r.clashes else ""
            say(f"  A{r.resno} {r.resname} {r.sidechain_atom} <-> {r.partner_kind} {ptag} "
                f"{r.partner_atom}  d={r.distance} {r.strength_bin}  "
                f"donor={r.hypothesized_donor} acceptor={r.hypothesized_acceptor}{flag}")
        by_res: dict[int, list] = {}
        for r in recs:
            by_res.setdefault(r.resno, []).append(r)
        cand_resnos: list[int] = []
        for resno, rs in sorted(by_res.items()):
            if rs[0].clashes and not args.conserve_keep_clashing:
                say(f"  excluding A{resno}: sidechain clashes with fixed backbone/ligand "
                    f"({rs[0].clash_with})  [--conserve_keep_clashing to keep]")
                continue
            cand_resnos.append(resno)
        conserve_candidates = [f"{chain}{rn}" for rn in cand_resnos]
        conserve_base = set(catres_labels) | set(user_fixed_labels)
        # The per-combo JSON is the sole fixed-residues source while conserving;
        # drop the shared one and any user-supplied fixed flags (folded into base).
        fixed_residues_json_shared = None
        extras = _strip_flags(extras, ["--fixed_residues", "--fixed_residues_multi"])
        conserve_active = True
        # Always have an explicit seed so every round's roll is reproducible
        # after the fact: use --conserve_seed if given, else auto-generate one
        # and tell the user how to replay it.
        if conserve_seed_base is None:
            conserve_seed_base = random.SystemRandom().randint(1, 2**31 - 1)
            say(f"  no --conserve_seed given; using auto seed {conserve_seed_base} "
                f"(rerun with --conserve_seed {conserve_seed_base} for identical rolls)")
        say(f"  conserving {len(conserve_candidates)} candidate residue(s) "
            f"{conserve_candidates} at p={conserve_prob:.2f} "
            f"({'one decision for all runs' if args.conserve_hbond_all_or_none else 'independent per run'}"
            f", seed={conserve_seed_base})")
        if args.conserve_hbond_all_or_none:
            conserve_once = _roll_conserved(
                conserve_candidates, conserve_prob, random.Random(str(conserve_seed_base)))

    runs, sweeping = parse_sweep(args, scanned)
    base_out = scanned.out_folder

    # ---- Pre-run plan: how many PDBs + per-combo breakdown ----------------
    say("")
    say(f"  PLAN: {len(runs)} MPNN run(s)"
        + ("" if sweeping else " (single passthrough)")
        + f"  ->  output dir: {base_out or '(run.py default)'}")
    total = 0
    for run in runs:
        d = _designs(run)
        total += d
        say(f"      [{run.run_tag or 'run'}] temperature={run.temperature} "
            f"number_of_batches={run.number_of_batches} batch_size={run.batch_size} "
            f"enhance={run.enhance or '(none)'}  ~{d} packed PDB(s)"
            + (f"  suffix={run.packed_suffix}" if sweeping else ""))
    say(f"      => ~{total} packed PDB(s) total, flattened into {base_out or '(run.py default)'}")
    if scanned.pdb_path and not args.no_copy_input_structure:
        say(f"      (+ a copy of the input structure: {Path(scanned.pdb_path).name})")

    if base_out and not args.dry_run:
        Path(base_out).mkdir(parents=True, exist_ok=True)

    # ---- Run MPNN (one apptainer exec per combo) --------------------------
    results: list[dict] = []
    any_ok = False
    for i, run in enumerate(runs, 1):
        banner(f"RUN {i}/{len(runs)}" + (f"  [{run.run_tag}]" if run.run_tag else ""))
        if sweeping:
            say(f"  temperature      = {run.temperature}")
            say(f"  number_of_batches= {run.number_of_batches}")
            say(f"  batch_size       = {run.batch_size}")
            say(f"  enhance          = {run.enhance if run.enhance else '(none)'}")
            say(f"  omit_AA          = {run.omit_AA}")
            say(f"  bias_AA          = {run.bias_AA}")
            say(f"  packed_suffix    = {run.packed_suffix}")
            say(f"  staging dir      = {run.out_folder}")
        # Per-run fixed-residues source: when conserving, roll candidates for THIS
        # combo and write a combined (base ∪ rolled) JSON; else use the shared one.
        rolled: set[str] = set()
        seed_str = ""
        if conserve_active:
            if args.conserve_hbond_all_or_none:
                rolled, seed_str = conserve_once, str(conserve_seed_base)
            else:
                seed_str = f"{conserve_seed_base}:{i}"
                rolled = _roll_conserved(conserve_candidates, conserve_prob,
                                         random.Random(seed_str))
            combined = sorted(conserve_base | rolled, key=_label_sort_key)
            combo_json = pre_dir / f"fixed_residues_{run.run_tag or f'run{i}'}.json"
            _fixed_residues_json(scanned.pdb_path, combined, combo_json)
            fixed_inject = ["--fixed_residues_multi", str(combo_json)]
            if conserve_candidates:
                say(f"  conserve roll (p={conserve_prob:.2f}, seed={seed_str}): fixing "
                    f"{sorted(rolled, key=_label_sort_key) or '(none)'} of {conserve_candidates}")
        else:
            fixed_inject = (["--fixed_residues_multi", str(fixed_residues_json_shared)]
                            if fixed_residues_json_shared else [])
        run_extras = build_run_extras(extras, run, injected + fixed_inject, sweeping)
        inner = ["python", run_script, *run_extras]
        binds = _bind_dirs(
            files=[scanned.pdb_path],
            dir_paths=[run.out_folder, base_out, str(pre_dir)],
        )
        cmd = assemble_apptainer_command(inner, binds, args.sif, nv)
        t_run = time.time()
        rc = run_streaming(cmd, args.dry_run, label="mpnn")
        elapsed = time.time() - t_run
        status = "dry-run" if args.dry_run else ("ok" if rc == 0 else f"FAIL({rc})")
        if status == "ok":
            any_ok = True
        rec = {"run": i, "tag": run.run_tag or "-", "mpnn": status, "elapsed": elapsed}
        if conserve_active:
            rec["fixed"] = sorted(rolled, key=_label_sort_key)
            rec["seed"] = seed_str
        results.append(rec)

    # ---- Post-process: flatten -> protonate -> transfer REMARKs -> copy input
    subdir = resolve_protonate_subdir(scanned, args)
    n_pdbs, prot_status, input_copy = 0, "off", None
    do_remarks = scanned.pdb_path and not multi and not (
        args.no_transfer_remarks and args.no_design_path_remark)

    if args.dry_run:
        banner("POST-PROCESS (dry-run preview)")
        say(f"  would flatten each run's {subdir}/*.pdb into {base_out} (one layer)"
            f" and remove seqs/backbones/packed"
            + ("" if args.keep_intermediates else " + the _chisel_pre JSONs"))
        if (not args.no_protonate) and base_out and not multi:
            if catres and not args.no_restore_catalytic:
                say(f"  would restore catalytic states (HIS tautomer/KCX/cat-H) for "
                    f"{sorted({cr.resno for cr in catres.values()})} from input")
            if args.ligand_params:
                say("  protonation: holo (with --ligand_params)")
                protonate_dir(base_out, args.seed_pdb or scanned.pdb_path,
                              list(args.ligand_params), args)
            elif not args.no_apo_protonate_fallback:
                say("  protonation: APO FALLBACK (no params) — protonate protein, "
                    "ligand re-added from input")
                protonate_dir(base_out, args.seed_pdb or scanned.pdb_path, [], args)
            else:
                say("  protonation would skip (no --ligand_params, apo fallback off)")
        if do_remarks:
            bits = []
            if not args.no_transfer_remarks:
                bits.append("transfer input REMARK lines")
            if not args.no_design_path_remark:
                bits.append("add 'REMARK DESIGN_PATH chisel_ligandmpnn output <path>'")
            say("  would " + " + ".join(bits) + " on each output PDB")
        if scanned.pdb_path and base_out and not args.no_copy_input_structure:
            say(f"  would copy input -> {Path(base_out) / Path(scanned.pdb_path).name}")
            if not multi and not args.no_transfer_remarks:
                say("    and (re)build its REMARK 668 block (states + --ptm)")
    elif base_out and any_ok:
        banner("POST-PROCESS")
        n_pdbs = finalize_outputs(base_out, runs, sweeping, subdir, args.keep_intermediates)
        if multi:
            prot_status = "skipped (pdb_path_multi)"
        elif args.no_protonate:
            prot_status = "off (--no_protonate)"
        elif (not args.ligand_params) and args.no_apo_protonate_fallback:
            LOGGER.warning("no --ligand_params and apo fallback disabled; "
                           "SKIPPING protonation (PDBs left raw).")
            prot_status = "skipped (no params)"
        else:
            params = list(args.ligand_params) if args.ligand_params else []
            mode = "holo" if params else "apo-fallback"
            say(f"  protonation mode: {mode}"
                + ("" if params else " (no params: protein protonated, ligand "
                   "re-added from input)"))
            # Restore catalytic HIS tautomers / KCX / cat-H from the input so the
            # catalytic residues protonate correctly (LigandMPNN emits plain HIS).
            if catres and not args.no_restore_catalytic:
                restore_catalytic_states_in_dir(
                    base_out, args.seed_pdb or scanned.pdb_path,
                    sorted({cr.resno for cr in catres.values()}),
                    next(iter(catres.values())).chain)
            pres = protonate_dir(base_out, args.seed_pdb or scanned.pdb_path, params, args)
            base_status = (f"{pres.get('overwritten', 0)}/{pres.get('total', 0)}"
                           if "overwritten" in pres
                           else pres.get("skipped") or f"FAIL({pres.get('failed')})")
            prot_status = f"{mode} {base_status}"
        # REMARK transfer + DESIGN_PATH on the design PDBs (before the input copy).
        if do_remarks:
            n = remarks.transfer_remarks_to_dir(
                base_out, scanned.pdb_path,
                transfer_input=not args.no_transfer_remarks,
                design_path_stage=(None if args.no_design_path_remark
                                   else "chisel_ligandmpnn"))
            bits = []
            if not args.no_transfer_remarks:
                bits.append("transferred input REMARK lines")
            if not args.no_design_path_remark:
                bits.append("added REMARK DESIGN_PATH chisel_ligandmpnn")
            say(f"  {' + '.join(bits)} on {n} design PDB(s)")
        if scanned.pdb_path and not args.no_copy_input_structure:
            input_copy = copy_input_structure(scanned.pdb_path, base_out)
            # Ensure the copied input carries a correct REMARK 668 block (built
            # from its own catalytic H + any --ptm overrides); add if missing,
            # override if stale.
            if input_copy and not multi and not args.no_transfer_remarks:
                annotate_input_copy_remark668(input_copy, args.ptm)

    # ---- Summary ----------------------------------------------------------
    banner("SUMMARY")
    say(f"  {'run':<4} {'tag':<18} {'mpnn':<10} {'time':<9}")
    for r in results:
        tm = "-" if args.dry_run else _fmt_dur(r["elapsed"])
        say(f"  {r['run']:<4} {r['tag']:<18} {r['mpnn']:<10} {tm:<9}")
    if not args.dry_run and len(results) > 1:
        say(f"  {'':<4} {'(total)':<18} {'':<10} {_fmt_dur(sum(r['elapsed'] for r in results)):<9}")

    if conserve_active:
        say("")
        say(f"  H-bond conservation: {len(conserve_candidates)} conservable "
            f"residue(s) found{' ' + str(conserve_candidates) if conserve_candidates else ''}")
        say(f"    seed base: {conserve_seed_base}"
            + ("" if args.conserve_seed is not None
               else f"  (auto; rerun with --conserve_seed {conserve_seed_base} to reproduce)"))
        for r in results:
            fixed = r.get("fixed", [])
            say(f"    run {r['run']} [{r['tag']}]  seed={r.get('seed')}  "
                f"fixed {len(fixed)}/{len(conserve_candidates)}: {fixed or '(none)'}")

    if not args.dry_run:
        say("")
        say(f"  output dir  : {base_out}")
        say(f"  packed PDBs : {n_pdbs}")
        say(f"  protonation : {prot_status}")
        say(f"  input copy  : {Path(input_copy).name if input_copy else '(none)'}")

    failed = sum(1 for r in results if r["mpnn"].startswith("FAIL"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
