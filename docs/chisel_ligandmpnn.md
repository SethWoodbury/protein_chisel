# `chisel_ligandMPNN.py` — LigandMPNN wrapper

A standalone, host-side wrapper around the fused_mpnn LigandMPNN `run.py`. It runs
MPNN **exactly** like a raw `apptainer exec … run.py …` command (every LigandMPNN
flag is passed straight through) while optionally layering on catalytic-site
fixing, protonation, a hyper-parameter sweep, probabilistic H-bond conservation,
REMARK provenance, and output de-duplication + diversity reporting.

- **Source:** [`scripts/chisel_ligandMPNN.py`](../scripts/chisel_ligandMPNN.py)
- **Tests:** [`tests/scripts/test_chisel_ligandmpnn_helpers.py`](../tests/scripts/test_chisel_ligandmpnn_helpers.py),
  [`tests/tools/test_conserved_hbonds.py`](../tests/tools/test_conserved_hbonds.py),
  [`tests/tools/test_remarks.py`](../tests/tools/test_remarks.py)
- **Container:** drives MPNN inside `universal.sif` (the fused_mpnn build) and
  protonation inside `pyrosetta.sif`. It is a thin orchestrator — **it does not
  submit SLURM jobs** (see [Running at scale](#running-at-scale)).

> **How it differs from the production pipeline.** `run_chisel_design.sh` /
> `iterative_design.py` is the full ESM-C + SaProt PLM-fusion *iterative* design
> loop (see [architecture.md](architecture.md)). `chisel_ligandMPNN.py` is the
> opposite end of the spectrum: a single-shot (or single-sweep) **plain
> LigandMPNN** run over one input PDB, with the surrounding bookkeeping
> (catalytic fixing, protonation, REMARKs, dedup) that you'd otherwise do by
> hand. Reach for it when you want raw MPNN designs, fast, without the loop.

---

## Pipeline (per input PDB)

```
PRE ─────────────►  MPNN (once per sweep combo) ─────►  POST ───────────────►  REPORT
fix REMARK 666      run.py inside universal.sif         flatten → protonate     summary:
catalytic residues  (catalytic site + conserved         → restore catalytic     per-run timing,
omit N-term Met     H-bonds held fixed per combo)        HIS/KCX → transfer      conservation rolls,
(+ conserve H-bonds)                                     REMARKs + DESIGN_PATH   dedup + diversity
                                                         → copy input → dedup
```

Every step except the MPNN call itself is toggleable. With everything off
(`--no_fix_remark666_catres --no_protonate --no_copy_input_structure
--no_transfer_remarks --no_design_path_remark --no_omit_nterm_met
--no_dedup_sequences`) the wrapper is a transparent passthrough to `run.py`.

---

## Quick start

```bash
# Reproduce a raw run.py command verbatim (no pre/post):
python scripts/chisel_ligandMPNN.py --no_fix_remark666_catres --no_protonate \
    --model_type ligand_mpnn --pdb_path in.pdb --out_folder out \
    --temperature 0.2 --number_of_batches 15 --batch_size 1 \
    --pack_side_chains 1 --omit_AA CX

# Auto-fix catalytic residues + protonate (apo fallback) + sweep three combos:
python scripts/chisel_ligandMPNN.py --device auto \
    --model_type ligand_mpnn --pdb_path in.pdb --out_folder out \
    --pack_side_chains 1 --omit_AA CX --bias_AA 'K:-0.5,R:-0.75' \
    --run 't=0.1;n=15' --run 't=0.2;n=15;bs=2' --run 't=0.3;n=10;bias=K:-1.0'

# Holo protonation (with params) + conserve sidechain H-bonds to the ligand:
python scripts/chisel_ligandMPNN.py --device auto \
    --pdb_path in.pdb --out_folder out --ligand_params lig.params \
    --model_type ligand_mpnn --pack_side_chains 1 \
    --conserve_hbonds --conserve_hbond_prob 0.8 \
    --run 'n=10;t=0.1'
```

Add `--dry_run` to any command to print the exact apptainer command(s) and a
full plan/summary **without** running MPNN. The conservation rolls are computed
identically in dry-run, so it is a faithful preview of which residues would be
fixed.

---

## Flags

All **unrecognized** flags pass straight through to `run.py`, so drive MPNN
(`--model_type`, `--temperature`, `--pack_side_chains`, `--bias_AA`, …) exactly
as you would raw. The wrapper-specific flags below are layered on top.

### Pre-processing

| Flag | Default | Effect |
|---|---|---|
| `--no_fix_remark666_catres` | off (fixing **on**) | Don't auto-fix catalytic residues. Default: if you didn't pass `--fixed_residues[_multi]`, parse the input's REMARK 666 motifs and inject them as `--fixed_residues_multi`. No REMARK 666 → warn + full redesign. |
| `--no_omit_nterm_met` | off (guard **on**) | Allow Met at residue 1. Default: omit `M` at the N-terminal residue (a Met tag is added at expression), unless that residue is fixed. Additive to `omit_AA`. |

### H-bond sidechain conservation (opt-in)

| Flag | Default | Effect |
|---|---|---|
| `--conserve_hbonds` | off | Enable: detect designable-residue **sidechain** H-bonds to anchors and probabilistically pin those residues into the fixed list. |
| `--conserve_hbond_prob P` | `0.8` | Per-residue fix probability. `0–1`, or a percentage `>1` (e.g. `80`) auto-converted with a warning. |
| `--conserve_hbond_all_or_none` | off | Roll each candidate **once** and apply to all sweep combos (default: roll **independently per combo**). |
| `--conserve_seed N` | random | Seed the rolls for reproducibility. If omitted, an explicit seed is auto-generated and printed so any run is replayable. |
| `--conserve_anchors …` | `ligand,catalytic,user_fixed` | Comma subset of anchor groups a sidechain must H-bond to. |
| `--conserve_hbond_max_dist D` | `3.9` | Heavy-atom donor···acceptor distance cutoff (Å). |
| `--conserve_hbond_max_angle DEG` | `90` | Antecedent–D–A angle gate; larger = more permissive (stock detector uses 70). |
| `--conserve_keep_clashing` | off | Keep candidates whose sidechain clashes with fixed backbone / ligand (default: exclude with a warning). |

See [H-bond sidechain conservation](#h-bond-sidechain-conservation) for the
mechanism.

### Protonation (POST)

| Flag | Default | Effect |
|---|---|---|
| `--no_protonate` | off (protonation **on**) | Skip protonation; leave raw MPNN PDBs. |
| `--ligand_params P …` | none | One or more `.params` → **holo** protonation (full protein + ligand). |
| `--no_apo_protonate_fallback` | off (fallback **on**) | If no `--ligand_params`: default strips the ligand, protonates the **apo** protein with PyRosetta, then re-adds the ligand from the input. This flag disables that fallback (→ skip protonation instead). |
| `--no_restore_catalytic` | off (restore **on**) | Don't restore catalytic HIS tautomer / KCX / catalytic-H from the input before protonating. Default restores them so catalytic residues protonate correctly (MPNN emits plain `HIS`). |
| `--seed_pdb PDB` | `--pdb_path` | Protonation reference (REMARK 666 + ligand source). |
| `--protonate_subdir DIR` | auto | Which `run.py` subdir to protonate (auto: packed/backbones). |
| `--ptm SPEC` | none | REMARK 668 PTM annotation, e.g. `A/LYS/4:KCX`. Annotation-only; the residue keeps its unmodified identity. |

### REMARK provenance & output layout

| Flag | Default | Effect |
|---|---|---|
| `--no_copy_input_structure` | off (copy **on**) | Don't copy the input PDB into `--out_folder`. |
| `--no_transfer_remarks` | off (transfer **on**) | Don't transfer input REMARK lines (666 etc.) onto each design, and don't (re)build the input copy's REMARK 668. |
| `--no_design_path_remark` | off (stamp **on**) | Don't add `REMARK DESIGN_PATH chisel_ligandmpnn output <path>`. |
| `--keep_intermediates` | off | Keep the sequence FASTAs and the `_chisel_pre` JSONs (flattened into `--out_folder`). Default: only the packed PDBs (+ input copy). |
| `--suffix_base S` | `""` | Prepended to every per-combo `--packed_suffix` tag. |

### Output de-dup & diversity

| Flag | Default | Effect |
|---|---|---|
| `--no_dedup_sequences` | off (dedup **on**) | Don't de-duplicate identical output sequences. Default: collapse duplicate sequences in the flat output dir. The **copied input is never removed** and always wins a tie; among MPNN-vs-MPNN duplicates the **oldest** is kept. |
| `--diversity_ligand_cutoff D` | `8.0` | Heavy-atom distance (Å) defining "near-ligand" positions for the pocket diversity metric in the summary. |

### Sweep

| Flag | Effect |
|---|---|
| `--run 'k=v;k=v'` (repeatable) | One sweep combo. Fields separated by **`;`** (so `bias=K:-0.5,R:-0.75` keeps its commas). Keys: `t`/`temperature`, `n`/`number_of_batches`, `bs`/`batch_size`, `enh`/`enhance` (or `none`), `omit`/`omit_AA`, `bias`/`bias_AA`, `tag`. Each combo overrides only the keys it names; everything else inherits your universal/base value. |
| `--sweep_config FILE.json` | A JSON list of combo objects (preferred for notebook/rich use). |

### Device / execution

| Flag | Default | Effect |
|---|---|---|
| `--device {auto,cpu,gpu}` | `auto` | `auto` uses the GPU (`--nv`) if `nvidia-smi` succeeds, else CPU. MPNN runs fine on CPU. **Independent of any SLURM partition** — this is where MPNN runs *inside* the job. |
| `--sif PATH` | `universal.sif` | Override the MPNN container. |
| `--mpnn_run_script PATH` | fused_mpnn `run.py` | Override the run.py. |
| `--quiet` | off | Less chatter. |
| `--dry_run` | off | Print the plan + apptainer commands + summary; run nothing. |

---

## Feature deep-dives

### Catalytic-site fixing (REMARK 666)

When you don't supply your own fixed residues, the wrapper parses the input
PDB's REMARK 666 catalytic-motif lines and writes a `--fixed_residues_multi`
JSON so the catalytic site is held during design. If you *do* pass
`--fixed_residues[_multi]`, it's respected and auto-fixing is skipped. No REMARK
666 in the PDB → a loud warning and a full unfixed redesign.

### N-terminal Met guard

Expression adds an N-terminal Met tag, so designing `M` at residue 1 is
wasteful. The wrapper injects an additive per-residue `omit` of `M` at the
N-terminal residue — unless that residue is itself fixed, or you passed your own
`--omit_AA_per_residue[_multi]`, or `--no_omit_nterm_met`.

### H-bond sidechain conservation

Designable (unfixed) residues whose **sidechain** hydrogen-bonds the ligand or a
fixed residue are often functionally important. With `--conserve_hbonds` the
wrapper detects them and probabilistically pins them into the fixed list.

- **Detection** ([`tools/conserved_hbonds.py`](../src/protein_chisel/tools/conserved_hbonds.py))
  is **heavy-atom only** — donor···acceptor distance + an antecedent-based angle,
  reusing the H-bond geometry in `tools/geometric_interactions.py`. No reliance
  on explicit/relaxed hydrogens.
- The bond must use the **designable residue's sidechain** (a backbone N/O bond
  doesn't count — the sidechain is still freely designable). The **anchor**
  (ligand / catalytic / user-fixed) may use sidechain *or* backbone.
- **Histidine is tautomer-agnostic**: both `ND1` and `NE2` are treated as donor
  *and* acceptor (a non-catalytic His may be either tautomer, or HIP).
- Each bond is **strength-binned** (`super_strong … super_weak`) from the
  distance-based Gaussian strength.
- **Clash filter**: candidates whose current rotamer clashes with anything that
  won't be redesigned (fixed/all backbone, ligand, fixed sidechains) are
  excluded by default (kept with `--conserve_keep_clashing`). Clashes against
  *other designable* sidechains are ignored — those get redesigned anyway.
- **Per-combo rolls**: each sweep combo independently rolls each candidate at
  `P` and writes a combined (`catalytic ∪ user-fixed ∪ rolled`) fixed-residues
  JSON for that combo. `--conserve_hbond_all_or_none` rolls once for all combos.
- **Reproducibility**: with `--conserve_seed N`, round *i* uses seed `N:i`
  (or `N` under `--all_or_none`) — bit-for-bit reproducible. With **no** seed,
  the wrapper auto-generates a base seed and prints it (`rerun with
  --conserve_seed <N> …`), so *any* run can be replayed afterward.

### Protonation

After the MPNN runs, designs are protonated **in place**:

- **holo** — with `--ligand_params`: full protein + ligand.
- **apo fallback** (default when no params) — strip the ligand, protonate the
  apo protein with PyRosetta, then paste the ligand back from the input.
- Before protonating, catalytic **HIS tautomers / KCX / catalytic-H** are
  restored from the input (`--no_restore_catalytic` to skip) because MPNN emits
  plain `HIS`. Verified to reproduce the input's catalytic protonation states.

### REMARK transfer, DESIGN_PATH & dedup

[`tools/remarks.py`](../src/protein_chisel/tools/remarks.py) (shared with
`iterative_design.py`) rescues REMARK lines from the input onto each design,
writes a canonical header order, drops PyRosetta `REMARK 0` dump artifacts, and
appends a `REMARK DESIGN_PATH chisel_ligandmpnn output <path>` provenance line.
It also **de-duplicates** REMARK header lines — prefix-aware for numbered
REMARKs (folds col-80 truncations and legend drift like `…anchors` vs
`…anchors.` into the complete variant; safe for equal-width 666/668 data),
exact-only for grouped lines so nested `DESIGN_PATH` provenance is preserved.
The copied input gets a correct REMARK 668 block (states + `--ptm`), added if
missing or overridden if stale.

### Output sequence de-dup

After the flat output dir is assembled (designs + input copy), identical
sequences are collapsed:

- The **copied input is never removed**, and always wins a tie — an MPNN design
  matching the input's sequence is dropped instead.
- Among MPNN-vs-MPNN duplicates, the **oldest is kept** (by mtime, then name)
  and newer ones removed.
- Sequences are read **independent of protonation/PTM state** (HID/HIE/HIP→H,
  KCX→K, …), so two PDBs differing only in protonation aren't treated as
  distinct.

### Diversity reporting

Computed over the **unique design sequences** (the input copy excluded), reusing
`hamming_distance` from [`scoring/diversity.py`](../src/protein_chisel/scoring/diversity.py):

- **Full-sequence** mean pairwise Hamming (+ % of positions).
- **Pocket** mean pairwise Hamming restricted to positions within
  `--diversity_ligand_cutoff` Å (default 8) of any ligand heavy atom.

Pocket diversity is typically lower than full — the active site is the most
constrained region (catalytic residues fixed, H-bonds conserved).

### Run summary

The final `SUMMARY` banner reports:

- a per-run table with **wall-clock time** per combo (and a total),
- the **H-bond conservation** block — candidates found, the seed base, and
  per-round seed + which residues were fixed,
- **seq dedup** (how many duplicate PDBs removed, unique designs remaining), and
- **diversity** (full + pocket).

---

## Output layout

`--out_folder` ends as a single flat layer:

```
out_folder/
├── <stem><suffix>_<i>_<c>.pdb     # packed designs (one per sweep sample)
└── <stem>.pdb                     # a copy of the input (REMARK 668 rebuilt)
```

`run.py`'s `seqs/`, `backbones/`, and per-combo staging subdirs are removed.
`--keep_intermediates` keeps the (tag-qualified) sequence FASTAs and the
`_chisel_pre` JSONs, still flat. `--no_copy_input_structure` suppresses the input
copy.

---

## Running at scale

The wrapper **does not submit SLURM jobs** — it builds and runs `apptainer`
commands. To fan out across many input PDBs, a notebook/driver cell builds **one
wrapper command per input PDB** (one combo sweep inlined via repeated `--run`)
and submits them as an array via `nb.submit_array_job`. So:

- **one array task = one wrapper invocation = one input PDB** (→ its own
  `<out>/<stem>/` dir, its own sweep combos).
- Pre/post, conservation, dedup, and diversity are therefore **per input PDB**,
  computed at the end of that task — **not** a campaign-wide pass across every
  design. (A global dedup/diversity across all inputs would be a separate
  post-array step.)

---

## Gotchas

- **Literal `--pdb_path` in injected JSONs.** The fixed-residues / omit-AA JSONs
  are keyed by the **literal** `--pdb_path` string, *not* `Path.resolve()`.
  `run.py` looks up `fixed_residues_multi[args.pdb_path]` verbatim, and on
  compute nodes scratch resolves through a symlink (`/net/scratch` →
  `/mnt/net/scratch`); a resolved key would not match → `KeyError` that fails
  every array task at the MPNN step. (Fixed; regression-tested.)
- **Apptainer `PYTHONPATH`.** `universal.sif` needs `/cifutils/src` on
  `PYTHONPATH`; the wrapper preserves it via the ApptainerCall keeper so it isn't
  clobbered by `--env PYTHONPATH=/code/src`.
- **`--pdb_path_multi`.** With multi-seed input, REMARK 666 auto-fix and
  protonation are disabled (single-seed post-processing can't map multiple
  seeds) — MPNN passthrough only.
- **`--device` vs SLURM partition.** `--device` controls where MPNN runs *inside*
  the job; it's unrelated to which SLURM queue you submit to. `auto` is safe for
  both CPU and GPU nodes.

---

## Reusable building blocks

These were built for / extracted from the wrapper and are importable on their own:

| Module | What it does |
|---|---|
| [`tools/conserved_hbonds.py`](../src/protein_chisel/tools/conserved_hbonds.py) | `find_conservable_sidechain_hbonds(...)` → `list[ConservableHbond]`. Heavy-atom detection of designable sidechain H-bonds to ligand/fixed anchors, with strength bins, His tautomer-agnostic handling, and clash flags. Host-side (reuses `geometric_interactions`). |
| [`tools/remarks.py`](../src/protein_chisel/tools/remarks.py) | `reorganize_pdb_remarks` / `transfer_remarks_to_dir` / `replace_remark_block` — transfer, canonical-order, de-dup, and `DESIGN_PATH`-stamp PDB REMARK headers. Pure-Python (no PyRosetta); shared with `iterative_design.py`. |
