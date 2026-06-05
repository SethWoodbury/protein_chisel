# Pluggable design experts

An **expert** produces a calibrated `(L, 20)` per-position amino-acid
log-probability array for a scaffold. The fusion
(`sampling/plm_fusion.fuse_experts`) combines N experts into the additive
LigandMPNN bias using our calibration (log-odds vs UniProt background →
entropy-match → structural class weights → shrink-at-disagreement).

The registry makes this **plug-and-play**: select experts by name; add a new
model without touching the fusion math.

> Multi-expert registry / product-of-experts design borrowed from Sebastian
> (sebols) and Joe Mi's `fused_mpnn_poe`. HERMES (Phase 4) from Visani et al.,
> Nourmohammad lab, UW.

## Default behavior (unchanged)

`--experts esmc,saprot` (the default) reproduces the legacy two-PLM fusion
**byte-for-byte**: `fuse_experts` delegates to the untouched `fuse_plm_logits`,
and `precompute_plm_artifacts.py` writes the same `esmc_log_probs.npy` /
`saprot_log_probs.npy` / `fusion_bias.npy` / `fusion_log_odds_*.npy` /
`fusion_weights.npy`. Running the pipeline with no new env/flags is identical to
before.

## Selecting experts

| Where | Knob | Default |
|---|---|---|
| `run_chisel_design.sh` | `EXPERTS=esmc,saprot` env | `esmc,saprot` |
| `precompute_plm_artifacts.py` | `--experts esmc,saprot` | `esmc,saprot` |
| `iterative_design.py` | `--experts esmc,saprot` | `esmc,saprot` |

The `--experts` value must match between precompute (writes `<name>_log_probs.npy`)
and the driver (loads them). The shell only emits `--experts` when it differs
from `esmc,saprot`, so the default command line is unchanged.

Per-expert model variants still come from `--esmc_model` / `--saprot_model`
(env `ESMC_MODEL` / `SAPROT_MODEL`).

## Adding a new expert

1. Subclass `protein_chisel.experts.base.Expert`:
   ```python
   class MyExpert(Expert):
       name = "myexpert"          # registry key + cache prefix (<name>_log_probs.npy)
       modality = "structure"     # or "sequence"
       def __init__(self, model_name=None): ...
       @property
       def version(self): return f"myexpert:{self.model_name}"
       def compute_log_probs(self, ctx: ExpertContext) -> np.ndarray:  # (L, 20)
           # lazy-import heavy deps here; read ctx.seq / ctx.pdb_path / ctx.chain
           ...
   ```
2. Register it in `experts/registry.py` `_FACTORIES` (or call `register_expert`).
3. Run with `--experts esmc,saprot,myexpert`.

`Expert.log_probs` caches `<out_dir>/<name>_log_probs.npy` (keyed by the
artifact dir) and computes one model at a time (load → compute → free) for memory
discipline. Heavy deps (torch/esm) are lazy-imported, so importing the registry
is cheap.

## Per-expert calibration knobs (`FusionConfig`)

Defaults are no-ops (preserve current behavior); set to tune a non-default set:
- `expert_weights={"hermes": 0.25}` — global down-weight (e.g. avoid
  double-counting structure MPNN already sees).
- `expert_class_weights={"hermes": {"primary_sphere": 0.0}}` — mute an expert at
  certain position classes.
- `expert_temperatures={"esmc": 1.5}` — extra temperature on an expert's log-odds.

A zero-weight expert still participates in entropy-match + shrink calibration; to
fully exclude an expert, omit it from `--experts`.
