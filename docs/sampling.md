# Sampling — PLM fusion, biased MPNN, ligand_mpnn wrapper

`src/protein_chisel/sampling/` is the small layer between PLM logits and the MPNN sampler. The asymmetry between MPNN's autoregressive logits and PLM masked-LM marginals is the key thing this layer handles correctly (see [docs/architecture.md](architecture.md) "Logit fusion — and an important asymmetry between samplers").

---

## `plm_fusion.py` — calibrate + fuse ESM-C and SaProt logits

[src/protein_chisel/sampling/plm_fusion.py](../src/protein_chisel/sampling/plm_fusion.py)

Both ESM-C and SaProt produce masked-LM marginals — `p(aa_i | seq_minus_i)` (ESM-C) and `p(aa_i | seq_minus_i, 3Di_minus_i)` (SaProt). They live in the same conditioning regime, so they fuse cleanly. The fusion is **not** a peer to LigandMPNN's autoregressive logits; it gets fed to MPNN as an additive `--bias_AA_per_residue` matrix.

### Pipeline (in `fuse_plm_logits`)

```
log_probs_esmc, log_probs_saprot   ->   bias  (L, 20)
   (L, 20) each in PLM_AA_ORDER
```

1. **Log-odds calibration** — subtract the AA background frequency:
   `log p(aa | ctx) − log p_background(aa)`.
   This decouples "AA is rare everywhere" (background) from "AA is wrong here" (context-specific). Default `aa_background = UNIPROT_AA_BG` (Swiss-Prot 2024) at [plm_fusion.py:33](../src/protein_chisel/sampling/plm_fusion.py#L33).

   Implementation: `calibrate_log_odds(log_probs, aa_bg)` ([plm_fusion.py:76](../src/protein_chisel/sampling/plm_fusion.py#L76)).

2. **Entropy match** — rescale each model's logits by a multiplier `m` so their median per-position entropies are equal.
   - Target: geometric mean `H_target = sqrt(H_a · H_b)`.
   - Multipliers: `m_a = H_a / H_target`, `m_b = H_b / H_target`.
   - `m > 1` → multiplying logits sharpens (lowers entropy); `m < 1` → softens.
   - Implementation: `entropy_match_temperature` ([plm_fusion.py:99](../src/protein_chisel/sampling/plm_fusion.py#L99)).

3. **Position-class–dependent weights** (β for ESM-C, γ for SaProt). Defaults from `FusionConfig.class_weights`:

   | class | weight |
   |---|---|
   | `active_site` | 0.0 (frozen — no PLM bias) |
   | `first_shell` | 0.05 |
   | `pocket` | 0.10 |
   | `buried` | 0.30 |
   | `surface` | 0.50 |
   | `ligand` | 0.0 |

4. **Shrinkage at disagreement** — per-position cosine similarity in probability space ([plm_fusion.py:123](../src/protein_chisel/sampling/plm_fusion.py#L123)). Where `cos < shrink_threshold` (default 0.7), weight is scaled by `max(cos, 0)`. Both β and γ shrink (the PLMs are signaling low confidence; MPNN should dominate at disputed positions).

### Final formula

```
bias[i, j] = β[i] · log_odds_esmc[i, j]  +  γ[i] · log_odds_saprot[i, j]
```

with `β`, `γ` already entropy-matched and shrunk.

### `FusionResult` ([plm_fusion.py:67](../src/protein_chisel/sampling/plm_fusion.py#L67))

| field | shape | content |
|---|---|---|
| `bias` | `(L, 20)` | additive bias, ready for LigandMPNN's `--bias_AA_per_residue_multi` |
| `log_odds_esmc` | `(L, 20)` | calibrated ESM-C log-odds (after entropy-match) |
| `log_odds_saprot` | `(L, 20)` | calibrated SaProt log-odds (after entropy-match) |
| `weights_per_position` | `(L, 2)` | final β, γ per position (post-shrinkage) |
| `config` | `FusionConfig` | — |

### Tests

[tests/test_plm_fusion.py](../tests/test_plm_fusion.py) — host-only. Checks log-odds, entropy match, cosine similarity, fusion shape rules, class-weight ordering, shrinkage at disagreement.

---

## `biased_mpnn.py` — orchestrator

[src/protein_chisel/sampling/biased_mpnn.py](../src/protein_chisel/sampling/biased_mpnn.py)

Thin layer over `plm_fusion.fuse_plm_logits` + `tools.ligand_mpnn.sample_with_ligand_mpnn`. Takes pre-computed PLM log-probs, builds the fused bias, fixes catalytic residues, calls MPNN.

```python
result = biased_sample(
    pdb_path=pdb,
    ligand_params=params,        # accepted; modern fused_mpnn ignores
    position_table=pt,           # rows must include 'class' and 'is_protein'
    log_probs_esmc=esmc_lp,      # (L, 20) PLM_AA_ORDER
    log_probs_saprot=saprot_lp,  # (L, 20)
    out_dir=out,
    config=BiasedSampleConfig(
        fusion=FusionConfig(),
        ligand_mpnn=LigandMPNNConfig(temperature=0.1),
        n_samples=100,
        fix_active_site=True,
        fix_first_shell=False,
        chain="A",
    ),
)
```

Key behaviors:
- **Validates `(L, 20)` shapes** match the PositionTable's protein-residue count ([biased_mpnn.py:93](../src/protein_chisel/sampling/biased_mpnn.py#L93)).
- **Forces REMARK 666 catalytic residues into the fixed set** even if their class drifted ([biased_mpnn.py:122](../src/protein_chisel/sampling/biased_mpnn.py#L122)).
- **Annotates the output `CandidateSet`** with `plm_fusion_mean_abs_bias`, `n_fixed_positions`, `fusion_class_weights` for traceability.
- **Static bias**, no refresh. The "PLM bias goes stale as MPNN drifts the seed sequence" caveat is documented in [docs/architecture.md](architecture.md) under "Why a static PLM bias can drift". Refresh / reranker / allowed-set variants are TODO; see [docs/future_plans.md](future_plans.md).

### Container coordination

The orchestrator is **container-agnostic** — it expects pre-computed log-probs, so it can run wherever numpy + pandas + PyRosetta-PDB are available. The actual stages are:

| Stage | sif |
|---|---|
| ESM-C / SaProt logits | `esmc.sif` |
| `classify_positions` (for class labels) | `pyrosetta.sif` |
| Fusion + MPNN call | wherever (LigandMPNN sampling itself dispatches into `universal.sif` via `apptainer.universal_call`) |

---

## `tools/ligand_mpnn.py` — fused_mpnn runner wrapper

[src/protein_chisel/tools/ligand_mpnn.py](../src/protein_chisel/tools/ligand_mpnn.py)

The actual MPNN call. Wraps `/net/software/lab/fused_mpnn/seth_temp/run.py` (the modern lab build). Critical knobs in `LigandMPNNConfig` ([ligand_mpnn.py:56](../src/protein_chisel/tools/ligand_mpnn.py#L56)):

- `temperature=0.1` — softmax temperature.
- `repack_everything=0` — **DO NOT repack fixed residues**. Critical for theozyme protection. Without this the catalytic sidechains move.
- `ligand_mpnn_use_side_chain_context=1` — model sees catalytic sidechain rotamers (clash avoidance).
- `omit_AA="CX"` — don't sample C or X by default.
- `enhance` — optional `plddt_residpo_*` checkpoint for pLDDT-enhanced models.

### Bias format conversions

`bias_per_residue` arrives as a `(L, 20)` numpy array in `PLM_AA_ORDER`. `_build_bias_per_residue_multi` ([ligand_mpnn.py:99](../src/protein_chisel/tools/ligand_mpnn.py#L99)) converts that to fused_mpnn's expected JSON:

```json
{
  "/abs/path/to/design.pdb": {
    "A10": {"A": 1.5, "C": 0.0, "D": -0.2, "E": 0.3, ...},
    "A12": {"A": 0.0, "C": 0.0, ..., "K": -2.0, ...}
  }
}
```

Rules:
- Keyed by `<chain><resno>` (no separator). Helper: `_residue_label("A", 10) == "A10"`.
- Rows that are all zeros are dropped (keeps the JSON small).
- Bias values are in nats (log-space); 0 = no bias.
- The wrapper asserts `PLM_AA_ORDER == "ACDEFGHIKLMNPQRSTVWY"` matches what fused_mpnn expects on the AA dimension.

### Fixed residues format

`_build_fixed_residues_multi` ([ligand_mpnn.py:93](../src/protein_chisel/tools/ligand_mpnn.py#L93)) emits:

```json
{"/abs/path/to/design.pdb": ["A41", "A64", "A148", "A184", "A187", "A188"]}
```

Sorted, deduped, prefixed with chain.

### Output FASTA parsing

fused_mpnn writes `seqs/<pdb_stem>.fa<file_ending>` with one entry per design + an input header. `_parse_output_fasta` + `_parse_header` ([ligand_mpnn.py:138](../src/protein_chisel/tools/ligand_mpnn.py#L138)) extract `T=`, `seed=`, `seq_rec=`, `overall_confidence=`, etc. as `mpnn_*` columns on the `CandidateSet`.

### Tests

[tests/test_ligand_mpnn_unit.py](../tests/test_ligand_mpnn_unit.py) — host-only. Covers the bias-matrix conversion, fixed-residue JSON layout, header parsing, output-FASTA parsing, config hashing, and the critical default `repack_everything=0`. The actual fused_mpnn execution is **not** in CI.

---

## Iterative walks (in pipelines, not sampling)

The single-mutation walks (`constrained_local_search`, `mh`) live in [pipelines/iterative_optimize.py](../src/protein_chisel/pipelines/iterative_optimize.py) — see [docs/pipelines.md](pipelines.md). They consume a calibrated `(L, 20)` log-prob array (the PLM-fused marginals) but the orchestration is at the pipeline layer.
