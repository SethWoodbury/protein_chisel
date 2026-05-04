# Directional position classification — design plan

**Status:** REVISED after codex review (2026-05-04). Awaiting user sign-off before implementation.
**Author:** Claude (Opus 4.7) on behalf of Seth, 2026-05-04.
**Tag to revert:** `pre-plm-bump-2026-05-04` (or HEAD of `main` if before merge).

## Changelog after codex review

Codex flagged 5 hard blockers + 5 soft blockers; all addressed below.

- §4 decision tree: secondary tier now also fires on `d_sidechain_catshell ≤ 6` (catches H-bond networks like PTE Asp301↔His254 that don't directly contact ligand).
- §4: `d_ca_ligand` redefined as `min(CA → any ligand atom)` instead of CA→centroid (better for elongated substrates like paraoxon).
- §4: `secondary_sasa_max_fraction` raised 0.30 → 0.40 (avoids demoting true 2nd-shell residues on flexible loops).
- §4: oxyanion-hole detection narrowed to backbone-N donors only (was any backbone atom).
- §4 (post-user feedback): backbone polar contact now covers BOTH N (donor, e.g. oxyanion hole) AND O (acceptor, e.g. substrate-orienting backbone carbonyls). Codex narrowed it too far; donor-only would miss real acceptor-side H-bonds. CA and C still excluded (not polar atoms).
- §4 (post-user-feedback v2): added a directional GATE on backbone polar proximity to fix the helix-against-ligand back-side false positive. A backbone N or O within 4.5 Å of a ligand atom now only triggers primary_sphere if EITHER (a) the residue's CB vector points roughly toward the ligand (θ ≤ 110°), OR (b) explicit H-bond geometry (donor/acceptor angles via `tools/geometric_interactions.py` antecedent tables) confirms a real H-bond. Path (a) catches the user's helix scenario; path (b) is the oxyanion-hole escape hatch. Sidechain-reach paths (`d_sidechain_lig`, `d_sidechain_catshell`) do NOT need this gate -- reaching is already directional.
- §4: precondition added — assert `min(d_sidechain_lig over catalytic residues) < 5` before classifying; loud warning if not (catches poorly-placed ligand poses).
- §4: empty `catalytic_resnos` → loud warning + return all-distal classification (no silent demotion).
- §5 phantom Cβ: chirality marked as PROVISIONAL — implementation must compute both `cross(â, b̂)` and `cross(b̂, â)`, test against real Cβ on a non-Gly residue, pick the one with mean deviation < 0.15 Å. The 54.7356° tilt is canonical (`acos(1/√3)`). Codex flagged the cross-product order; the unit test is the source of truth.
- §3 θ definition: clarified — θ uses CA→Cβ (cheap, deterministic, rotamer-stable); the secondary-tier θ-gate has a known limitation that long-chi-rotated Arg/Lys can fail it. Documented in §9 cons.
- §8 PLM weights: `nearby_surface` lowered 0.40 → 0.30 (was a flat spot; codex right that geometric constraints should dominate spatially-close positions).
- §11: implementation now mandates a Catalytic Site Atlas / M-CSA validation step on 5 well-studied enzymes before production use.
- §11: explicit consumer audit of `class` is now step 1, not step 5.
- §3: `ClassifyConfig` is serialized as JSON metadata into the parquet (deterministic re-classification across users).
- §12: added "PTE-tuned vs generic" config preset note — defaults are PTE-shaped, not universal.

---

## 1. Why we're doing this

Today `classify_positions._classify_one` is purely radial:

```python
if resno in catalytic_resnos:        return "active_site"
if d_ligand <= 5.0:                  return "first_shell"   # heavy-atom min
if in_pocket:                        return "pocket"
if sasa < 20.0:                      return "buried"
return                               "surface"
```

Three known failure modes:

1. **Spherical false positives.** A residue with CA at 4.5 Å whose Cβ vector points *away* from the ligand still gets `first_shell`. Sidechain orientation is ignored.
2. **Spherical false negatives.** An Arg whose CA sits 8 Å away but reaches NH1/NH2 to within 3 Å of the ligand gets `surface` — yet it's mechanistically primary-shell. Long sidechains escape detection.
3. **No "supportive but not contacting" tier.** A residue 5–7 Å away pointing inward (preorganization role, second shell) and one pointing outward (just nearby surface real estate) get the same `buried`/`surface` label, even though their PLM bias and design treatment should differ.

Literature (Richter/Baker enzdes 2011, Tawfik HG3 series, Markin PafA 2023, Warshel preorganization, Chien & Huang EXIA 2012) treats the catalytic neighborhood as a 4-tier hierarchy with directionality. Our taxonomy should match.

## 2. Proposed taxonomy

Six classes, replacing the current five protein classes plus `ligand`:

| New class           | Meaning                                                                 | Replaces                |
| ------------------- | ----------------------------------------------------------------------- | ----------------------- |
| `primary_sphere`    | catalytic + any residue whose **sidechain heavy atom** ≤ 4.5 Å of ligand-or-catalytic-sidechain | `active_site` + tight `first_shell` |
| `secondary_sphere`  | sidechain ≤ 6.0 Å of ligand AND oriented toward (θ ≤ 70°), **interior** (low sidechain SASA) | inward-pointing `first_shell`/`buried` |
| `nearby_surface`    | CA ≤ 10 Å of ligand-centroid AND (oriented away θ ≥ 110° OR exposed)    | outward `first_shell`/`surface` near pocket |
| `distal_buried`     | CA > 10 Å AND sidechain SASA fraction < 0.20                            | `buried` |
| `distal_surface`    | CA > 10 Å AND sidechain SASA fraction ≥ 0.20                            | `surface` |
| `ligand`            | non-protein                                                             | unchanged |

Naming choices:
- We use `primary_sphere` / `secondary_sphere` (matches Tawfik/Markin/Warshel) instead of `first_shell` / `second_shell` (also valid but ambiguous in a Rosetta-shell context). Both terms appear in the literature.
- We drop the standalone `pocket` class. The fpocket flag becomes a separate boolean column (`in_pocket`) on the PositionTable; classification doesn't need it.
- Catalytic residues remain force-included in `primary_sphere` regardless of geometry (they ARE the catalytic residues by definition).

## 3. Continuous → discrete: hybrid storage

Keep computing the underlying continuous metrics and **store them as separate columns** on PositionTable, then derive the discrete `class` from a deterministic decision tree. This way:

- **Discrete labels** stay simple for filters, telemetry, UI: `df[df["class"] == "primary_sphere"]`.
- **Continuous metrics** are available for advanced uses (e.g. smooth PLM weight modulation, ranking, ML features).
- **Auditable**: every class assignment can be back-explained from the metrics.

New columns to add to PositionTable (alongside existing ones):

| Column                  | Type    | Definition                                                              |
| ----------------------- | ------- | ----------------------------------------------------------------------- |
| `d_sidechain_lig`       | float   | min(sidechain heavy atom → any ligand heavy atom or catalytic-metal or μ-OH bridge) |
| `d_sidechain_catshell`  | float   | min(sidechain heavy atom → any catalytic-residue sidechain heavy atom) |
| `d_ca_ligand`           | float   | CA → ligand-centroid distance (renamed from current `dist_ligand`)      |
| `theta_orient_deg`      | float   | angle between (CA → CB_or_phantom) and (CA → ligand-centroid), 0–180°   |
| `sasa_sc_fraction`      | float   | sidechain SASA / max sidechain SASA for this AA type (Tien 2013 max)   |
| `is_backbone_polar_contact` | bool | Backbone N (donor) OR O (acceptor) ≤ 4.5 Å of ligand. Captures oxyanion-hole NH AND substrate-orienting backbone C=O. CA + C excluded as non-polar. |
| `is_rim_ambiguous`      | bool    | rotamer-library scan finds rotamers in BOTH primary and secondary buckets (optional, see §10) |

The legacy `dist_ligand` and `dist_catalytic` are kept verbatim for backwards compat with downstream code that reads them.

## 4. Decision algorithm

Pseudocode (revised post-codex):

```python
def classify(residue, ligand_atoms, catalytic_resnos, cfg):
    # ----- Sanity gates (codex finding 4) -----
    if not catalytic_resnos:
        log.warning("classify_positions: no catalytic residues; "
                    "all residues will be distal_*")
    else:
        d_check = min(d_sidechain_lig[c] for c in catalytic_resnos)
        if d_check > 5.0:
            log.warning("classify_positions: catalytic residue %s has "
                        "d_sidechain_lig=%.1f Å (>5). Likely poorly-placed "
                        "ligand; classification may be garbage.",
                        worst_cat, d_check)

    # ----- Force-include catalytic -----
    if residue.resno in catalytic_resnos:
        return "primary_sphere"

    # ----- Tier 1 — primary contact -----
    # Backbone polar contact: N (H-bond donor, e.g. oxyanion hole) OR
    # O (H-bond acceptor, e.g. substrate-orienting backbone carbonyls).
    # Excludes CA (sp3, not polar) and C (sp2 carbonyl carbon -- the
    # polar atom is its O, already covered).
    is_backbone_polar_contact_proximity = (
        any(||r.atom('N') - lig|| <= 4.5 for lig in ligand_atoms) OR
        any(||r.atom('O') - lig|| <= 4.5 for lig in ligand_atoms)
    )
    # GUARD against helix-against-ligand back-side false positives:
    # backbone polar proximity alone is not enough -- a residue on the
    # back of a helix can have its backbone N/O at 4.4 Å of the ligand
    # purely because the helix runs parallel, with no real H-bond and
    # the sidechain pointing into solvent. Require either:
    #   (a) the residue's CB vector points roughly toward the ligand
    #       (θ ≤ 110°) -- signals the residue is on the "facing" side
    #       of secondary structure, OR
    #   (b) explicit H-bond geometry (donor/acceptor angles consistent
    #       with the antecedent chemistry tables in
    #       tools/geometric_interactions.py) -- this is the oxyanion-
    #       hole escape hatch where the backbone NH is in true H-bond
    #       distance + angle to a ligand acceptor regardless of where
    #       Cβ sits.
    is_backbone_polar_contact = (
        is_backbone_polar_contact_proximity
        AND (
            theta_orient_deg <= 110.0   # path (a): CB not strongly outward
            OR has_explicit_backbone_hbond_to_ligand(r, ligand_atoms)
                                          # path (b): real H-bond geometry
        )
    )
    if (d_sidechain_lig <= cfg.primary_distance
        OR d_sidechain_catshell <= cfg.primary_distance
        OR is_backbone_polar_contact):
        return "primary_sphere"

    # ----- Tier 2 — secondary sphere (codex: also catshell-only path) -----
    in_secondary_distance = (
        d_sidechain_lig <= cfg.secondary_distance
        OR d_sidechain_catshell <= cfg.secondary_distance   # NEW: catches Asp301↔His254
    )
    if (in_secondary_distance
        AND theta_orient_deg <= cfg.orient_inward_max_deg
        AND sasa_sc_fraction < cfg.secondary_sasa_max_fraction):
        return "secondary_sphere"

    # ----- Tier 3 — nearby surface -----
    # d_ca_ligand uses min(CA → any ligand atom), not centroid (codex)
    if d_ca_ligand <= cfg.nearby_ca_distance:
        return "nearby_surface"

    # ----- Tier 4 — distal -----
    if sasa_sc_fraction < cfg.distal_buried_sasa_max_fraction:
        return "distal_buried"
    return "distal_surface"
```

`ClassifyConfig` (post-codex defaults):

```python
@dataclass
class ClassifyConfig:
    primary_distance: float = 4.5         # Å, sidechain → ligand/catshell
    secondary_distance: float = 6.0       # Å, sidechain → ligand OR catshell
    nearby_ca_distance: float = 10.0      # Å, min CA → any ligand atom (NOT centroid)
    orient_inward_max_deg: float = 70.0   # θ ≤ this = inward (Chien & Huang 2012)
    orient_outward_min_deg: float = 110.0 # θ ≥ this = outward (diagnostic only)
    # 0.40 (was 0.30 in v0 plan): codex flagged that 0.30 demotes true
    # 2nd-shell residues on flexible loops (e.g. PTE loops 7/8 around Y309).
    secondary_sasa_max_fraction: float = 0.40
    distal_buried_sasa_max_fraction: float = 0.20
    sasa_probe: float = 1.4
    backbone_contact_distance: float = 4.5
    poor_ligand_pose_warning_threshold: float = 5.0  # NEW
    altloc_policy: str = "min"            # "min" | "first"
    # Preset name persisted into parquet metadata for deterministic
    # re-classification across users (codex finding 3).
    preset_name: str = "pte_v1"
```

Cutoff justifications:
- 4.5 Å (Rosetta `enzdes` match-cstfile defaults, Richter 2011 doi:10.1371/journal.pone.0019230)
- 6.0 Å (Khersonsky/Tawfik HG3 series)
- 70°/110° (Chien & Huang EXIA 2012 doi:10.1371/journal.pone.0047951)
- 0.40 sidechain-SASA fraction (Tien 2013 max-SASA values, doi:10.1371/journal.pone.0080635, raised from 0.30 per codex)
- 10.0 Å CA → min ligand atom — Khersonsky 2012 design-shell convention

## 5. Glycine: phantom Cβ construction

Standard tetrahedral construction from N, Cα, C atoms (matches Engh & Huber 1991 ideal geometry, Rosetta internal):

```python
def phantom_cb(N, CA, C):
    a = (N - CA) / norm(N - CA)
    b = (C - CA) / norm(C - CA)
    bisector = -(a + b) / norm(a + b)        # opposite the N-CA-C bisector
    normal   = cross(a, b) / norm(cross(a, b))  # right-hand: L-aa convention
    direction = cos(54.75°) * bisector + sin(54.75°) * normal
    return CA + 1.522 * direction            # CA-CB bond length 1.522 Å
```

Sign convention: `cross(a, b)` with order (N, C) gives the L-amino-acid handedness. We will validate by computing phantom Cβ on a non-Gly residue and asserting mean deviation from real Cβ < 0.15 Å in a unit test.

## 6. Sidechain-tip distance (long-reaching residues)

For each non-Gly protein residue, "sidechain heavy atom" = every heavy atom NOT in `{N, CA, C, O, OXT, H, H2, H3, HA}`. This is a fixed set per residue type and matches the existing `SIDECHAIN_ATOM_NAMES` table in `clash_check.py` (we'll reuse it). For Gly: only the phantom Cβ is in the set.

Then `d_sidechain_lig = min(||sc_atom - lig_atom||)` over all (sidechain atom, ligand atom) pairs, where `lig_atom` ranges over:

- ligand HETATM heavy atoms
- catalytic metals (Zn, Mn, Fe, Mg, Ca, etc. — detected by element ∈ standard metal set)
- bridging hydroxide / water of catalytic relevance (for PTE: μ-OH between Zn and Mn, but in practice we just include any HETATM oxygen whose record ID matches a known cofactor)

`d_sidechain_catshell` is the same but vs catalytic-residue sidechain atoms instead.

This naturally handles the Arg-reaches-into-pocket case: if NH1 is at 3 Å from the ligand, `d_sidechain_lig = 3.0` regardless of where the CA sits.

## 7. PTE-specific fixes

The research agent flagged these. Most apply broadly but matter for PTE_i1:

1. **KCX (carbamylated Lys157)** — its functional atoms include OQ1/OQ2 (the carbamate). The existing `SIDECHAIN_ATOM_NAMES["KCX"]` table already covers this; reuse it. Confirm KCX rows survive in PositionTable.
2. **Binuclear Zn/Mn** — include both metal atoms in the `lig_atom` set so His60/64/128/132 register as primary_sphere via metal coordination, not just substrate proximity.
3. **μ-OH bridge** — any cofactor oxygen between the metals counts as a ligand atom.
4. **Altloc handling** — current `_read_atoms` keeps "first altloc only". For classification we should take **min over altlocs** (more inclusive; safer for design). Add a `--altloc-policy` option (default `min`), but keep `first` available for back-compat.
5. **Multi-pose ligands** — out of scope for this iteration. Single bound pose only. Add a TODO to revisit.

## 8. PLM weight remapping

Update `FusionConfig.class_weights` to the new keys. Proposed values (continuous gradient, fixed flat-spot codex flagged):

| Class             | Weight | Note |
| ----------------- | ------ | ---- |
| `primary_sphere`  | 0.05   | catalytic + direct contact (mostly fixed) |
| `secondary_sphere`| 0.20   | inward-pointing preorganization |
| `nearby_surface`  | **0.30** | spatially close: geometric constraints dominate (lowered from 0.40 per codex) |
| `distal_buried`   | 0.40   | folding/stability — PLM most informative |
| `distal_surface`  | 0.55   | solubility/expression — PLM most informative |
| `ligand`          | 0.0    | non-protein |

Rationale:
- `primary_sphere` = catalytic + direct contact: very low PLM (these positions are mostly fixed; for non-fixed contact residues the geometric constraint dominates).
- `secondary_sphere` = preorganization residues: meaningful PLM input, since these encode the natural sequence patterns that stabilize the active-site array (Tawfik HG3 series).
- `nearby_surface` = mid-range PLM. These don't see the substrate but they're spatially close so PLM evolutionary signal is informative.
- `distal_buried` = same as `nearby_surface`. Buried positions far from the active site benefit from PLM (folding/stability), but no more than nearby surface.
- `distal_surface` = highest. PLM signal is most useful here (solubility, expression, evolutionary accessible mutations).

The existing `--plm_strength` CLI knob still scales all six uniformly.

For backwards compatibility, the legacy keys (`active_site`, `first_shell`, `pocket`, `buried`, `surface`) map to the closest new class with a deprecation warning so existing configs don't crash:

```python
LEGACY_CLASS_REMAP = {
    "active_site": "primary_sphere",
    "first_shell": "primary_sphere",  # tight first_shell ≈ primary
    "pocket":      "secondary_sphere",
    "buried":      "distal_buried",
    "surface":     "distal_surface",
}
```

(The remap is for *PLM-config keys*. For PositionTable `class` values the new module always emits new names. Old `.parquet` files would need a migration helper — see §11.)

## 9. Pros & cons

### Pros of doing this rebuild

1. **Captures real biology.** Long-reach Arg in primary, inward-pointing buried in secondary, outward-pointing near-pocket in nearby_surface — all match how enzymologists actually reason.
2. **Better PLM gradient.** Smooth ramp from 0.05 → 0.55 across five classes instead of a step from 0.05 → 0.50, less binary.
3. **Diagnostic.** Continuous columns (`theta_orient_deg`, `d_sidechain_lig`, `sasa_sc_fraction`) let us audit any classification and write per-position diagnostics.
4. **Fixes known bugs.** No more "Y@35 looks like first-shell because CA is close, but actually points away into solvent" misclassifications.
5. **Future-proofing.** Adding directional classification once means downstream tools (preorganization, clash, expression rules) can leverage the same metrics for free.

### Cons / risks

1. **PositionTable schema changes.** New columns. Legacy `.parquet` files become non-conformant. We'll need a migration helper or version field. *Mitigation:* add a `schema_version` field, write a one-line `migrate_v1_to_v2(df)` helper.
2. **Cutoffs are research-defaults.** 4.5/6.0/10.0 Å + 70°/110° are defensible (Richter 2011, Chien 2012) but PTE-specific tuning may need different values once we see how it splits residues. *Mitigation:* expose every cutoff in `ClassifyConfig`. First production run uses defaults; iterate if class counts look weird.
3. **More compute per pose.** All-pairs sidechain × ligand distance + phantom-Cβ + per-AA-type SASA fraction. Estimated ~50 ms per pose (was ~10 ms). Negligible at the per-cycle level (we classify once per cycle, not per design).
4. **Risk of silent regressions.** Many places in code read `class` and assume the old vocabulary. *Mitigation:* grep for the five legacy class strings and either update or alias them. List of files touched: `sampling/plm_fusion.py`, `expression/builtin_rules.py` (kr_neighbor_dibasic uses `first_shell` to gate severity), `tools/ligand_mpnn.py` (no — uses positions, not classes), `pipelines/*` (need to grep). I'll enumerate in the audit step.
5. **The orientation test fails for Pro.** Pro's Cβ is constrained by the ring; the CB vector is less free. *Mitigation:* document that for Pro we use the standard CB (it exists), and the metric is just less informative. Don't special-case unless empirical results force it.

### Soft alternative (rejected)

The "soft" approach I floated earlier (multiply existing class weight by an orientation factor without re-classifying) avoids schema changes but:
- Doesn't fix long-reach Arg (since Arg's CA is still at 8 Å, it still gets `surface`).
- Doesn't give a `secondary_sphere` distinction for telemetry/expression rules.
- Mixes orientation into the PLM weight only, not into class. Other tools (preorganization, expression rules) don't see the directional info.

The hard rebuild is the right call given how much downstream code wants directional info.

## 10. Optional: rotamer-library scan for `is_rim_ambiguous`

Some rim residues have rotamer states that flip between primary and secondary depending on chi1/chi2. To flag these:

For each non-fixed residue at the boundary (e.g. `d_sidechain_lig` between 4.0 and 5.5 Å), scan a 9-rotamer chi1×chi2 stub grid (we already have this infrastructure for clash bias) and check whether ≥1 rotamer would push the residue into a different class.

Set `is_rim_ambiguous = True` if so. Use it for diagnostics; don't gate classification on it (otherwise we get unstable labels). This is OPTIONAL for v1 — tag it with a `--enable_rim_ambiguous_scan` flag, default off.

## 11. Implementation sequence (revised, post-codex)

**STEP 0 — consumer audit (NEW per codex):** before touching anything, grep every consumer of the legacy class names (`active_site`, `first_shell`, `pocket`, `buried`, `surface`) and produce a list. Decide per-consumer: update to new names, or use the legacy remap. Files to audit at minimum: `sampling/plm_fusion.py`, `expression/builtin_rules.py`, `pipelines/*`, `scripts/iterative_design_v2.py`, all tests. **No edits until the audit list is in the plan.**

1. **`utils/geometry.py` additions** — phantom_cb() with BOTH chirality candidates + auto-validation against real Cβ on a non-Gly residue (mean deviation < 0.15 Å). sidechain_atoms(), residue_max_sasa(Tien 2013 table). ~80 LOC + tests.
2. **`tools/classify_positions.py` rewrite** — new `_classify_one`, new `ClassifyConfig` fields, new metric columns, sanity-gate warnings. Keep current entrypoint signature stable. `ClassifyConfig` is serialized as JSON metadata into the parquet (codex finding 3).
3. **`io/schemas.py`** — add new optional columns to `PositionTable` schema. Bump `PositionTable.schema_version` to 2. Backfill defaults (NaN/False) on legacy load. Migration helper requires raw PDB + the original `ClassifyConfig` (or refuses to migrate).
4. **`sampling/plm_fusion.py`** — new `class_weights` defaults (post-codex, with smooth gradient); legacy class-name remap with `DeprecationWarning`. Keep `global_strength` as is.
5. **`expression/builtin_rules.py`** — audit + update kr_neighbor_dibasic etc. to either use new names or accept both via remap.
6. **Driver** — re-classify in v2 driver if the loaded PositionTable is schema_version < 2 AND the seed PDB is available; otherwise hard-fail with instructions.
7. **Validation against benchmark (NEW per codex)** — pick 5 well-studied enzymes from M-CSA / Catalytic Site Atlas (e.g. trypsin 1AKS, chymotrypsin 4CHA, dihydrofolate reductase 1RX2, kemp eliminase HG3.17, PTE 1HZY), manually annotate primary/secondary shells from the literature, run the new classifier, report agreement. Gate: must achieve ≥80% agreement on primary_sphere and ≥70% on secondary_sphere before production use.
8. **Tests** — phantom CB on a non-Gly residue (mean deviation < 0.15 Å), classification on a synthetic PTE pose with known categories, cutoff-boundary residues, KCX, altlocs, long-reach Arg, oxyanion-hole backbone-N-only contact, empty-catalytic warning, poorly-placed-ligand warning, schema migration with config-pinning.
9. **Smoke run** — re-classify the WT PTE_i1 seed and inspect the class distribution. Sanity-check that catalytic residues are all primary_sphere, that "obvious" first-shell residues (Y@35, F@135, M@145) classify correctly given orientation, that loop-7/8 residues land in `secondary_sphere` not `nearby_surface`, etc. **Measure actual classification time per pose** (codex flagged 50 ms claim is unverified).
10. **Production run** — `--plm_strength 1.0` with the new defaults; compare fitness / charge / SAP / preorganization / pocket-druggability to the pre-rebuild baseline. Easy to revert via `git reset --hard pre-plm-bump-2026-05-04`.

**Pre-prod freeze:** for one cycle after merge, run BOTH the v1 classifier (cached) and v2 classifier (new) and dual-write both `class_v1` and `class_v2` columns. Compare distributions on real production runs. After 1–2 cycles of confidence, drop `class_v1`.

Estimated effort: ~700 LOC source + ~500 LOC tests + ~200 LOC for benchmark validation harness, ~4 days of careful work with codex review at each step.

## 12. Open questions for the user

1. Are the post-codex cutoffs (4.5/6.0/10.0 Å + 70°/110° + 0.40 sidechain-SASA) acceptable as defaults? Or do you want to dial primary tighter (4.0 Å)?
2. Do you want `is_rim_ambiguous` rotamer scan in v1 (slow but informative) or deferred to v2?
3. Should the PLM `class_weights` for `secondary_sphere` be 0.20 (conservative) or 0.25 (slightly more aggressive)?
4. Migration: pure CLI command, runtime re-classify, or hard-fail if schema is old?
5. The PTE-shaped defaults — do you want a separate `generic` preset (looser cutoffs, no metal force-include) so this can also be applied to other scaffolds without re-tuning?
6. The benchmark validation step is a real gate (≥80% agreement on primary). Are you OK with deferring production until we have that, or should we ship without it and validate retroactively on PTE only?

---

## Sources

- Richter, Leaver-Fay, Khare, Bjelic, Baker. "De Novo Enzyme Design Using Rosetta3," *PLoS ONE* 2011, doi:10.1371/journal.pone.0019230.
- Chien & Huang. EXIA: A Web Server for Predicting Catalytic Residues from Sidechain Positioning, *PLoS ONE* 2012, doi:10.1371/journal.pone.0047951.
- Khersonsky, Tawfik et al. "Bridging the gaps in design methodologies …," *PNAS* 2012, doi:10.1073/pnas.1121063109.
- Blomberg, Tawfik, Hilvert et al. HG3.17, *Nature* 2013, doi:10.1038/nature12623.
- Markin, Fordyce et al. PafA second-shell residues, *Science* 2021/2023, PMID 37172218.
- Warshel preorganization, *Chem. Rev.* 2006, 106, 3210; Fried & Boxer *Annu. Rev. Biochem.* 2017.
- Tien et al. Maximum Allowed Solvent Accessibilities, *PLoS ONE* 2013, doi:10.1371/journal.pone.0080635.
- Engh & Huber, *Acta Cryst.* 1991, A47, 392 (idealized backbone geometry).
- Wankowicz et al. altloc/ligand binding, *eLife* 2022, doi:10.7554/eLife.74114.
- Bigley & Raushel, PTE structure/mechanism review, PMC6622166.
