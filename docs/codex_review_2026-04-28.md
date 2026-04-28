# Codex review — 2026-04-28

Verbatim output from running OpenAI Codex CLI (`codex exec`, model gpt-5.4, reasoning effort xhigh) against the docs/ and src/protein_chisel/ manifests at commit `07d44a8`. Prompt at `/tmp/codex_review_prompt.txt` (also reproduced below).

The response is reproduced unedited; the architectural changes triggered by it are in commits after `07d44a8`.

---

## Prompt

> You are reviewing the architecture and plans of a protein-design codebase that I'm building. Read README.md and the two markdown files under docs/ (architecture.md, dependencies.md). Skim the planned-tool manifests in src/protein_chisel/{tools,filters,scoring,sampling,pipelines,io}/__init__.py.
>
> Then give me a focused critique covering:
>
> 1. Is the layered tools/filters/scoring/sampling/pipelines architecture sensible for de novo enzyme sequence optimization with cheap filters before AF3? Where will it bend or break as the codebase grows?
> 2. The logit-fusion approach: I fuse ESM-C + SaProt masked-LM marginals into a per-position bias and feed it to LigandMPNN's native sampler via --bias_AA_per_residue. I'm explicitly NOT putting LigandMPNN's autoregressive logits into the same product-of-experts as the PLM marginals because they live in different conditioning regimes. Correct distinction? Better ways to combine an autoregressive structure-conditioned sampler with masked-LM PLMs? Anything missing about temperature/calibration of fused logits?
> 3. The iterative_optimize pipeline: Gibbs / Metropolis-Hastings single-mutation walk using PLM-only marginals (ESM-C + SaProt fused), with cheap filters (regex / ProtParam / quick PyRosetta repack-Δ) as MH acceptance. Does this make sense theoretically and in practice? When does it pathologically fail? Sensible mixing time? How would you tell convergence?
> 4. The "no AF in inner loop" philosophy and the cheap-filter cascade — fixed-backbone PyRosetta sidechain repacking + Rosetta ligand-binding ddG (apo vs holo) + fpocket geometry + SAP/charge/protease filters. Is the assumption valid that fixed-backbone repack-and-score is informative enough to discriminate good from bad sequences without folding? When does that assumption break?
> 5. Pareto ranking with sequence-identity diversity caps for final selection (~200 → 50-100). Strong or weak choice? Better alternatives at this size?
> 6. Multi-PDB inputs (single PDB OR multiple PDBs as input — designed model + AF3 conformers, or family of designs from one backbone). I'm planning a `pose_set` abstraction with metadata (sequence_id, fold_source, conformer_index). Aggregated vs per-conformer scoring. Critique?
> 7. Biophysical metrics I'm missing? Already planned: SAP, net charge w/o HIS, hbonds w/ energies, π-π, π-cation, salt bridges, fa_elec, ligand interface ddG, pocket geometry (fpocket), shape complementarity, holes, secondary structure, metal3d, electric field (APBS, late-stage), ProtParam (pI, instability, GRAVY, MW), Rg / Rg-normalized, asphericity, globularity, chainbreak, rCA_nonadj, loop_at_motif, sidechain bondlen deviation, cart_bonded + fa_dun on catalytic residues, ligand SASA. What ELSE is genuinely informative for de novo enzyme assessment that I'm missing? Flexibility predictors? Preorganization metrics? Active-site quality measures?
> 8. Synthetic-MSA approach: pool of N=1000 sampled sequences for per-position frequency profiles and pairwise mutual information, but explicitly NOT treating them as a natural MSA for DCA/EVcouplings. Is this caveat right? Ways to get genuine evolutionary signal even for de novo proteins where natural homologs don't exist?
> 9. Potholes / failure modes / things I'm probably overconfident about?
>
> Be terse and specific. Don't repeat my plan back to me — only flag things you'd change, contradict, or add. If something is fine as planned, you can say "fine, no comment."

## Response

1. Sensible overall. The main breakage point is not the layer count, it's missing artifact contracts. Add a small typed schema layer for `position_table`, `candidate_set`, `pose_set`, `metric_table`, plus per-stage provenance hashes; otherwise restartable file handoffs will turn into stale-TSV glue. Also, your boundaries already leak: `filters/sap_score` is structural, and `rosetta_metrics_xml` is not really a primitive. Expect "filter vs score" to collapse into the same metric with different policies.

2. Correct distinction. Do not PoE LigandMPNN autoregressive logits with MLM marginals as if they share a conditioning regime. The weak spot is the "compute PLM marginals once on the seed sequence" choice: that bias goes stale as MPNN drifts away. Keep it weak, refresh on top sampled sequences, or use PLMs for reranking / allowed-set truncation instead of a fixed additive bias. Also calibrate as log-odds, not raw log-prob sums: subtract AA background, temperature-match model entropies, and shrink low-confidence or disagreeing positions.

3. As written, that is not MH unless you include the proposal-ratio term and a real target energy. "Accept iff filters pass" is constrained local search. Practically, it can work for polishing a small mutable set. It fails on compensatory multi-site moves, charge-pair swaps, backbone-coupled core changes, and any case where a good solution requires traversing a bad single-mutant intermediate. Use occasional block moves or parallel tempering. Convergence is not "no accepts in N sweeps"; use multiple chains, objective plateaus, acceptance-rate collapse, autocorrelation, and stability of the top sequence clusters.

4. Fine as a veto layer, weak as a ranker among near-ties. Fixed-backbone repack is informative for clashes, gross desolvation, pocket destruction, and obvious ligand incompatibility. It is not reliably informative for foldability, loop gating, water-mediated catalysis, protonation-sensitive networks, or induced-fit pockets. Treat it as coarse feasibility screening, not as evidence that the design "works."

5. Weak if you throw many metrics into Pareto. With enough correlated objectives, almost everything becomes non-dominated. Stronger version: hard constraints first, then 3-5 real objectives, then crowding/epsilon-dominance, then diversity selection. Sequence diversity should be computed on mutable or pocket positions, not full-length identity, or you will mostly buy surface noise.

6. `pose_set` is the right abstraction. Internally, normalize everything to pose sets of size 1; don't maintain two codepaths. Aggregation must be metric-specific: worst-case or quantile for failure metrics, mean/std for descriptive robustness, explicit paired deltas for apo/holo. Do not average designed models and AF3 conformers as exchangeable samples; keep `source_model` and compare agreement across sources separately.

7. Missing metrics worth adding before more global shape descriptors:
- Buried unsatisfied polar atoms / buried unsat H-bonds.
- Internal packing quality / voids (`packstat`, buried cavity volume).
- Theozyme satisfaction metrics: catalytic distance/angle/dihedral deviations, motif RMSD, attack geometry.
- Active-site preorganization/flexibility: variance of catalytic geometry across restrained repack/min or backrub ensembles.
- Catalytic pKa / protonation-network estimates.
- A solubility predictor orthogonal to SAP.

8. The caveat is right. A synthetic pool gives sampler-induced co-variation, not evolutionary signal. If you want genuine evolutionary information, the only real route is remote homolog / structural-analog harvesting: Foldseek/PDB/AFDB analogs, motif-centered structural alignments, or superfamily-local residue statistics. Anything generated from your own models is still synthetic, even if useful.

9. Biggest overconfidence risks:
- Static PLM bias will over-regularize toward natural priors and erase intentionally weird catalytic environments.
- Rosetta ligand ddG / fpocket will look more quantitative than they are.
- Hard early filters will kill compensatory solutions unless you preserve an exploration tail.
- Protonation, catalytic waters, metal state, and alternate ligand poses are under-modeled.
- File-existence restarts without hashing params/model versions will eventually give you silent reuse bugs.
- Without a retrospective calibration set of known good/bad or decoy designs, your thresholds will drift into folklore.
