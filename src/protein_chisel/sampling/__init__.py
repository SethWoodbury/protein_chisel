"""Sequence sampling logic.

Important asymmetry between samplers (see docs/architecture.md):
LigandMPNN logits are autoregressive — `p(aa_i | structure, decoded_so_far)`.
ESM-C / SaProt logits are masked-LM marginals — `p(aa_i | seq_minus_i)`.
Don't naively product-of-experts across all three; instead, fuse the PLM
marginals into a per-position bias and feed it to LigandMPNN's native
sampler via --bias_AA_per_residue.

Planned:
- plm_fusion        # ESM-C + SaProt masked-LM marginals fused as
                    #   log p_plm = β·log p_ESMC + γ·log p_SaProt.
                    # These are peer distributions and fuse cleanly.
- biased_mpnn       # call LigandMPNN with --bias_AA_per_residue derived
                    # from plm_fusion. MPNN remains the primary sampler;
                    # PLMs are only a bias.
- iterative_walk    # Gibbs / Metropolis-Hastings single-mutation walk
                    # using PLM-only marginals (ESM-C + SaProt fusion).
                    # Cheap filters (regex/ProtParam/repack-Δ) act as the
                    # acceptance criterion. Used by the iterative_optimize
                    # pipeline.
- temperature       # categorical sampling helpers (temperature, top-k,
                    # nucleus, position-class–dependent τ).
"""
