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
                    # Calibrate the bias as log-odds (subtract AA bg),
                    # entropy-match models, shrink at disagreeing positions.
                    # See architecture.md "Calibration of fused logits".

- mpnn_with_refresh # alternative to biased_mpnn: refresh PLM bias each
                    # round on the median-naturalness top samples, since
                    # a static seed-derived bias goes stale as MPNN
                    # drifts the sequence away from the seed.

- plm_reranker      # alternative to biased sampling entirely: let MPNN
                    # sample freely, then rerank candidates by ESM-C +
                    # SaProt naturalness. Cleanest when bias calibration
                    # is uncertain.

- plm_allowed_set   # alternative: at each position, restrict MPNN's
                    # allowed AAs to top-k under PLM marginals. Strong
                    # restriction — use only at non-active, non-pocket
                    # positions when speed matters.
- iterative_walk    # Two modes (see architecture.md):
                    # 1) constrained_local_search (default): PLM-only
                    #    marginals propose, hard filters accept. Cheap
                    #    but cannot find compensatory multi-site moves.
                    # 2) mh: real Metropolis-Hastings with proposal-ratio
                    #    correction and a scalar target energy
                    #    (E = -log P_combined). Supports temperature
                    #    schedules, block moves, and parallel tempering
                    #    for compensatory mutations.
- temperature       # categorical sampling helpers (temperature, top-k,
                    # nucleus, position-class–dependent τ).
"""
