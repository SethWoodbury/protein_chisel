"""Sequence sampling logic.

Planned:
- logit_fusion      # product-of-experts over per-position log-probs
                    # log p_combined = α·log p_LMPNN + β·log p_ESM-C
                    #                + γ·log p_SaProt   (per position class)
- biased_mpnn       # call LigandMPNN with --bias_AA_per_residue derived
                    # from fused PLM logits
- temperature       # categorical sampling helpers
"""
