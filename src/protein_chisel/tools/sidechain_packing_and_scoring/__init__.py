"""Side-chain packing and rotamer-quality scoring.

This subpackage groups the family of tools that touch protein side chains:
either rebuilding them (packers) or evaluating their quality (scorers).
Most pipelines will only invoke one or two of these per run, but having
them all under one namespace makes head-to-head comparison clean.

Layers:
- rotamer_score   : Rosetta fa_dun (Shapovalov-Dunbrack 2011 BBDEP)
- rotalyze_score  : MolProbity rotalyze (Top8000 KDE) -- complementary to fa_dun
- pippack_score   : PIPPack neural plausibility (chi-bin distributions)
- faspr_pack      : FASPR fast classical CPU packer (Huang 2020)
- attnpacker_pack : AttnPacker SE(3) transformer (McPartlon 2023)
- flowpacker_pack : FlowPacker torsional flow matching (Lee 2025)
- opus_rota5_pack : OPUS-Rota5 RotaFormer (Xu 2024)

Most modules are optional and load lazily; e.g. importing this package does
not require PIPPack/AttnPacker to be installed.
"""
