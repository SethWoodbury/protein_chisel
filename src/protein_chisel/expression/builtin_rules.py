"""Builtin expression-risk rules.

Each rule is a concrete subclass of ``Rule`` and is registered into
the global ``REGISTRY`` at import time. The user can override severities
or disable rules via the profile.

This file is intentionally long but linear — one rule per checklist
item, with regex patterns and predicates explicit. Add new rules below
and they're automatically picked up by ``ExpressionRuleEngine``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from protein_chisel.expression.rules import REGISTRY, Rule, StructureContext
from protein_chisel.expression.severity import RuleHit, Severity


if TYPE_CHECKING:
    from protein_chisel.expression.profiles import ExpressionProfile


HYDROPHOBIC_AAS = set("AVLIMFWC")
BASIC_AAS = set("KR")
ACIDIC_AAS = set("DE")


# ----------------------------------------------------------------------
# C-terminal degron rules
# ----------------------------------------------------------------------


class SsrATagCTermRule(Rule):
    """ClpXP/SspB degradation tag patterns at the C-terminus.

    Native ssrA = AANDENYALAA. We also flag close terminal patterns
    like ...YALAA / ...ALAA / ...LAA / ...AA when preceded by a
    flexible region (loop or low-complexity GS run).
    """
    name = "ssra_tag_cterm"
    default_severity = Severity.HARD_FILTER

    _exact_ssra = re.compile(r"AANDENYALAA$")
    _ssra_like = re.compile(r"(YALAA|ALAA|LAA|AA)$")

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        sev = self.resolved_severity(profile)
        seq = ctx.sequence
        L = ctx.L
        hits: list[RuleHit] = []

        m = self._exact_ssra.search(seq)
        if m:
            hits.append(RuleHit(
                rule_name=self.name, severity=sev,
                start=m.start(), end=m.end(), matched=m.group(0),
                reason="exact ssrA tag (AANDENYALAA) at C-terminus",
            ))
            return hits

        m = self._ssra_like.search(seq)
        if m:
            # Only flag if preceded by 'flexible' context. Heuristic: last
            # ~5 residues before the match are mostly G/S/A/T (low-complexity),
            # OR ss says loop. If we have no SS, default-flag.
            tail_start = max(0, m.start() - 5)
            upstream = seq[tail_start:m.start()]
            low_comp = sum(1 for c in upstream if c in "GSAT") / max(len(upstream), 1)
            in_loop = (
                ctx.ss_reduced is not None
                and "L" in ctx.ss_reduced[max(0, m.start() - 5): m.start()]
            )
            if low_comp >= 0.6 or in_loop or ctx.ss_reduced is None:
                hits.append(RuleHit(
                    rule_name=self.name, severity=sev,
                    start=m.start(), end=m.end(), matched=m.group(0),
                    reason=f"ssrA-like C-terminal motif {m.group(0)!r} after "
                           f"flexible context (low-complexity={low_comp:.2f})",
                ))
        return hits


class HydrophobicCTailRule(Rule):
    """Tsp/Prc tail-specific protease prefers nonpolar C-termini.

    Flag if the last 3-5 residues are >=70% in {A V L I F M W}.
    Designed body's C-terminal — the user's tag overlay
    (e.g. GSAWSHPQFEK) is applied AFTER the designed body, so this rule
    runs against the body's tail, not the construct tail.
    """
    name = "hydrophobic_c_tail"
    default_severity = Severity.HARD_FILTER

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        sev = self.resolved_severity(profile)
        seq = ctx.sequence
        L = ctx.L
        if L < 5:
            return []
        for window in (5, 4, 3):
            tail = seq[L - window:]
            n_hydro = sum(1 for c in tail if c in HYDROPHOBIC_AAS)
            if n_hydro / window >= 0.70:
                return [RuleHit(
                    rule_name=self.name, severity=sev,
                    start=L - window, end=L, matched=tail,
                    reason=f"hydrophobic C-tail {tail!r} ({n_hydro}/{window} hydrophobic)",
                )]
        return []


# ----------------------------------------------------------------------
# N-terminal rules
# ----------------------------------------------------------------------


class NEndRuleDestabilizingRule(Rule):
    """N-end rule: bulky hydrophobic AA exposed at the mature N-terminus.

    After MetAP cleaves the leading Met (when position 2 is small),
    the new N-terminus is a degron if it's L / F / Y / W (ClpS recognition).
    R / K is also part of the bacterial N-end-rule code but milder; soft-bias.
    """
    name = "n_end_rule_destabilizing"
    default_severity = Severity.HARD_FILTER

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        sev = self.resolved_severity(profile)
        if ctx.n_terminal_pair is None:
            return []
        nt1, nt2 = ctx.n_terminal_pair
        if nt1 in {"L", "F", "Y", "W"}:
            return [RuleHit(
                rule_name=self.name, severity=sev,
                start=0, end=1, matched=nt1,
                reason=f"mature N-terminal residue {nt1!r} is a strong N-degron "
                       f"(ClpS substrate)",
            )]
        if nt1 in {"R", "K"}:
            return [RuleHit(
                rule_name=self.name, severity=sev.demote(),
                start=0, end=1, matched=nt1,
                reason=f"mature N-terminal residue {nt1!r} participates in the "
                       f"bacterial N-end rule (milder than L/F/Y/W)",
            )]
        return []


class MetAPWarningRule(Rule):
    """Tell the user MetAP will cleave the leading M.

    Not a risk by itself; informational. Used to populate
    ``n_terminal_pair`` in the engine.
    """
    name = "metap_warning"
    default_severity = Severity.WARN_ONLY

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        seq = ctx.sequence
        if (profile.metap_cleaves_n_terminal_M and len(seq) >= 2
                and seq[0] == "M" and seq[1] in "ACGPSTV"):
            return [RuleHit(
                rule_name=self.name, severity=self.resolved_severity(profile),
                start=0, end=1, matched="M",
                reason=f"leading M will be cleaved by MetAP (next residue "
                       f"{seq[1]!r} is small); mature N-term = {seq[1:3]!r}",
            )]
        return []


class SignalPeptideNTermRule(Rule):
    """Sec-pathway signal peptide pattern at the N-terminus.

    Heuristic SignalP-lite: first 30 residues have:
      - >=2 basic AAs (n-region)
      - a hydrophobic stretch >=8 (h-region)
      - small residue at -3 / -1 of cleavage (AXA-like)
    """
    name = "signal_peptide_n_term"
    default_severity = Severity.HARD_FILTER

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        if profile.compartment != "cytosolic":
            return []   # signal peptide expected in periplasmic/secreted
        seq = ctx.sequence
        nterm = seq[:30] if len(seq) >= 30 else seq
        if len(nterm) < 15:
            return []
        n_basic = sum(1 for c in nterm[:5] if c in BASIC_AAS)

        # Find longest run of hydrophobic
        max_hydro = 0; cur = 0
        for c in nterm:
            if c in HYDROPHOBIC_AAS:
                cur += 1
                max_hydro = max(max_hydro, cur)
            else:
                cur = 0

        # AXA-like cleavage motif (small at -3 and -1)
        small = set("AGSTV")
        cleavage_ok = False
        for i in range(15, min(len(nterm) - 1, 30)):
            if nterm[i] in small and nterm[i - 2] in small:
                cleavage_ok = True; break

        if n_basic >= 2 and max_hydro >= 8 and cleavage_ok:
            return [RuleHit(
                rule_name=self.name,
                severity=self.resolved_severity(profile),
                start=0, end=min(30, len(seq)), matched=nterm,
                reason=f"N-term resembles Sec signal peptide "
                       f"(basic_n={n_basic}, hydro_run={max_hydro}, AXA={cleavage_ok})",
            )]
        return []


class TatSignalMotifRule(Rule):
    """Twin-arginine translocation signal."""
    name = "tat_signal_motif"
    default_severity = Severity.HARD_FILTER
    _re = re.compile(r"[ST]RR.FLK")

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        if profile.compartment != "cytosolic":
            return []
        seq = ctx.sequence[:40]
        m = self._re.search(seq)
        if m:
            return [RuleHit(
                rule_name=self.name, severity=self.resolved_severity(profile),
                start=m.start(), end=m.end(), matched=m.group(0),
                reason=f"Tat motif {m.group(0)!r} in N-term 40 aa",
            )]
        return []


class LipoboxNTermRule(Rule):
    """Lipoprotein lipobox: [LVI][ASTVI][GAS]C in first 30."""
    name = "lipobox_n_term"
    default_severity = Severity.HARD_FILTER
    _re = re.compile(r"[LVI][ASTVI][GAS]C")

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        if profile.compartment != "cytosolic":
            return []
        seq = ctx.sequence[:30]
        m = self._re.search(seq)
        if m:
            return [RuleHit(
                rule_name=self.name, severity=self.resolved_severity(profile),
                start=m.start(), end=m.end(), matched=m.group(0),
                reason=f"lipobox motif {m.group(0)!r} in N-term 30 aa",
            )]
        return []


# ----------------------------------------------------------------------
# Internal-sequence rules
# ----------------------------------------------------------------------


class KRNeighborDibasicRule(Rule):
    """K/R immediately adjacent to a fixed K/R catalytic residue.

    Structure-aware severity:
      - surface-loop neighbor -> HARD_OMIT (KR forbidden at sample time)
      - surface-helix neighbor -> SOFT_BIAS (downweight K/R)
      - buried neighbor -> WARN_ONLY (low protease access)

    Only relevant in compartments where dibasic motifs are recognized
    (periplasm = OmpT, cytoplasm in K12 = OmpT-like). For BL21 cytosolic
    (ompT-, lon-) we still warn because Lon can recognize exposed
    dibasic loops, but the default is permissive.
    """
    name = "kr_neighbor_dibasic"
    default_severity = Severity.HARD_OMIT
    requires_structure = True

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        seq = ctx.sequence
        params = profile.rule_params.get(self.name, {})
        sasa_cutoff = params.get("sasa_cutoff", 30.0)
        loop_severity = params.get("loop_severity", Severity.HARD_OMIT)
        helix_severity = params.get("helix_severity", Severity.SOFT_BIAS)
        buried_severity = params.get("buried_severity", Severity.WARN_ONLY)

        # Apply profile-level override for the rule as a whole
        if self.name in profile.severity_overrides:
            forced = profile.severity_overrides[self.name]
            loop_severity = forced
            helix_severity = forced
            buried_severity = forced

        hits: list[RuleHit] = []
        # Neighbors of fixed K/R
        for fixed_idx in ctx.fixed_resnos_zero_idx:
            if fixed_idx < 0 or fixed_idx >= ctx.L:
                continue
            if seq[fixed_idx] not in ("K", "R"):
                continue
            for nb in (fixed_idx - 1, fixed_idx + 1):
                if nb < 0 or nb >= ctx.L:
                    continue
                if nb in ctx.fixed_resnos_zero_idx:
                    continue
                # Pick severity from local SS + SASA
                ss = ctx.ss_reduced[nb] if ctx.ss_reduced else "L"
                sasa = float(ctx.sasa[nb]) if ctx.sasa is not None else 100.0
                if sasa < sasa_cutoff:
                    sev = buried_severity
                elif ss == "L":
                    sev = loop_severity
                else:                    # H, E
                    sev = helix_severity
                hits.append(RuleHit(
                    rule_name=self.name, severity=Severity(sev),
                    start=nb, end=nb + 1, matched=seq[nb] if seq[nb] else "",
                    suggested_omit_AAs="KR",
                    reason=f"position {nb + 1} is adjacent to fixed K/R at {fixed_idx + 1} "
                           f"(ss={ss}, sasa={sasa:.0f})",
                    metadata={"fixed_idx": int(fixed_idx),
                              "neighbor_ss": ss, "neighbor_sasa": sasa},
                ))
        return hits


class LongHydrophobicStretchRule(Rule):
    """Membrane-mimic risk: ~15-20 mostly hydrophobic in a row."""
    name = "long_hydrophobic_stretch"
    default_severity = Severity.SOFT_BIAS

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        params = profile.rule_params.get(self.name, {})
        win_min = params.get("window_min", 15)
        frac = params.get("frac_hydro", 0.70)

        seq = ctx.sequence
        hits: list[RuleHit] = []
        for i in range(0, len(seq) - win_min + 1):
            window = seq[i:i + win_min]
            n_hydro = sum(1 for c in window if c in HYDROPHOBIC_AAS)
            if n_hydro / win_min >= frac:
                # If we have SS and the stretch is in a known fold helix,
                # demote severity (helix-bundle hydrophobics are normal).
                sev = self.resolved_severity(profile)
                if ctx.ss_reduced is not None:
                    ss_window = ctx.ss_reduced[i:i + win_min]
                    if ss_window.count("H") / win_min >= 0.6:
                        sev = sev.demote()
                hits.append(RuleHit(
                    rule_name=self.name, severity=sev,
                    start=i, end=i + win_min, matched=window,
                    suggested_omit_AAs="LFW",
                    reason=f"long hydrophobic stretch {window!r} "
                           f"({n_hydro}/{win_min} hydrophobic)",
                ))
                break    # one hit per sequence
        return hits


class AmpLikePeptideRule(Rule):
    """Antimicrobial peptide-like region: cationic + amphipathic +
    hydrophobic. Flagged near termini where the protein hasn't yet
    folded into something protected.
    """
    name = "amp_like_peptide"
    default_severity = Severity.HARD_FILTER

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        params = profile.rule_params.get(self.name, {})
        win = params.get("window", 18)
        net_charge_min = params.get("net_charge_min", 4)
        hydro_frac_min = params.get("hydro_frac_min", 0.40)

        seq = ctx.sequence
        L = ctx.L
        hits: list[RuleHit] = []
        # Only check the N-terminal 50 and C-terminal 50 regions
        regions = [(0, min(50, L)), (max(0, L - 50), L)]
        for r_start, r_end in regions:
            for i in range(r_start, r_end - win + 1):
                w = seq[i:i + win]
                n_basic = sum(1 for c in w if c in BASIC_AAS)
                n_acid = sum(1 for c in w if c in ACIDIC_AAS)
                net = n_basic - n_acid
                n_hydro = sum(1 for c in w if c in HYDROPHOBIC_AAS)
                if net >= net_charge_min and n_hydro / win >= hydro_frac_min:
                    hits.append(RuleHit(
                        rule_name=self.name,
                        severity=self.resolved_severity(profile),
                        start=i, end=i + win, matched=w,
                        reason=f"AMP-like region {w!r} (net charge={net}, "
                               f"hydrophobic frac={n_hydro/win:.2f})",
                    ))
                    return hits   # one is enough
        return hits


class PolyprolineStallRule(Rule):
    """PPP / PPG / repeated proline stalls bacterial ribosome (EF-P helps)."""
    name = "polyproline_stall"
    default_severity = Severity.SOFT_BIAS

    _re_ppp = re.compile(r"P{3,}")
    _re_ppg = re.compile(r"(?:PPG){2,}")

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        sev = self.resolved_severity(profile)
        hits: list[RuleHit] = []
        for m in self._re_ppp.finditer(ctx.sequence):
            hits.append(RuleHit(
                rule_name=self.name, severity=sev,
                start=m.start(), end=m.end(), matched=m.group(0),
                suggested_omit_AAs="P",
                reason=f"polyproline run {m.group(0)!r}",
            ))
        for m in self._re_ppg.finditer(ctx.sequence):
            hits.append(RuleHit(
                rule_name=self.name, severity=sev,
                start=m.start(), end=m.end(), matched=m.group(0),
                suggested_omit_AAs="P",
                reason=f"PPG repeat {m.group(0)!r}",
            ))
        return hits


class SecMArrestRule(Rule):
    """SecM-like nascent-peptide arrest motif (FXXXXWIXXXXGIRAGP)."""
    name = "secm_arrest"
    default_severity = Severity.HARD_FILTER

    _re_loose = re.compile(r"F.{4}WI.{4}GIRAGP")
    _exact = "FSTPVWISQAQGIRAGP"

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        sev = self.resolved_severity(profile)
        hits: list[RuleHit] = []
        seq = ctx.sequence
        # Exact first
        idx = seq.find(self._exact)
        if idx >= 0:
            hits.append(RuleHit(
                rule_name=self.name, severity=sev,
                start=idx, end=idx + len(self._exact),
                matched=self._exact,
                reason=f"exact SecM arrest motif {self._exact!r}",
            ))
            return hits
        m = self._re_loose.search(seq)
        if m:
            hits.append(RuleHit(
                rule_name=self.name, severity=sev,
                start=m.start(), end=m.end(), matched=m.group(0),
                reason=f"SecM-like arrest motif {m.group(0)!r}",
            ))
        return hits


class CytosolicDisulfideOverloadRule(Rule):
    """Many Cys in cytosolic expression: reducing environment, no folding."""
    name = "cytosolic_disulfide_overload"
    default_severity = Severity.SOFT_BIAS

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        if profile.compartment != "cytosolic":
            return []
        params = profile.rule_params.get(self.name, {})
        max_cys = params.get("max_cys", 2)
        n_cys = ctx.sequence.count("C")
        if n_cys <= max_cys:
            return []
        return [RuleHit(
            rule_name=self.name,
            severity=self.resolved_severity(profile),
            start=0, end=ctx.L, matched="",
            suggested_omit_AAs="C",
            reason=f"{n_cys} Cys exceeds cytosolic limit {max_cys}; "
                   f"reducing cytoplasm prevents disulfide formation",
        )]


# ----------------------------------------------------------------------
# Tag-protease cleavage sites (only acted on if user plans cleavage)
# ----------------------------------------------------------------------


class _TagProteaseRule(Rule):
    """Base for TEV / 3C / thrombin / EK-style internal-site rules.

    Severity is HARD_FILTER if profile.cleave_protease matches this rule's
    target, else WARN_ONLY.
    """
    target_protease: str = ""
    pattern: re.Pattern = re.compile("")

    def _severity(self, profile: "ExpressionProfile") -> Severity:
        if self.name in profile.severity_overrides:
            return profile.severity_overrides[self.name]
        if profile.cleave_protease == self.target_protease:
            return Severity.HARD_FILTER
        return Severity.WARN_ONLY

    def evaluate(self, ctx: StructureContext, profile: "ExpressionProfile") -> list[RuleHit]:
        sev = self._severity(profile)
        hits: list[RuleHit] = []
        for m in self.pattern.finditer(ctx.sequence):
            hits.append(RuleHit(
                rule_name=self.name, severity=sev,
                start=m.start(), end=m.end(), matched=m.group(0),
                reason=f"internal {self.target_protease} cleavage site "
                       f"{m.group(0)!r} (cleave_protease={profile.cleave_protease})",
            ))
        return hits


class TEVSiteRule(_TagProteaseRule):
    name = "tev_site_internal"
    default_severity = Severity.WARN_ONLY
    target_protease = "TEV"
    pattern = re.compile(r"E.LYFQ[GS]")


class PreScissionSiteRule(_TagProteaseRule):
    name = "prescission_site_internal"
    default_severity = Severity.WARN_ONLY
    target_protease = "PreScission"
    pattern = re.compile(r"LEVLFQGP")


class ThrombinSiteRule(_TagProteaseRule):
    name = "thrombin_site_internal"
    default_severity = Severity.WARN_ONLY
    target_protease = "Thrombin"
    pattern = re.compile(r"LVPRGS")


class EnterokinaseSiteRule(_TagProteaseRule):
    name = "enterokinase_site_internal"
    default_severity = Severity.WARN_ONLY
    target_protease = "Enterokinase"
    pattern = re.compile(r"D{4}K")


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


for _cls in (
    SsrATagCTermRule, HydrophobicCTailRule,
    NEndRuleDestabilizingRule, MetAPWarningRule, SignalPeptideNTermRule,
    TatSignalMotifRule, LipoboxNTermRule,
    KRNeighborDibasicRule, LongHydrophobicStretchRule, AmpLikePeptideRule,
    PolyprolineStallRule, SecMArrestRule, CytosolicDisulfideOverloadRule,
    TEVSiteRule, PreScissionSiteRule, ThrombinSiteRule, EnterokinaseSiteRule,
):
    REGISTRY.register(_cls())


__all__ = [c.__name__ for c in (
    SsrATagCTermRule, HydrophobicCTailRule,
    NEndRuleDestabilizingRule, MetAPWarningRule, SignalPeptideNTermRule,
    TatSignalMotifRule, LipoboxNTermRule,
    KRNeighborDibasicRule, LongHydrophobicStretchRule, AmpLikePeptideRule,
    PolyprolineStallRule, SecMArrestRule, CytosolicDisulfideOverloadRule,
    TEVSiteRule, PreScissionSiteRule, ThrombinSiteRule, EnterokinaseSiteRule,
)]
