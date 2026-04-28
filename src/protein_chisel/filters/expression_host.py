"""Expression-host avoidance patterns: E. coli (default) and yeast.

The lists target two failure modes:

1. **Cleavage / degradation** — proteases recognized by the host's
   intracellular or periplasmic machinery that would chew up the
   designed protein.
2. **Post-translational modifications** — sites that the host modifies
   without being asked. These are *usually* unwanted; mark them as
   forbidden unless the design intends the modification (e.g. KCX —
   carbamylation on a catalytic lysine — is intended in our PTE
   designs and is excluded from this list because we don't want the
   filter to flag those positions).

Each pattern is a Python regex against the 1-letter sequence. ``[^P]``
notation captures common "unless followed by Pro" exceptions.

The default ``HOST_E_COLI`` list is conservative: patterns that are
clear losses, not borderline ones. ``HOST_YEAST`` covers Pichia /
Saccharomyces production hosts. To compose with your own list, pass
``extra_patterns=`` to ``find_protease_sites``.
"""

from __future__ import annotations

# E. coli intracellular / periplasmic proteases. These are well-known
# and often documented in protein expression FAQs.
E_COLI_PROTEASE_SITES: list[tuple[str, str]] = [
    # OmpT — periplasmic, cleaves between two basic residues
    ("ompT_KK", r"KK"),
    ("ompT_KR", r"KR"),
    ("ompT_RK", r"RK"),
    ("ompT_RR", r"RR"),
    # Lon and Clp — generally degrade exposed hydrophobic stretches; we
    # use unstructured-charged-rich patches as a proxy at the regex level.
    # (Not regex-detectable cleanly — leave to SAP / instability_index.)
    # Trypsin (relevant if doing in-vitro digestion / mass-spec)
    ("trypsin", r"[KR][^P]"),
]

# E. coli post-translational modifications and their motifs.
# "intended_kcx" (carbamylated lysine) is intentionally excluded — our PTE
# designs use it. To re-enable: include it manually.
E_COLI_PTM_SITES: list[tuple[str, str]] = [
    # Met aminopeptidase removes N-terminal Met if pos 2 is small (A, C, G,
    # P, S, T, V). We usually rely on this — flag only if the user wants
    # the M kept (e.g. for tag stability).
    # ("met_aminopeptidase_at_pos2", r"^M[ACGSTV]"),
    # N-glycosylation (eukaryotic; absent in E. coli K12 but flag for
    # cross-host moves).
    # ("n_glycosylation", r"N[^P][ST]"),  # off by default for E. coli
    # Phosphorylation: rare in cytoplasmic E. coli but Ser-Thr kinases
    # exist; we leave this off to avoid over-filtering.
]

# Yeast (Pichia + Saccharomyces) host-specific patterns.
YEAST_PROTEASE_SITES: list[tuple[str, str]] = [
    # Kex2 — Golgi protease that cleaves dibasic sites (KR > RR > KK; the
    # canonical KR site is the most aggressive).
    ("kex2_KR", r"KR"),
    ("kex2_RR", r"RR"),
    # Yapsin / Mkc7 — cleaves single basic + Asn? More subtle; leaving off.
]

YEAST_PTM_SITES: list[tuple[str, str]] = [
    # N-glycosylation: NX[ST] where X != P. Yeast hyperglycosylates —
    # active-site N-glycans can wreck activity.
    ("n_glycosylation_NXS_NXT", r"N[^P][ST]"),
    # O-glycosylation (Ser/Thr in Pichia surface-exposed) — sequence-only
    # detection is unreliable; left out.
]

# General problematic motifs regardless of host.
GENERAL_FORBIDDEN: list[tuple[str, str]] = [
    # Internal Met start sites — could generate truncated alternative
    # products in some translation systems.
    ("internal_met_start", r"(?<=.)M[A-Z]{8,}M"),
    # Long polyG / polyN / polyQ runs — aggregation prone.
    ("poly_G_long", r"G{6,}"),
    ("poly_N_long", r"N{4,}"),
    ("poly_Q_long", r"Q{4,}"),
    # Three-residue runs of cysteines or prolines — folding pathologies.
    ("CCC", r"C{3,}"),
    ("PPPP", r"P{4,}"),
    # PEST-like (rough heuristic; full PEPstats is more nuanced)
    # ("pest_motif", r"[PEST]{6,}"),
]


def get_host_patterns(
    host: str = "ecoli",
    include_ptm: bool = True,
    include_general: bool = True,
) -> list[tuple[str, str]]:
    """Return the merged pattern list for a given expression host.

    Args:
        host: ``"ecoli"`` or ``"yeast"``.
        include_ptm: include PTM motifs.
        include_general: include the host-agnostic general pattern list.

    Each entry is ``(name, regex)``.
    """
    out: list[tuple[str, str]] = []
    h = host.lower()
    if h in {"ecoli", "e_coli", "e.coli"}:
        out.extend(E_COLI_PROTEASE_SITES)
        if include_ptm:
            out.extend(E_COLI_PTM_SITES)
    elif h in {"yeast", "saccharomyces", "pichia"}:
        out.extend(YEAST_PROTEASE_SITES)
        if include_ptm:
            out.extend(YEAST_PTM_SITES)
    else:
        raise ValueError(f"unknown host: {host!r}; expected 'ecoli' or 'yeast'")
    if include_general:
        out.extend(GENERAL_FORBIDDEN)
    return out


__all__ = [
    "E_COLI_PROTEASE_SITES",
    "E_COLI_PTM_SITES",
    "GENERAL_FORBIDDEN",
    "YEAST_PROTEASE_SITES",
    "YEAST_PTM_SITES",
    "get_host_patterns",
]
