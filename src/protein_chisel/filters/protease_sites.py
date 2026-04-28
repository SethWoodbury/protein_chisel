"""Sequence-only protease / motif blacklist.

Pure regex; runs on host. Built-in blacklist covers the most common
expression-toxic motifs in E. coli; users can pass an extra list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional


# Default blacklist patterns (regex; non-anchored).
# Names are descriptive; not all of these "cleave" — many are surrogate
# markers for expression-toxicity in our standard hosts.
DEFAULT_BLACKLIST: list[tuple[str, str]] = [
    # Kex2 dibasic site (yeast / sometimes E. coli surrogate)
    ("kex2_RR", r"RR"),
    # Trypsin cuts after K or R unless followed by P
    ("trypsin", r"[KR][^P]"),
    # OmpT outer-membrane protease — often cleaves between two Args/Lys
    ("ompT", r"[KR][KR]"),
    # Thrombin recognition (LVPR/GS-style) — only the conserved part
    ("thrombin", r"LVPR"),
    # Furin / proprotein convertase
    ("furin", r"R..R"),
    # Caspase cleavage sites (DXXD)
    ("caspase", r"D..D"),
    # PEST sequences (rough; full PEPstats-style scoring is more nuanced)
    # Skip here — too noisy for a regex filter.
]


@dataclass
class ProteaseHit:
    name: str
    pattern: str
    start: int   # 0-indexed
    end: int     # exclusive
    match: str   # actual matched substring


@dataclass
class ProteaseSitesResult:
    hits: list[ProteaseHit] = field(default_factory=list)

    def has_any(self) -> bool:
        return bool(self.hits)

    def by_name(self) -> dict[str, list[ProteaseHit]]:
        out: dict[str, list[ProteaseHit]] = {}
        for h in self.hits:
            out.setdefault(h.name, []).append(h)
        return out

    def to_dict(self, prefix: str = "protease__") -> dict[str, int]:
        out: dict[str, int] = {f"{prefix}n_total": len(self.hits)}
        for name, hits in self.by_name().items():
            out[f"{prefix}n_{name}"] = len(hits)
        return out


def find_protease_sites(
    sequence: str,
    extra_patterns: Iterable[tuple[str, str]] = (),
    skip_default: bool = False,
    host: Optional[str] = None,
) -> ProteaseSitesResult:
    """Find every blacklist motif occurrence in `sequence`.

    Args:
        sequence: protein sequence (1-letter, uppercase).
        extra_patterns: extra (name, regex) pairs beyond the default list.
        skip_default: if True, only ``extra_patterns`` are checked.
        host: optionally include a host-specific pattern set. Currently
            ``"ecoli"`` (default for our designs) or ``"yeast"``. When
            given, the host pattern list is merged into the default
            blacklist (or replaces it if ``skip_default=True``).
    """
    patterns: list[tuple[str, str]] = []
    if not skip_default:
        patterns.extend(DEFAULT_BLACKLIST)
    if host is not None:
        from protein_chisel.filters.expression_host import get_host_patterns

        patterns.extend(get_host_patterns(host))
    patterns.extend(extra_patterns)

    seq = sequence.upper()
    hits: list[ProteaseHit] = []
    for name, pat in patterns:
        for m in re.finditer(pat, seq):
            hits.append(ProteaseHit(
                name=name,
                pattern=pat,
                start=m.start(),
                end=m.end(),
                match=m.group(0),
            ))
    return ProteaseSitesResult(hits=hits)


__all__ = ["DEFAULT_BLACKLIST", "ProteaseHit", "ProteaseSitesResult", "find_protease_sites"]
