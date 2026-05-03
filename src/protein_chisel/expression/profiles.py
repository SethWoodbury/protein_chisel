"""Expression-host profiles.

A profile is a (host, compartment, tags, induction, severity overrides,
per-rule numeric knobs) bundle. Presets cover the common cases; users
can subclass / load from YAML for custom hosts.

Default for the Baker Lab PTE_i1 case:
- host = "BL21" (ompT-, lon-)
- compartment = "cytosolic"
- N-tag = "MSG", C-tag = "GSAWSHPQFEK" (Strep-tag II)
- induction = "IPTG"
- MetAP cleaves the leading M -> mature N-term = "SG..." -> safe
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from protein_chisel.expression.severity import Severity


LOGGER = logging.getLogger("protein_chisel.expression.profiles")


@dataclass
class ExpressionProfile:
    name: str = "default"
    host: str = "BL21"                          # "BL21" / "K12" / "yeast"
    compartment: str = "cytosolic"              # / "periplasmic" / "secreted"
    induction: str = "IPTG"                     # / "AHL" / "constitutive"
    n_tag: str = ""                             # tag added at N-term (NOT designed)
    c_tag: str = ""                             # tag added at C-term
    metap_cleaves_n_terminal_M: bool = True
    cleave_protease: Optional[str] = None       # "TEV" / "PreScission" / None
    preset: str = "standard"                    # "strict" / "standard" / "permissive"
    severity_overrides: dict[str, Severity] = field(default_factory=dict)
    rule_params: dict[str, dict] = field(default_factory=dict)

    def with_overrides(
        self, overrides: dict[str, Severity],
    ) -> "ExpressionProfile":
        """Return a new profile with ``overrides`` merged into severity_overrides."""
        new_overrides = {**self.severity_overrides, **overrides}
        return replace(self, severity_overrides=new_overrides)

    @classmethod
    def bl21_cytosolic_streptag(cls) -> "ExpressionProfile":
        """Baker-Lab default: BL21, cytosolic, IPTG, MSG-...-GSAWSHPQFEK."""
        return cls(
            name="bl21_cytosolic_streptag",
            host="BL21",
            compartment="cytosolic",
            induction="IPTG",
            n_tag="MSG",
            c_tag="GSAWSHPQFEK",
            metap_cleaves_n_terminal_M=True,
            cleave_protease=None,
            preset="standard",
        )

    @classmethod
    def k12_cytosolic(cls) -> "ExpressionProfile":
        """K-12 strain: ompT+ and lon+, so dibasic OmpT motifs are real."""
        return cls(
            name="k12_cytosolic",
            host="K12",
            compartment="cytosolic",
            induction="IPTG",
            preset="strict",
        )

    @classmethod
    def bl21_periplasmic(cls) -> "ExpressionProfile":
        """Periplasmic expression: signal peptide intentional, OmpT in periplasm."""
        return cls(
            name="bl21_periplasmic",
            host="BL21",
            compartment="periplasmic",
            induction="IPTG",
            preset="strict",
        )

    @classmethod
    def from_overrides_string(
        cls, base: "ExpressionProfile", overrides_string: str,
    ) -> "ExpressionProfile":
        """Parse a comma-sep "rule_name=SEVERITY" string into the profile.

        Used for sbatch / CLI: ``--expression_override
        kr_neighbor_dibasic=SOFT_BIAS,polyproline_stall=WARN_ONLY``.
        """
        if not overrides_string.strip():
            return base
        out: dict[str, Severity] = {}
        for entry in overrides_string.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise ValueError(f"override missing '=': {entry!r}")
            name, sev_str = entry.split("=", 1)
            try:
                sev = Severity[sev_str.strip().upper()]
            except KeyError as e:
                raise ValueError(
                    f"unknown severity {sev_str!r} for rule {name!r}; "
                    f"expected one of {[s.name for s in Severity]}"
                ) from e
            out[name.strip()] = sev
        return base.with_overrides(out)


__all__ = ["ExpressionProfile"]
