"""Length / terminal-AA constraints. Trivial sequence filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class LengthFilterConfig:
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    forbidden_n_terminal: Iterable[str] = ()  # AAs not allowed at position 1
    forbidden_c_terminal: Iterable[str] = ()  # AAs not allowed at position L
    must_start_with: Optional[str] = None
    must_end_with: Optional[str] = None


def passes_length_filter(sequence: str, cfg: LengthFilterConfig) -> tuple[bool, str]:
    """Return (passed, reason). `reason` is empty when passed."""
    n = len(sequence)
    if cfg.min_length is not None and n < cfg.min_length:
        return False, f"length {n} < min_length {cfg.min_length}"
    if cfg.max_length is not None and n > cfg.max_length:
        return False, f"length {n} > max_length {cfg.max_length}"
    # Empty sequence with terminal constraints fails — those constraints
    # by definition can't be satisfied.
    if not sequence:
        if cfg.must_start_with or cfg.must_end_with:
            return False, "empty sequence cannot satisfy terminal constraints"
        return True, ""

    nt = sequence[0].upper()
    ct = sequence[-1].upper()
    if cfg.must_start_with and nt != cfg.must_start_with.upper():
        return False, f"N-term is {nt!r}; must be {cfg.must_start_with!r}"
    if cfg.must_end_with and ct != cfg.must_end_with.upper():
        return False, f"C-term is {ct!r}; must be {cfg.must_end_with!r}"
    if nt in {a.upper() for a in cfg.forbidden_n_terminal}:
        return False, f"forbidden N-term residue {nt!r}"
    if ct in {a.upper() for a in cfg.forbidden_c_terminal}:
        return False, f"forbidden C-term residue {ct!r}"
    return True, ""


__all__ = ["LengthFilterConfig", "passes_length_filter"]
