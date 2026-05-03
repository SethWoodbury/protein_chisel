"""Structural feature extraction (SS, SASA) decoupled from PyRosetta."""

from protein_chisel.structure.secondary_structure import (
    SSConsensus, SSProvider,
)


__all__ = ["SSConsensus", "SSProvider"]
