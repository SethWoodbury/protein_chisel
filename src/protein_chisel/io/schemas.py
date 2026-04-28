"""Typed artifact contracts for stage handoffs.

Every pipeline stage reads/writes one of:
- PoseSet         (list of poses + metadata)
- PositionTable   (per-residue features for one parent design)
- CandidateSet    (a population of designed sequences)
- MetricTable     (per-(sequence, conformer) metric rows)

Each artifact has a sidecar `_manifest.json` carrying:
- input file SHA-256s
- tool / model checkpoint versions
- hashed CLI args / config
- python package versions

Restart logic checks the manifest hash, not just file existence: if the
manifest doesn't match what would be produced now, the stage re-runs.

Design rules:
- Tabular artifacts use parquet so columns are typed and compact.
- Containers (PoseSet) keep poses as files on disk and a JSON sidecar
  catalog; we don't pickle PyRosetta poses.
- Anything carrying `meta: dict` is free-form (per-pose annotations).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

try:
    import pyarrow  # noqa: F401

    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False


def _table_write(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to disk. Prefers parquet; falls back to TSV.

    Returns the actual path written (may differ from the requested path's
    suffix if we fell back). Tools should always write via this helper and
    read back via `_table_read`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_PARQUET and path.suffix == ".parquet":
        df.to_parquet(path, index=False)
        return path
    out = path.with_suffix(".tsv")
    df.to_csv(out, sep="\t", index=False)
    return out


def _table_read(path: str | Path) -> pd.DataFrame:
    """Read parquet or TSV based on suffix."""
    path = Path(path)
    if not path.exists():
        # Maybe we wrote .tsv where caller expected .parquet, or vice versa.
        for alt in (path.with_suffix(".tsv"), path.with_suffix(".parquet")):
            if alt.exists():
                path = alt
                break
        else:
            raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's contents, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    """SHA-256 of any JSON-serialisable Python value (sorted keys, no NaN)."""
    payload = json.dumps(obj, sort_keys=True, default=_json_default).encode()
    return hashlib.sha256(payload).hexdigest()


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class Manifest:
    """Provenance for a stage's output.

    A receiving stage compares its expected manifest against the on-disk one;
    matches mean the prior output is reusable. Anything different (input file
    changed, tool version bumped, CLI args different) → re-run.
    """

    stage: str  # e.g. "classify_positions", "esmc_logits"
    inputs: dict[str, str]  # {"design.pdb": sha256, ...}
    config: dict[str, Any]  # CLI args / parameters that affect the result
    tool_versions: dict[str, str]  # {"pyrosetta": "...", "esm": "3.2.3", ...}
    package_versions: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    host: str = field(default_factory=platform.node)
    python: str = field(default_factory=lambda: sys.version.split()[0])

    def hash(self) -> str:
        """Stable hash of the manifest (excluding `created_at` and `host`)."""
        body = {
            "stage": self.stage,
            "inputs": self.inputs,
            "config": self.config,
            "tool_versions": self.tool_versions,
            "package_versions": self.package_versions,
        }
        return sha256_obj(body)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2, default=_json_default))

    @classmethod
    def from_json(cls, path: str | Path) -> "Manifest":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    @classmethod
    def for_stage(
        cls,
        stage: str,
        input_paths: Iterable[str | Path],
        config: dict[str, Any],
        tool_versions: Optional[dict[str, str]] = None,
    ) -> "Manifest":
        inputs = {str(Path(p).name): sha256_file(p) for p in input_paths}
        return cls(
            stage=stage,
            inputs=inputs,
            config=config,
            tool_versions=tool_versions or {},
            package_versions=_collect_versions(),
        )


def _collect_versions() -> dict[str, str]:
    """Snapshot a few common dependencies. Cheap; never raises."""
    versions: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "torch", "transformers", "esm", "biotite", "biopython"):
        try:
            mod = __import__(pkg)
        except Exception:
            continue
        v = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
        if v is not None:
            versions[pkg] = str(v)
    return versions


def manifest_matches(expected: Manifest, manifest_path: str | Path) -> bool:
    """True iff a manifest at `manifest_path` matches `expected.hash()`."""
    p = Path(manifest_path)
    if not p.is_file():
        return False
    try:
        existing = Manifest.from_json(p)
    except Exception:
        return False
    return existing.hash() == expected.hash()


# ---------------------------------------------------------------------------
# Artifact: PoseSet
# ---------------------------------------------------------------------------


@dataclass
class PoseEntry:
    """One pose in a PoseSet.

    `path` is a filesystem path to a PDB. Metadata fields make the entry
    self-describing so downstream tools can route per-conformer / per-source
    logic correctly.
    """

    path: str
    sequence_id: str  # "design_001" — same value across conformers of one sequence
    fold_source: str  # "designed" | "AF3_seed1" | "Boltz" | "RFdiffusion" | ...
    conformer_index: int = 0
    parent_design_id: Optional[str] = None  # the original scaffold design, if any
    is_apo: bool = False
    chain_id: Optional[str] = None  # main protein chain if not "A"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PoseSet:
    """A list of poses with metadata. Single-pose inputs become size-1 sets."""

    entries: list[PoseEntry]
    name: str = "pose_set"

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def filter(self, **kw) -> "PoseSet":
        """Subset by metadata equality (e.g. .filter(fold_source="designed"))."""
        out = []
        for e in self.entries:
            if all(getattr(e, k, None) == v for k, v in kw.items()):
                out.append(e)
        return PoseSet(out, name=f"{self.name}__{kw}")

    def by_sequence(self) -> dict[str, list[PoseEntry]]:
        """Group entries by sequence_id."""
        out: dict[str, list[PoseEntry]] = {}
        for e in self.entries:
            out.setdefault(e.sequence_id, []).append(e)
        return out

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"name": self.name, "entries": [dataclasses.asdict(e) for e in self.entries]}
        path.write_text(json.dumps(payload, indent=2, default=_json_default))

    @classmethod
    def from_json(cls, path: str | Path) -> "PoseSet":
        data = json.loads(Path(path).read_text())
        entries = [PoseEntry(**e) for e in data["entries"]]
        return cls(entries=entries, name=data.get("name", "pose_set"))

    @classmethod
    def from_single_pdb(
        cls, pdb_path: str | Path, sequence_id: str = "design", fold_source: str = "designed"
    ) -> "PoseSet":
        return cls(
            entries=[
                PoseEntry(
                    path=str(Path(pdb_path).resolve()),
                    sequence_id=sequence_id,
                    fold_source=fold_source,
                )
            ],
            name=Path(pdb_path).stem,
        )


# ---------------------------------------------------------------------------
# Artifact: PositionTable
# ---------------------------------------------------------------------------


# Canonical schema. Tools may add extra columns; these are the required ones
# every PositionTable must carry so downstream code can rely on them.
POSITION_TABLE_REQUIRED: tuple[str, ...] = (
    "pose_id",         # PoseEntry.sequence_id (for joining across artifacts)
    "resno",           # 1-indexed pose residue number
    "chain",           # PDB chain id
    "name3",           # 3-letter residue name (e.g. "HIS", "MET")
    "name1",           # 1-letter (e.g. "H", "M"); "X" for non-canonical
    "is_protein",      # bool — False for ligand / virtual
    "is_catalytic",    # bool — True if listed in REMARK 666
    "class",           # active_site | first_shell | pocket | buried | surface | ligand
    "sasa",            # float Å² (CA-side or per-residue, tool-specific; see metadata)
    "dist_ligand",     # min Å to any ligand heavy atom (np.nan if no ligand)
    "dist_catalytic",  # min Å to any catalytic-residue heavy atom (np.nan if none)
    "ss",              # DSSP full alphabet (H E L T S B G I -)
    "ss_reduced",      # DSSP reduced (H | E | L)
    "in_pocket",       # bool — fpocket membership
    "phi",
    "psi",
)


@dataclass
class PositionTable:
    """Per-residue features for one design (one PoseEntry).

    Backed by a parquet on disk; pandas DataFrame in memory.
    """

    df: pd.DataFrame  # columns include POSITION_TABLE_REQUIRED + tool extras

    def __post_init__(self):
        missing = [c for c in POSITION_TABLE_REQUIRED if c not in self.df.columns]
        if missing:
            raise ValueError(f"PositionTable missing required columns: {missing}")

    def to_parquet(self, path: str | Path) -> Path:
        return _table_write(self.df, path)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "PositionTable":
        return cls(df=_table_read(path))


# ---------------------------------------------------------------------------
# Artifact: CandidateSet
# ---------------------------------------------------------------------------


@dataclass
class CandidateSet:
    """A pool of designed sequences.

    Two artifacts on disk: a FASTA carrying the sequences (universal format)
    and a parquet sidecar carrying metadata. Joined by candidate id.
    """

    df: pd.DataFrame  # columns: id, sequence, parent_design_id, sampler, sampler_params_hash, ...

    def to_disk(self, fasta_path: str | Path, meta_path: str | Path) -> tuple[Path, Path]:
        fasta_path = Path(fasta_path)
        fasta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fasta_path, "w") as fh:
            for _, row in self.df.iterrows():
                fh.write(f">{row['id']}\n{row['sequence']}\n")
        meta_actual = _table_write(self.df, meta_path)
        return fasta_path, meta_actual

    @classmethod
    def from_disk(cls, meta_path: str | Path) -> "CandidateSet":
        return cls(df=_table_read(meta_path))


# ---------------------------------------------------------------------------
# Artifact: MetricTable
# ---------------------------------------------------------------------------


@dataclass
class MetricTable:
    """One row per (sequence, conformer); columns prefixed by source.

    Examples of columns: `rosetta__total_score`, `rosetta__ligand_iface_ddg`,
    `fpocket__volume`, `esmc__pseudo_perplexity`, `saprot__naturalness`,
    `cms__total`, `buns__count_no_whitelist`, ...

    The "id columns" `sequence_id` and `conformer_index` identify the row;
    everything else is a metric. Pareto / ranking operates on this table.
    """

    df: pd.DataFrame  # must contain sequence_id + conformer_index

    REQUIRED = ("sequence_id", "conformer_index")

    def __post_init__(self):
        missing = [c for c in self.REQUIRED if c not in self.df.columns]
        if missing:
            raise ValueError(f"MetricTable missing required columns: {missing}")

    def merge(self, other: "MetricTable", how: str = "outer") -> "MetricTable":
        """Combine metric tables on (sequence_id, conformer_index)."""
        merged = self.df.merge(other.df, on=list(self.REQUIRED), how=how, suffixes=("", "__dup"))
        dup_cols = [c for c in merged.columns if c.endswith("__dup")]
        if dup_cols:
            merged = merged.drop(columns=dup_cols)
        return MetricTable(df=merged)

    def to_parquet(self, path: str | Path) -> Path:
        return _table_write(self.df, path)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "MetricTable":
        return cls(df=_table_read(path))
