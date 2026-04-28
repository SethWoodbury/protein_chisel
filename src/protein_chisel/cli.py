"""`chisel` CLI entrypoint.

Each subcommand is a thin wrapper around a tool or pipeline. As tools are
added, register them here so they're discoverable via `chisel --help`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from protein_chisel import __version__


@click.group(help="protein_chisel — refine sequences for de novo enzyme designs.")
@click.version_option(version=__version__)
def main() -> None:
    pass


# ---- Pipelines ------------------------------------------------------------


@main.command("comprehensive-metrics")
@click.argument("pdb", type=click.Path(exists=True, dir_okay=False, path_type=Path), nargs=-1, required=True)
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Output directory for per-pose subdirs and the merged MetricTable.")
@click.option("--params", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="Ligand .params files or directories containing them.")
@click.option("--target-atoms", multiple=True,
              help="Ligand atom names for per-atom SASA (e.g. C1 O1 O2).")
@click.option("--sequence-id", default=None, type=str,
              help="Sequence id (defaults to PDB stem). For multi-PDB invocations,"
                   " each PDB becomes its own conformer of this sequence_id.")
@click.option("--fold-source", default="designed", type=str,
              help="fold_source label (designed | AF3_seedN | RFdiffusion | Boltz | AF3_refined …).")
@click.option("--no-skip-existing", is_flag=True, help="Disable restart-skip; always rerun.")
def comprehensive_metrics_cli(
    pdb: tuple[Path, ...],
    out_dir: Path,
    params: tuple[Path, ...],
    target_atoms: tuple[str, ...],
    sequence_id: Optional[str],
    fold_source: str,
    no_skip_existing: bool,
) -> None:
    """Run the structural metric battery on one or more PDBs.

    Multiple PDBs become a multi-conformer PoseSet. Default sequence_id is
    the first PDB's stem; conformer_index is assigned by argument order.
    """
    from protein_chisel.io.schemas import PoseEntry, PoseSet
    from protein_chisel.pipelines.comprehensive_metrics import (
        ComprehensiveMetricsConfig, run_comprehensive_metrics,
    )

    sid = sequence_id or pdb[0].stem
    entries = [
        PoseEntry(
            path=str(p.resolve()),
            sequence_id=sid,
            fold_source=fold_source if i == 0 else f"{fold_source}_{i}",
            conformer_index=i,
        )
        for i, p in enumerate(pdb)
    ]
    pose_set = PoseSet(entries=entries, name=sid)
    cfg = ComprehensiveMetricsConfig(ligand_target_atoms=tuple(target_atoms))
    result = run_comprehensive_metrics(
        pose_set, out_dir=out_dir, params=list(params), config=cfg,
        skip_existing=not no_skip_existing,
    )
    click.echo(f"Wrote MetricTable with {len(result.metric_table.df)} rows to {out_dir}")
    # Print a small summary as JSON for downstream tools.
    click.echo(json.dumps(
        {
            "out_dir": str(out_dir),
            "n_rows": int(len(result.metric_table.df)),
            "sequence_ids": result.metric_table.df["sequence_id"].unique().tolist(),
        },
        indent=2,
    ))


# ---- Standalone tools (when running outside a pipeline) -------------------


@main.command("classify-positions")
@click.argument("pdb", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True,
              help="Output path for the PositionTable (parquet or .tsv).")
@click.option("--params", multiple=True, type=click.Path(exists=True, path_type=Path))
@click.option("--catres-spec", multiple=True, type=str,
              help="Fallback catalytic-residue spec (e.g. A94-96 B101).")
def classify_positions_cli(
    pdb: Path, out_path: Path, params: tuple[Path, ...], catres_spec: tuple[str, ...],
) -> None:
    """Build a PositionTable for a single PDB."""
    from protein_chisel.tools.classify_positions import classify_positions

    pt = classify_positions(
        pdb, params=list(params),
        catres_spec=list(catres_spec) if catres_spec else None,
    )
    actual = pt.to_parquet(out_path)
    click.echo(f"PositionTable: {len(pt.df)} rows -> {actual}")


@main.command("esmc-score")
@click.argument("sequence", type=str)
@click.option("--model", default="esmc_300m", type=str,
              help="esmc_300m or esmc_600m.")
@click.option("--device", default="auto", type=str)
def esmc_score_cli(sequence: str, model: str, device: str) -> None:
    """Compute ESM-C pseudo-perplexity for a single sequence."""
    from protein_chisel.tools.esmc import esmc_score

    res = esmc_score(sequence, model_name=model, device=device)
    click.echo(json.dumps(res.to_dict(), indent=2))


@main.command("saprot-score")
@click.argument("pdb", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--chain", default=None, type=str)
@click.option("--model", default="saprot_35m", type=str,
              help="saprot_35m / saprot_650m / saprot_1.3b.")
@click.option("--device", default="auto", type=str)
def saprot_score_cli(pdb: Path, chain: Optional[str], model: str, device: str) -> None:
    """Compute SaProt pseudo-perplexity for a structured protein."""
    from protein_chisel.tools.saprot import saprot_score

    res = saprot_score(pdb, chain=chain, model_name=model, device=device)
    click.echo(json.dumps(res.to_dict(), indent=2))


if __name__ == "__main__":
    main()
