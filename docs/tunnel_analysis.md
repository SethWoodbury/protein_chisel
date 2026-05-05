# Tunnel Analysis

Two complementary tools are used to evaluate solvent / substrate access tunnels
in protein-chisel designs:

| Tool                                                      | Use case             | Speed       |
|-----------------------------------------------------------|----------------------|-------------|
| `src/protein_chisel/tools/tunnel_metrics.py` (in-house)   | Inline filtering     | ~ms / PDB   |
| **CAVER 3.0.3** (this doc)                                | Offline validation   | 1-10 s/PDB  |

CAVER is the gold-standard tunnel-analysis package (Chovancova et al., PLoS
Comput Biol 2012). It performs Voronoi-based tunnel detection, average-link
clustering, and bottleneck profiling. It is **too slow** to run inside the
`iterative_design_v2` hot path, so we use it offline to validate the top
designs (e.g. top 5) after a campaign completes.

## Install

| Item             | Path                                                     |
|------------------|----------------------------------------------------------|
| CAVER home       | `/net/software/lab/CAVER/caver_3.0.3/caver`              |
| Main jar         | `/net/software/lab/CAVER/caver_3.0.3/caver/caver.jar`    |
| Examples         | `/net/software/lab/CAVER/caver_3.0.3/examples/`          |
| User guide       | `/net/software/lab/CAVER/caver_3.0.3/user_guide/caver_userguide.pdf` |
| Java             | `/usr/bin/java` (OpenJDK 25 — CAVER needs 8+, so any system Java works) |

The lab-shared install is world-readable and benefits other users on the
cluster. No site-packages or modules required: CAVER is a self-contained Java
distribution.

Source: downloaded from
`https://www.caver.cz/fil/download/caver30/303/caver_3.0.3.zip` (33 MB, 2016
release — current latest of the 3.x line).

## Invocation

Use the wrapper at `scripts/run_caver.sh`:

```bash
scripts/run_caver.sh \
    --pdb path/to/design.pdb \
    --catres_resnos 60,64,128,132,157 \
    --out_dir caver_out/design_001 \
    --min_probe_radius 0.9
```

Either supply explicit `--start_x X --start_y Y --start_z Z` coordinates, or
pass `--catres_resnos` (comma-separated residue numbers, default chain `A`,
override with `--chain`) — the wrapper computes the Cα centroid as the CAVER
starting point.

The wrapper writes raw CAVER outputs into `--out_dir` (CSV + per-tunnel PDBs
under `analysis/` and `data/tunnels/`) and prints a JSON summary on stdout:

```json
{
  "status": "ok",
  "wall_time_s": 1.513,
  "n_tunnels": 2,
  "starting_point": [3.50, 2.18, 0.77],
  "min_probe_radius": 0.9,
  "narrowest_bottleneck_radius": 1.44,
  "longest_tunnel_length": 20.18,
  "best_throughput": 0.76,
  "tunnels": [...]
}
```

## Wall time

On a typical PTE_i1 design (ZAPP_…_FS269.pdb, ~3200 atoms, single snapshot,
probe radius 0.9 Å) with 12 approximating balls and clustering threshold 3.5:

- **~1.5 s wall-time per PDB** on the cluster login node (Java 25, single-threaded).

For a top-5 validation pass this is fine; for inline use across thousands of
sampled designs it is not.

## When to use CAVER

Run CAVER **offline** on the top N designs after `iterative_design_v2`
finishes. Typical workflow:

1. Run `iterative_design_v2` with inline `tunnel_metrics` filtering
   (millisecond/PDB).
2. Pick top 5-10 designs by composite score.
3. For each design, run `scripts/run_caver.sh` and inspect the JSON summary +
   `analysis/tunnel_characteristics.csv`.
4. Visualise the cheapest tunnels (`out/data/tunnels/*.pdb`) in PyMOL/VMD using
   the auto-generated scripts under `out/pymol/` and `out/vmd/`.

## CAVER vs. inline tunnel_metrics

The in-house `tunnel_metrics` (located at
`src/protein_chisel/tools/tunnel_metrics.py`) is a fast pocket-channel
estimator built for high-throughput filtering, not for definitive geometric
tunnel reporting. Expected differences:

- **Speed**: tunnel_metrics is ~1000× faster (ms vs s).
- **Geometry**: CAVER builds an exact Voronoi diagram and traces multiple
  alternative tunnels with bottleneck profiles; tunnel_metrics produces a
  coarser channel-radius proxy.
- **Clustering**: CAVER clusters tunnels across snapshots; tunnel_metrics is
  per-frame.
- **Outputs**: CAVER emits per-residue contact lists, bottleneck residues,
  PyMOL/VMD scenes; tunnel_metrics emits scalar summary metrics only.

Treat the inline metric as a screen and CAVER as ground truth. Disagreement
between the two on a top-ranked design is a signal worth investigating.

## Constraints / gotchas

- CAVER is **offline-only** — do not import into the hot path.
- CAVER requires a *directory* of PDBs, not a single file; the wrapper handles
  the staging.
- Probe radius 0.9 Å is the convention used in our PTE_i1 work (water + small
  substrate). The CAVER default is 0.9 Å as well.
- The starting point must be inside the protein; if CAVER reports
  `Starting point not inside the protein`, double-check the centroid you
  passed.
- The 3.0.3 jar is from 2016 and is the latest of the 3.x line — newer CAVER
  features live in CAVER Analyst 2.0 (GUI) and CaverDock (docking), not the
  command-line tunnel calculator.

## References

- Chovancova et al. (2012) *CAVER 3.0: A Tool for the Analysis of Transport
  Pathways in Dynamic Protein Structures*, PLoS Comput Biol 8:e1002708.
  doi:10.1371/journal.pcbi.1002708
- CAVER home: <https://www.caver.cz/>
