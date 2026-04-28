# configs/

YAML configs for runs. Each pipeline accepts `--config configs/<name>.yaml`.

Conventions:
- One file per run.
- Inputs (paths to PDB, theozyme spec, ligand) at the top.
- Pipeline parameters (sample counts, filter thresholds, fusion weights) in the middle.
- Output directory at the bottom.
