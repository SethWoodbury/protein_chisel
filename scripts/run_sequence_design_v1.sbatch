#!/bin/bash
#SBATCH --job-name=chisel_design_v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-chisel-design-%j.out
#SBATCH --error=slurm-chisel-design-%j.err

# protein_chisel sequence_design_v1 — orchestrates 5 stages across 3 sifs.
#
# Usage:
#   sbatch run_sequence_design_v1.sbatch <pdb> <ligand_params> <out_dir>
#
# Stages 0 (classify) runs in pyrosetta.sif; stages 1-2 (PLM logits +
# fusion) in esmc.sif (gpu); stage 3 (sample) in mlfold.sif; stages 4-5
# (filter + diversity) in any sif with biopython.
#
# This sbatch runs all stages from a single invocation; each stage is
# restartable so you can break it up if needed.

set -euo pipefail

PDB="$1"
PARAMS="$2"
OUT="$3"

REPO=/home/woodbuse/codebase_projects/protein_chisel

# Stage 0: classify_positions — pyrosetta.sif, no GPU
echo "=== stage 0: classify_positions ==="
apptainer exec \
    --bind "${REPO}:/code" \
    --env "PYTHONPATH=/code/src:/pyrosetta" \
    /net/software/containers/pyrosetta.sif \
    python -c "
from pathlib import Path
from protein_chisel.io.schemas import PoseSet
from protein_chisel.pipelines.sequence_design_v1 import (
    SequenceDesignV1Config, stage_classify, stage_filter, stage_diversity,
    stage_fusion,
)

# Just stage 0 here — we do PLM/sampling in subsequent steps.
out = Path('$OUT')
out.mkdir(parents=True, exist_ok=True)
stage_classify(Path('$PDB'), [Path('$PARAMS').parent], out / '0_classify')
"

# Stages 1-2: PLM logits + fusion — esmc.sif, GPU
echo "=== stages 1-2: PLM logits + fusion ==="
apptainer exec --nv \
    --bind "${REPO}:/code" \
    --bind /net/databases/huggingface/esmc \
    --bind /net/databases/huggingface/saprot \
    --env "PYTHONPATH=/code/src" \
    --env "HF_HOME=/net/databases/huggingface/esmc" \
    --env "HF_HUB_CACHE=/net/databases/huggingface/esmc/hub" \
    /net/software/containers/users/woodbuse/esmc.sif \
    python -c "
from pathlib import Path
import numpy as np
from protein_chisel.io.pdb import extract_sequence
from protein_chisel.io.schemas import PositionTable
from protein_chisel.pipelines.sequence_design_v1 import (
    SequenceDesignV1Config, stage_plm_logits, stage_fusion,
)

cfg = SequenceDesignV1Config()
out = Path('$OUT')
pdb = Path('$PDB')
seq = extract_sequence(pdb, chain=cfg.chain)
esmc_lp, saprot_lp = stage_plm_logits(pdb, seq, out / '1_logits', cfg)
pt = PositionTable.from_parquet(out / '0_classify' / 'positions.tsv')
stage_fusion(esmc_lp, saprot_lp, pt, out / '2_fusion', cfg)
"

# Stage 3: LigandMPNN — mlfold.sif, GPU
echo "=== stage 3: LigandMPNN sampling ==="
apptainer exec --nv \
    --bind "${REPO}:/code" \
    --bind /net/software/lab/mpnn:/net/software/lab/mpnn \
    --bind /net/databases/mpnn:/net/databases/mpnn \
    --env "PYTHONPATH=/code/src" \
    /net/software/containers/mlfold.sif \
    python -c "
from pathlib import Path
import numpy as np
from protein_chisel.io.pdb import parse_remark_666
from protein_chisel.pipelines.sequence_design_v1 import (
    SequenceDesignV1Config, stage_sample,
)

cfg = SequenceDesignV1Config()
out = Path('$OUT')
bias = np.load(out / '2_fusion' / 'bias.npy')
catres = parse_remark_666(Path('$PDB'))
fixed = list(catres.keys())
stage_sample(Path('$PDB'), Path('$PARAMS'), bias, fixed, out / '3_sample', cfg)
"

# Stages 4-5: filters + diversity — esmc.sif (has Bio + numpy)
echo "=== stages 4-5: filter + diversity ==="
apptainer exec \
    --bind "${REPO}:/code" \
    --env "PYTHONPATH=/code/src" \
    /net/software/containers/users/woodbuse/esmc.sif \
    python -c "
from pathlib import Path
from protein_chisel.io.schemas import CandidateSet, PositionTable
from protein_chisel.pipelines.sequence_design_v1 import (
    SequenceDesignV1Config, stage_filter, stage_diversity,
)

cfg = SequenceDesignV1Config()
out = Path('$OUT')
candidates = CandidateSet.from_disk(out / '3_sample' / 'candidates.tsv')
filtered = stage_filter(candidates, out / '4_filter', cfg)
pt = PositionTable.from_parquet(out / '0_classify' / 'positions.tsv')
diverse = stage_diversity(filtered, pt, out / '5_diversity', cfg)
print(f'Final library: {len(diverse.df)} sequences in', out / '5_diversity' / 'library.fasta')
"

echo "=== done ==="
echo "Output: $OUT"
