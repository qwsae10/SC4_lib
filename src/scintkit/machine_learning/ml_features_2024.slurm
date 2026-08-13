#!/bin/bash
#SBATCH --job-name=scintpi_ml_2024
#SBATCH --output=/home/igw180000/scratch/scintpi_ml_2024_%j.out
#SBATCH --error=/home/igw180000/scratch/scintpi_ml_2024_%j.err
#SBATCH --partition=rodrigues,cpu-preempt
#SBATCH --exclude=c-05-08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Use all memory available on the assigned node. Each worker loads one raw file.
#SBATCH --mem=0
#SBATCH --time=48:00:00

set -euo pipefail

unset PYTHONHOME
unset PYTHONPATH

source /opt/ohpc/pub/unpackaged/apps/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate scintkit

# Four file workers share four Slurm CPUs. Prevent NumPy/SciPy/Arrow from
# starting additional threads inside each worker process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export ARROW_NUM_THREADS=1

# Slurm runs a spool copy, so locate the checkout through SLURM_SUBMIT_DIR.
if [[ -f "$SLURM_SUBMIT_DIR/src/scintkit/machine_learning/batch_ml_features.py" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
    SCRIPT="$PROJECT_ROOT/src/scintkit/machine_learning/batch_ml_features.py"
elif [[ -f "$SLURM_SUBMIT_DIR/batch_ml_features.py" ]]; then
    SCRIPT="$SLURM_SUBMIT_DIR/batch_ml_features.py"
    PROJECT_ROOT="$(cd -- "$SLURM_SUBMIT_DIR/../../.." && pwd)"
else
    echo "Could not find batch_ml_features.py from: $SLURM_SUBMIT_DIR" >&2
    echo "Submit from the scintkit root or src/scintkit/machine_learning." >&2
    exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/src"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Workers/CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated memory: ${SLURM_MEM_PER_NODE:-all node memory}"
echo "Project: $PROJECT_ROOT"

cd "$PROJECT_ROOT"
python -u "$SCRIPT" \
    --input-root /titan/frodrigues/scintpi_storage \
    --output-dir /titan/frodrigues/scintpi_storage/ml_features \
    --year 2024 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --n-thresh 1190 \
    --s4-thresh 0.15 \
    --coordinate -7.213 -35.907 \
    --coordinate 43.271 -120.358 \
    --coordinate 46.907 -96.793
