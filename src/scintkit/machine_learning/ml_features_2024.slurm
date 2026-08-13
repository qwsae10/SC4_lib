#!/bin/bash
#SBATCH --job-name=scintpi_ml_2024
# Logs are redirected after startup into ~/features_run_<array-job-id>/.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=rodrigues,cpu-preempt
#SBATCH --exclude=c-05-08
# Four simultaneous array tasks, each on its own machine.
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Prevent two array tasks from sharing the same machine.
#SBATCH --exclusive
# Use all memory available on the assigned node. Each worker loads one raw file.
#SBATCH --mem=0
#SBATCH --time=48:00:00

set -euo pipefail

# Slurm cannot create a job-ID-named directory while opening #SBATCH logs, so
# redirect immediately after the task starts and the array job ID is known.
RUN_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_DIR="$HOME/features_run_${RUN_ID}"
mkdir -p "$RUN_DIR"
exec >"$RUN_DIR/ml_features_${RUN_ID}_${TASK_ID}.out" \
     2>"$RUN_DIR/ml_features_${RUN_ID}_${TASK_ID}.err"

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
echo "Array job ID: $RUN_ID"
echo "Array task: $TASK_ID/${SLURM_ARRAY_TASK_COUNT:-1}"
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
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --shard-count "$SLURM_ARRAY_TASK_COUNT" \
    --n-thresh 1190 \
    --s4-thresh 0.15 \
    --coordinate -7.213 -35.907 \
    --coordinate 43.271 -120.358 \
    --coordinate 46.907 -96.793
