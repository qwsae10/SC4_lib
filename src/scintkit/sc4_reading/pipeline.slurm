#!/bin/bash
#SBATCH --job-name=sc4_pipeline
#SBATCH --output=sc4_pipeline_%j.out
#SBATCH --error=sc4_pipeline_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=16G

# Load modules (if required on your cluster)
module load miniconda

# Activate your conda environment
source activate scinkit_env

# Go to your project directory
cd /home/dal674840/scintkit/src/scintkit/sc4_reading

# Run the pipeline
python pipeline_trial.py