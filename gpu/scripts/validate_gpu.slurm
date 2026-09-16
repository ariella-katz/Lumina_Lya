#!/bin/bash
#SBATCH --job-name=lya-validation
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=gpu/lya-validation-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
lya_executable="${LYA_EXECUTABLE:-${SLURM_SUBMIT_DIR}/gpu/build/lumina_lya}"
# --cuda is required: ordinary CTest and this test without the flag use CPU.
srun python gpu/tests/test_pipeline.py "$lya_executable" --cuda -v
