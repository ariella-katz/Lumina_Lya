#!/bin/bash
#SBATCH --job-name=lya-validation
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=gpu/lya-validation-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dal440490@utdallas.edu

set -euo pipefail
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v miniconda3 | paste -sd: || true)
export LD_LIBRARY_PATH=/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:${LD_LIBRARY_PATH}
source "${SLURM_SUBMIT_DIR}/Lya-env/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
lya_executable="${LYA_EXECUTABLE:-${SLURM_SUBMIT_DIR}/gpu/build/lumina_lya}"
# --cuda is required: ordinary CTest and this test without the flag use CPU.
srun python gpu/tests/test_pipeline.py "$lya_executable" --cuda -v
