#!/bin/bash
#SBATCH --job-name=lumina-lya
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=gpu/lya-%j.out

# Submit from the repository root with input, source list, output directory,
# then any additional native options. This script never submits another job.
set -euo pipefail
if (( $# < 3 )); then
    echo "Usage: sbatch gpu/scripts/run.slurm INPUT SOURCES OUTPUT [OPTIONS...]" >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
lya_executable="${LYA_EXECUTABLE:-${SLURM_SUBMIT_DIR}/gpu/build/lumina_lya}"
srun "$lya_executable" run --input "$1" --z0-file "$2" --output-dir "$3" \
    "${@:4}" --backend cuda
