#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=8g
#SBATCH -J "RPS simulation" 
#SBATCH -p short
#SBATCH -t 4:00:00 
#SBATCH --gres=gpu:1

echo "loading modules"
module load cuda12.6
source ./.venv/bin/activate
export HOSTNAME

echo "begin execute python"

python
