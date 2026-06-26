#!/usr/bin/env bash
#SBATCH -J superresolution-cbottle
#SBATCH --output /home/k/k202223/cBottle/logs/cbottle_super_resolution_%j.log
#SBATCH -p gpu
#SBATCH -A bk1444
#SBATCH --time=01:00:00
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:1

cd /home/k/k202223/cBottle

module load python3/2025.01-gcc-13.3.0
source /home/k/k202223/cBottle/.venv/bin/activate


python scripts/inference_multidiffusion.py \
/home/k/k202223/cBottle/checkpoints/cBottle-SR.zip \
/home/k/k202223/cBottle/output/super_resolution \
  --input-path /home/k/k202223/cBottle/output/coarse-inference/0.nc