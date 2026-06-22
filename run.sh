#!/usr/bin/env bash
#SBATCH -J coarse-inference-cbottle
#SBATCH --output /home/k/k202223/logs/cbottle_coarse_inference_%j.log
#SBATCH -p gpu
#SBATCH -A bk1444
#SBATCH --time=00:25:00
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:1

cd /home/k/k202223/cBottle
source /home/k/k202223/.venv/bin/activate

export AMIP_MID_MONTH_SST=/home/k/k202223/.cache/cbottle/amip_midmonth_sst.nc
export PROJECT_ROOT="/home/k/k202223/cBottle"

python scripts/inference_coarse.py \
/home/k/k202223/cBottle/checkpoints/training-state-009856000.checkpoint \
/home/k/k202223/cBottle/output/coarse-inference \
  --dataset amip \
  --sample.mode sample \
  --start_time "2020-01-01 12:00:00" \
  --end_time   "2020-01-01 13:00:00" \
  --timestamp_frequency "h"