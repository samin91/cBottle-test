#!/usr/bin/env bash
#SBATCH -J train-coarse-inference-cbottle
#SBATCH --output /home/k/k202223/cBottle/logs/train_coarse_inference_cbottle_%j.log
#SBATCH -p gpu
#SBATCH -A bk1444
#SBATCH --time=12:00:00
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:1

cd /home/k/k202223/cBottle
module load python3/2025.01-gcc-13.3.0
source /home/k/k202223/cBottle/.venv/bin/activate

export V6_ICON_ZARR=/work/kd1453/rechunked_ngc3028/ngc3028_PT3H_6.zarr
export RAW_DATA_URL=/work/kd1453/rechunked_ngc3028/ngc3028_PT30M_10.zarr
export LAND_DATA_URL_6=/work/kd1453/rechunked_ngc3028/ngc3028_P1D_6.zarr
export SST_MONMEAN_DATA_URL_6=/work/bk1444/k202222/resources_cBottle/datasets/ngc3028_P1D_ts_monmean_6_renamed.zarr

python3 scripts/train_coarse.py \
    --output_dir /home/k/k202223/cBottle/output/training_coarse_inference \
    --loop.total_ticks 80 \
    --loop.steps_per_tick 30 \
    --loop.snapshot_ticks 6 \
    --loop.state_dump_ticks 6 \
    --loop.valid_min_samples 6 \
    --loop.batch_size 48 \
    --loop.batch_gpu 24 \
    --loop.monthly_sst_input