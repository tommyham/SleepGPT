#!/bin/bash -l
# Local CAP fine-tuning runner (no srun/slurm)

set -e

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Replace with your pretrained SleepGPT checkpoint path.
pretrained_ckpt=/mnt/e/DataSet/Local/OpenData/capslpdb/checkpoint/pretrain/ModelCheckpoint.ckpt
cap_root=/mnt/e/DataSet/Local/OpenData/capslpdb/process

python3 run_cap_local.py train \
  --pretrained-ckpt "$pretrained_ckpt" \
  --cap-root "$cap_root" \
  --folds 1-4 \
  --num-gpus 1 \
  --num-workers 4 \
  --train-batch-size 8
