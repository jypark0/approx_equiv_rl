#!/bin/bash

SNAPSHOT_DIR="exp_local/exp/acrobot_swingup_plain_sky_no_grid_original/RelaxedEquivDrQv2/seed1001"
NOISE_STD=0.02

CMD=(python -u eval.py
  --snapshot_dir="${SNAPSHOT_DIR}"
  --test_num_episodes=50
  --noise_std="${NOISE_STD}"
)

echo "[Executing command] ${CMD[*]}"
"${CMD[@]}"
