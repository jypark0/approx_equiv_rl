#!/bin/bash

DOMAIN="acrobot"
TASK="swingup"
VARIANT="original"
AGENT="relaxed_equiv"
SEED=1001

CMD=(python -u train.py
  device=cuda
  seed="${SEED}"
  task="${DOMAIN}_${TASK}_${VARIANT}"
  agent="${AGENT}/${DOMAIN}_${TASK}_${VARIANT}"
)

echo "[Executing command] ${CMD[*]}"
OMP_NUM_THREADS=8 "${CMD[@]}"
