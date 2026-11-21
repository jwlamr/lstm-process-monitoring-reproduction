#!/bin/bash
# Runs the official scripts (README commands) 3 times per task, 100 epochs
export TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3
cd /home/claude/repro
mkdir -p logs
for run in 1; do
  for t in next_activity next_time remaining_time; do
    python3 $t.py --dataset=helpdesk --epochs=100 --model_dir=./models/run$run --result_dir=./results/run$run > logs/${t}_run$run.log 2>&1 &
  done
done
wait
echo ALLDONE > logs/done.txt
