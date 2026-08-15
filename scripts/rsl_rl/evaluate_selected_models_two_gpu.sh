#!/usr/bin/env bash
# Evaluate the two selected G1 checkpoints concurrently: one Isaac Sim
# process per GPU. Run this from the repository root on the Linux server.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 /absolute/path/leg_only_high_cadence_model.pt /absolute/path/locked_elbow_arm_model.pt [output_dir]" >&2
  exit 2
fi

leg_checkpoint=$1
arm_checkpoint=$2
output_root=${3:-logs/rsl_rl/evaluation/selected_models_$(date +%Y%m%d_%H%M%S)}
episodes=${NUM_EPISODES:-100}
envs=${NUM_ENVS:-64}
seed=${SEED:-42}
suite=${ROBUSTNESS_SUITE:-nominal}
reality_gap_suite=${REALITY_GAP_SUITE:-nominal}
leg_gpu=${EVAL_GPU_LEG:-0}
arm_gpu=${EVAL_GPU_ARM:-1}

mkdir -p "$output_root"

# Each child sees one physical GPU as cuda:0. Do not add --distributed: these
# are independent inference jobs, not one distributed policy.
CUDA_VISIBLE_DEVICES="$leg_gpu" python scripts/rsl_rl/evaluate_race.py \
  --task Template-Race-400m-LegOnly-HighCadence --headless \
  --checkpoint "$leg_checkpoint" --num_episodes "$episodes" --num_envs "$envs" --seed "$seed" \
  --robustness_suite "$suite" \
  --reality_gap_suite "$reality_gap_suite" \
  --output_dir "$output_root/leg_only_high_cadence" \
  > "$output_root/leg_only_high_cadence.console.log" 2>&1 &
leg_pid=$!

CUDA_VISIBLE_DEVICES="$arm_gpu" python scripts/rsl_rl/evaluate_race.py \
  --task Template-Race-400m --headless \
  --checkpoint "$arm_checkpoint" --num_episodes "$episodes" --num_envs "$envs" --seed "$seed" \
  --robustness_suite "$suite" \
  --reality_gap_suite "$reality_gap_suite" \
  --output_dir "$output_root/locked_elbow_arm" \
  > "$output_root/locked_elbow_arm.console.log" 2>&1 &
arm_pid=$!

status=0
wait "$leg_pid" || status=1
wait "$arm_pid" || status=1

if [[ $status -ne 0 ]]; then
  echo "Evaluation failed. Inspect $output_root/*.console.log" >&2
  exit "$status"
fi

echo "Evaluation complete. Compare:"
echo "  $output_root/leg_only_high_cadence/report.md"
echo "  $output_root/locked_elbow_arm/report.md"
