#!/usr/bin/env bash
# Evaluation only: each lane runs moderate robustness first, then strong
# stress.  No checkpoint, reward, or training source is modified.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 LANE1_MODEL.pt LANE2_MODEL.pt LANE3_MODEL.pt LANE4_MODEL.pt LANE5_MODEL.pt" >&2
  exit 2
fi

models=("$@")
gpus=(${EVAL_GPUS:-2 3 4 5 6})
if [[ ${#gpus[@]} -ne 5 ]]; then
  echo "EVAL_GPUS must contain exactly five physical GPU indices." >&2
  exit 2
fi

episodes="${NUM_EPISODES:-256}"
envs="${NUM_ENVS:-128}"
seed="${SEED:-42}"
output_root="${EVAL_OUTPUT_ROOT:-outputs/eval/five_lanes_$(date +%Y%m%d_%H%M%S)}"
isaaclab_app="${ISAACLAB_APP:-isaaclab.sh}"
mkdir -p "$output_root"

run_stage() {
  local lane=$1 gpu=$2 model=$3 stage=$4 physics=$5 reality_gap=$6
  local out="$output_root/lane${lane}/${stage}"
  mkdir -p "$out"
  echo "[lane ${lane}] ${stage}: GPU ${gpu}, physics=${physics}, reality_gap=${reality_gap}"
  CUDA_VISIBLE_DEVICES="$gpu" "$isaaclab_app" -p scripts/rsl_rl/evaluate_race.py \
    --task "Template-Race-400m-Lane${lane}-SingleGPU" --headless \
    --checkpoint "$model" --num_episodes "$episodes" --num_envs "$envs" --seed "$seed" \
    --robustness_suite "$physics" --reality_gap_suite "$reality_gap" \
    --output_dir "$out" >"$out/console.log" 2>&1
}

run_lane() {
  local lane=$1 gpu=$2 model=$3
  run_stage "$lane" "$gpu" "$model" robustness_moderate moderate moderate
  run_stage "$lane" "$gpu" "$model" stress_strong strong strong
}

pids=()
for lane in 1 2 3 4 5; do
  run_lane "$lane" "${gpus[$((lane - 1))]}" "${models[$((lane - 1))]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one lane evaluation failed. See ${output_root}/lane*/**/console.log" >&2
  exit 1
fi
echo "All robustness and stress evaluations completed: ${output_root}"
