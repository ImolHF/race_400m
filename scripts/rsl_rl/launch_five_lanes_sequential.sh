#!/usr/bin/env bash
# Launch five independent one-GPU training runs in lane order.  A later launch
# cannot alter an earlier one: its Gym task names a lane-specific config class.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 GPU_LANE1 GPU_LANE2 GPU_LANE3 GPU_LANE4 GPU_LANE5" >&2
  exit 2
fi

gpus=("$@")
num_envs="${NUM_ENVS:-4096}"
max_iterations="${MAX_ITERATIONS:-8000}"
run_stamp="${RUN_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
log_dir="logs/lane_launch/${run_stamp}"
# Isaac Lab training must run through its bundled Python runtime.  Override
# this on the server, e.g. ISAACLAB_APP=/d/home/wei-chen/IsaacLab/isaaclab.sh.
isaaclab_app="${ISAACLAB_APP:-isaaclab.sh}"
if ! [[ -x "$isaaclab_app" ]] && ! command -v "$isaaclab_app" >/dev/null 2>&1; then
  echo "Isaac Lab launcher not found: $isaaclab_app (set ISAACLAB_APP)" >&2
  exit 2
fi
mkdir -p "$log_dir"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

pids=()
for lane in 1 2 3 4 5; do
  gpu="${gpus[$((lane - 1))]}"
  task="Template-Race-400m-Lane${lane}-SingleGPU"
  log_file="$log_dir/lane${lane}.log"
  run_name="lane${lane}_${run_stamp}"

  echo "Launching lane ${lane} on GPU ${gpu}: ${task}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$isaaclab_app" -p scripts/rsl_rl/train.py \
    --task "$task" --headless --num_envs "$num_envs" \
    --max_iterations "$max_iterations" --run_name "$run_name" \
    >"$log_file" 2>&1 &
  pid=$!
  pids+=("$pid")

  # Do not proceed to the next lane until this process has imported the task
  # and logged its own raw waypoint-zero reset position.
  ready=0
  for _ in $(seq 1 90); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Lane ${lane} exited during startup; see ${log_file}" >&2
      exit 1
    fi
    if grep -q "Robot reset aligned to waypoint 0" "$log_file"; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "Lane ${lane} did not confirm its configuration within 180 s; see ${log_file}" >&2
    exit 1
  fi
  echo "Lane ${lane} confirmed (pid=${pid})."
done

printf 'All five training jobs started. PIDs: %s\n' "${pids[*]}"
printf 'Logs: %s\n' "$log_dir"
