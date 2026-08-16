# G1 400 m Ordered-Waypoint Race

Train a Unitree G1 to complete one 400 m lap by passing fixed, ordered
ground-plane targets.  The course contains 201 coordinates: the 0 m start plus
one target every 2 m through 400 m.  The policy has no camera, LiDAR, or map
input.  It receives proprioception and the direction/distance to the next
target, and outputs 12 leg joint-position residual actions.

The task name for RSL-RL training is exactly:

```bash
Template-Race-400m
```

`Race400m-v0` is a legacy Gym registration and must not be used with
`scripts/rsl_rl/train.py`.

## Server setup

The server needs a compatible Isaac Lab / Isaac Sim installation and a Conda
environment named `isaaclab`.  Clone this repository, then install the local
package in editable mode:

```bash
git clone https://github.com/ImolHF/race_400m.git
cd race_400m
conda activate isaaclab
python -m pip install -e source/race_400m
```

The G1 USD is included through the repository-relative `unitree_model` path;
do not replace it with a local Windows absolute path.

## Two-GPU configuration

The project is configured for two GPUs. `torchrun` starts one Isaac Sim and one
PhysX scene per GPU; `--num_envs` is therefore the environment count **per
GPU**, not the total.

Before training, limit CPU BLAS/OpenMP thread oversubscription.  This does not
remove all CPU work, but prevents two training processes from creating many
competing CPU worker threads.

```bash
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

The environment uses GPU PhysX, Fabric, GPU-resident waypoint reward tensors,
and GPU contact sensors.  It does not render cameras or use height scans during
training.  PhysX buffers are sized for 4096 G1 environments per GPU.

## Smoke test

Run this first. It confirms distributed launch, package installation, contact
sensors, and PPO training without committing significant server time:

```bash
torchrun --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \
  --task Template-Race-400m --headless --distributed \
  --num_envs 256 --max_iterations 2 --run_name smoke_test
```

## Training from scratch

Use this only when you intentionally want a new policy. It learns both running
and the 200-point navigation task from scratch:

```bash
torchrun --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \
  --task Template-Race-400m --headless --distributed \
  --num_envs 4096 --max_iterations 5000 --run_name race_from_scratch
```

This creates logs under:

```text
logs/rsl_rl/g1_track_400m/<timestamp>_race_from_scratch/
```

## Safety note: self-collision is enabled

The G1 articulation now uses PhysX self-collision. Legs are therefore not
allowed to pass through one another during training. This changes the contact
dynamics, so do not continue a checkpoint that visibly used leg penetration;
start a fresh run from the compact-gait configuration instead.

The compact-gait reward set also includes compatible posture/stability terms
adapted from Unitree RL Gym and Isaac Lab: contact-foot velocity, extra ankle
soft-limit cost, and a non-foot (hip/knee) contact penalty. These complement
the existing G1 hip-neutral, contact-phase, swing-clearance, foot-slide,
joint-acceleration, and torque terms rather than duplicating them.

## Locked-elbow shoulder-swing training

The default `Template-Race-400m` task has **14 actions**: 12 leg joints plus
left/right shoulder pitch. Both elbows are held at a fixed, moderate `+0.95 rad`
(about 54 degrees) using dedicated high-stiffness, high-damping PD actuators;
shoulder roll/yaw and the wrists are also held at their default posture. The arm
reward is contact-synchronized: when one foot swings, the opposite elbow is
rewarded for forward motion and the same-side elbow for backward motion. It
uses elbow-link velocity in the pelvis yaw frame, so it does not rely on an
assumed shoulder-joint sign. A small shoulder-pitch deviation cost prevents
flailing.

This task must train **from scratch**: the previous arm-swing checkpoint has
16 actions / 86 observations, while this task has 14 actions / 84 observations.
It also includes a rear-swing-foot penalty, which limits an airborne ankle to
0.16 m behind the pelvis and reduces exaggerated heel kick without suppressing
normal forward swing.

```bash
torchrun --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \
  --task Template-Race-400m --headless --distributed \
  --num_envs 4096 --max_iterations 8000 --run_name locked_elbow_arm_from_scratch
```

## Recommended recovery path: leg-only race task

If the arm-swing policy has reduced lap-completion reliability, use the
separate `Template-Race-400m-LegOnly` task. It restores the original
**12 leg actions / 82 policy observations** interface, leaves the arms at the
configured bent default posture, and disables every arm-control reward. It
retains the ordered 200-point race logic, self-collision, low-center-of-mass,
compact-stride, low-air-time, toe-heading, and rear-swing-foot constraints.

Do **not** resume a 16-action arm-swing checkpoint in this task. Resume the
best earlier 12-action checkpoint that completed the course:

```bash
torchrun --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \
  --task Template-Race-400m-LegOnly --headless --distributed \
  --num_envs 4096 --max_iterations 1500 --run_name leg_only_recovery \
  --resume --load_run 'YOUR_12_ACTION_COMPLETION_RUN' \
  --checkpoint 'model_.*.pt'
```

For a clean comparison, keep the old checkpoint untouched and use a new
`--run_name`. If it completes consistently after 500--1000 iterations,
continue to 1500; if completion falls, stop and compare against the unchanged
baseline rather than continuing the degraded model.

## Faster leg-only gait: high-cadence fine-tuning

`Template-Race-400m-LegOnly-HighCadence` keeps the same **12-action / 82-observation**
policy interface as the completed leg-only model, so resume the best completion
checkpoint rather than starting over. It raises cadence moderately from a
0.55 s to a 0.48 s gait period, rewards forward track speed more strongly, and
uses a shorter 0.16 s swing-time limit with 5.5 cm swing clearance. It does
not increase stride reach, so the intended speed gain is from quicker leg
cycling rather than hopping or exaggerated rear kick.

```bash
torchrun --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \
  --task Template-Race-400m-LegOnly-HighCadence --headless --distributed \
  --num_envs 4096 --max_iterations 1000 --run_name leg_only_high_cadence \
  --resume --load_run 'YOUR_COMPLETED_LEG_ONLY_RUN' \
  --checkpoint 'model_.*.pt'
```

First inspect the checkpoint at 300 iterations. Continue only if completion
does not fall and the average track-forward speed increases; otherwise keep
the original completion model as the deployment baseline.

## Smooth start and controlled stop

`Template-Race-400m-LegOnly-StartStop` starts from the normal default standing
pose; it does not use a separate crouch or launch pose. The policy receives one
extra desired-speed observation: it ramps from a small positive speed across
the first 20 m and holds a `1.8 m/s` cruise command through the exact finish
line. Only after crossing that line does it smoothly brake for `3 s` along the
exit direction. Passing the final point is not sufficient: success requires an
upright, planar-speed-below-`0.18 m/s` stand for `1.5 s`.

The extra phase observation changes the interface from **82 to 83
observations**, so do not resume a prior 12-action / 82-observation policy.
Train this task from scratch:

```bash
torchrun --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \
  --task Template-Race-400m-LegOnly-StartStop --headless --distributed \
  --num_envs 4096 --max_iterations 8000 --run_name leg_only_start_stop
```

Before considering deployment, verify in playback that the robot begins from
the default stand without a lunge, decelerates before the red finish marker,
and remains upright for the full hold interval.

## Recommended: gait-quality fine-tuning

If you already have a 12-action checkpoint that completes the lap, fine-tune
the `Template-Race-400m-LegOnly` task instead of training from scratch. This
task adds official-G1-style gait shaping:
hip yaw/roll deviation, joint acceleration/torque penalties, biped air-time,
and foot-slide penalties. It also adapts the phase-based alternating-contact
and swing-foot-clearance terms used by Unitree RL Lab and Unitree RL Mjlab,
then adds a yaw-frame crossed-feet penalty. The 200-point logic, policy
observation size (82), and 12-leg-joint action space are unchanged.

First list prior runs:

```bash
ls -lt logs/rsl_rl/g1_track_400m/
```

Assuming the best run is named
`2026-08-14_12-00-00_race_baseline`, run:

```bash
torchrun --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \
  --task Template-Race-400m-LegOnly --headless --distributed \
  --num_envs 4096 --max_iterations 1500 --run_name gait_finetune \
  --resume --load_run '2026-08-14_12-00-00_race_baseline' \
  --checkpoint 'model_.*.pt'
```

Replace `2026-08-14_12-00-00_race_baseline` with your actual directory name.
The checkpoint expression selects the latest `model_*.pt` in that directory.
Start with 1500 iterations, inspect the result, and keep the original
checkpoint unchanged as the rollback baseline.

## What to monitor

Use TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/g1_track_400m --bind_all
```

## Visualize a checkpoint with the 200-point track

For a single-environment evaluation, `--track-markers` draws the actual
ordered track points without changing the training scene or physics:
blue spheres are ordinary checkpoints, green is the start, red is the finish,
and the moving yellow sphere is the target currently used by the reward.
The terminal also prints simulation time, target index, and planar speed once
per second of simulation time.

```bash
python scripts/rsl_rl/play.py \
  --task Template-Race-400m-LegOnly-HighCadence \
  --num_envs 1 --track-markers --real-time \
  --checkpoint /absolute/path/to/model.pt
```

Key metrics:

- `Episode_Reward/reached_checkpoint`: should remain strong; a collapse means
  gait regularization is interfering with navigation.
- `Episode_Termination/completed`: completion frequency; this is the primary
  success measure.
- `Episode_Termination/robot_fallen`: should decrease during gait fine-tuning.
- `Episode_Reward/feet_slide`, `hip_deviation`, and `action_rate`: use them to
  compare gait quality across checkpoints, not as stand-alone success metrics.
- `Episode_Reward/alternating_gait`, `swing_clearance`, and `crossed_feet`:
  new gait-quality terms. They should improve without a material reduction in
  checkpoint rewards or full-lap completion.
- `Episode_Reward/course_heading`, `forward_course_speed`, and
  `lateral_velocity`: the primary anti-crab-walk terms. Heading and forward
  speed should rise while lateral velocity falls.
- `Episode_Reward/swing_forward` and `forward_landing`: verify that airborne
  feet swing and land ahead of the pelvis in the robot's yaw frame.
- `Episode_Reward/base_height`: the target is now 0.70 m (previously 0.74 m)
  so the policy learns a modest knee bend and a lower moving center of mass.
- `Episode_Reward/excess_swing_time` and `compact_stride`: compact-gait
  constraints. Air time beyond 0.24 s and touchdown more than 0.22 m ahead of
  the pelvis are penalized. The gait schedule is 0.65 s and swing clearance
  target is 0.06 m, reducing hopping and vertical excursion.

## References used for gait shaping

- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab):
  phase-based left/right contact schedule and foot-clearance shaping.
- [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym): G1
  12-leg-joint control scale and conservative gait/stability regularization.
- [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab):
  contact-gated swing-foot clearance and tighter hip roll/yaw treatment.

Only compatible reward concepts were adapted. No external policy checkpoint,
29-DoF action convention, or MuJoCo-specific runtime code is used.

## Safety and scope

This is a simulation training project. The current navigation method relies on
preconfigured ordered coordinates and simulator state; it is not visual or
LiDAR self-navigation. Test every new checkpoint in simulation before any
sim2sim or physical-robot work.

## Checkpoint evaluation (no training)

`scripts/rsl_rl/evaluate_race.py` evaluates an existing checkpoint in headless
mode without training, exporting a policy, recording video, or mutating any
reward / task / checkpoint file.  It writes `episodes.csv`, `summary.json` and
`report.md` into `--output_dir`.

The two checkpoints use different action/observation interfaces and must be
evaluated with their own matching `--task`; never cross-load them:

| checkpoint | task | action / observation |
|---|---|---|
| `leg_only_high_cadence_model.pt` | `Template-Race-400m-LegOnly-HighCadence` | 12 / 82 |
| `locked_elbow_arm_model.pt` | `Template-Race-400m` | 14 / 84 |

```bash
python scripts/rsl_rl/evaluate_race.py \
  --task Template-Race-400m-LegOnly-HighCadence \
  --checkpoint /abs/path/leg_only_high_cadence_model.pt \
  --num_episodes 100 --num_envs 64 --headless \
  --output_dir outputs/eval/leg_only_high_cadence

python scripts/rsl_rl/evaluate_race.py \
  --task Template-Race-400m \
  --checkpoint /abs/path/locked_elbow_arm_model.pt \
  --num_episodes 100 --num_envs 64 --headless \
  --output_dir outputs/eval/locked_elbow_arm
```

Compare the two summaries:

```bash
python scripts/rsl_rl/compare_summaries.py \
  --baseline outputs/eval/leg_only_high_cadence/summary.json \
  --candidate outputs/eval/locked_elbow_arm/summary.json \
  --output outputs/eval/comparison.md
```

Comparison priority: completion rate > fall rate > finish time > lateral speed
/ heading error > foot spacing & crossing > action smoothness.

These runs measure repeatability and base stability only; they are **not** a
sim2real safety certificate.

### Two-GPU evaluation of the selected models

The launcher below starts two independent headless evaluation processes: GPU 0
for `leg_only_high_cadence_model.pt`, GPU 1 for `locked_elbow_arm_model.pt`.
It is intentionally not a distributed PPO job, so do not add `torchrun` or
`--distributed`.

```bash
chmod +x scripts/rsl_rl/evaluate_selected_models_two_gpu.sh
scripts/rsl_rl/evaluate_selected_models_two_gpu.sh \
  /absolute/path/leg_only_high_cadence_model.pt \
  /absolute/path/locked_elbow_arm_model.pt
```

## Locked-elbow start, run, and stop training (eight GPUs)

`Template-Race-400m-LockedElbow-StartStop` is a new policy, not a resume of
the existing locked-elbow checkpoint. It retains the **14 actions** (12 legs
plus two shoulder-pitch joints) and adds the phase-speed observation used to
learn a smooth default-stand start, a 3.6 m/s cruise, finish-line crossing,
three-second brake, and stable stand. Its interface is therefore **85
observations / 14 actions** and it must start from scratch.

For eight 72-GB GPUs, use 4,096 environments per GPU. The task uses 12 rollout
steps per environment, giving 393,216 transitions per PPO update across eight
processes. This matches the former six-GPU global batch rather than making each
PPO update 33% larger, while retaining high PhysX GPU occupancy.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

torchrun --standalone --nproc_per_node=8 scripts/rsl_rl/train.py \
  --task Template-Race-400m-LockedElbow-StartStop --headless --distributed \
  --num_envs 4096 --max_iterations 10000 \
  --run_name locked_elbow_start_stop_8gpu
```

Run a short launch check before committing a long job:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \
  --task Template-Race-400m-LockedElbow-StartStop --headless --distributed \
  --num_envs 256 --max_iterations 2 --run_name locked_elbow_start_stop_smoke
```

Do not load `locked_elbow_arm_model.pt` with this task: its 84-observation
interface is intentionally incompatible. Keep it unchanged as the deployed
main-model candidate while this new start/stop candidate trains.

It evaluates 100 episodes with 64 parallel environments on each GPU by
default. To override those values and choose an output directory:

```bash
NUM_EPISODES=200 NUM_ENVS=256 \
scripts/rsl_rl/evaluate_selected_models_two_gpu.sh \
  /absolute/path/leg_only_high_cadence_model.pt \
  /absolute/path/locked_elbow_arm_model.pt \
  outputs/eval/selected_models
```

Each model gets its own report and console log below the specified directory;
run `compare_summaries.py` on the two `summary.json` files afterward.

### Isaac Lab robustness suite

Use the same two-GPU launcher with `ROBUSTNESS_SUITE=moderate` to compare the
policies outside their nominal training conditions. This is evaluation-only:
it does not train, fine-tune, or alter either checkpoint. Each parallel
environment samples robot contact material, mass/inertia, center of mass, and
PD gains at startup; every episode also samples a small initial position,
orientation, and velocity error.

```bash
ROBUSTNESS_SUITE=moderate NUM_EPISODES=256 NUM_ENVS=128 \
scripts/rsl_rl/evaluate_selected_models_two_gpu.sh \
  /absolute/path/leg_only_high_cadence_model.pt \
  /absolute/path/locked_elbow_arm_model.pt \
  outputs/eval/moderate_robustness
```

The launcher defaults to physical GPUs 0 and 1. On a shared server, select
different cards explicitly; for example, use GPUs 6 and 7:

```bash
EVAL_GPU_LEG=6 EVAL_GPU_ARM=7 ROBUSTNESS_SUITE=moderate \
scripts/rsl_rl/evaluate_selected_models_two_gpu.sh \
  /absolute/path/leg_only_high_cadence_model.pt \
  /absolute/path/locked_elbow_arm_model.pt \
  outputs/eval/moderate_robustness
```

`strong` widens all of those ranges and should be treated as a stress test,
not an estimate of expected real-world performance. Compare `completion_rate`,
`fall_rate`, and `mean_max_progress_m` before comparing speed or gait style.

### Deployment-gap robustness: delay, noise, and odometry drift

Physics randomization alone is not sim2sim and does not test the largest
remaining risk of this no-vision waypoint strategy: a real policy receives
delayed, noisy state estimates rather than simulator ground truth. Add
`REALITY_GAP_SUITE=moderate` or `strong` to inject, only into the policy input,
action delay, joint/IMU noise, waypoint noise, persistent odometry scale error,
and yaw bias. The simulator's true state remains unchanged for fair metrics.

Run the locked-elbow model through the realistic combined test first:

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/rsl_rl/evaluate_race.py \
  --task Template-Race-400m --headless \
  --checkpoint /absolute/path/locked_elbow_arm_model.pt \
  --robustness_suite=strong --reality_gap_suite=moderate \
  --num_envs 256 --num_episodes 256 --seed 42 \
  --output_dir outputs/eval/locked_elbow_physics_strong_gap_moderate_seed42
```

Repeat with seeds `43` and `44`. Then use `--reality_gap_suite=strong` as a
stress boundary. These are still Isaac Lab evaluations. Strict sim2sim comes
next: run the same checkpoint through an independently implemented MuJoCo G1
adapter with exactly the same observation order, action scaling, waypoint
interface, and test cases.

If a combined deployment-gap run fails, do not infer its cause from the
aggregate result. Run the five moderate single-factor ablations first:

```bash
for component in action_delay joint_state base_state waypoint odometry; do
  CUDA_VISIBLE_DEVICES=6 python scripts/rsl_rl/evaluate_race.py \
    --task Template-Race-400m --headless \
    --checkpoint /absolute/path/locked_elbow_arm_model.pt \
    --robustness_suite=nominal --reality_gap_suite=moderate \
    --reality_gap_components "$component" \
    --num_envs 256 --num_episodes 256 --seed 42 \
    --output_dir "outputs/eval/locked_elbow_gap_ablation_${component}"
done
```

Omit `--reality_gap_components` (or pass `all`) for the existing combined
suite. The five components are: action delay, joint position/velocity noise,
base velocity/projected-gravity noise, waypoint noise, and persistent
odometry scale/yaw error.
