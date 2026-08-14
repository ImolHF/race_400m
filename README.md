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

## Recommended: gait-quality fine-tuning

If you already have a checkpoint that completes the lap, fine-tune it instead
of training from scratch. The current task adds official-G1-style gait shaping:
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
  --task Template-Race-400m --headless --distributed \
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
