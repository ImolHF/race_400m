# G1 U2 first-day runbook

Use only the exported `policy.pt` of the chosen main policy. Keep `dry_run=true` until the suspended tests pass.

## Before arrival

1. Generate `deploy/config/waypoints_training.json` with `generate_waypoints.py`.
2. Run `preflight.py` with the exported policy and waypoint JSON.
3. Measure the real track origin, forward X direction, robot start position and yaw. Modify a copy of the JSON only for rigid translation/rotation; preserve the 201-point order.
4. Confirm E-stop, support harness, operator roles, power and network.

## Hard safety gates

Do not progress if any condition fails: wrong left/right joint direction, command timeout above 80 ms, tilt above 0.30 rad, joint target jump above the configured limit, unexpected contact, heating, oscillation, or unreliable E-stop.

## Test order

1. Read-only: 29 joint angles/velocities, IMU quaternion, timestamps. The SDK adapter must convert each received state timestamp to local `time.monotonic()` time before handing it to the runtime; raw robot-clock time is not valid for the 80 ms timeout gate.
2. Suspended: default pose, then low-amplitude single-joint direction checks.
3. Ground: stand 60 s.
4. Protected 3–5 m walk, repeated three times.
5. Protected 5–10 m runs, log every trial.
6. Only after repeatable success, change **only** `max_target_step_rad` in `deployment_config.json` gradually from 0.03 to 0.05, 0.10, then towards 0.20–0.25 rad. The last range is needed to recover the trained gait; never jump directly there.
7. At each change, repeat suspended and protected short tests. Only after repeatable success: 10–20 m; do not attempt 400 m on first day.

## Required log fields

Timestamp, q[29], dq[29], IMU quaternion, body angular velocity, policy obs[84], action[14], q_target[29], target index, XY estimate, E-stop/fault reason.
