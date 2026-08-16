# G1 U2 deployment layer

`g1_u2_safe_runtime.py` is an offline-safe policy runtime. It builds the exact
84-value observation for the 14-action locked-elbow policy, applies a 40 ms
action FIFO, maps the output to the official 29-motor G1 order, and refuses to
send hardware commands unless a lab SDK2 adapter, explicit arming gate, and
`dry_run=False` are supplied.

Before any hardware command, verify: the firmware's 29 motor order; JIT policy
export from the selected checkpoint; IMU quaternion convention; body-frame
velocity and track-frame position estimator; E-stop; and suspended command
tests. Do not use this file to command an unsupported or unverified robot.
