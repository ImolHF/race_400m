#!/usr/bin/env python3
"""Read-only G1 live shadow runner: no publisher, LowCmd, SDK client, or control switch."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import torch

N_JOINTS, N_OBS, N_ACTIONS, DT = 29, 84, 14, 0.04


def yaw_xyzw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def gravity_wxyz(q):
    w, x, y, z = q
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-6:
        raise ValueError("zero IMU quaternion")
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    return [-2*(x*z-y*w), -2*(y*z+x*w), -(1-2*(x*x+y*y))]


class OdomReader(Node):
    """Subscriber only; no ROS publishers are constructed."""
    def __init__(self, topic):
        super().__init__("g1_policy_shadow_read_only")
        self.latest = None
        self.create_subscription(Odometry, topic, self.callback, 10)

    def callback(self, msg):
        p, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
        self.latest = dict(receipt=time.monotonic(), x=float(p.x), y=float(p.y),
                           yaw=yaw_xyzw(q.x, q.y, q.z, q.w), vx=float(v.x), vy=float(v.y), vz=float(v.z))


def snapshot(path):
    values = [float(v) for v in path.read_text().strip().split(",")]
    if len(values) != 67:
        raise ValueError(f"snapshot needs 67 values, got {len(values)}")
    return dict(stamp=values[0], q=values[1:30], dq=values[30:59], quat=values[59:63], gyro=values[63:66], mode=values[66])


def spin_until_state(node, path):
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.02)
        try:
            state = snapshot(path)
            if node.latest is not None and time.time() - state["stamp"] <= 0.15:
                return state
        except (OSError, ValueError):
            pass


def capture(node, path, output, samples):
    print("CALIBRATION: robot must be stably standing at physical start and facing race forward.")
    qs = [0.0] * N_JOINTS
    xs = ys = ss = cs = 0.0
    for _ in range(samples):
        state = spin_until_state(node, path)
        for i, value in enumerate(state["q"]): qs[i] += value
        xs += node.latest["x"]; ys += node.latest["y"]
        ss += math.sin(node.latest["yaw"]); cs += math.cos(node.latest["yaw"])
        time.sleep(0.02)
    output.write_text(json.dumps({
        "format": "g1_race_shadow_calibration_v1", "samples": samples,
        "joint_reference_q": [v/samples for v in qs],
        "origin_odom_xy": [xs/samples, ys/samples],
        "origin_yaw_rad": math.atan2(ss, cs),
        "note": "Read-only standing calibration; no robot command sent.",
    }, indent=2))
    print(f"CALIBRATION PASS: {output}")


def load_calibration(path):
    value = json.loads(path.read_text())
    if len(value.get("joint_reference_q", [])) != N_JOINTS or len(value.get("origin_odom_xy", [])) != 2:
        raise ValueError("calibration must contain 29 joint references and origin XY")
    return value


def run(node, args):
    cal = load_calibration(args.calibration)
    qref = [float(v) for v in cal["joint_reference_q"]]
    ox, oy = [float(v) for v in cal["origin_odom_xy"]]
    oyaw = float(cal["origin_yaw_rad"])
    waypoints = json.loads(args.waypoints.read_text())
    if len(waypoints) != 201:
        raise ValueError("expected 201 waypoints")
    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(selected_device if args.device == "auto" else args.device)
    policy = torch.jit.load(str(args.policy), map_location=device).eval()
    previous, target = [0.0]*N_ACTIONS, 1
    args.log.parent.mkdir(parents=True, exist_ok=True)
    header = ["wall_s","snapshot_age_s","odom_age_s","mode_machine","target_index","track_x","track_y","target_dx_body","target_dy_body","target_distance_m","obs_min","obs_max","action_min","action_max"] + [f"action_{i}" for i in range(N_ACTIONS)]
    with args.log.open("w", newline="") as log:
        writer = csv.writer(log); writer.writerow(header)
        print("SHADOW RUN: reads only; hardware_commands=0 by design.")
        done, next_tick = 0, time.monotonic()
        while args.steps <= 0 or done < args.steps:
            next_tick += DT; rclpy.spin_once(node, timeout_sec=0.0)
            try:
                state = snapshot(args.snapshot)
                sa, oa = time.time()-state["stamp"], time.monotonic()-node.latest["receipt"]
                if node.latest is None or sa > .15 or oa > .15: raise RuntimeError(f"stale state snapshot={sa:.3f}, odom={oa:.3f}")
                dxw, dyw = node.latest["x"]-ox, node.latest["y"]-oy
                c, s = math.cos(oyaw), math.sin(oyaw)
                tx, ty = c*dxw+s*dyw, -s*dxw+c*dyw
                wx, wy = float(waypoints[target][0])-tx, float(waypoints[target][1])-ty
                ry = node.latest["yaw"]-oyaw; c, s = math.cos(ry), math.sin(ry)
                bx, by, distance = c*wx+s*wy, -s*wx+c*wy, math.hypot(wx, wy)
                if distance < 1.0 and target < len(waypoints)-1: target += 1
                obs = ([a-b for a,b in zip(state["q"],qref)] + state["dq"] +
                       [node.latest["vx"],node.latest["vy"],node.latest["vz"]] +
                       [.25*v for v in state["gyro"]] + gravity_wxyz(state["quat"]) + previous + [.25*bx,.25*by,.1*distance])
                if len(obs) != N_OBS or not all(math.isfinite(v) for v in obs): raise RuntimeError("invalid 84D observation")
                with torch.inference_mode(): action = policy(torch.tensor([obs],dtype=torch.float32,device=device)).squeeze(0).cpu().tolist()
                action = [float(v) for v in action]
                if len(action) != N_ACTIONS or not all(math.isfinite(v) for v in action): raise RuntimeError("invalid policy action")
                previous = action
                writer.writerow([time.time(),sa,oa,state["mode"],target,tx,ty,bx,by,distance,min(obs),max(obs),min(action),max(action),*action]); log.flush()
                done += 1
                if done % 25 == 0: print(f"shadow_steps={done} target={target} dist={distance:.2f} action=[{min(action):.3f},{max(action):.3f}]")
            except (OSError, ValueError, RuntimeError, TypeError) as error:
                print(f"SHADOW WAIT/FAULT: {error}"); time.sleep(.05)
            sleep = next_tick-time.monotonic()
            if sleep > 0: time.sleep(sleep)
    print(f"SHADOW PASS: rows={done}, log={args.log}, hardware_commands=0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,default=Path("/tmp/g1_shadow_lowstate.csv")); parser.add_argument("--odom-topic",default="/state_estimator/odom_pelvis")
    mode = parser.add_mutually_exclusive_group(required=True); mode.add_argument("--capture-calibration",type=Path); mode.add_argument("--calibration",type=Path)
    parser.add_argument("--samples",type=int,default=100); parser.add_argument("--policy",type=Path); parser.add_argument("--waypoints",type=Path,default=Path(__file__).parent/"config"/"waypoints_training.json"); parser.add_argument("--log",type=Path,default=Path("/tmp/g1_shadow_log.csv")); parser.add_argument("--steps",type=int,default=250); parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    args = parser.parse_args()
    if args.samples <= 0: parser.error("--samples must be positive")
    if args.calibration and args.policy is None: parser.error("--policy is required for a shadow run")
    rclpy.init(); node = OdomReader(args.odom_topic)
    try:
        capture(node,args.snapshot,args.capture_calibration,args.samples) if args.capture_calibration else run(node,args)
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__": main()
