# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare two evaluate_race.py summary.json files into a ranked Markdown table.

Usage:
    python scripts/rsl_rl/compare_summaries.py \
        --baseline outputs/eval/leg_only_high_cadence/summary.json \
        --candidate outputs/eval/locked_elbow_arm/summary.json \
        --output outputs/eval/comparison.md
"""

import argparse
import json
import os


# Metric ranking used for the priority table.  Each entry is (label, json key,
# higher_is_better).  Completion/fall/timeout are treated specially below.
PRIORITY_METRICS = [
    ("Completion rate (%)", "completion_rate_pct", True),
    ("Fall rate (%)", "fall_rate_pct", False),
    ("Mean finish time (s)", "mean_finish_time_s", False),
    ("Mean lateral speed", "mean_lateral_speed", False),
    ("Mean heading error (rad)", "mean_heading_err_rad", False),
    ("Mean min foot spacing", "mean_min_foot_spacing", True),
    ("Mean crossed-feet events", "mean_crossed_feet_events", False),
    ("Mean action rate", "mean_action_rate", False),
]

# Additional metrics reported for context but not used for ranking.
CONTEXT_METRICS = [
    "mean_planar_speed",
    "max_planar_speed",
    "mean_trunk_tilt_rad",
    "max_trunk_tilt_rad",
    "max_heading_err_rad",
    "mean_joint_vel",
    "max_joint_vel",
    "mean_air_time",
    "max_air_time",
    "mean_max_rear_swing",
    "mean_foot_slide_events",
    "mean_undesired_contact_events",
    "median_finish_time_s",
    "p90_finish_time_s",
]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="First summary.json (baseline).")
    parser.add_argument("--candidate", required=True, help="Second summary.json (candidate).")
    parser.add_argument("--output", required=True, help="Output Markdown path.")
    args = parser.parse_args()

    base = _load(args.baseline)
    cand = _load(args.candidate)

    lines = ["# Race 400m Checkpoint Comparison\n"]
    lines.append("| Metric | {} | {} |".format(
        os.path.basename(os.path.dirname(args.baseline)),
        os.path.basename(os.path.dirname(args.candidate)),
    ))
    lines.append("|---|---|---|")

    lines.append(f"| task | `{base.get('task')}` | `{cand.get('task')}` |")
    lines.append(f"| action dim | {base.get('action_dim')} | {cand.get('action_dim')} |")
    lines.append(f"| observation dim | {base.get('observation_dim')} | {cand.get('observation_dim')} |")
    lines.append(f"| num episodes | {base.get('num_episodes')} | {cand.get('num_episodes')} |")

    lines.append("\n## Priority metrics (ranked)\n")
    lines.append("| Priority | Metric | {} | {} | Winner |".format(
        os.path.basename(os.path.dirname(args.baseline)),
        os.path.basename(os.path.dirname(args.candidate)),
    ))
    lines.append("|---|---|---|---|---|")

    for i, (label, key, higher) in enumerate(PRIORITY_METRICS, 1):
        b = base.get(key)
        c = cand.get(key)
        if b is None or c is None:
            winner = "N/A"
        elif key in ("mean_finish_time_s", "median_finish_time_s"):
            # lower is better, but only meaningful when both completed
            if b is None or c is None:
                winner = "N/A"
            else:
                winner = "baseline" if b <= c else "candidate"
        elif higher:
            winner = "baseline" if b >= c else "candidate"
        else:
            winner = "baseline" if b <= c else "candidate"
        lines.append(f"| {i} | {label} | {_fmt(b)} | {_fmt(c)} | {winner} |")

    lines.append("\n## Context metrics\n")
    lines.append("| Metric | {} | {} |".format(
        os.path.basename(os.path.dirname(args.baseline)),
        os.path.basename(os.path.dirname(args.candidate)),
    ))
    lines.append("|---|---|---|")
    for key in CONTEXT_METRICS:
        lines.append(f"| {key} | {_fmt(base.get(key))} | {_fmt(cand.get(key))} |")

    lines.append("\n> Priority order: completion rate > fall rate > finish time > "
                "lateral speed / heading error > foot spacing & crossing > action smoothness.\n")
    lines.append("> These runs measure repeatability and base stability only; "
                "they are **not** a sim2real safety certificate.\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Wrote {args.output}")


if __name__ == "__main__":
    main()
