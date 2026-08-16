"""Offline preflight: validates the one-file deployment contract."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from g1_u2_safe_runtime import RuntimeConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=Path, required=True, help="Explicit policy override for the checked-in placeholder.")
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config" / "deployment_config.json")
    a = p.parse_args()
    cfg = RuntimeConfig.from_json(a.config, policy_path=a.policy)
    if not cfg.policy_path.is_file():
        raise FileNotFoundError(cfg.policy_path)
    policy = torch.jit.load(str(cfg.policy_path), map_location="cpu").eval()
    try:
        with torch.inference_mode():
            action = policy(torch.zeros(1, 84))
    except RuntimeError as error:
        raise RuntimeError(
            "Selected policy is incompatible with this 84-observation locked-elbow deployment package. "
            "Do not use an 85-observation start/stop export here."
        ) from error
    if tuple(action.shape)!=(1,14): raise RuntimeError(f"Expected policy output [1,14], got {tuple(action.shape)}")
    print(f"PREFLIGHT PASS: obs=84, action=14, waypoints={len(cfg.waypoints_xy)}, policy={cfg.policy_path}")
if __name__=="__main__": main()
