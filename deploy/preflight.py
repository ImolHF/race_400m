"""Offline preflight: validates policy I/O and the editable waypoint file."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument("--policy",type=Path,required=True); p.add_argument("--waypoints",type=Path,required=True); a=p.parse_args()
    pts=json.loads(a.waypoints.read_text())
    if len(pts)!=201 or any(not isinstance(x,list) or len(x)!=2 for x in pts): raise ValueError("Expected exactly 201 [x,y] waypoints.")
    policy=torch.jit.load(str(a.policy),map_location="cpu").eval()
    with torch.inference_mode(): action=policy(torch.zeros(1,84))
    if tuple(action.shape)!=(1,14): raise RuntimeError(f"Expected policy output [1,14], got {tuple(action.shape)}")
    print(f"PREFLIGHT PASS: obs=84, action=14, waypoints={len(pts)}, policy={a.policy}")
if __name__=="__main__": main()
