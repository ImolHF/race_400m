"""Generate the exact 201 fixed-course waypoints used during training."""
from __future__ import annotations
import json, math
from pathlib import Path
import argparse

def points():
    out=[]; straight=110.43; radius=23.24; start=32.5; half=math.pi*radius
    for i in range(201):
        d=2.0*i
        if d<start: p=(d,0.0)
        elif d<start+half:
            t=(d-start)/radius; p=(start+radius*math.sin(t),radius*(1-math.cos(t)))
        elif d<start+half+straight: p=(start-(d-start-half),2*radius)
        elif d<start+2*half+straight:
            t=(d-start-half-straight)/radius; p=(-77.93-radius*math.sin(t),2*radius-radius*(1-math.cos(t)))
        else: p=(-77.0+d-start-2*half-straight,0.0)
        out.append([round(p[0],6),round(p[1],6)])
    return out

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path("deploy/config/waypoints_training.json")); args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(points(),indent=2)+"\n")
    print(f"wrote {len(points())} waypoints to {args.output}")
