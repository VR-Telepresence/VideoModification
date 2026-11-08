import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", default="yolo26n.pt")
p.add_argument("--imgsz", type=int, default=256)
args = p.parse_args()

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available.")

model = YOLO(args.weights)
out = Path(model.export(format="engine", half=True, imgsz=args.imgsz, batch=1,
                        device=0, simplify=True, dynamic=False))

w = Path(args.weights)
target = w.with_name(f"{w.stem}_{args.imgsz}.engine")
shutil.move(str(out), str(target))
print(f"Engine saved to {target}")