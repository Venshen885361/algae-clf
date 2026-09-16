"""產生假的長尾 ImageFolder 資料集，用於在沒下載真資料前 smoke test 整條 pipeline。"""
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
random.seed(0)
# 長尾：前幾類很多、後面很少
species = [f"Genus{g}_sp{s}" for g in range(4) for s in range(3)]
for rank, sp in enumerate(species):
    n = max(12, int(150 / (rank + 1)))
    d = root / sp
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        w, h = random.randint(150, 600), random.randint(100, 400)
        img = Image.new("L", (w, h), 30)
        dr = ImageDraw.Draw(img)
        # 每類不同的形狀 + 大小 → 模型要學得起來
        k = rank % 3
        bbox = (w * 0.2, h * 0.2, w * 0.8, h * 0.8)
        (dr.ellipse if k == 0 else dr.rectangle if k == 1 else dr.chord)(bbox, fill=80 + rank * 12, **({"start": 0, "end": 200} if k == 2 else {}))
        img.save(d / f"{i:04d}.png")
print(f"fake dataset → {root}")
