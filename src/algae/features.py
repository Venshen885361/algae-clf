"""Extract frozen embeddings once → run_dir/features.npz (linear probe / zero-shot 都重用)."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from algae.backbones import load_backbone
from algae.common import device, load_config, load_splits
from algae.data import AlgaeDataset, build_transforms


@torch.inference_mode()
def main() -> None:
    cfg = load_config()
    dev = device()
    df, classes = load_splits(cfg)
    bb = load_backbone(cfg["model"]["backbone"], cfg["model"]["pretrained"])
    enc = bb.encoder.to(dev).eval()
    tf = build_transforms(cfg, bb.mean, bb.std, train=False)
    t = cfg["train"]

    out = {}
    for split in ("train", "val", "test"):
        sub = df[df.split == split]
        dl = DataLoader(AlgaeDataset(sub, tf), batch_size=t["batch_size"] * 2, num_workers=t["num_workers"], pin_memory=dev.type == "cuda")
        feats = []
        for x, _ in tqdm(dl, desc=split):
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=t["bf16"] and dev.type == "cuda"):
                feats.append(enc(x.to(dev, non_blocking=True)).float().cpu())
        out[f"{split}_x"] = torch.cat(feats).numpy()
        out[f"{split}_y"] = sub["y"].to_numpy()

    run = Path(t["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    np.savez(run / "features.npz", classes=np.array(classes), **out)
    print(f"saved {run / 'features.npz'}  dim={out['train_x'].shape[1]}")


if __name__ == "__main__":
    main()
