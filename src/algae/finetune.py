"""End-to-end fine-tune：backbone 小 lr、head 大 lr，bf16 autocast，val macro-F1 選 best checkpoint。"""
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from algae.backbones import Classifier, load_backbone
from algae.common import device, evaluate, load_config, load_splits, save_report
from algae.data import AlgaeDataset, build_transforms


@torch.inference_mode()
def predict(model, dl, dev, bf16):
    model.eval()
    out, ys = [], []
    for x, y in dl:
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
            out.append(model(x.to(dev, non_blocking=True)).float().cpu())
        ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def main() -> None:
    cfg = load_config()
    t = cfg["train"]
    dev = device()
    bf16 = t["bf16"] and dev.type == "cuda"
    torch.manual_seed(cfg["data"]["seed"])

    df, classes = load_splits(cfg)
    bb = load_backbone(cfg["model"]["backbone"], cfg["model"]["pretrained"])
    model = Classifier(bb, len(classes)).to(dev)

    def loader(split, train):
        ds = AlgaeDataset(df[df.split == split], build_transforms(cfg, bb.mean, bb.std, train))
        return DataLoader(ds, batch_size=t["batch_size"], shuffle=train, drop_last=train,
                          num_workers=t["num_workers"], pin_memory=dev.type == "cuda", persistent_workers=t["num_workers"] > 0)

    dl_tr, dl_va, dl_te = loader("train", True), loader("val", False), loader("test", False)
    counts = np.bincount(df.loc[df.split == "train", "y"], minlength=len(classes))
    prior_log = torch.log(torch.tensor(counts + 1, device=dev, dtype=torch.float32) / counts.sum())

    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": t["lr_backbone"]},
        {"params": model.head.parameters(), "lr": t["lr_head"]},
    ], weight_decay=t["weight_decay"])
    total = t["epochs"] * len(dl_tr)
    warmup = max(len(dl_tr) // 2, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    fwd = torch.compile(model) if t["compile"] else model
    run = Path(t["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    best = -1.0

    for ep in range(t["epochs"]):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in tqdm(dl_tr, desc=f"ep{ep}", leave=False):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
                logits = fwd(x)
            loss = F.cross_entropy(logits.float() + t["logit_adjust_tau"] * prior_log, y, label_smoothing=t["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()

        lv, yv = predict(model, dl_va, dev, bf16)
        mv = evaluate(lv, yv, classes, counts)
        print(f"ep{ep} loss={running / len(dl_tr):.4f} val top1={mv['top1']:.4f} macroF1={mv['macro_f1']:.4f} ({time.time() - t0:.0f}s)")
        if mv["macro_f1"] > best:
            best = mv["macro_f1"]
            torch.save({"model": model.state_dict(), "classes": classes, "cfg": cfg}, run / "best.pt")

    model.load_state_dict(torch.load(run / "best.pt", weights_only=False)["model"])
    lt, yt = predict(model, dl_te, dev, bf16)
    save_report(str(run), "finetune", evaluate(lt, yt, classes, counts), lt, yt, classes)


if __name__ == "__main__":
    main()
