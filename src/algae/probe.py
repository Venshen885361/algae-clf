"""Linear probe on frozen features（GPU full-batch），weight decay 用 val macro-F1 選。"""
import numpy as np
import torch
import torch.nn.functional as F

from algae.common import device, evaluate, load_config, save_report


def fit(x, y, n_cls, wd, prior_log, tau, epochs=300, lr=1e-2):
    head = torch.nn.Linear(x.shape[1], n_cls, device=x.device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        # logit adjustment（Menon et al., ICLR 2021）：訓練時加 τ·log prior，推論時不加 → 平衡長尾
        loss = F.cross_entropy(head(x) + tau * prior_log, y)
        loss.backward()
        opt.step()
        sched.step()
    return head


def main() -> None:
    cfg = load_config()
    dev = device()
    run, tau = cfg["train"]["run_dir"], cfg["train"]["logit_adjust_tau"]
    z = np.load(f"{run}/features.npz", allow_pickle=True)
    classes = z["classes"].tolist()
    n = len(classes)

    def t(k, norm=True):
        a = torch.from_numpy(z[k]).to(dev)
        return F.normalize(a.float(), dim=-1) if norm else a.long()

    xtr, ytr, xva, yva, xte = t("train_x"), t("train_y", False), t("val_x"), z["val_y"], t("test_x")
    counts = np.bincount(z["train_y"], minlength=n)
    prior_log = torch.log(torch.tensor(counts + 1, device=dev, dtype=torch.float32) / counts.sum())

    best = None
    for wd in (0.0, 1e-4, 1e-3, 1e-2):
        head = fit(xtr, ytr, n, wd, prior_log, tau)
        with torch.no_grad():
            mv = evaluate(head(xva).cpu().numpy(), yva, classes, counts)
        print(f"wd={wd:g}  val top1={mv['top1']:.4f}  macroF1={mv['macro_f1']:.4f}")
        if best is None or mv["macro_f1"] > best[0]:
            best = (mv["macro_f1"], wd, head)

    _, wd, head = best
    with torch.no_grad():
        logits = head(xte).cpu().numpy()
    m = evaluate(logits, z["test_y"], classes, counts) | {"weight_decay": wd, "logit_adjust_tau": tau}
    save_report(run, "probe", m, logits, z["test_y"], classes)
    torch.save(head.state_dict(), f"{run}/probe_head.pt")


if __name__ == "__main__":
    main()
