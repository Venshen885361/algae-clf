"""BioCLIP zero-shot baseline：完全不用訓練資料，測「通用生物模型對顯微藻類認得多少」。"""
import numpy as np
import torch
import torch.nn.functional as F

from algae.backbones import load_backbone
from algae.common import device, evaluate, load_config, save_report


@torch.inference_mode()
def main() -> None:
    cfg = load_config()
    if not cfg["model"]["backbone"].startswith("bioclip:"):
        raise SystemExit("zero-shot 需要 CLIP 類 backbone（bioclip:*）")
    dev = device()
    run = cfg["train"]["run_dir"]
    z = np.load(f"{run}/features.npz", allow_pickle=True)
    classes = z["classes"].tolist()

    bb = load_backbone(cfg["model"]["backbone"])
    clip = bb.clip.to(dev).eval()
    # BioCLIP 以學名訓練；Genus_species → "a photo of Genus species."
    prompts = [f"a photo of {c.replace('_', ' ')}." for c in classes]
    txt = []
    for i in range(0, len(prompts), 256):
        txt.append(clip.encode_text(bb.tokenizer(prompts[i:i + 256]).to(dev)).float())
    txt = F.normalize(torch.cat(txt), dim=-1)

    img = F.normalize(torch.from_numpy(z["test_x"]).to(dev), dim=-1)
    logits = (100.0 * img @ txt.T).cpu().numpy()

    train_counts = np.bincount(z["train_y"], minlength=len(classes))
    m = evaluate(logits, z["test_y"], classes, train_counts)
    save_report(run, "zeroshot", m, logits, z["test_y"], classes)


if __name__ == "__main__":
    main()
