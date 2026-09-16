from dataclasses import dataclass

import torch
from torch import nn

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class Backbone:
    encoder: nn.Module          # image -> embedding (B, dim)
    dim: int
    mean: tuple
    std: tuple
    clip: nn.Module | None = None   # 完整 CLIP model（zero-shot 用）
    tokenizer: object | None = None


def load_backbone(spec: str, pretrained: bool = True) -> Backbone:
    kind, name = spec.split(":", 1)
    if kind == "bioclip":
        import open_clip

        # "imageomics/bioclip-2" → HF Hub；不含 "/" 則視為 open_clip 內建架構（無權重，僅供 smoke test）
        model_id = f"hf-hub:{name}" if "/" in name else name
        model, _, _ = open_clip.create_model_and_transforms(model_id)
        tok = open_clip.get_tokenizer(model_id)
        v = model.visual
        mean = tuple(getattr(v, "image_mean", None) or CLIP_MEAN)
        std = tuple(getattr(v, "image_std", None) or CLIP_STD)
        return Backbone(v, v.output_dim, mean, std, model, tok)
    if kind == "timm":
        import timm

        m = timm.create_model(name, pretrained=pretrained, num_classes=0)
        pc = m.pretrained_cfg
        return Backbone(m, m.num_features, tuple(pc.get("mean", CLIP_MEAN)), tuple(pc.get("std", CLIP_STD)))
    raise ValueError(f"unknown backbone kind: {kind}")


class Classifier(nn.Module):
    def __init__(self, bb: Backbone, n_classes: int):
        super().__init__()
        self.encoder = bb.encoder
        self.head = nn.Linear(bb.dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
