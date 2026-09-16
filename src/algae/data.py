import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T


class PadToCanvas:
    """Center pad/crop 到固定像素畫布；之後統一縮放 → 所有影像縮放倍率相同，保留物種的相對尺寸資訊。"""

    def __init__(self, px: int, fill: int = 0):
        self.px, self.fill = px, fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        canvas = Image.new(img.mode, (self.px, self.px), (self.fill,) * len(img.getbands()))
        # 大於畫布的部分從中心裁掉
        left, top = max((w - self.px) // 2, 0), max((h - self.px) // 2, 0)
        img = img.crop((left, top, left + min(w, self.px), top + min(h, self.px)))
        canvas.paste(img, ((self.px - img.width) // 2, (self.px - img.height) // 2))
        return canvas


def build_transforms(cfg: dict, mean, std, train: bool):
    im = cfg["image"]
    geo = [PadToCanvas(im["canvas_px"]), T.Resize(im["size"])] if im["resize_mode"] == "canvas" else [T.Resize((im["size"], im["size"]))]
    aug = [
        # 顯微影像沒有「上下左右」：旋轉/翻轉不改變類別
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(180)], p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),  # 模擬不同顯微鏡曝光
    ] if train else []
    return T.Compose([
        *geo,
        *aug,
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean, std),
    ])


class AlgaeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths, self.ys, self.tf = df["path"].tolist(), df["y"].to_numpy(), transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        # UDE 為 8-bit 灰階；轉 RGB 以沿用 ImageNet/CLIP 預訓練權重
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), int(self.ys[i])
