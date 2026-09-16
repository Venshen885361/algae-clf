"""Build data/splits.csv: path,label,genus,split  (label = Genus_species)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from algae.common import load_config

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def from_imagefolder(root: Path) -> pd.DataFrame:
    rows = [{"path": str(p), "label": p.parent.name} for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    return pd.DataFrame(rows)


def from_csv(d: dict) -> pd.DataFrame:
    meta = pd.read_csv(d["csv"])
    # 只收鑑定到「種」的樣本；只到屬的樣本不能當 species label
    meta = meta.dropna(subset=[d["col_genus"], d["col_species"]])
    df = pd.DataFrame({
        "path": [str(Path(d["root"]) / p) for p in meta[d["col_path"]]],
        "label": meta[d["col_genus"]].str.strip() + "_" + meta[d["col_species"]].str.strip(),
    })
    if d.get("col_group"):
        df["group"] = meta[d["col_group"]].values
    return df


def split(df: pd.DataFrame, ratios: list[float], seed: int) -> pd.Series:
    tr, va, te = ratios
    s = pd.Series("train", index=df.index)
    if "group" in df:
        # 同一玻片/樣本只會落在同一個 split → 避免背景、染色條件洩漏造成虛高
        gss = GroupShuffleSplit(n_splits=1, test_size=te, random_state=seed)
        _, test_idx = next(gss.split(df, groups=df["group"]))
        s.iloc[test_idx] = "test"
        rest = df[s == "train"]
        gss = GroupShuffleSplit(n_splits=1, test_size=va / (tr + va), random_state=seed)
        _, val_idx = next(gss.split(rest, groups=rest["group"]))
        s.loc[rest.index[val_idx]] = "val"
        return s
    trva, test = train_test_split(df.index, test_size=te, stratify=df["label"], random_state=seed)
    train, val = train_test_split(trva, test_size=va / (tr + va), stratify=df.loc[trva, "label"], random_state=seed)
    s.loc[val], s.loc[test] = "val", "test"
    return s


def main() -> None:
    cfg = load_config()
    d = cfg["data"]
    df = from_imagefolder(Path(d["root"])) if d["mode"] == "imagefolder" else from_csv(d)

    counts = df["label"].value_counts()
    keep = counts[counts >= d["min_samples"]].index
    print(f"{len(df)} images / {len(counts)} classes → keep {df['label'].isin(keep).sum()} images / {len(keep)} classes (min_samples={d['min_samples']})")
    df = df[df["label"].isin(keep)].reset_index(drop=True)

    df["genus"] = df["label"].str.split("_").str[0]
    df["split"] = split(df, d["split"], d["seed"])

    Path(d["out"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(d["out"], index=False)
    print(df.groupby("split").agg(images=("path", "size"), classes=("label", "nunique")))
    missing = set(df.loc[df.split == "train", "label"]) ^ set(df["label"])
    if missing:
        print(f"⚠️ {len(missing)} classes have no train images (group split) — consider lowering test ratio")
    print(f"imbalance ratio (max/min per class): {counts[keep].max() / counts[keep].min():.1f}")


if __name__ == "__main__":
    np.random.seed(0)
    main()
