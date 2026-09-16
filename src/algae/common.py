import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score


def load_config(path: str | None = None) -> dict:
    path = path or (sys.argv[1] if len(sys.argv) > 1 else "config.toml")
    with open(path, "rb") as f:
        return tomllib.load(f)


def device() -> torch.device:
    if torch.cuda.is_available():
        # 5090 = sm_120；wheel 沒編到就會在第一個 kernel 爆掉，提早檢查
        cap = torch.cuda.get_device_capability()
        arch = f"sm_{cap[0]}{cap[1]}"
        if arch not in torch.cuda.get_arch_list():
            raise RuntimeError(f"{arch} not in torch arch list {torch.cuda.get_arch_list()} — install a cu128+ wheel")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


def load_splits(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(cfg["data"]["out"])
    classes = sorted(df["label"].unique())
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})
    return df, classes


def evaluate(logits: np.ndarray, y: np.ndarray, classes: list[str], train_counts: np.ndarray) -> dict:
    """長尾分類指標：只看 top-1 會被大類主導，科展報告要同時呈現 macro 與分組結果。"""
    pred = logits.argmax(1)
    top5 = np.argsort(-logits, 1)[:, :5]
    per_class_acc = np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(len(classes))])

    # many / medium / few-shot 分組（依訓練集樣本數，門檻沿用長尾文獻慣例）
    groups = {"many(>100)": train_counts > 100, "medium(20-100)": (train_counts >= 20) & (train_counts <= 100), "few(<20)": train_counts < 20}

    # 屬層級正確率：種猜錯但屬對 = 形態相近，可作為錯誤分析素材
    genus = np.array([c.split("_")[0] for c in classes])

    return {
        "n_test": int(len(y)),
        "n_classes": len(classes),
        "top1": float((pred == y).mean()),
        "top5": float((top5 == y[:, None]).any(1).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=np.arange(len(classes)), zero_division=0)),
        "balanced_acc": float(np.nanmean(per_class_acc)),
        "genus_top1": float((genus[pred] == genus[y]).mean()),
        "group_acc": {k: (float(np.nanmean(per_class_acc[m])) if m.any() else None) for k, m in groups.items()},
    }


def confused_pairs(logits: np.ndarray, y: np.ndarray, classes: list[str], k: int = 30) -> pd.DataFrame:
    pred = logits.argmax(1)
    wrong = pd.DataFrame({"true": np.array(classes)[y], "pred": np.array(classes)[pred]})
    wrong = wrong[wrong.true != wrong.pred]
    return wrong.value_counts().head(k).rename("count").reset_index()


def save_report(run_dir: str, name: str, metrics: dict, logits: np.ndarray, y: np.ndarray, classes: list[str]) -> None:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    confused_pairs(logits, y, classes).to_csv(out / f"{name}_confused.csv", index=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
