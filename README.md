# algae-clf — 矽藻 Species-level 影像分類（科展版）

## 0. 範圍現實檢查

「分辨所有藻類」在 species level 不可行：沒有任何公開資料集涵蓋所有藻類，且不同類群的影像型態（顯微 vs 野外照片）完全不同。
**科展主題收斂為：淡水矽藻（diatoms）種級分類 + 長尾問題**，理由是目前最大、授權最乾淨的 species-level 藻類資料集就是矽藻。

| Dataset | 影像數 | 類別 | 層級 | 型態 | 授權 | 用途 |
|---|---|---|---|---|---|---|
| [UDE Diatoms in the Wild 2024](https://zenodo.org/records/10410655) | 83,570 | 611 taxa（542 species + 69 genera）；≥100 張的僅 101 類 | Species | 光學顯微、focus stacking、灰階 PNG、0.09 µm/px | CC0 | **主資料集** |
| [LMFM-12](https://github.com/aimialina/LMFM-12) | 7,555 | 12 | Species | 明視野顯微 400×/1000× | CC-BY-4.0 | 延伸：跨類群（綠藻/藍綠菌等） |
| [WHOI-Plankton](https://github.com/hsosik/WHOI-Plankton) | 3.5M+ | 103 | ⚠️ 未確認（README 未明說層級） | IFCB 流式影像、海洋 | MIT | 延伸：海洋浮游植物 |

Foundation model：[BioCLIP 2](https://huggingface.co/imageomics/bioclip-2)（ViT-L/14，TreeOfLife-200M，~214M 張 / 952K taxa，MIT）與 [BioCLIP 2.5 Huge](https://huggingface.co/imageomics/bioclip-2.5-vith14)（ViT-H/14）。
⚠️ 模型卡未提供顯微影像的表現數據 → 這正好是 RQ1 的研究空白。

## 1. 研究問題

| RQ | 問題 | 自變數 | 看的指標 |
|---|---|---|---|
| RQ1 | 通用生物基礎模型不經訓練，認得顯微矽藻嗎？ | zero-shot BioCLIP 2 / 2.5 | top-1、top-5、genus_top1 |
| RQ2 | 要多少「適應」才夠？ | zero-shot → linear probe → full fine-tune；BioCLIP vs ImageNet ConvNeXt | macro-F1、top-1 |
| RQ3 | 長尾校正能救少樣本物種嗎？ | `logit_adjust_tau` = 0 vs 1 | few(<20) / medium / many 組別正確率 |
| RQ4 | 保留絕對尺寸有幫助嗎？（矽藻鑑定高度依賴殼長） | `resize_mode` = resize vs canvas | macro-F1、混淆對（`*_confused.csv`） |

> 為什麼主指標用 **macro-F1** 而不是 accuracy：類別極度不平衡，大類猜對就能把 top-1 撐高，macro-F1 每類權重相同。

## 2. 實驗矩陣

每列複製一份 `config.toml`，只改表內欄位與 `run_dir`，**seed 固定**，至少跑 3 個 seed 報平均 ± 標準差。

| run_dir | backbone | 方法 | resize_mode | τ |
|---|---|---|---|---|
| runs/zs-b2 | bioclip:imageomics/bioclip-2 | zeroshot | canvas | – |
| runs/zs-b25 | bioclip:imageomics/bioclip-2.5-vith14 | zeroshot | canvas | – |
| runs/lp-b2 | bioclip:imageomics/bioclip-2 | probe | canvas | 1 |
| runs/lp-b2-tau0 | bioclip:imageomics/bioclip-2 | probe | canvas | 0 |
| runs/ft-b2 | bioclip:imageomics/bioclip-2 | finetune | canvas | 1 |
| runs/ft-b2-resize | bioclip:imageomics/bioclip-2 | finetune | resize | 1 |
| runs/ft-convnext | timm:convnext_base.fb_in22k | finetune | canvas | 1 |

## 3. 環境（RTX 5090）

```bash
# uv：Python 版 Bun（CLI-first、lockfile、自管 Python）
curl -LsSf https://astral.sh/uv/install.sh | sh     # Arch 可改 pacman -S uv
uv sync

# 5090 = compute capability 12.0，arch list 必須含 sm_120，否則 wheel 不對
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_arch_list())"
```

- PyTorch ≥ 2.7 + CUDA 12.8 wheel 才原生支援 sm_120（[來源](https://docs.salad.com/container-engine/tutorials/machine-learning/pytorch-rtx5090)）；`pyproject.toml` 已用 [uv 官方寫法](https://docs.astral.sh/uv/guides/integration/pytorch/) 鎖 cu128 index。
- `common.device()` 啟動時會自動檢查 sm_120，裝錯 wheel 會直接報錯而不是跑到一半炸。
- ViT-H/14（BioCLIP 2.5）full fine-tune 若 OOM：`batch_size = 32` 或開 `compile = true`。⚠️ 32 GB 下的實際上限未實測。

## 4. 執行

```bash
# 0) 沒有真資料前先 smoke test（CPU 也能跑）
uv run python scripts/make_fake.py data/raw

# 1) 下載 UDE（~2.1 GB zip）解壓到 data/raw，確認 CSV 欄位後改 config.toml [data]
head -1 data/raw/*.csv

# 2) 建 split
uv run python -m algae.prep config.toml

# 3) 抽特徵（一次）→ zero-shot / linear probe 共用
uv run python -m algae.features config.toml
uv run python -m algae.zeroshot config.toml
uv run python -m algae.probe config.toml

# 4) full fine-tune
uv run python -m algae.finetune config.toml
```

輸出（`run_dir/`）：

| 檔案 | 內容 |
|---|---|
| `*_metrics.json` | top1 / top5 / macro_f1 / balanced_acc / genus_top1 / many-medium-few 組別正確率 |
| `*_confused.csv` | 最常混淆的 (true, pred) 前 30 對 → 錯誤分析、找形態相似種 |
| `features.npz` | 凍結特徵，可再做 t-SNE/UMAP 視覺化 |
| `best.pt` | fine-tune 最佳 checkpoint（依 val macro-F1） |

## 5. 專案結構

```
src/algae/
  common.py     config、device 檢查（sm_120）、長尾指標
  prep.py       ImageFolder / CSV → splits.csv；min_samples 過濾；stratified 或 group split
  data.py       PadToCanvas（保留尺寸）、旋轉翻轉 augmentation
  backbones.py  bioclip:* (open_clip) / timm:* 統一介面
  features.py   凍結特徵抽取
  zeroshot.py   學名 prompt zero-shot
  probe.py      GPU full-batch linear probe + logit adjustment + wd sweep
  finetune.py   差異化 lr、bf16、cosine warmup、best-by-macro-F1
scripts/make_fake.py  長尾假資料
```

## 6. 科展報告要主動寫的限制

- `min_samples` 排除了多少物種 → 模型實際「會認」的只有剩下的類別。
- UDE 只有淡水矽藻、單一顯微設定；換顯微鏡/染色會有 domain shift。
- 若 CSV 有玻片/樣本 ID，填 `col_group` 做 group split；random split 可能因同玻片背景相似而虛高（可當額外實驗對照）。⚠️ UDE CSV 是否含此欄位未確認。
- zero-shot 的 prompt 只用學名，未做 prompt engineering。

## 7. 建議時程（MVP 導向）

| 週 | 目標 |
|---|---|
| 1 | 下載 UDE、跑 prep、確認 split 與類別分佈圖 |
| 2 | RQ1 zero-shot + RQ2 linear probe（幾分鐘就跑完，先拿到 baseline 表） |
| 3–4 | RQ2 fine-tune、RQ3 τ、RQ4 resize_mode，各 3 seeds |
| 5 | 混淆對錯誤分析 + UMAP 視覺化 + 寫報告 |
| 延伸 | LMFM-12 跨資料集測試、Svelte 5 + ONNX Runtime Web 做線上 demo |
