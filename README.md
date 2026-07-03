# Abdominal Ultrasound — Organ Classification & Disease Detection

A deep-learning project on a multi-organ **abdominal ultrasound** dataset, tackling
two tasks:

1. **Organ classification** — identify which organ a B-mode ultrasound frame shows
   (10 classes).
2. **Disease / anomaly detection** — classify a frame as **normal** vs **abnormal**.

The code fine-tunes strong ImageNet-pretrained backbones as reproducible baselines
and includes a hook to fine-tune the **USFM ultrasound foundation model** — the
current state of the art for ultrasound image analysis (see [State of the art](#state-of-the-art)).

> Environment note: this repo was authored where PyTorch was not installed, so the
> training/evaluation code is provided ready-to-run on a GPU machine (`pip install
> -r requirements.txt`). The dataset-analysis script (`analyze_dataset.py`) runs with
> only Pillow and was used to produce the real numbers below.

---

## Dataset

`US_Dateset_V01.zip` — **13,065** RGB JPG frames (~568×782), organised as an
ImageFolder layout (`<task>/<class>/*.jpg`). Measured class distribution
(`python analyze_dataset.py --zip ../US_Dateset_V01.zip`):

| Task directory | Type | Classes | Images | Max/Min imbalance |
|---|---|---|---|---|
| `organ_classification_1` | organ (multi-class) | 10 | 2,784 | 9.0× |
| `organ_classification_2` | organ (multi-class) | 10 | 1,293 | 86.8× |
| `Anomaly_detection_1` | normal/abnormal | 2 | 2,813 | 2.5× |
| `Anomaly_detection_2` | normal/abnormal | 2 | 925 | 2.4× |
| `organ_classification+anomaly_detection` | organ × normal/abnormal | 18 | 2,924 | 418× |
| `Patient_Wise/` | per-patient folders (170 patients) | — | 2,326 | — |

**Organ classes:** AA (abdominal aorta), GB (gallbladder), Hepatic, Kidney, Liver,
Ovaries, Pancreas, Portal, Spleen, UB+prostate+uterus+cervix.

**Recommended targets:** `organ_classification_1` (cleanest 10-class set) and
`Anomaly_detection_1` (binary). `organ_classification_2` and the combined 18-class
split are usable but **severely imbalanced** (some classes have 1–6 images) — treat
their rare-class metrics with caution.

---

## State of the art

For ultrasound specifically, the strongest published models in 2024–2025 are
**ViT-based ultrasound foundation models** pretrained with masked image modeling
on millions of ultrasound frames:

- **USFM** — Jiao et al., *"USFM: A Universal Ultrasound Foundation Model Generalized
  to Tasks and Organs towards Label-Efficient Image Analysis"*, *Medical Image
  Analysis*, 2024. A ViT-B/16 encoder trained with **spatial-frequency masked image
  modeling** on **>2M** ultrasound images; state of the art across ultrasound
  classification/segmentation and especially strong in the low-label regime.
- **URFM** (2025) and **USF-MAE** (2025) — follow-up ultrasound foundation models.

This project therefore:

1. Provides **transfer-learning baselines** (`ConvNeXt`, `ViT-B`, `Swin`,
   `EfficientNetV2`, `ResNet-50`) via `timm` — strong, fully reproducible, no
   external downloads. **ConvNeXt-Tiny** is the default.
2. Provides `--model usfm` to **fine-tune the USFM ViT-B encoder** once you download
   its checkpoint from the official repo (https://github.com/openmedlab/USFM) and
   pass `--usfm-weights path.pth`.

---

## Folder structure

```
Self Project/
├── README.md
├── requirements.txt
├── analyze_dataset.py          # EDA — runs on the .zip or an extracted dir
├── configs/
│   ├── organ_classification.yaml
│   └── anomaly_detection.yaml
├── src/
│   ├── datasets.py             # indexing, stratified & patient-grouped splits
│   ├── transforms.py           # ultrasound-appropriate augmentations
│   ├── models.py               # timm backbones + USFM fine-tuning hook
│   ├── engine.py               # train / eval loops (AMP, metrics)
│   ├── train.py                # training entrypoint
│   ├── evaluate.py             # test-set metrics + folder inference
│   └── utils.py                # seeding, metrics, checkpoints, early stopping
└── outputs/                    # checkpoints, logs, confusion matrices (gitignored)
```

---

## Setup

```bash
cd "Self Project"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # install a CUDA torch build for GPU

# extract the dataset once (≈1.5 GB):
unzip ../US_Dateset_V01.zip -d ../US_Dateset_V01
# -> ../US_Dateset_V01/dataset publish/<task>/<class>/*.jpg
```

## Usage

**1) Analyze the data** (no torch needed):
```bash
python analyze_dataset.py --zip ../US_Dateset_V01.zip --save-figs outputs/eda
```

**2) Smoke-test the pipeline** (tiny subset, 1 epoch, runs on CPU):
```bash
python -m src.train --smoke --task organ \
  --data "../US_Dateset_V01/dataset publish/organ_classification_1"
```

**3) Train organ classification** (10 classes):
```bash
python -m src.train --config configs/organ_classification.yaml \
  --data "../US_Dateset_V01/dataset publish/organ_classification_1"
```

**4) Train disease/anomaly detection** (binary):
```bash
python -m src.train --config configs/anomaly_detection.yaml \
  --data "../US_Dateset_V01/dataset publish/Anomaly_detection_1"
```

**5) Fine-tune the USFM foundation model:**
```bash
python -m src.train --config configs/organ_classification.yaml \
  --data "../US_Dateset_V01/dataset publish/organ_classification_1" \
  --model usfm --usfm-weights /path/to/usfm_vitb.pth
```

**6) Evaluate / run inference:**
```bash
# reproduce held-out test metrics + confusion matrix
python -m src.evaluate --checkpoint outputs/organ_convnext_tiny/best_model.pt \
  --data "../US_Dateset_V01/dataset publish/organ_classification_1"

# predict every image in a new folder -> CSV
python -m src.evaluate --checkpoint outputs/organ_convnext_tiny/best_model.pt \
  --predict-dir /path/to/new_images --out outputs/predictions.csv
```

Swap the backbone anytime with `--model {convnext_tiny,vit_base,swin_tiny,efficientnetv2_s,resnet50,usfm}`.

---

## Design choices

- **Transfer learning** from ImageNet-22k → 1k checkpoints; the classifier head is
  re-initialized for the task's class count.
- **Class imbalance** is handled with inverse-frequency class weights in the loss
  (disable with `--no-class-weights`) and **macro-F1 / balanced accuracy** as the
  primary metrics rather than raw accuracy.
- **Mild augmentation** (crop, flip, ±10° rotation, light brightness/contrast) —
  ultrasound frames are near-grayscale with fixed geometry, so heavy augmentation
  hurts.
- **Metrics:** accuracy, balanced accuracy, macro-F1/precision/recall, ROC-AUC
  (binary) or one-vs-rest macro AUC (multi-class), plus a saved confusion matrix and
  per-class classification report.
- **Reproducibility:** fixed seed, deterministic stratified split, best checkpoint by
  val macro-F1, full training log (`outputs/<run>/training_log.csv`).

### ⚠️ Patient-wise evaluation (important caveat)
The default split is **stratified at the image level**. Because consecutive frames
from the same patient are correlated, an image-level split can *overestimate*
performance. For a clinically honest estimate, use the `Patient_Wise/` folders (170
patients) with the provided `datasets.grouped_split(..., groups=patient_ids)` so no
patient appears in more than one split.

---

## Results

Fill in after training (`outputs/<run>/test_results.json` has these values):

| Task | Model | Accuracy | Balanced Acc | Macro-F1 | AUC |
|---|---|---|---|---|---|
| Organ (10-class) | ConvNeXt-Tiny | — | — | — | — |
| Organ (10-class) | USFM ViT-B | — | — | — | — |
| Anomaly (binary) | ConvNeXt-Tiny | — | — | — | — |

---

## References
- USFM: A Universal Ultrasound Foundation Model — https://arxiv.org/abs/2401.00153 · https://www.sciencedirect.com/science/article/abs/pii/S1361841524001270 · code: https://github.com/openmedlab/USFM
- URFM: A general Ultrasound Representation Foundation Model — https://www.cell.com/iscience/fulltext/S2589-0042(25)01178-2
- ConvNeXt (*A ConvNet for the 2020s*) — https://arxiv.org/abs/2201.03545
- `timm` (PyTorch Image Models) — https://github.com/huggingface/pytorch-image-models
