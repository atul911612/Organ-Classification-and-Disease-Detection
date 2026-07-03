"""Utilities: seeding, metrics, checkpoints, logging, early stopping."""
import os
import csv
import json
import random
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Running average
# --------------------------------------------------------------------------- #
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(targets, preds, probs=None, num_classes=None):
    """Return a dict of classification metrics.

    targets, preds: 1-D int arrays. probs: (N, C) softmax array (optional).
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, roc_auc_score,
    )
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    m = {
        "accuracy": float(accuracy_score(targets, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(targets, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, preds, average="macro", zero_division=0)),
    }
    # ROC-AUC (binary or macro one-vs-rest) when probabilities are available
    if probs is not None:
        probs = np.asarray(probs)
        try:
            if num_classes == 2 or probs.shape[1] == 2:
                m["roc_auc"] = float(roc_auc_score(targets, probs[:, 1]))
            else:
                m["roc_auc_ovr_macro"] = float(
                    roc_auc_score(targets, probs, multi_class="ovr", average="macro")
                )
        except ValueError:
            pass  # a class may be missing from this split
    return m


def classification_report_str(targets, preds, class_names):
    from sklearn.metrics import classification_report
    return classification_report(
        targets, preds, target_names=class_names, digits=4, zero_division=0
    )


def save_confusion_matrix(targets, preds, class_names, out_path):
    """Save a confusion-matrix figure. No-op (prints) if matplotlib is missing."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed - skipping confusion-matrix figure")
        return cm
    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 2,
                                    1.2 * len(class_names) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok] saved confusion matrix -> {out_path}")
    return cm


# --------------------------------------------------------------------------- #
# Class imbalance
# --------------------------------------------------------------------------- #
def class_weights_from_samples(samples, num_classes):
    """Inverse-frequency class weights (normalized) as a float32 numpy array."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, label in samples:
        counts[label] += 1
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    return (weights / weights.mean()).astype(np.float32)


# --------------------------------------------------------------------------- #
# Checkpoint / logging
# --------------------------------------------------------------------------- #
def save_checkpoint(state: dict, out_path):
    import torch
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)


def save_json(obj, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)


class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)


class EarlyStopper:
    """Stop when the monitored metric stops improving."""
    def __init__(self, patience=8, mode="max", min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.count = 0

    def step(self, value) -> bool:
        """Return True if this is a new best value."""
        if self.best is None:
            self.best = value
            return True
        improved = (value > self.best + self.min_delta) if self.mode == "max" \
            else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.count = 0
            return True
        self.count += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.count >= self.patience
