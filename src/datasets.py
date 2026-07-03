"""Dataset indexing and train/val/test splitting.

The dataset is organised as ``<task_root>/<class_name>/*.jpg`` (an ImageFolder
layout). Some tasks nest an extra level (e.g. the combined organ+anomaly split
uses ``normal/<organ>`` and ``abnormal/<organ>``); ``rglob`` handles that by
collecting every image under each top-level class directory.

Splitting:
  * default  -> stratified image-level split (fixed seed).
  * grouped  -> GroupShuffleSplit on a per-image ``group`` id (e.g. patient),
                which prevents images from the same patient leaking across
                train/val/test. Use this whenever a patient id is available.
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
Sample = Tuple[str, int]


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def build_index(root) -> Tuple[List[Sample], List[str]]:
    """Scan ``root`` for class subfolders and return (samples, class_names)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root not found: {root}")
    class_names = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not class_names:
        raise ValueError(f"no class subfolders found under {root}")
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    samples: List[Sample] = []
    for c in class_names:
        for p in sorted((root / c).rglob("*")):
            if p.is_file() and _is_image(p):
                samples.append((str(p), class_to_idx[c]))
    if not samples:
        raise ValueError(f"no images found under {root}")
    return samples, class_names


def class_distribution(samples: List[Sample], class_names: List[str]) -> dict:
    counts = {c: 0 for c in class_names}
    for _, label in samples:
        counts[class_names[label]] += 1
    return counts


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def stratified_split(samples, val_size=0.15, test_size=0.15, seed=42):
    from sklearn.model_selection import train_test_split
    labels = [lbl for _, lbl in samples]
    idx = np.arange(len(samples))
    train_idx, temp_idx = train_test_split(
        idx, test_size=val_size + test_size, stratify=labels, random_state=seed
    )
    temp_labels = [labels[i] for i in temp_idx]
    rel_test = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=rel_test, stratify=temp_labels, random_state=seed
    )
    take = lambda ids: [samples[i] for i in ids]
    return take(train_idx), take(val_idx), take(test_idx)


def grouped_split(samples, groups, val_size=0.15, test_size=0.15, seed=42):
    """Patient- (group-) disjoint split. ``groups[i]`` is the id for sample i."""
    from sklearn.model_selection import GroupShuffleSplit
    groups = np.asarray(groups)
    idx = np.arange(len(samples))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size,
                             random_state=seed)
    train_idx, temp_idx = next(gss1.split(idx, groups=groups))
    rel_test = test_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
    v_rel, t_rel = next(gss2.split(temp_idx, groups=groups[temp_idx]))
    val_idx, test_idx = temp_idx[v_rel], temp_idx[t_rel]
    take = lambda ids: [samples[i] for i in ids]
    return take(train_idx), take(val_idx), take(test_idx)


def subsample_per_class(samples, max_per_class, num_classes, seed=42):
    """Cap images per class - used by --smoke to verify the pipeline fast."""
    rng = np.random.default_rng(seed)
    buckets = {c: [] for c in range(num_classes)}
    for s in samples:
        buckets[s[1]].append(s)
    out = []
    for c, items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        out.extend(items[:max_per_class])
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class UltrasoundDataset(Dataset):
    def __init__(self, samples: List[Sample], transform: Optional[Callable] = None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
