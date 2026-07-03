#!/usr/bin/env python3
"""Exploratory analysis of the ultrasound dataset.

Works either on an extracted directory OR directly on the .zip (no extraction
needed). Reports, per task: number of classes, images per class, class-imbalance
ratio, and a sample of image sizes/modes. Optionally saves bar charts if
matplotlib is available.

Usage
-----
python analyze_dataset.py --zip "../US_Dateset_V01.zip"
python analyze_dataset.py --root "../US_Dateset_V01/dataset publish"
python analyze_dataset.py --zip "../US_Dateset_V01.zip" --save-figs outputs/eda
"""
import argparse
import io
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Top-level task directories we care about and how deep the class label sits
# relative to the task directory.
TASK_HINTS = [
    "organ_classification_1", "organ_classification_2",
    "Anomaly_detection_1", "Anomaly_detection_2",
    "organ_classification+anomaly_detection",
]


def _is_img(name: str) -> bool:
    return name.lower().endswith(IMG_EXTS)


def collect_from_zip(zip_path):
    z = zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist() if _is_img(n) and not n.endswith("/")]
    return z, names


def collect_from_dir(root):
    root = Path(root)
    names = [str(p.relative_to(root)) for p in root.rglob("*")
             if p.is_file() and _is_img(p.name)]
    return None, names


def task_of(path_parts):
    """Return (task, class_label) for a normalized path split on '/'.

    Expects paths like  <...>/<task>/<class>/img.jpg  or
    <...>/<task>/<normal|abnormal>/<organ>/img.jpg (combined split).
    """
    for i, part in enumerate(path_parts):
        if part in TASK_HINTS:
            rest = path_parts[i + 1:-1]  # class-path between task and filename
            if not rest:
                return None
            label = "/".join(rest)
            return part, label
    return None


def summarize(names):
    per_task = defaultdict(Counter)
    for n in names:
        parts = n.replace("\\", "/").split("/")
        res = task_of(parts)
        if res:
            task, label = res
            per_task[task][label] += 1
    return per_task


def sample_image_props(zf, names, n=40):
    try:
        from PIL import Image
    except ImportError:
        return None
    import random
    random.seed(0)
    sizes, modes = Counter(), Counter()
    for name in random.sample(names, min(n, len(names))):
        if zf is not None:
            with zf.open(name) as f:
                data = f.read()
            im = Image.open(io.BytesIO(data))
        else:
            im = Image.open(name)
        sizes[im.size] += 1
        modes[im.mode] += 1
    return sizes, modes


def maybe_save_fig(per_task, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed - skipping figures")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, counter in per_task.items():
        labels = list(counter.keys())
        values = [counter[k] for k in labels]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4))
        ax.bar(range(len(labels)), values, color="#4C78A8")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("images")
        ax.set_title(f"{task} - class distribution")
        fig.tight_layout()
        safe = task.replace("/", "_").replace("+", "-")
        fig.savefig(out_dir / f"dist_{safe}.png", dpi=150)
        plt.close(fig)
    print(f"[ok] saved distribution figures -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="path to US_Dateset_V01.zip")
    ap.add_argument("--root", default=None, help="path to extracted dataset dir")
    ap.add_argument("--save-figs", default=None, help="dir to save bar charts")
    args = ap.parse_args()
    if not args.zip and not args.root:
        ap.error("provide --zip or --root")

    if args.zip:
        zf, names = collect_from_zip(args.zip)
    else:
        zf, names = collect_from_dir(args.root)

    print(f"total images: {len(names)}\n")
    per_task = summarize(names)

    for task in sorted(per_task):
        counter = per_task[task]
        total = sum(counter.values())
        n_cls = len(counter)
        mx, mn = max(counter.values()), min(counter.values())
        print(f"### {task}")
        print(f"    classes: {n_cls} | images: {total} | "
              f"imbalance max/min: {mx}/{mn} = {mx / max(mn, 1):.1f}x")
        for label, c in counter.most_common():
            print(f"      {label:40s} {c:5d}  ({100 * c / total:4.1f}%)")
        print()

    props = sample_image_props(zf, names)
    if props:
        sizes, modes = props
        print("### sample image properties (40 random images)")
        print(f"    sizes: {sizes.most_common(5)}")
        print(f"    modes: {dict(modes)}")

    if args.save_figs:
        maybe_save_fig(per_task, args.save_figs)


if __name__ == "__main__":
    main()
