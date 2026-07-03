"""Evaluate a trained checkpoint on a test split, or run inference on a folder.

Examples
--------
# reproduce the held-out test metrics for a trained model:
python -m src.evaluate --checkpoint outputs/organ_convnext_tiny/best_model.pt \
    --data "/path/.../organ_classification_1"

# predict every image in an arbitrary folder (no labels needed):
python -m src.evaluate --checkpoint outputs/organ_convnext_tiny/best_model.pt \
    --predict-dir "/path/to/new_images" --out outputs/predictions.csv
"""
import argparse
import csv
from pathlib import Path

from . import datasets as D
from . import engine
from . import models as M
from . import transforms as Tr
from . import utils


def get_args():
    p = argparse.ArgumentParser(description="Evaluate / run inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default=None,
                   help="labelled task root -> recompute test-split metrics")
    p.add_argument("--predict-dir", default=None,
                   help="unlabelled folder -> write per-image predictions CSV")
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(checkpoint, device):
    import torch
    ckpt = torch.load(checkpoint, map_location=device)
    class_names = ckpt["class_names"]
    arch = ckpt.get("arch", M.DEFAULT_MODEL)
    img_size = ckpt.get("img_size", 224)
    model = M.create(arch, len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, class_names, img_size


def evaluate_labelled(args, model, class_names, img_size, device):
    import torch
    samples, ds_classes = D.build_index(args.data)
    if ds_classes != class_names:
        print("[warn] dataset classes differ from checkpoint's class order; "
              "using checkpoint order for metrics.")
    # reproduce the SAME test split used during training (same seed/sizes)
    _, _, test_s = D.stratified_split(
        samples, val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    test_ds = D.UltrasoundDataset(test_s, Tr.build_transforms(img_size, train=False))
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    res = engine.evaluate(model, loader, device)
    metrics = utils.compute_metrics(res["targets"], res["preds"], res["probs"],
                                    len(class_names))
    print("=== TEST METRICS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(utils.classification_report_str(res["targets"], res["preds"], class_names))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    utils.save_confusion_matrix(res["targets"], res["preds"], class_names,
                                out_dir / "confusion_matrix.png")
    utils.save_json({"metrics": metrics, "class_names": class_names},
                    out_dir / "metrics.json")


def predict_folder(args, model, class_names, img_size, device):
    import torch
    from PIL import Image
    tfm = Tr.build_transforms(img_size, train=False)
    paths = sorted(p for p in Path(args.predict_dir).rglob("*")
                   if p.is_file() and p.suffix.lower() in D.IMG_EXTS)
    if not paths:
        print(f"no images found under {args.predict_dir}")
        return
    out_csv = Path(args.out)
    if out_csv.suffix != ".csv":
        out_csv = out_csv / "predictions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "pred_label", "confidence"] + class_names)
        with torch.no_grad():
            for i in range(0, len(paths), args.batch_size):
                batch_paths = paths[i:i + args.batch_size]
                imgs = torch.stack([tfm(Image.open(p).convert("RGB"))
                                    for p in batch_paths]).to(device)
                probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
                for p, pr in zip(batch_paths, probs):
                    j = int(pr.argmax())
                    writer.writerow([str(p), class_names[j], f"{pr[j]:.4f}",
                                     *[f"{x:.4f}" for x in pr]])
    print(f"wrote predictions for {len(paths)} images -> {out_csv}")


def main():
    args = get_args()
    utils.set_seed(args.seed)
    device = utils.get_device()
    model, class_names, img_size = load_model(args.checkpoint, device)
    print(f"loaded {args.checkpoint} | classes={class_names} | img_size={img_size}")
    if args.predict_dir:
        predict_folder(args, model, class_names, img_size, device)
    elif args.data:
        evaluate_labelled(args, model, class_names, img_size, device)
    else:
        raise SystemExit("provide --data (labelled) or --predict-dir (inference)")


if __name__ == "__main__":
    main()
