"""Train an ultrasound classifier (organ classification or anomaly detection).

Examples
--------
# fast pipeline check (a few images/class, 1 epoch, CPU ok):
python -m src.train --data "../US_Dateset_V01/dataset publish/organ_classification_1" \
    --task organ --model convnext_tiny --smoke

# real organ-classification run from a YAML preset:
python -m src.train --config configs/organ_classification.yaml \
    --data "/path/to/dataset publish/organ_classification_1"

# fine-tune the USFM ultrasound foundation model:
python -m src.train --config configs/organ_classification.yaml \
    --data "/path/.../organ_classification_1" --model usfm --usfm-weights usfm_vitb.pth
"""
import argparse
import time
from pathlib import Path

from . import datasets as D
from . import engine
from . import models as M
from . import transforms as Tr
from . import utils


def get_args():
    p = argparse.ArgumentParser(description="Train ultrasound classifier")
    p.add_argument("--config", default=None, help="YAML preset (optional)")
    p.add_argument("--data", required=False, help="task root: <root>/<class>/*.jpg")
    p.add_argument("--task", default="organ", choices=["organ", "anomaly"],
                   help="only affects output naming / default metric")
    p.add_argument("--out", default="outputs")
    p.add_argument("--model", default=M.DEFAULT_MODEL)
    p.add_argument("--usfm-weights", default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--drop-rate", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-class-weights", action="store_true",
                   help="disable inverse-frequency class weighting")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--smoke", action="store_true",
                   help="tiny subset + 1 epoch to verify the pipeline")

    args, _ = p.parse_known_args()
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        p.set_defaults(**cfg)
    args = p.parse_args()
    if not args.data:
        p.error("--data is required (path to the task's class folder)")
    return args


def build_loaders(args, num_classes):
    import torch
    samples, class_names = D.build_index(args.data)
    if args.smoke:
        samples = D.subsample_per_class(samples, max_per_class=20,
                                        num_classes=len(class_names), seed=args.seed)
    train_s, val_s, test_s = D.stratified_split(
        samples, val_size=args.val_size, test_size=args.test_size, seed=args.seed)

    train_ds = D.UltrasoundDataset(train_s, Tr.build_transforms(args.img_size, train=True))
    val_ds = D.UltrasoundDataset(val_s, Tr.build_transforms(args.img_size, train=False))
    test_ds = D.UltrasoundDataset(test_s, Tr.build_transforms(args.img_size, train=False))

    mk = lambda ds, shuf: torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuf,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    return (mk(train_ds, True), mk(val_ds, False), mk(test_ds, False),
            class_names, train_s)


def main():
    import torch
    import torch.nn as nn

    args = get_args()
    utils.set_seed(args.seed)
    device = utils.get_device()
    out_dir = Path(args.out) / f"{args.task}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}")

    # peek at classes first (needed to size the head)
    _, class_names = D.build_index(args.data)
    num_classes = len(class_names)
    train_loader, val_loader, test_loader, class_names, train_s = \
        build_loaders(args, num_classes)
    print(f"classes ({num_classes}): {class_names}")
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}")

    model = M.create(args.model, num_classes, pretrained=not args.no_pretrained,
                     usfm_weights=args.usfm_weights, drop_rate=args.drop_rate).to(device)
    print(f"model={args.model}  trainable params={utils.count_parameters(model):,}")

    # loss (optionally class-weighted for imbalance)
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    else:
        w = torch.tensor(utils.class_weights_from_samples(train_s, num_classes),
                         device=device)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    epochs = 1 if args.smoke else args.epochs
    steps = max(len(train_loader), 1) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    logger = utils.CSVLogger(out_dir / "training_log.csv",
                             ["epoch", "train_loss", "train_acc", "val_loss",
                              "val_acc", "val_macro_f1", "lr", "seconds"])
    stopper = utils.EarlyStopper(patience=args.patience, mode="max")
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = engine.train_one_epoch(model, train_loader, optimizer, criterion,
                                    device, scaler=scaler, scheduler=scheduler,
                                    grad_clip=args.grad_clip)
        val = engine.evaluate(model, val_loader, device, criterion)
        vm = utils.compute_metrics(val["targets"], val["preds"], val["probs"],
                                   num_classes)
        dt = time.time() - t0
        print(f"epoch {epoch:02d}/{epochs} | train_loss {tr['loss']:.4f} "
              f"acc {tr['acc']:.4f} | val_loss {val['loss']:.4f} "
              f"acc {vm['accuracy']:.4f} macroF1 {vm['macro_f1']:.4f} | {dt:.1f}s")
        logger.log({"epoch": epoch, "train_loss": round(tr["loss"], 5),
                    "train_acc": round(tr["acc"], 5), "val_loss": round(val["loss"], 5),
                    "val_acc": round(vm["accuracy"], 5),
                    "val_macro_f1": round(vm["macro_f1"], 5),
                    "lr": optimizer.param_groups[0]["lr"], "seconds": round(dt, 1)})

        if stopper.step(vm["macro_f1"]):
            utils.save_checkpoint(
                {"model": model.state_dict(), "class_names": class_names,
                 "arch": args.model, "img_size": args.img_size,
                 "val_metrics": vm}, best_path)
            print(f"  ^ new best (macro_f1={vm['macro_f1']:.4f}) -> {best_path}")
        if stopper.should_stop:
            print(f"early stopping at epoch {epoch}")
            break

    # ---- final test with best checkpoint ----
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    test = engine.evaluate(model, test_loader, device, criterion)
    tm = utils.compute_metrics(test["targets"], test["preds"], test["probs"],
                               num_classes)
    print("\n=== TEST ===")
    for k, v in tm.items():
        print(f"  {k}: {v:.4f}")
    print(utils.classification_report_str(test["targets"], test["preds"], class_names))
    utils.save_confusion_matrix(test["targets"], test["preds"], class_names,
                                out_dir / "confusion_matrix_test.png")
    utils.save_json({"test_metrics": tm, "class_names": class_names,
                     "config": vars(args)}, out_dir / "test_results.json")
    print(f"\nartifacts saved under: {out_dir}")


if __name__ == "__main__":
    main()
