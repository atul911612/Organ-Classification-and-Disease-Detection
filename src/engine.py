"""Training and evaluation loops."""
import torch
from tqdm import tqdm

from .utils import AverageMeter


def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, scheduler=None, grad_clip=None):
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    use_amp = scaler is not None
    for imgs, targets in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_meter.update(loss.item(), imgs.size(0))
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return {"loss": loss_meter.avg, "acc": correct / max(total, 1)}


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    loss_meter = AverageMeter()
    all_logits, all_targets = [], []
    for imgs, targets in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(imgs)
        if criterion is not None:
            loss_meter.update(criterion(logits, targets).item(), imgs.size(0))
        all_logits.append(logits.float().cpu())
        all_targets.append(targets.cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = logits.argmax(1).numpy()
    return {
        "loss": loss_meter.avg,
        "probs": probs,
        "preds": preds,
        "targets": targets.numpy(),
    }
