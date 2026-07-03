"""Model factory.

State of the art for ultrasound image analysis (2024-2025) is a *ViT-based
ultrasound foundation model* pretrained with masked image modeling on millions
of ultrasound frames:

  * USFM  - Jiao et al., "USFM: A Universal Ultrasound Foundation Model...",
            Medical Image Analysis, 2024.  ViT-B/16 + spatial-frequency MIM,
            pretrained on >2M US images.  Repo: https://github.com/openmedlab/USFM
  * URFM / USF-MAE (2025) - follow-up ultrasound foundation models.

Because those weights must be downloaded separately, this factory gives you:

  1. Strong, fully-reproducible *ImageNet transfer-learning baselines* via `timm`
     (ConvNeXt / ViT / Swin / EfficientNetV2 / ResNet).  ConvNeXt and ViT are the
     recommended defaults and are competitive baselines for this dataset.
  2. `build_usfm(...)` - a drop-in hook to fine-tune the actual USFM ViT-B
     encoder once you provide its checkpoint (`--usfm-weights path.pth`).

All models output raw logits of shape (B, num_classes).
"""
from typing import Optional

# short-name -> timm model id (with recommended pretrained tag)
SUPPORTED = {
    "convnext_tiny":    "convnext_tiny.fb_in22k_ft_in1k",
    "convnext_small":   "convnext_small.fb_in22k_ft_in1k",
    "vit_base":         "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "swin_tiny":        "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "resnet50":         "resnet50.a1_in1k",
}
DEFAULT_MODEL = "convnext_tiny"


def _require_timm():
    try:
        import timm
        return timm
    except ImportError as e:
        raise ImportError(
            "timm is required to build models. Install with `pip install timm`."
        ) from e


def build_model(name: str, num_classes: int, pretrained: bool = True,
                drop_rate: float = 0.1):
    """Build an ImageNet-pretrained backbone with a fresh classification head."""
    timm = _require_timm()
    timm_id = SUPPORTED.get(name, name)  # allow passing a raw timm id too
    model = timm.create_model(
        timm_id, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate
    )
    return model


def build_usfm(num_classes: int, weights_path: Optional[str] = None,
               drop_rate: float = 0.1):
    """Fine-tune the USFM ultrasound foundation model (ViT-B/16 backbone).

    Download the checkpoint from https://github.com/openmedlab/USFM and pass its
    path. Without a checkpoint this returns a randomly-initialized ViT-B (useful
    only as a from-scratch control, not as the foundation model).
    """
    timm = _require_timm()
    import torch
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=False,
        num_classes=num_classes, drop_rate=drop_rate,
    )
    if weights_path:
        ckpt = torch.load(weights_path, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # strip common wrapper prefixes so keys line up with a timm ViT
        cleaned = {}
        for k, v in state.items():
            for pre in ("module.", "encoder.", "backbone.", "model."):
                if k.startswith(pre):
                    k = k[len(pre):]
            cleaned[k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[USFM] loaded {len(cleaned)} tensors from {weights_path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("[USFM] WARNING: no --usfm-weights given -> random ViT-B init "
              "(from-scratch control, NOT the foundation model).")
    return model


def create(name: str, num_classes: int, pretrained: bool = True,
           usfm_weights: Optional[str] = None, drop_rate: float = 0.1):
    """Unified entrypoint used by train.py / evaluate.py."""
    if name == "usfm":
        return build_usfm(num_classes, weights_path=usfm_weights, drop_rate=drop_rate)
    return build_model(name, num_classes, pretrained=pretrained, drop_rate=drop_rate)
