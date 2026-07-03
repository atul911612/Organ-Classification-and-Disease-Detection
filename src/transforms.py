"""Image transforms for ultrasound classification.

We fine-tune ImageNet-pretrained backbones, so inputs are normalized with the
ImageNet statistics. Augmentations are intentionally mild: ultrasound B-mode
frames are (near) grayscale with a fixed acquisition geometry, so heavy colour
or aggressive geometric distortion hurts more than it helps.
"""
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 224, train: bool = True):
    import torchvision.transforms as T
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(int(round(img_size * 1.14))),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
