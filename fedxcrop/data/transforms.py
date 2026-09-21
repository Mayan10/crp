"""Image transforms.

Augmentation is applied to the training split only. Validation and test images
are only resized, center cropped, and normalized, so the numbers reported for
them are not influenced by random augmentation.
"""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(image_size: int = 224):
    """Augmented pipeline for training images."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(0.4, 0.4, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(image_size: int = 224):
    """Deterministic pipeline for validation, test, and explainability."""
    resize_size = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def mask_transform(image_size: int = 224):
    """Geometry only pipeline, used to bring a segmented image into pixel
    correspondence with its center cropped color counterpart."""
    resize_size = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def denormalize(tensor):
    """Undo ImageNet normalization, giving a tensor back in [0, 1]."""
    import torch

    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)
