"""BraTS 2024 input pipeline shared with the local SwinUNETR experiments.

The JSON and NIfTI files are read in place.  No converted dataset is written.
"""

import json
import os
from collections import Counter

import torch
from monai import data, transforms


CHANNEL_ORDER = ("t1n", "t1c", "t2w", "t2f")
EVAL_MASKS = {
    "all": (1, 1, 1, 1),
    "no_t1n": (0, 1, 1, 1),
    "no_t1c": (1, 0, 1, 1),
    "no_t2w": (1, 1, 0, 1),
    "no_t2f": (1, 1, 1, 0),
}


def read_fold(json_path, data_dir, fold=2):
    with open(json_path, encoding="utf-8") as stream:
        entries = json.load(stream)["training"]
    resolved = []
    for source in entries:
        item = dict(source)
        item["image"] = [os.path.join(data_dir, path) for path in source["image"]]
        item["label"] = os.path.join(data_dir, source["label"])
        _check_channel_order(item["image"])
        resolved.append(item)
    train = [item for item in resolved if item.get("fold") != fold]
    validation = [item for item in resolved if item.get("fold") == fold]
    overlap = {item["patient"] for item in train} & {item["patient"] for item in validation}
    if overlap:
        raise ValueError("Patient overlap between training and validation: {}".format(sorted(overlap)[:5]))
    return train, validation


def _check_channel_order(paths):
    if len(paths) != 4:
        raise ValueError("Expected four MRI paths, got {}".format(len(paths)))
    for path, modality in zip(paths, CHANNEL_ORDER):
        if not os.path.basename(path).endswith("-{}.nii.gz".format(modality)):
            raise ValueError("Expected {} in channel order, got {}".format(modality, path))


def raw_to_regions(label):
    """SwinUNETR's BraTS 2024 target order: [TC, WT, ET], excluding label 4."""
    return torch.stack(
        (
            (label == 1) | (label == 3),
            (label == 1) | (label == 2) | (label == 3),
            label == 3,
        ),
        dim=0,
    ).float()


def raw_to_mmformer_classes(label):
    """Four exclusive classes for the official softmax losses; RC (4) is background."""
    foreground = torch.stack((label == 1, label == 2, label == 3), dim=0)
    background = ~torch.any(foreground, dim=0, keepdim=True)
    return torch.cat((background, foreground), dim=0).float()


def classes_to_regions(classes):
    """Convert class probabilities/one-hot [B,4,...] to binary [TC,WT,ET]."""
    if classes.ndim != 5 or classes.shape[1] != 4:
        raise ValueError("Expected [B,4,H,W,D], got {}".format(tuple(classes.shape)))
    labels = torch.argmax(classes, dim=1)
    return torch.stack(
        ((labels == 1) | (labels == 3), labels > 0, labels == 3), dim=1
    ).float()


def targets_to_regions(targets):
    return classes_to_regions(targets)


def build_transforms(roi=(96, 96, 96)):
    # This order intentionally mirrors SwinUNETR/utils/data_utils.py exactly.
    label_transform = transforms.Lambdad(keys="label", func=raw_to_mmformer_classes)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.CopyItemsd(keys="label", names="raw_label"),
            transforms.EnsureChannelFirstd(keys="raw_label", channel_dim="no_channel"),
            label_transform,
            transforms.CropForegroundd(
                keys=["image", "label", "raw_label"],
                source_key="image",
                k_divisible=roi,
                allow_smaller=True,
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label", "raw_label"], roi_size=roi, random_size=False
            ),
            transforms.RandFlipd(keys=["image", "label", "raw_label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label", "raw_label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label", "raw_label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label", "raw_label"]),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.CopyItemsd(keys="label", names="raw_label"),
            transforms.EnsureChannelFirstd(keys="raw_label", channel_dim="no_channel"),
            label_transform,
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label", "raw_label"]),
        ]
    )
    return train_transform, validation_transform


def build_loaders(json_path, data_dir, fold=2, roi=(96, 96, 96), batch_size=1, workers=8):
    train_files, validation_files = read_fold(json_path, data_dir, fold)
    train_transform, validation_transform = build_transforms(roi)
    train_dataset = data.Dataset(train_files, train_transform)
    validation_dataset = data.Dataset(validation_files, validation_transform)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    validation_loader = data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_loader, validation_loader, train_files, validation_files


def sample_modality_masks(batch_size, device=None, generator=None):
    """Official strategy: iid Bernoulli(0.5), conditioned on at least one modality."""
    masks = torch.zeros((batch_size, 4), dtype=torch.bool, device=device)
    missing = torch.ones(batch_size, dtype=torch.bool, device=device)
    while torch.any(missing):
        draw = torch.rand((int(missing.sum()), 4), device=device, generator=generator) < 0.5
        masks[missing] = draw
        missing = ~torch.any(masks, dim=1)
    return masks


def mask_frequencies(samples=1000, seed=1024):
    generator = torch.Generator().manual_seed(seed)
    masks = sample_modality_masks(samples, generator=generator)
    counts = Counter(tuple(int(value) for value in row) for row in masks.tolist())
    return {"".join(map(str, key)): counts[key] for key in sorted(counts)}
