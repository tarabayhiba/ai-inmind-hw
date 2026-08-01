"""Correctness checks for lejepa_data.py.
Runs against the real, already-downloaded CIFAR-10
train split
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lejepa_data import CIFAR10_MEAN, CIFAR10_STD, MultiViewCIFAR10, get_pretrain_loader

torch.manual_seed(0)

TRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cifar10", "train")

_all_ok = True


def check(name, cond, detail=""):
    global _all_ok
    _all_ok = _all_ok and bool(cond)
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def test_multiview_dataset_item_shape():
    views = 4
    dataset = MultiViewCIFAR10(root=TRAIN_DIR, views=views)
    item = dataset[0]
    check(
        "dataset item has shape (V, C, H, W)",
        tuple(item.shape) == (views, 3, 32, 32),
        f"shape={tuple(item.shape)}",
    )


def test_views_are_independently_augmented():
    dataset = MultiViewCIFAR10(root=TRAIN_DIR, views=4)
    item = dataset[0]
    check(
        "different views of the same image are not identical (augmentation is actually applied)",
        not torch.allclose(item[0], item[1]),
    )


def test_normalization_matches_dataset_constants():
    # Average over many images so per-image crop/color-jitter noise
    # washes out and only the Normalize() call's effect remains.
    dataset = MultiViewCIFAR10(root=TRAIN_DIR, views=1)
    batch = torch.stack([dataset[i][0] for i in range(256)])  # (256, C, H, W)
    per_channel_mean = batch.mean(dim=(0, 2, 3))
    per_channel_std = batch.std(dim=(0, 2, 3))
    check(
        "normalized per-channel mean is near 0",
        per_channel_mean.abs().max().item() < 0.5,
        f"mean={per_channel_mean.tolist()}",
    )
    check(
        "normalized per-channel std is near 1",
        (per_channel_std - 1.0).abs().max().item() < 0.5,
        f"std={per_channel_std.tolist()}",
    )
    check(
        "dataset exposes the mean/std constants it normalizes with",
        len(CIFAR10_MEAN) == 3 and len(CIFAR10_STD) == 3,
    )


def test_pretrain_loader_batch_shape():
    config = {
        "paths": {"train_dir": TRAIN_DIR},
        "lejepa": {
            "views": 4,
            "pretrain": {"batch_size": 8, "num_workers": 0},
        },
    }
    loader = get_pretrain_loader(config)
    batch = next(iter(loader))
    check(
        "loader batch has shape (N, V, C, H, W)",
        tuple(batch.shape) == (8, 4, 3, 32, 32),
        f"shape={tuple(batch.shape)}",
    )


def test_dataset_uses_train_split_only():
    dataset = MultiViewCIFAR10(root=TRAIN_DIR, views=1)
    check(
        "dataset length matches CIFAR-10's 50k train split, not the 10k test split",
        len(dataset) == 50000,
        f"len={len(dataset)}",
    )


if __name__ == "__main__":
    test_multiview_dataset_item_shape()
    test_views_are_independently_augmented()
    test_normalization_matches_dataset_constants()
    test_pretrain_loader_batch_shape()
    test_dataset_uses_train_split_only()

    print()
    if _all_ok:
        print("All checks passed.")
    else:
        print("Some checks FAILED -- see above.")
        sys.exit(1)
