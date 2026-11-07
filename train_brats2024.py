#!/usr/bin/env python3
"""Fold-2 mmFormer reproduction on the local TRDP BraTS 2024 protocol."""

import argparse
import importlib.util
import json
import logging
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.utils.enums import MetricReduction

import mmformer
from data.brats2024 import (
    CHANNEL_ORDER,
    EVAL_MASKS,
    build_loaders,
    classes_to_regions,
    mask_frequencies,
    raw_to_regions,
    sample_modality_masks,
    targets_to_regions,
)
from utils import criterions
from utils.lr_scheduler import LR_Scheduler


DEFAULT_DATA = "/net/cremi/abennabi/espaces/travail/trdp/SwinUNETR/data/brats2024"
DEFAULT_JSON = "/net/cremi/abennabi/espaces/travail/trdp/SwinUNETR/jsons/brats2024_folds.json"
DEFAULT_SWIN = "/net/cremi/abennabi/espaces/travail/trdp/SwinUNETR"
DEFAULT_CONFIG = "configs/mmformer_brats2024_fold2.yaml"

CONFIG_FIELDS = {
    "experiment": {"name": "experiment_name", "fold": "fold", "seed": "seed"},
    "data": {
        "data_dir": "data_dir",
        "json_list": "json_list",
        "swin_repo": "swin_repo",
        "workers": "workers",
    },
    "model": {"roi": "roi"},
    "training": {
        "batch_size": "batch_size",
        "epochs": "epochs",
        "learning_rate": "lr",
        "weight_decay": "weight_decay",
        "optimizer": "optimizer",
        "scheduler": "scheduler",
    },
    "validation": {
        "every": "val_every",
        "sliding_window_batch_size": "sw_batch_size",
        "overlap": "overlap",
    },
    "checkpoint": {"resume": "resume"},
    "output": {"save_dir": "save_dir"},
    "runtime": {"device": "device"},
    "sanity": {
        "only": "sanity_only",
        "skip_forward": "skip_forward_sanity",
    },
}


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown_sections = set(config) - set(CONFIG_FIELDS)
    if unknown_sections:
        raise ValueError("Unknown configuration sections: {}".format(sorted(unknown_sections)))
    defaults = {}
    for section, values in config.items():
        if not isinstance(values, dict):
            raise ValueError("Configuration section '{}' must be a mapping".format(section))
        unknown_keys = set(values) - set(CONFIG_FIELDS[section])
        if unknown_keys:
            raise ValueError(
                "Unknown keys in '{}': {}".format(section, sorted(unknown_keys))
            )
        for key, value in values.items():
            defaults[CONFIG_FIELDS[section][key]] = value
    return config, defaults


def parse_args():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=DEFAULT_CONFIG)
    config_args, _ = bootstrap.parse_known_args()
    _config, config_defaults = load_config(config_args.config)

    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.add_argument("--experiment-name", default="mmformer_brats2024")
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--json-list", default=DEFAULT_JSON)
    parser.add_argument("--swin-repo", default=DEFAULT_SWIN)
    parser.add_argument("--save-dir", default="runs/mmformer_brats2024/fold_2")
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--roi", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adam_amsgrad")
    parser.add_argument("--scheduler", default="poly")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--resume")
    parser.add_argument("--sanity-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--skip-forward-sanity", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--device", default="cuda")
    parser.set_defaults(**config_defaults)
    return parser.parse_args()


def configure_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(save_dir, "train.log"))],
        force=True,
    )


def save_resolved_config(args):
    path = os.path.join(args.save_dir, "resolved_config.yaml")
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(vars(args), stream, sort_keys=True)
    return path


def load_training_checkpoint(path, device, require_optimizer=True):
    """Load a trusted full-state checkpoint created by this training script.

    PyTorch 2.6+ defaults ``torch.load`` to weights-only mode, which rejects
    optimizer/legacy NumPy metadata. Resuming requires the complete trusted
    training state, not only a state_dict.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("Checkpoint does not exist: {}".format(path))
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint mapping, got {}".format(type(checkpoint)))
    required = {"epoch", "state_dict"}
    if require_optimizer:
        required.add("optimizer")
    missing = required - set(checkpoint)
    if missing:
        raise ValueError("Checkpoint is missing required fields: {}".format(sorted(missing)))
    return checkpoint


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_determinism(seed=seed)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def plain_tensor(value):
    """Drop MONAI metadata before mmFormer's boolean feature-mask indexing."""
    return value.as_tensor() if hasattr(value, "as_tensor") else value


def component_losses(predictions, target):
    cross = target.new_zeros(())
    dice = target.new_zeros(())
    for prediction in predictions:
        cross = cross + criterions.softmax_weighted_loss(prediction, target, num_cls=4)
        dice = dice + criterions.dice_loss(prediction, target, num_cls=4)
    return cross, dice


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    unwrap(model).is_training = True
    totals = Counter()
    mask_counts = Counter()
    started = time.time()
    for step, batch in enumerate(loader, start=1):
        image = plain_tensor(batch["image"]).to(device, non_blocking=True)
        target = plain_tensor(batch["label"]).to(device, non_blocking=True)
        mask = sample_modality_masks(image.shape[0], device=device)
        mask_counts.update("".join(str(int(v)) for v in row) for row in mask.tolist())

        fused, separate, prm = model(image, mask)
        fuse_cross = criterions.softmax_weighted_loss(fused, target, num_cls=4)
        fuse_dice = criterions.dice_loss(fused, target, num_cls=4)
        sep_cross, sep_dice = component_losses(separate, target)
        prm_cross, prm_dice = component_losses(prm, target)
        loss = fuse_cross + fuse_dice + sep_cross + sep_dice + prm_cross + prm_dice

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        values = {
            "loss": loss.item(),
            "fuse_cross": fuse_cross.item(),
            "fuse_dice": fuse_dice.item(),
            "sep_cross": sep_cross.item(),
            "sep_dice": sep_dice.item(),
            "prm_cross": prm_cross.item(),
            "prm_dice": prm_dice.item(),
        }
        totals.update(values)
        if step == 1 or step % 20 == 0:
            logging.info("epoch=%d step=%d/%d %s", epoch, step, len(loader), values)
    means = {name: value / len(loader) for name, value in totals.items()}
    logging.info(
        "epoch=%d train=%s mask_counts=%s seconds=%.1f",
        epoch,
        means,
        dict(sorted(mask_counts.items())),
        time.time() - started,
    )
    return means, mask_counts


def _predictor(model, mask_values):
    def predict(patches):
        mask = torch.as_tensor(mask_values, dtype=torch.bool, device=patches.device)
        mask = mask.view(1, 4).repeat(patches.shape[0], 1)
        return model(patches, mask)

    return predict


def validate(model, loader, device, roi, sw_batch_size, overlap, conditions):
    model.eval()
    unwrap(model).is_training = False
    results = {}
    with torch.no_grad():
        for condition, mask in conditions.items():
            metric = DiceMetric(
                include_background=True,
                reduction=MetricReduction.MEAN_BATCH,
                get_not_nans=True,
            )
            for batch in loader:
                image = plain_tensor(batch["image"]).to(device, non_blocking=True)
                target = targets_to_regions(
                    plain_tensor(batch["label"]).to(device, non_blocking=True)
                )
                prediction = sliding_window_inference(
                    image,
                    roi_size=(roi, roi, roi),
                    sw_batch_size=sw_batch_size,
                    predictor=_predictor(model, mask),
                    overlap=overlap,
                )
                metric(y_pred=classes_to_regions(prediction), y=target)
            values, _not_nans = metric.aggregate()
            values = values.detach().cpu().numpy()  # repository order: TC, WT, ET
            results[condition] = {
                "ET": float(values[2]),
                "TC": float(values[0]),
                "WT": float(values[1]),
                "mean": float(np.mean(values)),
            }
            logging.info("validation condition=%s metrics=%s", condition, results[condition])
    model.train()
    return results


def save_metrics(path, epoch, results):
    record = {"epoch": epoch, **results}
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)


def _load_swin_label_transform(swin_repo):
    path = os.path.join(swin_repo, "utils", "data_utils.py")
    spec = importlib.util.spec_from_file_location("swin_reference_data_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._brats_label_transform("brats2024")


def run_sanity_checks(args, train_loader, train_files, validation_files, device):
    train_patients = {item["patient"] for item in train_files}
    validation_patients = {item["patient"] for item in validation_files}
    logging.info(
        "dataset train_cases=%d validation_cases=%d train_patients=%d validation_patients=%d "
        "train_folds=%s validation_folds=%s overlap=%d",
        len(train_files),
        len(validation_files),
        len(train_patients),
        len(validation_patients),
        sorted({item["fold"] for item in train_files}),
        sorted({item["fold"] for item in validation_files}),
        len(train_patients & validation_patients),
    )

    batch = next(iter(train_loader))
    regions = targets_to_regions(batch["label"])
    raw_crop = batch["raw_label"]
    logging.info(
        "input image_shape=%s label_shape=%s channel_order=%s raw_labels=%s "
        "ET_voxels=%d TC_voxels=%d WT_voxels=%d",
        tuple(batch["image"].shape),
        tuple(batch["label"].shape),
        list(CHANNEL_ORDER),
        sorted(int(value) for value in torch.unique(raw_crop)),
        int(regions[:, 2].sum()),
        int(regions[:, 0].sum()),
        int(regions[:, 1].sum()),
    )

    raw = plain_tensor(raw_crop)[0, 0]
    swin_transform = _load_swin_label_transform(args.swin_repo)
    swin_regions = swin_transform({"label": raw})["label"]
    ours = raw_to_regions(raw)
    names = ("TC", "WT", "ET")
    for index, name in enumerate(names):
        if not torch.equal(swin_regions[index].bool(), ours[index].bool()):
            raise AssertionError("{} differs from SwinUNETR target construction".format(name))
        if not torch.equal(swin_regions[index].bool(), regions[0, index].bool()):
            raise AssertionError("{} differs after mmFormer target construction".format(name))
    logging.info("label_consistency voxelwise_equal=True regions=%s", names)

    frequencies = mask_frequencies(1000, args.seed)
    if len(frequencies) != 15 or "0000" in frequencies:
        raise AssertionError("Invalid modality-mask sample: {}".format(frequencies))
    logging.info("mask_samples=1000 frequencies=%s all_zero=0", frequencies)

    if args.skip_forward_sanity:
        logging.warning("forward sanity was explicitly skipped")
        return
    model = mmformer.Model(num_cls=4, input_size=args.roi).to(device)
    model.train()
    model.is_training = True
    image = plain_tensor(batch["image"]).to(device)
    target = plain_tensor(batch["label"]).to(device)
    mask = sample_modality_masks(image.shape[0], device=device)
    fused, separate, prm = model(image, mask)
    fuse_cross = criterions.softmax_weighted_loss(fused, target, num_cls=4)
    fuse_dice = criterions.dice_loss(fused, target, num_cls=4)
    sep_cross, sep_dice = component_losses(separate, target)
    prm_cross, prm_dice = component_losses(prm, target)
    loss = fuse_cross + fuse_dice + sep_cross + sep_dice + prm_cross + prm_dice
    if not torch.isfinite(loss):
        raise AssertionError("Non-finite sanity-check loss")
    loss.backward()
    if tuple(fused.shape[2:]) != tuple(target.shape[2:]):
        raise AssertionError("Output and target spatial shapes differ")
    logging.info(
        "forward_sanity success=True loss=%.6f backward=True output_shape=%s target_shape=%s",
        loss.item(),
        tuple(fused.shape),
        tuple(target.shape),
    )


def main():
    args = parse_args()
    configure_logging(args.save_dir)
    resolved_path = save_resolved_config(args)
    seed_everything(args.seed)
    if args.fold != 2:
        raise ValueError("This fair-comparison entry point is intentionally restricted to Fold 2")
    if args.roi % 16:
        raise ValueError("ROI must be divisible by 16")
    if args.optimizer != "adam_amsgrad":
        raise ValueError("Faithful mmFormer reproduction requires optimizer: adam_amsgrad")
    if args.scheduler != "poly":
        raise ValueError("Faithful mmFormer reproduction requires scheduler: poly")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu only for diagnostics")
    device = torch.device(args.device)
    logging.info("configuration=%s", vars(args))
    logging.info("resolved_config=%s", resolved_path)

    loaders = build_loaders(
        args.json_list,
        args.data_dir,
        fold=args.fold,
        roi=(args.roi,) * 3,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    train_loader, validation_loader, train_files, validation_files = loaders
    run_sanity_checks(args, train_loader, train_files, validation_files, device)
    if args.sanity_only:
        logging.info("Sanity checks complete; training was not started")
        return

    model = mmformer.Model(num_cls=4, input_size=args.roi).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}],
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=True,
    )
    scheduler = LR_Scheduler(args.lr, args.epochs)
    start_epoch = 0
    best_mean = -1.0
    if args.resume:
        checkpoint = load_training_checkpoint(args.resume, device, require_optimizer=True)
        unwrap(model).load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_mean = float(checkpoint.get("best_mean", best_mean))
        logging.info(
            "resumed checkpoint=%s completed_epoch=%d next_epoch=%d best_mean=%.6f",
            args.resume,
            checkpoint["epoch"],
            start_epoch,
            best_mean,
        )

    cumulative_masks = Counter()
    for epoch in range(start_epoch, args.epochs):
        learning_rate = scheduler(optimizer, epoch)
        losses, epoch_masks = train_epoch(model, train_loader, optimizer, device, epoch)
        cumulative_masks.update(epoch_masks)
        logging.info(
            "epoch=%d learning_rate=%.8g total_loss=%.6f cumulative_mask_counts=%s",
            epoch,
            learning_rate,
            losses["loss"],
            dict(sorted(cumulative_masks.items())),
        )
        state = {
            "epoch": epoch,
            "state_dict": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_mean": best_mean,
            "seed": args.seed,
            "fold": args.fold,
            "channel_order": CHANNEL_ORDER,
            "training_args": vars(args),
        }
        torch.save(state, os.path.join(args.save_dir, "model_last.pth"))
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            results = validate(
                model,
                validation_loader,
                device,
                args.roi,
                args.sw_batch_size,
                args.overlap,
                {"all": EVAL_MASKS["all"]},
            )
            save_metrics(os.path.join(args.save_dir, "validation_latest.json"), epoch, results)
            if results["all"]["mean"] > best_mean:
                best_mean = float(results["all"]["mean"])
                state["best_mean"] = best_mean
                torch.save(state, os.path.join(args.save_dir, "model_best_all_mean.pth"))
                logging.info(
                    "checkpoint path=%s criterion=all_mean value=%.6f",
                    os.path.join(args.save_dir, "model_best_all_mean.pth"),
                    best_mean,
                )
            state["best_mean"] = best_mean
            torch.save(state, os.path.join(args.save_dir, "model_last.pth"))

    best_path = os.path.join(args.save_dir, "model_best_all_mean.pth")
    checkpoint = load_training_checkpoint(best_path, device, require_optimizer=False)
    unwrap(model).load_state_dict(checkpoint["state_dict"])
    final_results = validate(
        model,
        validation_loader,
        device,
        args.roi,
        args.sw_batch_size,
        args.overlap,
        EVAL_MASKS,
    )
    save_metrics(
        os.path.join(args.save_dir, "fold2_missing_modality_results.json"),
        checkpoint["epoch"],
        final_results,
    )


if __name__ == "__main__":
    main()
