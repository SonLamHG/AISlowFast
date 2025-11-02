from __future__ import annotations

import argparse
import importlib
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import AgeRatingDataset
from .model import SlowFastAgeModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SlowFast for age rating classification")
    parser.add_argument("--train-csv", type=Path, dest="train_csv", default=None)
    parser.add_argument("--video-root", type=Path, dest="video_root", default=None)
    parser.add_argument("--output-dir", type=Path, dest="output_dir", default=None)
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=None)
    parser.add_argument("--max-epochs", type=int, dest="max_epochs", default=None)
    parser.add_argument("--learning-rate", type=float, dest="learning_rate", default=None)
    parser.add_argument("--weight-decay", type=float, dest="weight_decay", default=None)
    parser.add_argument("--device", type=str, dest="device", default=None)
    parser.add_argument("--checkpoint", type=Path, dest="checkpoint_path", default=None)
    parser.add_argument("--num-workers", type=int, dest="num_workers", default=None)
    parser.add_argument("--alpha", type=int, dest="alpha", default=None)
    parser.add_argument("--frames", type=int, dest="fast_num_frames", default=None)
    parser.add_argument("--clip-duration", type=float, dest="clip_duration", default=None)
    parser.add_argument("--val-split", type=float, dest="val_split", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig()
    replacements = {}
    for field in (
        "train_csv",
        "video_root",
        "output_dir",
    "val_split",
        "batch_size",
        "max_epochs",
        "learning_rate",
        "weight_decay",
        "device",
        "checkpoint_path",
        "num_workers",
        "alpha",
        "fast_num_frames",
        "clip_duration",
    ):
        value = getattr(args, field)
        if value is not None:
            replacements[field] = value
    return replace(config, **replacements)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slowfast_collate_fn(batch: List[Tuple[List[torch.Tensor], int]]):
    slow = torch.stack([item[0][0] for item in batch], dim=0)
    fast = torch.stack([item[0][1] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return [slow, fast], labels


def create_dataloaders(config: TrainConfig) -> Tuple[DataLoader, DataLoader | None]:
    base_dataset = AgeRatingDataset(config.train_csv, config, is_train=True)
    rng = random.Random(config.seed)
    val_ratio = min(max(config.val_split, 0.0), 1.0)

    samples_by_class: Dict[int, List[Tuple[Path, int]]] = {}
    for sample in base_dataset.samples:
        samples_by_class.setdefault(sample[1], []).append(sample)

    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []

    for label_samples in samples_by_class.values():
        label_samples = label_samples.copy()
        rng.shuffle(label_samples)
        if val_ratio <= 0.0:
            train_samples.extend(label_samples)
            continue

        val_count = int(len(label_samples) * val_ratio)
        if val_ratio > 0.0 and val_count == 0 and len(label_samples) > 1:
            val_count = 1
        if val_count >= len(label_samples):
            val_count = max(len(label_samples) - 1, 0)

        val_samples.extend(label_samples[:val_count])
        train_samples.extend(label_samples[val_count:])

    if not train_samples and val_samples:
        train_samples, val_samples = val_samples, []

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    train_dataset = AgeRatingDataset(
        config.train_csv,
        config,
        is_train=True,
        samples=train_samples,
        class_to_idx=base_dataset.class_to_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=slowfast_collate_fn,
    )

    val_loader = None
    if val_samples:
        val_dataset = AgeRatingDataset(
            config.train_csv,
            config,
            is_train=False,
            samples=val_samples,
            class_to_idx=base_dataset.class_to_idx,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=slowfast_collate_fn,
        )

    return train_loader, val_loader


def train_one_epoch(
    model: SlowFastAgeModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = [pathway.to(device, non_blocking=True) for pathway in inputs]
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * targets.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        if batch_idx % 10 == 0:
            avg_loss = total_loss / max(total, 1)
            avg_acc = correct / max(total, 1)
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.3f}"
            )

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(
    model: SlowFastAgeModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    total_mse = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = [pathway.to(device, non_blocking=True) for pathway in inputs]
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

            probs = torch.softmax(outputs, dim=1)
            class_indices = torch.arange(
                probs.shape[1], device=probs.device, dtype=probs.dtype
            )
            expected_value = torch.sum(probs * class_indices, dim=1)
            total_mse += F.mse_loss(
                expected_value, targets.float(), reduction="sum"
            ).item()

    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    try:
        metrics_module = importlib.import_module("sklearn.metrics")
        precision_recall_fscore_support = getattr(
            metrics_module, "precision_recall_fscore_support"
        )
    except ModuleNotFoundError as err:  # pragma: no cover - better error surface
        raise ImportError(
            "scikit-learn is required for precision/recall/F1 metrics. Install with 'pip install scikit-learn'."
        ) from err

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average="macro",
        zero_division=0,
    )
    accuracy = correct / total
    mse = total_mse / total
    return (
        total_loss / total,
        accuracy,
        float(precision),
        float(recall),
        float(f1),
        float(mse),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    train_loader, val_loader = create_dataloaders(config)
    num_classes = len(train_loader.dataset.class_to_idx)

    model = SlowFastAgeModel(config, num_classes=num_classes).to(device)
    if config.checkpoint_path:
        print(f"Loading checkpoint from {config.checkpoint_path}")
        model.load_checkpoint(config.checkpoint_path)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_acc = 0.0
    final_val_metrics: dict[str, float] | None = None
    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch
        )
        print(
            f"Epoch {epoch} completed | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}"
        )

        if val_loader is None:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                config.output_dir / "latest.pt",
            )
            continue

        val_loss, val_acc, val_prec, val_rec, val_f1, val_mse = evaluate(
            model, val_loader, device
        )
        print(
            "Validation | Loss: {:.4f} | Acc: {:.3f} | Prec: {:.3f} | Rec: {:.3f} | F1: {:.3f} | MSE: {:.4f}".format(
                val_loss,
                val_acc,
                val_prec,
                val_rec,
                val_f1,
                val_mse,
            )
        )
        final_val_metrics = {
            "loss": val_loss,
            "accuracy": val_acc,
            "precision": val_prec,
            "recall": val_rec,
            "f1": val_f1,
            "mse": val_mse,
        }

        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        }
        torch.save(checkpoint, config.output_dir / "latest.pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, config.output_dir / "best.pt")
            print(f"New best model saved with acc {best_val_acc:.3f}")

    if final_val_metrics is None:
        if val_loader is None:
            print("Validation set not provided; skipping final metric summary.")
        else:
            print("Validation metrics could not be computed.")
    else:
        print(
            "Final validation metrics -> Acc: {accuracy:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}, F1: {f1:.3f}, MSE: {mse:.4f}".format(
                **final_val_metrics
            )
        )


if __name__ == "__main__":
    main()
