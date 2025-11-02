from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from .config import TrainConfig
from .dataset import AgeRatingDataset
from .model import SlowFastAgeModel


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _prepare_inputs(video_path: Path, config: TrainConfig) -> List[torch.Tensor]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    dataset = AgeRatingDataset(
        csv_path=None,
        config=config,
        is_train=False,
        samples=[(video_path, 0)],
        class_to_idx={name: idx for idx, name in enumerate(config.class_names)},
    )
    slowfast_inputs, _ = dataset[0]
    return slowfast_inputs


def _predict_probabilities(
    video_path: Path,
    checkpoint_path: Path,
    config: TrainConfig | None,
    device: str | torch.device | None,
) -> tuple[TrainConfig, torch.Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    cfg = config or TrainConfig()
    model = SlowFastAgeModel(cfg, num_classes=cfg.num_classes, pretrained=False)
    model.load_checkpoint(checkpoint_path)
    torch_device = _resolve_device(device or cfg.device)
    model = model.to(torch_device).eval()

    inputs = _prepare_inputs(video_path, cfg)
    inputs = [tensor.unsqueeze(0).to(torch_device) for tensor in inputs]

    with torch.inference_mode():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

    return cfg, probs


def predict_age_rating(
    video_path: str | Path,
    checkpoint_path: str | Path,
    config: TrainConfig | None = None,
    device: str | torch.device | None = None,
) -> str:
    """Return the predicted age rating label for ``video_path``."""

    video_path = Path(video_path)
    checkpoint_path = Path(checkpoint_path)
    cfg, probs = _predict_probabilities(video_path, checkpoint_path, config, device)
    top_idx = int(probs.argmax().item())
    return cfg.class_names[top_idx]


def predict_age_rating_with_scores(
    video_path: str | Path,
    checkpoint_path: str | Path,
    config: TrainConfig | None = None,
    device: str | torch.device | None = None,
) -> Dict[str, float]:
    """Return the per-class confidence scores for ``video_path``."""

    video_path = Path(video_path)
    checkpoint_path = Path(checkpoint_path)
    cfg, probs = _predict_probabilities(video_path, checkpoint_path, config, device)
    return {cfg.class_names[idx]: float(probs[idx].item()) for idx in range(cfg.num_classes)}
