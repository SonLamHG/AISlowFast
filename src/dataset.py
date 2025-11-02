from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_video

from .config import TrainConfig


def _uniform_temporal_subsample(
    video: torch.Tensor, num_samples: int
) -> torch.Tensor:
    """Uniformly sample ``num_samples`` frames along the temporal axis."""

    total_frames = video.shape[0]
    if total_frames == num_samples:
        return video

    indices = torch.linspace(0, max(total_frames - 1, 0), num_samples)
    indices = indices.clamp(0, max(total_frames - 1, 0)).long()
    return video.index_select(0, indices)


def _resize_short_side(video: torch.Tensor, min_size: int) -> torch.Tensor:
    """Resize so the shortest spatial side equals ``min_size``."""

    _, _, height, width = video.shape
    short_side = min(height, width)
    if short_side == min_size:
        return video

    scale = min_size / float(short_side)
    new_height = int(round(height * scale))
    new_width = int(round(width * scale))
    video = F.interpolate(
        video,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )
    return video


def _random_crop(video: torch.Tensor, size: int) -> torch.Tensor:
    """Randomly crop the spatial dimension to ``size`` square."""

    _, _, height, width = video.shape
    if height == size and width == size:
        return video

    if height < size or width < size:
        pad_height = max(size - height, 0)
        pad_width = max(size - width, 0)
        video = F.pad(
            video,
            (0, pad_width, 0, pad_height),
            mode="replicate",
        )
        _, _, height, width = video.shape

    top = random.randint(0, height - size)
    left = random.randint(0, width - size)
    return video[:, :, top : top + size, left : left + size]


def _center_crop(video: torch.Tensor, size: int) -> torch.Tensor:
    """Center crop the spatial dimension to ``size`` square."""

    _, _, height, width = video.shape
    if height < size or width < size:
        pad_height = max(size - height, 0)
        pad_width = max(size - width, 0)
        video = F.pad(
            video,
            (0, pad_width, 0, pad_height),
            mode="replicate",
        )
        _, _, height, width = video.shape

    top = (height - size) // 2
    left = (width - size) // 2
    return video[:, :, top : top + size, left : left + size]


@dataclass
class _SampledClip:
    frames: torch.Tensor
    fps: float


class AgeRatingDataset(Dataset[Tuple[List[torch.Tensor], int]]):
    """Dataset for SlowFast-based movie age rating classification."""

    def __init__(
        self,
        csv_path: Optional[Path],
        config: TrainConfig,
        is_train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        samples: Optional[List[Tuple[Path, int]]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.config = config
        self.is_train = is_train
        self.transform = transform

        self.class_to_idx: Dict[str, int] = class_to_idx or {
            name: idx for idx, name in enumerate(config.class_names)
        }

        if samples is not None:
            self.samples = list(samples)
        else:
            if self.csv_path is None or not self.csv_path.exists():
                raise FileNotFoundError(
                    f"Could not find metadata CSV at {self.csv_path or config.train_csv}"
                )

            self.samples: List[Tuple[Path, int]] = []
            with self.csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    raw_label = row.get(config.label_column)
                    raw_path = row.get(config.path_column)
                    if raw_label is None or raw_path is None:
                        continue

                    label_idx = self._encode_label(raw_label)
                    video_path = Path(raw_path)
                    if not video_path.is_absolute():
                        video_path = config.video_root / video_path

                    if not video_path.exists():
                        continue
                    self.samples.append((video_path, label_idx))
        # Ensure internal list exists even if CSV empty
        if not hasattr(self, "samples"):
            self.samples = []

        if not self.samples:
            source = self.csv_path or config.train_csv
            raise RuntimeError(
                f"No valid samples found in {source}. Check paths and labels."
            )
        self._mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        self._std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)

    def _encode_label(self, label: str) -> int:
        if label in self.class_to_idx:
            return self.class_to_idx[label]
        try:
            numeric = int(label)
            if 0 <= numeric < len(self.config.class_names):
                return numeric
        except ValueError:
            pass
        source = self.csv_path or self.config.train_csv
        raise ValueError(f"Unknown label '{label}' in {source}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], int]:
        video_path, label = self.samples[index]
        clip = self._load_clip(video_path)
        processed = self._preprocess_clip(clip)
        slowfast_inputs = self._pack_pathways(processed.frames)
        return slowfast_inputs, label

    def _load_clip(self, video_path: Path) -> _SampledClip:
        try:
            video, _, info = read_video(
                str(video_path),
                pts_unit="sec",
            )
        except RuntimeError as err:
            raise RuntimeError(f"Failed to decode video {video_path}: {err}") from err

        fps = float(info.get("video_fps", 30.0))
        total_frames = video.shape[0]
        if total_frames == 0:
            raise RuntimeError(f"Decoded zero frames from {video_path}")
        clip_length = max(int(round(self.config.clip_duration * fps)), 1)
        if total_frames <= clip_length:
            padded = clip_length - total_frames
            pads = video[-1:].repeat(padded, 1, 1, 1)
            video = torch.cat([video, pads], dim=0)
            total_frames = video.shape[0]

        max_start = max(total_frames - clip_length, 0)
        if self.is_train:
            start_frame = random.randint(0, max_start)
        else:
            start_frame = max_start // 2
        end_frame = start_frame + clip_length
        clip_frames = video[start_frame:end_frame]
        return _SampledClip(frames=clip_frames, fps=fps)

    def _preprocess_clip(self, clip: _SampledClip) -> _SampledClip:
        video = clip.frames.float() / 255.0
        video = video.permute(0, 3, 1, 2)  # T, C, H, W
        video = video.permute(1, 0, 2, 3)  # C, T, H, W for interpolation
        video = _resize_short_side(video, 256)

        if self.is_train:
            video = _random_crop(video, 224)
            if random.random() < 0.5:
                video = video.flip(-1)
        else:
            video = _center_crop(video, 224)

        video = video.permute(1, 0, 2, 3)  # T, C, H, W
        if self.transform is not None:
            video = self.transform(video)
        video = video.permute(1, 0, 2, 3)  # C, T, H, W
        video = (video - self._mean) / self._std
        return _SampledClip(frames=video, fps=clip.fps)

    def _pack_pathways(self, video: torch.Tensor) -> List[torch.Tensor]:
        video_t_first = video.permute(1, 0, 2, 3)
        fast_pathway = _uniform_temporal_subsample(
            video_t_first, self.config.fast_num_frames
        ).permute(1, 0, 2, 3)
        slow_frames = max(self.config.fast_num_frames // self.config.alpha, 1)
        slow_pathway = _uniform_temporal_subsample(video_t_first, slow_frames).permute(1, 0, 2, 3)
        return [slow_pathway, fast_pathway]
