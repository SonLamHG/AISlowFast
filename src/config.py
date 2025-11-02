from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainConfig:
    """Configuration container for SlowFast age rating training."""

    train_csv: Path = Path("data/train_metadata.csv")
    video_root: Path = Path("data/videos")
    output_dir: Path = Path("runs/slowfast")
    val_split: float = 0.2  # portion of samples reserved for validation

    fast_num_frames: int = 32
    alpha: int = 4
    clip_duration: float = 2.0  # seconds
    sample_rate: int = 1

    batch_size: int = 4
    num_workers: int = 4

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 15
    warmup_epochs: int = 2

    device: str = "cuda"
    seed: int = 1337

    label_column: str = "label"
    path_column: str = "path"

    class_names: List[str] = field(
        default_factory=lambda: ["U0", "U10", "U14", "U16", "U18"]
    )

    checkpoint_path: Optional[Path] = None

    @property
    def num_classes(self) -> int:
        return len(self.class_names)
