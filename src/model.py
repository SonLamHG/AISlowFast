from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from pytorchvideo.models.hub import slowfast_r50

from .config import TrainConfig


class SlowFastAgeModel(nn.Module):
    """SlowFast backbone adapted for age rating classification."""

    def __init__(
        self,
        config: TrainConfig,
        num_classes: Optional[int] = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = slowfast_r50(pretrained=pretrained)
        in_features = self.model.blocks[-1].proj.in_features
        head_classes = num_classes if num_classes is not None else config.num_classes
        if head_classes <= 0:
            raise ValueError("Number of classes must be greater than zero")
        self.model.blocks[-1].proj = nn.Linear(in_features, head_classes)

    def forward(self, inputs: Any) -> torch.Tensor:  # type: ignore[override]
        return self.model(inputs)

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, torch.Tensor]:
        state = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = self.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=False)
        if missing or unexpected:
            details: Dict[str, Any] = {"missing": missing, "unexpected": unexpected}
            raise RuntimeError(f"Checkpoint mismatch: {details}")
        return state

    def freeze_backbone(self, except_head: bool = True) -> None:
        for name, param in self.model.named_parameters():
            if except_head and name.startswith("blocks.5"):  # head block
                continue
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)
