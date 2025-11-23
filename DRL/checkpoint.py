from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class CheckpointInfo:
    path: Path
    iteration: int
    score: float


class CheckpointManager:
    def __init__(self, directory: str, max_keep: int = 5) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep

    def list_checkpoints(self) -> list[CheckpointInfo]:
        infos = []
        for file in self.directory.glob("checkpoint_iter*_timestamp.pt"):
            parts = file.stem.split("_")
            iteration = 0
            score = 0.0
            for idx, token in enumerate(parts):
                if token.startswith("iter"):
                    iteration = int(token.replace("iter", ""))
                if token.startswith("score"):
                    try:
                        score = float(token.replace("score", ""))
                    except ValueError:
                        score = 0.0
            infos.append(CheckpointInfo(file, iteration, score))
        infos.sort(key=lambda x: x.iteration)
        return infos

    def latest_checkpoint(self) -> Optional[Path]:
        infos = self.list_checkpoints()
        return infos[-1].path if infos else None

    def save(self, state: Dict, iteration: int, score: float = 0.0) -> Path:
        timestamp = int(time.time())
        filename = f"checkpoint_iter{iteration:06d}_score{score:.3f}_{timestamp}_timestamp.pt"
        target_path = self.directory / filename
        tmp_path = target_path.with_suffix(".tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, target_path)
        self._enforce_retention()
        return target_path

    def load(self, path: Optional[str] = None) -> Dict:
        ckpt_path = Path(path) if path else self.latest_checkpoint()
        if ckpt_path is None:
            raise FileNotFoundError("No checkpoint available")
        return _safe_torch_load(ckpt_path, map_location="cpu")

    def _enforce_retention(self) -> None:
        infos = self.list_checkpoints()
        if len(infos) <= self.max_keep:
            return
        to_remove = infos[: len(infos) - self.max_keep]
        for info in to_remove:
            info.path.unlink(missing_ok=True)


def capture_rng_state() -> Dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict) -> None:
    if not state:
        return
    random.setstate(state.get("python"))
    np.random.set_state(state.get("numpy"))
    torch.set_rng_state(state.get("torch"))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _safe_torch_load(path: Path, map_location: str = "cpu") -> Dict:
    """Load checkpoints compatibly across torch versions."""
    load_kwargs = {"map_location": map_location}
    try:
        # torch>=2.6 defaults weights_only=True; force False for full checkpoints
        return torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)
