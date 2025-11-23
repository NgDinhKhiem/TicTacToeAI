import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
yaml = None

try:
    import yaml as _yaml
    yaml = _yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to run this project") from exc


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config from disk."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    return data


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_pref: Optional[str] = None, max_vram_gb: Optional[float] = None) -> torch.device:
    pref = (device_pref or "auto").lower()
    device = torch.device("cpu")

    def _limit_cuda_memory(dev: torch.device) -> None:
        if max_vram_gb is None or max_vram_gb <= 0 or dev.type != "cuda" or not torch.cuda.is_available():
            return
        try:
            device_index = dev.index if dev.index is not None else torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device_index).total_memory
            bytes_limit = max_vram_gb * (1024**3)
            fraction = max(0.05, min(1.0, bytes_limit / total_mem))
            torch.cuda.set_per_process_memory_fraction(float(fraction), device=dev)
        except Exception:
            pass

    if pref == "cpu":
        return device

    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return device

    if pref in ("cuda", "gpu"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            _limit_cuda_memory(device)
            return device
        print("CUDA requested but unavailable; falling back to CPU.")
        return device

    # auto or unknown preference
    if torch.cuda.is_available():
        device = torch.device("cuda")
        _limit_cuda_memory(device)
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return device


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply mask to logits and return a valid probability distribution."""
    mask = mask.to(dtype=logits.dtype)
    masked_logits = logits + (mask + 1e-45).log()  # -inf where mask==0
    return torch.nn.functional.softmax(masked_logits, dim=-1)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)
