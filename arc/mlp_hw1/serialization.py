from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_checkpoint(path: str | Path, state_dict: dict[str, np.ndarray], metadata: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: value.astype(np.float32) for name, value in state_dict.items()}
    payload["metadata_json"] = np.array(json.dumps(metadata), dtype=object)
    np.savez_compressed(path, **payload)
    return path


def load_checkpoint(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    archive = np.load(Path(path), allow_pickle=True)
    metadata = json.loads(archive["metadata_json"].item())
    state_dict = {name: archive[name] for name in archive.files if name != "metadata_json"}
    return state_dict, metadata
