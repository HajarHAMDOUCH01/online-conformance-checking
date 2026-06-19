"""
dataset_utils.py
For trying offline reinforcement learning model later
Helper for persisting offline-RL transition records collected from
AlignmentEnv rollouts.
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch


def _to_native(obj):
    if isinstance(obj, torch.Tensor):
        return _to_native(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def save_episode_transitions(path: str, episode_transitions: list[dict]) -> None:
    """
    Append a list of transition dicts to a JSONL file, one JSON object
    per line. Safe to call once per episode rather than
    once per step, to minimize file-open overhead across millions of
    transitions.
    """
    if not episode_transitions:
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "a", buffering=1024 * 1024) as f:
        for transition in episode_transitions:
            clean = _to_native(transition)
            f.write(json.dumps(clean, separators=(",", ":")))
            f.write("\n")