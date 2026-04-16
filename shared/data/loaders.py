"""
Dataset loading and stratified split utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pair_generator import ContrastivePair


def load_jsonl(path: str | Path) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_dataset_splits(
    dataset_dir: str | Path,
    train_file: str = "train.json",
    val_file: str = "validation.json",
    test_file: str = "test.json",
) -> Dict[str, List[Dict]]:
    """Load train/val/test splits from a HuggingFace-style directory."""
    dataset_dir = Path(dataset_dir)
    splits = {}
    for name, fname in [("train", train_file), ("val", val_file), ("test", test_file)]:
        fpath = dataset_dir / fname
        if fpath.exists():
            splits[name] = load_jsonl(fpath)
    return splits


def stratified_split(
    pairs: List[ContrastivePair],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ContrastivePair], List[ContrastivePair], List[ContrastivePair]]:
    """Split pairs into train/val/test preserving pair_type distribution."""
    rng = np.random.RandomState(seed)

    by_type: Dict[str, List[ContrastivePair]] = {}
    for p in pairs:
        by_type.setdefault(p.pair_type, []).append(p)

    train, val, test = [], [], []
    for type_pairs in by_type.values():
        indices = rng.permutation(len(type_pairs))
        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)

        for i, idx in enumerate(indices):
            if i < n_train:
                train.append(type_pairs[idx])
            elif i < n_train + n_val:
                val.append(type_pairs[idx])
            else:
                test.append(type_pairs[idx])

    return train, val, test
