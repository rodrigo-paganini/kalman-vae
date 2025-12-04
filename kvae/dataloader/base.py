"""Dataset utilities for loading pymunk-generated `.npz` datasets.

This module provides a small, modular hierarchy:
- BaseDataset: abstract convenience wrapper for torch datasets
"""
from torch.utils.data import Dataset
from typing import Any, Dict, List

class BaseDataset(Dataset):
    """Minimal base dataset exposing load / preprocess hooks.

    Subclasses should override `load()` to populate `self.raw` and
    `build_index()` to construct indexing for `__len__` and `__getitem__`.
    """

    def __init__(self):
        self.raw: Dict[str, Any] = {}
        self.index: List[Any] = []

    def load(self, *args, **kwargs):
        raise NotImplementedError()

    def build_index(self):
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx):
        raise NotImplementedError()