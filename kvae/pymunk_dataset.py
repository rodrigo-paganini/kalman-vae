"""Dataset utilities for loading pymunk-generated `.npz` datasets.

This module provides a small, modular hierarchy:
- BaseDataset: abstract convenience wrapper for torch datasets
- PymunkBaseDataset: handles loading `.npz` files created by pymunk pipelines
- PymunkNPZDataset: concrete dataset that yields image sequences (T, C, H, W)

The concrete class is intentionally flexible to support several common
serialization layouts produced by simulation toolchains.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


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


class PymunkBaseDataset(BaseDataset):
    """Base class for datasets stored as `.npz` files by pymunk pipelines.

    This class handles loading the `.npz` archive and exposes `self.raw`
    (a dict-like mapping of arrays). Subclasses decide which keys to use.
    """

    def __init__(self, npz_path: str | Path, load_in_memory: bool = True):
        super().__init__()
        self.path = Path(npz_path)
        self.load_in_memory = load_in_memory
        self.npz = None
        self.load()

    def load(self):
        if not self.path.exists():
            raise FileNotFoundError(f"NPZ file not found: {self.path}")
        # Use numpy.load; allow_pickle to be safe for object arrays
        self.npz = np.load(self.path, allow_pickle=True)
        # copy into raw if requested
        if self.load_in_memory:
            for k in self.npz.files:
                self.raw[k] = self.npz[k]
        else:
            # keep the NpzFile-like object for lazy access
            self.raw = {k: self.npz[k] for k in self.npz.files}


class PymunkNPZDataset(PymunkBaseDataset):
    """Concrete dataset that returns image sequences from a `.npz` file.

    The dataset supports several input formats produced by simulation/tooling:
    - sequences stored as (N, T, C, H, W)
    - sequences stored as (N, T, H, W)  (assumes C=1)
    - flat frames stored as (F, C, H, W) or (F, H, W)  -> sliding windows built

    By default the dataset looks for an array under the key `images` but you
    can override the key. Optionally additional arrays (e.g., `states`) are
    returned alongside the image sequence as a dict under the `meta` key.
    """

    def __init__(self,
                 npz_path: str | Path,
                 image_key: str = 'images',
                 state_key: Optional[str] = 'state',
                 seq_len: int = 10,
                 stride: int = 1,
                 normalize: bool = True,
                 load_in_memory: bool = True):
        super().__init__(npz_path, load_in_memory=load_in_memory)
        self.image_key = image_key
        self.state_key = state_key
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.normalize = bool(normalize)

        # after loading, build internal index
        self._prepare()

    def _prepare(self):
        if self.image_key not in self.raw:
            raise KeyError(f"Image key '{self.image_key}' not in NPZ. Available: {list(self.raw.keys())}")

        imgs = self.raw[self.image_key]
        imgs = np.asarray(imgs)

        # standardize shape to (N, T, C, H, W) or flatten frames -> we'll handle
        if imgs.ndim == 5:
            # (N, T, C, H, W)
            self.seq_data = imgs
        elif imgs.ndim == 4:
            # ambiguous: could be (N, T, H, W) or (F, C, H, W)
            N, D1, D2, D3 = imgs.shape
            # Heuristic: if D1 equals seq_len or D1 > 1 and D2 small -> treat as (N,T,H,W)
            if D1 == self.seq_len:
                # (N, T, H, W) -> add channel dim
                self.seq_data = imgs[:, :, None, :, :]
            elif N == self.seq_len:
                # (T, C, H, W) for single sequence -> wrap
                self.seq_data = imgs[None, :, :, :, :]
            else:
                # treat as flat frames (F, H, W) or (F, C, H, W) -> build sliding windows
                # ensure channel dim
                if D1 in (1, 3):
                    # ambiguous but assume (F, C, H, W)
                    frames = imgs
                else:
                    # assume (F, H, W)
                    frames = imgs[:, None, :, :]
                self._build_from_frames(frames)
                return
        elif imgs.ndim == 3:
            # (F, H, W) -> add channel
            frames = imgs[:, None, :, :]
            self._build_from_frames(frames)
            return
        else:
            raise ValueError(f"Unsupported image array shape: {imgs.shape}")

        # if we have seq_data already as (N,T,C,H,W), build index trivially
        self.N, self.T, self.C, self.H, self.W = self.seq_data.shape

        # load states if available and aligned
        self.state_data = None
        if self.state_key is not None and self.state_key in self.raw:
            states = np.asarray(self.raw[self.state_key])
            # expected shape (N, T, D)
            if states.ndim != 3:
                raise ValueError(f"Expected state array of shape (N,T,D), got {states.shape}")
            if states.shape[0] != self.N or states.shape[1] != self.T:
                raise ValueError(f"State array shape {states.shape} does not match images {(self.N, self.T)}")
            self.state_data = states.astype(np.float32)

        # Build index as simple list of sequence indices
        self.index = list(range(self.N))

    def _build_from_frames(self, frames: np.ndarray):
        # frames: (F, C, H, W)
        F = frames.shape[0]
        if F < self.seq_len:
            raise ValueError(f'Not enough frames ({F}) for seq_len={self.seq_len}')
        seqs = []
        for start in range(0, F - self.seq_len + 1, self.stride):
            seq = frames[start: start + self.seq_len]  # (T, C, H, W)
            seqs.append(seq)
        seqs = np.stack(seqs, axis=0)  # (N, T, C, H, W)
        self.seq_data = seqs
        self.N, self.T, self.C, self.H, self.W = self.seq_data.shape
        self.index = list(range(self.N))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # return dict with 'images': Tensor[T,C,H,W] and optional meta
        if isinstance(idx, slice):
            raise NotImplementedError('Slicing not implemented')

        seq = self.seq_data[self.index[idx]]  # numpy array
        # convert to float32 and optionally normalize to [0,1]
        seq = seq.astype(np.float32)
        if self.normalize:
            # per-frame normalization
            seq = seq - seq.min(axis=(2, 3), keepdims=True)
            denom = seq.max(axis=(2, 3), keepdims=True)
            denom[denom == 0] = 1.0
            seq = seq / denom

        # convert to torch tensor with shape (T, C, H, W)
        seq_t = torch.from_numpy(seq)

        out: Dict[str, Any] = {'images': seq_t}
        if self.state_data is not None:
            state = self.state_data[self.index[idx]]  # (T, D)
            state_t = torch.from_numpy(state.astype(np.float32))
            out['state'] = state_t

        return out

    @classmethod
    def from_npz(cls, npz_path: str | Path, **kwargs) -> 'PymunkNPZDataset':
        """Convenience constructor."""
        return cls(npz_path, **kwargs)
