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

from kvae.dataloader.base import BaseDataset


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
            # Better heuristic:
            # - If the last two dims look like image HxW (>=8) treat as temporal axis T -> (N,T,H,W)
            # - Else if D1 looks like channels (1 or 3) treat as frames (F,C,H,W)
            # - Fallback: treat as frames without explicit channel -> add channel axis
            if D2 >= 8 and D3 >= 8:
                # (N, T, H, W) -> add channel dim
                self.seq_data = imgs[:, :, None, :, :]
            elif D1 in (1, 3) and D2 >= 8 and D3 >= 8:
                # ambiguous but likely (F, C, H, W)
                frames = imgs
                self._build_from_frames(frames)
            else:
                # assume (F, H, W) -> add channel and build sliding windows
                frames = imgs[:, None, :, :]
                self._build_from_frames(frames)
        elif imgs.ndim == 3:
            # (F, H, W) -> add channel
            frames = imgs[:, None, :, :]
            self._build_from_frames(frames)
        else:
            raise ValueError(f"Unsupported image array shape: {imgs.shape}")

        # normalize seq_data shape into (N, T, C, H, W)
        shape = self.seq_data.shape
        if len(shape) == 5:
            self.N, self.T, self.C, self.H, self.W = shape
        elif len(shape) == 4:
            # (N, T, H, W) -> assume single channel
            self.N, self.T, self.H, self.W = shape
            self.C = 1
            # reshape to (N, T, C, H, W)
            self.seq_data = self.seq_data.reshape(self.N, self.T, self.C, self.H, self.W)
        elif len(shape) > 5:
            # Unexpected extra dims: collapse middle dims into channel dim.
            # e.g., (N, T, d1, d2, ..., H, W) -> C = prod(d1..dK)
            self.N = shape[0]
            self.T = shape[1]
            self.H = shape[-2]
            self.W = shape[-1]
            mid = shape[2:-2]
            C = 1
            for d in mid:
                C *= int(d)
            self.C = C
            # reshape by collapsing middle dims into channel
            self.seq_data = self.seq_data.reshape(self.N, self.T, self.C, self.H, self.W)
        else:
            raise ValueError(f'Unexpected sequence data shape: {shape}')

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
        # frames: expected (F, C, H, W) but be robust to extra middle dims
        F = frames.shape[0]
        if F < self.seq_len:
            raise ValueError(f'Not enough frames ({F}) for seq_len={self.seq_len}')

        # Ensure frames have shape (F, C, H, W). If frames have extra middle dims
        # (e.g. (F, d1, d2, H, W)), collapse them into a single channel dimension.
        frame_shape = frames.shape[1:]
        if len(frame_shape) == 3:
            C, H, W = frame_shape
        elif len(frame_shape) > 3:
            # collapse middle dims into channel
            mid = frame_shape[:-2]
            H = frame_shape[-2]
            W = frame_shape[-1]
            C = 1
            for d in mid:
                C *= int(d)
            frames = frames.reshape(F, C, H, W)
        else:
            raise ValueError(f'Unsupported frame shape: {frames.shape}')

        seqs = []
        for start in range(0, F - self.seq_len + 1, self.stride):
            seq = frames[start: start + self.seq_len]  # (T, C, H, W)
            seqs.append(seq)
        seqs = np.stack(seqs, axis=0)  # (N, T, C, H, W)
        self.seq_data = seqs

        # Now shape should be (N, T, C, H, W)
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
