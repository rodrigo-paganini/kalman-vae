"""Small data utilities for KVAE experiments.

Includes a toy dataset generator useful for smoke tests and development.
"""
from typing import Optional

import torch
from torch.utils.data import TensorDataset


def make_toy_dataset(num_seq: int = 200,
					 T: int = 10,
					 C: int = 1,
					 H: int = 32,
					 W: int = 32,
					 seed: Optional[int] = None) -> TensorDataset:
	"""Create a simple random toy dataset of shape (num_seq, T, C, H, W).

	Returns a TensorDataset where each item is a single-tensor tuple (x,) and
	x has shape [T, C, H, W]. This is convenient to pass directly to a
	DataLoader which will produce batches of shape [B, T, C, H, W].

	Args:
		num_seq: number of sequences
		T: time-steps per sequence
		C: channels
		H: height
		W: width
		seed: optional RNG seed for reproducibility

	Returns:
		TensorDataset
	"""
	if seed is not None:
		rng = torch.manual_seed(seed)

	x = torch.randn(num_seq, T, C, H, W)
	return TensorDataset(x)
