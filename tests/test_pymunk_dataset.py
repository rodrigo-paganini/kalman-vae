import json
import numpy as np
import torch
from pathlib import Path

from kvae.data_loading.pymunk_dataset import PymunkNPZDataset


def make_npz(path: Path, N=5, T=20, H=32, W=32, D=4):
    images = np.random.randint(0, 256, size=(N, T, H, W), dtype=np.uint8)
    states = np.random.randn(N, T, D).astype(np.float32)
    np.savez_compressed(path, images=images, state=states)


def test_pymunk_npz_dataset(tmp_path):
    npz_path = tmp_path / 'data.npz'
    make_npz(npz_path)

    ds = PymunkNPZDataset.from_npz(str(npz_path), seq_len=20)
    assert len(ds) == 5

    item = ds[0]
    assert 'images' in item
    assert 'state' in item
    imgs = item['images']
    st = item['state']
    assert isinstance(imgs, torch.Tensor)
    assert isinstance(st, torch.Tensor)
    # imgs returned as (T, C, H, W)
    assert imgs.shape[0] == 20
    assert imgs.shape[2] == 32 and imgs.shape[3] == 32
    # state shape (T, D)
    assert st.shape[0] == 20
    assert st.shape[1] == 4
