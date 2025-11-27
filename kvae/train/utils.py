import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
import os
import yaml
import numpy as np
import random
import matplotlib.pyplot as plt

from kvae.dataloader.pymunk_dataset import PymunkNPZDataset
from kvae.vae.train_vae import TrainingConfig


def parse_device(device_pref: str):
    mps_available = getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()

    if device_pref == 'auto':
        if cuda_available:
            device_pref = 'cuda'
        elif mps_available:
            device_pref = 'mps'
        else:
            device_pref = 'cpu'
    elif device_pref == 'cuda' and not cuda_available:
        print("CUDA not available, switching to CPU.")
        device_pref = 'cpu'
    elif device_pref == 'mps' and not mps_available:
        print("MPS not available, switching to CPU.")
        device_pref = 'cpu'

    return torch.device(device_pref)
    


def seed_all_modules(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_config() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='kvae/train/config.yaml', help='Path to YAML/JSON config file')
    args = p.parse_args()

    config_path = args.config
    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


def load_config(path):
    if path is None: return {}
    p = Path(path)
    if not p.exists(): return {}
    return yaml.safe_load(p.read_text()) or {}


def build_pymunk_dataset(path: str, T: int):
    dataset = PymunkNPZDataset.from_npz(
        path,
        seq_len=T,
        load_in_memory=True,
        normalize=False,
    )
    return dataset

def build_dataloaders(ds_path, batch_size, T):
    ds_path = str(Path(ds_path).resolve()) 
    dataset = PymunkNPZDataset.from_npz(ds_path, seq_len=T, load_in_memory=True, normalize=False)

    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,         
        pin_memory=True,
        persistent_workers=False, 
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
    )
    return train_loader, val_loader

def build_dataloaders_refactored(
        dataset_cfg: dict,
        batch_size: int,
    ):
    if dataset_cfg:
        ds_type = dataset_cfg.get('type', 'toy')
        ds_path = dataset_cfg.get('path')
        ds_kwargs = dataset_cfg.get('kwargs', {})
    else:
        raise ValueError("Dataset configuration is required")

    if ds_type == 'pymunk':
        ds_path = str(Path(ds_path).resolve()) 
        dataset = PymunkNPZDataset.from_npz(
            ds_path,
            **ds_kwargs
        )
    else:
        raise NotImplementedError(f"Unsupported dataset type: {ds_type}")

    n_val = max(1, int(dataset_cfg.get('val_split') * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataset_cfg.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=False, 
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataset_cfg.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=False,
    )
    return train_loader, val_loader


def save_checkpoint(path, model, optimizer, epoch, train_loss, val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(payload, path)