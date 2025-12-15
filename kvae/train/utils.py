import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
import os
import yaml
import numpy as np
import random
import matplotlib.pyplot as plt
import datetime
import logging

from kvae.dataloader.pymunk_dataset import PymunkNPZDataset


logger = logging.getLogger(__name__)


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
        logger.warning("CUDA not available, switching to CPU.")
        device_pref = 'cpu'
    elif device_pref == 'mps' and not mps_available:
        logger.warning("MPS not available, switching to CPU.")
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


def build_dataloaders(
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


def create_runs_dir(logdir: str) -> Path:
    if logdir is None:
        return None
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    ckpt_root = os.path.join(logdir, ts)
    os.makedirs(ckpt_root)

    return Path(ckpt_root)


def plot_state_probabilities(state_probs):
    """
    Create a heatmap figure of regime probabilities over time for a single sequence.
    """
    if state_probs is None:
        return None
    if isinstance(state_probs, list):
        state_probs = torch.stack(state_probs, dim=1)
    if state_probs.dim() == 3:
        state_probs = state_probs[0]  # take first sequence in batch
    if state_probs.dim() == 1:
        state_probs = state_probs.unsqueeze(0)

    state_np = state_probs.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(
        state_np.T,
        aspect='auto',
        origin='lower',
        interpolation='nearest',
        vmin=0.0,
        vmax=1.0,
        cmap='magma',
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("State")
    ax.set_title("Switch state")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("probability")
    fig.tight_layout()
    return fig


class Checkpointer:
    def __init__(
        self,
        checkpoint_dir: Path,
        ckpt_every: int
        ):
        self.checkpoint_dir = checkpoint_dir
        self.ckpt_every = ckpt_every
        self.best_val = float("inf")
        self.checkpoint_dir.mkdir(parents=True)
        logger.info(f"\nCheckpoints will be saved to: {self.checkpoint_dir}\n")

    @staticmethod
    def payload(model, optimizer, epoch, train_loss, val_loss):
        return {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

    def save_checkpoints(
        self,
        train_loss,
        val_loss,
        model,
        optimizer,
        epoch,
        ):
        best_path = self.checkpoint_dir / "kvae-best.pt"

        if val_loss < self.best_val:
            self.best_val = val_loss
            self.save_checkpoint(best_path, model, optimizer, epoch, train_loss, val_loss)

        if self.ckpt_every > 0 and epoch % self.ckpt_every == 0:
            ckpt_path = self.checkpoint_dir / f"kvae-ckpt-epoch={epoch:03d}.pt"
            self.save_checkpoint(ckpt_path, model, optimizer, epoch, train_loss, val_loss)

    def save_checkpoint(self, path, model, optimizer, epoch, train_loss, val_loss):
        torch.save(
            self.payload(model, optimizer, epoch, train_loss, val_loss),
            path
        )
        logger.info(f"Saved checkpoint at epoch {epoch} to {path}")