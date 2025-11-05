"""PyTorch Lightning training script for KVAE.

Provides:
- KVAELit: LightningModule wrapping the KVAE model and loss
- KVAEDataModule: DataModule using the toy dataset for quick experiments
- CLI entrypoint that launches a Lightning Trainer with TensorBoard logging

Usage:
    pip install -e '.[dev]'
    python train_lightning.py --max-epochs 10 --batch-size 32

"""
from __future__ import annotations

import argparse
import json
import yaml
import os
import shutil
from typing import Optional, Callable

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from kvae.utils import KVAEConfig
from kvae.model import KVAE
from kvae.dataloader import make_toy_dataset
from kvae.pymunk_dataset import PymunkNPZDataset


class KVAELit(pl.LightningModule):
    """LightningModule wrapper for KVAE."""

    def __init__(self, cfg: KVAEConfig, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters({'lr': lr})
        self.cfg = cfg
        self.model = KVAE(cfg)
        self.lr = lr

        # placeholders for validation-image logging
        self._val_orig = None
        self._val_recon = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        out = self.model(x)
        losses = self.model.compute_loss(x, out)
        self.log('train/total_loss', losses['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/recon_loss', losses['recon_loss'], on_step=True, on_epoch=True)
        self.log('train/kl_loss', losses['kl_loss'], on_step=True, on_epoch=True)
        return losses['total_loss']

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        out = self.model(x)
        losses = self.model.compute_loss(x, out)
        self.log('val/total_loss', losses['total_loss'], on_step=False, on_epoch=True, prog_bar=True)

        # keep first batch's first sequence for image logging at epoch end
        if batch_idx == 0:
            self._val_orig = x[:1].detach().cpu()
            self._val_recon = out['x_recon'][:1].detach().cpu()

        return losses['total_loss']

    def training_epoch_end(self, outputs):
        # outputs is a list of tensors (losses) returned by training_step
        try:
            vals = torch.stack([o for o in outputs])
            mean = float(vals.mean().detach().cpu().item())
        except Exception:
            # fallback: convert items to float
            vals = [float(o) for o in outputs]
            mean = sum(vals) / max(1, len(vals))
        import logging
        logging.getLogger('kvae.train').info(f'Epoch {self.current_epoch} train_loss={mean:.6f}')

    def validation_epoch_end(self, outputs):
        # outputs is a list of tensors returned by validation_step
        try:
            vals = torch.stack([o for o in outputs])
            mean = float(vals.mean().detach().cpu().item())
        except Exception:
            vals = [float(o) for o in outputs if o is not None]
            mean = sum(vals) / max(1, len(vals)) if vals else 0.0
        import logging
        logging.getLogger('kvae.train').info(f'Epoch {self.current_epoch} val_loss={mean:.6f}')

    def on_validation_epoch_end(self):
        # log images to TensorBoard if logger supports experiment
        if self._val_orig is None or self._val_recon is None:
            return
        logger = self.logger
        # pytorch_lightning logger exposes experiment (tensorboard SummaryWriter)
        if hasattr(logger, 'experiment'):
            tb = logger.experiment
            # self._val_orig shape: [B=1, T, C, H, W] -> remove batch dim and log frames as images
            orig = self._val_orig.squeeze(0)  # [T, C, H, W]
            recon = self._val_recon.squeeze(0)
            # normalize per-frame
            def norm(imgs: torch.Tensor) -> torch.Tensor:
                imgs = imgs.clone()
                imgs -= imgs.view(imgs.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
                m = imgs.view(imgs.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
                m[m == 0] = 1.0
                return imgs / m

            orig_n = norm(orig)
            recon_n = norm(recon)
            tb.add_images('val/orig', orig_n, self.current_epoch)
            tb.add_images('val/recon', recon_n, self.current_epoch)

        # drop references
        self._val_orig = None
        self._val_recon = None

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt


class KVAEDataModule(pl.LightningDataModule):
    def __init__(self, cfg: KVAEConfig, batch_size: int = 16, num_seq: int = 200, T: int = 10,
                 dataset_type: str = 'toy', dataset_path: Optional[str] = None,
                 dataset_kwargs: Optional[dict] = None, transform_fn: Optional[Callable] = None):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_seq = num_seq
        self.T = T
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        self.dataset_kwargs = dataset_kwargs or {}
        self.transform_fn = transform_fn
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: Optional[str] = None):
        if self.dataset_type == 'toy':
            ds = make_toy_dataset(num_seq=self.num_seq, T=self.T, C=self.cfg.img_channels,
                                  H=self.cfg.img_size, W=self.cfg.img_size)
        elif self.dataset_type == 'pymunk':
            if not self.dataset_path:
                raise ValueError('dataset_path must be set for pymunk dataset')
            ds = PymunkNPZDataset.from_npz(self.dataset_path, seq_len=self.T, **self.dataset_kwargs)
        else:
            raise ValueError(f'Unknown dataset_type: {self.dataset_type}')

        # Optionally wrap with transform
        if self.transform_fn is not None:
            ds = TransformDataset(ds, self.transform_fn)

        n_val = max(1, int(0.2 * len(ds)))
        n_train = len(ds) - n_val
        self.train_ds, self.val_ds = random_split(ds, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)


class TransformDataset(torch.utils.data.Dataset):
    """Wraps an existing dataset and applies a transform function to each item.

    The transform function should accept and return a dict-like item.
    """
    def __init__(self, base_ds, transform_fn: Callable):
        self.base = base_ds
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        return self.transform_fn(item)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--logdir', type=str, default='logs')
    p.add_argument('--config', type=str, default='config.yaml', help='Path to YAML/JSON config file (defaults to ./config.yaml if present)')
    p.add_argument('--num-seq', type=int, default=200)
    p.add_argument('--T', type=int, default=10)
    p.add_argument('--gpus', type=int, default=0)
    p.add_argument('--ckpt-every', type=int, default=5)
    args = p.parse_args()

    # If a JSON config is provided, load it and override/extend CLI args
    config_path = args.config
    config = None
    if config_path and os.path.exists(config_path):
        # support YAML or JSON configs
        with open(config_path, 'r') as f:
            if config_path.lower().endswith(('.yml', '.yaml')):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

    # build components according to config
    cfg = KVAEConfig()

    # determine dataset settings
    dataset_cfg = config.get('dataset') if config else None
    if dataset_cfg:
        ds_type = dataset_cfg.get('type', 'toy')
        ds_path = dataset_cfg.get('path')
        ds_kwargs = dataset_cfg.get('kwargs', {})
        transforms_cfg = config.get('transforms', {}) if config else {}
    else:
        ds_type = 'toy'
        ds_path = None
        ds_kwargs = {}
        transforms_cfg = {}

    # build optional transform function
    def build_transform(tcfg: dict):
        # simple transform builder: supports add_noise (std)
        noise_std = float(tcfg.get('add_noise_std', 0.0)) if tcfg else 0.0
        def transform(item: dict):
            # item expected to be {'images': Tensor[T,C,H,W], ...}
            imgs = item['images'].float()
            if noise_std > 0:
                imgs = imgs + torch.randn_like(imgs) * noise_std
            item['images'] = imgs
            return item
        return transform if noise_std > 0 else None

    transform_fn = build_transform(transforms_cfg)

    lit = KVAELit(cfg, lr=args.lr)
    dm = KVAEDataModule(cfg, batch_size=args.batch_size, num_seq=args.num_seq, T=args.T,
                        dataset_type=ds_type, dataset_path=ds_path, dataset_kwargs=ds_kwargs,
                        transform_fn=transform_fn)

    # create logger and copy config into the run dir for reproducibility
    logger = TensorBoardLogger(save_dir=args.logdir, name='kvae_lightning')
    if config_path:
        # ensure logger has been initialized and has a log_dir
        run_logdir = logger.log_dir
        os.makedirs(run_logdir, exist_ok=True)
        shutil.copy(config_path, os.path.join(run_logdir, 'config.json'))

    # Configure python logging to write into the run directory as well as stdout
    import logging
    run_logdir = logger.log_dir
    os.makedirs(run_logdir, exist_ok=True)
    log_file = os.path.join(run_logdir, 'train.log')
    root = logging.getLogger()
    # reset handlers to avoid duplicate logs in repeated runs
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    ckpt_callback = ModelCheckpoint(dirpath=os.path.join(args.logdir, 'checkpoints'), filename='ckpt-{epoch}', every_n_epochs=args.ckpt_every)

    # decide accelerator: support cuda, mps (Apple), or cpu. Allow override from config.training.device
    training_cfg = config.get('training', {}) if config else {}
    device_pref = training_cfg.get('device', 'auto')

    trainer_kwargs = dict(max_epochs=args.max_epochs, logger=logger, callbacks=[ckpt_callback], log_every_n_steps=10)

    # helper to set accelerator/devices
    def set_accelerator_for(trainer_kwargs: dict, device_pref: str):
        pref = (device_pref or 'auto').lower()
        # explicit CPU
        if pref == 'cpu':
            trainer_kwargs.update({'accelerator': 'cpu', 'devices': 1})
            return trainer_kwargs

        # prefer CUDA if requested/available
        if pref in ('cuda', 'gpu') or (pref == 'auto' and torch.cuda.is_available() and args.gpus > 0):
            devices = args.gpus if args.gpus > 0 else 1
            trainer_kwargs.update({'accelerator': 'gpu', 'devices': devices})
            return trainer_kwargs

        # MPS support
        try:
            mps_available = getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
        except Exception:
            mps_available = False

        if pref == 'mps' or (pref == 'auto' and mps_available):
            trainer_kwargs.update({'accelerator': 'mps', 'devices': 1})
            return trainer_kwargs

        # fallback to CPU
        trainer_kwargs.update({'accelerator': 'cpu', 'devices': 1})
        return trainer_kwargs

    trainer_kwargs = set_accelerator_for(trainer_kwargs, device_pref)

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(lit, datamodule=dm)


if __name__ == '__main__':
    main()
