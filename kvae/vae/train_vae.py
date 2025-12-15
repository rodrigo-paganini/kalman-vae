"""PyTorch Lightning training script for the VAE-only component.

This mirrors the style of `train_lightning.py` but trains only the
Encoder/Decoder (VAE) using the same `KVAEConfig` for compatibility.

It is mostly intended for testing the VAE capabilities, but is not to be included in the main training pipeline.

Usage:
    python -m kvae.train.train_vae

"""
from __future__ import annotations

import argparse
import json
import yaml
import os
import shutil
from typing import Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from kvae.utils.config import KVAEConfig
from kvae.vae.vae import VAE
from kvae.dataloader.dataloader import make_toy_dataset
from kvae.dataloader.pymunk_dataset import PymunkNPZDataset
from kvae.vae.losses import vae_loss

@dataclass
class TrainingConfig:
    max_epochs: int = 10
    gpus: int = 1
    lr: float = 1e-3
    batch_size: int = 32
    ckpt_every: int = 5
    device: str = 'auto'
    logdir: str = 'runs'


class VAELit(pl.LightningModule):
    """LightningModule wrapper for a simple VAE (Encoder + Decoder).

    The VAE encodes per-frame images a_t ~ q(a|x) and decodes them back.
    We train with a standard ELBO: MSE reconstruction + KL(q||N(0,I)).
    """

    def __init__(self, cfg: KVAEConfig, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters({'lr': lr})
        self.cfg = cfg
        self.model = VAE(cfg)
        self.lr = lr
        # placeholders for validation-image logging
        self._val_orig = None
        self._val_recon = None

        # Running aggregates for epoch-level metrics
        self._train_loss_sum = 0.0
        self._train_loss_count = 0
        self._val_loss_sum = 0.0
        self._val_loss_count = 0

    # Forwarding is handled by the VAE wrapper. Use self.model(x) in steps.

    def training_step(self, batch, batch_idx):
        # support dicts and simple TensorDataset
        if isinstance(batch, dict):
            x = batch.get('images')
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        out = self.model(x)

        # Calculate losses
        x_mu, x_var = out['x_recon_mu'], out['x_recon_var']
        a, a_mu, a_var = out['a_vae'], out['a_mu'], out['a_var']

        total, recon, kl = vae_loss(x, x_mu, x_var, a, a_mu, a_var, scale_reconstruction=self.cfg.scale_reconstruction)
        self.log('train/total_loss', total, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/recon_loss', recon, on_step=True, on_epoch=True)
        self.log('train/kl_loss', kl, on_step=True, on_epoch=True)

        # accumulate for epoch-level logging
        try:
            self._train_loss_sum += float(total.detach().cpu().item())
        except Exception:
            self._train_loss_sum += float(total)
        self._train_loss_count += 1

        return total

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x = batch.get('images')
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        out = self.model(x)

        # Calculate losses
        x_mu, x_var = out['x_recon_mu'], out['x_recon_var']
        a, a_mu, a_var = out['a_vae'], out['a_mu'], out['a_var']

        total, recon, kl = vae_loss(x, x_mu, x_var, a, a_mu, a_var, scale_reconstruction=self.cfg.scale_reconstruction)
        self.log('val/total_loss', total, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recon_loss', recon, on_step=False, on_epoch=True)
        self.log('val/kl_loss', kl, on_step=False, on_epoch=True)

        # keep first batch's first sequence for image logging at epoch end
        if batch_idx == 0:
            # x has shape [B, T, C, H, W]
            self._val_orig = x[:1].detach().cpu()
            self._val_recon = out['x_recon'][:1].detach().cpu()

        # accumulate for epoch-level logging
        try:
            self._val_loss_sum += float(total.detach().cpu().item())
        except Exception:
            self._val_loss_sum += float(total)
        self._val_loss_count += 1

        return total

    def on_train_epoch_end(self):
        """Aggregate and log training epoch metrics (replacement for training_epoch_end)."""
        if self._train_loss_count > 0:
            mean = self._train_loss_sum / float(self._train_loss_count)
        else:
            mean = 0.0
        import logging
        logging.getLogger('kvae.train').info(f'Epoch {self.current_epoch} train_loss={mean:.6f}')
        try:
            self.log('train/epoch_loss', mean, prog_bar=True, logger=True)
        except Exception:
            pass
        # reset counters
        self._train_loss_sum = 0.0
        self._train_loss_count = 0

    def on_validation_epoch_end(self):
        # Aggregate and log validation epoch loss
        if self._val_loss_count > 0:
            val_mean = self._val_loss_sum / float(self._val_loss_count)
        else:
            val_mean = 0.0
        import logging
        logging.getLogger('kvae.train').info(f'Epoch {self.current_epoch} val_loss={val_mean:.6f}')
        try:
            self.log('val/epoch_loss', val_mean, prog_bar=True, logger=True)
        except Exception:
            pass
        # reset validation counters
        self._val_loss_sum = 0.0
        self._val_loss_count = 0

        # log images to TensorBoard if logger supports experiment
        logger = self.logger
        if hasattr(logger, 'experiment') and self._val_orig is not None and self._val_recon is not None:
            tb = logger.experiment
            orig = self._val_orig.squeeze(0)  # [T, C, H, W]
            recon = self._val_recon.squeeze(0)

            def norm(imgs: torch.Tensor) -> torch.Tensor:
                imgs = imgs.clone()
                mins = imgs.view(imgs.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
                imgs = imgs - mins
                m = imgs.view(imgs.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
                m = m.masked_fill(m == 0, 1.0)
                return imgs / m

            orig_n = norm(orig)
            recon_n = norm(recon)
            tb.add_images('val/orig', orig_n, self.current_epoch)
            tb.add_images('val/recon', recon_n, self.current_epoch)

        # drop references used for image logging
        self._val_orig = None
        self._val_recon = None

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt


class VAEDataModule(pl.LightningDataModule):
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
    p.add_argument('--config', type=str, default='kvae/vae/config.yaml', help='Path to YAML/JSON config file')
    # VAE training specific args
    p.add_argument('--T', type=int, default=10, help='Length of each sequence in the (toy) dataset.')
    p.add_argument('--num-seq', type=int, default=200, help='Number of sequences in the (toy) dataset.')
    args = p.parse_args()

    config_path = args.config
    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            if config_path.lower().endswith(('.yml', '.yaml')):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

    train_cfg = TrainingConfig(**(config.get('training', {}) if config else {}))
    cfg = KVAEConfig()

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

    def build_transform(tcfg: dict):
        noise_std = float(tcfg.get('add_noise_std', 0.0)) if tcfg else 0.0
        def transform(item: dict):
            imgs = item['images'].float()
            if noise_std > 0:
                imgs = imgs + torch.randn_like(imgs) * noise_std
            item['images'] = imgs
            return item
        return transform if noise_std > 0 else None

    transform_fn = build_transform(transforms_cfg)

    lit = VAELit(
        cfg,
        lr=train_cfg.lr
    )
    dm = VAEDataModule(
        cfg,
        batch_size=train_cfg.batch_size,
        num_seq=args.num_seq,
        T=args.T,
        dataset_type=ds_type,
        dataset_path=ds_path,
        dataset_kwargs=ds_kwargs,
        transform_fn=transform_fn
    )

    # Create a timestamped run folder so each experiment is isolated: <logdir>/<timestamp>/
    import datetime
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    ckpt_root = os.path.join(train_cfg.logdir, ts)
    os.makedirs(ckpt_root, exist_ok=True)

    # Prepare a dedicated checkpoints subfolder inside the run folder
    ckpt_dir = os.path.join(ckpt_root, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # initialize TensorBoard logger so its logs are stored directly under the run folder
    # set name and version empty so Lightning doesn't create nested tb/version_x folders
    logger = TensorBoardLogger(save_dir=ckpt_root, name='', version='')

    # copy config into ckpt_root for reproducibility
    if config_path and os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(ckpt_root, 'config.yaml'))

    # write hparams.yaml with CLI args and model config
    try:
        from dataclasses import asdict
        hparams = {'args': vars(args), 'cfg': asdict(cfg)}
    except Exception:
        hparams = {'args': vars(args), 'cfg': cfg.__dict__}
    with open(os.path.join(ckpt_root, 'hparams.yaml'), 'w') as hf:
        yaml.safe_dump(hparams, hf)

    import logging
    # write training log into the run folder so each experiment is self-contained
    log_file = os.path.join(ckpt_root, 'train.log')
    root = logging.getLogger()
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

    # Save the checkpoint with the best VAE validation loss (minimize)
    ckpt_callback_best = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='vae-best',
        monitor='val/total_loss',
        mode='min',
        save_top_k=1,
    )
    # Also keep periodic checkpoints for previews/history (every N epochs).
    ckpt_callback_periodic = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='vae-ckpt-{epoch}',
        every_n_epochs=train_cfg.ckpt_every,
        save_top_k=-1,
    )

    training_cfg = config.get('training', {}) if config else {}
    device_pref = training_cfg.get('device', 'auto')

    trainer_kwargs = dict(
        max_epochs=train_cfg.max_epochs,
        logger=logger,
        callbacks=[ckpt_callback_best, ckpt_callback_periodic],
        log_every_n_steps=10,
    )

    def set_accelerator_for(trainer_kwargs: dict, device_pref: str):
        pref = (device_pref or 'auto').lower()
        if pref == 'cpu':
            trainer_kwargs.update({'accelerator': 'cpu', 'devices': 1})
            return trainer_kwargs
        if pref in ('cuda', 'gpu') or (pref == 'auto' and torch.cuda.is_available() and train_cfg.gpus > 0):
            devices = train_cfg.gpus if train_cfg.gpus > 0 else 1
            trainer_kwargs.update({'accelerator': 'gpu', 'devices': devices})
            return trainer_kwargs
        try:
            mps_available = getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
        except Exception:
            mps_available = False
        if pref == 'mps' or (pref == 'auto' and mps_available):
            trainer_kwargs.update({'accelerator': 'mps', 'devices': 1})
            return trainer_kwargs
        trainer_kwargs.update({'accelerator': 'cpu', 'devices': 1})
        return trainer_kwargs

    trainer_kwargs = set_accelerator_for(trainer_kwargs, device_pref)

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(lit, datamodule=dm)


if __name__ == '__main__':
    main()
