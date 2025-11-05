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
import os
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from kvae.utils import KVAEConfig
from kvae.model import KVAE
from kvae.dataloader import make_toy_dataset


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
    def __init__(self, cfg: KVAEConfig, batch_size: int = 16, num_seq: int = 200, T: int = 10):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_seq = num_seq
        self.T = T
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: Optional[str] = None):
        ds = make_toy_dataset(num_seq=self.num_seq, T=self.T, C=self.cfg.img_channels,
                              H=self.cfg.img_size, W=self.cfg.img_size)
        n_val = max(1, int(0.2 * len(ds)))
        n_train = len(ds) - n_val
        self.train_ds, self.val_ds = random_split(ds, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--logdir', type=str, default='logs')
    p.add_argument('--num-seq', type=int, default=200)
    p.add_argument('--T', type=int, default=10)
    p.add_argument('--gpus', type=int, default=0)
    p.add_argument('--ckpt-every', type=int, default=5)
    args = p.parse_args()

    cfg = KVAEConfig()
    lit = KVAELit(cfg, lr=args.lr)
    dm = KVAEDataModule(cfg, batch_size=args.batch_size, num_seq=args.num_seq, T=args.T)

    logger = TensorBoardLogger(save_dir=args.logdir, name='kvae_lightning')
    ckpt_callback = ModelCheckpoint(dirpath=os.path.join(args.logdir, 'checkpoints'), filename='ckpt-{epoch}', every_n_epochs=args.ckpt_every)

    trainer_kwargs = dict(max_epochs=args.max_epochs, logger=logger, callbacks=[ckpt_callback], log_every_n_steps=10)
    if args.gpus > 0 and torch.cuda.is_available():
        trainer_kwargs.update({'accelerator': 'gpu', 'devices': args.gpus})

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(lit, datamodule=dm)


if __name__ == '__main__':
    main()
