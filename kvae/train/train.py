import logging
from multiprocessing import process
import random
from venv import logger
import numpy as np
from pathlib import Path
import torch
import numpy as np
from dataclasses import dataclass
import yaml
from tqdm import tqdm

from kvae.train.imputation import impute_epoch
from kvae.train.logging_utils import setup_logging, TensorBoardLogger
from kvae.vae.config import KVAEConfig
from kvae.model.model import KVAE
from kvae.train.utils import Checkpointer, build_dataloaders, parse_config, parse_device, seed_all_modules, \
    create_runs_dir
from kvae.train.testing import reconstruct_and_save, kalman_prediction_test, pre_vidsave_trans, save_frames


def train_one_epoch(model, loader, optimizer, device, grad_clip_norm, scheduler=None, kf_weight=1.0,
                   vae_weight=1.0, tb_logger=None, epoch : int = None):
    model.train()
    total_loss = 0.0
    total_vae  = 0.0
    total_kf   = 0.0
    n_batches  = 0
    model.beta = model.scheduler.get_beta(epoch) if model.config.scheduled_beta else 1.0

    for batch in loader:
        # Reset Kalman LSTM state at the start of each sequence
        model.kalman_filter.dyn_params.reset_state()

        # Get data
        x = batch["images"].float().to(device)
        B, T = x.shape[:2]

        # Fully observed training (no masking)
        mask = torch.ones(B, T, device=device, dtype=x.dtype)

        # Forward + loss
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, mask=mask)
        losses = model.compute_loss(x, outputs, kf_weight=kf_weight, vae_weight=vae_weight,
                                    mask=mask)

        loss         = losses["loss"]
        elbo_kf      = losses["elbo_kf"]
        elbo_vae_tot = losses["elbo_vae_total"]

        loss.backward()

        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        total_loss += float(loss.detach())
        total_kf   += float(elbo_kf.detach())
        total_vae  += float(elbo_vae_tot.detach())
        n_batches  += 1

    denom = max(n_batches, 1)
    epoch_losses = {
        "loss":           total_loss / denom,
        "elbo_kf":        total_kf   / denom,
        "elbo_vae_total": total_vae  / denom,
        "active_units":   losses["active_units"]
    }

    if tb_logger is not None:
        tb_logger.log_epoch_metrics(epoch_losses, 'train')

    return epoch_losses


@torch.no_grad()
def evaluate(model, loader, device, kf_weight=1.0, tb_logger=None):
    model.eval()
    total_loss = 0.0
    total_vae  = 0.0
    total_kf   = 0.0
    active_units = 0.0
    latent_vars = [0.0, 0.0]
    n_batches  = 0

    for batch in tqdm(loader, desc="Evaluating:"):
        model.kalman_filter.dyn_params.reset_state()

        x = batch["images"].float().to(device)
        B, T = x.shape[:2]

        # Fully observed evaluation (no masking)
        mask = torch.ones(B, T, device=device, dtype=x.dtype)

        outputs = model(x, mask=mask)
        losses = model.compute_loss(x, outputs, kf_weight=kf_weight, mask=mask)

        loss          = losses["loss"]
        elbo_kf       = losses["elbo_kf"]
        elbo_vae_tot  = losses["elbo_vae_total"]

        total_loss += loss.detach()
        total_kf   += elbo_kf.detach()
        total_vae  += elbo_vae_tot.detach()
        active_units += losses["active_units"]
        latent_vars[0] += losses["latent_var_0"]
        latent_vars[1] += losses["latent_var_1"]
        n_batches  += 1

    denom = max(n_batches, 1)
    epoch_losses = {
        "loss":          total_loss / denom,
        "elbo_kf":       total_kf   / denom,
        "elbo_vae_total": total_vae / denom,
    }

    training_vae = {
        "active_units":   active_units / denom, # ToDo: Cambiarlo de Lugar
        "latent_vars_0":   latent_vars[0] / denom,
        "latent_vars_1": latent_vars[1] / denom,
        "beta_scheduler": model.beta,
    }

    if tb_logger:
        tb_logger.log_epoch_metrics(epoch_losses, 'val')
        tb_logger.log_image(batch["images"][:1], name='val/orig')
        tb_logger.log_image(outputs["x_recon"][:1], name='val/recon')
        tb_logger.log_video(batch["images"][:1], name='val/seq_orig')
        tb_logger.log_video(outputs["x_recon"][:1], name='val/seq_recon')
    return epoch_losses


def set_training_phase(model, phase: str):
    assert phase in {"vae", "warmup", "all"}

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    dyn = model.kalman_filter.dyn_params
    # VAE only
    if phase == "vae":
        for m in (model.encoder, model.decoder):
            for p in m.parameters():
                p.requires_grad = True

        dyn.A.requires_grad = False
        dyn.B.requires_grad = False
        dyn.C.requires_grad = False

        if dyn.K > 1:
            if hasattr(dyn, "lstm"):
                for p in dyn.lstm.parameters():
                    p.requires_grad = False
            if hasattr(dyn, "mlp"):
                for p in dyn.mlp.parameters():
                    p.requires_grad = False
            if hasattr(dyn, "head_w"):
                for p in dyn.head_w.parameters():
                    p.requires_grad = False

    # Warmup: train VAE + global A,B,C (mixture weights frozen)
    elif phase == "warmup":
        for m in (model.encoder, model.decoder):
            for p in m.parameters():
                p.requires_grad = True

        dyn.A.requires_grad = True
        dyn.B.requires_grad = True
        dyn.C.requires_grad = True

        # Keep dynamics network (mixture) frozen in warmup
        if dyn.K > 1:
            if hasattr(dyn, "lstm"):
                for p in dyn.lstm.parameters():
                    p.requires_grad = False
            if hasattr(dyn, "mlp"):
                for p in dyn.mlp.parameters():
                    p.requires_grad = False
            if hasattr(dyn, "head_w"):
                for p in dyn.head_w.parameters():
                    p.requires_grad = False

    # Fine-tune everything
    elif phase == "all":
        for p in model.parameters():
            p.requires_grad = True


def main():
    config = parse_config()
    train_cfg = TrainingConfig(**config['training'])
    runs_dir = create_runs_dir(train_cfg.logdir)
    setup_logging(str(runs_dir / "train.log"))
    logger = logging.getLogger(__name__)
    tb_logger = TensorBoardLogger(str(runs_dir))
    logger.info("Starting training with configuration:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    ckpt_dir = runs_dir / "checkpoints" if runs_dir else None
    ckpt = Checkpointer(ckpt_dir, train_cfg.ckpt_every)
    with open(runs_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    seed_all_modules(train_cfg.seed)
    cfg = KVAEConfig()
    device = parse_device(train_cfg.device)
    logger.info(f"Using device: {device}")
    dataset_cfg = config['dataset']
    
    train_loader, val_loader = build_dataloaders(dataset_cfg, train_cfg.batch_size)

    model = KVAE(cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    num_batches = len(train_loader)
    # LR decays every (decay_steps * num_batches) updates
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg.decay_rate,   
    )

    for epoch in range(1, train_cfg.max_epochs + 1):
        if epoch <= train_cfg.pretrain_vae_epochs:
            phase = "vae"
            kf_weight = 0.0
            vae_weight = 1.0          
        elif epoch <= train_cfg.pretrain_vae_epochs + train_cfg.warmup_epochs:
            phase = "warmup"
            kf_weight = 1.0  
            vae_weight = 1.0         
        else:
            phase = "all"
            kf_weight = 1.0 
            vae_weight = 1.0       

        set_training_phase(model, phase)

        if epoch == 1 or epoch == train_cfg.pretrain_vae_epochs + 1 or epoch == train_cfg.pretrain_vae_epochs + train_cfg.warmup_epochs + 1:
            logger.info(f"\n=== Switched to training phase '{phase}' at epoch {epoch} ===")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, cfg.grad_clip_norm, scheduler, kf_weight, vae_weight, tb_logger, epoch = epoch
        )
        if scheduler is not None and epoch % cfg.decay_steps == 0:
            scheduler.step()
        # Evaluate on fully observed data
        val_metrics   = evaluate(
            model, val_loader, device, kf_weight, tb_logger
        )

        train_loss = train_metrics["loss"]
        val_loss   = val_metrics["loss"]
        inputation_log_msg = ""

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        tb_logger.log_scalar('train/learning_rate', current_lr, num_epoch=epoch)

        if train_cfg.add_imputation_plots and epoch % 5 == 0:
            # Kalman prediction test
            kf_mse, mse_naive = kalman_prediction_test(model, val_loader, device, max_batches=5)
            inputation_log_msg += f"Kalman pred MSE {kf_mse:.6e} vs naive {mse_naive:.6e}\n\n"
            # VAE reconstruction test
            # Imputation testing on full validation set
            imp_metrics = impute_epoch(
                model, val_loader, device,
                t_init_mask=cfg.t_init_mask,
                t_steps_mask=cfg.t_steps_mask,
            )

            if imp_metrics is not None:
                logger.info(
                    f"Testing - Imputation planning "
                    f"(t_init={cfg.t_init_mask}, t_steps={cfg.t_steps_mask}) "
                    f"MSE (smooth: {imp_metrics['mse_smooth']:.6e}, "
                    f"filt: {imp_metrics['mse_filt']:.6e}, "
                    f"recon: {imp_metrics['mse_recon']:.6e})"
                    f" | baseline: {imp_metrics['baseline']:.6e}"
                )

                sample = imp_metrics.get("sample", None)
                if sample is not None:
                    tb_logger.log_image(sample["x_real"][:1], name="val_inputation/seq_impute_real")
                    tb_logger.log_image(sample["x_recon"][:1], name="val_inputation/seq_impute_recon")
                    tb_logger.log_image(sample["x_filtered"][:1], name="val_inputation/seq_impute_filt")
                    tb_logger.log_image(sample["x_imputed"][:1], name="val_inputation/seq_impute_smooth")

                    tb_logger.log_video(sample["x_real"][:1], name="val_inputation/seq_impute_real.mp4")
                    tb_logger.log_video(sample["x_recon"][:1], name="val_inputation/seq_impute_recon.mp4")
                    tb_logger.log_video(sample["x_filtered"][:1], name="val_inputation/seq_impute_filt.mp4")
                    tb_logger.log_video(sample["x_imputed"][:1], name="val_inputation/seq_impute_smooth.mp4")
            # reconstruct_and_save(model, val_loader, device, runs_dir / "videos", prefix=f"vae_epoch{epoch:03d}")
        # Logging
        logger.info(
            f"Epoch {epoch:03d} [phase={phase}]\n"
            f"Train loss (min) {train_loss:.6f} | ELBOs (max)"
            f"(VAE {train_metrics['elbo_vae_total']:.6f}, KF {train_metrics['elbo_kf']:.6f})\n"
            f"Val loss (min) {val_loss:.6f} "
            f"(VAE {val_metrics['elbo_vae_total']:.6f}, KF {val_metrics['elbo_kf']:.6f})\n"
            + inputation_log_msg
        )
        # Checkpointing
        if ckpt_dir:
            ckpt.save_checkpoints(train_loss, val_loss, model, optimizer, epoch)

        if tb_logger is not None:
            tb_logger.incr_epoch()


@dataclass
class TrainingConfig:
    seed: int = 10
    max_epochs: int = 80
    gpus: int = 1
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 0.0
    ckpt_every: int = 5
    pretrain_vae_epochs: int = 5  
    warmup_epochs: int = 10
    device: str = 'auto'
    logdir: str = 'runs'
    T: int = 20
    add_imputation_plots: bool = False


if __name__ == "__main__":
    main()