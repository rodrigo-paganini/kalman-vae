import random
import numpy as np
from tqdm import tqdm
import yaml
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvae.vae.config import KVAEConfig
from kvae.model.model import KVAE
from kvae.dataloader.pymunk_dataset import PymunkNPZDataset

import imageio
import numpy as np


# ============= DEBUG TESTING  =============
def _pad_to_block(x, block = 16):
    """Pad H/W to the next multiple of `block` for video codecs."""
    if x.ndim != 4:
        return x
    H, W = x.shape[1], x.shape[2]
    pad_h = (block - H % block) % block
    pad_w = (block - W % block) % block
    if pad_h == 0 and pad_w == 0:
        return x
    return np.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="constant")


def save_frames(x, filename, fps=10):
    """
    Save a sequence of frames as an MP4 video.

    Expected shapes:
      - (B, T, H, W, C)  -> we save only the FIRST sequence (batch index 0)
      - (T, H, W)        -> grayscale sequence
      - (T, H, W, C)     -> color/grayscale sequence

    If you have (B, T, H, W), slice the batch and add the channel dim
    before calling this function (e.g. x[0, ..., None]).
    """
    x = np.asarray(x)

    # If batch dimension is present, take first sequence.
    if x.ndim == 5:          # (B, T, H, W, C)
        x = x[0]             # (T, H, W, C)

    # Now x should be (T, H, W) or (T, H, W, C)
    if x.ndim == 3:          # (T, H, W) -> add channel dim
        x = x[..., None]     # (T, H, W, 1)

    # Normalize to 0–255 uint8 for video
    x_min, x_max = x.min(), x.max()
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros_like(x)
    x = (x * 255).astype(np.uint8)

    # Pad to multiples of 16 to avoid ffmpeg resizing warnings
    x = _pad_to_block(x, block=16)

    # imageio expects (num_frames, H, W, C)
    imageio.mimwrite(filename, x, fps=fps)
    print(f"Saved video to {filename}")


def pre_vidsave_trans(x, index=0):
    """
    x: torch.Tensor of shape (B, T, C, H, W)

    Returns:
        np.ndarray of shape (T, H, W, C) for the first sequence in the batch.
    """
    x_seq = x[index].detach().cpu().numpy()  # (T, C, H, W)
    if x_seq.ndim == 4:
        x_seq = np.transpose(x_seq, (0, 2, 3, 1))  # (T, H, W, C)
    return x_seq


# NOTE: Not used anymore, using imputation directly
def reconstruct_and_save(
    model: KVAE,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    prefix: str = "vae",
) -> None:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Take the first batch from the *validation* loader
    batch = next(iter(loader))
    x = batch["images"].float().to(device)  # (B, T, C, H, W)

    with torch.no_grad():
        model.kalman_filter.dyn_params.reset_state()
        outputs = model(x)
        x_recon = outputs["x_recon"]      # (B, T, C, H, W)

    save_frames(pre_vidsave_trans(x[0]),  str(out_dir / f"{prefix}_true.mp4"))
    save_frames(pre_vidsave_trans(x_recon[0]), str(out_dir / f"{prefix}_recon.mp4"))


# === IMPUTATION FUNCTIONS ===
def mask_impute_planning(batch_size, T, t_init_mask = 4, t_steps_mask = 12, device=None):
    """
    Observe first t_init_mask steps, hide the next t_steps_mask steps, then observe the rest.
    """
    mask = torch.ones(batch_size, T, device=device)
    t_end = t_init_mask + t_steps_mask
    t_end = min(t_end, T)
    mask[:, t_init_mask:t_end] = 0.0
    return mask


def mask_impute_random(batch_size, T, t_init_mask = 4, drop_prob = 0.5, device=None):
    """
    Observe first t_init_mask steps, then randomly drop later steps with probability drop_prob.
    """
    mask = torch.ones(batch_size, T, device=device)
    n_steps = T - t_init_mask
    if n_steps > 0:
        mask[:, t_init_mask:] = torch.bernoulli(
            torch.full((batch_size, n_steps), 1.0 - drop_prob, device=device)
        )
    return mask


def make_training_mask(batch_size, T, t_init_mask=4, drop_prob=0.0, device=None, strategy="random", t_steps_mask=12):
    strategy = strategy.lower()
    if strategy == "block":
        return mask_impute_planning(batch_size, T, t_init_mask=t_init_mask, t_steps_mask=t_steps_mask, device=device)
    if drop_prob <= 0:
        return torch.ones(batch_size, T, device=device)
    return mask_impute_random(batch_size, T, t_init_mask=t_init_mask, drop_prob=drop_prob, device=device)


@torch.no_grad()
def impute_batch(model, batch, mask, device):
    """
    Imputation evaluation on a single batch:
    - uses model.impute(...) with a given mask
    - reconstructs x from:
        * VAE a (x_recon)
        * smoothed a (x_imputed)
        * filtered a (x_filtered)
    - computes Hamming distances on unobserved frames.
    """
    model.eval()

    x = batch["images"].float().to(device)      # [B,T,C,H,W]
    B, T, C, H, W = x.shape

    # Optional controls
    u = batch.get("controls", None)
    if u is not None:
        u = u.to(device)

    mask = mask.to(device)

    # KVAE imputation
    imp_out = model.impute(x, mask=mask, u=u)

    x_recon    = imp_out["x_recon"]    # [B,T,C,H,W] (VAE baseline)
    x_imputed  = imp_out["x_imputed"]  # [B,T,C,H,W] (smoothing)
    x_filtered = imp_out["x_filtered"] # [B,T,C,H,W] (filtering)

    # 1 = observed, 0 = missing -> we want missing
    unobs = (mask < 0.5)                 # [B,T]
    if unobs.sum() == 0:
        return None

    # Expand mask to pixelwise shape for broadcasting
    unobs_px = unobs.view(B, T, 1, 1, 1)   # [B,T,1,1,1]

    # MSE on missing frames
    def mse_on_unobs(x_hat):
        diff2 = (x - x_hat) ** 2
        mask_full = unobs_px.expand_as(x)       # [B,T,C,H,W]
        diff2 = diff2[mask_full.bool()]        # flatten all missing pixels
        return diff2.mean().item()

    # MSE on baseline (comparing random unobserved frames)
    baseline = 0.0
    for i in [0, min(3, T-1), min(6, T-1)]:
        for j in [min(9, T-1), min(12, T-1), min(15, T-1)]:
            if i >= T or j >= T:
                continue

            # Sequences where both timesteps are unobserved
            pair_unobs = (mask[:, i] < 0.5) & (mask[:, j] < 0.5)   # [B]
            if pair_unobs.sum() == 0:
                continue

            xi = x[pair_unobs, i]  # [B',C,H,W]
            xj = x[pair_unobs, j]  # [B',C,H,W]

            dist = ((xi - xj) ** 2).mean().item()   
            baseline = max(baseline, dist)


    mse_smooth   = mse_on_unobs(x_imputed)      # MSE using smoothed reconstruction
    mse_filt     = mse_on_unobs(x_filtered)     # MSE using filtered reconstruction  
    mse_recon    = mse_on_unobs(x_recon)        # MSE using VAE reconstruction

    return {
        "x_real":     x,
        "x_recon":    x_recon,
        "x_imputed": x_imputed,
        "x_filtered": x_filtered,
        "mse_smooth": mse_smooth,
        "mse_filt":   mse_filt,
        "mse_recon":  mse_recon,
        "baseline":   baseline,
    }


@torch.no_grad()
def impute_epoch(model, loader, device, t_init_mask=4, t_steps_mask=12):
    """
    Run imputation over the full loader and aggregate metrics.
    Returns averaged mse_smooth, mse_filt, mse_recon, baseline and one sample batch for visualization.
    """
    totals = {"mse_smooth": 0.0, "mse_filt": 0.0, "mse_recon": 0.0, "baseline": 0.0}
    n = 0
    sample = None

    for batch in loader:
        B, T = batch["images"].shape[:2]
        mask_planning = mask_impute_planning(
            batch_size=B, T=T, t_init_mask=t_init_mask, t_steps_mask=t_steps_mask, device=device
        )
        metrics = impute_batch(model, batch, mask_planning, device)
        if metrics is None:
            continue
        for k in totals:
            totals[k] += metrics[k]
        n += 1
        if sample is None:
            sample = metrics

    if n == 0:
        return None

    averaged = {k: v / n for k, v in totals.items()}
    averaged["sample"] = sample
    return averaged


# ==============================================================

def load_config(path):
    if path is None: return {}
    p = Path(path)
    if not p.exists(): return {}
    return yaml.safe_load(p.read_text()) or {}


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


def train_one_epoch(model, loader, optimizer, device, grad_clip_norm, scheduler=None, kf_weight=1.0,
                   vae_weight=1.0):
    model.train()
    total_loss = 0.0
    total_vae  = 0.0
    total_kf   = 0.0
    n_batches  = 0

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
        losses = model.compute_loss(x, outputs, kf_weight=kf_weight, vae_weight=vae_weight, mask=mask)

        loss         = losses["loss"]
        elbo_kf      = losses["elbo_kf"]
        elbo_vae_tot = losses["elbo_vae_total"]

        loss.backward()

        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.detach())
        total_kf   += float(elbo_kf.detach())
        total_vae  += float(elbo_vae_tot.detach())
        n_batches  += 1

    denom = max(n_batches, 1)
    return {
        "loss":           total_loss / denom,
        "elbo_kf":        total_kf   / denom,
        "elbo_vae_total": total_vae  / denom,
    }


@torch.no_grad()
def evaluate(model, loader, device, kf_weight=1.0):
    model.eval()
    total_loss = 0.0
    total_vae  = 0.0
    total_kf   = 0.0
    n_batches  = 0

    for batch in loader:
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
        n_batches  += 1

    denom = max(n_batches, 1)
    return {
        "loss":          total_loss / denom,
        "elbo_kf":       total_kf   / denom,
        "elbo_vae_total": total_vae / denom,
    }


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


if __name__ == "__main__":
    # Fix random seeds for reproducibility
    seed = 15
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Core settings 
    max_epochs = 80          # num_epochs
    batch_size = 32          # batch_size
    ckpt_every = 0
    logdir = "runs"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 20  
    pretrain_vae_epochs = 5   # VAE only
    warmup_epochs = 5        # VAE + global A,B,C, mixture frozen

    cfg = KVAEConfig()
    print(f"Using device: {device}")

    pathfile_videos = "/home/daniel/Documents/MVA/PGM/box.npz"
    train_loader, val_loader = build_dataloaders(pathfile_videos, batch_size=batch_size, T=T)

    # # DEBUG
    # # Plot a sample batch
    # sample_batch = next(iter(train_loader))
    # fig, axes = plt.subplots(1, TT, figsize=(TT * 2, 2))
    # for t in range(TT):
    #     ax = axes[t]
    #     img = sample_images[0, t].squeeze().cpu().numpy()
    #     ax.imshow(img, cmap="gray")
    #     ax.axis("off")
    #     ax.set_title(f"t={t}")
    # plt.suptitle("Sample training sequence")
    # plt.show()

    model = KVAE(cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.init_lr, weight_decay=0.0)
    num_batches = len(train_loader)
    # LR decays every (decay_steps * num_batches) updates
    step_size = cfg.decay_steps * num_batches  
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=cfg.decay_rate,   
    )

    best_val = float("inf")
    ckpt_dir = Path(logdir) if logdir else None

    GENERATE_STEP = 5          # cfg.generate_step
    T_INIT_MASK   = 4          # cfg.t_init_mask
    T_STEPS_MASK  = 12         # cfg.t_steps_mask

    for epoch in range(1, max_epochs + 1):
        if epoch <= pretrain_vae_epochs:
            phase = "vae"
            kf_weight = 0.0
            vae_weight = 1.0
        elif epoch <= pretrain_vae_epochs + warmup_epochs:
            phase = "warmup"
            kf_weight = 1.0  
            vae_weight = 1.0
        else:
            phase = "all"
            kf_weight = 1.0 
            vae_weight = 1.0

        set_training_phase(model, phase)

        # Print when phase changes
        if epoch == 1 or epoch == pretrain_vae_epochs + 1 or epoch == pretrain_vae_epochs + warmup_epochs + 1:
            print(f"\n=== Switched to training phase '{phase}' at epoch {epoch} ===")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, cfg.grad_clip_norm, scheduler, kf_weight,
            vae_weight=vae_weight,
        )
        # Evaluate on fully observed data
        val_metrics   = evaluate(
            model, val_loader, device, kf_weight,
        )

        train_loss = train_metrics["loss"]
        val_loss   = val_metrics["loss"]

        # Logging
        print(
            f"Epoch {epoch:03d} [phase={phase}]\n"
            f"Train loss (min) {train_loss:.6f} | ELBOs (max)"
            f"(VAE {train_metrics['elbo_vae_total']:.6f}, KF {train_metrics['elbo_kf']:.6f})\n"
            f"Val loss (min) {val_loss:.6f} "
            f"(VAE {val_metrics['elbo_vae_total']:.6f}, KF {val_metrics['elbo_kf']:.6f})"
        )
        
        if (epoch % GENERATE_STEP == 0) or epoch == 1 or epoch == max_epochs:
            # Imputation testing on full validation set
            imp_metrics = impute_epoch(
                model, val_loader, device,
                t_init_mask=T_INIT_MASK,
                t_steps_mask=T_STEPS_MASK,
            )

            if imp_metrics is not None:
                print(
                    f"Testing - Imputation planning "
                    f"(t_init={T_INIT_MASK}, t_steps={T_STEPS_MASK}) "
                    f"MSE (smooth: {imp_metrics['mse_smooth']:.6e}, "
                    f"filt: {imp_metrics['mse_filt']:.6e}, "
                    f"recon: {imp_metrics['mse_recon']:.6e})"
                    f" | baseline: {imp_metrics['baseline']:.6e}"
                )

                sample = imp_metrics.get("sample", None)
                if sample is not None:
                    save_frames(pre_vidsave_trans(sample["x_real"]), Path('./runs/') / f"epoch_{epoch}_real.mp4")
                    save_frames(pre_vidsave_trans(sample["x_recon"]),    Path('./runs/') / f"epoch_{epoch}_vae_recon.mp4")
                    save_frames(pre_vidsave_trans(sample["x_imputed"]),  Path('./runs/') / f"epoch_{epoch}_impute_smooth.mp4")
                    save_frames(pre_vidsave_trans(sample["x_filtered"]), Path('./runs/') / f"epoch_{epoch}_impute_filt.mp4")

        # Checkpointing
        if ckpt_dir:
            best_path = ckpt_dir / "best.pt"
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best_path, model, optimizer, epoch, train_loss, val_loss)

            if ckpt_every > 0 and epoch % ckpt_every == 0:
                ckpt_path = ckpt_dir / f"epoch-{epoch:03d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, epoch, train_loss, val_loss)
