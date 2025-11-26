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

    # Use only the first sequence in the batch
    x_true_seq = x[0].detach().cpu().numpy()       # (T, C, H, W)
    x_recon_seq = x_recon[0].detach().cpu().numpy()

    # Convert (T, C, H, W) -> (T, H, W, C)
    if x_true_seq.ndim == 4:
        x_true_seq = np.transpose(x_true_seq, (0, 2, 3, 1))
    if x_recon_seq.ndim == 4:
        x_recon_seq = np.transpose(x_recon_seq, (0, 2, 3, 1))

    save_frames(x_true_seq,  str(out_dir / f"{prefix}_true.mp4"))
    save_frames(x_recon_seq, str(out_dir / f"{prefix}_recon.mp4"))


@torch.no_grad()
def kalman_prediction_test(
    model: KVAE,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 5,
):
    """
    Check whether the Kalman part learns useful dynamics by measuring
    one-step-ahead prediction error on the VAE latents a_t.
    """
    model.eval()

    mse_kf_sum = 0.0
    mse_naive_sum = 0.0
    n_batches = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        # Reset LSTM state for this sequence batch
        model.kalman_filter.dyn_params.reset_state()

        # Get data
        x = batch["images"].float().to(device)  # (B, T, C, H, W)
        # Forward pass
        outputs = model(x)
        a = outputs["a_samples"]          
        mus_smooth = outputs["mus_smooth"]  
        A_list, B_list, C_list = outputs["ABC"] 
        u = outputs["u"]                  

        # normalize mus_smooth to (B, T, n)
        z = mus_smooth
        if z.dim() == 4:
            z = z.squeeze(-1)
        elif z.dim() == 5:
            z = z.squeeze(-1).squeeze(-1)
        z = z.unsqueeze(-1)

        # normalize u to (B, T, m) then columnize 
        u_col = u
        if u_col.dim() == 4:
            u_col = u_col.squeeze(-1)   
        elif u_col.dim() == 5:
            u_col = u_col.squeeze(-1).squeeze(-1)
        u_col = u_col.unsqueeze(-1)     

        # Slice time
        z_t   = z[:, :-1]      
        u_tp1 = u_col[:, 1:]   
        A_t   = A_list[:, :-1]   
        B_t   = B_list[:, :-1]   
        C_tp1 = C_list[:, 1:]    

        # Predict
        z_pred = A_t @ z_t + B_t @ u_tp1        
        a_pred = (C_tp1 @ z_pred).squeeze(-1)  

        # True next latent (from VAE encoder)
        a_true = a[:, 1:, :]                       

        # KF Mean Squared Error (Observation space prediction vs VAE)
        mse_kf = torch.mean((a_pred - a_true) ** 2)

        # Naive baseline: Persistence MSE (a_naive a_t and a_true a_{t+1})
        a_naive = a[:, :-1, :]                   
        mse_naive = torch.mean((a_naive - a_true) ** 2)

        mse_kf_sum      += mse_kf.item()
        mse_naive_sum += mse_naive.item()
        n_batches       += 1

    denom = max(n_batches, 1)
    mse_kf_avg      = mse_kf_sum / denom
    mse_naive_avg = mse_naive_sum / denom

    return mse_kf_avg, mse_naive_avg


# ======================================================================

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


def train_one_epoch(model, loader, optimizer, device, grad_clip_norm, scheduler=None, kf_weight=1.0):
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

        # Forward + loss
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        losses = model.compute_loss(x, outputs, kf_weight=kf_weight)

        loss         = losses["loss"]
        elbo_kf      = losses["elbo_kf"]
        elbo_vae_tot = losses["elbo_vae_total"]

        loss.backward()

        # if grad_clip_norm and grad_clip_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

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
        outputs = model(x)
        losses = model.compute_loss(x, outputs, kf_weight=kf_weight)

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
    assert phase in {"vae", "vae_kf", "all"}

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    dyn = model.kalman_filter.dyn_params
    
    # Train only VAE
    if phase == "vae":
        
        for m in (model.encoder, model.decoder):
            for p in m.parameters():
                p.requires_grad = True

    # Freeze VAE, train only Kalman dynamics
    elif phase == "vae_kf":
        dyn.A.requires_grad = True
        dyn.B.requires_grad = True
        dyn.C.requires_grad = True

        if dyn.K > 1:
            if hasattr(dyn, "lstm"):
                for p in dyn.lstm.parameters():
                    p.requires_grad = True
            if hasattr(dyn, "mlp"):
                for p in dyn.mlp.parameters():
                    p.requires_grad = True
            if hasattr(dyn, "head_w"):
                for p in dyn.head_w.parameters():
                    p.requires_grad = True

    # Fine-tune everything
    elif phase == "all":
        for m in (model.encoder, model.decoder):
            for p in m.parameters():
                p.requires_grad = True

        dyn.A.requires_grad = True
        dyn.B.requires_grad = True
        dyn.C.requires_grad = True

        if dyn.K > 1:
            if hasattr(dyn, "lstm"):
                for p in dyn.lstm.parameters():
                    p.requires_grad = True
            if hasattr(dyn, "mlp"):
                for p in dyn.mlp.parameters():
                    p.requires_grad = True
            if hasattr(dyn, "head_w"):
                for p in dyn.head_w.parameters():
                    p.requires_grad = True


if __name__ == "__main__":
    # Fix random seeds for reproducibility
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Core settings 
    max_epochs = 80          # num_epochs
    batch_size = 32          # batch_size
    lr = 0.001               # init_lr
    ckpt_every = 0
    logdir = "runs"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 20  
    only_vae_epochs = 5    
    kf_update_epochs = 5   

    cfg = KVAEConfig()
    print(f"Using device: {device}")

    pathfile_videos = "/home/daniel/Documents/MVA/PGM/box.npz"
    train_loader, val_loader = build_dataloaders(pathfile_videos, batch_size=batch_size, T=T)

    # DEBUG
    # # Plot a sample batch
    # sample_batch = next(iter(train_loader))
    # sample_images = extract_images(sample_batch)
    # B, TT, C, H, W = sample_images.shape
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

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.init_lr)
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

    for epoch in range(1, max_epochs + 1):
        if epoch <= only_vae_epochs:
            phase = "vae"
            kf_weight = 0.0           
        elif epoch <= only_vae_epochs + kf_update_epochs:
            phase = "vae_kf"
            kf_weight = 0.3           
        else:
            phase = "all"
            kf_weight = 1.0           

        set_training_phase(model, phase)

        # Print when phase changes
        if epoch == 1 or epoch == only_vae_epochs + 1 or epoch == only_vae_epochs + kf_update_epochs + 1:
            print(f"\n=== Switched to training phase '{phase}' at epoch {epoch} ===")

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg.grad_clip_norm, scheduler, kf_weight)        
        val_metrics   = evaluate(model, val_loader, device, kf_weight)

        train_loss = train_metrics["loss"]
        val_loss   = val_metrics["loss"]
        
        # Kalman prediction test
        kf_mse, mse_naive = kalman_prediction_test(model, val_loader, device, max_batches=5)
        # VAE reconstruction test
        reconstruct_and_save(model, val_loader, device, Path('./runs/'), prefix=f"vae_epoch{epoch:03d}")
        # Logging
        print(
            f"Epoch {epoch:03d} [phase={phase}]\n"
            f"Train loss (min) {train_loss:.6f} | ELBOs (max)"
            f"(VAE {train_metrics['elbo_vae_total']:.6f}, KF {train_metrics['elbo_kf']:.6f})\n"
            f"Val loss (min) {val_loss:.6f} "
            f"(VAE {val_metrics['elbo_vae_total']:.6f}, KF {val_metrics['elbo_kf']:.6f})\n"
            f"Kalman prediction MSE {kf_mse:.6e} vs naive {mse_naive:.6e}\n\n"
        )
        # Checkpointing
        if ckpt_dir:
            best_path = ckpt_dir / "best.pt"
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best_path, model, optimizer, epoch, train_loss, val_loss)

            if ckpt_every > 0 and epoch % ckpt_every == 0:
                ckpt_path = ckpt_dir / f"epoch-{epoch:03d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, epoch, train_loss, val_loss)