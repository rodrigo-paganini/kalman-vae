#Minimal KVAE training loop using plain PyTorch.
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


def train_one_epoch(model, loader, optimizer, device, grad_clip_norm):
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
        losses = model.compute_loss(x, outputs)

        loss          = losses["loss"]
        elbo_kf       = losses["elbo_kf"]
        elbo_vae_tot  = losses["elbo_vae_total"]

        loss.backward()

        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        # Accumulate for epoch averages
        total_loss += float(loss.detach())
        total_kf   += float(elbo_kf.detach())
        total_vae  += float(elbo_vae_tot.detach())
        n_batches  += 1

    denom = max(n_batches, 1)
    return {
        "loss":          total_loss / denom,
        "elbo_kf":       total_kf   / denom,   # Kalman term (ELBO part)
        "elbo_vae_total": total_vae / denom,   # VAE term (ELBO part)
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_vae  = 0.0
    total_kf   = 0.0
    n_batches  = 0

    for batch in loader:
        model.kalman_filter.dyn_params.reset_state()

        x = batch["images"].float().to(device)
        outputs = model(x)
        losses = model.compute_loss(x, outputs)

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


def set_kalman_trainable(model, trainable):
    for p in model.kalman_filter.dyn_params.parameters():
        p.requires_grad = trainable


if __name__ == "__main__":
    # Core settings live right here; adjust to taste.
    max_epochs = 50
    batch_size = 50
    lr = 7e-3
    ckpt_every = 0
    logdir = "runs"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warmup_alpha_epochs = 5
    T = 20

    cfg = KVAEConfig()
    print(f"Using device: {device}")

    pathfile_videos = "box.npz"
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    ckpt_dir = Path(logdir) if logdir else None

    for epoch in tqdm(range(1, max_epochs + 1), desc="Training epochs"):
        # Freeze alpha during the first few epochs
        # train_alpha = epoch > warmup_alpha_epochs
        # if epoch == warmup_alpha_epochs + 1: print("Unfreezing alpha for training")
        # set_kalman_trainable(model, train_alpha)
    
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg.grad_clip_norm)
        val_metrics   = evaluate(model, val_loader, device)

        train_loss = train_metrics["loss"]
        val_loss   = val_metrics["loss"]

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.6f} "
            f"(VAE {train_metrics['elbo_vae_total']:.6f}, KF {train_metrics['elbo_kf']:.6f}) | "
            f"val loss {val_loss:.6f} "
            f"(VAE {val_metrics['elbo_vae_total']:.6f}, KF {val_metrics['elbo_kf']:.6f})"
        )

        if ckpt_dir:
            best_path = ckpt_dir / "best.pt"
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best_path, model, optimizer, epoch, train_loss, val_loss)

            if ckpt_every > 0 and epoch % ckpt_every == 0:
                ckpt_path = ckpt_dir / f"epoch-{epoch:03d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, epoch, train_loss, val_loss)

    out_dir = Path(logdir) if logdir else Path(".")
    reconstruct_and_save(model, val_loader, device, out_dir, prefix=f"vae")