import numpy as np
import torch
import imageio
from torch.utils.data import DataLoader
from pathlib import Path

from kvae.model.model import KVAE


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