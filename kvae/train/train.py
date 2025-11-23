#Minimal KVAE training loop using plain PyTorch.
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvae.model.config import KVAEConfig
from kvae.model.model import KVAE
from kvae.data_loading.dataloader import make_toy_dataset
from kvae.data_loading.pymunk_dataset import PymunkNPZDataset

import imageio
import numpy as np

def _pad_to_block(x: np.ndarray, block: int = 16) -> np.ndarray:
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



def load_config(path: Optional[str]) -> dict:
    """Load a YAML/JSON config if it exists; otherwise return an empty dict."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r") as f:
        if p.suffix.lower() in (".yml", ".yaml"):
            return yaml.safe_load(f) or {}
        return json.load(f)


def resolve_device(preference: str) -> torch.device:
    """Pick a device based on preference ('auto'|'cpu'|'cuda'|'mps')."""
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    try:
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        mps_available = False
    if pref == "mps" and mps_available:
        return torch.device("mps")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
    return torch.device("cpu")


def extract_images(batch) -> torch.Tensor:
    """Handle batch formats from different datasets and return the image tensor."""
    if isinstance(batch, dict):
        x = batch.get("images")
    elif isinstance(batch, (list, tuple)):
        x = batch[0]
    else:
        x = batch
    if x is None:
        raise ValueError("Batch does not contain images")
    return x


def build_dataloaders(
    cfg: KVAEConfig,
    dataset_cfg: dict,
    batch_size: int,
    num_seq: int,
    T: int,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders for either the toy or pymunk dataset."""
    ds_type = (dataset_cfg.get("type") if dataset_cfg else None) or "toy"
    ds_kwargs = (dataset_cfg.get("kwargs") if dataset_cfg else None) or {}
    ds_path = dataset_cfg.get("path") if dataset_cfg else None

    if ds_type == "toy":
        dataset = make_toy_dataset(num_seq=num_seq, T=T, C=cfg.img_channels, H=cfg.img_size, W=cfg.img_size)
    elif ds_type == "pymunk":
        if not ds_path:
            raise ValueError("dataset.path must be set when using the pymunk dataset")
        dataset = PymunkNPZDataset.from_npz(ds_path, seq_len=T, **ds_kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def maybe_add_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return x
    return x + torch.randn_like(x) * noise_std


def preprocess_batch(batch, device: torch.device, noise_std: float):
    """Move batch to device and apply optional noise in one place."""
    if isinstance(batch, dict):
        data = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                data[k] = v.to(device)
            else:
                data[k] = v
        imgs = data.get("images")
        if imgs is None:
            raise ValueError("Batch does not contain images")
        imgs = maybe_add_noise(imgs.float(), noise_std)
        data["images"] = imgs
        return data

    if isinstance(batch, (list, tuple)):
        imgs = maybe_add_noise(batch[0].float(), noise_std).to(device)
        rest = [b.to(device) if torch.is_tensor(b) else b for b in batch[1:]]
        return (imgs, *rest)

    if torch.is_tensor(batch):
        return maybe_add_noise(batch.float(), noise_std).to(device)

    raise TypeError(f"Unsupported batch type: {type(batch)}")


class PreprocessedLoader:
    """Wrapper that applies preprocessing to each batch."""

    def __init__(self, loader, device: torch.device, noise_std: float):
        self.loader = loader
        self.device = device
        self.noise_std = noise_std

    def __iter__(self):
        for batch in self.loader:
            yield preprocess_batch(batch, self.device, self.noise_std)

    def __len__(self):
        return len(self.loader)


def train_one_epoch(model, loader, optimizer, device, grad_clip_norm):
    model.train()
    total = 0.0
    for batch in loader:
        # Reset Kalman LSTM state at the start of each sequence
        model.kalman_filter.dyn_params.reset_state()

        # Get data
        x = extract_images(batch).to(device)

        # Forward + loss
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        losses = model.compute_loss(x, outputs)
        loss = losses['loss']

        # Debugging purposes
        # elbo_kf = losses['elbo_kf']
        # elbo_vae_total = losses['elbo_vae_total']
        # print(f"Train step loss: {loss.item():.4f} | ELBO KF: {elbo_kf.item():.4f} | ELBO VAE total: {elbo_vae_total.item():.4f}")

        loss.backward()

        # if grad_clip_norm and grad_clip_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()
        total += float(loss.detach().cpu())

    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model: KVAE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        model.kalman_filter.dyn_params.reset_state()

        x = extract_images(batch).to(device)
        outputs = model(x)
        losses = model.compute_loss(x, outputs)
        loss = losses['loss']
        total += float(loss.detach().cpu())

    return total / max(len(loader), 1)


def save_checkpoint(path: Path, model: KVAE, optimizer: torch.optim.Optimizer, epoch: int, train_loss: float, val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(payload, path)


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
    x = extract_images(batch).to(device)  # (B, T, C, H, W)

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


def main():
    # Core settings live right here; adjust to taste.
    max_epochs = 20
    batch_size = 16
    lr = 1e-3
    ckpt_every = 0
    logdir = "runs"
    device_pref = "auto"
    num_seq = 200
    T = 10
    noise_std = 0.0

    def override(base, section, key, cast=None):
        if key not in section:
            return base
        val = section[key]
        return cast(val) if cast else val

    # Optional config file overrides
    config = load_config(ROOT / "kvae/train/config.yaml")
    training_cfg = (config or {}).get("training", {})
    dataset_cfg = (config or {}).get("dataset", {})
    transforms_cfg = (config or {}).get("transforms", {})

    max_epochs = override(max_epochs, training_cfg, "max_epochs", int)
    batch_size = override(batch_size, training_cfg, "batch_size", int)
    lr = override(lr, training_cfg, "lr", float)
    ckpt_every = override(ckpt_every, training_cfg, "ckpt_every", int)
    logdir = override(logdir, training_cfg, "logdir")
    device_pref = override(device_pref, training_cfg, "device")
    num_seq = override(num_seq, dataset_cfg, "num_seq", int)
    T = override(T, dataset_cfg, "T", int)
    T = override(T, dataset_cfg, "seq_len", int)  # accept either name
    noise_std = override(noise_std, transforms_cfg, "add_noise_std", float)

    cfg = KVAEConfig()
    device = resolve_device(device_pref)
    print(f"Using device: {device}")

    raw_train_loader, raw_val_loader = build_dataloaders(cfg, dataset_cfg, batch_size=batch_size, num_seq=num_seq, T=T)
    train_loader = PreprocessedLoader(raw_train_loader, device, noise_std)
    val_loader = PreprocessedLoader(raw_val_loader, device, noise_std)

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

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg.grad_clip_norm)
        val_loss   = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if ckpt_dir:
            # Save best checkpoint
            best_path = ckpt_dir / "best.pt"
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best_path, model, optimizer, epoch, train_loss, val_loss)
            # Optional periodic checkpoints
            if ckpt_every > 0 and epoch % ckpt_every == 0:
                ckpt_path = ckpt_dir / f"epoch-{epoch:03d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, epoch, train_loss, val_loss)


        out_dir = Path(logdir) if logdir else Path(".")
        reconstruct_and_save(model, val_loader, device, out_dir, prefix="vae")


if __name__ == "__main__":
    main()
