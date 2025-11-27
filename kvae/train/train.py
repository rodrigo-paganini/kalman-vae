import random
import numpy as np
from pathlib import Path
import torch
import numpy as np

from kvae.vae.config import KVAEConfig
from kvae.model.model import KVAE
from kvae.train.utils import build_dataloaders, save_checkpoint
from kvae.train.testing import reconstruct_and_save, kalman_prediction_test


def train_one_epoch(model, loader, optimizer, device, scheduler=None, kf_weight=1.0):
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
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    T = 20  
    only_vae_epochs = 5    
    kf_update_epochs = 5   

    cfg = KVAEConfig()
    print(f"Using device: {device}")

    pathfile_videos = "/Users/rodrigopaganini/master/data/pgm/kvae/box.npz"
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

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scheduler, kf_weight)        
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