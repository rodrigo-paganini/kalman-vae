import torch


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
        diff2 = diff2 * unobs_px
        return diff2.sum().item() / unobs_px.sum().item()
    

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