
import torch
import torch.nn.functional as F

from kvae.vae.config import KVAEConfig
import math


def log_gaussian(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """
    Compute log N(x; mean, var).
    
    Args:
        x: Sample to evaluate [..., D]
        mean: Mean of the underlying gaussian distribution [..., D]
        var: Variance of the underlying gaussian distribution [..., D]
    Returns:
        log_pdf: Tensor of shape [...] with log probabilities
    """
    const_log_pdf = -0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=x.device, dtype=x.dtype))
    return const_log_pdf - torch.log(var) / 2 - torch.square(x - mean) / (2 * var)

def log_likelihood(
        x: torch.Tensor,
        x_mu: torch.Tensor,
        x_var: torch.Tensor,
        a: torch.Tensor,
        a_mu: torch.Tensor,
        a_var: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log likelihood terms for VAE loss.

    Args:
        x: Original input image sequences [batch, seq_len, channels, height, width]
        x_mu: Reconstructed image means [batch, seq_len, channels, height, width
        x_var: Reconstructed image variances [batch, seq_len, channels, height, width]
        a: Sampled latent encodings [batch, seq_len, a_dim]
        a_mu: Latent means [batch, seq_len, a_dim]
        a_var: Latent variances [batch, seq_len, a_dim]
        mask:  [B, T] with 1 for observed frames, 0 for missing.
               If None, all frames are treated as observed.

    Returns:
        log_px_given_a: log p(x|a) averaged over batch
        log_qa_given_x: log q(a|x) averaged over batch
    """
    B, T = x.shape[:2]

    log_lik_x_per_frame = log_gaussian(x, x_mu, x_var).sum(dim=(2, 3, 4))
    log_q_per_frame = log_gaussian(a, a_mu, a_var).sum(dim=-1)
    # Mask [B, T]
    if mask is None:
        mask_ = torch.ones(B, T, device=x.device, dtype=x.dtype)
    else:
        mask_ = mask.to(device=x.device, dtype=x.dtype)
        if mask_.shape != (B, T):
            mask_ = mask_.view(B, T)

    log_px_given_a = (log_lik_x_per_frame * mask_).sum()
    log_qa_given_x = (log_q_per_frame * mask_).sum()

    return log_px_given_a, log_qa_given_x

def vae_loss(
    x: torch.Tensor,
    x_mu: torch.Tensor,
    x_var: torch.Tensor,
    a: torch.Tensor,
    a_mu: torch.Tensor,
    a_var: torch.Tensor,
    scale_reconstruction: float = 0.3,
    mask: torch.Tensor | None = None,
):
    B, T, C, H, W = x.shape
    if mask is None:
        denom = B * T
    else:
        mask_ = mask.to(device=x.device, dtype=x.dtype)
        if mask_.shape != (B, T):
            mask_ = mask_.view(B, T)
        denom = mask_.sum().clamp(min=1.0)

    log_px_given_a, log_qa_given_x = log_likelihood(
        x, x_mu, x_var,
        a, a_mu, a_var,
        mask=mask,
    )

    recon   = -scale_reconstruction * (log_px_given_a / denom)
    entropy = -(log_qa_given_x / denom)

    vae_elbo = (scale_reconstruction * log_px_given_a - log_qa_given_x) / denom
    return vae_elbo, recon, entropy