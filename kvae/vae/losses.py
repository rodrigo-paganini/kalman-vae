
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

    Returns:
        log_px_given_a: log p(x|a) averaged over batch
        log_qa_given_x: log q(a|x) averaged over batch
    """
    log_lik = log_gaussian(x, x_mu, x_var)

    log_lik = log_lik.sum((1,2,3,4))
    log_px_given_a = log_lik.mean()

    log_qa_given_x = torch.sum(log_gaussian(a, a_mu, a_var), (1,2))
    log_qa_given_x = log_qa_given_x.mean()

    return log_px_given_a, log_qa_given_x

def vae_loss(
    x: torch.Tensor,
    x_mu: torch.Tensor,
    x_var: torch.Tensor,
    a: torch.Tensor,
    a_mu: torch.Tensor,
    a_var: torch.Tensor,
    scale_reconstruction: float = 0.3,
):
    B, T, C, H, W = x.shape
    denom = B * T

    log_px_given_a, log_qa_given_x = log_likelihood(x, x_mu, x_var, a, a_mu, a_var)

    # Normalized terms 
    recon   = -scale_reconstruction * (log_px_given_a / denom)
    entropy = - (log_qa_given_x / denom)

    # ELBO_vae normalized the same way
    vae_elbo = (scale_reconstruction * log_px_given_a - log_qa_given_x) / denom

    return vae_elbo, recon, entropy