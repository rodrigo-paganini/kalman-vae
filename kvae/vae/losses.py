
from sympy import denom
import torch

from kvae.vae.config import KVAEConfig

const_log_pdf = torch.tensor(- 0.5) * torch.log(torch.tensor(2) * torch.pi)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss components.

    Args:
        x: Original input image sequences [batch, seq_len, channels, height, width]
        x_mu: Reconstructed image means [batch, seq_len, channels, height, width
        x_var: Reconstructed image variances [batch, seq_len, channels, height, width]
        a: Sampled latent encodings [batch, seq_len, a_dim]
        a_mu: Latent means [batch, seq_len, a_dim]
        a_var: Latent variances [batch, seq_len, a_dim]
        scale_reconstruction: Scaling factor for reconstruction loss. Defaults to 0.3.

    Returns:
        total: Total VAE loss.
        recon: Reconstruction loss term.
        kl: KL divergence loss.
    """
    B, T, C, H, W = x.shape
    num_pixels = C * H * W
    denom = B * T * num_pixels # normalize to per (B*T*pixels)

    log_px_given_a, log_qa_given_x = log_likelihood(x, x_mu, x_var, a, a_mu, a_var)

    # Normalize 
    log_px_bt = log_px_given_a / denom
    log_qa_bt = log_qa_given_x / denom

    # Positive losses
    recon   = -log_px_bt       
    entropy = -log_qa_bt       

    # ELBO_vae = scale * log p(x|a) - log q(a|x)
    vae_elbo = scale_reconstruction * log_px_given_a - log_qa_given_x

    return vae_elbo, recon, entropy