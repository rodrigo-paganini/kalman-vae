
import torch
import torch.nn.functional as F
from kvae.vae.config import KVAEConfig

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
    beta: float = 1.0, 
    mask: torch.Tensor | None = None,
    out_distr: str = "gaussian",
):
    B, T, C, H, W = x.shape
    if mask is None:
        mask_ = torch.ones(B, T, device=x.device, dtype=x.dtype)
    else:
        mask_ = mask.to(device=x.device, dtype=x.dtype)
        if mask_.shape != (B, T):
            mask_ = mask_.view(B, T)
    denom = mask_.sum().clamp(min=1.0)

    if out_distr.lower() == "bernoulli":
        bce = F.binary_cross_entropy_with_logits(x_mu, x, reduction="none")
        log_px_per_frame = -bce.sum(dim=(2, 3, 4))
        log_px_given_a = (log_px_per_frame * mask_).sum()
        log_q_per_frame = log_gaussian(a, a_mu, a_var).sum(dim=-1)
        log_qa_given_x = (log_q_per_frame * mask_).sum()
    else:
        log_px_given_a, log_qa_given_x = log_likelihood(
            x, x_mu, x_var, a, a_mu, a_var, mask=mask
        )

    # Compute Log Prior p(a) ~ N(0, 1) 
    # To compute KL divergence we need log p(a)
    # # KL: KL = log q(a|x) - log p(a)
    zeros = torch.zeros_like(a)
    ones = torch.ones_like(a)
    log_p_per_frame = log_gaussian(a, zeros, ones).sum(dim=-1)
    log_pa = (log_p_per_frame * mask_).sum()

    # ELBO = Recon - Beta * KL
    # ELBO = Recon + Beta * (log p(a) - log q(a|x))
    
    recon_term = (log_px_given_a / denom)
    
    regularization_term = (log_pa - log_qa_given_x) / denom 

    vae_elbo = scale_reconstruction * recon_term + beta * regularization_term

    return vae_elbo, recon_term, regularization_term

class LinearScheduler:
    def __init__(self, config: KVAEConfig):
    
        self.start_epoch = config.start_epoch
        self.end_epoch = config.end_epoch
        self.start_val = config.start_val
        self.end_val = config.end_val

    def get_beta(self, current_epoch: int) -> float:
        if current_epoch < self.start_epoch:
            return self.start_val
        
        if current_epoch >= self.end_epoch:
            return self.end_val
            
        # Interpolate linearly
        progress = (current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        
        # Formula de la linea: y = mx + b
        beta = self.start_val + progress * (self.end_val - self.start_val)
        
        return beta

def count_active_units(mu_tensor, threshold=1e-2):
        # Compute variance of the latent means across the batch
        # Active units are those whose variance exceeds the threshold.
        if mu_tensor.dim() == 3:
            # mu_tensor: [B, T, a_dim]
            mu = mu_tensor.reshape(-1, mu_tensor.shape[-1])  # [B*T, a_dim]
        else:
            mu = mu_tensor  # [B, a_dim]
        variances = mu.var(dim=0)  # a_dim
        active_units = (variances > threshold).sum().item()
        # print(f"Active units: {active_units}")
        # print(f"Latent variances: {variances}")
        return active_units, variances

#def vae_loss(
#    x: torch.Tensor,
#    x_mu: torch.Tensor,
#    x_var: torch.Tensor,
#    a: torch.Tensor,
#    a_mu: torch.Tensor,
#    a_var: torch.Tensor,
#    scale_reconstruction: float = 0.3,
#    beta = None,
#    mask: torch.Tensor | None = None,
#    out_distr: str = "gaussian",
#):
#    B, T, C, H, W = x.shape
#    if mask is None:
#        mask_ = torch.ones(B, T, device=x.device, dtype=x.dtype)
#    else:
#        mask_ = mask.to(device=x.device, dtype=x.dtype)
#        if mask_.shape != (B, T):
#            mask_ = mask_.view(B, T)
#    denom = mask_.sum().clamp(min=1.0)
#
#    if out_distr.lower() == "bernoulli":
#        # x_mu is treated as logits for Bernoulli
#        bce = F.binary_cross_entropy_with_logits(x_mu, x, reduction="none")
#        log_px_per_frame = -bce.sum(dim=(2, 3, 4))  # [B,T]
#        log_px_given_a = (log_px_per_frame * mask_).sum()
#        # q(a|x) stays Gaussian
#        log_q_per_frame = log_gaussian(a, a_mu, a_var).sum(dim=-1)
#        log_qa_given_x = (log_q_per_frame * mask_).sum()
#    else:
#        log_px_given_a, log_qa_given_x = log_likelihood(
#            x, x_mu, x_var,
#            a, a_mu, a_var,
#            mask=mask,
#        )
#
#    recon_logprob = (log_px_given_a / denom)  
#    entropy      = -(log_qa_given_x / denom)                         
#    vae_elbo     = scale_reconstruction * recon_logprob + entropy
#
#    return vae_elbo, recon_logprob, entropy