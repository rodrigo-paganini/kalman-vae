from torch import nn
import torch
from torch.nn import functional as F

from typing import Optional, Tuple, Dict

from kvae.utils import KVAEConfig
from kvae.vae import Encoder, Decoder
from kvae.lgssm import LGSSM
from kvae.lstm import DynamicsParameterNetwork



class KVAE(nn.Module):
    """
    Kalman Variational Auto-Encoder
    Combines VAE for recognition with LGSSM for dynamics modeling
    """
    
    def __init__(self, config: KVAEConfig):
        super().__init__()
        self.config = config
        
        # VAE components
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # LGSSM
        self.lgssm = LGSSM(config)
        
        # Dynamics parameter network
        self.dynamics_net = DynamicsParameterNetwork(config)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode full sequence with VAE
        
        Args:
            x: Input sequence [batch, T, channels, height, width]
        Returns:
            a_samples, a_mu, a_logvar
        """
        batch_size, T = x.shape[:2]
        
        # Flatten batch and time for encoding
        x_flat = x.view(-1, *x.shape[2:])
        mu, logvar = self.encoder(x_flat)
        
        # Sample
        a_samples = self.reparameterize(mu, logvar)
        
        # Reshape back to sequences
        a_samples = a_samples.view(batch_size, T, -1)
        mu = mu.view(batch_size, T, -1)
        logvar = logvar.view(batch_size, T, -1)
        
        return a_samples, mu, logvar
    
    def decode_sequence(self, a: torch.Tensor) -> torch.Tensor:
        """
        Decode sequence from latent encodings
        
        Args:
            a: Latent encodings [batch, T, a_dim]
        Returns:
            x_recon: Reconstructed sequence
        """
        batch_size, T = a.shape[:2]
        
        # Flatten for decoding
        a_flat = a.view(-1, a.shape[-1])
        x_recon_flat = self.decoder(a_flat)
        
        # Reshape back
        x_recon = x_recon_flat.view(batch_size, T, *x_recon_flat.shape[1:])
        
        return x_recon
    
    def forward(self, x: torch.Tensor, u: Optional[torch.Tensor] = None,
                use_smoothing: bool = True) -> Dict:
        """
        Full forward pass
        
        Args:
            x: Input sequence [batch, T, channels, height, width]
            u: Control inputs [batch, T, z_dim] (optional)
            use_smoothing: Whether to use Kalman smoothing
        
        Returns:
            Dictionary with reconstructions and latent variables
        """
        # Encode sequence
        a_samples, a_mu, a_logvar = self.encode_sequence(x)
        
        # Get dynamics parameters
        alpha = self.dynamics_net(a_samples)
        
        # Kalman filtering
        filtered = self.lgssm.kalman_filter(a_samples, alpha, u)
        
        # Kalman smoothing (optional)
        if use_smoothing:
            smoothed = self.lgssm.kalman_smoother(
                filtered['means'], filtered['covs'], alpha, u
            )
            z_mean = smoothed['means']
        else:
            z_mean = filtered['means']
        
        # Decode
        x_recon = self.decode_sequence(a_samples)
        
        return {
            'x_recon': x_recon,
            'a_samples': a_samples,
            'a_mu': a_mu,
            'a_logvar': a_logvar,
            'z_mean': z_mean,
            'alpha': alpha,
            'filtered': filtered
        }
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict) -> Dict:
        """
        Compute ELBO loss
        
        Args:
            x: Original input sequence
            outputs: Output dictionary from forward pass
        
        Returns:
            Dictionary with loss components
        """
        batch_size, T = x.shape[:2]
        
        # Reconstruction loss (downweighted as in paper)
        recon_loss = F.mse_loss(outputs['x_recon'], x, reduction='sum')
        recon_loss = recon_loss / batch_size * self.config.recon_weight
        
        # KL divergence for VAE: KL(q(a|x) || p(a|z))
        # This is approximated by computing p(a|z) using the LGSSM
        a_mu = outputs['a_mu']
        a_logvar = outputs['a_logvar']
        
        # Get predicted a from z through emission matrix
        z_mean = outputs['z_mean']
        alpha = outputs['alpha']
        _, _, C_t = self.lgssm.get_matrices(alpha)
        
        a_pred = torch.bmm(C_t.view(-1, C_t.shape[-2], C_t.shape[-1]),
                          z_mean.view(-1, z_mean.shape[-1], 1)).squeeze(-1)
        a_pred = a_pred.view(batch_size, T, -1)
        
        # KL term (simplified)
        kl_loss = -0.5 * torch.sum(1 + a_logvar - a_mu.pow(2) - a_logvar.exp())
        kl_loss = kl_loss / batch_size
        
        # Total loss
        total_loss = recon_loss + kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
