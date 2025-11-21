from torch import nn
import torch
from torch.nn import functional as F

from typing import Optional, Tuple, Dict

from kvae.model.config import KVAEConfig
from kvae.model.vae import Encoder, Decoder
from kvae.model.lgssm import LGSSM
from kvae.kalman.kalman_filter import DynamicsParameter, KalmanFilter
from kvae.model.lstm import DynamicsParameterNetwork
from kvae.utils.losses import vae_loss



class KVAE(nn.Module):
    """
    Kalman Variational Auto-Encoder
    Combines VAE for recognition with LGSSM for dynamics modeling

    TODO: Correct the implementation of this class.
    """
    
    def __init__(self, config: KVAEConfig):
        super().__init__()
        self.config = config
        
        # VAE components
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # LGSSM
        # self.lgssm = LGSSM(config)
        
        # Dynamics parameter network
        self.K = config.num_modes
        self.z_dim = config.z_dim
        self.a_dim = config.a_dim
        A_in = torch.randn(self.K, self.z_dim, self.z_dim) * 0.1
        B_in = torch.randn(self.K, self.z_dim, self.z_dim) * 0.1
        C_in = torch.randn(self.K, self.a_dim, self.z_dim) * 0.1
        dynamics_net = DynamicsParameter(A_in, B_in, C_in)
        #self.dynamics_net = DynamicsParameterNetwork(config)

        # Kalman filter
        mu0 = torch.zeros(self.z_dim, dtype=torch.float32)
        Sigma0 = torch.diag(torch.ones(self.z_dim))
        std_dyn = 1.0
        std_obs = 1.0
        self.kalman_filter = KalmanFilter(std_dyn, std_obs, mu0, Sigma0, dynamics_net)

    
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
        # alpha = self.dynamics_net(a_samples)
        
        # Kalman filtering
        # filtered = self.lgssm.kalman_filter(a_samples, alpha, u)
        if u is None:
            u = torch.zeros(x.shape[0], x.shape[1], self.z_dim, device=x.device)
        # Defensive clone: avoid downstream in-place mutations of `a_samples`
        # that could corrupt autograd for the encoder outputs.
        mus_filt, sigms_filt, mus_pred, sigmas_pred, A_list, B_list, C_list = self.kalman_filter.filter(a_samples.clone(), u)
        
        # Kalman smoothing (optional)
        if use_smoothing:
            mus_smooth, Sigmas_smooth, _, _, _ = self.kalman_filter.smooth(a_samples.clone(), u)
            z_mean = mus_smooth
        else:
            z_mean = mus_filt
        
        # Decode
        x_recon = self.decode_sequence(a_samples)
        
        return {
            'x_recon': x_recon,
            'a_samples': a_samples,
            'a_mu': a_mu,
            'a_logvar': a_logvar,
            'z_mean': z_mean,
            'alpha': (A_list, B_list, C_list),
            'filtered': (mus_filt, sigms_filt),
            'u': u,
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
        # recon_loss = F.mse_loss(outputs['x_recon'], x, reduction='sum')
        # recon_loss = recon_loss / batch_size * self.config.recon_weight
        
        # KL divergence for VAE: KL(q(a|x) || p(a|z))
        # This is approximated by computing p(a|z) using the LGSSM
        a_mu = outputs['a_mu']
        a_logvar = outputs['a_logvar']
        
        # Get predicted a from z through emission matrix
        z_mean = outputs['z_mean']
        a = outputs['a_samples']
        x_mu = outputs['x_recon']

        # FALTAN
        x_var = torch.tensor(self.config.noise_pixel_var, device=x.device, dtype=x_mu.dtype)    
        Sigmas_smooth = torch.eye(self.z_dim).unsqueeze(0).unsqueeze(0).repeat(batch_size, T, 1, 1).to(x.device)
        
        #_, _, C_t = self.lgssm.get_matrices(alpha)
        
        #a_pred = torch.bmm(C_t.view(-1, C_t.shape[-2], C_t.shape[-1]),
        #                  z_mean.view(-1, z_mean.shape[-1], 1)).squeeze(-1)
        #a_pred = a_pred.view(batch_size, T, -1)
        
        # KL term (simplified)
        #kl_loss = -0.5 * torch.sum(1 + a_logvar - a_mu.pow(2) - a_logvar.exp())
        #kl_loss = kl_loss / batch_size
        total, recon, kl = vae_loss(x, x_mu, x_var, a, a_mu, a_logvar, scale_reconstruction=self.config.scale_reconstruction)

        u = outputs['u']
        A_list, B_list, C_list = outputs['alpha']
        # Clone `a` defensively before passing to the KalmanFilter in case
        # downstream code performs any in-place operations that would
        # corrupt the autograd graph for `a` (this avoids the
        # "variable needed for gradient computation has been modified"
        # RuntimeError).
        elbo_kf = self.kalman_filter.elbo(z_mean, Sigmas_smooth, a.clone(), u, A_list, B_list, C_list)
        
        # Total loss
        # total = recon * scale_reconstruction + kl
        # total = -log_px_given_a*scale_reconstruction + log_qa_given_x
        elbo_total = elbo_kf - total 
        
        return {
            'total_loss': elbo_total,
            'recon_loss': recon,
            'kl_loss': kl,
            'elbo_kf': elbo_kf,
            'elbo_total': elbo_total,
        }
