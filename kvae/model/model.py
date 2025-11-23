from torch import nn
import torch

from kvae.vae.vae import Encoder, Decoder
from kvae.kalman.kalman_filter import KalmanFilter
from kvae.kalman.dyn_param import DynamicsParameter
from kvae.vae.losses import vae_loss


class KVAE(nn.Module):
    """
    Kalman Variational Auto-Encoder
    Combines VAE for recognition with LGSSM for dynamics modeling

    TODO: Correct the implementation of this class.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # VAE components
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # Dynamics parameter network
        self.K = config.num_modes
        self.z_dim = config.z_dim
        self.a_dim = config.a_dim
        A_in = torch.randn(self.K, self.z_dim, self.z_dim) * 0.1 # TODO: why?
        B_in = torch.randn(self.K, self.z_dim, self.z_dim) * 0.1
        C_in = torch.randn(self.K, self.a_dim, self.z_dim) * 0.1
        dynamics_net = DynamicsParameter(A_in, B_in, C_in)

        # Kalman filter
        mu0 = torch.zeros(self.z_dim, dtype=torch.float32)
        Sigma0 = torch.diag(torch.ones(self.z_dim))
        std_dyn = 1.0
        std_obs = 1.0
        self.kalman_filter = KalmanFilter(std_dyn, std_obs, mu0, Sigma0, dynamics_net)

    
    def reparameterize(self, mu, var):
            std = torch.sqrt(var + 1e-6)
            eps = torch.randn_like(std)
            return mu + eps * std

    
    def encode_sequence(self, x):
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
        mu, var = self.encoder(x_flat)
        
        # Sample
        a_samples = self.reparameterize(mu, var)
        
        # Reshape back to sequences
        a_samples = a_samples.view(batch_size, T, -1)
        mu = mu.view(batch_size, T, -1)
        var = var.view(batch_size, T, -1)
        
        return a_samples, mu, var
    

    def decode_sequence(self, a):
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
    

    def forward(self, x, u=None):
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
        a_samples, a_mu, a_var = self.encode_sequence(x)

        # Kalman filtering
        if u is None:
            u = torch.zeros(x.shape[0], x.shape[1], self.z_dim, device=x.device)

        # Kalman smoothing (optional)
        mus_smooth, Sigmas_smooth, A_list, B_list, C_list = self.kalman_filter.smooth(a_samples.clone(), u.clone())

        # Decode
        x_recon = self.decode_sequence(a_samples)
        
        return {
            'x_recon': x_recon,
            'a_samples': a_samples,
            'a_mu': a_mu,
            'a_var': a_var,
            'mus_smooth': mus_smooth,
            'Sigmas_smooth': Sigmas_smooth,
            'ABC': (A_list, B_list, C_list),
            'u': u,
        }
    
    
    def compute_loss(self, x, outputs):
        batch_size, T = x.shape[:2]

        a      = outputs['a_samples']      # [B, T, a_dim]
        a_mu   = outputs['a_mu']           # [B, T, a_dim]
        a_var  = outputs['a_var']          # [B, T, a_dim]

        mus_smooth             = outputs['mus_smooth']
        Sigmas_smooth          = outputs['Sigmas_smooth']
        A_list, B_list, C_list = outputs['ABC']

        # Controls
        if 'u' in outputs:
            u = outputs['u']
        else:
            u = torch.zeros(batch_size, T, self.z_dim, device=x.device, dtype=x.dtype)

        # Reconstruction distribution p(x | a)
        x_mu  = outputs['x_recon']                          # [B, T, C, H, W]
        x_var = torch.tensor(
            self.config.noise_pixel_var,
            device=x.device,
            dtype=x_mu.dtype
        )

        #  VAE ELBO
        vae_elbo, recon, entropy = vae_loss(
            x, x_mu, x_var,
            a, a_mu, a_var,
            scale_reconstruction=self.config.scale_reconstruction,
        )

        # Kalman ELBO
        elbo_kf = self.kalman_filter.elbo(
            mus_smooth, Sigmas_smooth,
            a, u, A_list, B_list, C_list
        )

        # Combine
        elbo_total = elbo_kf + vae_elbo        # sum of ELBO contributions
        loss = -elbo_total                     # minimize negative ELBO

        return {
            "loss": loss,
            "elbo_total": elbo_total,
            "elbo_kf": elbo_kf,
            "elbo_vae_total": vae_elbo,
            "recon": recon,
            "entropy": entropy,
        }

