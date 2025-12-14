from torch import nn
import torch

from kvae.vae.vae import Encoder, Decoder
from kvae.kalman.kalman_filter import KalmanFilter
from kvae.kalman import dyn_param as base_dyn_param
from kvae.kalman import switch_dyn_param as switch_dyn_param
from kvae.vae.losses import vae_loss


class KVAE(nn.Module):
    """
    Kalman Variational Auto-Encoder
    Combines VAE for recognition with LGSSM for dynamics modeling
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
        self.u_dim = config.u_dim

        A_in = torch.eye(self.z_dim).unsqueeze(0).repeat(self.K, 1, 1)  # [K, z, z]
        init_std = config.init_kf_matrices 
        B_in = torch.randn(self.K, self.z_dim, self.u_dim) * init_std
        C_in = torch.randn(self.K, self.a_dim, self.z_dim) * init_std
        # Choose either switching or original KVAE dynamics
        if config.use_switching_dynamics:
            prior = switch_dyn_param.StickyRegimePrior(self.K, p_stay=config.sticky_p_stay)
            regime_post = switch_dyn_param.MarkovVariationalRegimePosterior(
                self.K, input_dim=self.a_dim, hidden_size=config.dynamics_hidden_dim
            )
            Q_in = torch.eye(self.z_dim, dtype=torch.float32).unsqueeze(0).repeat(self.K, 1, 1) * config.noise_transition
            dynamics_net = switch_dyn_param.SwitchingDynamicsParameter(
                A_in, B_in, C_in,
                Q=Q_in,
                prior=prior,
                hidden_lstm=config.dynamics_hidden_dim,
                markov_regime_posterior=regime_post,
            )
            dynamics_net.tau = config.tau_init
        else:
            dynamics_net = base_dyn_param.DynamicsParameter(
                A_in, B_in, C_in,
                hidden_lstm=config.dynamics_hidden_dim,
            )

        mu0 = torch.zeros(self.z_dim, dtype=torch.float32)
        Sigma0 = torch.eye(self.z_dim, dtype=torch.float32) * config.init_cov 

        # Noise from config (note: config values are variances )
        std_dyn = (config.noise_transition) ** 0.5   
        std_obs = (config.noise_emission) ** 0.5     
        # Kalman filter
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
    

    def forward(self, x, u=None, mask=None):
        """
        Full forward pass
        
        Args:
            x: Input sequence [batch, T, channels, height, width]
            u: Control inputs [batch, T, u_dim] (optional)
            mask: [B, T] observation mask (1=observed, 0=missing)
        
        Returns:
            Dictionary with reconstructions and latent variables
        """
        # Encode sequence
        a_samples, a_mu, a_var = self.encode_sequence(x)

        if u is None:
            u = torch.zeros(x.shape[0], x.shape[1], self.u_dim, device=x.device, dtype=x.dtype)

        # Reset LSTM state before running KF on this sequence
        self.kalman_filter.dyn_params.reset_state()
        
        # Kalman smoothing
        (mus_smooth, Sigmas_smooth,
        mus_filt, Sigmas_filt,
        mus_pred, Sigmas_pred,
        A_list, B_list, C_list) = self.kalman_filter.smooth(
            a_samples.clone(), u.clone(), mask=mask
        )
        
        # Decode
        x_logits = self.decode_sequence(a_samples)
        if self.config.out_distr.lower() == "bernoulli":
            x_recon = torch.sigmoid(x_logits)
        else:
            x_recon = x_logits

        return {
            "x_recon":       x_recon,
            "x_logits":      x_logits,
            "a_samples":     a_samples,
            "a_mu":          a_mu,
            "a_var":         a_var,
            "mus_smooth":    mus_smooth,
            "Sigmas_smooth": Sigmas_smooth,
            "mus_filt":      mus_filt,
            "Sigmas_filt":   Sigmas_filt,
            "mus_pred":      mus_pred,
            "Sigmas_pred":   Sigmas_pred,
            "ABC":           (A_list, B_list, C_list),
            "u":             u,
        }
        
    
    def compute_loss(self, x, outputs, kf_weight=1.0, vae_weight=1.0, mask=None):
        batch_size, T = x.shape[:2]

        a      = outputs["a_samples"]
        a_mu   = outputs["a_mu"]
        a_var  = outputs["a_var"]

        mus_smooth             = outputs["mus_smooth"]
        Sigmas_smooth          = outputs["Sigmas_smooth"]
        A_list, B_list, C_list = outputs["ABC"]

        # Controls
        u = outputs.get("u", torch.zeros(batch_size, T, self.u_dim, device=x.device, dtype=x.dtype))
        
        # Reconstruction distribution p(x | a)
        x_mu  = outputs.get("x_logits", outputs["x_recon"])
        x_var = torch.tensor(self.config.noise_pixel_var, device=x.device, dtype=x_mu.dtype)

        # VAE ELBO
        vae_elbo, recon, entropy = vae_loss(
            x, x_mu, x_var,
            a, a_mu, a_var,
            scale_reconstruction=self.config.scale_reconstruction,
            mask=mask,
            out_distr=self.config.out_distr,
        )
        
        # Kalman ELBO
        elbo_kf = self.kalman_filter.elbo(
            mus_smooth, Sigmas_smooth,
            a, u, A_list, B_list, C_list,
            mask=mask,
        )

        # Combine ELBOs
        elbo_total = vae_weight * vae_elbo + kf_weight * elbo_kf
        loss = -elbo_total

        return {
            "loss": loss,
            "elbo_total": elbo_total,
            "elbo_kf": elbo_kf,
            "elbo_vae_total": vae_elbo,
            "recon": recon,
            "kl": entropy,
        }

    @torch.no_grad()
    def impute(self, x, mask, u=None):
        """
        Impute missing data in sequence x given mask
        Args:
            x: Input sequence with missing data [B,T,C,H,W]
            mask: Observation mask [B,T] (1=observed, 0=missing)
            u: Control inputs [B,T,u_dim] (optional)
        Returns:
            Dictionary with:
                x_recon: VAE reconstruction from a_samples [B,T,C,H,W]
                x_imputed: Imputation using smoothed states [B,T,C,H,W]
                x_filtered: Imputation using filtered states [B,T,C,H,W]
                a_vae: VAE latent encodings [B,T,a_dim]
                a_imputed: Imputed latent encodings from smoothed states [B,T,a_dim]
                a_filtered: Imputed latent encodings from filtered states [B,T,a_dim]
        """
        self.eval()
        device = x.device
        B, T = x.shape[:2]

        mask = mask.to(device=device, dtype=x.dtype)

        # Forward pass with mask
        outputs = self.forward(x, u=u, mask=mask)

        a_vae                  = outputs["a_samples"]           # [B,T,a_dim]
        mus_smooth             = outputs["mus_smooth"]          # [B,T,n]
        mus_filt               = outputs["mus_filt"]            # [B,T,n]
        A_list, B_list, C_list = outputs["ABC"]       

        # Baseline decoder reconstruction from original a_samples
        x_recon_logits = self.decode_sequence(a_vae)       # [B,T,C,H,W]
        x_recon = torch.sigmoid(x_recon_logits) if self.config.out_distr.lower() == "bernoulli" else x_recon_logits

        # Compute a_imputed = C_t z_{t|1:T}
        z_smooth = mus_smooth                          # [B,T,n,1]
        a_imputed = (C_list @ z_smooth).squeeze(-1)    # [B,T,p] 
        # Uses smoothed latent states (past + future) then decodes.
        x_imputed_logits  = self.decode_sequence(a_imputed)
        x_imputed = torch.sigmoid(x_imputed_logits) if self.config.out_distr.lower() == "bernoulli" else x_imputed_logits

        # Compute a_filtered = C_t z_{t|1:t}
        z_filt = mus_filt                # [B,T,n,1]
        a_filtered = (C_list @ z_filt).squeeze(-1)     # [B,T,p]
        # Uses filtered latent states (past only) then decodes (online baseline)
        x_filtered_logits = self.decode_sequence(a_filtered)
        x_filtered = torch.sigmoid(x_filtered_logits) if self.config.out_distr.lower() == "bernoulli" else x_filtered_logits
    
        return {
            "x_recon":    x_recon,
            "x_imputed":  x_imputed,
            "x_filtered": x_filtered,
            "a_vae":      a_vae,
            "a_imputed":  a_imputed,
            "a_filtered": a_filtered,
        }
