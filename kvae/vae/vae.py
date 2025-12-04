import numpy as np
import os
import torch.nn as nn

from typing import Tuple

from kvae.vae.config import KVAEConfig
import torch


class Encoder(nn.Module):
    """Convolutional encoder that maps images to latent encodings a_t"""
    
    def __init__(self, config: KVAEConfig):
        super().__init__()
        self.config = config
        
        # Build convolutional layers
        layers = []
        in_channels = config.img_channels
        for out_channels in config.encoder_channels:
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=self.config.encoder_kernel_size,
                    stride=self.config.encoder_stride,
                    padding=self.config.encoder_padding,
                ),
                nn.ReLU()
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size after convolutions
        self.flat_size = self._get_flat_size()
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(self.flat_size, config.a_dim)
        self.fc_var = nn.Sequential(
            nn.Linear(self.flat_size, config.a_dim),
            nn.Sigmoid()
        )
    
    def _get_flat_size(self):
        """Calculate size after convolutions"""
        x = torch.zeros(1, self.config.img_channels, 
                       self.config.img_size, self.config.img_size)
        x = self.conv_layers(x)
        return int(np.prod(x.shape[1:]))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images [batch, channels, height, width]
        Returns:
            mu, var: Mean and log variance of q(a|x)
        """
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        var = self.fc_var(h)
        return mu, self.config.noise_emission * var


class Decoder(nn.Module):
    """Convolutional decoder that reconstructs images from latent encodings a_t"""
    
    def __init__(self, config: KVAEConfig):
        super().__init__()
        self.config = config
        
        # Calculate initial spatial size
        self.init_size = config.img_size // (2 ** len(config.decoder_channels))
        self.init_channels = config.decoder_channels[0]
        
        # Linear layer to expand latent to initial feature maps
        self.fc = nn.Linear(config.a_dim, 
                           self.init_channels * self.init_size * self.init_size)
        # Build upsampling stack using Sub-Pixel (PixelShuffle) layers.
        # Each upsampling step increases spatial size by factor r (2) via
        # Conv2d -> PixelShuffle(r). We perform one upsample per entry in
        # `decoder_channels` so that init_size * (2**len(decoder_channels)) == img_size.
        layers = []
        channels = config.decoder_channels.copy()
        r = 2  # upsampling factor per step

        # For each intermediate upsampling step: produce channels[i+1] * r^2
        # features and then rearrange with PixelShuffle to get channels[i+1]
        for i in range(len(channels) - 1):
            in_c = channels[i]
            out_c = channels[i+1]
            layers.extend([
                nn.Conv2d(in_c, out_c * (r * r), kernel_size=3, padding=1),
                nn.PixelShuffle(r),
                nn.ReLU()
            ])

        # Final conv to map to image channels and upsample one last time
        layers.append(nn.Conv2d(channels[-1], config.img_channels * (r * r), kernel_size=3, padding=1))
        layers.append(nn.PixelShuffle(r))

        self.deconv_layers = nn.Sequential(*layers)
    
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: Latent encodings [batch, a_dim]
        Returns:
            x_recon: Reconstructed images [batch, channels, height, width]
        """
        h = self.fc(a)
        h = h.view(h.size(0), self.init_channels, self.init_size, self.init_size)
        x_recon = self.deconv_layers(h)
        return x_recon


class VAE(nn.Module):
    """High-level VAE wrapper that composes Encoder + Decoder.

    Provides convenience methods for encoding, sampling and decoding so the
    training code and utilities can rely on a single class.
    """

    def __init__(self, config: KVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def encode(self, x: torch.Tensor):
        """Encode images.

        Args:
            x: [N, C, H, W]
        Returns:
            mu, var: [N, a_dim]
        """
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, a: torch.Tensor):
        """Decode latents to image means.

        Args:
            a: [N, a_dim]
        Returns:
            x_recon_mu: [N, C, H, W]
        """
        return self.decoder(a)

    def forward(self, x: torch.Tensor) -> dict:
        """Run full VAE on input sequences.

        Input:
            x: [B, T, C, H, W]
        Returns dict with keys:
            'x_recon', 'x_recon_mu', 'x_recon_var', 'a_vae', 'a_mu', 'a_var'
        """
        B, T = x.shape[:2]
        x_flat = x.view(-1, *x.shape[2:])  # [B*T, C, H, W]
        mu, var = self.encode(x_flat)
        a = self.reparameterize(mu, var)
        x_recon_mu = self.decode(a)  # logits if Bernoulli, mean if Gaussian

        if self.config.out_distr.lower() == "bernoulli":
            x_recon = torch.sigmoid(x_recon_mu)
        else:
            x_recon = x_recon_mu

        # scalar reconstruction variance from config
        x_recon_var = torch.tensor(self.config.noise_pixel_var, device=x.device, dtype=x_recon_mu.dtype)

        # reshape back
        x_recon = x_recon_mu.view(B, T, *x_recon_mu.shape[1:])
        x_recon_mu = x_recon_mu.view(B, T, *x_recon_mu.shape[1:])
        mu = mu.view(B, T, -1)
        var = var.view(B, T, -1)
        a = a.view(B, T, -1)

        return {
            'x_recon': x_recon,
            'x_recon_mu': x_recon_mu,
            'x_recon_var': x_recon_var,
            'a_vae': a,
            'a_mu': mu,
            'a_var': var,
        }

    def sample_from_prior(self, n: int = 1, device: str | None = None) -> torch.Tensor:
        """Sample images by drawing latents from the prior N(0,I) and decoding.

        Returns:
            samples: [n, C, H, W]
        """
        device = device or next(self.parameters()).device
        a = torch.randn(n, self.config.a_dim, device=device)
        x_mu = self.decode(a)
        return x_mu

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, config: KVAEConfig | None = None, device: str = 'cpu'):
        """Construct a VAE instance and load encoder/decoder weights from a checkpoint.

        Accepts Lightning-style checkpoints (dict with 'state_dict') or plain state_dict files.
        If `config` is None the default `KVAEConfig()` will be used; for exact parity
        you should pass the same config used during training.
        """
        if config is None:
            config = KVAEConfig()
        vae = cls(config)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)

        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # pull encoder/decoder keys (support both 'encoder.' and 'model.encoder.' prefixes)
        enc = {k.split('encoder.',1)[-1]: v for k, v in state_dict.items() if 'encoder.' in k}
        dec = {k.split('decoder.',1)[-1]: v for k, v in state_dict.items() if 'decoder.' in k}

        if not enc or not dec:
            # fallback: maybe checkpoints saved top-level keys for encoder/decoder
            enc = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            dec = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}

        if enc:
            vae.encoder.load_state_dict(enc)
        if dec:
            vae.decoder.load_state_dict(dec)

        vae.to(device)
        return vae
