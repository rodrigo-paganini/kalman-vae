import numpy as np
import torch.nn as nn

from typing import Tuple

from kvae.utils import KVAEConfig
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
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size after convolutions
        self.flat_size = self._get_flat_size()
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(self.flat_size, config.a_dim)
        self.fc_logvar = nn.Linear(self.flat_size, config.a_dim)
    
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
            mu, logvar: Mean and log variance of q(a|x)
        """
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


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
        
        # Build deconvolutional layers
        layers = []
        channels = config.decoder_channels.copy()
        for i in range(len(channels) - 1):
            layers.extend([
                nn.ConvTranspose2d(channels[i], channels[i+1], 
                                  kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            ])
        
        # Final layer to output image
        layers.append(
            nn.ConvTranspose2d(channels[-1], config.img_channels,
                              kernel_size=4, stride=2, padding=1)
        )
        
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