from dataclasses import dataclass

from typing import Tuple

@dataclass
class KVAEConfig:
    """Configuration for KVAE model"""
    # Data dimensions
    img_channels: int = 1
    img_size: int = 32
    
    # Latent dimensions
    a_dim: int = 2  # Recognition latent space
    z_dim: int = 4  # Dynamics latent space
    
    # LGSSM parameters
    num_modes: int = 3  # K in the paper
    
    # VAE architecture
    encoder_channels: Tuple[int] = None
    encoder_kernel_size: int = 3
    encoder_stride: int = 2
    encoder_padding: int = 1
    decoder_channels: Tuple[int] = None
    decoder_kernel_size: int = 3
    decoder_stride: int = 2
    decoder_padding: int = 1

    # Dynamics network
    dynamics_hidden_dim: int = 64
    
    # Training parameters
    recon_weight: float = 0.3
    
    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [32, 32, 32]
        if self.decoder_channels is None:
            self.decoder_channels = [32, 32, 32]