"""Simple utility to load VAE weights from Lightning checkpoints.

Usage:
    from kvae.utils.weights import load_vae_weights
    
    # vae is any object with .encoder and .decoder attributes
    load_vae_weights(vae, 'runs/.../checkpoints/vae-best.ckpt')
"""
import os
import torch


def load_vae_weights(vae, checkpoint_path: str, device: str = 'cpu'):
    """Load encoder and decoder weights into a VAE instance from a checkpoint.
    
    Args:
        vae: Object with .encoder and .decoder attributes (nn.Module instances)
        checkpoint_path: Path to Lightning checkpoint (.ckpt file)
        device: Device to load weights on (default: 'cpu')
    
    Example:
        from kvae.model.vae import Encoder, Decoder
        from kvae.model.config import KVAEConfig
        from kvae.utils.weights import load_vae_weights
        
        cfg = KVAEConfig()
        
        class VAE:
            def __init__(self):
                self.encoder = Encoder(cfg)
                self.decoder = Decoder(cfg)
        
        vae = VAE()
        load_vae_weights(vae, 'runs/20241108-.../checkpoints/vae-best.ckpt')
    """
    if not hasattr(vae, 'encoder') or not hasattr(vae, 'decoder'):
        raise AttributeError('VAE instance must have .encoder and .decoder attributes')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Extract state_dict (Lightning checkpoints have a 'state_dict' key)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    # Extract encoder and decoder weights
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}
    
    if not encoder_state:
        raise RuntimeError(f"No encoder weights found in checkpoint (looked for 'encoder.*' keys)")
    if not decoder_state:
        raise RuntimeError(f"No decoder weights found in checkpoint (looked for 'decoder.*' keys)")
    
    # Load weights
    vae.encoder.load_state_dict(encoder_state)
    vae.decoder.load_state_dict(decoder_state)
    
    # Move to device
    vae.encoder.to(device)
    vae.decoder.to(device)
    
    print(f"✓ Loaded VAE weights from: {checkpoint_path}")
    print(f"  - Encoder: {len(encoder_state)} parameter groups")
    print(f"  - Decoder: {len(decoder_state)} parameter groups")
