import torch
from kvae.model.config import KVAEConfig
from kvae.model.vae import VAE


def test_vae_forward_shapes():
    cfg = KVAEConfig()
    vae = VAE(cfg)
    B, T = 2, 5
    C, H, W = cfg.img_channels, cfg.img_size, cfg.img_size
    x = torch.randn(B, T, C, H, W)

    out = vae(x)
    # keys
    assert 'x_recon' in out
    assert 'x_recon_mu' in out
    assert 'x_recon_var' in out
    assert 'a_vae' in out
    assert 'a_mu' in out
    assert 'a_var' in out

    # shapes
    assert out['x_recon'].shape == (B, T, C, H, W)
    assert out['x_recon_mu'].shape == (B, T, C, H, W)
    # x_recon_var is a scalar tensor (broadcastable)
    assert torch.is_tensor(out['x_recon_var'])
    assert out['a_vae'].shape == (B, T, cfg.a_dim)
    assert out['a_mu'].shape == (B, T, cfg.a_dim)
    assert out['a_var'].shape == (B, T, cfg.a_dim)


def test_sample_from_prior_shape():
    cfg = KVAEConfig()
    vae = VAE(cfg)
    n = 4
    samples = vae.sample_from_prior(n=n)
    assert samples.shape == (n, cfg.img_channels, cfg.img_size, cfg.img_size)
