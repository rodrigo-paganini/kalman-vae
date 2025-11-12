import torch
import torch.nn.functional as F
from kvae.model.config import KVAEConfig
from kvae.model.vae import VAE


def _make_dummy_batch(cfg, B=2, T=4):
    C, H, W = cfg.img_channels, cfg.img_size, cfg.img_size
    return torch.randn(B, T, C, H, W)


def test_deterministic_forward_with_seed():
    cfg = KVAEConfig()
    torch.manual_seed(1234)
    vae1 = VAE(cfg)
    x1 = _make_dummy_batch(cfg, B=2, T=3)
    out1 = vae1(x1)
    # repeat same seed -> same init + same input -> identical outputs
    torch.manual_seed(1234)
    vae2 = VAE(cfg)
    x2 = _make_dummy_batch(cfg, B=2, T=3)
    out2 = vae2(x2)

    assert out1['x_recon_mu'].shape == out2['x_recon_mu'].shape
    assert torch.allclose(out1['x_recon_mu'], out2['x_recon_mu'], atol=1e-6), "Forward is not deterministic with fixed seed"


def test_deterministic_forward_with_seed():
    '''
    Test that VAE forward doesn't change over time. If VAE behavior changes,
    update `tests/fixtures/out1.pt` with the new output.
    '''
    cfg = KVAEConfig()
    torch.manual_seed(1234)
    vae1 = VAE(cfg)
    x1 = _make_dummy_batch(cfg, B=2, T=3)
    out1 = vae1(x1)

    out2 = torch.load('tests/fixtures/out1.pt')

    assert out1['x_recon_mu'].shape == out2['x_recon_mu'].shape, "Output shape has changed. Update the fixture if this is intentional."
    assert torch.allclose(out1['x_recon_mu'], out2['x_recon_mu'], atol=1e-6), "Output values have changed. Update the fixture if this is intentional."


def test_forward_backward_and_parameter_update():
    cfg = KVAEConfig()
    torch.manual_seed(0)
    vae = VAE(cfg)
    vae.train()

    x = _make_dummy_batch(cfg, B=2, T=3)
    out = vae(x)

    # outputs finite and correct shapes
    recon = out['x_recon_mu']
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()

    # simple scalar loss and backward
    loss = F.mse_loss(recon, x)
    assert torch.isfinite(loss).item(), "Loss is not finite"

    loss.backward()

    # check at least some gradients are nonzero
    grads = [p.grad for p in vae.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients were computed"
    total_grad_norm = sum(g.abs().sum().item() for g in grads)
    assert total_grad_norm > 0, "All gradients are zero"

    # optimizer step updates parameters
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    params_before = [p.clone().detach() for p in vae.parameters() if p.requires_grad]
    opt.step()
    params_after = [p.clone().detach() for p in vae.parameters() if p.requires_grad]

    # at least one parameter tensor changed
    changed = any(not torch.allclose(a, b) for a, b in zip(params_before, params_after))
    assert changed, "Optimizer step did not change model parameters"