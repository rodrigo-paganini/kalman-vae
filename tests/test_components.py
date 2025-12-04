# TODO: restore when the components are rebuilt

# from kvae.model.model import KVAE
# from kvae.vae.vae import Encoder, Decoder
# from kvae.kalman.dyn_param import DynamicsParameterNetwork
# from kvae.vae.config import KVAEConfig
# import torch
# import torch.nn.functional as F

# def test_kvae_components():
#     """Test individual components of KVAE"""
#     print("Testing KVAE Components...")
#     print("=" * 60)
    
#     # Configuration
#     config = KVAEConfig(
#         img_channels=1,
#         img_size=32,
#         a_dim=2,
#         z_dim=4,
#         num_modes=3
#     )
    
#     batch_size = 4
#     T = 10
    
#     # Test data
#     x = torch.randn(batch_size, T, config.img_channels, 
#                    config.img_size, config.img_size)
    
#     print(f"\n1. Testing Encoder...")
#     encoder = Encoder(config)
#     x_single = x[:, 0]  # Test single frame
#     mu, logvar = encoder(x_single)
#     print(f"   Input shape: {x_single.shape}")
#     print(f"   Output mu shape: {mu.shape}")
#     print(f"   Output logvar shape: {logvar.shape}")
#     assert mu.shape == (batch_size, config.a_dim), "Encoder output shape mismatch"
#     print("   ✓ Encoder working correctly")
    
#     print(f"\n2. Testing Decoder...")
#     decoder = Decoder(config)
#     a = torch.randn(batch_size, config.a_dim)
#     x_recon = decoder(a)
#     print(f"   Input shape: {a.shape}")
#     print(f"   Output shape: {x_recon.shape}")
#     assert x_recon.shape == x_single.shape, "Decoder output shape mismatch"
#     print("   ✓ Decoder working correctly")
    
#     print(f"\n3. Testing LGSSM...")
#     lgssm = LGSSM(config)
#     a_seq = torch.randn(batch_size, T, config.a_dim)
#     alpha = F.softmax(torch.randn(batch_size, T, config.num_modes), dim=-1)
#     filtered = lgssm.kalman_filter(a_seq, alpha)
#     print(f"   Input a sequence shape: {a_seq.shape}")
#     print(f"   Filtered means shape: {filtered['means'].shape}")
#     print(f"   Filtered covs shape: {filtered['covs'].shape}")
#     assert filtered['means'].shape == (batch_size, T, config.z_dim)
#     print("   ✓ Kalman filter working correctly")
    
#     smoothed = lgssm.kalman_smoother(filtered['means'], filtered['covs'], alpha)
#     print(f"   Smoothed means shape: {smoothed['means'].shape}")
#     assert smoothed['means'].shape == (batch_size, T, config.z_dim)
#     print("   ✓ Kalman smoother working correctly")
    
#     print(f"\n4. Testing Dynamics Parameter Network...")
#     dyn_net = DynamicsParameterNetwork(config)
#     alpha_pred = dyn_net(a_seq)
#     print(f"   Input shape: {a_seq.shape}")
#     print(f"   Output alpha shape: {alpha_pred.shape}")
#     print(f"   Alpha sum (should be ~1.0): {alpha_pred[0, 0].sum().item():.4f}")
#     assert alpha_pred.shape == (batch_size, T, config.num_modes)
#     assert torch.allclose(alpha_pred.sum(dim=-1), torch.ones(batch_size, T), atol=1e-5)
#     print("   ✓ Dynamics network working correctly")
    
#     print(f"\n5. Testing Complete KVAE...")
#     kvae = KVAE(config)
#     outputs = kvae(x)
#     print(f"   Input shape: {x.shape}")
#     print(f"   Reconstruction shape: {outputs['x_recon'].shape}")
#     print(f"   a_samples shape: {outputs['a_samples'].shape}")
#     print(f"   z_mean shape: {outputs['z_mean'].shape}")
#     assert outputs['x_recon'].shape == x.shape
#     print("   ✓ KVAE forward pass working correctly")
    
#     print(f"\n6. Testing Loss Computation...")
#     losses = kvae.compute_loss(x, outputs)
#     print(f"   Total loss: {losses['total_loss'].item():.4f}")
#     print(f"   Recon loss: {losses['recon_loss'].item():.4f}")
#     print(f"   KL loss: {losses['kl_loss'].item():.4f}")
#     assert losses['total_loss'].requires_grad, "Loss should require gradients"
#     print("   ✓ Loss computation working correctly")


# if __name__ == "__main__":
#     test_kvae_components()
#     print("All tests passed! ✓")
#     print("=" * 60)