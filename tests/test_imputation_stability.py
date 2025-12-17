"""
Regression tests to ensure imputation functionality remains stable during refactoring.
"""
import torch
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kvae.model.model import KVAE
from kvae.utils.config import KVAEConfig


def set_deterministic_weights(model, seed=42):
    """Set model weights deterministically for reproducible testing."""
    torch.manual_seed(seed)
    for param in model.parameters():
        if param.requires_grad:
            # Initialize with small random values
            param.data = torch.randn_like(param.data) * 0.01


def create_dummy_batch(batch_size=2, T=10, C=1, H=32, W=32, device='cpu'):
    """Create a dummy batch for testing."""
    torch.manual_seed(123)
    return {
        'images': torch.randn(batch_size, T, C, H, W, device=device),
    }


def test_lstm_imputation_outputs_unchanged():
    """
    Test that imputation produces the same outputs with fixed random weights.
    This acts as a regression test - if refactoring changes outputs, this fails.
    """
    device = 'cpu'
    cfg = KVAEConfig(dynamics_model="lstm")
    
    # Create model with deterministic weights
    model = KVAE(cfg).to(device)
    set_deterministic_weights(model, seed=42)
    model.eval()
    
    # Create dummy data
    batch = create_dummy_batch(batch_size=2, T=10, device=device)
    x = batch['images']
    B, T = x.shape[:2]
    
    # Create a planning mask (observe first 4, hide next 6)
    mask = torch.ones(B, T, device=device)
    mask[:, 4:10] = 0.0
    
    # Run imputation twice - should get identical results
    with torch.no_grad():
        with open('tests/fixtures/imputation_output_lstm.pt', 'rb') as f:
            try:
                out1 = torch.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Fixture file not found. Run `save_reference_outputs(model.impute(x, mask=mask), 'tests/fixtures/imputation_output_lstm.pt')` to create it." +\
                    "You can also run `pytest ./tests --no-stability` to skip these type of tests temporarily."
                )
        
        out2 = model.impute(x, mask=mask)
    
    # Check outputs are identical
    keys = ['x_recon', 'x_imputed', 'x_filtered']
    for key in keys:
        assert key in out1, f"Missing key: {key}"
        assert key in out2, f"Missing key: {key}"
        
        diff = torch.abs(out1[key] - out2[key]).max().item()
        assert diff < 1e-8, f"{key}: outputs differ by {diff}. If you are intentionally changing imputation, update the reference output."
    
    print("✓ Imputation produces identical outputs with same weights")


def test_imputation_outputs_unchanged():
    """
    Test that imputation produces the same outputs with fixed random weights.
    This acts as a regression test - if refactoring changes outputs, this fails.
    """
    device = 'cpu'
    cfg = KVAEConfig(dynamics_model="switching")
    
    # Create model with deterministic weights
    model = KVAE(cfg).to(device)
    set_deterministic_weights(model, seed=42)
    model.eval()
    
    # Create dummy data
    batch = create_dummy_batch(batch_size=2, T=10, device=device)
    x = batch['images']
    B, T = x.shape[:2]
    
    # Create a planning mask (observe first 4, hide next 6)
    mask = torch.ones(B, T, device=device)
    mask[:, 4:10] = 0.0
    
    # Run imputation twice - should get identical results
    with torch.no_grad():
        try:
            with open('tests/fixtures/imputation_output_switching.pt', 'rb') as f:
                out1 = torch.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Fixture file not found. Run `save_reference_outputs(model.impute(x, mask=mask), 'tests/fixtures/imputation_output_switching.pt')` to create it." +\
                "You can also run `pytest ./tests --no-stability` to skip these type of tests temporarily."
            )
        
        out2 = model.impute(x, mask=mask)
    
    # Check outputs are identical
    keys = ['x_recon', 'x_imputed', 'x_filtered']
    for key in keys:
        assert key in out1, f"Missing key: {key}"
        assert key in out2, f"Missing key: {key}"
        
        diff = torch.abs(out1[key] - out2[key]).max().item()
        assert diff < 1e-8, f"{key}: outputs differ by {diff}. If you are intentionally changing imputation, update the reference output."
    
    print("✓ Imputation produces identical outputs with same weights")


def save_reference_outputs(output_dict, filepath='tests/imputation_reference.npz'):
    """Save reference outputs to compare against after refactoring."""
    Path(filepath).parent.mkdir(exist_ok=True)
    
    data = {
        key: val.cpu().numpy() 
        for key, val in output_dict.items() 
        if isinstance(val, torch.Tensor)
    }
    np.savez(filepath, **data)
    print(f"✓ Saved reference outputs to {filepath}")


def test_against_reference(reference_path='tests/imputation_reference.npz'):
    """
    Compare current imputation outputs against saved reference.
    Run this after refactoring to ensure outputs haven't changed.
    """
    if not Path(reference_path).exists():
        print(f"⚠ No reference file found at {reference_path}")
        print("  Run with --save-reference first to create baseline")
        return
    
    device = 'cpu'
    cfg = KVAEConfig()
    
    # Create model with same deterministic weights
    model = KVAE(cfg).to(device)
    set_deterministic_weights(model, seed=42)
    model.eval()
    
    # Same dummy data
    batch = create_dummy_batch(batch_size=2, T=10, device=device)
    x = batch['images']
    B, T = x.shape[:2]
    
    mask = torch.ones(B, T, device=device)
    mask[:, 4:10] = 0.0
    
    # Current outputs
    with torch.no_grad():
        current = model.impute(x, mask=mask)
    
    # Load reference
    reference = np.load(reference_path)
    
    # Compare
    max_diff = 0.0
    for key in ['x_recon', 'x_imputed', 'x_filtered']:
        ref_arr = reference[key]
        cur_arr = current[key].cpu().numpy()
        
        diff = np.abs(ref_arr - cur_arr).max()
        max_diff = max(max_diff, diff)
        
        status = "✓" if diff < 1e-5 else "✗"
        print(f"{status} {key}: max diff = {diff:.2e}")
    
    if max_diff < 1e-5:
        print("\n✓ All outputs match reference (refactoring is safe!)")
    else:
        print(f"\n✗ Outputs differ from reference by up to {max_diff:.2e}")
        print("  This may indicate a breaking change in refactoring")
    
    assert max_diff < 1e-5, "Outputs changed after refactoring!"


def create_kvae_fixture(filename, device='cpu'):
    '''
    Utility function to create or update the saved fixture output.
    NEVER call this function during normal testing! It's just an auxiliary function to call when fixtures 
    need to be created or updated.
    Run this if VAE behavior changes intentionally.
    '''
    cfg = KVAEConfig(out_distr="gaussian")
    torch.manual_seed(1234)
    kvae = KVAE(cfg).to(device)
    x1 = create_dummy_batch(batch_size=2, T=10, device=device)
    out1 = kvae(x1)

    with open(filename, 'wb') as f:
        torch.save(out1, f)