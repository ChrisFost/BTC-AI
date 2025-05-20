#!/usr/bin/env python
"""
Test for gradient flow to calibration parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

# Test gradient flow to calibration parameters
def test_calibration_gradient_flow():
    """Test gradient flow to calibration parameters using a simpler approach"""
    # Create a simple model with calibration-like parameters
    class SimpleModel(nn.Module):
        def __init__(self, device='cpu'):
            super().__init__()
            self.base = nn.Linear(10, 1)
            
            # Create calibration parameters similar to the main model
            self.calib_scale = nn.Parameter(torch.ones(1, device=device))
            self.calib_bias = nn.Parameter(torch.zeros(1, device=device))
            
        def forward(self, x):
            # Basic prediction
            pred = self.base(x)
            std = torch.abs(pred) + 0.1  # Some base uncertainty
            
            # Apply calibration similar to the main model
            calibrated_std = std * self.calib_scale + self.calib_bias
            calibrated_std = F.softplus(calibrated_std)
            
            return pred, calibrated_std
    
    # Create test input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(5, 10, device=device)
    
    # Create model and move to device
    model = SimpleModel(device=device).to(device)
    
    # Ensure the model is in training mode
    model.train()
    
    # Run forward pass
    pred, std = model(x)
    
    # Create loss that depends on both prediction and calibrated std
    pred_loss = pred.mean()
    std_loss = std.pow(2).mean()
    loss = pred_loss + std_loss * 5.0  # Weight std more heavily
    
    # Zero gradients
    model.zero_grad()
    
    # Check gradients are None before backward
    assert model.calib_scale.grad is None, "Scale param already has gradients"
    assert model.calib_bias.grad is None, "Bias param already has gradients"
    
    # Backpropagate
    loss.backward()
    
    # Verify gradients flowed to calibration parameters
    assert model.calib_scale.grad is not None, "No gradients for scale param"
    assert model.calib_bias.grad is not None, "No gradients for bias param"
    
    # Verify gradients are non-zero
    assert torch.abs(model.calib_scale.grad).sum() > 0, "Zero gradient for scale param"
    assert torch.abs(model.calib_bias.grad).sum() > 0, "Zero gradient for bias param"
    
    # Log success
    print("✓ Calibration parameters received gradients")
    print(f"  Scale grad: {model.calib_scale.grad.item()}")
    print(f"  Bias grad: {model.calib_bias.grad.item()}")
    
    # Document the pattern that works
    print("\nPattern that ensures gradient flow:")
    print("1. Direct multiplication: std * scale_param + bias_param")
    print("2. Apply F.softplus() to the result")
    print("3. Include calibrated result in loss function with sufficient weight")

# Add this pattern to the main ActorCritic model documentation
def test_document_fix_for_actor_critic():
    """Document how to apply the fix to ActorCritic"""
    print("\nTo fix gradient flow in ActorCritic:")
    print("1. In both apply_calibration and forward methods, use:")
    print("   - scale_key = f\"{horizon_name}_scale\"")
    print("   - bias_key = f\"{horizon_name}_bias\"")
    print("2. When applying calibration:")
    print("   - Get params with: scale = self.calibration_params[scale_key]")
    print("   - Apply with direct multiplication: pred_std * scale + bias")
    print("   - Use F.softplus() to ensure positivity")
    print("3. Ensure the loss includes the calibrated std with sufficient weight")

if __name__ == "__main__":
    # Run the test directly
    test_calibration_gradient_flow()
    test_document_fix_for_actor_critic() 