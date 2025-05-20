#!/usr/bin/env python
"""
Test script to verify gradient flow to calibration parameters in the ActorCritic model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

# Use dynamic import for models
models_module = importlib.import_module("src.models.models")
ActorCritic = models_module.ActorCritic
PolicyNetwork = models_module.PolicyNetwork
load_model = models_module.load_model
save_model = models_module.save_model

def test_gradient_flow():
    """Test that gradients flow properly to calibration parameters."""
    print("Testing gradient flow to calibration parameters...")
    
    # Initialize model
    input_dim = 32
    hidden_size = 64
    horizons = [12, 36, 72, 144]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ActorCritic(
        input_dim=input_dim, 
        hidden_size=hidden_size,
        horizons=horizons,
        device=device
    )
    
    # Generate sample input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    
    # Enable calibration
    model.set_calibration_enabled(True)
    
    # Ensure model is in training mode
    model.train()
    
    # Forward pass
    outputs = model(x)
    
    # Create a loss that depends on all outputs including calibrated std values
    loss = outputs['value'].mean()
    loss += outputs['actions_mean'].pow(2).mean()
    loss += outputs['actions_std'].pow(2).mean()
    
    # Add predictions and confidence to loss
    for h in model.horizons:
        horizon_name = f"h{h}"
        loss += outputs['predictions'][horizon_name].mean()
        # Give prediction_stds more weight in loss
        loss += outputs['prediction_stds'][horizon_name].pow(2).mean() * 10.0
        loss += outputs['confidence'][horizon_name].pow(2).mean() * 5.0
    
    # Zero gradients
    model.zero_grad()
    
    # Backpropagate
    loss.backward()
    
    # Check gradients for calibration parameters
    calibration_has_grad = True
    for h in horizons:
        horizon_name = f"h{h}"
        scale_key = f"{horizon_name}_scale"
        bias_key = f"{horizon_name}_bias"
        
        # Check scale parameter
        if scale_key in model.calibration_params:
            scale_param = model.calibration_params[scale_key]
            if scale_param.grad is None:
                print(f"ERROR: No gradient for {scale_key}")
                calibration_has_grad = False
            else:
                print(f"✓ {scale_key} has gradient: {scale_param.grad.item()}")
        
        # Check bias parameter
        if bias_key in model.calibration_bias:
            bias_param = model.calibration_bias[bias_key]
            if bias_param.grad is None:
                print(f"ERROR: No gradient for {bias_key}")
                calibration_has_grad = False
            else:
                print(f"✓ {bias_key} has gradient: {bias_param.grad.item()}")
    
    # Check confidence parameters
    for h in horizons:
        horizon_name = f"h{h}"
        conf_key = f"conf_{horizon_name}.0.weight"
        
        for name, param in model.named_parameters():
            if conf_key in name and param.requires_grad:
                if param.grad is None:
                    print(f"ERROR: No gradient for confidence parameter {name}")
                else:
                    print(f"✓ Confidence parameter {name} has gradient")
    
    # Final result
    if calibration_has_grad:
        print("\n✓ SUCCESS: Gradient flow to calibration parameters is working correctly")
    else:
        print("\n✗ FAILURE: Some calibration parameters are not receiving gradients")

if __name__ == "__main__":
    test_gradient_flow() 