#!/usr/bin/env python
"""
Verification script for gradient flow to all model parameters including calibration.
"""
import torch
import sys
import os
import importlib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use dynamic import for models
models_module = importlib.import_module("src.models.models")
ActorCritic = models_module.ActorCritic
PolicyNetwork = models_module.PolicyNetwork
load_model = models_module.load_model
save_model = models_module.save_model

def verify_gradient_flow():
    """Verify that gradients flow to all model parameters including calibration."""
    print("Verifying gradient flow to all model parameters...")
    
    # Create model
    model = ActorCritic(
        input_dim=32,
        hidden_size=64,
        horizons=[12, 36, 72, 144],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Ensure model is in training mode
    model.train()
    
    # Enable calibration
    model.set_calibration_enabled(True)
    
    # Create input data
    x = torch.randn(2, 10, 32, device=model.device)
    
    # Forward pass
    outputs = model(x)
    
    # Create loss with all outputs
    loss = outputs['value'].mean()
    loss += outputs['actions_mean'].pow(2).mean()
    loss += outputs['actions_std'].pow(2).mean()
    
    # Include all predictions in loss
    for h in model.horizons:
        horizon_name = f"h{h}"
        loss += outputs['predictions'][horizon_name].mean()
        loss += outputs['prediction_stds'][horizon_name].pow(2).mean() * 10.0
        loss += outputs['confidence'][horizon_name].pow(2).mean() * 5.0
    
    # Zero gradients
    model.zero_grad()
    
    # Backpropagate
    loss.backward()
    
    # Check gradients
    missing_grads = []
    calibration_grads = {}
    
    # Print gradient information for calibration parameters
    print("\nCalibration Parameters:")
    for name, param in model.named_parameters():
        if 'calibration_params' in name or 'calibration_bias' in name:
            if param.grad is None:
                print(f"  ✗ {name}: NO GRADIENT")
                missing_grads.append(name)
            else:
                grad_val = param.grad.abs().sum().item()
                print(f"  ✓ {name}: {grad_val:.6f}")
                calibration_grads[name] = grad_val
    
    # Print gradient information for confidence parameters
    print("\nConfidence Parameters:")
    for name, param in model.named_parameters():
        if 'conf_heads' in name:
            if param.grad is None:
                print(f"  ✗ {name}: NO GRADIENT")
                missing_grads.append(name)
            else:
                grad_val = param.grad.abs().sum().item()
                print(f"  ✓ {name}: {grad_val:.6f}")
    
    # Final result
    if not missing_grads:
        print("\n✓ SUCCESS: Gradients flow to all model parameters")
    else:
        print("\n✗ FAILURE: Some parameters are missing gradients:")
        for name in missing_grads:
            print(f"  - {name}")

if __name__ == "__main__":
    verify_gradient_flow() 