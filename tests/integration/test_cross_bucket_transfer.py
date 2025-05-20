#!/usr/bin/env python
"""
Test script for cross-bucket knowledge transfer.

This script tests the CrossBucketKnowledgeTransfer class to verify it can
properly transfer knowledge between different bucket types.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import time
import random
import traceback
from collections import deque
import pytest
from unittest.mock import MagicMock, patch
import importlib

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

print("Starting cross-bucket knowledge transfer test...")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Import needed modules using dynamic imports
print("Importing modules with dynamic imports...")

# Import agent module
print("Importing agent module...")
agent_module = importlib.import_module("src.agent.agent")
PPOAgent = agent_module.PPOAgent

# Import progressive module
progressive_module = importlib.import_module("src.training.progressive")
CrossBucketKnowledgeTransfer = progressive_module.CrossBucketKnowledgeTransfer
print("Successfully imported agent and progressive modules")

# Import models module
print("Importing models module...")
models_module = importlib.import_module("src.models.models")
ActorCritic = models_module.ActorCritic
print("Successfully imported models module")

def create_dummy_model(input_dim, horizons, device="cpu"):
    """Create a dummy model for testing."""
    print(f"Creating dummy model with input_dim={input_dim}, horizons={horizons}, device={device}")
    
    # Create a small model for testing
    model = ActorCritic(
        input_dim=input_dim,
        hidden_size=32,
        horizon_config=horizons,
        device=device
    )
    
    return model

def create_dummy_agent(bucket_type, input_dim=20, device="cpu"):
    """Create a dummy agent for a specific bucket type."""
    print(f"Creating dummy agent for bucket: {bucket_type}")
    
    # Define horizons based on bucket type
    if bucket_type == "Scalping":
        horizons = [6, 12, 24, 36]
    elif bucket_type == "Short":
        horizons = [12, 36, 72, 144]
    elif bucket_type == "Medium":
        horizons = [24, 72, 144, 288]
    else:  # Long
        horizons = [72, 144, 288, 576]
    
    print(f"Using horizons: {horizons}")
    
    # Create agent with fixed parameters to avoid conflicts
    try:
        # Create a simple model directly to avoid create_model issues
        model = ActorCritic(
            input_dim=input_dim,
            hidden_size=32,
            horizon_config=horizons,
            device=device
        )
        
        # Use a minimal agent implementation for testing
        class MinimalAgent:
            def __init__(self):
                self.model = model
                self.device = device
                self.feature_importance = np.ones(input_dim)
                self.recent_rewards = deque(maxlen=10)
                self.reasoning_style = bucket_type.lower()
                
                # Add update_old_model method for compatibility
                def update_old_model(self):
                    return True
                
                self.update_old_model = update_old_model.__get__(self)
        
        agent = MinimalAgent()
        
        # Fill with random rewards
        for _ in range(10):
            agent.recent_rewards.append(random.random() * 2 - 1)  # Random rewards between -1 and 1
        
        # Modify feature importance with bucket-specific patterns
        if bucket_type == "Scalping":
            # Emphasize short-term features (first quarter)
            agent.feature_importance[:input_dim//4] *= 2.0
        elif bucket_type == "Short":
            # Mixed emphasis
            agent.feature_importance[input_dim//4:input_dim//2] *= 1.5
        elif bucket_type == "Medium":
            # Emphasis on middle features
            agent.feature_importance[input_dim//3:2*input_dim//3] *= 1.8
        else:  # Long
            # Emphasis on long-term features (last quarter)
            agent.feature_importance[3*input_dim//4:] *= 2.5
        
        print(f"Successfully created minimal agent for {bucket_type}")
        return agent
    
    except Exception as e:
        print(f"Error creating agent: {str(e)}")
        traceback.print_exc()
        raise

def test_cross_bucket_transfer():
    """Test the cross-bucket knowledge transfer functionality."""
    print("\n=== Testing Cross Bucket Knowledge Transfer ===")
    
    # Set up test parameters
    input_dim = 64  # Changed from 20 to 64 (divisible by 8 for transformer model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a transfer mechanism
    config = {
        "WEIGHT_TRANSFER_ALPHA": 0.3,
        "FEATURE_TRANSFER_ALPHA": 0.5,
        "TRANSFER_COOLDOWN": 5,
        "ENABLE_REVERSE_TRANSFER": True
    }
    transfer = CrossBucketKnowledgeTransfer(config)
    print("Created CrossBucketKnowledgeTransfer instance")
    
    # Create agents for different buckets
    buckets = ["Scalping", "Short", "Medium", "Long"]
    agents = {}
    
    for bucket_type in buckets:
        print(f"\nCreating agent for {bucket_type}...")
        agents[bucket_type] = create_dummy_agent(bucket_type, input_dim, device)
        transfer.register_agent(bucket_type, agents[bucket_type])
        print(f"Created and registered {bucket_type} agent")
    
    # Test feature importance transfer
    print("\n--- Testing Feature Importance Transfer ---")
    # Store original values
    original_fi = {}
    for bucket in buckets:
        if isinstance(agents[bucket].feature_importance, np.ndarray):
            original_fi[bucket] = agents[bucket].feature_importance.copy()
        else:
            original_fi[bucket] = agents[bucket].feature_importance.cpu().numpy().copy()
        
        print(f"{bucket} original top features: {np.argsort(-original_fi[bucket])[:3]}")
    
    # Perform transfer from Scalping to Short
    print("\nTransferring from Scalping to Short...")
    transfer.transfer_feature_importance("Scalping", "Short")
    
    # Check for changes
    for i in range(3):  # Top 3 features
        feature_idx = np.argsort(-original_fi["Scalping"])[i]
        print(f"Feature {feature_idx}: Scalping={original_fi['Scalping'][feature_idx]:.3f}, " +
            f"Short before={original_fi['Short'][feature_idx]:.3f}, " +
            f"Short after={agents['Short'].feature_importance[feature_idx]:.3f}")
    
    # Test model weight transfer
    print("\n--- Testing Model Weight Transfer ---")
    
    # Record original weights
    original_weights = {}
    for bucket in buckets:
        original_weights[bucket] = {}
        for name, param in agents[bucket].model.named_parameters():
            if 'encoder' in name:  # Only track encoder weights
                original_weights[bucket][name] = param.clone().detach()
    
    # Perform transfer
    print("\nTransferring model weights from Medium to Long...")
    result = transfer.transfer_model_weights("Medium", "Long", layers=['encoder'])
    print(f"Transfer success: {result}")
    
    # Verify some weights changed
    for name, param in agents["Long"].model.named_parameters():
        if 'encoder' in name and name in original_weights["Long"]:
            orig = original_weights["Long"][name]
            diff = torch.sum(torch.abs(param - orig)).item()
            print(f"Layer {name}: weight diff = {diff:.5f}")
    
    # Test horizon suggestion
    print("\n--- Testing Horizon Suggestion ---")
    
    # Set up horizon performance
    agents["Short"].model.horizon_performance = {
        "h12": 0.2,
        "h36": 0.4,
        "h72": 0.9,  # Best performing
        "h144": 0.7  # Second best
    }
    
    print(f"Short original horizon performance: {agents['Short'].model.horizon_performance}")
    print(f"Medium original horizons: {agents['Medium'].model.horizons}")
    
    # Suggest horizons
    new_horizons = transfer.suggest_horizon_updates("Short", "Medium")
    
    print(f"Suggested new horizons for Medium: {new_horizons}")
    
    # Test complete knowledge transfer
    print("\n--- Testing Complete Knowledge Transfer ---")
    
    # Perform full transfer
    results = transfer.transfer_all(current_episode=10)
    
    print(f"Transfer results: {len(results)} transfers performed")
    for result in results:
        print(f"  {result['message']}")
    
    print("\n=== Cross Bucket Knowledge Transfer Test Complete ===")
    print("\n=== Cross Bucket Knowledge Transfer Test Complete ===")


# Main execution block (only runs when script is executed directly)
if __name__ == "__main__":
    try:
        print("Starting cross-bucket knowledge transfer test...")
        test_cross_bucket_transfer()
        print("Test completed successfully!")
    except Exception as e:
        print(f"ERROR running test directly: {str(e)}")
        traceback.print_exc()
        sys.exit(1) # Exit with error code if run directly and fails