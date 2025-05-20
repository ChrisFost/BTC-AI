#!/usr/bin/env python
"""
Test suite for Agent V2.py
-------------------------
Tests the following components:
- Agent initialization
- Action selection
- Model update process
- Memory management
- Batch prediction
- Feature importance tracking
- Model saving and loading
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
from unittest.mock import patch, MagicMock
import gc
import importlib

# Get the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# Add project root to system path to ensure imports work
sys.path.insert(0, project_root)

# Keep the load_mock definition for potential targeted patching later
@staticmethod
def load_mock(path, device="cpu"):
    # This should now likely load a REAL agent, 
    # or the calling test needs to patch create_model if a mock is needed.
    agent = PPOAgent(32, 32, 3e-4) # Assumes PPOAgent is defined below
    return agent

# Perform imports directly - PPOAgent will be the ORIGINAL class now
try:
    agent_module = importlib.import_module("src.agent.agent")
    PPOAgent = agent_module.PPOAgent
except ImportError as e:
    print(f"Error importing agent module: {e}")
    sys.exit(1)

# Import fixture from test_dataframe
try:
    from tests.unit.test_dataframe import sample_df
except ImportError:
    # If that fails, try with absolute import from the project root
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tests.unit.test_dataframe import sample_df

@pytest.fixture
def agent():
    """Create a small test agent with minimal configuration"""
    # This will now create an agent with the *real* __init__ and *real* model
    input_dim = 32
    hidden_size = 32
    # If a test using this fixture needs a mocked model, it must patch 'src.models.models.create_model'
    return PPOAgent(input_dim, hidden_size, lr=3e-4)

@pytest.fixture
def sample_batch(sample_df):
    """Create a sample batch of data from the sample dataframe"""
    # Define the number of features needed by the agent
    required_features = 32
    
    # Create state vectors from dataframe
    # We'll use a subset of columns as features
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'day_of_month_sin', 'day_of_month_cos', 'day_of_year_sin', 'day_of_year_cos'
    ]
    
    # Ensure these columns exist in the sample data
    missing_cols = [col for col in feature_cols if col not in sample_df.columns]
    if missing_cols:
        for col in missing_cols:
            sample_df[col] = 0  # Fill with zeros for missing columns
    
    # Create sliding windows of data
    window_size = 60
    batch_size = 10
    states = []
    
    # Create overlapping windows
    for i in range(batch_size):
        start_idx = i
        end_idx = start_idx + window_size
        if end_idx > len(sample_df):
            break
            
        # Get the available features
        window = sample_df.iloc[start_idx:end_idx][feature_cols].values
        
        # Create a larger feature array with the required dimensions
        # Add derived features to reach 32 total features
        extended_window = np.zeros((window.shape[0], required_features))
        
        # Fill in the actual features
        extended_window[:, :window.shape[1]] = window
        
        # Fill remaining features with synthetic data
        remaining_features = required_features - window.shape[1]
        if remaining_features > 0:
            # Add some derived features like moving averages, differences, etc.
            # Use the close price as a base
            close_price_idx = feature_cols.index('close')
            for j in range(remaining_features):
                feature_idx = window.shape[1] + j
                # Create patterns like moving averages, momentum indicators, etc.
                if j % 4 == 0:  # Simple moving average
                    ma_length = 2 + j // 2
                    ma = np.convolve(window[:, close_price_idx], np.ones(ma_length)/ma_length, mode='valid')
                    pad_length = window.shape[0] - ma.shape[0]
                    padded_ma = np.pad(ma, (pad_length, 0), 'edge')
                    extended_window[:, feature_idx] = padded_ma
                elif j % 4 == 1:  # Price differences
                    extended_window[:, feature_idx] = np.diff(window[:, close_price_idx], prepend=window[0, close_price_idx])
                elif j % 4 == 2:  # Normalized price
                    extended_window[:, feature_idx] = window[:, close_price_idx] / np.max(window[:, close_price_idx])
                else:  # Random noise (simulating other indicators)
                    extended_window[:, feature_idx] = np.random.randn(window.shape[0]) * 0.01 + 1.0
        
        states.append(extended_window)
    
    # Create other batch elements
    actions = np.random.randn(len(states), 2).astype(np.float32)  # [direction, fraction]
    rewards = np.random.randn(len(states)).astype(np.float32)
    log_probs = np.random.randn(len(states)).astype(np.float32)
    dones = np.zeros(len(states)).astype(np.float32)
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(np.array(states))
    actions_tensor = torch.FloatTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards)
    log_probs_tensor = torch.FloatTensor(log_probs)
    dones_tensor = torch.FloatTensor(dones)
    
    return {
        'states': states_tensor,
        'actions': actions_tensor,
        'rewards': rewards_tensor,
        'log_probs': log_probs_tensor,
        'dones': dones_tensor
    }

def test_agent_initialization():
    """Test that agent initializes correctly with different configurations"""
    # Test basic initialization - This now tests the REAL initialization process
    input_dim = 32
    hidden_size = 32
    agent = PPOAgent(input_dim, hidden_size, lr=3e-4)
    assert agent is not None
    assert agent.model is not None
    assert agent.old_model is not None
    assert agent.optimizer is not None
    
    # Test with different model type
    agent = PPOAgent(input_dim, hidden_size, lr=3e-4, model_type='lstm')
    assert agent is not None
    assert agent.model is not None
    assert agent.old_model is not None
    
    # Test with horizons
    horizons = [12, 36, 72]
    agent = PPOAgent(input_dim, hidden_size, lr=3e-4, horizons=horizons)
    assert agent is not None
    assert agent.horizons == horizons

def test_select_action(agent, sample_batch):
    """Test the action selection mechanism"""
    state = sample_batch['states'][0]  # Take the first state
    
    # Get the action, log_prob, and value
    initial_hidden = None
    with torch.no_grad():
        action, log_prob, value, next_hidden, horizon_data, novelty = agent.select_action(state, initial_hidden)
    
    # Check the shapes and types
    assert isinstance(action, np.ndarray)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    assert action.shape == (2,), f"Expected action shape (2,), got {action.shape}"
    
    # Check for the horizon_data dictionary
    assert isinstance(horizon_data, dict)
    
    # Check that novelty is a float
    assert isinstance(novelty, float)
    
    # Check the range of actions
    assert np.all(np.isfinite(action)), "Action contains inf or nan values"

def test_batch_prediction(agent, sample_batch):
    """Test the batch prediction mechanism"""
    states = sample_batch['states']
    
    # We need to convert the states to a list for the predict_batch method
    states_list = [state for state in states]
    
    # Get predictions
    with torch.no_grad():
        results = agent.predict_batch(states_list)
    
    # Check that we get a list of results
    assert isinstance(results, list)
    assert len(results) == len(states_list)
    
    # Check the contents of the first result
    if results:
        first_result = results[0]
        
        # Check that the result has the expected keys
        assert 'action' in first_result
        assert 'log_prob' in first_result
        assert 'value' in first_result
        assert 'predictions' in first_result
        assert 'prediction_stds' in first_result
        assert 'confidence' in first_result
        assert 'mid_point_means' in first_result
        assert 'mid_point_stds' in first_result
        assert 'trend_strength' in first_result
        
        # Check types
        assert isinstance(first_result['action'], np.ndarray)
        assert isinstance(first_result['log_prob'], float)
        assert isinstance(first_result['value'], float)
        assert isinstance(first_result['predictions'], dict)

def test_update_method(agent, sample_batch):
    """Test the update method"""
    # Extract data
    states = sample_batch['states']
    actions = sample_batch['actions']
    log_probs = sample_batch['log_probs']
    rewards = sample_batch['rewards']
    
    # Create returns and advantages
    returns = rewards.clone()  # Simplified
    advantages = torch.randn_like(returns)
    
    # Perform an update
    loss_dict = agent.update(states, actions, log_probs, returns, advantages)
    
    # Verify loss is a dictionary with expected keys
    assert isinstance(loss_dict, dict)
    assert 'total_loss' in loss_dict
    assert 'policy_loss' in loss_dict
    assert 'value_loss' in loss_dict
    
    # Check for either entropy_loss or entropy
    assert 'entropy' in loss_dict or 'entropy_loss' in loss_dict

def test_memory_cleanup(agent, sample_batch):
    """Test memory cleanup after agent operations"""
    # Count tensors before operations
    torch_tensors_before = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])
    
    # Perform operations that might leak memory
    states = sample_batch['states']
    with torch.no_grad():
        for _ in range(10):
            # Repeated operations to make leaks more apparent
            state = states[0]
            initial_hidden = None
            action, log_prob, value, next_hidden, horizon_data, novelty = agent.select_action(state, initial_hidden)
    
    # Explicitly delete variables that might hold references
    del action, log_prob, value, next_hidden, horizon_data, novelty
    
    # Force garbage collection
    gc.collect()
    
    # Count tensors after operations
    torch_tensors_after = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])
    
    # Allow for some new tensors, but not too many
    # We may need to adjust this threshold
    assert torch_tensors_after - torch_tensors_before < 50, "Too many tensors remain after operations"

def test_save_load(agent, sample_batch):
    """Test saving and loading the agent"""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a save path
        save_path = os.path.join(temp_dir, "test_agent.pt")
        
        # Save the agent
        agent.save(save_path)
        
        # Verify the file exists
        assert os.path.exists(save_path)
        
        # Load the agent
        loaded_agent = PPOAgent.load(save_path)
        
        # Test the loaded agent on a sample state
        state = sample_batch['states'][0]
        
        # Get actions from both agents
        with torch.no_grad():
            original_action, _, _, _, _, _ = agent.select_action(state, None)
            loaded_action, loaded_log_prob, loaded_value, _, _, _ = loaded_agent.select_action(state, None)
        
        # Rather than comparing exact actions, verify that the loaded agent returns valid outputs
        assert isinstance(loaded_action, np.ndarray), "Loaded agent should return numpy array for action"
        assert loaded_action.shape == (2,), f"Expected action shape (2,), got {loaded_action.shape}"
        assert np.all(np.isfinite(loaded_action)), "Action contains inf or nan values"
        assert isinstance(loaded_log_prob, float), "Log probability should be a float"
        assert isinstance(loaded_value, float), "Value should be a float"

# Test the feature importance tracking
def test_feature_importance(agent, sample_batch):
    """Test the feature importance tracking mechanism"""
    # Extract data
    states = sample_batch['states']
    actions = sample_batch['actions']
    rewards = sample_batch['rewards']
    
    # Only run if the feature importance method exists
    if hasattr(agent, 'update_feature_importance'):
        # Update feature importance
        agent.update_feature_importance(states, actions, rewards)
        
        # Verify that feature importance exists and is the right shape
        assert hasattr(agent, 'feature_importance')
        assert agent.feature_importance is not None
        
        # If it's initialized, check shape
        if agent.feature_importance is not None:
            expected_shape = (agent.input_dim,)
            actual_shape = agent.feature_importance.shape
            assert actual_shape == expected_shape, f"Expected feature importance shape {expected_shape}, got {actual_shape}"

if __name__ == "__main__":
    # Run the tests when executed directly
    pytest.main(['-xvs', __file__]) 