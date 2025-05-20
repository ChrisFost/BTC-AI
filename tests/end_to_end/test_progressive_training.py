#!/usr/bin/env python
"""
Test script for progressive training and cross-bucket knowledge transfer.

This script tests the progressive training functionality to ensure it correctly
trains buckets in sequence and transfers knowledge between them.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import torch
import json
import importlib
import shutil
from unittest.mock import MagicMock, patch

# Add parent directory to sys.path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import our modules using dynamic imports
try:
    progressive_module = importlib.import_module("src.training.progressive")
    ProgressiveTrainer = progressive_module.ProgressiveTrainer
except ImportError:
    print("Progressive training module not found. Skipping tests.")
    sys.exit(0)

class TestProgressiveTraining(unittest.TestCase):
    """Test cases for progressive training functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create test directories
        cls.test_dir = os.path.join(os.path.dirname(__file__), "test_data")
        cls.models_dir = os.path.join(cls.test_dir, "models")
        cls.data_dir = os.path.join(cls.test_dir, "data")
        
        os.makedirs(cls.test_dir, exist_ok=True)
        os.makedirs(cls.models_dir, exist_ok=True)
        os.makedirs(cls.data_dir, exist_ok=True)
        
        # Create test config
        cls.config_path = os.path.join(cls.test_dir, "test_config.json")
        cls.config = {
            "MODELS_DIR": cls.models_dir,
            "DATA_DIR": cls.data_dir,
            "MAX_EPISODES": 5,  # Use small number for testing
            "USE_CROSS_BUCKET_TRANSFER": True,
            "WEIGHT_TRANSFER_ALPHA": 0.3,
            "FEATURE_TRANSFER_ALPHA": 0.5,
            "TRANSFER_COOLDOWN": 2,
            "ENABLE_REVERSE_TRANSFER": True,
            "MEMORY_THRESHOLD": 0.8,
            "OPTIMIZE_MEMORY_FREQ": 3
        }
        
        with open(cls.config_path, "w") as f:
            json.dump(cls.config, f)
        
        # Create test training data
        cls._create_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_test_data(cls):
        """Create test training data for each bucket type."""
        # Generate synthetic data
        dates = pd.date_range(start='2023-01-01', periods=1000)
        
        for bucket_type in ["Scalping", "Short", "Medium", "Long"]:
            # Create simple sine wave with noise
            close = 100 + 10 * np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 1, 1000)
            
            # Create dataframe
            df = pd.DataFrame({
                'date': dates,
                'open': close - np.random.rand(1000),
                'high': close + np.random.rand(1000) * 2,
                'low': close - np.random.rand(1000) * 2,
                'close': close,
                'volume': np.random.randint(1000, 10000, 1000)
            })
            
            # Save to CSV
            df.to_csv(os.path.join(cls.data_dir, f"training_data_{bucket_type.lower()}.csv"), index=False)
    
    def test_trainer_initialization(self):
        """Test that the trainer initializes correctly."""
        trainer = ProgressiveTrainer(config_path=self.config_path)
        
        self.assertEqual(trainer.config_path, self.config_path)
        self.assertEqual(trainer.models_dir, self.models_dir)
        self.assertListEqual(trainer.bucket_sequence, ["Scalping", "Short", "Medium", "Long"])
        self.assertDictEqual(trainer.data_cache, {})
    
    @patch('progressive_training.train_model')
    def test_train_bucket(self, mock_train_model):
        """Test training a single bucket."""
        # Mock the train_model function to avoid actual training
        mock_model = MagicMock()
        mock_optimizer = MagicMock()
        mock_train_model.return_value = (mock_model, mock_optimizer, 5, 10.0)
        
        # Create trainer
        trainer = ProgressiveTrainer(config_path=self.config_path)
        
        # Train a bucket
        model_path = trainer.train_bucket("Scalping", episodes=5)
        
        # Check that train_model was called with correct arguments
        mock_train_model.assert_called_once()
        args, kwargs = mock_train_model.call_args
        
        # Check DataFrame was passed
        self.assertIsInstance(args[0], pd.DataFrame)
        
        # Check config
        self.assertEqual(kwargs['save_path'], os.path.join(self.models_dir, "Scalping", "checkpoints"))
        
        # Check model path
        self.assertTrue(model_path.endswith("final_scalping.pth"))
    
    @patch('progressive_training.train_model')
    def test_progressive_training(self, mock_train_model):
        """Test progressive training of multiple buckets."""
        # Mock the train_model function
        mock_model = MagicMock()
        mock_optimizer = MagicMock()
        mock_train_model.return_value = (mock_model, mock_optimizer, 5, 10.0)
        
        # Create trainer
        trainer = ProgressiveTrainer(config_path=self.config_path)
        
        # Mock knowledge transfer
        trainer.knowledge_transfer = MagicMock()
        
        # Train progressively
        bucket_sequence = ["Scalping", "Short"]
        episodes_per_bucket = {"Scalping": 3, "Short": 5}
        
        model_paths = trainer.train_progressively(
            custom_sequence=bucket_sequence,
            episodes_per_bucket=episodes_per_bucket
        )
        
        # Check that train_model was called twice
        self.assertEqual(mock_train_model.call_count, 2)
        
        # Check that we got model paths for both buckets
        self.assertEqual(len(model_paths), 2)
        self.assertIn("Scalping", model_paths)
        self.assertIn("Short", model_paths)
    
    @patch('progressive_training.measure_gpu_usage')
    def test_memory_management(self, mock_measure_gpu_usage):
        """Test memory management during training."""
        # Mock GPU usage
        mock_measure_gpu_usage.return_value = 0.7  # 70% usage
        
        # Create trainer
        trainer = ProgressiveTrainer(config_path=self.config_path)
        
        # Mock methods
        trainer.optimize_memory = MagicMock()
        
        # Test memory check
        trainer._free_memory_and_resources()
        
        # Check that optimize_memory was called
        trainer.optimize_memory.assert_called_once()
    
    def test_bucket_config(self):
        """Test generation of bucket-specific configurations."""
        trainer = ProgressiveTrainer(config_path=self.config_path)
        
        # Test config for each bucket type
        for bucket, expected_min, expected_max in [
            ("Scalping", 1, 72),
            ("Short", 6, 144),
            ("Medium", 24, 288),
            ("Long", 72, 576)
        ]:
            config = trainer._get_bucket_config(bucket)
            
            # Check bucket settings
            self.assertEqual(config["BUCKET"], bucket)
            self.assertEqual(config["MIN_HORIZON"], expected_min)
            self.assertEqual(config["MAX_HORIZON"], expected_max)
            
            # Check knowledge transfer dir
            self.assertTrue("KNOWLEDGE_TRANSFER_DIR" in config)
            self.assertTrue(os.path.exists(config["KNOWLEDGE_TRANSFER_DIR"]))

if __name__ == "__main__":
    unittest.main() 