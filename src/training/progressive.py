#!/usr/bin/env python
"""
Progressive Training Orchestration Module

This module implements progressive training strategies for different trading buckets,
allowing for sequential training and knowledge transfer between buckets.
"""

import os
import sys
import time
import json
import logging
import argparse
import importlib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch

# Add the current directory to sys.path to ensure module imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Get the project root directory
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# Add project root to system path to ensure imports work
sys.path.insert(0, project_root)

# Import utility functions
from src.utils.utils import log, measure_gpu_usage, optimize_memory
from src.training.training import train_model
from src.utils.trade_config import get_trade_config

# Define module-level logger
logger = logging.getLogger('progressive_training')
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Directory constants
MODELS_DIR_DEFAULT = os.path.join(os.path.dirname(current_dir), "Models")
DATA_DIR = os.path.join(os.path.dirname(current_dir), "Data")
CONFIG_FILE = os.path.join(current_dir, "config.json")

# Initialize TradeConfig
trade_config = get_trade_config()

class CrossBucketKnowledgeTransfer:
    """
    Handles knowledge transfer between different bucket models.
    
    This class provides mechanisms for transferring learned knowledge from one bucket
    model to another, facilitating progressive learning across different timeframes.
    
    Attributes:
        source_agent: The source agent to transfer knowledge from
        target_agent: The target agent to transfer knowledge to
        strategy: The transfer strategy to use ('weights', 'distillation', or 'feature')
        transfer_rate: The proportion of knowledge to transfer (0.0 to 1.0)
        shared_layers: List of shared layer names for partial transfer
        agents: Dictionary of registered agents by bucket type
        config: Configuration parameters for transfer
    """
    
    def __init__(self, config=None, source_agent=None, target_agent=None, strategy='weights', 
                 transfer_rate=0.5, shared_layers=None):
        """
        Initialize the knowledge transfer module.
        
        Args:
            config: Dictionary of configuration parameters
            source_agent: Agent to transfer knowledge from
            target_agent: Agent to receive knowledge
            strategy: Transfer strategy ('weights', 'distillation', or 'feature')
            transfer_rate: How much knowledge to transfer (0.0 to 1.0)
            shared_layers: List of layer names for partial transfer
        """
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.strategy = strategy
        self.transfer_rate = max(0.0, min(1.0, transfer_rate))  # Clamp to [0,1]
        self.shared_layers = shared_layers or []
        self.logger = logging.getLogger('knowledge_transfer')
        self.agents = {}  # Dictionary to store agents by bucket type
        
        # Parse configuration
        self.config = config or {}
        self.weight_transfer_alpha = self.config.get('WEIGHT_TRANSFER_ALPHA', 0.3)
        self.feature_transfer_alpha = self.config.get('FEATURE_TRANSFER_ALPHA', 0.5)
        self.transfer_cooldown = self.config.get('TRANSFER_COOLDOWN', 5)
        self.enable_reverse_transfer = self.config.get('ENABLE_REVERSE_TRANSFER', True)
    
    def register_agent(self, bucket_type, agent):
        """
        Register an agent for a specific bucket type.
        
        Args:
            bucket_type: The bucket identifier (e.g., 'Scalping', 'Short')
            agent: The agent to register
            
        Returns:
            bool: Whether registration was successful
        """
        if not agent:
            self.logger.warning(f"Attempted to register None agent for {bucket_type}")
            return False
            
        try:
            self.agents[bucket_type] = agent
            self.logger.info(f"Registered agent for bucket: {bucket_type}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register agent for {bucket_type}: {str(e)}")
            return False
    
    def get_agent(self, bucket_type):
        """
        Get the registered agent for a bucket type.
        
        Args:
            bucket_type: The bucket identifier
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(bucket_type, None)
    
    def transfer_feature_importance(self, source_bucket, target_bucket, alpha=None):
        """
        Transfer feature importance from source to target bucket.
        
        Args:
            source_bucket: Bucket type of the source agent
            target_bucket: Bucket type of the target agent
            alpha: The blend factor for importance transfer (None uses default)
            
        Returns:
            bool: Success status
        """
        alpha = alpha if alpha is not None else self.feature_transfer_alpha
        source_agent = self.get_agent(source_bucket)
        target_agent = self.get_agent(target_bucket)
        
        if not source_agent or not target_agent:
            self.logger.warning(f"Missing agent for transfer: source={source_bucket}, target={target_bucket}")
            return False
            
        try:
            # Get feature importances
            if hasattr(source_agent, 'feature_importance') and hasattr(target_agent, 'feature_importance'):
                source_fi = source_agent.feature_importance
                target_fi = target_agent.feature_importance
                
                # Convert to numpy if they're tensors
                if isinstance(source_fi, torch.Tensor):
                    source_fi = source_fi.cpu().detach().numpy()
                if isinstance(target_fi, torch.Tensor):
                    target_fi = target_fi.cpu().detach().numpy()
                
                # Blend the feature importance values
                new_fi = alpha * source_fi + (1 - alpha) * target_fi
                
                # Update the target agent
                if isinstance(target_agent.feature_importance, torch.Tensor):
                    device = target_agent.feature_importance.device
                    target_agent.feature_importance = torch.tensor(new_fi, device=device)
                else:
                    target_agent.feature_importance = new_fi
                
                self.logger.info(f"Transferred feature importance from {source_bucket} to {target_bucket}")
                return True
            else:
                self.logger.warning("Feature importance not available in agents")
                return False
                
        except Exception as e:
            self.logger.error(f"Feature importance transfer failed: {str(e)}")
            return False
    
    def transfer_model_weights(self, source_bucket, target_bucket, layers=None, alpha=None):
        """
        Transfer model weights from source to target bucket.
        
        Args:
            source_bucket: Bucket type of the source agent
            target_bucket: Bucket type of the target agent
            layers: List of layer name patterns to transfer (None for all compatible layers)
            alpha: The blend factor for weight transfer (None uses default)
            
        Returns:
            bool: Success status
        """
        alpha = alpha if alpha is not None else self.weight_transfer_alpha
        source_agent = self.get_agent(source_bucket)
        target_agent = self.get_agent(target_bucket)
        
        if not source_agent or not target_agent:
            self.logger.warning(f"Missing agent for weight transfer: source={source_bucket}, target={target_bucket}")
            return False
            
        try:
            # Get models
            if hasattr(source_agent, 'model') and hasattr(target_agent, 'model'):
                source_model = source_agent.model
                target_model = target_agent.model
            elif hasattr(source_agent, 'actor_critic') and hasattr(target_agent, 'actor_critic'):
                source_model = source_agent.actor_critic
                target_model = target_agent.actor_critic
            else:
                self.logger.warning("Models not available in agents")
                return False
                
            # Get state dictionaries
            source_dict = source_model.state_dict()
            target_dict = target_model.state_dict()
            
            # Find compatible layers
            transferable_layers = []
            
            # Filter by specified layer patterns
            if layers:
                for layer_name in layers:
                    for key in source_dict.keys():
                        if layer_name in key and key in target_dict and source_dict[key].size() == target_dict[key].size():
                            transferable_layers.append(key)
            else:
                # All compatible layers
                transferable_layers = [key for key in source_dict.keys() 
                                       if key in target_dict and source_dict[key].size() == target_dict[key].size()]
            
            if not transferable_layers:
                self.logger.warning("No compatible layers found for transfer")
                return False
                
            self.logger.info(f"Transferring {len(transferable_layers)} layers with alpha={alpha}")
            
            # Create new state dict with blended weights
            new_dict = target_dict.copy()
            for key in transferable_layers:
                new_dict[key] = source_dict[key] * alpha + target_dict[key] * (1 - alpha)
                
            # Load the updated weights
            target_model.load_state_dict(new_dict)
            
            # Update old model if using PPO
            if hasattr(target_agent, 'update_old_model'):
                target_agent.update_old_model()
                
            self.logger.info(f"Transferred model weights from {source_bucket} to {target_bucket}")
            return True
                
        except Exception as e:
            self.logger.error(f"Model weight transfer failed: {str(e)}")
            return False
    
    def suggest_horizon_updates(self, source_bucket, target_bucket):
        """
        Suggest horizon updates for the target bucket based on the source bucket's performance.
        
        Args:
            source_bucket: The source bucket with horizon performance data
            target_bucket: The target bucket to suggest horizons for
            
        Returns:
            list: Suggested new horizons for target bucket
        """
        source_agent = self.get_agent(source_bucket)
        target_agent = self.get_agent(target_bucket)
        
        if not source_agent or not target_agent:
            self.logger.warning(f"Missing agent for horizon suggestion: source={source_bucket}, target={target_bucket}")
            return []
            
        try:
            # Get models
            if hasattr(source_agent, 'model') and hasattr(target_agent, 'model'):
                source_model = source_agent.model
                target_model = target_agent.model
            elif hasattr(source_agent, 'actor_critic') and hasattr(target_agent, 'actor_critic'):
                source_model = source_agent.actor_critic
                target_model = target_agent.actor_critic
            else:
                self.logger.warning("Models not available in agents")
                return []
                
            # Check if source has horizon performance data
            if not hasattr(source_model, 'horizon_performance') or not source_model.horizon_performance:
                self.logger.warning(f"No horizon performance data for {source_bucket}")
                return []
                
            # Get current horizons
            if not hasattr(target_model, 'horizons'):
                self.logger.warning(f"No horizons attribute for {target_bucket}")
                return []
                
            current_horizons = target_model.horizons
            
            # Analyze source horizon performance
            perf = source_model.horizon_performance
            sorted_horizons = sorted(perf.items(), key=lambda x: x[1], reverse=True)
            
            # Extract best performing horizons
            best_horizons = []
            for key, value in sorted_horizons:
                if key.startswith('h'):
                    try:
                        horizon = int(key[1:])  # Extract number from 'h12' -> 12
                        best_horizons.append(horizon)
                    except ValueError:
                        continue
            
            if not best_horizons:
                return current_horizons
                
            # Keep some of the current horizons for stability
            num_to_keep = max(1, len(current_horizons) // 2)
            num_to_replace = len(current_horizons) - num_to_keep
            
            # Sort current horizons (we'll keep the lowest and highest)
            sorted_current = sorted(current_horizons)
            
            # Start with keeping the lowest horizons
            new_horizons = sorted_current[:num_to_keep]
            
            # Add best performing horizons from source, scaled appropriately for target timeframe
            bucket_scale_factor = self._get_bucket_scale_factor(source_bucket, target_bucket)
            
            for i in range(min(num_to_replace, len(best_horizons))):
                scaled_horizon = int(best_horizons[i] * bucket_scale_factor)
                if scaled_horizon not in new_horizons:
                    new_horizons.append(scaled_horizon)
            
            # If we still need more horizons, add from the original set
            while len(new_horizons) < len(current_horizons):
                for h in sorted_current:
                    if h not in new_horizons:
                        new_horizons.append(h)
                        break
            
            # Sort the final horizons
            new_horizons.sort()
            
            self.logger.info(f"Suggested new horizons for {target_bucket}: {new_horizons}")
            return new_horizons
                
        except Exception as e:
            self.logger.error(f"Horizon suggestion failed: {str(e)}")
            return []
    
    def _get_bucket_scale_factor(self, source_bucket, target_bucket):
        """Calculate the scale factor between two buckets for horizon scaling."""
        bucket_scales = {
            "Scalping": 1,
            "Short": 3,
            "Medium": 6,
            "Long": 12
        }
        
        source_scale = bucket_scales.get(source_bucket, 1)
        target_scale = bucket_scales.get(target_bucket, 1)
        
        if source_scale == 0:
            return 1.0
            
        return target_scale / source_scale
    
    def transfer_all(self, current_episode=0):
        """
        Perform all applicable transfers between buckets.
        
        Args:
            current_episode: Current training episode/iteration
            
        Returns:
            list: Results of transfer operations with messages
        """
        if current_episode < self.transfer_cooldown:
            return []
            
        results = []
        buckets = list(self.agents.keys())
        
        if len(buckets) < 2:
            return []
            
        # Define transfer pairs (which buckets transfer to which)
        transfer_pairs = [
            ("Scalping", "Short"),
            ("Short", "Medium"),
            ("Medium", "Long")
        ]
        
        # Add reverse transfers if enabled
        if self.enable_reverse_transfer:
            reverse_pairs = [
                ("Short", "Scalping"),
                ("Medium", "Short"),
                ("Long", "Medium")
            ]
            transfer_pairs.extend(reverse_pairs)
        
        # Perform transfers for each pair
        for source, target in transfer_pairs:
            if source in buckets and target in buckets:
                # Transfer feature importance
                fi_result = self.transfer_feature_importance(source, target)
                if fi_result:
                    results.append({
                        "source": source,
                        "target": target,
                        "type": "feature_importance",
                        "success": True,
                        "message": f"Transferred feature importance from {source} to {target}"
                    })
                
                # Transfer model weights
                weight_result = self.transfer_model_weights(source, target, layers=['encoder'])
                if weight_result:
                    results.append({
                        "source": source,
                        "target": target,
                        "type": "model_weights",
                        "success": True,
                        "message": f"Transferred encoder weights from {source} to {target}"
                    })
                
                # Suggest horizon updates
                new_horizons = self.suggest_horizon_updates(source, target)
                if new_horizons:
                    # Get target model
                    target_agent = self.get_agent(target)
                    if hasattr(target_agent, 'model'):
                        target_agent.model.horizons = new_horizons
                    elif hasattr(target_agent, 'actor_critic'):
                        target_agent.actor_critic.horizons = new_horizons
                        
                    results.append({
                        "source": source,
                        "target": target,
                        "type": "horizon_update",
                        "success": True,
                        "message": f"Updated {target} horizons based on {source} performance: {new_horizons}"
                    })
        
        return results
        
    def transfer_knowledge(self, source_bucket=None, target_bucket=None, verbose=True):
        """
        Execute knowledge transfer between source and target agents.
        
        Args:
            source_bucket: Bucket type of the source agent (takes precedence over source_agent)
            target_bucket: Bucket type of the target agent (takes precedence over target_agent)
            verbose: Whether to log detailed information
            
        Returns:
            bool: Success status of the transfer
        """
        # Use bucket-specific agents if provided
        source = self.get_agent(source_bucket) if source_bucket else self.source_agent
        target = self.get_agent(target_bucket) if target_bucket else self.target_agent
        
        if not source or not target:
            self.logger.warning("Source or target agent not provided")
            return False
            
        # Store temporarily for the transfer operation
        self.source_agent = source
        self.target_agent = target
        
        try:
            if verbose:
                log(f"Transferring knowledge using strategy: {self.strategy}")
                log(f"Transfer rate: {self.transfer_rate:.2f}")
            
            if self.strategy == 'weights':
                return self._transfer_weights(verbose)
            elif self.strategy == 'distillation':
                return self._transfer_via_distillation(verbose)
            elif self.strategy == 'feature':
                return self._transfer_feature_extractors(verbose)
            else:
                self.logger.warning(f"Unknown transfer strategy: {self.strategy}")
                return False
        except Exception as e:
            self.logger.error(f"Knowledge transfer failed: {str(e)}")
            if verbose:
                log(f"Knowledge transfer error: {str(e)}")
            return False
    
    def _transfer_weights(self, verbose=True):
        """
        Transfer weights directly from source model to target model.
        
        Args:
            verbose: Whether to log detailed information
            
        Returns:
            bool: Success status
        """
        try:
            # Get models from agents
            source_model = self.source_agent.actor_critic
            target_model = self.target_agent.actor_critic
            
            source_state_dict = source_model.state_dict()
            target_state_dict = target_model.state_dict()
            
            # Create a new state dict for target model
            new_state_dict = {}
            
            # Determine which layers to transfer
            transferable_layers = []
            if self.shared_layers:
                # Only transfer specified layers
                for layer_name in self.shared_layers:
                    matching_keys = [k for k in source_state_dict.keys() 
                                    if layer_name in k and k in target_state_dict]
                    transferable_layers.extend(matching_keys)
            else:
                # Transfer all compatible layers
                transferable_layers = [k for k in source_state_dict.keys() 
                                      if k in target_state_dict and 
                                      source_state_dict[k].size() == target_state_dict[k].size()]
            
            if verbose:
                log(f"Found {len(transferable_layers)} compatible layers for transfer")
                
            # Create new state_dict with transferred weights
            for k in target_state_dict.keys():
                if k in transferable_layers:
                    # Blend weights according to transfer rate
                    new_state_dict[k] = (source_state_dict[k] * self.transfer_rate + 
                                         target_state_dict[k] * (1 - self.transfer_rate))
                else:
                    # Keep original weights for non-transferable layers
                    new_state_dict[k] = target_state_dict[k]
            
            # Load the new weights
            target_model.load_state_dict(new_state_dict)
            
            # Update old model for PPO if applicable
            if hasattr(self.target_agent, 'update_old_model'):
                self.target_agent.update_old_model()
                
            if verbose:
                log("Weight transfer completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Weight transfer failed: {str(e)}")
            if verbose:
                log(f"Weight transfer error: {str(e)}")
            return False
    
    def _transfer_via_distillation(self, verbose=True):
        """
        Use knowledge distillation to transfer knowledge.
        
        Args:
            verbose: Whether to log detailed information
            
        Returns:
            bool: Success status
        """
        # Implementation of knowledge distillation
        # This would involve using the source model as a teacher
        # and training the target model to match its outputs
        if verbose:
            log("Knowledge distillation not fully implemented yet")
        return False
    
    def _transfer_feature_extractors(self, verbose=True):
        """
        Transfer feature extractor layers between models.
        
        Args:
            verbose: Whether to log detailed information
            
        Returns:
            bool: Success status
        """
        # Implementation of feature extractor transfer
        # This focuses on just transferring the feature extraction layers
        if verbose:
            log("Feature extractor transfer not fully implemented yet")
        return False

class ProgressiveTrainer:
    """
    Handles progressive training across multiple buckets with knowledge transfer.
    
    This orchestrates the sequential training of bucket models, allowing knowledge
    to flow from one bucket to another based on the training progress.
    """
    
    def __init__(self, config_path: str = CONFIG_FILE, progress_callback=None):
        """
        Initialize the progressive trainer.
        
        Args:
            config_path: Path to the configuration file
            progress_callback: Callback function for reporting progress
        """
        self.config_path = config_path
        self.progress_callback = progress_callback
        
        # Load base configuration
        self.config = self._load_config()
        
        # Training data cache
        self.data_cache = {}
        
        # Knowledge transfer module
        self._initialize_knowledge_transfer()
        
        # Training state
        self.current_bucket = None
        self.training_history = {}
        
        # Define the standard bucket training sequence
        self.bucket_sequence = ["Scalping", "Short", "Medium", "Long"]
        
        # Get directory paths
        self.models_dir = self.config.get("MODELS_DIR", MODELS_DIR_DEFAULT)
        self.log_dir = os.path.join(self.models_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging to file
        log_file = os.path.join(self.log_dir, "progressive_training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Initialized ProgressiveTrainer with config from {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        # Use TradeConfig as the base configuration
        config = trade_config.as_dict()
        
        # Override with any local config file settings
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                local_config = json.load(f)
                config.update(local_config)
        
        return config
    
    def _initialize_knowledge_transfer(self):
        """Initialize the knowledge transfer module."""
        try:
            # Import the CrossBucketKnowledgeTransfer class
            agent_module = importlib.import_module("src.agent.agent")
            self.CrossBucketKnowledgeTransfer = agent_module.CrossBucketKnowledgeTransfer
            
            # Create knowledge transfer instance
            self.knowledge_transfer = self.CrossBucketKnowledgeTransfer(self.config)
            logger.info("Initialized cross-bucket knowledge transfer")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge transfer: {e}")
            self.knowledge_transfer = None
    
    def _load_data(self, bucket_type: str) -> pd.DataFrame:
        """
        Load training data for a specific bucket type.
        
        Args:
            bucket_type: The bucket type to load data for
            
        Returns:
            DataFrame with training data
        """
        # Check if data is already loaded
        if bucket_type in self.data_cache:
            return self.data_cache[bucket_type]
        
        # Determine data file based on bucket type
        data_file = f"training_data_{bucket_type.lower()}.csv"
        data_path = os.path.join(DATA_DIR, data_file)
        
        if not os.path.exists(data_path):
            # Try alternative: generic training data
            data_path = os.path.join(DATA_DIR, "training_data.csv")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Training data not found for {bucket_type}")
        
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows from {data_path}")
        
        # Cache data
        self.data_cache[bucket_type] = df
        
        return df
    
    def _get_bucket_config(self, bucket_type: str) -> Dict[str, Any]:
        """
        Create a bucket-specific configuration.
        
        Args:
            bucket_type: Bucket type to configure
            
        Returns:
            Bucket-specific configuration dictionary
        """
        # Start with base config
        bucket_config = self.config.copy()
        
        # Update with bucket-specific settings
        bucket_config["BUCKET"] = bucket_type
        
        # Prediction horizons might be different for different buckets
        if bucket_type == "Scalping":
            bucket_config["MIN_HORIZON"] = 1
            bucket_config["MAX_HORIZON"] = 72
        elif bucket_type == "Short":
            bucket_config["MIN_HORIZON"] = 6
            bucket_config["MAX_HORIZON"] = 144
        elif bucket_type == "Medium":
            bucket_config["MIN_HORIZON"] = 24
            bucket_config["MAX_HORIZON"] = 288
        elif bucket_type == "Long":
            bucket_config["MIN_HORIZON"] = 72
            bucket_config["MAX_HORIZON"] = 576
        
        # Create knowledge transfer directory for storing transferable insights
        bucket_config["KNOWLEDGE_TRANSFER_DIR"] = os.path.join(self.models_dir, "knowledge_transfer")
        os.makedirs(bucket_config["KNOWLEDGE_TRANSFER_DIR"], exist_ok=True)
        
        return bucket_config
    
    def _free_memory_and_resources(self):
        """Free memory and resources to prepare for next bucket training."""
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run garbage collection
        optimize_memory()
        
        # Clear data cache for buckets we're not using anymore
        completed_buckets = []
        for bucket in self.bucket_sequence:
            if bucket == self.current_bucket:
                break
            completed_buckets.append(bucket)
        
        for bucket in completed_buckets:
            if bucket in self.data_cache:
                del self.data_cache[bucket]
        
        logger.info(f"Freed memory and resources. Current GPU usage: {measure_gpu_usage()*100:.1f}%")
    
    def train_bucket(self, bucket_type: str, episodes: int = None, save_path: str = None, 
                    transfer_from: str = None, resume: bool = False) -> str:
        """
        Train a specific bucket model.
        
        Args:
            bucket_type: Bucket type to train
            episodes: Number of episodes to train for (if None, use config default)
            save_path: Directory to save model (if None, use default bucket path)
            transfer_from: Bucket to transfer knowledge from
            resume: Whether to resume training from a checkpoint
            
        Returns:
            Path to the trained model
        """
        self.current_bucket = bucket_type
        
        # Set up save path
        if save_path is None:
            save_path = os.path.join(self.models_dir, bucket_type, "checkpoints")
        os.makedirs(save_path, exist_ok=True)
        
        # Get bucket-specific config
        bucket_config = self._get_bucket_config(bucket_type)
        
        # Set episodes if specified
        if episodes is not None:
            bucket_config["MAX_EPISODES"] = episodes
        
        # Log training start
        logger.info(f"Starting {bucket_type} bucket training for {bucket_config.get('MAX_EPISODES', 100)} episodes")
        if self.progress_callback:
            self.progress_callback(f"Starting {bucket_type} bucket training")
        
        # Load recovery state if resuming
        recovery_state = None
        if resume:
            recovery_path = os.path.join(os.path.dirname(save_path), "recovery_state.json")
            if os.path.exists(recovery_path):
                try:
                    with open(recovery_path, "r") as f:
                        recovery_state = json.load(f)
                    logger.info(f"Resuming training from episode {recovery_state.get('current_episode', 0)}")
                except Exception as e:
                    logger.error(f"Failed to load recovery state: {e}")
                    recovery_state = None
        
        # Transfer knowledge from another bucket if specified
        if transfer_from and self.knowledge_transfer:
            logger.info(f"Transferring knowledge from {transfer_from} to {bucket_type}")
            # This will be implemented in train_model when both agents are created
            bucket_config["TRANSFER_FROM_BUCKET"] = transfer_from
        
        # Load training data
        try:
            df = self._load_data(bucket_type)
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return None
        
        # Train the model
        try:
            # Update progress callback for nested training function
            def nested_progress_callback(msg):
                if self.progress_callback:
                    self.progress_callback(f"[{bucket_type}] {msg}")
                logger.info(f"[{bucket_type}] {msg}")
            
            # Train the model
            model, optimizer, episodes_completed, best_reward = train_model(
                df,
                bucket_config,
                save_path=save_path,
                recovery_state=recovery_state,
                progress_callback=nested_progress_callback
            )
            
            # Update training history
            self.training_history[bucket_type] = {
                "episodes_completed": episodes_completed,
                "best_reward": best_reward,
                "timestamp": time.time()
            }
            
            # Save checkpoint path
            final_path = os.path.join(save_path, f"final_{bucket_type.lower()}.pth")
            if model is not None:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                    "episodes": episodes_completed,
                    "reward": best_reward,
                    "config": bucket_config
                }, final_path)
                logger.info(f"Saved final model to {final_path}")
            
            # Free memory
            self._free_memory_and_resources()
            
            return final_path
        
        except Exception as e:
            logger.error(f"Error training {bucket_type} bucket: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def train_progressively(self, custom_sequence: List[str] = None, initial_bucket: str = None,
                          episodes_per_bucket: Dict[str, int] = None) -> Dict[str, str]:
        """
        Train buckets progressively, transferring knowledge between them.
        
        Args:
            custom_sequence: Custom sequence of buckets to train (default uses standard sequence)
            initial_bucket: Bucket to start with (if None, start with first in sequence)
            episodes_per_bucket: Dictionary of bucket -> episodes mappings
            
        Returns:
            Dictionary mapping bucket types to trained model paths
        """
        # Determine training sequence
        bucket_sequence = custom_sequence or self.bucket_sequence
        
        # Find starting bucket
        start_index = 0
        if initial_bucket:
            if initial_bucket in bucket_sequence:
                start_index = bucket_sequence.index(initial_bucket)
            else:
                logger.warning(f"Initial bucket {initial_bucket} not in sequence. Starting from beginning.")
        
        # Get episodes for each bucket
        if episodes_per_bucket is None:
            episodes_per_bucket = {}
        
        # Initialize results
        model_paths = {}
        
        # Train each bucket in sequence
        prev_bucket = None
        for i in range(start_index, len(bucket_sequence)):
            bucket = bucket_sequence[i]
            episodes = episodes_per_bucket.get(bucket, None)
            
            logger.info(f"Progressive training: {i+1}/{len(bucket_sequence)} - {bucket}")
            if self.progress_callback:
                self.progress_callback(f"Progressive training: {i+1}/{len(bucket_sequence)} - {bucket}")
            
            # Train with knowledge transfer from previous bucket (if any)
            model_path = self.train_bucket(
                bucket,
                episodes=episodes,
                transfer_from=prev_bucket
            )
            
            # Store model path
            if model_path:
                model_paths[bucket] = model_path
            
            # Update previous bucket for next iteration
            prev_bucket = bucket
        
        # Log completion
        logger.info(f"Progressive training complete. Trained {len(model_paths)}/{len(bucket_sequence)} buckets.")
        
        return model_paths

def main():
    """Command line interface for the progressive trainer."""
    parser = argparse.ArgumentParser(description="Progressive Trading Bucket Training")
    parser.add_argument("--config", type=str, default=CONFIG_FILE, help="Path to config file")
    parser.add_argument("--bucket", type=str, help="Single bucket to train (skip progressive training)")
    parser.add_argument("--sequence", type=str, help="Comma-separated bucket sequence (e.g., 'Scalping,Short,Medium,Long')")
    parser.add_argument("--episodes", type=int, help="Number of episodes (for single bucket mode)")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint if available")
    parser.add_argument("--transfer", type=str, help="Bucket to transfer knowledge from (for single bucket mode)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    def print_progress(msg):
        print(f"[PROGRESS] {msg}")
    
    trainer = ProgressiveTrainer(config_path=args.config, progress_callback=print_progress)
    
    # Training mode
    if args.bucket:
        # Single bucket mode
        print(f"Training single bucket: {args.bucket}")
        model_path = trainer.train_bucket(
            args.bucket,
            episodes=args.episodes,
            transfer_from=args.transfer,
            resume=args.resume
        )
        if model_path:
            print(f"Training complete. Model saved to {model_path}")
        else:
            print("Training failed.")
    else:
        # Progressive training mode
        sequence = None
        if args.sequence:
            sequence = args.sequence.split(",")
            print(f"Using custom bucket sequence: {sequence}")
        
        print("Starting progressive training...")
        model_paths = trainer.train_progressively(custom_sequence=sequence)
        
        print("\nProgressive training complete. Results:")
        for bucket, path in model_paths.items():
            print(f"  {bucket}: {path}")

if __name__ == "__main__":
    main() 