#!/usr/bin/env python
"""
Training Recovery System

This module implements a robust error handling and recovery system for the training process.
It provides mechanisms to handle errors, create checkpoints, and recover from failures.
"""

import os
import sys
import time
import json
import glob
import logging
import traceback
import importlib.util
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("training_recovery")

# Try to import error handling framework
try:
    from src.ui.error_handler import handle_error, ErrorSeverity
    error_handler_available = True
except ImportError:
    error_handler_available = False
    # Create stub function for error handling if not available
    def handle_error(error, context="", window=None, retry_func=None, additional_context=None):
        logger.error(f"Error in {context}: {str(error)}")
        return {"message": str(error), "handled": False}


def _get_preset_default_config(bucket: str) -> Dict[str, Any]:
    """Load default parameters for a bucket from the preset system."""
    try:
        from src.ui import preset_manager
    except Exception as e:
        logger.warning(f"Preset manager unavailable: {e}")
        return {}

    bucket_presets = preset_manager.DEFAULT_PRESETS.get(bucket, {})
    for _name, data in bucket_presets.items():
        params = data.get("params")
        if params:
            return params
    return {}

# Try to import trade_config for default configurations
try:
    from src.utils.trade_config import get_trade_config, TradeConfig
    trade_config_available = True
except ImportError:
    trade_config_available = False
    logger.warning("TradeConfig not available - some recovery features will be limited")

class TrainingRecoverySystem:
    """
    A comprehensive system for handling errors, creating checkpoints, and recovering
    from failures during the training process.
    
    This class integrates with the existing error handling and checkpointing mechanisms
    in the BTC AI system.
    """
    
    def __init__(self, 
                checkpoint_dir: str = "checkpoints", 
                max_retries: int = 3,
                min_checkpoint_interval: int = 5,
                enable_emergency_checkpoints: bool = True):
        """
        Initialize the training recovery system.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_retries: Maximum number of retry attempts for recoverable errors
            min_checkpoint_interval: Minimum episodes between checkpoints
            enable_emergency_checkpoints: Whether to create emergency checkpoints on errors
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_retries = max_retries
        self.min_checkpoint_interval = min_checkpoint_interval
        self.enable_emergency_checkpoints = enable_emergency_checkpoints
        
        # Initialize state variables
        self.retry_count = 0
        self.active_training = False
        self.last_checkpoint_episode = 0
        self.recovery_state = None
        self.last_error = None
        self.performance_history = []
        self.last_checkpoint_path = None
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Error classification
        self.recoverable_errors = {
            'ValueError': True,          # Recoverable with config changes
            'KeyError': True,            # Missing config key
            'IndexError': True,          # Likely recoverable
            'AttributeError': True,      # May be recoverable with fallbacks
            'RuntimeError': True,        # May be recoverable depending on cause
            'DataValidationError': True, # Can retry with different constraints
            'JSONDecodeError': True,     # Can try backup configs
            'ZeroDivisionError': True,   # Can add checks for this
            'NaNError': True,            # Can restart with adjusted params
            'EnvironmentError': True,    # Can recreate environment
        }
        
        self.unrecoverable_errors = {
            'ImportError': False,        # Critical module missing
            'ModuleNotFoundError': False, # Critical module missing
            'FileNotFoundError': False,  # Critical file missing
            'PermissionError': False,    # System permission issue
            'MemoryError': False,        # System resource issue
            'OSError': False,            # System level error
            'GPUError': False,           # Hardware issue
        }
        
        logger.info(f"Training recovery system initialized with max_retries={max_retries}")
        
    def start_training_with_recovery(self, 
                                    training_func: Callable, 
                                    config: Dict[str, Any],
                                    df=None,
                                    save_path: str = None,
                                    recovery_state: Dict[str, Any] = None,
                                    progress_callback: Callable = None) -> Tuple[Any, Dict[str, Any], float, float]:
        """
        Wrap training process with recovery capabilities.
        
        Args:
            training_func: Function that performs the training
            config: Configuration parameters
            df: DataFrame for training data
            save_path: Path to save checkpoints
            recovery_state: Optional recovery state for resuming training
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (model, metrics, best_reward, elapsed_time)
        """
        self.retry_count = 0
        self.active_training = True
        self.recovery_state = recovery_state
        start_time = time.time()
        
        # Validate save path
        if save_path:
            self.checkpoint_dir = save_path
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        # Validate config before starting
        if not self._validate_config(config):
            logger.warning("Configuration validation found issues - attempting to fix")
            config = self._repair_config(config)
            
        while self.retry_count <= self.max_retries:
            try:
                # Run training with monitoring
                logger.info(f"Starting training attempt {self.retry_count + 1}/{self.max_retries + 1}")
                
                # Call the training function
                result = training_func(
                    df=df, 
                    config=config, 
                    save_path=save_path,
                    recovery_state=self.recovery_state,
                    progress_callback=progress_callback
                )
                
                # Training completed successfully
                logger.info("Training completed successfully")
                elapsed_time = time.time() - start_time
                self.active_training = False
                
                # Return the result from training
                if result:
                    if isinstance(result, tuple) and len(result) >= 3:
                        # Assume result is (model, metrics, best_reward) or similar
                        return (*result, elapsed_time)
                    else:
                        # Just return whatever we got plus elapsed time
                        return (result, {}, 0.0, elapsed_time)
                else:
                    return (None, {}, 0.0, elapsed_time)
                
            except Exception as e:
                self.last_error = e
                error_type = type(e).__name__
                logger.error(f"Training error ({error_type}): {str(e)}")
                
                # Create detailed error report
                self._create_error_report(e, config)
                
                # Create emergency checkpoint if enabled
                if self.enable_emergency_checkpoints:
                    self._create_emergency_checkpoint(e, config)
                
                # Use BTC AI's error handler if available
                if error_handler_available:
                    handle_error(
                        e,
                        "training_recovery.training_error",
                        additional_context={
                            "retry_count": self.retry_count,
                            "config": config,
                            "error_type": error_type,
                            "traceback": traceback.format_exc()
                        }
                    )
                
                # Check if error is recoverable
                is_recoverable = self._is_recoverable_error(e)
                
                if is_recoverable and self.retry_count < self.max_retries:
                    self.retry_count += 1
                    logger.info(f"Attempting recovery (try {self.retry_count}/{self.max_retries})")
                    
                    # Try to recover config
                    config = self._recover_config(e, config)
                    
                    # Get a recovery state if available
                    self.recovery_state = self._get_recovery_state()
                    
                    # Continue with next attempt
                    continue
                else:
                    # We've exhausted recovery attempts or error is unrecoverable
                    logger.error(f"Training failed after {self.retry_count} recovery attempts")
                    self.active_training = False
                    elapsed_time = time.time() - start_time
                    
                    # Include elapsed time in the result
                    return (None, {"error": str(e)}, 0.0, elapsed_time)
        
        # We should never reach here, but just in case
        self.active_training = False
        elapsed_time = time.time() - start_time
        return (None, {"error": "Max retries exceeded"}, 0.0, elapsed_time)
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for required parameters and valid values.
        
        Args:
            config: Configuration parameters
            
        Returns:
            bool: True if config is valid, False otherwise
        """
        if not config:
            logger.error("Configuration is empty or None")
            return False
            
        # Essential parameters that must be present
        required_params = [
            "LEARNING_RATE",
            "HIDDEN_SIZE",
            "BUCKET",
            "MAX_STEPS_PER_EPISODE",
            "ES_POPULATION"
        ]
        
        # Check for required parameters
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            logger.warning(f"Missing required parameters: {missing_params}")
            return False
            
        # Parameter validation rules
        validation_rules = {
            "LEARNING_RATE": lambda x: 0 < x < 1,
            "HIDDEN_SIZE": lambda x: x > 0,
            "ES_POPULATION": lambda x: x > 0,
            "MAX_STEPS_PER_EPISODE": lambda x: x > 0,
        }
        
        # Check parameter values
        invalid_params = []
        for param, rule in validation_rules.items():
            if param in config:
                try:
                    if not rule(config[param]):
                        invalid_params.append(param)
                except:
                    invalid_params.append(param)
                    
        if invalid_params:
            logger.warning(f"Invalid parameter values: {invalid_params}")
            return False
            
        return True
    
    def _repair_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to repair invalid configuration.
        
        Args:
            config: Configuration parameters
            
        Returns:
            Dict: Repaired configuration
        """
        # Start with the original config
        repaired_config = config.copy() if config else {}
        
        # Default values for common parameters
        defaults = {
            "LEARNING_RATE": 0.0003,
            "HIDDEN_SIZE": 512,
            "ES_POPULATION": 8,
            "BUCKET": "Scalping",
            "MAX_STEPS_PER_EPISODE": 500,
            "EARLY_STOP_PATIENCE": 50,
            "MIN_ENVS_PER_AGENT": 1,
            "MAX_ENVS_PER_AGENT": 4,
            "USE_DYNAMIC_HORIZONS": True,
            "RECOVERY_INTERVAL": 10,
            "PERF_LOG_INTERVAL": 5,
            "ES_INTERVAL": 5,
            "TRANSFER_INTERVAL": 20,
        }
        
        # Try to get bucket-specific defaults from preset system
        bucket = repaired_config.get("BUCKET", "Scalping")
        preset_defaults = {}

        try:
            preset_defaults = _get_preset_default_config(bucket)
            if preset_defaults:
                logger.info(f"Found preset defaults for bucket {bucket}")
        except Exception as e:
            logger.warning(f"Could not get preset defaults: {e}")
        
        # If we have preset defaults, prefer them over our hardcoded defaults
        if preset_defaults:
            # Update defaults with preset values
            for key, value in preset_defaults.items():
                defaults[key] = value
        
        # Or try to use TradeConfig if available
        elif trade_config_available:
            try:
                trade_config = get_trade_config()
                config_dict = trade_config.as_dict()
                for key, value in config_dict.items():
                    defaults[key] = value
                logger.info(f"Using TradeConfig defaults")
            except Exception as e:
                logger.warning(f"Could not get TradeConfig defaults: {e}")
        
        # Fill in missing parameters
        for param, value in defaults.items():
            if param not in repaired_config or repaired_config[param] is None:
                repaired_config[param] = value
                logger.info(f"Added missing parameter {param}={value}")
        
        # Fix invalid parameter values
        validation_rules = {
            "LEARNING_RATE": lambda x: max(1e-6, min(x, 0.1)) if isinstance(x, (int, float)) else defaults["LEARNING_RATE"],
            "HIDDEN_SIZE": lambda x: max(32, x) if isinstance(x, (int, float)) else defaults["HIDDEN_SIZE"],
            "ES_POPULATION": lambda x: max(2, x) if isinstance(x, (int, float)) else defaults["ES_POPULATION"],
            "MAX_STEPS_PER_EPISODE": lambda x: max(100, x) if isinstance(x, (int, float)) else defaults["MAX_STEPS_PER_EPISODE"],
        }
        
        for param, rule in validation_rules.items():
            if param in repaired_config:
                old_value = repaired_config[param]
                try:
                    repaired_config[param] = rule(old_value)
                    if repaired_config[param] != old_value:
                        logger.info(f"Fixed parameter {param}: {old_value} -> {repaired_config[param]}")
                except:
                    repaired_config[param] = defaults[param]
                    logger.info(f"Reset invalid parameter {param} to default: {defaults[param]}")
        
        # Ensure BUCKET is valid
        if "BUCKET" in repaired_config:
            if repaired_config["BUCKET"] not in ["Scalping", "Short", "Medium", "Long"]:
                logger.info(f"Invalid BUCKET {repaired_config['BUCKET']}, defaulting to Scalping")
                repaired_config["BUCKET"] = "Scalping"
        
        return repaired_config
    
    def _recover_config(self, error: Exception, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to recover configuration based on the error.
        
        Args:
            error: Exception that occurred
            current_config: Current configuration
            
        Returns:
            Dict: Recovered configuration
        """
        error_msg = str(error)
        error_type = type(error).__name__
        
        # Start with repairing the current config
        recovered_config = self._repair_config(current_config)
        
        # Specific error handling based on error type
        if error_type == "ValueError":
            # Check for common value errors
            if "learning rate" in error_msg.lower():
                # Reduce learning rate if it's too high
                if "LEARNING_RATE" in recovered_config:
                    recovered_config["LEARNING_RATE"] *= 0.5
                    logger.info(f"Reduced learning rate to {recovered_config['LEARNING_RATE']}")
            
            # Hyperparameter validation errors
            if "invalid hyperparameter" in error_msg.lower():
                # Extract parameter name from error message if possible
                param_match = error_msg.lower().find("parameter")
                if param_match >= 0:
                    error_part = error_msg[param_match:].split()
                    if len(error_part) > 1:
                        param_name = error_part[1].strip(":'\"")
                        # Reset this parameter to default
                        if param_name.upper() in recovered_config:
                            if trade_config_available:
                                trade_config = get_trade_config()
                                default_value = getattr(trade_config, param_name.upper(), None)
                                if default_value is not None:
                                    recovered_config[param_name.upper()] = default_value
                                    logger.info(f"Reset {param_name.upper()} to default: {default_value}")
            
        elif error_type == "RuntimeError":
            # Check for memory-related errors
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                # Reduce model size or batch size
                if "HIDDEN_SIZE" in recovered_config:
                    recovered_config["HIDDEN_SIZE"] = max(64, recovered_config["HIDDEN_SIZE"] // 2)
                    logger.info(f"Reduced hidden size to {recovered_config['HIDDEN_SIZE']}")
                
                # Force CPU if GPU issues
                recovered_config["DEVICE_PREFERENCE"] = "cpu"
                logger.info(f"Forcing CPU due to GPU memory issues")
                
        elif error_type == "KeyError":
            # Missing configuration key
            key_name = error_msg.strip("'\"")
            logger.info(f"Adding missing configuration key: {key_name}")
            
            # Try to get the value from TradeConfig
            if trade_config_available:
                trade_config = get_trade_config()
                default_value = getattr(trade_config, key_name, None)
                if default_value is not None:
                    recovered_config[key_name] = default_value
            
        elif error_type == "NaNError" or "nan" in error_msg.lower():
            # NaN values in training - adjust learning rate and add gradient clipping
            if "LEARNING_RATE" in recovered_config:
                recovered_config["LEARNING_RATE"] = max(1e-6, recovered_config["LEARNING_RATE"] * 0.1)
                logger.info(f"Reduced learning rate to {recovered_config['LEARNING_RATE']} due to NaN values")
            
            # Add gradient clipping
            recovered_config["GRADIENT_CLIP_VALUE"] = 1.0
            logger.info(f"Added gradient clipping to handle NaN values")
            
        # Check for preset recovery on the last retry attempt
        if self.retry_count == self.max_retries - 1:
            # Last attempt - try to load a preset as a last resort
            bucket = recovered_config.get("BUCKET", "Scalping")
            try:
                # Attempt to get a preset config
                preset_config = _get_preset_default_config(bucket)
                if preset_config:
                    logger.info(f"Using preset defaults for {bucket} as last recovery attempt")
                    # Keep some original settings but use preset for most
                    for key, value in preset_config.items():
                        if key not in ["DEVICE_PREFERENCE"]:  # Keep some original settings
                            recovered_config[key] = value
            except Exception as e:
                logger.warning(f"Could not load preset for recovery: {e}")
        
        return recovered_config
    
    def _get_recovery_state(self) -> Dict[str, Any]:
        """
        Attempt to find a recovery state from checkpoints or state files.
        
        Returns:
            Dict: Recovery state if found, None otherwise
        """
        # First, check for recovery state file
        recovery_file = os.path.join(os.path.dirname(self.checkpoint_dir), "recovery_state.json")
        if os.path.exists(recovery_file):
            try:
                with open(recovery_file, 'r') as f:
                    state = json.load(f)
                    logger.info(f"Found recovery state in {recovery_file}")
                    return state
            except Exception as e:
                logger.warning(f"Failed to load recovery state from {recovery_file}: {e}")
        
        # Next, check for the latest checkpoint
        try:
            checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pth"))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                logger.info(f"Found latest checkpoint: {latest_checkpoint}")
                
                # Load checkpoint to extract state
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                
                # Create recovery state from checkpoint
                recovery_state = {
                    'episode': checkpoint.get('episode', 0),
                    'best_reward': checkpoint.get('best_reward', 0),
                    'best_agent_idx': checkpoint.get('best_agent_idx', 0),
                    'checkpoint_path': latest_checkpoint
                }
                
                logger.info(f"Created recovery state from checkpoint: Episode {recovery_state['episode']}")
                return recovery_state
        except Exception as e:
            logger.warning(f"Failed to extract recovery state from checkpoints: {e}")
        
        # No recovery state found
        return None
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """
        Determine if an error is recoverable.
        
        Args:
            error: Exception that occurred
            
        Returns:
            bool: True if error is recoverable, False otherwise
        """
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Check in predefined lists
        if error_type in self.recoverable_errors:
            return self.recoverable_errors[error_type]
        
        if error_type in self.unrecoverable_errors:
            return self.unrecoverable_errors[error_type]
        
        # Check message content for specific cases
        unrecoverable_patterns = [
            "cuda runtime error",
            "cuda driver version is insufficient",
            "no cuda-capable device is detected",
            "out of memory",
            "permission denied",
            "cannot open file",
            "file not found"
        ]
        
        for pattern in unrecoverable_patterns:
            if pattern in error_msg:
                return False
        
        # Default to recoverable for unknown errors
        return True
    
    def _create_error_report(self, error: Exception, config: Dict[str, Any]) -> None:
        """
        Create a detailed error report.
        
        Args:
            error: Exception that occurred
            config: Configuration being used
        """
        import platform
        
        # Create a timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(os.path.dirname(self.checkpoint_dir), f"error_report_{timestamp}.json")
        
        # Build the report
        report = {
            "timestamp": timestamp,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "config": config,
            "system_info": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            try:
                report["system_info"]["cuda_version"] = torch.version.cuda
                report["system_info"]["gpu_name"] = torch.cuda.get_device_name(0)
                report["system_info"]["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
            except:
                pass
        
        # Save the report
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Created error report at {report_path}")
            
            # Create a user-friendly summary
            user_report = f"""
=== TRAINING ERROR REPORT ===
Error: {type(error).__name__} - {str(error)}
Time: {timestamp}
Attempt: {self.retry_count + 1} of {self.max_retries + 1}
Device: {"CUDA" if torch.cuda.is_available() else "CPU"}

Technical report saved to: {report_path}

Possible next steps:
1. Check the error message above for clues
2. Verify data and configuration parameters
3. Try a different set of parameters or a smaller model
4. Check system resources (memory, GPU)
5. Review the technical report for more details
"""
            # Save user-friendly report
            user_report_path = os.path.join(os.path.dirname(self.checkpoint_dir), f"error_summary_{timestamp}.txt")
            with open(user_report_path, 'w') as f:
                f.write(user_report)
                
            logger.info(f"Created user-friendly error summary at {user_report_path}")
            
        except Exception as e:
            logger.error(f"Failed to create error report: {e}")
    
    def _create_emergency_checkpoint(self, error: Exception, config: Dict[str, Any]) -> bool:
        """
        Create an emergency checkpoint for debugging and recovery.
        
        Args:
            error: Exception that occurred
            config: Configuration parameters
            
        Returns:
            bool: True if checkpoint was created successfully, False otherwise
        """
        # This is a stub implementation since we need model and optimizer state
        # to create a proper checkpoint, which we don't have access to here.
        # In practice, the training code itself would need to create the checkpoint.
        
        # Create a state file with information about the error
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        state_path = os.path.join(self.checkpoint_dir, f"emergency_state_{timestamp}.json")
        
        try:
            state = {
                "timestamp": timestamp,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "config": config,
                "retry_count": self.retry_count,
                "recovery_needed": True
            }
            
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Created emergency state file at {state_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create emergency state file: {e}")
            return False
    
    def register_checkpoint(self, episode: int, checkpoint_path: str) -> None:
        """
        Register a checkpoint created during training.
        
        Args:
            episode: Episode number
            checkpoint_path: Path to the checkpoint
        """
        self.last_checkpoint_episode = episode
        self.last_checkpoint_path = checkpoint_path
        logger.info(f"Registered checkpoint at episode {episode}: {checkpoint_path}")
    
    def register_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Register performance metrics from training.
        
        Args:
            metrics: Performance metrics
        """
        self.performance_history.append(metrics)
        
    def should_create_checkpoint(self, current_episode: int) -> bool:
        """
        Determine if a checkpoint should be created.
        
        Args:
            current_episode: Current episode number
            
        Returns:
            bool: True if checkpoint should be created, False otherwise
        """
        # If no checkpoints yet, create one
        if self.last_checkpoint_episode == 0:
            return True
            
        # Check if enough episodes have passed
        return (current_episode - self.last_checkpoint_episode) >= self.min_checkpoint_interval
    
    def create_training_checkpoint(self, model, optimizer, episode: int, 
                                 best_reward: float, config: Dict[str, Any], 
                                 additional_state: Dict[str, Any] = None) -> str:
        """
        Create a training checkpoint.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer to save state
            episode: Current episode number
            best_reward: Best reward achieved
            config: Configuration parameters
            additional_state: Additional state to save
            
        Returns:
            str: Path to the created checkpoint
        """
        # Create checkpoint filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{episode}_{timestamp}.pt")
        
        try:
            # Create state to save
            checkpoint = {
                "episode": episode,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_reward": best_reward,
                "config": config,
                "timestamp": timestamp
            }
            
            # Add additional state if provided
            if additional_state:
                for key, value in additional_state.items():
                    checkpoint[key] = value
            
            # Save performance history if available
            if self.performance_history:
                checkpoint["performance_history"] = self.performance_history
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Register checkpoint
            self.register_checkpoint(episode, checkpoint_path)
            
            logger.info(f"Created checkpoint at {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return ""
    
    def cleanup_old_checkpoints(self, keep: int = 5, keep_best: bool = True) -> None:
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep: Number of recent checkpoints to keep
            keep_best: Whether to keep the best checkpoint regardless of age
        """
        try:
            # Get all checkpoints
            checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pt"))
            if not checkpoint_files:
                return
                
            # Sort by creation time (newest first)
            checkpoint_files.sort(key=os.path.getctime, reverse=True)
            
            # Keep the most recent ones
            to_keep = checkpoint_files[:keep]
            
            # Keep the best checkpoint if requested
            best_checkpoint = None
            if keep_best:
                # Find checkpoint with highest reward
                best_reward = float('-inf')
                for checkpoint_file in checkpoint_files:
                    try:
                        checkpoint = torch.load(checkpoint_file, map_location='cpu')
                        reward = checkpoint.get('best_reward', float('-inf'))
                        if reward > best_reward:
                            best_reward = reward
                            best_checkpoint = checkpoint_file
                    except:
                        pass
                
                if best_checkpoint and best_checkpoint not in to_keep:
                    to_keep.append(best_checkpoint)
            
            # Delete the rest
            for checkpoint_file in checkpoint_files:
                if checkpoint_file not in to_keep:
                    try:
                        os.remove(checkpoint_file)
                        logger.info(f"Deleted old checkpoint: {checkpoint_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete checkpoint {checkpoint_file}: {e}")
                        
        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {e}")
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """
        Get the current recovery status.
        
        Returns:
            Dict: Current recovery status
        """
        return {
            "active_training": self.active_training,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_checkpoint_episode": self.last_checkpoint_episode,
            "last_checkpoint_path": self.last_checkpoint_path,
            "last_error": str(self.last_error) if self.last_error else None,
            "recovery_state": self.recovery_state
        } 