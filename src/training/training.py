#!/usr/bin/env python
"""
Training module for BTC prediction agents.
This module handles the training loop, model optimization, and convergence checks.
"""

import torch
import os
import sys
import importlib
import gc
import numpy as np
import pandas as pd
import json
import logging
import time
import signal
import traceback
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("training")

# Import error handling framework
try:
    from src.ui.error_handler import handle_error, ErrorSeverity
    error_handler_available = True
except ImportError:
    error_handler_available = False
    # Create stub function for error handling if not available
    def handle_error(error, context="", window=None, retry_func=None, additional_context=None):
        logger.error(f"Error in {context}: {str(error)}")
        return {"message": str(error), "handled": False}

# Add the current directory to sys.path to ensure module imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Get the project root directory
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# Add project root to system path to ensure imports work
sys.path.insert(0, project_root)

# Directory constants
MODELS_DIR_DEFAULT = os.path.join(os.path.dirname(current_dir), "Models")

# See if stable-baselines3 is available
try:
    from stable_baselines3.common.vec_env import SubprocVecEnv
    SUBPROC_VEC_ENV_AVAILABLE = True
except ImportError:
    SUBPROC_VEC_ENV_AVAILABLE = False
    # Use dynamic imports instead of direct imports
    try:
        env_base_module = importlib.import_module("src.environment.env_base")
        VecEnvWrapper = env_base_module.VecEnvWrapper
    except ImportError:
        print("Error: Could not import VecEnvWrapper")

# Import modules dynamically
try:
    # Import utils module
    utils_module = importlib.import_module("src.utils.utils")
    log = utils_module.log
    validate_dataframe = utils_module.validate_dataframe
    calculate_metrics = utils_module.calculate_metrics
    calculate_env_metrics = utils_module.calculate_env_metrics
    cleanup_checkpoints = utils_module.cleanup_checkpoints
    measure_gpu_usage = utils_module.measure_gpu_usage
    get_optimal_gpu_targets = utils_module.get_optimal_gpu_targets
    optimize_memory = utils_module.optimize_memory
    check_multi_gpu = utils_module.check_multi_gpu
    save_checkpoint = utils_module.save_checkpoint
    write_metrics_history = utils_module.write_metrics_history
    visualize_metrics = utils_module.visualize_metrics
    optimize_memory_for_long_training = utils_module.optimize_memory_for_long_training
    
    try:
        # Try to import these functions from utils, but they might not exist
        preprocess_data = utils_module.preprocess_data
        adapt_prediction_horizons = utils_module.adapt_prediction_horizons
        save_recovery_state = utils_module.save_recovery_state
    except AttributeError:
        # Define them below if they don't exist in utils
        pass
    
    # Import environment module
    env_base_module = importlib.import_module("src.environment.env_base")
    create_environment = env_base_module.create_environment
    make_env_creator = env_base_module.make_env_creator
    
    # Import TradeConfig
    trade_config_module = importlib.import_module("src.utils.trade_config")
    trade_config = trade_config_module.get_trade_config()
    
    # Import agent components
    agent_module = importlib.import_module("src.agent.agent")
    PPOAgent = agent_module.PPOAgent
    ESPopulation = agent_module.ESPopulation
    
except ImportError as e:
    print(f"Error importing modules in training.py: {e}")
    # Define fallback functions if imports fail
    def log(message, level="info"):
        print(f"[{level.upper()}] {message}")
        
    def optimize_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config():
        return {}

# Add human-readable metrics logging function
def log_human_metrics(metrics, episode, time_elapsed=None):
    """
    Log a simplified, human-readable version of metrics focused on what humans care about.
    
    Args:
        metrics (dict): Dictionary containing metrics
        episode (int): Current episode number
        time_elapsed (float, optional): Time elapsed in seconds
    """
    # Format time if provided
    time_str = ""
    if time_elapsed is not None:
        hours, remainder = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f" [{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}]"
    
    # Extract and format key metrics humans care about
    net_profit = metrics.get("net_profit", 0)
    profit_pct = metrics.get("profit_pct", 0) * 100 if "profit_pct" in metrics else 0
    win_rate = metrics.get("win_rate", 0) * 100
    total_trades = metrics.get("total_trades", 0)
    max_dd = metrics.get("max_drawdown", 0) * 100
    sharpe = metrics.get("sharpe_ratio", 0)
    
    # Create compact, readable output
    log_msg = (f"[EPISODE {episode}]{time_str} "
              f"Profit: ${net_profit:.2f} ({profit_pct:.1f}%) | "
              f"Win Rate: {win_rate:.1f}% | "
              f"Trades: {total_trades} | "
              f"Max DD: {max_dd:.1f}% | "
              f"Sharpe: {sharpe:.2f}")
    
    # Log with info level
    log(log_msg, "info")

# Define utility functions if they're not imported from elsewhere
def preprocess_data(df, config):
    """
    Prepare dataframe for training by:
    1. Validating required columns
    2. Applying any preprocessing steps based on config
    
    Args:
        df (pd.DataFrame): Input dataframe with market data
        config (dict): Configuration parameters
        
    Returns:
        pd.DataFrame: Processed dataframe ready for training
    """
    # Validate DataFrame if validation function is available
    if 'validate_dataframe' in globals():
        df = validate_dataframe(df)
    
    # Fill any NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Ensure timestamp column is datetime if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Apply any additional preprocessing steps from config
    if config.get('NORMALIZE_DATA', False):
        for col in df.columns:
            if col not in ['timestamp', 'date', 'symbol'] and df[col].dtype != 'object':
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
    
    return df

def adapt_prediction_horizons(model, market_data, config):
    """
    Adapt prediction horizons based on market conditions and past performance.
    
    Args:
        model: The trained model
        market_data: Recent market data
        config: Configuration parameters
        
    Returns:
        list: Updated prediction horizons
    """
    bucket_type = config.get("BUCKET", "Scalping")
    
    # Default horizons by bucket type
    default_horizons = {
        "Scalping": [6, 12, 24, 36],
        "Short": [12, 36, 72, 144],
        "Medium": [24, 72, 144, 288],
        "Long": [72, 144, 288, 576]
    }
    
    # Use default horizons if market_data or model is not valid
    if market_data is None or model is None:
        return default_horizons.get(bucket_type, [12, 36, 72, 144])
    
    # Otherwise, keep the existing horizons
    return default_horizons.get(bucket_type, [12, 36, 72, 144])

def save_recovery_state(episode, best_reward, best_agent_idx, filename):
    """
    Save recovery state to JSON file for potential training interruptions.
    
    Args:
        episode: Current episode number
        best_reward: Best reward so far
        best_agent_idx: Index of the best agent
        filename: Path to save the recovery state
    """
    try:
        with open(filename, 'w') as f:
            json.dump({
                'episode': episode,
                'best_reward': best_reward,
                'best_agent_idx': best_agent_idx,
                'timestamp': datetime.now().isoformat()
            }, f)
    except Exception as e:
        log(f"Error saving recovery state: {e}", level="error")

def make_vec_env(df, config, num_envs, device="cpu"):
    """
    Create a vectorized environment.
    
    Args:
        df: DataFrame with market data
        config: Configuration dictionary
        num_envs: Number of environments to create
        device: Device to use (cpu/cuda)
        
    Returns:
        Vectorized environment
    """
    # Define prediction horizons (store in config for environment access)
    prediction_horizons = {
        "short": config.get("SHORT_HORIZON", 60),
        "medium": config.get("MEDIUM_HORIZON", 240),
        "long": config.get("LONG_HORIZON", 720)
    }
    
    # Store horizons in config for environment to access
    config["PREDICTION_HORIZONS"] = prediction_horizons
    
    # Create environment creator function with correct signature
    env_creator = make_env_creator(df, config, device)
    
    # Create vectorized environment
    if SUBPROC_VEC_ENV_AVAILABLE and num_envs > 1:
        env = SubprocVecEnv([env_creator for _ in range(num_envs)])
    else:
        env = VecEnvWrapper([env_creator() for _ in range(num_envs)])
    
    return env

def train_agent_episode(agent_envs, agent, horizons, config, device, episode_num):
    """
    Train a single agent for one episode across multiple environments.
    
    Args:
        agent_envs: Environments for this agent
        agent: The agent to train
        horizons: Prediction horizons
        config: Configuration parameters
        device: Computing device (CPU/GPU)
        episode_num: Current episode number
        
    Returns:
        tuple: (rewards, metrics)
    """
    # Initialize environment
    try:
        observations = agent_envs.reset()
    except Exception as e:
        error_msg = f"Error resetting environment: {str(e)}"
        logger.error(error_msg)
        if error_handler_available:
            handle_error(
                e,
                "train_agent_episode.reset_env",
                additional_context={"error_details": str(e)}
            )
        # Return empty results
        return [], {"reward": 0, "steps": 0, "error": str(e)}
    
    dones = [False] * len(agent_envs.envs)
    
    # Storage for episode data
    rewards = []
    episode_metrics = []
    hidden_states = [None] * len(agent_envs.envs)
    trajectories = []
    
    # Track NaN detection
    nan_actions_detected = 0
    nan_rewards_detected = 0
    
    # Main episode loop
    step_count = 0
    max_steps = config.get("MAX_STEPS_PER_EPISODE", 500)
    
    while not all(dones) and step_count < max_steps:
        step_count += 1
        
        # Select actions
        actions = []
        log_probs = []
        values = []
        
        for env_idx, obs in enumerate(observations):
            if dones[env_idx]:
                # Skip done environments
                actions.append(np.zeros(2, dtype=np.float32))
                log_probs.append(0.0)
                values.append(0.0)
                continue
            
            # Convert to tensor
            try:
                obs_tensor = torch.FloatTensor(obs).to(device)
                
                # Check for NaN values in observation
                if torch.isnan(obs_tensor).any():
                    if error_handler_available:
                        handle_error(
                            ValueError("NaN values detected in observation"),
                            "train_agent_episode.nan_observation",
                            additional_context={"env_idx": env_idx, "step": step_count}
                        )
                    # Use zero tensor as fallback
                    obs_tensor = torch.zeros_like(obs_tensor)
                
                # Select action with adaptive exploration
                action, log_prob, value, preds, confs, mids, trend, novelty, hidden_states[env_idx] = agent.select_action(
                    obs_tensor, hidden_states[env_idx]
                )
                
                # Check for NaN values in action
                if np.isnan(action).any():
                    nan_actions_detected += 1
                    logger.warning(f"NaN action detected at step {step_count}, env {env_idx}")
                    
                    if error_handler_available:
                        handle_error(
                            ValueError("NaN action detected"),
                            "train_agent_episode.nan_action",
                            additional_context={"env_idx": env_idx, "step": step_count}
                        )
                    
                    # Replace with random action
                    action = np.array([random.uniform(-1, 1), random.uniform(0, 1)], dtype=np.float32)
                    log_prob = 0.0
                    value = 0.0
                
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                
            except Exception as e:
                logger.error(f"Error selecting action: {e}")
                if error_handler_available:
                    handle_error(
                        e,
                        "train_agent_episode.select_action",
                        additional_context={"env_idx": env_idx, "step": step_count}
                    )
                # Fallback to random action
                actions.append([random.uniform(-1, 1), random.uniform(0, 1)])
                log_probs.append(0.0)
                values.append(0.0)
        
        # Step environments
        try:
            next_obs, step_rewards, new_dones, infos = agent_envs.step(actions)
            
            # Check for NaN rewards
            for i, reward in enumerate(step_rewards):
                if np.isnan(reward):
                    nan_rewards_detected += 1
                    step_rewards[i] = 0.0  # Replace NaN with zero
                    if error_handler_available:
                        handle_error(
                            ValueError("NaN reward detected"),
                            "train_agent_episode.nan_reward",
                            additional_context={"env_idx": i, "step": step_count}
                        )
            
        except Exception as e:
            logger.error(f"Error stepping environment: {e}")
            if error_handler_available:
                handle_error(
                    e,
                    "train_agent_episode.step_env",
                    additional_context={"step": step_count}
                )
            # Break the episode
            break
        
        # Process each environment
        for env_idx in range(len(agent_envs.envs)):
            if dones[env_idx]:
                continue
                
            # Store trajectory data
            trajectories.append((
                observations[env_idx],
                actions[env_idx],
                log_probs[env_idx],
                step_rewards[env_idx],
                values[env_idx],
                dones[env_idx]
            ))
            
            # Store surprise-based experience in replay buffer
            if hasattr(agent, 'process_step_outcome'):
                # Provide additional info for surprise calculation
                step_info = {
                    'state': observations[env_idx],
                    'prediction_confidence': np.mean([c for c in confs]) if confs else 0.5,
                    'done': new_dones[env_idx],
                    'predictions': preds  # Include predictions for measuring surprise
                }
                try:
                    agent.process_step_outcome(actions[env_idx], step_rewards[env_idx], next_obs[env_idx], step_info)
                except Exception as e:
                    logger.error(f"Error processing step outcome: {e}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "train_agent_episode.process_outcome",
                            additional_context={"env_idx": env_idx, "step": step_count}
                        )
            
            # Handle completed trades for post-trade analysis
            if hasattr(agent, 'analyze_trade') and infos[env_idx] and 'trade_completed' in infos[env_idx]:
                trade_info = infos[env_idx].get('trade_info', {})
                if trade_info:
                    try:
                        lesson = agent.analyze_trade(trade_info)
                    except Exception as e:
                        logger.error(f"Error analyzing trade: {e}")
                        if error_handler_available:
                            handle_error(
                                e,
                                "train_agent_episode.analyze_trade",
                                additional_context={"env_idx": env_idx, "step": step_count}
                            )
        
        # Update for next step
        observations = next_obs
        dones = new_dones
        rewards.extend([r for r in step_rewards if not np.isnan(r)])
    
    # Perform PPO update with trajectories
    if trajectories:
        try:
            # Check if we need to enable gradient anomaly detection
            if nan_actions_detected > 0 or nan_rewards_detected > 0:
                # Enable anomaly detection for this update
                prev_anomaly_detection = torch.is_anomaly_enabled()
                torch.set_anomaly_enabled(True)
                
                # Perform update with extra care
                try:
                    agent.update_with_rollouts(trajectories)
                except RuntimeError as e:
                    # This could be a NaN gradient error
                    logger.error(f"Error during PPO update (NaN suspected): {e}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "train_agent_episode.nan_gradients",
                            additional_context={"nan_actions": nan_actions_detected, "nan_rewards": nan_rewards_detected}
                        )
                finally:
                    # Restore previous anomaly detection setting
                    torch.set_anomaly_enabled(prev_anomaly_detection)
            else:
                # Normal update
                agent.update_with_rollouts(trajectories)
                
            # Utilize experience replay with prioritization
            if hasattr(agent, 'update_with_prioritized_replay') and random.random() < 0.3:  # 30% chance each episode
                try:
                    agent.update_with_prioritized_replay()
                except Exception as e:
                    logger.error(f"Error during prioritized replay: {e}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "train_agent_episode.prioritized_replay",
                            additional_context={"error_details": str(e)}
                        )
        except Exception as e:
            logger.error(f"Error during agent update: {e}")
            if error_handler_available:
                handle_error(
                    e,
                    "train_agent_episode.agent_update",
                    additional_context={"error_details": str(e)}
                )
    
    # Calculate episode metrics
    avg_reward = np.mean(rewards) if rewards else 0
    metrics = {
        'reward': avg_reward, 
        'steps': step_count,
        'nan_actions': nan_actions_detected,
        'nan_rewards': nan_rewards_detected
    }
    
    # Update hyperparameters through meta-learning
    if hasattr(agent, 'adapt_hyperparameters') and hasattr(agent, 'recent_rewards'):
        try:
            # Record performance for hyperparameter optimization
            if hasattr(agent, 'record_hyperparameter_performance'):
                agent.record_hyperparameter_performance(avg_reward, episode_num)
            
            # Adapt hyperparameters if appropriate
            agent.adapt_hyperparameters(episode_num, list(agent.recent_rewards))
        except Exception as e:
            logger.error(f"Error adapting hyperparameters: {e}")
            if error_handler_available:
                handle_error(
                    e,
                    "train_agent_episode.adapt_hyperparameters",
                    additional_context={"error_details": str(e)}
                )
    
    return rewards, metrics

# Global flag to track interruption
_interrupted = False

def handle_interrupt(signum, frame):
    """Handle interrupt signals gracefully."""
    global _interrupted
    _interrupted = True
    logger.warning("Interrupt received. Finishing current episode and saving checkpoint...")
    # Don't exit immediately - let the training loop handle it gracefully

# Register signal handlers
signal.signal(signal.SIGINT, handle_interrupt)
signal.signal(signal.SIGTERM, handle_interrupt)

def calculate_weighted_fitness(metrics: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Calculate a weighted fitness score based on key metrics and config priorities.
    
    Args:
        metrics: Dictionary of calculated metrics for an agent/episode 
                 (e.g., {'net_profit': 100.0, 'win_rate': 0.6, 'max_drawdown': 0.1, ...}).
        config: Configuration dictionary containing priority weights 
                (e.g., {'priority_net_profit': 4.0, 'priority_win_rate': 3.0, ...}).
                
    Returns:
        Weighted fitness score.
    """
    # Get weights from config, defaulting to 1.0 if missing
    weights = {
        'net_profit': config.get('priority_net_profit', 1.0),
        'win_rate': config.get('priority_win_rate', 1.0),
        'max_drawdown': config.get('priority_max_drawdown', 1.0),
        'profit_factor': config.get('priority_profit_factor', 1.0)
    }
    
    # Normalize weights to sum to 1 (or handle zero sum)
    total_weight = sum(weights.values())
    if total_weight <= 1e-6: # Avoid division by zero or near-zero
        # If all weights are zero, return raw reward or a default value
        return metrics.get('reward', 0.0) 
        
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    # Get metric values, potentially normalizing or scaling them
    # Note: Normalization/scaling might be needed for metrics on different scales!
    # For simplicity here, we use raw values but assume higher is better (except drawdown).
    
    # Net Profit (Higher is better)
    net_profit_score = metrics.get('net_profit', 0.0) 
    # --- Add scaling/normalization if needed --- 
    # Example: net_profit_score = net_profit_score / config.get("INITIAL_CAPITAL", 100000.0) 
    
    # Win Rate (Higher is better, 0-1 scale)
    win_rate_score = metrics.get('win_rate', 0.0)
    
    # Max Drawdown (Lower is better, 0-1 scale - Invert for scoring: 1 - drawdown)
    max_drawdown_score = 1.0 - metrics.get('max_drawdown', 0.0) 
    
    # Profit Factor (Higher is better, typically > 1 is good)
    profit_factor = metrics.get('profit_factor', 1.0)
    # --- Add scaling/normalization if needed --- 
    # Example: Scale logarithmically or cap? For now, use directly.
    profit_factor_score = max(0, profit_factor) # Ensure non-negative
    
    # Calculate weighted score
    fitness_score = (
        normalized_weights['net_profit'] * net_profit_score +
        normalized_weights['win_rate'] * win_rate_score +
        normalized_weights['max_drawdown'] * max_drawdown_score +
        normalized_weights['profit_factor'] * profit_factor_score
    )
    
    # Optional: Combine with raw reward? 
    # fitness_score = 0.7 * fitness_score + 0.3 * metrics.get('reward', 0.0)
    
    return fitness_score

# Add training recovery integration helper
def create_recovery_checkpoint(model, optimizer, episode, best_reward, config, checkpoint_path,
                            additional_state=None, recovery_system=None):
    """
    Create a training checkpoint with optional recovery system integration.
    
    This function serves as a bridge between the existing checkpoint mechanism and the 
    recovery system if available.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer to save state
        episode: Current episode number
        best_reward: Best reward achieved
        config: Configuration parameters
        checkpoint_path: Path to save the checkpoint
        additional_state: Additional state to save
        recovery_system: Optional TrainingRecoverySystem instance
        
    Returns:
        bool: True if checkpoint was saved successfully, False otherwise
    """
    try:
        # If recovery system is available, use it for checkpoint creation
        if recovery_system:
            # Use the recovery system to create the checkpoint
            path = recovery_system.create_training_checkpoint(
                model=model, 
                optimizer=optimizer, 
                episode=episode, 
                best_reward=best_reward, 
                config=config, 
                additional_state=additional_state
            )
            
            # Return success based on if a path was returned
            return bool(path)
        
        # Otherwise, use the standard save_checkpoint function
        elif "save_checkpoint" in globals():
            # Use standard checkpoint function
            return save_checkpoint(
                model=model,
                optimizer=optimizer,
                episode=episode,
                best_reward=best_reward,
                config=config,
                path=checkpoint_path,
                performance_history=additional_state.get("performance_history") if additional_state else None,
                recent_rewards=additional_state.get("recent_rewards") if additional_state else None,
                horizons=additional_state.get("horizons") if additional_state else None,
                additional_state=additional_state
            )
        
        # If no checkpoint mechanism is available, create a basic one
        else:
            # Create a minimal checkpoint
            checkpoint = {
                "episode": episode,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict() if optimizer else None,
                "best_reward": best_reward,
                "config": config,
                "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
            }
            
            # Add additional state if provided
            if additional_state:
                for key, value in additional_state.items():
                    checkpoint[key] = value
                    
            # Create directory if needed
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Created checkpoint at {checkpoint_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error creating checkpoint: {str(e)}")
        return False

def train_model(df, config=None, save_path=None, recovery_state=None, progress_callback=None, recovery_system=None):
    """
    Train the trading model using the provided data and configuration.
    
    Args:
        df: DataFrame containing training data
        config: Optional configuration dictionary (will use TradeConfig if None)
        save_path: Optional path to save the model
        recovery_state: Optional recovery state for resuming training
        progress_callback: Optional callback function for progress updates
        recovery_system: Optional TrainingRecoverySystem instance for robust error handling
    """
    global _interrupted  # Ensure we can modify the global flag
    _interrupted = False # Reset interruption flag at the start of training

    # Use TradeConfig if no config provided
    if config is None:
        config = trade_config.as_dict()
    
    # Set up logging and directories
    if save_path is None:
        save_path = os.path.join(MODELS_DIR_DEFAULT, config.get("BUCKET", "Scalping"), "checkpoints")
    
    try:
        os.makedirs(save_path, exist_ok=True)
    except Exception as e:
        error_msg = f"Failed to create save directory: {str(e)}"
        logger.error(error_msg)
        if error_handler_available:
            handle_error(
                e,
                "train_model.create_directory",
                additional_context={"save_path": save_path}
            )
        # Use a default save path as fallback
        save_path = os.path.join(MODELS_DIR_DEFAULT, "emergency")
        os.makedirs(save_path, exist_ok=True)
        
    log_file = os.path.join(os.path.dirname(save_path), "training_log.txt")
    perf_log_file = os.path.join(os.path.dirname(save_path), "performance_log.json")
    
    # Configure logger
    def _log(msg):
        # Use logger.info or logger.warning etc directly for better level handling
        level_str = msg.split("]")[0][1:] if msg.startswith("[") else "INFO"
        actual_msg = msg.split("]", 1)[1].strip() if msg.startswith("[") else msg
        
        if level_str == "ERROR":
            logger.error(actual_msg)
        elif level_str == "WARNING":
            logger.warning(actual_msg)
        else: # INFO or other
            logger.info(actual_msg)

        # Still call progress callback if provided
        if progress_callback:
            # Send the original message format to callback
            progress_callback(msg + "\n") # Add newline for GUI log

    # Prepare DataFrame
    _log("[INFO] Preparing data...")
    try:
        df = preprocess_data(df, config)
    except Exception as e:
        error_msg = f"Error preprocessing data: {str(e)}"
        _log(f"[ERROR] {error_msg}")
        if error_handler_available:
            handle_error(
                e,
                "train_model.preprocess_data",
                additional_context={"error_msg": error_msg}
            )
        # Continue with original data
        _log("[WARNING] Continuing with original data due to preprocessing error")
    
    # Data checks
    if 'close' not in df.columns:
        error_msg = "Price data missing 'close' column"
        _log(f"[ERROR] {error_msg}")
        if error_handler_available:
            handle_error(
                ValueError(error_msg),
                "train_model.data_validation",
                additional_context={"missing_column": "close"}
            )
        return None, None, 0, 0
    
    # Check if training data is sufficient
    min_data_size = config.get("MIN_DATA_SIZE", 1000)
    if len(df) < min_data_size:
        error_msg = f"Not enough data. Got {len(df)} rows, need at least {min_data_size}"
        _log(f"[ERROR] {error_msg}")
        if error_handler_available:
            handle_error(
                ValueError(error_msg),
                "train_model.insufficient_data",
                additional_context={"rows": len(df), "min_required": min_data_size}
            )
        return None, None, 0, 0
    
    # Validate configuration parameters
    try:
        # Essential hyperparameters validation
        num_agents = config.get("ES_POPULATION", 8)
        if num_agents <= 0:
            error_msg = f"Invalid ES_POPULATION value: {num_agents}, must be positive"
            _log(f"[ERROR] {error_msg}")
            if error_handler_available:
                handle_error(
                    ValueError(error_msg),
                    "train_model.invalid_hyperparameter",
                    additional_context={"parameter": "ES_POPULATION", "value": num_agents}
                )
            # Use default value
            num_agents = 8
            _log(f"[INFO] Using default ES_POPULATION: {num_agents}")
            
        # Validate environment counts
        min_envs_per_agent = config.get("MIN_ENVS_PER_AGENT", 1)
        max_envs_per_agent = config.get("MAX_ENVS_PER_AGENT", 4)
        
        if min_envs_per_agent <= 0:
            error_msg = f"Invalid MIN_ENVS_PER_AGENT value: {min_envs_per_agent}, must be positive"
            _log(f"[WARNING] {error_msg}")
            if error_handler_available:
                handle_error(
                    ValueError(error_msg),
                    "train_model.invalid_hyperparameter",
                    additional_context={"parameter": "MIN_ENVS_PER_AGENT", "value": min_envs_per_agent}
                )
            min_envs_per_agent = 1
            
        if max_envs_per_agent <= 0:
            error_msg = f"Invalid MAX_ENVS_PER_AGENT value: {max_envs_per_agent}, must be positive"
            _log(f"[WARNING] {error_msg}")
            if error_handler_available:
                handle_error(
                    ValueError(error_msg),
                    "train_model.invalid_hyperparameter",
                    additional_context={"parameter": "MAX_ENVS_PER_AGENT", "value": max_envs_per_agent}
                )
            max_envs_per_agent = 4
            
        # Ensure min_envs doesn't exceed max_envs
        if min_envs_per_agent > max_envs_per_agent:
            error_msg = f"MIN_ENVS_PER_AGENT ({min_envs_per_agent}) cannot exceed MAX_ENVS_PER_AGENT ({max_envs_per_agent})"
            _log(f"[WARNING] {error_msg}")
            if error_handler_available:
                handle_error(
                    ValueError(error_msg),
                    "train_model.invalid_hyperparameter_relationship",
                    additional_context={"min_envs": min_envs_per_agent, "max_envs": max_envs_per_agent}
                )
            # Swap values if min > max
            min_envs_per_agent, max_envs_per_agent = max_envs_per_agent, min_envs_per_agent
            
        envs_per_agent = min(max_envs_per_agent, max(min_envs_per_agent, 1))
        num_envs = num_agents * envs_per_agent
        
        # Validate learning rate
        learning_rate = config.get("LEARNING_RATE", 0.0003)
        if learning_rate <= 0 or learning_rate > 1.0:
            error_msg = f"Invalid LEARNING_RATE value: {learning_rate}, must be between 0 and 1"
            _log(f"[WARNING] {error_msg}")
            if error_handler_available:
                handle_error(
                    ValueError(error_msg),
                    "train_model.invalid_hyperparameter",
                    additional_context={"parameter": "LEARNING_RATE", "value": learning_rate}
                )
            learning_rate = 0.0003
            
        # Validate hidden size
        hidden_size = config.get("HIDDEN_SIZE", 512)
        if hidden_size <= 0:
            error_msg = f"Invalid HIDDEN_SIZE value: {hidden_size}, must be positive"
            _log(f"[WARNING] {error_msg}")
            if error_handler_available:
                handle_error(
                    ValueError(error_msg),
                    "train_model.invalid_hyperparameter",
                    additional_context={"parameter": "HIDDEN_SIZE", "value": hidden_size}
                )
            hidden_size = 512
            
    except Exception as e:
        error_msg = f"Error validating configuration: {str(e)}"
        _log(f"[ERROR] {error_msg}")
        if error_handler_available:
            handle_error(
                e,
                "train_model.config_validation",
                additional_context={"error_details": str(e)}
            )
    
    # Environment dimensions
    _log("[INFO] Setting up model...")
    try:
        # Define explicit feature sets based on bucket type 
        standard_price_features = ['close', 'high', 'low', 'open', 'volume']
        
        # Technical indicators commonly used
        standard_technical_features = [
            'SMA9', 'SMA21', 'SMA50', 'SMA100', 'SMA200',
            'RSI14', 'Stoch_K', 'Stoch_D', 'CCI', 'BB_upper20', 
            'BB_mid20', 'BB_lower20', 'ATR', 'MACD', 'MACD_signal'
        ]
        
        # Get available columns that are actually in the dataframe
        available_price_features = [col for col in standard_price_features if col in df.columns]
        available_technical_features = [col for col in standard_technical_features if col in df.columns]
        
        # Add custom columns that appear to be numeric features
        # But exclude known non-feature columns
        non_feature_cols = ['timestamp', 'date', 'symbol', 'target', 'index']
        custom_numeric_cols = [
            col for col in df.columns 
            if col not in non_feature_cols 
            and col not in standard_price_features 
            and col not in standard_technical_features
            and df[col].dtype in ['float64', 'float32', 'int64', 'int32']
        ]
        
        # Combine all available features
        feature_cols = available_price_features + available_technical_features + custom_numeric_cols
        
        # Log what was found and what will be used
        _log(f"[INFO] Available price features: {available_price_features}")
        _log(f"[INFO] Available technical features: {available_technical_features}")
        _log(f"[INFO] Additional numeric features: {custom_numeric_cols}")
        
        # Make sure we have at least price data
        if not available_price_features:
            error_msg = "No price features found in data. At minimum 'close' is required."
            _log(f"[ERROR] {error_msg}")
            if error_handler_available:
                handle_error(
                    ValueError(error_msg),
                    "train_model.missing_price_features",
                    additional_context={"columns_found": list(df.columns)}
                )
            return None, None, 0, 0
        
        # Calculate input dimension based on actual features
        input_dim = len(feature_cols)
        
        # Store the feature columns in config for use by the environment
        config["feature_columns"] = feature_cols
        
        _log(f"[INFO] Determined input dimension: {input_dim}")
        _log(f"[INFO] Using features: {feature_cols}")
        
        # Check if input_dim makes sense
        if input_dim <= 0:
            error_msg = f"Invalid input dimension calculated: {input_dim}. Columns found: {df.columns.tolist()}"
            _log(f"[ERROR] {error_msg}")
            if error_handler_available:
                handle_error(
                    ValueError(error_msg),
                    "train_model.invalid_input_dim",
                    additional_context={"input_dim": input_dim, "df_columns": df.columns.tolist()}
                )
            return None, None, 0, 0
        
    except Exception as e:
        error_msg = f"Error determining input dimensions: {str(e)}"
        _log(f"[ERROR] {error_msg}")
        if error_handler_available:
            handle_error(
                e,
                "train_model.input_dim_calculation",
                additional_context={"error_details": str(e)}
            )
        return None, None, 0, 0
    
    # Get prediction horizons
    if config.get("USE_DYNAMIC_HORIZONS", True):
        # Generate initial horizons for bucket type
        bucket_type = config.get("BUCKET", "Scalping")
        # Adapt horizon ranges based on bucket type
        if bucket_type == "Scalping":
            min_horizon = 1
            max_horizon = 72
        elif bucket_type == "Short":
            min_horizon = 6
            max_horizon = 144
        elif bucket_type == "Medium":
            min_horizon = 24
            max_horizon = 288
        elif bucket_type == "Long":
            min_horizon = 72
            max_horizon = 576
        else:
            min_horizon = 12
            max_horizon = 144
            
        # Import dynamic horizon generation
        try:
            from src.utils.utils import generate_dynamic_horizons
        except ImportError:
             _log("[ERROR] Failed to import generate_dynamic_horizons from src.utils.utils")
             return None, None, 0, 0

        # Generate horizons with distribution based on bucket type
        if bucket_type == "Scalping":
            mode = "log"  # Emphasize shorter horizons
        elif bucket_type == "Long":
            mode = "exp"  # Emphasize longer horizons
        else:
            mode = "mixed"  # Balanced approach
            
        horizons = generate_dynamic_horizons(
            min_horizon=min_horizon,
            max_horizon=max_horizon,
            num_horizons=6,
            mode=mode,
            seed=config.get("RANDOM_SEED", 42)
        )
        
        _log(f"[INFO] Generated dynamic prediction horizons for {bucket_type}: {horizons}")
    else:
        # Use fixed horizons by bucket type
        prediction_horizons = {
            "Scalping": [6, 12, 24, 36],
            "Short": [12, 36, 72, 144],
            "Medium": [24, 72, 144, 288],
            "Long": [72, 144, 288, 576]
        }
        horizons = prediction_horizons.get(config.get("BUCKET", "Scalping"), [12, 36, 72, 144])
        _log(f"[INFO] Using fixed prediction horizons: {horizons}")
    
    _log(f"[INFO] Training {num_agents} agents with {envs_per_agent} environments each ({num_envs} total)")
    
    # Get device
    try:
        _log("[INFO] Determining device for training...")
        # Use config setting for device preference
        preferred_device = config.get("DEVICE_PREFERENCE", "auto").lower()
        if preferred_device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif preferred_device == "cpu":
            device = torch.device("cpu")
        else: # auto or invalid setting
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        _log(f"[INFO] Using device: {device}")
        
        # Get optimal GPU targets if using GPU
        if device.type == 'cuda':
            try:
                gpu_target_low, gpu_target_high = get_optimal_gpu_targets()
                _log(f"[INFO] GPU target range: {gpu_target_low*100:.1f}% - {gpu_target_high*100:.1f}%")
            except Exception as e:
                error_msg = f"Error getting GPU targets: {str(e)}"
                _log(f"[WARNING] {error_msg}")
                if error_handler_available:
                    handle_error(
                        e,
                        "train_model.gpu_targets",
                        additional_context={"error_details": str(e)}
                    )
                # Use default values
                gpu_target_low, gpu_target_high = 0.65, 0.85
                _log(f"[INFO] Using default GPU targets: {gpu_target_low*100:.1f}% - {gpu_target_high*100:.1f}%")
        else:
            gpu_target_low, gpu_target_high = 0.65, 0.85
    except Exception as e:
        error_msg = f"Error determining device: {str(e)}"
        _log(f"[ERROR] {error_msg}")
        if error_handler_available:
            handle_error(
                e,
                "train_model.device_detection",
                additional_context={"error_details": str(e)}
            )
        # Use CPU as fallback
        device = torch.device("cpu")
        gpu_target_low, gpu_target_high = 0.65, 0.85
        _log("[INFO] Falling back to CPU device")
    
    # Create vectorized environments
    _log("[INFO] Creating vectorized environments...")
    try:
        envs = make_vec_env(df, config, num_envs, device)
        # Check the observation space after creation
        obs_space = envs.observation_space
        _log(f"[INFO] Environment observation space: {obs_space}")
        
        # CRITICAL FIX: Resolve dimension mismatch between agent and environment
        if obs_space.shape[0] != input_dim:
            _log(f"[WARNING] Observation space dimension ({obs_space.shape[0]}) does not match calculated input dimension ({input_dim})")
            
            # Validate which dimension is correct by checking environment features
            env_feature_count = len(config.get("feature_columns", []))
            
            if env_feature_count > 0 and obs_space.shape[0] == env_feature_count:
                # Environment dimension is correct, update input_dim
                _log(f"[INFO] Using environment observation dimension ({obs_space.shape[0]}) as correct input_dim")
                input_dim = obs_space.shape[0]
                config["input_dim"] = input_dim
            elif env_feature_count > 0 and input_dim == env_feature_count:
                # Calculated input_dim is correct, need to fix environment
                _log(f"[ERROR] Environment observation space is incorrect. Expected {input_dim}, got {obs_space.shape[0]}")
                if error_handler_available:
                    handle_error(
                        ValueError(f"Environment observation space mismatch: expected {input_dim}, got {obs_space.shape[0]}"),
                        "train_model.observation_space_mismatch",
                        additional_context={"expected_dim": input_dim, "actual_dim": obs_space.shape[0]}
                    )
                return None, None, 0, 0
            else:
                # Use environment dimension as fallback and update feature tracking
                _log(f"[INFO] Adjusting to environment observation dimension ({obs_space.shape[0]})")
                input_dim = obs_space.shape[0]
                config["input_dim"] = input_dim
                # Update feature columns to match actual observation space
                if len(config.get("feature_columns", [])) != input_dim:
                    _log("[WARNING] Feature columns list doesn't match observation space. Using environment dimension.")
        else:
            _log(f"[INFO] Input dimension validation passed: {input_dim}")
            
    except Exception as e:
        error_msg = f"Error creating environments: {str(e)}"
        _log(f"[ERROR] {error_msg}")
        if error_handler_available:
            handle_error(
                e,
                "train_model.create_environments",
                additional_context={"error_details": str(e)}
            )
        return None, None, 0, 0
    
    # Initialize agents
    _log("[INFO] Initializing population...")
    
    # Initialize ES population
    population = None # Initialize to None
    agents = []
    start_time = time.time() # Initialize start_time here
    try:
        _log("[INFO] Creating model with input dimension: " + str(input_dim))
        # Check model type is valid before trying to create it
        model_type = config.get("MODEL_TYPE", "actor_critic")
        valid_model_types = ["actor_critic", "lstm", "hybrid", "transformer"]
        if model_type not in valid_model_types:
            error_msg = f"Invalid model type: {model_type}. Must be one of {valid_model_types}"
            _log(f"[WARNING] {error_msg}")
            if error_handler_available:
                handle_error(
                    ValueError(error_msg),
                    "train_model.invalid_model_type",
                    additional_context={"model_type": model_type, "valid_types": valid_model_types}
                )
            # Use default model type
            model_type = "actor_critic"
            _log(f"[INFO] Using default model type: {model_type}")
        
        # CRITICAL FIX: Validate bucket compatibility between agent and environment
        agent_bucket = config.get("BUCKET", "Scalping")
        env_bucket = config.get("trading_mode", agent_bucket)  # Environment uses trading_mode
        
        if agent_bucket != env_bucket:
            _log(f"[WARNING] Agent bucket ({agent_bucket}) doesn't match environment bucket ({env_bucket})")
            # Standardize to agent bucket preference
            config["trading_mode"] = agent_bucket
            _log(f"[INFO] Synchronized environment to use agent bucket: {agent_bucket}")
        
        # Validate bucket-specific configuration requirements
        bucket_requirements = {
            "Scalping": {"min_window_size": 30, "max_horizon": 72, "risk_level": "aggressive"},
            "Short": {"min_window_size": 60, "max_horizon": 144, "risk_level": "moderate"},
            "Medium": {"min_window_size": 144, "max_horizon": 288, "risk_level": "moderate"},
            "Long": {"min_window_size": 288, "max_horizon": 576, "risk_level": "conservative"}
        }
        
        bucket_req = bucket_requirements.get(agent_bucket, {})
        if bucket_req:
            # Validate window size
            current_window = config.get("WINDOW_SIZE", config.get("window_size", 60))
            min_window = bucket_req.get("min_window_size", 60)
            if current_window < min_window:
                _log(f"[WARNING] Window size ({current_window}) too small for {agent_bucket} bucket. Adjusting to {min_window}")
                config["WINDOW_SIZE"] = min_window
                config["window_size"] = min_window
            
            # Set appropriate risk level if not specified
            if "risk_level" not in config:
                config["risk_level"] = bucket_req["risk_level"]
                _log(f"[INFO] Set risk level to {bucket_req['risk_level']} for {agent_bucket} bucket")
        
        _log(f"[INFO] Bucket compatibility validation completed for: {agent_bucket}")
        
        # Create population with error catching
        population = ESPopulation(
            num_agents, 
            input_dim, 
            config.get("HIDDEN_SIZE", 512), 
            config.get("LEARNING_RATE", 0.0003), 
            horizons=horizons,
            use_mixed_precision=config.get("USE_MIXED_PRECISION", False) and device.type == 'cuda', # Only use MP on CUDA
            model_type=model_type, 
            device=device, 
            config=config
        )
        agents = population.agents
        _log(f"[INFO] Successfully created {len(agents)} agents with model type: {model_type}")
    except RuntimeError as e:
        # Check specifically for CUDA out-of-memory errors
        if "CUDA out of memory" in str(e):
            error_msg = f"GPU out of memory error: {str(e)}"
            _log(f"[ERROR] {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "train_model.gpu_out_of_memory",
                    additional_context={
                        "error_details": str(e),
                        "input_dim": input_dim,
                        "hidden_size": config.get("HIDDEN_SIZE", 512)
                    }
                )
            # Try to recover by using CPU instead
            _log("[INFO] Attempting to recover by using CPU instead...")
            try:
                device = torch.device("cpu")
                _log("[INFO] Switched to CPU for model creation")
                population = ESPopulation(
                    num_agents, 
                    input_dim, 
                    config.get("HIDDEN_SIZE", 512), 
                    config.get("LEARNING_RATE", 0.0003), 
                    horizons=horizons,
                    use_mixed_precision=False,  # Disable mixed precision on CPU
                    model_type=config.get("MODEL_TYPE", "actor_critic"), 
                    device=device, 
                    config=config
                )
                agents = population.agents
                _log("[INFO] Successfully created agents on CPU")
            except Exception as cpu_e:
                error_msg = f"Failed to create model on CPU: {str(cpu_e)}"
                _log(f"[ERROR] {error_msg}")
                if error_handler_available:
                    handle_error(
                        cpu_e,
                        "train_model.model_creation_cpu_fallback",
                        additional_context={"error_details": str(cpu_e)}
                    )
                # Clean up environments and exit
                try:
                    envs.close()
                except:
                    pass
                return None, None, 0, 0
        else:
            # Handle other runtime errors
            error_msg = f"Runtime error initializing agent population: {str(e)}"
            _log(f"[ERROR] {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "train_model.initialize_population_runtime",
                    additional_context={
                        "error_details": str(e),
                        "num_agents": num_agents,
                        "input_dim": input_dim
                    }
                )
            # Clean up environments
            try:
                envs.close()
            except:
                pass
            return None, None, 0, 0
    except Exception as e:
        error_msg = f"Error initializing agent population: {str(e)}"
        _log(f"[ERROR] {error_msg}")
        if error_handler_available:
            handle_error(
                e,
                "train_model.initialize_population",
                additional_context={
                    "error_details": str(e),
                    "num_agents": num_agents,
                    "input_dim": input_dim
                }
            )
        # Clean up environments
        try:
            envs.close()
        except:
            pass
        return None, None, 0, 0
    
    # Initialize Cross-Bucket Knowledge Transfer if enabled
    use_cross_bucket_transfer = config.get("USE_CROSS_BUCKET_TRANSFER", True)
    knowledge_transfer = None
    
    if use_cross_bucket_transfer:
        try:
            from src.agent.agent import CrossBucketKnowledgeTransfer
            knowledge_transfer = CrossBucketKnowledgeTransfer(config)
            
            # Register this bucket's agent (initially the first one)
            bucket_type = config.get("BUCKET", "Scalping")
            if agents: # Ensure agents list is not empty
                knowledge_transfer.register_agent(bucket_type, agents[0])
                _log(f"[INFO] Initialized cross-bucket knowledge transfer for {bucket_type}")
            else:
                 _log(f"[WARNING] Cannot initialize knowledge transfer: No agents available.")
                 use_cross_bucket_transfer = False

            # Try to load knowledge from previous buckets if available
            knowledge_dir = config.get("KNOWLEDGE_TRANSFER_DIR", "knowledge_transfer")
            if os.path.exists(knowledge_dir):
                _log(f"[INFO] Looking for knowledge transfer data in {knowledge_dir}...")
                # This would normally be implemented with a loading mechanism
                # For now, we'll just log the intent
        except ImportError as e:
            error_msg = f"Could not import CrossBucketKnowledgeTransfer: {str(e)}"
            _log(f"[WARNING] {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "train_model.import_knowledge_transfer",
                    additional_context={"error_details": str(e)}
                )
            # Disable cross-bucket transfer
            use_cross_bucket_transfer = False
            _log("[INFO] Cross-bucket knowledge transfer disabled due to import error")
        except Exception as e:
            error_msg = f"Error initializing knowledge transfer: {str(e)}"
            _log(f"[WARNING] {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "train_model.initialize_knowledge_transfer",
                    additional_context={"error_details": str(e)}
                )
            # Disable cross-bucket transfer
            use_cross_bucket_transfer = False
            _log("[INFO] Cross-bucket knowledge transfer disabled due to initialization error")
    
    # ===== PREDICTIVE AGENT INITIALIZATION =====
    # Create a dedicated predictive agent for this bucket that will train alongside the main agent
    # and provide predictions to inform the bucket's decision making
    predictive_agent = None
    predictive_agent_enabled = config.get("USE_PREDICTIVE_AGENT", True)
    
    if predictive_agent_enabled and agents:
        try:
            _log(f"[INFO] Initializing dedicated predictive agent for bucket {bucket_type}")
            
            # Create bucket-specific predictive agent directory
            bucket_dir = os.path.dirname(save_path)  # Parent of checkpoints dir
            predictive_agent_dir = os.path.join(bucket_dir, "predictive_agent")
            os.makedirs(predictive_agent_dir, exist_ok=True)
            
            # Create predictive agent config tailored for this bucket
            predictive_config = config.copy()
            predictive_config.update({
                "AGENT_TYPE": "predictive",
                "BUCKET_TYPE": bucket_type,
                "PREDICTION_FOCUS": True,
                "SAVE_DIR": predictive_agent_dir,
                # Smaller learning rate for more stable predictions
                "LEARNING_RATE": config.get("LEARNING_RATE", 0.0003) * 0.5,
                # Focus on prediction accuracy over trading performance
                "PREDICTION_WEIGHT": 0.8,
                "TRADING_WEIGHT": 0.2
            })
            
            # Create a single predictive agent (not a population)
            predictive_agent = PPOAgent(
                input_dim=input_dim,
                hidden_size=config.get("HIDDEN_SIZE", 512),
                lr=predictive_config["LEARNING_RATE"],
                horizons=horizons,
                use_mixed_precision=config.get("USE_MIXED_PRECISION", False) and device.type == 'cuda',
                model_type=config.get("MODEL_TYPE", "actor_critic"),
                device=device,
                config=predictive_config
            )
            
            # Register predictive agent with knowledge transfer system if available
            if knowledge_transfer:
                # Use a special key to identify this as a predictive agent
                predictive_bucket_key = f"{bucket_type}_predictive"
                knowledge_transfer.register_agent(predictive_bucket_key, predictive_agent)
                _log(f"[INFO] Registered predictive agent for cross-bucket knowledge transfer")
            
            # Set up predictive agent save path
            predictive_save_path = os.path.join(predictive_agent_dir, f"{bucket_type.lower()}_predictive_agent.pth")
            
            # Try to load existing predictive agent if available
            if os.path.exists(predictive_save_path):
                try:
                    checkpoint = torch.load(predictive_save_path, map_location=device)
                    predictive_agent.model.load_state_dict(checkpoint['model_state_dict'])
                    predictive_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'scheduler_state_dict' in checkpoint:
                        predictive_agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    _log(f"[INFO] Loaded existing predictive agent from {predictive_save_path}")
                except Exception as load_e:
                    _log(f"[WARNING] Could not load existing predictive agent: {str(load_e)}")
                    _log("[INFO] Starting with fresh predictive agent")
            
            _log(f"[INFO] Predictive agent for {bucket_type} bucket initialized successfully")
            _log(f"[INFO] Predictive agent will save to: {predictive_save_path}")
            
        except Exception as e:
            error_msg = f"Error initializing predictive agent: {str(e)}"
            _log(f"[WARNING] {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "train_model.initialize_predictive_agent",
                    additional_context={"error_details": str(e), "bucket_type": bucket_type}
                )
            predictive_agent = None
            _log("[INFO] Continuing training without predictive agent")
    
    # ===== END PREDICTIVE AGENT INITIALIZATION =====
    
    # Track best agent and metrics
    best_agent = None
    best_fitness_score = -float('inf') # Changed from best_reward
    recent_rewards = deque(maxlen=5) # Keep raw rewards for logging/analysis if needed
    recent_fitness = deque(maxlen=config.get("FITNESS_SMOOTHING_WINDOW", 5)) # Track recent fitness
    recent_errors = deque(maxlen=10) # Keep track of recent errors if needed
    
    # Set up performance tracking
    performance_history = []
    
    # Resume from saved state if provided
    start_episode = 0
    episode = 0 # Initialize episode here
    no_improvement_count = 0 # Initialize counter
    
    # Try to load from recovery state if provided
    if recovery_state:
        try:
            _log("[INFO] Attempting to resume training from recovery state...")
            
            if isinstance(recovery_state, str) and os.path.exists(recovery_state):
                # Load from file path
                try:
                    with open(recovery_state, 'r') as f:
                        recovery_data = json.load(f)
                    _log(f"[INFO] Loaded recovery state from {recovery_state}")
                except Exception as e:
                    error_msg = f"Error loading recovery state file: {str(e)}"
                    _log(f"[ERROR] {error_msg}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "train_model.load_recovery_file",
                            additional_context={"recovery_path": recovery_state}
                        )
                    recovery_data = None
            elif isinstance(recovery_state, dict):
                # Already a dictionary
                recovery_data = recovery_state
            else:
                recovery_data = None
                _log("[WARNING] Invalid recovery state format")
            
            if recovery_data:
                # Extract recovery data
                try:
                    start_episode = recovery_data.get('episode', 0)
                    best_agent_idx = recovery_data.get('best_agent_idx', 0) # This index is now based on fitness
                    best_fitness_score = recovery_data.get('best_fitness_score', -float('inf')) # Changed from best_reward
                    checkpoint_path = recovery_data.get('checkpoint_path', '') # Get path if available
                    
                    _log(f"[INFO] Resuming from episode {start_episode}, best fitness: {best_fitness_score:.4f}") # Updated log message
                    
                    # Check if there's a specific agent checkpoint to load
                    # Prefer checkpoint_path from recovery state if it exists
                    if not checkpoint_path:
                         # If not in recovery state, try finding the latest checkpoint
                         checkpoint_files = sorted(Path(save_path).glob('checkpoint_*.pth'), key=os.path.getmtime, reverse=True)
                         if checkpoint_files:
                             checkpoint_path = str(checkpoint_files[0])
                             _log(f"[INFO] Found latest checkpoint automatically: {checkpoint_path}")
                         else:
                              _log("[INFO] No checkpoint path provided or found, starting fresh.")

                    if checkpoint_path and os.path.exists(checkpoint_path):
                        _log(f"[INFO] Loading checkpoint from {checkpoint_path}...")
                        try:
                            # Ensure population and agents exist before trying to load
                            if population is None or not agents:
                                raise ValueError("Population/Agents not initialized before loading checkpoint.")

                            # Load checkpoint onto the correct device
                            checkpoint = torch.load(checkpoint_path, map_location=device)
                            
                            # Check if checkpoint is valid
                            required_keys = ["model_state", "optimizer_state", "episode", "best_fitness_score"] # Added best_fitness_score check
                            if not all(key in checkpoint for key in required_keys):
                                raise ValueError(f"Checkpoint is missing required keys: {[k for k in required_keys if k not in checkpoint]}")
                            
                            # Load state into best agent (or all agents if needed)
                            # Assuming ES evolves the whole population, maybe load state for all agents?
                            # For now, loading only the agent identified as best in the checkpoint
                            
                            # Get the index stored in the checkpoint if available, otherwise use from recovery data
                            agent_idx_to_load = checkpoint.get('best_agent_idx', best_agent_idx) 
                            
                            if agent_idx_to_load < len(agents):
                                agent_to_load = agents[agent_idx_to_load]
                                agent_to_load.model.load_state_dict(checkpoint["model_state"])
                                agent_to_load.optimizer.load_state_dict(checkpoint["optimizer_state"])
                                best_agent = agent_to_load # Set the loaded agent as best_agent (potentially temporary until next eval)
                                _log(f"[INFO] Loaded state for agent {agent_idx_to_load}")
                            else:
                                _log(f"[WARNING] Best agent index {agent_idx_to_load} out of range, cannot load state.")

                            # Extract additional state if available
                            if "performance_history" in checkpoint:
                                performance_history = checkpoint["performance_history"]
                                _log(f"[INFO] Loaded performance history with {len(performance_history)} entries")
                            
                            # Load recent fitness if available (replace recent_rewards loading)
                            if "recent_fitness" in checkpoint:
                                recent_fitness = deque(checkpoint["recent_fitness"], maxlen=config.get("FITNESS_SMOOTHING_WINDOW", 5))
                                _log("[INFO] Loaded recent fitness history")
                            elif "recent_rewards" in checkpoint: # Fallback for older checkpoints
                                recent_rewards = deque(checkpoint["recent_rewards"], maxlen=5) 
                                # Estimate fitness from rewards if possible, otherwise reset
                                recent_fitness = deque([-np.inf]*len(recent_rewards), maxlen=config.get("FITNESS_SMOOTHING_WINDOW", 5))
                                _log("[INFO] Loaded recent rewards history (fitness estimated/reset)")

                            # ... (load horizons, no_improvement_count, etc.) ...
                            
                            # Restore the best fitness score from the checkpoint
                            best_fitness_score = checkpoint.get("best_fitness_score", best_fitness_score)
                            
                            _log(f"[INFO] Successfully loaded checkpoint and resumed state. Best fitness score: {best_fitness_score:.4f}")
                            
                        except RuntimeError as e:
                            error_msg = f"Error loading model weights: {str(e)}"
                            _log(f"[ERROR] {error_msg}")
                            if error_handler_available:
                                handle_error(
                                    e,
                                    "train_model.load_checkpoint_weights",
                                    additional_context={"checkpoint_path": checkpoint_path}
                                )
                            _log("[WARNING] Continuing with newly initialized model")
                            
                        except Exception as e:
                            error_msg = f"Error loading checkpoint: {str(e)}"
                            _log(f"[ERROR] {error_msg}")
                            if error_handler_available:
                                handle_error(
                                    e,
                                    "train_model.load_checkpoint",
                                    additional_context={"checkpoint_path": checkpoint_path}
                                )
                            _log("[WARNING] Continuing with newly initialized models")
                    
                    # Update episode counter for resumption
                    episode = start_episode # Set the current episode
                    
                except Exception as e:
                    error_msg = f"Error extracting recovery data: {str(e)}"
                    _log(f"[ERROR] {error_msg}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "train_model.process_recovery_data",
                            additional_context={"recovery_data": str(recovery_data)[:100] + "..." if len(str(recovery_data)) > 100 else str(recovery_data)}
                        )
                    # Reset to defaults
                    start_episode = 0
                    episode = 0
        except Exception as e:
            error_msg = f"Error in resumption process: {str(e)}"
            _log(f"[ERROR] {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "train_model.resumption_process",
                    additional_context={"recovery_state_type": type(recovery_state).__name__}
                )
            # Reset to defaults
            start_episode = 0
            episode = 0
    
    # Training loop
    horizon_update_freq = config.get("HORIZON_UPDATE_FREQ", 10)  # Update horizons every 10 episodes
    max_episodes = config.get("MAX_EPISODES", 1000)

    try:
        _log(f"[INFO] Beginning training from episode {episode + 1} up to {max_episodes}...")
        start_time = time.time() # Record overall start time

        while episode < max_episodes:
            _log(f"[EPISODE {episode + 1}/{max_episodes}] Starting training iteration")
            episode_start_time = time.time()
            
            # Reset population fitness scores for this episode evaluation
            population.reset_fitness() # Changed from reset_rewards
            
            # Run parallel episodes for all agents
            total_rewards = [] # Still track raw rewards for logging/analysis
            agent_metrics = []
            agent_fitness_scores = [] # Store fitness scores for this episode
            
            # Process each agent
            for agent_idx in range(num_agents):
                agent_envs_list = envs.envs[agent_idx * envs_per_agent : (agent_idx + 1) * envs_per_agent]
                agent_env_wrapper = VecEnvWrapper(agent_envs_list) # Create a temporary wrapper for the slice

                # Train agent
                rewards, metrics = train_agent_episode(
                    agent_env_wrapper, # Pass the wrapper for this agent's envs
                    agents[agent_idx],
                    horizons,
                    config,
                    device,
                    episode
                )
                
                # Record raw rewards and detailed metrics
                total_rewards.extend(rewards)
                agent_metrics.append(metrics)
                
                # Calculate Weighted Fitness Score
                fitness_score = calculate_weighted_fitness(metrics, config)
                agent_fitness_scores.append(fitness_score)
                
                # Record fitness score in the population object
                population.record_fitness(agent_idx, fitness_score) # Changed from record_reward

            # ===== PREDICTIVE AGENT TRAINING =====
            # Train the dedicated predictive agent alongside the main agents
            if predictive_agent is not None:
                try:
                    _log(f"[PREDICTIVE] Training predictive agent for {bucket_type} bucket (Episode {episode + 1})")
                    
                    # Use a single environment for predictive agent training (first env)
                    predictive_env = envs.envs[0]
                    predictive_env_wrapper = VecEnvWrapper([predictive_env])
                    
                    # Train predictive agent with focus on prediction accuracy
                    predictive_rewards, predictive_metrics = train_agent_episode(
                        predictive_env_wrapper,
                        predictive_agent,
                        horizons,
                        predictive_config,  # Use the predictive agent's specialized config
                        device,
                        episode
                    )
                    
                    # ===== ENHANCED PREDICTIVE EVALUATION =====
                    # Evaluate predictive agent quality using enhanced system
                    try:
                        from .predictive_agent_evaluator import PredictiveAgentEvaluator
                        
                        # Create evaluator for this bucket
                        evaluator = PredictiveAgentEvaluator(bucket_type)
                        
                        # Get recent market data for evaluation
                        eval_data = df.tail(min(1000, len(df))).copy()
                        
                        # Run evaluation if we have enough data
                        if len(eval_data) > 100:
                            _log(f"[PREDICTIVE] Running enhanced evaluation on {len(eval_data)} data points")
                            
                            # Simulate some predictions for evaluation (in real training, these would come from actual agent predictions)
                            # This is a placeholder - in practice, you'd collect actual predictions during training
                            evaluation_results = evaluator.evaluate_predictions(
                                predictions=[],  # Empty for now - would be filled with actual predictions
                                data=eval_data,
                                current_step=len(df) - 1
                            )
                            
                            # Extract reward adjustment from evaluation
                            reward_adjustment = evaluation_results.get("reward_adjustment", 0.0)
                            overall_score = evaluation_results.get("overall_score", 0.0)
                            
                            # Apply reward adjustment to predictive agent
                            if reward_adjustment != 0.0:
                                predictive_agent.apply_reward_adjustment(reward_adjustment)
                                _log(f"[PREDICTIVE] Applied reward adjustment: {reward_adjustment:+.3f}")
                            
                            # Log evaluation summary
                            _log(f"[PREDICTIVE] Evaluation Score: {overall_score:.3f}")
                            
                            # Store evaluation metrics
                            predictive_metrics.update({
                                "evaluation_score": overall_score,
                                "reward_adjustment": reward_adjustment,
                                "enhanced_evaluation": True
                            })
                            
                        else:
                            _log(f"[PREDICTIVE] Skipping evaluation - insufficient data ({len(eval_data)} points)")
                            
                    except ImportError:
                        _log(f"[PREDICTIVE] Enhanced evaluation not available - using standard metrics")
                    except Exception as eval_e:
                        _log(f"[WARNING] Error during enhanced evaluation: {str(eval_e)}")
                    
                    # ===== END ENHANCED PREDICTIVE EVALUATION =====
                    
                    # Generate predictions for the main agents to use
                    if len(predictive_rewards) > 0:
                        # Store predictions in the bucket's predictive agent directory
                        predictions_file = os.path.join(predictive_agent_dir, f"{bucket_type.lower()}_predictions.json")
                        predictions_data = {
                            "episode": episode + 1,
                            "bucket_type": bucket_type,
                            "timestamp": time.time(),
                            "predicted_performance": np.mean(predictive_rewards),
                            "prediction_confidence": predictive_metrics.get("confidence", 0.5),
                            "market_sentiment": predictive_metrics.get("market_sentiment", "neutral"),
                            "horizons": horizons,
                            "enhanced_evaluation_score": predictive_metrics.get("evaluation_score", 0.0),
                            "reward_adjustment": predictive_metrics.get("reward_adjustment", 0.0),
                            "recommendations": {
                                "suggested_action": "hold" if np.mean(predictive_rewards) > 0 else "caution",
                                "confidence_level": predictive_metrics.get("confidence", 0.5),
                                "prediction_accuracy": predictive_metrics.get("prediction_accuracy", 0.0),
                                "horizon_appropriateness": predictive_metrics.get("evaluation_score", 0.0)
                            }
                        }
                        
                        # Save predictions for the main agents to access
                        try:
                            with open(predictions_file, 'w') as f:
                                json.dump(predictions_data, f, indent=2)
                            _log(f"[PREDICTIVE] Saved predictions to {predictions_file}")
                        except Exception as save_e:
                            _log(f"[WARNING] Could not save predictions: {str(save_e)}")
                        
                        # ===== SAVE FORECAST HISTORY WITH TIMESTAMPS =====
                        # Save detailed forecast history from the predictive agent
                        if hasattr(predictive_agent, 'model') and hasattr(predictive_agent.model, 'forecast_history'):
                            try:
                                forecast_history_file = os.path.join(predictive_agent_dir, f"{bucket_type.lower()}_forecast_history.json")
                                # Get forecast history from the predictive agent's model
                                forecast_history = predictive_agent.model.forecast_history
                                
                                if forecast_history:
                                    # Convert datetime objects to ISO strings for JSON serialization
                                    serializable_history = []
                                    for record in forecast_history:
                                        serialized_record = record.copy()
                                        if isinstance(serialized_record.get("timestamp"), datetime):
                                            serialized_record["timestamp"] = serialized_record["timestamp"].isoformat()
                                        serializable_history.append(serialized_record)
                                    
                                    with open(forecast_history_file, 'w') as f:
                                        json.dump({
                                            "bucket_type": bucket_type,
                                            "episode": episode + 1,
                                            "saved_at": datetime.now().isoformat(),
                                            "total_forecasts": len(serializable_history),
                                            "forecast_history": serializable_history
                                        }, f, indent=2)
                                    
                                    _log(f"[PREDICTIVE] Saved {len(serializable_history)} forecast records to {forecast_history_file}")
                                else:
                                    _log(f"[PREDICTIVE] No forecast history to save for {bucket_type}")
                                    
                            except Exception as history_save_e:
                                _log(f"[WARNING] Could not save forecast history: {str(history_save_e)}")
                        
                        # ===== END SAVE FORECAST HISTORY =====
                        
                        # Make predictions available to main agents through knowledge transfer
                        if knowledge_transfer:
                            predictive_bucket_key = f"{bucket_type}_predictive"
                            knowledge_transfer.register_agent(predictive_bucket_key, predictive_agent)
                            _log(f"[PREDICTIVE] Updated predictive agent in knowledge transfer system")
                    
                    # Save predictive agent checkpoint periodically
                    if (episode + 1) % config.get("PREDICTIVE_SAVE_INTERVAL", 10) == 0:
                        try:
                            predictive_checkpoint = {
                                'model_state_dict': predictive_agent.model.state_dict(),
                                'optimizer_state_dict': predictive_agent.optimizer.state_dict(),
                                'scheduler_state_dict': predictive_agent.scheduler.state_dict() if hasattr(predictive_agent, 'scheduler') else None,
                                'episode': episode + 1,
                                'config': predictive_config,
                                'bucket_type': bucket_type,
                                'performance_metrics': predictive_metrics
                            }
                            torch.save(predictive_checkpoint, predictive_save_path)
                            _log(f"[PREDICTIVE] Saved predictive agent checkpoint (Episode {episode + 1})")
                        except Exception as save_e:
                            _log(f"[WARNING] Could not save predictive agent checkpoint: {str(save_e)}")
                    
                    _log(f"[PREDICTIVE] Predictive agent training completed for episode {episode + 1}")
                    
                except Exception as pred_e:
                    _log(f"[WARNING] Error training predictive agent: {str(pred_e)}")
                    if error_handler_available:
                        handle_error(
                            pred_e,
                            "train_model.predictive_agent_training",
                            additional_context={"episode": episode + 1, "bucket_type": bucket_type}
                        )
                    # Continue training without predictive agent for this episode
            
            # ===== END PREDICTIVE AGENT TRAINING =====

            # Calculate statistics (raw reward stats for logging)
            mean_reward = np.mean(total_rewards) if total_rewards else 0
            std_reward = np.std(total_rewards) if total_rewards else 0
            
            # Calculate fitness stats for tracking improvement
            mean_fitness = np.mean(agent_fitness_scores) if agent_fitness_scores else -np.inf
            std_fitness = np.std(agent_fitness_scores) if agent_fitness_scores else 0
            
            # Update dynamic horizons if enabled
            if config.get("USE_DYNAMIC_HORIZONS", True) and episode > 0 and episode % horizon_update_freq == 0:
                _log(f"[INFO] Updating prediction horizons (episode {episode})...")
                best_agent_idx_current_eval = population.get_best_agent_idx() # Get best from current eval
                best_agent_model = agents[best_agent_idx_current_eval].model
                
                # Get current market data slice
                market_slice = df.iloc[-min(100, len(df)):].copy()
                
                # Generate new horizons based on market conditions and past performance
                updated_horizons = adapt_prediction_horizons(best_agent_model, market_slice, config)
                
                # Apply horizon updates to all agents
                for idx, agent in enumerate(agents):
                    updated = agent.model.update_horizons(updated_horizons)
                    if updated and idx == best_agent_idx_current_eval:
                        _log(f"[INFO] Updated horizons to: {updated_horizons}")
            
            # Perform cross-bucket knowledge transfer if enabled
            if use_cross_bucket_transfer and knowledge_transfer and episode > 0 and episode % config.get("TRANSFER_INTERVAL", 20) == 0:
                _log(f"[INFO] Attempting cross-bucket knowledge transfer (episode {episode})...")
                best_agent_idx_current_eval = population.get_best_agent_idx() # Get best from current eval
                bucket_type = config.get("BUCKET", "Scalping")
                knowledge_transfer.register_agent(bucket_type, agents[best_agent_idx_current_eval])
                
                # Perform knowledge transfer
                transfer_results = knowledge_transfer.transfer_all(episode)
                
                # Log transfer results
                if transfer_results:
                    for result in transfer_results:
                        _log(f"[TRANSFER] {result['message']}")
                else:
                    _log("[TRANSFER] No knowledge transfer performed")
            
            # Evolution step
            if (episode + 1) % config.get("ES_INTERVAL", 5) == 0:
                _log(f"[INFO] Running evolution step at episode {episode+1}...")
                population.evolution_step() # No argument needed now
                agents = population.agents # Update local reference to agents list
            
            # Log performance
            recent_rewards.append(mean_reward) # Log raw reward mean
            recent_fitness.append(mean_fitness) # Log fitness mean
            avg_recent_reward = np.mean(recent_rewards)
            avg_recent_fitness = np.mean([f for f in recent_fitness if not np.isinf(f)]) if any(not np.isinf(f) for f in recent_fitness) else -np.inf

            # Calculate elapsed time
            episode_time = time.time() - episode_start_time
            total_elapsed_time = time.time() - start_time
            
            # Combine metrics from all agents for human-readable logging
            combined_metrics = {}
            for m in agent_metrics:
                for k, v in m.items():
                    if k not in combined_metrics:
                        combined_metrics[k] = []
                    combined_metrics[k].append(v)
            
            # Average metrics
            avg_metrics = {k: np.mean(v) for k, v in combined_metrics.items() if v}
            
            # Log human-readable metrics
            log_human_metrics(avg_metrics, episode + 1, total_elapsed_time)
            
            # Log detailed fitness metrics
            _log(f"[EPISODE {episode + 1}] Fitness: {mean_fitness:.4f} (Â±{std_fitness:.4f}), " +
                 f"Avg Recent Fitness: {avg_recent_fitness:.4f}") # Updated log message
            _log(f"[EPISODE {episode + 1}] Raw Reward: {mean_reward:.2f} (Â±{std_reward:.2f}), " +
                 f"Avg Recent Reward: {avg_recent_reward:.2f}") # Keep raw reward log too
            
            # Track best agent based on fitness
            current_best_fitness_this_episode = population.get_current_best_fitness()
            
            if current_best_fitness_this_episode > best_fitness_score:
                best_fitness_score = current_best_fitness_this_episode
                best_agent_idx = population.get_best_agent_idx() # Get index of the best agent from this episode
                best_agent = agents[best_agent_idx] # Update the best_agent reference
                
                # Save checkpoint
                if save_path:
                    checkpoint_path = os.path.join(save_path, f"checkpoint_{episode + 1}.pth")
                    try:
                        _log(f"[INFO] Saving best model checkpoint (Fitness: {best_fitness_score:.4f}) to {checkpoint_path}...")
                        
                        # Use our recovery-integrated checkpoint function
                        save_success = create_recovery_checkpoint(
                            best_agent.model,
                            best_agent.optimizer,
                            episode + 1,
                            best_fitness_score, # Pass best fitness score
                            config,
                            checkpoint_path,
                            additional_state={
                                "performance_history": performance_history,
                                "recent_fitness": recent_fitness, # Save recent fitness
                                "horizons": horizons,
                                "no_improvement_count": no_improvement_count,
                                "best_agent_idx": best_agent_idx # Store the index too
                            },
                            recovery_system=recovery_system
                        )
                        
                        if not save_success:
                            # Try fallback location if primary save failed
                            fallback_path = os.path.join(os.path.dirname(save_path), 
                                                       "emergency", 
                                                       f"checkpoint_{episode + 1}_emergency.pth")
                            _log(f"[WARNING] Primary checkpoint save failed, trying fallback location: {fallback_path}")
                            
                            # Create directory if needed
                            os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
                            
                            # Try with fallback path
                            create_recovery_checkpoint(
                                best_agent.model,
                                best_agent.optimizer,
                                episode + 1,
                                best_fitness_score,
                                config,
                                fallback_path,
                                additional_state={
                                    "performance_history": performance_history,
                                    "recent_fitness": recent_fitness,
                                    "horizons": horizons,
                                    "no_improvement_count": no_improvement_count,
                                    "best_agent_idx": best_agent_idx,
                                    "is_emergency": True
                                },
                                recovery_system=recovery_system
                            )
                    except Exception as e:
                        error_msg = f"Error saving checkpoint: {str(e)}"
                        _log(f"[ERROR] {error_msg}")
                        if error_handler_available:
                            handle_error(
                                e,
                                "train_model.save_checkpoint",
                                additional_context={
                                    "episode": episode + 1, 
                                    "best_fitness_score": best_fitness_score,
                                    "checkpoint_path": checkpoint_path
                                }
                            )
                
                # Reset no improvement counter
                no_improvement_count = 0
                
                # If cross-bucket knowledge transfer is enabled, save knowledge
                if use_cross_bucket_transfer:
                    # Save knowledge for transfer (in a real implementation)
                    bucket_type = config.get("BUCKET", "Scalping")
                    _log(f"[INFO] Updated knowledge for bucket {bucket_type}")
                else:
                    no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= config.get("EARLY_STOP_PATIENCE", 50):
                _log(f"[INFO] Early stopping: no improvement in best fitness for {config.get('EARLY_STOP_PATIENCE', 50)} episodes")
                break
            
            # Save recovery state periodically
            if save_path and (episode + 1) % config.get("RECOVERY_INTERVAL", 10) == 0:
                recovery_path = os.path.join(os.path.dirname(save_path), "recovery_state.json")
                try:
                    _log(f"[INFO] Saving recovery state to {recovery_path}...")
                    current_best_agent_idx = population.get_best_agent_idx() # Index of best in current eval
                    
                    # Create recovery state
                    state_to_save = {
                         'episode': episode + 1,
                         'best_fitness_score': best_fitness_score, # Save overall best fitness
                         'best_agent_idx': population.best_agent_idx_overall, # Save index of overall best agent
                         'current_episode_best_idx': current_best_agent_idx, # Optionally save current best index
                         'no_improvement_count': no_improvement_count,
                    }
                    
                    # If recovery system is available, register the performance metrics
                    if recovery_system:
                        # Register current performance
                        recovery_system.register_performance_metrics(avg_metrics)
                        
                        # Update recovery system state
                        recovery_system.recovery_state = state_to_save
                    
                    # Save recovery state using existing function or direct write
                    save_recovery_state(
                        episode + 1,
                        best_fitness_score, # Save overall best fitness
                        population.best_agent_idx_overall, # Save overall best agent index
                        recovery_path
                    )
                    
                    _log(f"[INFO] Recovery state saved successfully")
                except Exception as e:
                    error_msg = f"Error saving recovery state: {str(e)}"
                    _log(f"[ERROR] {error_msg}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "train_model.save_recovery_state",
                            additional_context={
                                "episode": episode + 1, 
                                "best_fitness_score": best_fitness_score,
                                "recovery_path": recovery_path
                            }
                        )
            
            # Log performance metrics
            if save_path and (episode + 1) % config.get("PERF_LOG_INTERVAL", 5) == 0:
                # Average metrics across agents for this episode
                ep_avg_metrics = {k: np.mean([m.get(k, 0) for m in agent_metrics]) for k in agent_metrics[0].keys()} if agent_metrics else {}

                # Create performance entry including fitness
                performance_entry = {
                    "episode": episode + 1,
                    "reward": mean_reward, 
                    "fitness": mean_fitness, # Add mean fitness
                    "avg_recent_reward": avg_recent_reward, 
                    "avg_recent_fitness": avg_recent_fitness, # Add avg recent fitness
                    "best_fitness_so_far": best_fitness_score, # Add overall best fitness
                    "metrics": ep_avg_metrics, 
                    "timestamp": time.time()
                }
                performance_history.append(performance_entry)
                
                # Save to file with error handling
                try:
                    _log(f"[INFO] Saving performance metrics to {perf_log_file}...")
                    
                    # Create a temp file to avoid corrupting the original if interrupted
                    temp_perf_log_file = f"{perf_log_file}.tmp"
                    with open(temp_perf_log_file, 'w') as f:
                        json.dump(performance_history, f, indent=2)
                    
                    # If successful, replace the original file
                    if os.path.exists(temp_perf_log_file):
                        if os.path.exists(perf_log_file):
                            # Keep a backup of the previous file
                            backup_file = f"{perf_log_file}.bak"
                            try:
                                if os.path.exists(backup_file):
                                    os.remove(backup_file)
                                os.rename(perf_log_file, backup_file)
                            except:
                                pass
                        
                        os.rename(temp_perf_log_file, perf_log_file)
                        _log(f"[INFO] Performance metrics saved successfully")

                except Exception as e:
                    error_msg = f"Error saving performance metrics: {str(e)}"
                    _log(f"[ERROR] {error_msg}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "train_model.save_performance_metrics",
                            additional_context={
                                "episode": episode + 1, 
                                "metrics_file": perf_log_file
                            }
                        )
                    
                    # Try to save in an alternative location
                    try:
                        alt_metrics_path = os.path.join(os.path.dirname(save_path), "metrics_backup", f"performance_{episode + 1}.json")
                        os.makedirs(os.path.dirname(alt_metrics_path), exist_ok=True)
                        _log(f"[WARNING] Attempting to save metrics to alternative location: {alt_metrics_path}")
                        with open(alt_metrics_path, 'w') as f:
                            json.dump(performance_history, f, indent=2)
                        _log(f"[INFO] Performance metrics saved to alternative location")
                    except Exception as alt_e:
                        _log(f"[ERROR] Failed to save metrics to alternative location: {str(alt_e)}")
            
            # Increment episode counter
            episode += 1
            
            # Memory cleanup
            optimize_memory()
            
            # Check for interruption
            if _interrupted:
                _log("[WARNING] Training interrupted by user signal. Finishing current operations and saving...")
                
                # Save emergency checkpoint using the *overall* best agent found so far
                if population and hasattr(population, 'best_agent_idx_overall'):
                    overall_best_agent_idx = population.best_agent_idx_overall
                    if 0 <= overall_best_agent_idx < len(agents):
                        emergency_agent = agents[overall_best_agent_idx]
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        emergency_path = os.path.join(save_path, f"emergency_checkpoint_{episode}.pth")
                        last_interrupted_path = os.path.join(save_path, "last_interrupted.pth")
                        try:
                            _log(f"[EMERGENCY] Saving state for overall best agent (Index: {overall_best_agent_idx}, Fitness: {best_fitness_score:.4f})...")
                            # Save checkpoint for the overall best agent
                            save_success = save_checkpoint(
                                emergency_agent.model, emergency_agent.optimizer, episode, best_fitness_score, config, emergency_path,
                                timestamp=timestamp, is_emergency=True, performance_history=performance_history,
                                recent_fitness=recent_fitness, horizons=horizons, # Save recent fitness
                                additional_state={"error_recovery": True, "no_improvement_count": no_improvement_count, "best_agent_idx": overall_best_agent_idx}
                            )
                            if save_success: _log(f"[EMERGENCY] Saved checkpoint to {emergency_path}")
                            else: _log(f"[ERROR] Failed to save emergency checkpoint.")
                            
                            # Also save as last_interrupted
                            save_checkpoint(
                               emergency_agent.model, emergency_agent.optimizer, episode, best_fitness_score, config, last_interrupted_path,
                               timestamp=timestamp, is_emergency=True, performance_history=performance_history,
                               recent_fitness=recent_fitness, horizons=horizons,
                               additional_state={"error_recovery": True, "no_improvement_count": no_improvement_count, "best_agent_idx": overall_best_agent_idx}
                            )
                            _log(f"[EMERGENCY] Also saved as last_interrupted.pth")
                        except Exception as e:
                            error_msg = f"Error saving emergency checkpoint: {str(e)}"
                            _log(f"[ERROR] {error_msg}")
                            if error_handler_available: handle_error(e, "train_model.save_emergency_checkpoint", ...)
                    else:
                         _log("[ERROR] Could not save emergency checkpoint: Overall best agent index invalid.")
                else:
                     _log("[ERROR] Could not save emergency checkpoint: Population or best agent tracking missing.")

                _log(f"[WARNING] Training stopped at episode {episode} due to interruption.")
                # Can't use break here since we're not in a loop
    
    except KeyboardInterrupt:
        _log("[INFO] Training interrupted by KeyboardInterrupt")
        _interrupted = True # Set flag in case it wasn't set by signal handler
    except Exception as e:
        _log(f"[ERROR] An unexpected error occurred during training: {e}")
        import traceback
        _log(traceback.format_exc())
        
        # Save emergency checkpoint
        if population and hasattr(population, 'best_agent_idx_overall'):
            overall_best_agent_idx = population.best_agent_idx_overall
            if 0 <= overall_best_agent_idx < len(agents):
                emergency_agent = agents[overall_best_agent_idx]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                emergency_path = os.path.join(save_path, f"emergency_checkpoint_{episode}.pth")
                last_interrupted_path = os.path.join(save_path, "last_interrupted.pth")
                try:
                    _log(f"[EMERGENCY] Saving state for overall best agent (Index: {overall_best_agent_idx}, Fitness: {best_fitness_score:.4f})...")
                    # Save checkpoint for the overall best agent
                    save_success = save_checkpoint(
                        emergency_agent.model, emergency_agent.optimizer, episode, best_fitness_score, config, emergency_path,
                        timestamp=timestamp, is_emergency=True, performance_history=performance_history,
                        recent_fitness=recent_fitness, horizons=horizons, # Save recent fitness
                        additional_state={"error_recovery": True, "no_improvement_count": no_improvement_count, "best_agent_idx": overall_best_agent_idx}
                    )
                    if save_success: _log(f"[EMERGENCY] Saved checkpoint to {emergency_path}")
                    else: _log(f"[ERROR] Failed to save emergency checkpoint.")
                    
                    # Also save as last_interrupted
                    save_checkpoint(
                       emergency_agent.model, emergency_agent.optimizer, episode, best_fitness_score, config, last_interrupted_path,
                       timestamp=timestamp, is_emergency=True, performance_history=performance_history,
                       recent_fitness=recent_fitness, horizons=horizons,
                       additional_state={"error_recovery": True, "no_improvement_count": no_improvement_count, "best_agent_idx": overall_best_agent_idx}
                    )
                    _log(f"[EMERGENCY] Also saved as last_interrupted.pth")
                except Exception as e:
                    error_msg = f"Error saving emergency checkpoint: {str(e)}"
                    _log(f"[ERROR] {error_msg}")
                    if error_handler_available: handle_error(e, "train_model.save_emergency_checkpoint", ...)
            else:
                 _log("[ERROR] Could not save emergency checkpoint: Overall best agent index invalid.")
        else:
             _log("[ERROR] Could not save emergency checkpoint: Population or best agent tracking missing.")

        _log(f"[WARNING] Training stopped at episode {episode} due to interruption.")
        # Can't use break here since we're not in a loop
    
    finally:
        # Final cleanup regardless of how loop exited
        _log("[INFO] Cleaning up environments...")
        try:
            if 'envs' in locals() and envs is not None:
                envs.close()
                _log("[INFO] Environments closed.")
            # Explicit memory cleanup
            optimize_memory()
        except Exception as cleanup_e:
            _log(f"[WARNING] Error during environment cleanup: {cleanup_e}")
    
    _log("[INFO] Training loop finished.")

    # Return the overall best agent found
    final_best_agent = None
    final_best_optimizer = None
    if 'population' in locals() and population and hasattr(population, 'best_agent_idx_overall'):
         final_best_idx = population.best_agent_idx_overall
         if 0 <= final_best_idx < len(agents):
             final_best_agent = agents[final_best_idx]
             final_best_optimizer = final_best_agent.optimizer
             _log(f"[INFO] Returning overall best agent (Index: {final_best_idx}) with fitness {best_fitness_score:.4f}")
         else:
              _log("[WARNING] Could not determine final best agent: Index out of bounds.")
    else:
         _log("[WARNING] Could not determine final best agent: Population tracking incomplete.")

    # ===== FINAL PREDICTIVE AGENT SAVE =====
    # Save the final state of the predictive agent
    if 'predictive_agent' in locals() and predictive_agent is not None:
        try:
            _log(f"[PREDICTIVE] Saving final predictive agent for {bucket_type} bucket...")
            
            # Save final predictive agent checkpoint
            final_predictive_checkpoint = {
                'model_state_dict': predictive_agent.model.state_dict(),
                'optimizer_state_dict': predictive_agent.optimizer.state_dict(),
                'scheduler_state_dict': predictive_agent.scheduler.state_dict() if hasattr(predictive_agent, 'scheduler') else None,
                'episode': episode,
                'config': predictive_config,
                'bucket_type': bucket_type,
                'training_completed': True,
                'final_fitness_score': best_fitness_score
            }
            
            # Save to the predictive agent directory
            if 'predictive_save_path' in locals():
                torch.save(final_predictive_checkpoint, predictive_save_path)
                _log(f"[PREDICTIVE] Final predictive agent saved to {predictive_save_path}")
            
            # Also save a final predictions summary
            if 'predictive_agent_dir' in locals():
                final_summary_file = os.path.join(predictive_agent_dir, f"{bucket_type.lower()}_final_summary.json")
                summary_data = {
                    "bucket_type": bucket_type,
                    "training_completed": True,
                    "total_episodes": episode,
                    "final_fitness_score": best_fitness_score,
                    "predictive_agent_path": predictive_save_path if 'predictive_save_path' in locals() else None,
                    "timestamp": time.time(),
                    "status": "ready_for_prediction"
                }
                
                try:
                    with open(final_summary_file, 'w') as f:
                        json.dump(summary_data, f, indent=2)
                    _log(f"[PREDICTIVE] Final summary saved to {final_summary_file}")
                except Exception as summary_e:
                    _log(f"[WARNING] Could not save final summary: {str(summary_e)}")
            
        except Exception as final_save_e:
            _log(f"[WARNING] Error saving final predictive agent: {str(final_save_e)}")
            if error_handler_available:
                handle_error(
                    final_save_e,
                    "train_model.final_predictive_agent_save",
                    additional_context={"bucket_type": bucket_type}
                )
    
    # ===== END FINAL PREDICTIVE AGENT SAVE =====

    _log("[INFO] Returning final results.")
    return final_best_agent.model if final_best_agent else None, final_best_optimizer, episode, best_fitness_score # Return best fitness


if __name__ == "__main__":
    import argparse
    import logging
    
    # Set up proper logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("training")
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Training script for BTC AI Trading Agent')
    
    # Basic training parameters
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--price_column', type=str, help='Name of price column')
    parser.add_argument('--seq_len', type=int, help='Sequence length for time series')
    parser.add_argument('--test_size', type=float, help='Test size for splitting data')
    parser.add_argument('--models_dir', type=str, help='Directory to save models')
    parser.add_argument('--bucket', type=str, choices=['Scalping', 'Short', 'Medium', 'Long'], 
                       help='Trading timeframe bucket')
    parser.add_argument('--episodes', type=int, help='Number of episodes to train')
    parser.add_argument('--initial_capital', type=float, help='Initial capital for trading')
    parser.add_argument('--max_positions', type=int, help='Maximum number of positions')
    parser.add_argument('--commission', type=float, help='Trading commission rate')
    
    # Advanced training parameters
    parser.add_argument('--hidden_size', type=int, help='Hidden size for neural networks')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--entropy_weight', type=float, help='Entropy weight for exploration')
    parser.add_argument('--es_population', type=int, help='Population size for evolution strategy')
    parser.add_argument('--env_count', type=int, help='Number of environments')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], 
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--model_type', type=str, 
                       choices=['lstm', 'gru', 'transformer', 'cnn', 'hybrid'], 
                       help='Model architecture to use')
    
    # Recovery and robustness parameters
    parser.add_argument('--recovery_path', type=str, help='Path to recovery state file')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum number of recovery attempts')
    parser.add_argument('--checkpoint_interval', type=int, help='Episodes between checkpoints')
    parser.add_argument('--enable_emergency_checkpoints', action='store_true', 
                       help='Enable emergency checkpoints on errors')
    
    args = parser.parse_args()
    
    try:
        # Import pandas for reading data
        import pandas as pd
        
        # Import the training recovery system
        try:
            from src.training.training_recovery import TrainingRecoverySystem
            recovery_available = True
            logger.info("Training recovery system available")
        except ImportError:
            recovery_available = False
            logger.warning("Training recovery system not available - proceeding without recovery capabilities")
        
        # Load data file if provided
        df = None
        if args.data_path:
            df = pd.read_csv(args.data_path)
            logger.info(f"Loaded data from {args.data_path}: {len(df)} rows")
        else:
            logger.warning("No data path provided - will attempt to use default data")
        
        # Create config dictionary from arguments
        config = {k: v for k, v in vars(args).items() if v is not None}
        
        # Convert parameter names to uppercase to match BTC AI convention
        config = {k.upper(): v for k, v in config.items()}
        
        # Determine save path
        if args.models_dir:
            save_path = args.models_dir
        else:
            # Use default path based on bucket
            bucket = args.bucket or "Scalping"
            save_path = os.path.join(MODELS_DIR_DEFAULT, bucket, "checkpoints")
        
        # Create recovery state from file if provided
        recovery_state = None
        if args.recovery_path and os.path.exists(args.recovery_path):
            try:
                with open(args.recovery_path, 'r') as f:
                    recovery_state = json.load(f)
                logger.info(f"Loaded recovery state from {args.recovery_path}")
            except Exception as e:
                logger.error(f"Error loading recovery state: {e}")
        
        # Use training recovery system if available
        if recovery_available:
            # Initialize recovery system
            recovery_system = TrainingRecoverySystem(
                checkpoint_dir=save_path,
                max_retries=config.get("MAX_RETRIES", 3),
                min_checkpoint_interval=config.get("CHECKPOINT_INTERVAL", 5),
                enable_emergency_checkpoints=config.get("ENABLE_EMERGENCY_CHECKPOINTS", True)
            )
            
            # Start training with recovery
            logger.info("Starting training with recovery capabilities")
            model, metrics, best_reward, elapsed_time = recovery_system.start_training_with_recovery(
                training_func=train_model,
                config=config,
                df=df,
                save_path=save_path,
                recovery_state=recovery_state
            )
            
            # Report final results
            if model:
                logger.info(f"Training completed successfully in {elapsed_time:.2f} seconds")
                logger.info(f"Best reward: {best_reward}")
            else:
                logger.error(f"Training failed after {elapsed_time:.2f} seconds")
                logger.error(f"Error: {metrics.get('error', 'Unknown error')}")
                sys.exit(1)
        else:
            # Fall back to direct training without recovery
            logger.info("Starting training without recovery capabilities")
            result = train_model(df, config, save_path, recovery_state)
            
            if not result or not result[0]:
                logger.error("Training failed")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
