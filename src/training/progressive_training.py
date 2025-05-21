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
from typing import Optional, Dict

# Forecast helper
try:
    from src.training.forecast_helper import ForecastHelperManager

    forecast_helper_available = True
except Exception:
    forecast_helper_available = False
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch

# Add the current directory to sys.path to ensure module imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import modules dynamically
try:
    # Import utils module
    utils_module = importlib.import_module("src.utils.utils")
    log = utils_module.log
    measure_gpu_usage = utils_module.measure_gpu_usage
    optimize_memory = utils_module.optimize_memory

    # Import training module
    training_module = importlib.import_module("src.training.training")
    train_model = training_module.train_model

    # Import config module
    config_module = importlib.import_module("src.utils.config")
    get_config = config_module.get_config

except ImportError as e:
    print(f"Error importing modules in progressive_training.py: {e}")

    # Define fallback functions
    def log(message, level="info"):
        print(f"[{level.upper()}] {message}")

    def measure_gpu_usage():
        return 0.0

    def optimize_memory():
        import gc

        gc.collect()

    def train_model(*args, **kwargs):
        print("Error: train_model function not available")
        return None

    def get_config():
        return {}


# Define module-level logger
logger = logging.getLogger("progressive_training")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Directory constants
MODELS_DIR_DEFAULT = os.path.join(os.path.dirname(current_dir), "Models")
DATA_DIR = os.path.join(os.path.dirname(current_dir), "Data")
CONFIG_FILE = os.path.join(current_dir, "config.json")


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

        # Forecast helpers per bucket
        if forecast_helper_available:
            self.forecast_manager = ForecastHelperManager()
        else:
            self.forecast_manager = None

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
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                config = json.load(f)
            return config
        else:
            logger.warning(
                f"Config file {self.config_path} not found. Using default config."
            )
            return {}

    def _initialize_knowledge_transfer(self):
        """Initialize the knowledge transfer module."""
        try:
            # Import the CrossBucketKnowledgeTransfer class
            agent_module = importlib.import_module("src.agent.agent")
            self.CrossBucketKnowledgeTransfer = (
                agent_module.CrossBucketKnowledgeTransfer
            )

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

    def _get_bucket_config(
        self, bucket_type: str, ui_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a bucket-specific configuration.

        Args:
            bucket_type: Bucket type to configure
            ui_params: Optional dictionary of parameters supplied by the UI. The
                following keys are recognized:
                - ``horizon_range``: ``(min, max)`` tuple of prediction horizons
                - ``frequency``: timeframe or data frequency string
                - ``capital_allocation``: float representing portion of capital

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

        # Apply UI-provided overrides
        if ui_params:
            horizon_range = ui_params.get("horizon_range")
            if (
                horizon_range
                and isinstance(horizon_range, (list, tuple))
                and len(horizon_range) == 2
            ):
                bucket_config["MIN_HORIZON"], bucket_config["MAX_HORIZON"] = (
                    horizon_range
                )
            if "frequency" in ui_params:
                bucket_config["FREQUENCY"] = ui_params["frequency"]
            if "capital_allocation" in ui_params:
                bucket_config["CAPITAL_ALLOCATION"] = ui_params["capital_allocation"]

        # Create knowledge transfer directory for storing transferable insights
        bucket_config["KNOWLEDGE_TRANSFER_DIR"] = os.path.join(
            self.models_dir, "knowledge_transfer"
        )
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

        logger.info(
            f"Freed memory and resources. Current GPU usage: {measure_gpu_usage()*100:.1f}%"
        )

    def train_bucket(
        self,
        bucket_type: str,
        episodes: int = None,
        save_path: str = None,
        transfer_from: str = None,
        resume: bool = False,
        ui_params: Optional[Dict] = None,
    ) -> str:
        """
        Train a specific bucket model.

        Args:
            bucket_type: Bucket type to train
            episodes: Number of episodes to train for (if None, use config default)
            save_path: Directory to save model (if None, use default bucket path)
            transfer_from: Bucket to transfer knowledge from
            resume: Whether to resume training from a checkpoint
            ui_params: Optional parameters from the UI used during bucket
                initialization. Supported keys are ``horizon_range``,
                ``frequency`` and ``capital_allocation``.

        Returns:
            Path to the trained model
        """
        self.current_bucket = bucket_type

        # Set up save path
        if save_path is None:
            save_path = os.path.join(self.models_dir, bucket_type, "checkpoints")
        os.makedirs(save_path, exist_ok=True)

        # Get bucket-specific config including any UI overrides
        bucket_config = self._get_bucket_config(bucket_type, ui_params)

        # Set episodes if specified
        if episodes is not None:
            bucket_config["MAX_EPISODES"] = episodes

        # Log training start
        logger.info(
            f"Starting {bucket_type} bucket training for {bucket_config.get('MAX_EPISODES', 100)} episodes"
        )
        if self.progress_callback:
            self.progress_callback(f"Starting {bucket_type} bucket training")

        # Load recovery state if resuming
        recovery_state = None
        if resume:
            recovery_path = os.path.join(
                os.path.dirname(save_path), "recovery_state.json"
            )
            if os.path.exists(recovery_path):
                try:
                    with open(recovery_path, "r") as f:
                        recovery_state = json.load(f)
                    logger.info(
                        f"Resuming training from episode {recovery_state.get('current_episode', 0)}"
                    )
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

        # Initialize or retrieve forecast helper
        if self.forecast_manager is not None:
            numeric_cols = [c for c in df.columns if df[c].dtype != object]
            feature_size = len(numeric_cols)
            helper = self.forecast_manager.get_helper(
                bucket_type, feature_size, bucket_config
            )
            try:
                sample = torch.tensor(
                    df[numeric_cols].iloc[[0]].values, dtype=torch.float32
                )
                forecast = helper.get_forecast(sample)
                logger.info(f"Initial forecast for {bucket_type}: {forecast}")
            except Exception as e:
                logger.warning(f"Forecast helper failed: {e}")
        else:
            helper = None

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
                progress_callback=nested_progress_callback,
            )

            # Update training history
            self.training_history[bucket_type] = {
                "episodes_completed": episodes_completed,
                "best_reward": best_reward,
                "timestamp": time.time(),
            }

            # Save checkpoint path
            final_path = os.path.join(save_path, f"final_{bucket_type.lower()}.pth")
            if model is not None:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": (
                            optimizer.state_dict() if optimizer else None
                        ),
                        "episodes": episodes_completed,
                        "reward": best_reward,
                        "config": bucket_config,
                    },
                    final_path,
                )
                logger.info(f"Saved final model to {final_path}")

            # Free memory
            self._free_memory_and_resources()

            return final_path

        except Exception as e:
            logger.error(f"Error training {bucket_type} bucket: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def train_progressively(
        self,
        custom_sequence: List[str] = None,
        initial_bucket: str = None,
        episodes_per_bucket: Dict[str, int] = None,
        ui_params: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """
        Train buckets progressively, transferring knowledge between them.

        Args:
            custom_sequence: Custom sequence of buckets to train (default uses standard sequence)
            initial_bucket: Bucket to start with (if None, start with first in sequence)
            episodes_per_bucket: Dictionary mapping buckets to episode counts
            ui_params: Optional parameters forwarded to ``train_bucket`` for
                each bucket. Supported keys are ``horizon_range``, ``frequency``
                and ``capital_allocation``.

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
                logger.warning(
                    f"Initial bucket {initial_bucket} not in sequence. Starting from beginning."
                )

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

            logger.info(
                f"Progressive training: {i+1}/{len(bucket_sequence)} - {bucket}"
            )
            if self.progress_callback:
                self.progress_callback(
                    f"Progressive training: {i+1}/{len(bucket_sequence)} - {bucket}"
                )

            # Train with knowledge transfer from previous bucket (if any)
            model_path = self.train_bucket(
                bucket,
                episodes=episodes,
                transfer_from=prev_bucket,
                ui_params=ui_params,
            )

            # Store model path
            if model_path:
                model_paths[bucket] = model_path

            # Update previous bucket for next iteration
            prev_bucket = bucket

        # Log completion
        logger.info(
            f"Progressive training complete. Trained {len(model_paths)}/{len(bucket_sequence)} buckets."
        )

        return model_paths


def main():
    """Command line interface for the progressive trainer."""
    parser = argparse.ArgumentParser(description="Progressive Trading Bucket Training")
    parser.add_argument(
        "--config", type=str, default=CONFIG_FILE, help="Path to config file"
    )
    parser.add_argument(
        "--bucket", type=str, help="Single bucket to train (skip progressive training)"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Comma-separated bucket sequence (e.g., 'Scalping,Short,Medium,Long')",
    )
    parser.add_argument(
        "--episodes", type=int, help="Number of episodes (for single bucket mode)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint if available",
    )
    parser.add_argument(
        "--transfer",
        type=str,
        help="Bucket to transfer knowledge from (for single bucket mode)",
    )

    args = parser.parse_args()

    # Initialize trainer
    def print_progress(msg):
        print(f"[PROGRESS] {msg}")

    trainer = ProgressiveTrainer(
        config_path=args.config, progress_callback=print_progress
    )

    # Training mode
    if args.bucket:
        # Single bucket mode
        print(f"Training single bucket: {args.bucket}")
        model_path = trainer.train_bucket(
            args.bucket,
            episodes=args.episodes,
            transfer_from=args.transfer,
            resume=args.resume,
            ui_params=None,
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
        model_paths = trainer.train_progressively(
            custom_sequence=sequence, ui_params=None
        )

        print("\nProgressive training complete. Results:")
        for bucket, path in model_paths.items():
            print(f"  {bucket}: {path}")


if __name__ == "__main__":
    main()
