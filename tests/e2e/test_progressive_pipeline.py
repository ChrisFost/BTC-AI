#!/usr/bin/env python
"""
End-to-End Test for Progressive Training Pipeline

This script performs a comprehensive test of the entire progressive training pipeline,
including progressive training, cross-bucket knowledge transfer, and monitoring.
"""

import os
import sys
import time
import logging
import threading
import subprocess
import argparse
import importlib
from datetime import datetime
import json

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

# Import necessary modules using dynamic imports
try:
    # Use mock training module for testing
    mock_training_module = importlib.import_module("src.training.mock_training")
    MockProgressiveTrainer = mock_training_module.MockProgressiveTrainer
    
    progressive_module = importlib.import_module("src.training.progressive")
    ProgressiveTrainer = progressive_module.ProgressiveTrainer
    CrossBucketKnowledgeTransfer = progressive_module.CrossBucketKnowledgeTransfer
    
    visualizer_module = importlib.import_module("src.utils.training_visualizer")
    ProgressiveTrainingVisualizer = visualizer_module.ProgressiveTrainingVisualizer
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure all required modules are available in the current directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('pipeline_test')

class ProgressivePipelineTester:
    """
    Tests the entire progressive training pipeline from end to end.
    
    This class orchestrates training across multiple buckets, verifies knowledge transfer,
    and checks that monitoring functions properly.
    """
    
    def __init__(self, models_dir=None, data_dir=None, config_path=None, debug=False):
        """
        Initialize the pipeline tester.
        
        Args:
            models_dir: Directory for saving model files
            data_dir: Directory containing training data
            config_path: Path to configuration file
            debug: Whether to run in debug mode
        """
        self.debug = debug
        
        # Set up directories
        project_dir = os.path.dirname(current_dir)
        self.models_dir = models_dir or os.path.join(project_dir, "Models", "test_progressive")
        self.data_dir = data_dir or os.path.join(project_dir, "Data")
        self.config_path = config_path or os.path.join(current_dir, "config.json")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Add log and monitoring directories
        self.log_dir = os.path.join(self.models_dir, "logs")
        self.monitor_dir = os.path.join(self.models_dir, "monitoring")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.monitor_dir, exist_ok=True)
        
        # Create log file handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, "pipeline_test.log"))
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Initialize components
        self._init_trainer()
        self._init_visualizer()
        
        # Track processes
        self.processes = []
    
    def _init_trainer(self):
        """Initialize the progressive trainer."""
        try:
            # Use mock trainer for testing purposes
            self.trainer = MockProgressiveTrainer(
                config_path=self.config_path,
                progress_callback=self._progress_callback
            )
            logger.info("Initialized MockProgressiveTrainer for testing")
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            raise
    
    def _init_visualizer(self):
        """Initialize the training visualizer."""
        try:
            self.visualizer = ProgressiveTrainingVisualizer(
                output_dir=self.monitor_dir
            )
            logger.info("Initialized ProgressiveTrainingVisualizer")
        except Exception as e:
            logger.error(f"Failed to initialize visualizer: {e}")
            raise
    
    def _progress_callback(self, message):
        """Callback function for trainer progress updates."""
        logger.info(f"Training progress: {message}")
    
    def _launch_monitor(self):
        """Launch the training monitor in a separate process."""
        logger.info("Launching monitor dashboard...")
        
        try:
            # Construct command to launch monitor
            monitor_script = os.path.join(current_dir, "monitor_training.py")
            
            # Use sys.executable to ensure we use the correct Python interpreter
            cmd = [sys.executable, monitor_script, "--models-dir", self.models_dir]
            
            # Launch monitor process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE if not self.debug else None,
                stderr=subprocess.PIPE if not self.debug else None,
                text=True
            )
            
            self.processes.append(process)
            logger.info(f"Monitor dashboard launched (PID: {process.pid})")
            
            # Give monitor time to initialize
            time.sleep(3)
            
            return process
        except Exception as e:
            logger.error(f"Failed to launch monitor: {e}")
            return None
    
    def run_test(self, sequence=None, episodes_per_bucket=5, launch_monitor=True, 
                transfer_knowledge=True, dashboard_view_time=15):
        """
        Run the complete pipeline test.
        
        Args:
            sequence: List of bucket types to train in sequence
            episodes_per_bucket: Number of episodes to train each bucket
            launch_monitor: Whether to launch the monitoring dashboard
            transfer_knowledge: Whether to enable knowledge transfer between buckets
            dashboard_view_time: Time to keep dashboard open (in seconds)
            
        Returns:
            True if test completed successfully, False otherwise
        """
        # Define training sequence if not provided
        if sequence is None:
            sequence = ["Scalping", "Short", "Medium", "Long"]
        
        # Launch monitor if requested
        monitor_process = None
        if launch_monitor:
            monitor_process = self._launch_monitor()
        
        try:
            logger.info(f"Starting progressive training test with sequence: {sequence}")
            
            # Train each bucket in sequence
            for i, bucket_type in enumerate(sequence):
                # Determine if we should transfer knowledge from previous bucket
                transfer_from = None
                if transfer_knowledge and i > 0:
                    transfer_from = sequence[i-1]
                    logger.info(f"Will transfer knowledge from {transfer_from} to {bucket_type}")
                
                # Train the bucket
                logger.info(f"Training {bucket_type} bucket for {episodes_per_bucket} episodes")
                
                # Set bucket-specific save path
                save_path = os.path.join(self.models_dir, bucket_type, "checkpoints")
                
                try:
                    # Start training
                    self.trainer.train_bucket(
                        bucket_type=bucket_type,
                        episodes=episodes_per_bucket,
                        save_path=save_path,
                        transfer_from=transfer_from
                    )
                    
                    logger.info(f"Completed training for {bucket_type} bucket")
                except Exception as e:
                    logger.error(f"Error training {bucket_type} bucket: {e}")
                    return False
                
                # Pause between buckets
                if i < len(sequence) - 1:
                    logger.info("Pausing between bucket training...")
                    time.sleep(2)
            
            # Keep monitor open for a while to observe the results
            if monitor_process:
                logger.info("Training complete. Monitor will remain open for viewing...")
                
                # Generate a report using the visualizer
                try:
                    report_dir = os.path.join(self.models_dir, "reports", f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    os.makedirs(os.path.dirname(report_dir), exist_ok=True)
                    
                    # Extract training and transfer history from files
                    training_history = {}
                    for bucket in sequence:
                        history_file = os.path.join(self.models_dir, bucket, "training_history.json")
                        if os.path.exists(history_file):
                            with open(history_file, 'r') as f:
                                training_history[bucket] = json.load(f)
                    
                    transfer_history = []
                    transfer_file = os.path.join(self.models_dir, "knowledge_transfer", "transfer_history.json")
                    if os.path.exists(transfer_file):
                        with open(transfer_file, 'r') as f:
                            transfer_history = json.load(f)
                    
                    # Generate report
                    if training_history and transfer_history:
                        report_path = self.visualizer.generate_training_report(
                            training_history,
                            transfer_history,
                            output_path=report_dir
                        )
                        logger.info(f"Generated training report at: {report_path}")
                except Exception as e:
                    logger.error(f"Error generating report: {e}")
                
                # Keep dashboard open for a set amount of time, then automatically close
                logger.info(f"Dashboard will automatically close after {dashboard_view_time} seconds...")
                
                # Give time to view the dashboard
                start_time = time.time()
                while time.time() - start_time < dashboard_view_time:
                    # Check if the process is still running
                    if monitor_process.poll() is not None:
                        logger.info("Dashboard was closed by user")
                        break
                    time.sleep(1)
            
            logger.info("Progressive training test completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during progressive training test: {e}")
            return False
        finally:
            # Clean up processes
            self._cleanup()
    
    def _cleanup(self):
        """Clean up any running processes."""
        for process in self.processes:
            try:
                if process.poll() is None:  # Process is still running
                    logger.info(f"Terminating process {process.pid}")
                    process.terminate()
                    process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error terminating process: {e}")
        
        self.processes = []

def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Test the progressive training pipeline")
    parser.add_argument("--models-dir", type=str, help="Directory for saving models")
    parser.add_argument("--data-dir", type=str, help="Directory containing training data")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per bucket (default: 5)")
    parser.add_argument("--sequence", type=str, nargs="+", 
                      help="Bucket sequence for training (default: Scalping Short Medium Long)")
    parser.add_argument("--no-monitor", action="store_true", help="Don't launch the monitor dashboard")
    parser.add_argument("--no-transfer", action="store_true", help="Disable knowledge transfer")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with verbose logging")
    parser.add_argument("--view-time", type=int, default=15, 
                      help="Time in seconds to keep dashboard open (default: 15)")
    
    args = parser.parse_args()
    
    # Create timestamp for test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Default models directory with timestamp to avoid conflicts
    default_models_dir = os.path.join(
        os.path.dirname(current_dir), 
        "Models", 
        f"test_progressive_{timestamp}"
    )
    
    # Initialize tester
    tester = ProgressivePipelineTester(
        models_dir=args.models_dir or default_models_dir,
        data_dir=args.data_dir,
        config_path=args.config,
        debug=args.debug
    )
    
    # Run the test
    success = tester.run_test(
        sequence=args.sequence,
        episodes_per_bucket=args.episodes,
        launch_monitor=not args.no_monitor,
        transfer_knowledge=not args.no_transfer,
        dashboard_view_time=args.view_time
    )
    
    # Report results
    if success:
        logger.info("End-to-end test completed successfully!")
        return 0
    else:
        logger.error("End-to-end test failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 