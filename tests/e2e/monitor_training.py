#!/usr/bin/env python
"""
Simplified Monitor Training Script for Testing

This is a test version of the monitor_training.py script that simply loads the
visualizer and generates some plots for end-to-end testing purposes.
"""

import os
import sys
import argparse
import importlib
import time
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

try:
    visualizer_module = importlib.import_module("src.utils.training_visualizer")
    ProgressiveTrainingVisualizer = visualizer_module.ProgressiveTrainingVisualizer
    print(f"Successfully imported ProgressiveTrainingVisualizer from {project_root}")
except ImportError as e:
    print(f"Error importing ProgressiveTrainingVisualizer: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Monitor progressive training")
    parser.add_argument("--models-dir", type=str, default=None, help="Directory containing model files")
    args = parser.parse_args()
    
    # Use provided models directory or default
    models_dir = args.models_dir
    if not models_dir:
        models_dir = os.path.join(project_root, "Models", "test_progressive")
    
    print(f"Monitoring training in: {models_dir}")
    
    # Create the visualizer
    try:
        visualizer = ProgressiveTrainingVisualizer(output_dir=os.path.join(models_dir, "monitoring"))
        print("Visualizer created successfully")
    except Exception as e:
        print(f"Error creating visualizer: {e}")
        return 1
    
    # Mock monitoring loop
    print("Starting mock monitoring loop")
    for i in range(5):
        print(f"Monitoring iteration {i+1}/5")
        time.sleep(1)
    
    print("Monitoring complete")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 