#!/usr/bin/env python
"""
Test Script for Progressive Training Visualizer

This script demonstrates the functionality of the ProgressiveTrainingVisualizer
by generating example visualizations with sample data.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import importlib

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the visualizer using dynamic import
utils_module = importlib.import_module("src.utils.training_visualizer") 
ProgressiveTrainingVisualizer = utils_module.ProgressiveTrainingVisualizer

def generate_sample_training_history():
    """Generate sample training history data for testing."""
    # Define buckets
    buckets = ["Scalping", "Short", "Medium", "Long"]
    
    # Initialize training history
    training_history = {}
    
    # Generate data for each bucket
    for i, bucket in enumerate(buckets):
        # Simulate later start for each subsequent bucket
        offset = i * 25
        episodes = 100 - i * 15  # Fewer episodes for later buckets
        
        history = []
        for ep in range(episodes):
            # Create simulated metrics that improve over time
            reward = np.sin(ep/10) * 2 + ep/10 + np.random.normal(0, 0.5)
            loss = max(0.1, 2.0 * np.exp(-ep/30) + np.random.normal(0, 0.1))
            win_rate = min(0.95, 0.4 + ep/200 + np.random.normal(0, 0.03))
            profit_factor = min(3.0, 1.0 + ep/50 + np.random.normal(0, 0.2))
            
            # Add entry to history
            history.append({
                'episode': ep + offset,
                'reward': reward,
                'loss': loss,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            })
        
        training_history[bucket] = history
    
    return training_history

def generate_sample_transfer_history(training_history):
    """Generate sample knowledge transfer events."""
    transfer_history = []
    buckets = list(training_history.keys())
    
    # Add sequential transfers from earlier buckets to later ones
    for i in range(len(buckets) - 1):
        source = buckets[i]
        target = buckets[i + 1]
        
        # Find episode numbers from the source bucket history
        if training_history[source]:
            # Initial transfer after 20 episodes
            episode = training_history[source][20]['episode']
            
            # Add transfer event
            transfer_history.append({
                'episode': episode,
                'source': source,
                'target': target,
                'transfer_types': ['weights', 'features'],
                'success': True,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'duration': 15.3,
                'metrics_before': {
                    'reward': 3.2,
                    'win_rate': 0.45,
                    'profit_factor': 1.2
                },
                'metrics_after': {
                    'reward': 5.1,
                    'win_rate': 0.52,
                    'profit_factor': 1.5
                }
            })
            
            # Add another transfer later
            if len(training_history[source]) > 50:
                episode = training_history[source][50]['episode']
                
                # Add transfer event
                transfer_history.append({
                    'episode': episode,
                    'source': source,
                    'target': target,
                    'transfer_types': ['horizons'],
                    'success': False,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'duration': 8.7
                })
    
    # Add a reverse transfer from longer time frame to shorter
    if len(buckets) > 2:
        source = buckets[2]  # Medium
        target = buckets[1]  # Short
        
        if training_history[source] and len(training_history[source]) > 30:
            episode = training_history[source][30]['episode']
            
            # Add transfer event
            transfer_history.append({
                'episode': episode,
                'source': source,
                'target': target,
                'transfer_types': ['features', 'horizons'],
                'success': True,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'duration': 12.8,
                'metrics_before': {
                    'reward': 6.7,
                    'win_rate': 0.58,
                    'profit_factor': 1.7
                },
                'metrics_after': {
                    'reward': 7.2,
                    'win_rate': 0.62,
                    'profit_factor': 1.9
                }
            })
    
    return transfer_history

def generate_sample_memory_usage(training_history, transfer_history):
    """Generate sample memory usage data."""
    # Find all unique episode numbers
    all_episodes = []
    for bucket, history in training_history.items():
        for entry in history:
            all_episodes.append(entry['episode'])
    
    transfer_episodes = [entry['episode'] for entry in transfer_history]
    
    # Sort episodes
    all_episodes = sorted(list(set(all_episodes)))
    
    # Generate memory usage for each episode
    memory_usage = {
        'episodes': all_episodes,
        'usage': []
    }
    
    for ep in all_episodes:
        # Base memory usage increases slightly over time
        base = 0.3 + ep * 0.001
        
        # Spike during transfer events
        is_transfer = ep in transfer_episodes
        transfer_spike = 0.25 if is_transfer else 0
        
        # Add noise
        noise = np.random.normal(0, 0.03)
        
        # Calculate usage (capped at 98%)
        usage = min(0.98, base + transfer_spike + noise)
        memory_usage['usage'].append(usage)
    
    return memory_usage

def generate_sample_feature_importance():
    """Generate sample feature importance data."""
    # Define buckets and features
    buckets = ["Scalping", "Short", "Medium", "Long"]
    num_features = 30
    
    # Generate feature names
    feature_names = [f"Feature_{i+1}" for i in range(num_features)]
    
    # Initialize feature importance dict
    feature_importances = {}
    
    # Generate importance values for each bucket
    for bucket in buckets:
        # Base importance array
        importance = np.random.rand(num_features)
        
        # Make some features more important
        key_features = np.random.choice(num_features, 5, replace=False)
        for idx in key_features:
            importance[idx] += np.random.rand() * 0.5
        
        # Normalize
        importance = importance / importance.sum()
        
        # Add to dict
        feature_importances[bucket] = {'importance': importance}
    
    return feature_importances, feature_names

def main():
    """Main test function."""
    # Create output directory
    output_dir = os.path.join(current_dir, "test_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating test visualizations in: {output_dir}")
    
    # Create visualizer
    visualizer = ProgressiveTrainingVisualizer(output_dir=output_dir)
    
    # Generate sample data
    training_history = generate_sample_training_history()
    transfer_history = generate_sample_transfer_history(training_history)
    memory_usage = generate_sample_memory_usage(training_history, transfer_history)
    feature_importances, feature_names = generate_sample_feature_importance()
    
    # Generate visualizations
    print("Generating training progress visualization...")
    fig1 = visualizer.plot_training_progress(
        training_history,
        metrics=['reward', 'loss', 'win_rate'],
        smooth_window=5,
        save_path=os.path.join(output_dir, "training_progress.png")
    )
    
    print("Generating knowledge transfer visualization...")
    fig2 = visualizer.plot_knowledge_transfer(
        transfer_history,
        buckets=list(training_history.keys()),
        save_path=os.path.join(output_dir, "knowledge_transfer.png")
    )
    
    print("Generating memory usage visualization...")
    fig3 = visualizer.plot_memory_usage(
        memory_usage['episodes'],
        memory_usage['usage'],
        transfer_episodes=[entry['episode'] for entry in transfer_history],
        save_path=os.path.join(output_dir, "memory_usage.png")
    )
    
    print("Generating feature importance visualization...")
    fig4 = visualizer.plot_feature_importance(
        feature_importances,
        feature_names=feature_names,
        top_n=10,
        save_path=os.path.join(output_dir, "feature_importance.png")
    )
    
    print("Generating dashboard visualization...")
    fig5 = visualizer.plot_training_dashboard(
        training_history,
        transfer_history,
        memory_usage,
        save_path=os.path.join(output_dir, "dashboard.png")
    )
    
    # Generate full report
    print("Generating comprehensive report...")
    report_path = visualizer.generate_training_report(
        training_history,
        transfer_history,
        output_path=os.path.join(output_dir, "test_report")
    )
    
    print(f"Report generated at: {report_path}")
    print("All visualizations generated successfully!")
    
    # Display figures if interactive mode
    plt.show()

if __name__ == "__main__":
    main() 