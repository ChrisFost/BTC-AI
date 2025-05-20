#!/usr/bin/env python
"""
Progressive Training Visualization

This module provides visualization tools for monitoring progressive training and
displaying knowledge transfer between different bucket types.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
from typing import Dict, List, Tuple, Any, Optional, Union

class ProgressiveTrainingVisualizer:
    """
    Visualizer for progressive training and knowledge transfer.
    
    This class provides tools to visualize:
    - Training progress across multiple buckets
    - Knowledge transfer between buckets
    - Performance comparisons between buckets
    - Memory usage during training
    - Feature importance evolution
    """
    
    def __init__(self, style='darkgrid', context='talk', palette='viridis', 
                figsize=(12, 8), output_dir='visualizations'):
        """
        Initialize progressive training visualizer.
        
        Args:
            style (str): Seaborn style name
            context (str): Seaborn context name
            palette (str): Color palette name
            figsize (tuple): Default figure size (width, height)
            output_dir (str): Directory for saving visualizations
        """
        # Set up styling
        sns.set_style(style)
        sns.set_context(context)
        
        self.palette = sns.color_palette(palette)
        self.figsize = figsize
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define bucket colors
        self.bucket_colors = {
            'Scalping': self.palette[0],
            'Short': self.palette[1],
            'Medium': self.palette[2],
            'Long': self.palette[3]
        }
        
        # Define marker styles for different transfer types
        self.transfer_markers = {
            'weights': 'o',
            'features': 's',
            'horizons': '^'
        }
    
    def plot_training_progress(self, training_history: Dict[str, List[Dict]], 
                              metrics: List[str] = ['reward', 'loss'], 
                              smooth_window: int = 5,
                              title: str = "Training Progress",
                              save_path: str = None) -> plt.Figure:
        """
        Plot training progress for multiple buckets.
        
        Args:
            training_history: Dictionary with bucket names as keys and lists of metrics as values
            metrics: List of metric names to plot
            smooth_window: Window size for smoothing curves
            title: Plot title
            save_path: Path to save figure (if None, use default naming)
            
        Returns:
            Figure object
        """
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=self.figsize, sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        
        # For each bucket in training history
        for bucket, history in training_history.items():
            color = self.bucket_colors.get(bucket, self.palette[0])
            
            # Extract episodes
            episodes = [entry.get('episode', i) for i, entry in enumerate(history)]
            
            # Plot each metric
            for ax_idx, metric in enumerate(metrics):
                # Extract metric values
                values = [entry.get(metric, np.nan) for entry in history]
                
                # Apply smoothing if needed
                if smooth_window > 1 and len(values) > smooth_window:
                    smooth_values = np.convolve(values, np.ones(smooth_window)/smooth_window, mode='valid')
                    smooth_episodes = episodes[smooth_window-1:]
                    
                    # Plot raw data with low alpha
                    axes[ax_idx].plot(episodes, values, alpha=0.2, color=color)
                    # Plot smoothed data
                    axes[ax_idx].plot(smooth_episodes, smooth_values, 
                                     label=bucket, color=color, linewidth=2)
                else:
                    axes[ax_idx].plot(episodes, values, label=bucket, color=color, linewidth=2)
                
                # Format subplot
                axes[ax_idx].set_ylabel(metric.capitalize(), fontsize=12)
                axes[ax_idx].grid(True, alpha=0.3)
        
        # Add legend to top subplot
        axes[0].legend(loc='upper right')
        
        # Format figure
        fig.suptitle(title, fontsize=16)
        axes[-1].set_xlabel("Episodes", fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.output_dir:
            file_path = os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_knowledge_transfer(self, transfer_history: List[Dict],
                               buckets: List[str] = None,
                               metrics: Dict[str, List] = None,
                               title: str = "Knowledge Transfer Between Buckets",
                               save_path: str = None) -> plt.Figure:
        """
        Visualize knowledge transfer between buckets.
        
        Args:
            transfer_history: List of knowledge transfer events
            buckets: List of bucket names (if None, extract from history)
            metrics: Dictionary of metrics to overlay (optional)
            title: Plot title
            save_path: Path to save figure (if None, use default naming)
            
        Returns:
            Figure object
        """
        # Extract buckets if not provided
        if not buckets:
            source_buckets = [entry.get('source') for entry in transfer_history]
            target_buckets = [entry.get('target') for entry in transfer_history]
            buckets = sorted(list(set(source_buckets + target_buckets)))
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create y-positions for each bucket
        bucket_positions = {bucket: i for i, bucket in enumerate(buckets)}
        
        # Plot timeline
        episodes = [entry.get('episode', i) for i, entry in enumerate(transfer_history)]
        max_episode = max(episodes) if episodes else 100
        
        # Draw horizontal lines for buckets
        for bucket, pos in bucket_positions.items():
            ax.axhline(y=pos, color='gray', linestyle='-', alpha=0.3)
            
        # Draw transfer events
        for entry in transfer_history:
            source = entry.get('source')
            target = entry.get('target')
            episode = entry.get('episode', 0)
            transfer_types = entry.get('transfer_types', [])
            success = entry.get('success', True)
            
            if source in bucket_positions and target in bucket_positions:
                source_pos = bucket_positions[source]
                target_pos = bucket_positions[target]
                
                # Draw arrow for each transfer type
                for i, transfer_type in enumerate(transfer_types):
                    marker = self.transfer_markers.get(transfer_type, 'o')
                    offset = 0.1 * i  # Offset for multiple types
                    
                    # Determine color based on success
                    color = 'green' if success else 'red'
                    alpha = 0.8 if success else 0.4
                    
                    # Draw arrow from source to target
                    ax.annotate('', 
                               xy=(episode, target_pos + offset),
                               xytext=(episode, source_pos + offset),
                               arrowprops=dict(arrowstyle='->', color=color, alpha=alpha))
                    
                    # Add marker at source
                    ax.scatter(episode, source_pos + offset, marker=marker, 
                              color=color, s=50, alpha=alpha)
        
        # Set y-ticks at bucket positions
        ax.set_yticks(list(bucket_positions.values()))
        ax.set_yticklabels(list(bucket_positions.keys()))
        
        # Set x limits
        ax.set_xlim(-max_episode*0.05, max_episode*1.05)
        
        # Add legend for transfer types
        legend_elements = []
        for transfer_type, marker in self.transfer_markers.items():
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w',
                                  markerfacecolor='gray', markersize=10,
                                  label=transfer_type.capitalize()))
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Format plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Bucket", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.output_dir:
            file_path = os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, feature_importances: Dict[str, Dict[str, np.ndarray]],
                              feature_names: List[str] = None,
                              top_n: int = 20,
                              title: str = "Feature Importance Across Buckets",
                              save_path: str = None) -> plt.Figure:
        """
        Plot feature importance comparison across buckets.
        
        Args:
            feature_importances: Dict with bucket names as keys and feature importance dicts as values
            feature_names: List of feature names (if None, use indices)
            top_n: Number of top features to show
            title: Plot title
            save_path: Path to save figure (if None, use default naming)
            
        Returns:
            Figure object
        """
        buckets = list(feature_importances.keys())
        n_buckets = len(buckets)
        
        # Create figure
        fig, axes = plt.subplots(1, n_buckets, figsize=(self.figsize[0] * n_buckets / 2, self.figsize[1]),
                               sharey=True)
        if n_buckets == 1:
            axes = [axes]
        
        # Process each bucket
        for i, (bucket, importance) in enumerate(feature_importances.items()):
            # Get the latest feature importance (assuming it's a time series)
            if isinstance(importance, dict) and 'importance' in importance:
                latest_importance = importance['importance']
            else:
                latest_importance = importance
            
            # Ensure importance is a numpy array
            if not isinstance(latest_importance, np.ndarray):
                latest_importance = np.array(latest_importance)
            
            # Get feature names if not provided
            if feature_names is None:
                feat_names = [f"Feature {j}" for j in range(len(latest_importance))]
            else:
                feat_names = feature_names
            
            # Sort features by importance
            sorted_idx = np.argsort(latest_importance)[-top_n:]
            top_features = [feat_names[j] for j in sorted_idx]
            top_importances = latest_importance[sorted_idx]
            
            # Create barplot
            color = self.bucket_colors.get(bucket, self.palette[i % len(self.palette)])
            bars = axes[i].barh(range(len(top_features)), top_importances, color=color, alpha=0.7)
            
            # Add feature names
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features)
            
            # Format subplot
            axes[i].set_title(bucket, fontsize=14)
            axes[i].set_xlabel("Importance", fontsize=12)
            if i == 0:
                axes[i].set_ylabel("Features", fontsize=12)
            
            # Add grid
            axes[i].grid(True, axis='x', alpha=0.3)
        
        # Format figure
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.output_dir:
            file_path = os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_memory_usage(self, episodes: List[int], memory_usages: List[float],
                         transfer_episodes: List[int] = None,
                         gpu_target_thresholds: Tuple[float, float] = (0.65, 0.85),
                         title: str = "Memory Usage During Training",
                         save_path: str = None) -> plt.Figure:
        """
        Plot memory usage during training.
        
        Args:
            episodes: List of episode numbers
            memory_usages: List of memory usage values (0-1 range)
            transfer_episodes: List of episodes where knowledge transfer occurred
            gpu_target_thresholds: Tuple of (optimal, warning) thresholds
            title: Plot title
            save_path: Path to save figure (if None, use default naming)
            
        Returns:
            Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot memory usage
        ax.plot(episodes, memory_usages, 'b-', linewidth=2, label='GPU Memory Usage')
        
        # Add thresholds
        optimal, warning = gpu_target_thresholds
        ax.axhline(y=optimal, color='g', linestyle='--', alpha=0.6, 
                  label=f'Optimal Threshold ({optimal:.0%})')
        ax.axhline(y=warning, color='orange', linestyle='--', alpha=0.6,
                  label=f'Warning Threshold ({warning:.0%})')
        ax.axhline(y=1.0, color='r', linestyle='-', alpha=0.4,
                  label='Maximum Memory')
        
        # Color regions
        ax.fill_between(episodes, 0, optimal, color='g', alpha=0.1)
        ax.fill_between(episodes, optimal, warning, color='y', alpha=0.1)
        ax.fill_between(episodes, warning, 1.0, color='r', alpha=0.1)
        
        # Mark transfer events
        if transfer_episodes:
            for ep in transfer_episodes:
                if ep in episodes:
                    idx = episodes.index(ep)
                    memory = memory_usages[idx]
                    ax.scatter([ep], [memory], color='purple', s=100, marker='*', 
                              zorder=5, label='Knowledge Transfer' if ep == transfer_episodes[0] else "")
                    ax.axvline(x=ep, color='purple', linestyle=':', alpha=0.4)
        
        # Format plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("GPU Memory Usage", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        # Add legend (without duplicate entries)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.output_dir:
            file_path = os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_dashboard(self, training_history: Dict[str, List[Dict]],
                              transfer_history: List[Dict],
                              memory_usage: Dict[str, List[float]] = None,
                              title: str = "Progressive Training Dashboard",
                              save_path: str = None) -> plt.Figure:
        """
        Create a comprehensive dashboard visualization.
        
        Args:
            training_history: Dictionary with bucket names as keys and lists of metrics as values
            transfer_history: List of knowledge transfer events
            memory_usage: Dictionary with episodes and memory usage values
            title: Dashboard title
            save_path: Path to save figure (if None, use default naming)
            
        Returns:
            Figure object
        """
        # Extract buckets from training history
        buckets = list(training_history.keys())
        
        # Create figure with complex grid
        fig = plt.figure(figsize=(self.figsize[0] * 1.5, self.figsize[1] * 1.5))
        gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1])
        
        # Training progress subplot (top left)
        ax_progress = fig.add_subplot(gs[0, 0])
        self._plot_training_progress_subplot(ax_progress, training_history)
        
        # Knowledge transfer subplot (top middle)
        ax_transfer = fig.add_subplot(gs[0, 1])
        self._plot_transfer_subplot(ax_transfer, transfer_history, buckets)
        
        # Bucket metrics summary (top right)
        ax_metrics = fig.add_subplot(gs[0, 2])
        self._plot_bucket_metrics_subplot(ax_metrics, training_history)
        
        # Memory usage (middle row, spans all columns)
        ax_memory = fig.add_subplot(gs[1, :])
        if memory_usage and 'episodes' in memory_usage and 'usage' in memory_usage:
            transfer_episodes = [entry.get('episode', 0) for entry in transfer_history]
            self._plot_memory_subplot(ax_memory, memory_usage['episodes'], 
                                    memory_usage['usage'], transfer_episodes)
        
        # Transfer success rate (bottom left)
        ax_success = fig.add_subplot(gs[2, 0])
        self._plot_transfer_success_subplot(ax_success, transfer_history)
        
        # Latest transfer details (bottom middle and right)
        ax_latest = fig.add_subplot(gs[2, 1:])
        self._plot_latest_transfer_subplot(ax_latest, transfer_history)
        
        # Add dashboard title
        fig.suptitle(title, fontsize=18, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.output_dir:
            file_path = os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_training_progress_subplot(self, ax, training_history, metric='reward', smooth_window=5):
        """Helper method to plot training progress on a given axis."""
        for bucket, history in training_history.items():
            color = self.bucket_colors.get(bucket, 'blue')
            
            # Extract episodes and metric values
            episodes = [entry.get('episode', i) for i, entry in enumerate(history)]
            values = [entry.get(metric, np.nan) for entry in history]
            
            # Apply smoothing if needed
            if smooth_window > 1 and len(values) > smooth_window:
                smooth_values = np.convolve(values, np.ones(smooth_window)/smooth_window, mode='valid')
                smooth_episodes = episodes[smooth_window-1:]
                
                # Plot raw data with low alpha
                ax.plot(episodes, values, alpha=0.2, color=color)
                # Plot smoothed data
                ax.plot(smooth_episodes, smooth_values, label=bucket, color=color, linewidth=2)
            else:
                ax.plot(episodes, values, label=bucket, color=color, linewidth=2)
        
        # Format subplot
        ax.set_title("Training Progress", fontsize=14)
        ax.set_xlabel("Episodes", fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
    
    def _plot_transfer_subplot(self, ax, transfer_history, buckets):
        """Helper method to plot knowledge transfer on a given axis."""
        # Create y-positions for each bucket
        bucket_positions = {bucket: i for i, bucket in enumerate(buckets)}
        
        # Plot timeline
        episodes = [entry.get('episode', i) for i, entry in enumerate(transfer_history)]
        max_episode = max(episodes) if episodes else 100
        
        # Draw horizontal lines for buckets
        for bucket, pos in bucket_positions.items():
            ax.axhline(y=pos, color='gray', linestyle='-', alpha=0.3)
            
        # Draw transfer events
        for entry in transfer_history:
            source = entry.get('source')
            target = entry.get('target')
            episode = entry.get('episode', 0)
            transfer_types = entry.get('transfer_types', [])
            success = entry.get('success', True)
            
            if source in bucket_positions and target in bucket_positions:
                source_pos = bucket_positions[source]
                target_pos = bucket_positions[target]
                
                # Draw arrow for transfer
                color = 'green' if success else 'red'
                alpha = 0.8 if success else 0.4
                
                # Draw arrow from source to target
                ax.annotate('', 
                           xy=(episode, target_pos),
                           xytext=(episode, source_pos),
                           arrowprops=dict(arrowstyle='->', color=color, alpha=alpha))
        
        # Set y-ticks at bucket positions
        ax.set_yticks(list(bucket_positions.values()))
        ax.set_yticklabels(list(bucket_positions.keys()))
        
        # Format subplot
        ax.set_title("Knowledge Transfer", fontsize=14)
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel("Bucket", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_bucket_metrics_subplot(self, ax, training_history):
        """Helper method to plot bucket metric summary on a given axis."""
        # Prepare data
        buckets = list(training_history.keys())
        metrics = ['reward', 'win_rate', 'profit_factor']
        available_metrics = []
        
        # Check which metrics are available
        for metric in metrics:
            for bucket, history in training_history.items():
                if history and any(metric in entry for entry in history):
                    available_metrics.append(metric)
                    break
        
        if not available_metrics:
            ax.text(0.5, 0.5, "No metrics available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Bucket Metrics", fontsize=14)
            return
        
        # Get latest values for each metric and bucket
        data = {}
        for bucket, history in training_history.items():
            if not history:
                continue
                
            latest = history[-1]
            for metric in available_metrics:
                if metric in latest:
                    if metric not in data:
                        data[metric] = []
                    data[metric].append(latest[metric])
        
        # Plot bar charts for each metric
        bar_width = 0.8 / len(available_metrics)
        for i, metric in enumerate(available_metrics):
            if metric in data:
                positions = np.arange(len(buckets)) + (i - len(available_metrics)/2 + 0.5) * bar_width
                bars = ax.bar(positions, data[metric], width=bar_width, 
                             label=metric.capitalize(), alpha=0.7)
        
        # Format subplot
        ax.set_title("Latest Metrics by Bucket", fontsize=14)
        ax.set_xticks(np.arange(len(buckets)))
        ax.set_xticklabels(buckets, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    def _plot_memory_subplot(self, ax, episodes, memory_usage, transfer_episodes=None):
        """Helper method to plot memory usage on a given axis."""
        # Plot memory usage
        ax.plot(episodes, memory_usage, 'b-', linewidth=2)
        
        # Mark transfer events
        if transfer_episodes:
            for ep in transfer_episodes:
                if ep in episodes:
                    idx = episodes.index(ep)
                    memory = memory_usage[idx]
                    ax.scatter([ep], [memory], color='purple', s=80, marker='*', zorder=5)
                    ax.axvline(x=ep, color='purple', linestyle=':', alpha=0.4)
        
        # Format subplot
        ax.set_title("Memory Usage", fontsize=14)
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel("GPU Memory Usage", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    def _plot_transfer_success_subplot(self, ax, transfer_history):
        """Helper method to plot transfer success rate on a given axis."""
        if not transfer_history:
            ax.text(0.5, 0.5, "No transfer data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Transfer Success Rate", fontsize=14)
            return
        
        # Count success and failure by transfer type
        transfer_types = {}
        for entry in transfer_history:
            types = entry.get('transfer_types', [])
            success = entry.get('success', True)
            
            for tp in types:
                if tp not in transfer_types:
                    transfer_types[tp] = {'success': 0, 'failure': 0}
                
                if success:
                    transfer_types[tp]['success'] += 1
                else:
                    transfer_types[tp]['failure'] += 1
        
        # Prepare data for plotting
        types = list(transfer_types.keys())
        successes = [transfer_types[tp]['success'] for tp in types]
        failures = [transfer_types[tp]['failure'] for tp in types]
        
        # Create stacked bar chart
        bar_width = 0.5
        positions = np.arange(len(types))
        
        # Plot bars
        ax.bar(positions, successes, bar_width, label='Success', color='green', alpha=0.7)
        ax.bar(positions, failures, bar_width, bottom=successes, 
              label='Failure', color='red', alpha=0.7)
        
        # Add success rate text
        for i, tp in enumerate(types):
            total = successes[i] + failures[i]
            if total > 0:
                success_rate = successes[i] / total * 100
                ax.text(i, total + 0.5, f"{success_rate:.0f}%", 
                       ha='center', va='bottom', fontsize=9)
        
        # Format subplot
        ax.set_title("Transfer Success by Type", fontsize=14)
        ax.set_xticks(positions)
        ax.set_xticklabels([t.capitalize() for t in types])
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    def _plot_latest_transfer_subplot(self, ax, transfer_history):
        """Helper method to plot latest transfer details on a given axis."""
        if not transfer_history:
            ax.text(0.5, 0.5, "No transfer data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Latest Transfer Details", fontsize=14)
            ax.axis('off')
            return
        
        # Get the latest transfer
        latest = transfer_history[-1]
        
        # Clear axis and remove frame
        ax.clear()
        ax.axis('off')
        
        # Create a formatted text summary
        source = latest.get('source', 'Unknown')
        target = latest.get('target', 'Unknown')
        episode = latest.get('episode', 0)
        transfer_types = latest.get('transfer_types', [])
        success = latest.get('success', True)
        timestamp = latest.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        duration = latest.get('duration', 0)
        metrics_before = latest.get('metrics_before', {})
        metrics_after = latest.get('metrics_after', {})
        
        # Header
        header = f"Latest Transfer (Episode {episode})"
        
        # Status text with color
        status_color = 'green' if success else 'red'
        status_text = f"Status: {'Success' if success else 'Failed'}"
        
        # Basic info
        basic_info = (
            f"From: {source} → To: {target}\n"
            f"Time: {timestamp}\n"
            f"Duration: {duration:.1f}s\n"
            f"Types: {', '.join(t.capitalize() for t in transfer_types)}"
        )
        
        # Performance impact (if available)
        impact_text = "Performance Impact:\n"
        if metrics_before and metrics_after:
            for metric in metrics_before:
                if metric in metrics_after:
                    before = metrics_before[metric]
                    after = metrics_after[metric]
                    change = after - before
                    change_pct = (change / before * 100) if before != 0 else float('inf')
                    
                    impact_text += f"  {metric.capitalize()}: {before:.4f} → {after:.4f} "
                    if change > 0:
                        impact_text += f"(↑ +{change_pct:.1f}%)\n"
                    else:
                        impact_text += f"(↓ {change_pct:.1f}%)\n"
        else:
            impact_text += "  No performance data available"
        
        # Add text to plot
        ax.text(0.02, 0.98, header, fontsize=14, weight='bold', va='top')
        ax.text(0.02, 0.90, status_text, fontsize=12, color=status_color, va='top')
        ax.text(0.02, 0.82, basic_info, fontsize=10, va='top')
        ax.text(0.02, 0.55, impact_text, fontsize=10, va='top')
        
        # Add a border
        ax.set_title("Latest Transfer Details", fontsize=14)
    
    def generate_training_report(self, training_history: Dict[str, List[Dict]],
                              transfer_history: List[Dict],
                              output_path: str = None) -> str:
        """
        Generate a comprehensive training report with visualizations.
        
        Args:
            training_history: Dictionary with bucket names as keys and lists of metrics as values
            transfer_history: List of knowledge transfer events
            output_path: Path to save the report (if None, use default)
            
        Returns:
            Path to the generated report
        """
        # Create output directory
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"training_report_{timestamp}")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Generate visualizations
        fig_progress = self.plot_training_progress(
            training_history, 
            metrics=['reward', 'loss'],
            save_path=os.path.join(output_path, "training_progress.png")
        )
        
        fig_transfer = self.plot_knowledge_transfer(
            transfer_history,
            save_path=os.path.join(output_path, "knowledge_transfer.png")
        )
        
        # Generate dashboard
        fig_dashboard = self.plot_training_dashboard(
            training_history,
            transfer_history,
            save_path=os.path.join(output_path, "dashboard.png")
        )
        
        # Close all figures to free memory
        plt.close(fig_progress)
        plt.close(fig_transfer)
        plt.close(fig_dashboard)
        
        # Create a markdown report
        report_file = os.path.join(output_path, "report.md")
        with open(report_file, 'w') as f:
            # Report header
            f.write("# Progressive Training Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Training summary
            f.write("## Training Summary\n\n")
            
            # Table of metrics by bucket
            f.write("### Final Metrics\n\n")
            f.write("| Bucket | Episodes | Reward | Win Rate | Profit Factor |\n")
            f.write("|--------|----------|--------|----------|---------------|\n")
            
            for bucket, history in training_history.items():
                if history:
                    latest = history[-1]
                    episodes = latest.get('episode', 0)
                    reward = latest.get('reward', 'N/A')
                    win_rate = latest.get('win_rate', 'N/A')
                    profit_factor = latest.get('profit_factor', 'N/A')
                    
                    f.write(f"| {bucket} | {episodes} | {reward:.4f} | {win_rate:.2%} | {profit_factor:.2f} |\n")
            
            f.write("\n")
            
            # Knowledge transfer summary
            f.write("## Knowledge Transfer Summary\n\n")
            
            transfer_count = len(transfer_history)
            success_count = sum(1 for entry in transfer_history if entry.get('success', True))
            success_rate = success_count / transfer_count if transfer_count else 0
            
            f.write(f"Total transfers: {transfer_count}\n\n")
            f.write(f"Success rate: {success_rate:.2%}\n\n")
            
            # Transfer table
            f.write("### Transfer Events\n\n")
            f.write("| Episode | Source | Target | Types | Success | Duration |\n")
            f.write("|---------|--------|--------|-------|---------|----------|\n")
            
            for entry in transfer_history:
                episode = entry.get('episode', 0)
                source = entry.get('source', 'Unknown')
                target = entry.get('target', 'Unknown')
                types = ', '.join(entry.get('transfer_types', []))
                success = 'Success' if entry.get('success', True) else 'Failed'
                duration = entry.get('duration', 0)
                
                f.write(f"| {episode} | {source} | {target} | {types} | {success} | {duration:.1f}s |\n")
            
            f.write("\n")
            
            # Include visualizations
            f.write("## Visualizations\n\n")
            
            f.write("### Training Progress\n\n")
            f.write("![Training Progress](training_progress.png)\n\n")
            
            f.write("### Knowledge Transfer\n\n")
            f.write("![Knowledge Transfer](knowledge_transfer.png)\n\n")
            
            f.write("### Dashboard\n\n")
            f.write("![Dashboard](dashboard.png)\n\n")
        
        return report_file


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training visualizations')
    parser.add_argument('--models-dir', type=str, help='Directory containing model files')
    parser.add_argument('--output-dir', type=str, default='visualizations', 
                       help='Directory to save visualizations')
    parser.add_argument('--report', action='store_true', help='Generate a full report')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ProgressiveTrainingVisualizer(output_dir=args.output_dir)
    
    # Generate example visualizations (replace with actual data loading)
    print("Generating example visualizations...")
    
    # Example training history
    training_history = {
        'Scalping': [{'episode': i, 'reward': np.sin(i/10) + i/50, 'loss': 1/(i+1)} for i in range(100)],
        'Short': [{'episode': i, 'reward': np.sin(i/10) + i/40, 'loss': 1.2/(i+1)} for i in range(80)],
        'Medium': [{'episode': i, 'reward': np.sin(i/10) + i/30, 'loss': 1.5/(i+1)} for i in range(50)]
    }
    
    # Example transfer history
    transfer_history = [
        {'episode': 20, 'source': 'Scalping', 'target': 'Short', 
         'transfer_types': ['weights', 'features'], 'success': True},
        {'episode': 40, 'source': 'Scalping', 'target': 'Medium', 
         'transfer_types': ['weights'], 'success': False},
        {'episode': 60, 'source': 'Short', 'target': 'Medium', 
         'transfer_types': ['weights', 'features', 'horizons'], 'success': True}
    ]
    
    # Generate and save visualizations
    visualizer.plot_training_progress(training_history)
    visualizer.plot_knowledge_transfer(transfer_history)
    
    # Generate dashboard
    memory_usage = {
        'episodes': list(range(100)),
        'usage': [0.3 + 0.1 * np.sin(i/10) + i/200 for i in range(100)]
    }
    visualizer.plot_training_dashboard(training_history, transfer_history, memory_usage)
    
    # Generate report if requested
    if args.report:
        report_path = visualizer.generate_training_report(training_history, transfer_history)
        print(f"Generated report at: {report_path}")
    
    print("Visualizations saved to:", args.output_dir) 