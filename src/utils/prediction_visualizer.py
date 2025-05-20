#!/usr/bin/env python
"""
Visualization Tools for Probabilistic Predictions

This module provides visualization functions for displaying probabilistic predictions,
confidence intervals, calibration plots, and other uncertainty-related visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats
from matplotlib.gridspec import GridSpec
from collections import Counter

class PredictionVisualizer:
    """
    Class for visualizing probabilistic predictions and uncertainty.
    
    Features:
    - Price predictions with uncertainty bands
    - Confidence metric visualization
    - Prediction calibration plots
    - Prediction samples visualization
    - Interval coverage analysis
    """
    
    def __init__(self, style='darkgrid', context='talk', palette='viridis'):
        """
        Initialize visualizer with style settings.
        
        Args:
            style (str): Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
            context (str): Plotting context ('paper', 'notebook', 'talk', 'poster')
            palette (str): Color palette name
        """
        # Set visualization style
        sns.set_style(style)
        sns.set_context(context)
        
        # Set color palette
        self.palette = sns.color_palette(palette)
        self.colors = {
            "mean": self.palette[0],
            "actual": self.palette[1],
            "band_68": self.palette[2],
            "band_95": self.palette[3],
            "band_99": self.palette[4] if len(self.palette) > 4 else self.palette[0],
            "samples": self.palette[5] if len(self.palette) > 5 else self.palette[1]
        }
    
    def plot_price_predictions(self, actual_prices, pred_means, pred_stds, timestamps=None, title="Price Predictions with Uncertainty", save_path=None):
        """
        Plot price predictions with uncertainty bands.
        
        Args:
            actual_prices (array-like): Actual observed prices
            pred_means (array-like): Predicted mean prices
            pred_stds (array-like): Predicted standard deviations
            timestamps (array-like, optional): Timestamps for x-axis. If None, use indices.
            title (str, optional): Plot title
            save_path (str, optional): Path to save the figure. If None, display instead.
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create x-axis values
        if timestamps is None:
            x = np.arange(len(actual_prices))
        else:
            x = timestamps
            
        # Plot actual prices
        ax.plot(x, actual_prices, label="Actual", color=self.colors["actual"], linewidth=2)
        
        # Plot prediction mean
        ax.plot(x, pred_means, label="Predicted", color=self.colors["mean"], linewidth=2)
        
        # Plot uncertainty bands (68%, 95%, 99% confidence intervals)
        alpha_values = [0.4, 0.3, 0.2]
        z_values = [1.0, 2.0, 3.0]  # Z-values for confidence intervals
        band_labels = ["68% CI", "95% CI", "99% CI"]
        
        for z, alpha, band_label in zip(z_values, alpha_values, band_labels):
            upper = pred_means + z * pred_stds
            lower = pred_means - z * pred_stds
            ax.fill_between(x, lower, upper, alpha=alpha, 
                           color=self.colors[f"band_{int(z*68)}"], 
                           label=band_label)
        
        # Format plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Time" if timestamps is None else "Date", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format dates if timestamps provided and are datetime objects
        if timestamps is not None and isinstance(timestamps[0], (datetime, np.datetime64, pd.Timestamp)):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Tight layout
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confidence_metrics(self, prediction_times, confidence_scores, horizons=None, title="Prediction Confidence Over Time", save_path=None):
        """
        Plot confidence metrics over time.
        
        Args:
            prediction_times (array-like): Times when predictions were made
            confidence_scores (dict): Dictionary with horizons as keys and confidence arrays as values
            horizons (list, optional): List of horizons to plot. If None, plot all.
            title (str, optional): Plot title
            save_path (str, optional): Path to save the figure. If None, display instead.
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter horizons if specified
        if horizons is None:
            horizons = list(confidence_scores.keys())
        
        # Plot confidence for each horizon
        for i, horizon in enumerate(horizons):
            if horizon in confidence_scores:
                scores = confidence_scores[horizon]
                color = self.palette[i % len(self.palette)]
                ax.plot(prediction_times[:len(scores)], scores, 
                       label=f"Horizon {horizon}", color=color, alpha=0.8)
        
        # Format plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Confidence Score", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        # Format dates if prediction_times contains datetime objects
        if isinstance(prediction_times[0], (datetime, np.datetime64, pd.Timestamp)):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
        
        # Add legend
        ax.legend(loc='lower right')
        
        # Tight layout
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_samples(self, actual_prices, pred_means, pred_stds, num_samples=10, timestamps=None, title="Prediction Sample Paths", save_path=None):
        """
        Plot random samples from the predicted distribution.
        
        Args:
            actual_prices (array-like): Actual observed prices
            pred_means (array-like): Predicted mean prices
            pred_stds (array-like): Predicted standard deviations
            num_samples (int, optional): Number of random samples to generate
            timestamps (array-like, optional): Timestamps for x-axis. If None, use indices.
            title (str, optional): Plot title
            save_path (str, optional): Path to save the figure. If None, display instead.
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create x-axis values
        if timestamps is None:
            x = np.arange(len(actual_prices))
        else:
            x = timestamps
        
        # Generate random samples from the predicted distribution
        np.random.seed(42)  # For reproducibility
        samples = []
        
        for i in range(num_samples):
            # Sample from normal distribution with predicted mean and std
            sample = np.random.normal(pred_means, pred_stds)
            samples.append(sample)
            
            # Plot sample with low alpha
            ax.plot(x, sample, color=self.colors["samples"], alpha=0.3, linewidth=1)
        
        # Plot actual prices
        ax.plot(x, actual_prices, label="Actual", color=self.colors["actual"], linewidth=2)
        
        # Plot prediction mean
        ax.plot(x, pred_means, label="Predicted Mean", color=self.colors["mean"], linewidth=2)
        
        # Format plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Time" if timestamps is None else "Date", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format dates if timestamps provided
        if timestamps is not None and isinstance(timestamps[0], (datetime, np.datetime64, pd.Timestamp)):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
        
        # Add sample line to legend via custom patch
        sample_patch = Patch(color=self.colors["samples"], alpha=0.3, label="Sample Paths")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(sample_patch)
        ax.legend(handles=handles, loc='upper left')
        
        # Tight layout
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_curve(self, targets, pred_means, pred_stds, bins=10, title="Prediction Calibration", save_path=None):
        """
        Create a calibration plot to evaluate probabilistic predictions.
        
        Args:
            targets (array-like): Actual observed values
            pred_means (array-like): Predicted mean values
            pred_stds (array-like): Predicted standard deviations
            bins (int, optional): Number of bins for calibration. Defaults to 10.
            title (str, optional): Plot title
            save_path (str, optional): Path to save the figure. If None, display instead.
            
        Returns:
            matplotlib.figure.Figure: Figure object
            float: Calibration error (mean absolute deviation from ideal line)
        """
        # Calculate predicted quantiles
        pred_quantiles = []
        
        for target, mean, std in zip(targets, pred_means, pred_stds):
            # Convert to standard normal CDF
            z_score = (target - mean) / (std + 1e-8)
            quantile = stats.norm.cdf(z_score)
            pred_quantiles.append(quantile)
            
        # Create histogram bins
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Count observed frequencies in each bin
        observed_freq = np.zeros(bins)
        for q in pred_quantiles:
            bin_idx = min(int(q * bins), bins - 1)
            observed_freq[bin_idx] += 1
            
        # Convert to proportions
        observed_prop = observed_freq / len(pred_quantiles)
        cumulative_prop = np.cumsum(observed_prop)
        
        # Calculate calibration error
        expected_prop = bin_centers
        calibration_error = np.mean(np.abs(cumulative_prop - expected_prop))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot ideal calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Ideal Calibration')
        
        # Plot observed calibration
        ax.plot(bin_centers, cumulative_prop, 'o-', color=self.colors["mean"], 
               linewidth=2, label='Observed Calibration')
        
        # Format plot
        ax.set_title(f"{title}\nCalibration Error: {calibration_error:.4f}", fontsize=16)
        ax.set_xlabel("Expected Cumulative Probability", fontsize=12)
        ax.set_ylabel("Observed Cumulative Probability", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right')
        
        # Add diagonal fill to highlight deviations
        ax.fill_between(bin_centers, bin_centers, cumulative_prop, alpha=0.2, 
                      color=self.colors["band_68"])
        
        # Tight layout
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, calibration_error
    
    def plot_interval_coverage(self, targets, pred_means, pred_stds, title="Prediction Interval Coverage", save_path=None):
        """
        Plot interval coverage analysis to evaluate confidence intervals.
        
        Args:
            targets (array-like): Actual observed values
            pred_means (array-like): Predicted mean values
            pred_stds (array-like): Predicted standard deviations
            title (str, optional): Plot title
            save_path (str, optional): Path to save the figure. If None, display instead.
            
        Returns:
            matplotlib.figure.Figure: Figure object
            dict: Coverage statistics for different intervals
        """
        # Calculate coverage for different confidence intervals
        intervals = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
        z_values = [stats.norm.ppf((1 + interval) / 2) for interval in intervals]
        
        coverage_stats = {}
        coverage_counts = []
        
        for interval, z in zip(intervals, z_values):
            # Calculate bounds
            upper_bounds = pred_means + z * pred_stds
            lower_bounds = pred_means - z * pred_stds
            
            # Count targets within bounds
            within_bounds = np.sum((targets >= lower_bounds) & (targets <= upper_bounds))
            coverage = within_bounds / len(targets)
            coverage_stats[f"{interval*100:.0f}%"] = coverage
            coverage_counts.append(coverage)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot expected vs actual coverage
        bar_width = 0.35
        x = np.arange(len(intervals))
        
        ax.bar(x - bar_width/2, intervals, bar_width, label='Expected Coverage', 
               color=self.colors["band_68"], alpha=0.7)
        ax.bar(x + bar_width/2, coverage_counts, bar_width, label='Actual Coverage', 
               color=self.colors["mean"], alpha=0.7)
        
        # Add line for ideal ratio = 1
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        
        # Add text with coverage values
        for i, (interval, coverage) in enumerate(zip(intervals, coverage_counts)):
            ratio = coverage / interval
            color = 'green' if 0.9 <= ratio <= 1.1 else 'red'
            ax.text(i, max(interval, coverage) + 0.05, f"Ratio: {ratio:.2f}", 
                   ha='center', color=color, fontweight='bold')
        
        # Format plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Confidence Interval", fontsize=12)
        ax.set_ylabel("Coverage Proportion", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{interval*100:.0f}%" for interval in intervals])
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Tight layout
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, coverage_stats
    
    def plot_reasoning_chain(self, timestamps, reasoning_data, price_data=None, figsize=(14, 10)):
        """
        Visualize the agent's reasoning chain over time.
        
        Args:
            timestamps (list): List of timestamps or step indices
            reasoning_data (list): List of reasoning chain dictionaries
            price_data (list, optional): Price data for reference. Defaults to None.
            figsize (tuple, optional): Figure size. Defaults to (14, 10).
            
        Returns:
            matplotlib.figure.Figure: Figure with reasoning visualizations
        """
        if not reasoning_data:
            print("No reasoning data available for visualization")
            return None
            
        # Prepare data
        market_regimes = []
        regime_confidences = []
        entry_signals = []
        exit_signals = []
        support_signals = []
        resistance_signals = []
        volatility_values = []
        liquidity_values = []
        
        # Extract reasoning components
        for reasoning in reasoning_data:
            # Skip if missing reasoning chain
            if not reasoning or 'reasoning_chain' not in reasoning:
                continue
                
            reasoning_chain = reasoning['reasoning_chain']
            
            # Get market regime
            if 'market_regime' in reasoning_chain:
                regime_idx = np.argmax(reasoning_chain['market_regime'].cpu().numpy()[0])
                regime_names = ['trending', 'ranging', 'volatile', 'mixed']
                market_regimes.append(regime_names[regime_idx])
                regime_confidences.append(float(np.max(reasoning_chain['market_regime'].cpu().numpy()[0])))
            else:
                market_regimes.append('unknown')
                regime_confidences.append(0.0)
            
            # Get entry/exit signals
            if 'entry_exit' in reasoning_chain:
                entry_exit = reasoning_chain['entry_exit'].cpu().numpy()[0]
                entry_signals.append(float(entry_exit[0]))
                exit_signals.append(float(entry_exit[1]))
            else:
                entry_signals.append(0.0)
                exit_signals.append(0.0)
            
            # Get support/resistance
            if 'support_resistance' in reasoning_chain:
                sr = reasoning_chain['support_resistance'].cpu().numpy()[0]
                support_signals.append(float(sr[0]))
                resistance_signals.append(float(sr[1]))
            else:
                support_signals.append(0.0)
                resistance_signals.append(0.0)
            
            # Get risk factors
            if 'volatility' in reasoning_chain:
                volatility_values.append(float(reasoning_chain['volatility'].cpu().numpy()[0][0]))
            else:
                volatility_values.append(0.0)
                
            if 'liquidity' in reasoning_chain:
                liquidity_values.append(float(reasoning_chain['liquidity'].cpu().numpy()[0][0]))
            else:
                liquidity_values.append(0.0)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Define grid layout
        gs = GridSpec(4, 2, figure=fig)
        
        # Plot price if available
        if price_data is not None:
            ax_price = fig.add_subplot(gs[0, :])
            ax_price.plot(timestamps, price_data, color='black', linewidth=1.5)
            ax_price.set_title('Price Chart')
            ax_price.set_ylabel('Price')
            ax_price.grid(True, alpha=0.3)
        
        # Plot market regimes (as colored background)
        ax_regime = fig.add_subplot(gs[1, 0])
        
        # Convert regimes to numeric values for coloring
        regime_values = []
        for regime in market_regimes:
            if regime == 'trending':
                regime_values.append(0)
            elif regime == 'ranging':
                regime_values.append(1)
            elif regime == 'volatile':
                regime_values.append(2)
            else:  # mixed or unknown
                regime_values.append(3)
        
        # Plot regime confidence
        ax_regime.plot(timestamps, regime_confidences, color='blue', linewidth=1.5)
        
        # Add colored background for different regimes
        regime_colors = ['#90EE90', '#ADD8E6', '#FFB6C1', '#D3D3D3']  # green, blue, pink, gray
        
        # Create colored spans for each regime
        prev_regime = None
        start_idx = 0
        
        for i, regime in enumerate(regime_values):
            if regime != prev_regime and i > 0:
                # Add colored span
                ax_regime.axvspan(timestamps[start_idx], timestamps[i-1], 
                                 alpha=0.2, color=regime_colors[prev_regime])
                start_idx = i
            prev_regime = regime
            
        # Add the last span
        if len(regime_values) > 0:
            ax_regime.axvspan(timestamps[start_idx], timestamps[-1], 
                             alpha=0.2, color=regime_colors[prev_regime])
        
        ax_regime.set_title('Market Regime Confidence')
        ax_regime.set_ylabel('Confidence')
        ax_regime.set_ylim(0, 1)
        ax_regime.grid(True, alpha=0.3)
        
        # Plot entry/exit signals
        ax_signals = fig.add_subplot(gs[1, 1])
        ax_signals.plot(timestamps, entry_signals, color='green', linewidth=1.5, label='Entry Signal')
        ax_signals.plot(timestamps, exit_signals, color='red', linewidth=1.5, label='Exit Signal')
        ax_signals.set_title('Entry/Exit Signals')
        ax_signals.set_ylabel('Signal Strength')
        ax_signals.set_ylim(0, 1)
        ax_signals.grid(True, alpha=0.3)
        ax_signals.legend(loc='upper right')
        
        # Plot support/resistance
        ax_sr = fig.add_subplot(gs[2, 0])
        ax_sr.plot(timestamps, support_signals, color='green', linewidth=1.5, label='Support')
        ax_sr.plot(timestamps, resistance_signals, color='red', linewidth=1.5, label='Resistance')
        ax_sr.set_title('Support/Resistance Strength')
        ax_sr.set_ylabel('Strength')
        ax_sr.set_ylim(0, 1)
        ax_sr.grid(True, alpha=0.3)
        ax_sr.legend(loc='upper right')
        
        # Plot risk factors
        ax_risk = fig.add_subplot(gs[2, 1])
        ax_risk.plot(timestamps, volatility_values, color='orange', linewidth=1.5, label='Volatility')
        ax_risk.plot(timestamps, liquidity_values, color='purple', linewidth=1.5, label='Liquidity')
        ax_risk.set_title('Risk Factors')
        ax_risk.set_ylabel('Risk Level')
        ax_risk.set_ylim(0, 1)
        ax_risk.grid(True, alpha=0.3)
        ax_risk.legend(loc='upper right')
        
        # Plot regime distribution
        ax_dist = fig.add_subplot(gs[3, :])
        regime_counts = Counter(market_regimes)
        labels = list(regime_counts.keys())
        sizes = list(regime_counts.values())
        ax_dist.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=regime_colors)
        ax_dist.set_title('Market Regime Distribution')
        
        plt.tight_layout()
        return fig
        
    def create_reasoning_summary_table(self, reasoning_data, n_rows=10):
        """
        Create a pandas DataFrame summarizing the reasoning chain data.
        
        Args:
            reasoning_data (list): List of reasoning chain dictionaries
            n_rows (int, optional): Number of rows to include. Defaults to 10.
            
        Returns:
            pandas.DataFrame: DataFrame with reasoning summary
        """
        # Check if we have data
        if not reasoning_data or len(reasoning_data) == 0:
            print("No reasoning data available for table")
            return None
            
        # Prepare data
        data = []
        
        # Get the most recent n_rows
        for i, reasoning in enumerate(reasoning_data[-n_rows:]):
            if not reasoning or 'reasoning_chain' not in reasoning:
                continue
                
            row = {}
            row['step'] = i
            
            reasoning_chain = reasoning['reasoning_chain']
            
            # Get market regime
            if 'market_regime' in reasoning_chain:
                regime_probs = reasoning_chain['market_regime'].cpu().numpy()[0]
                regimes = ['trending', 'ranging', 'volatile', 'mixed']
                primary_regime = regimes[np.argmax(regime_probs)]
                row['regime'] = primary_regime
                row['regime_conf'] = f"{float(np.max(regime_probs)):.2f}"
            
            # Get entry/exit signals
            if 'entry_exit' in reasoning_chain:
                entry_exit = reasoning_chain['entry_exit'].cpu().numpy()[0]
                row['entry'] = f"{float(entry_exit[0]):.2f}"
                row['exit'] = f"{float(entry_exit[1]):.2f}"
            
            # Get risk factors
            if 'volatility' in reasoning_chain:
                row['volatility'] = f"{float(reasoning_chain['volatility'].cpu().numpy()[0][0]):.2f}"
                
            if 'liquidity' in reasoning_chain:
                row['liquidity'] = f"{float(reasoning_chain['liquidity'].cpu().numpy()[0][0]):.2f}"
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df 