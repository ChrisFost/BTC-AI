"""
Visualization tools for probabilistic trading models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
from datetime import datetime
import logging

class PredictionVisualizer:
    """Class for visualizing model predictions and uncertainty"""
    
    def __init__(
        self,
        output_dir: str = 'visualization_output',
        save_plots: bool = True,
        plot_style: str = 'seaborn-v0_8-whitegrid',
        confidence_levels: List[float] = [0.5, 0.8, 0.9, 0.95],
        colors: Dict[str, str] = None
    ):
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.plot_style = plot_style
        self.confidence_levels = confidence_levels
        
        # Set default colors if not provided
        if colors is None:
            self.colors = {
                'actual': 'black',
                'prediction': 'blue',
                'uncertainty': 'skyblue',
                'buy': 'green',
                'sell': 'red',
                'hold': 'gray',
                'scalping': 'lightcoral',
                'short': 'coral',
                'medium': 'orangered',
                'long': 'firebrick'
            }
        else:
            self.colors = colors
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use(self.plot_style)
    
    def plot_price_predictions(
        self,
        prices: np.ndarray,
        pred_means: np.ndarray,
        pred_stds: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        horizon_name: str = '',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = False,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot price predictions with uncertainty bands
        
        Args:
            prices: Actual prices
            pred_means: Predicted mean prices
            pred_stds: Predicted standard deviations
            timestamps: Timestamps for x-axis (optional)
            horizon_name: Name of prediction horizon (optional)
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Create x-axis
        x = timestamps if timestamps is not None else np.arange(len(prices))
        
        # Plot uncertainty bands for confidence levels
        for conf_level in reversed(self.confidence_levels):
            z_score = np.sqrt(2) * np.abs(np.arctanh(conf_level)) if conf_level < 1 else 3
            plt.fill_between(
                x,
                pred_means - z_score * pred_stds,
                pred_means + z_score * pred_stds,
                alpha=0.2,
                color=self.colors['uncertainty'],
                label=f'{conf_level:.0%} CI' if conf_level == self.confidence_levels[0] else None
            )
        
        # Plot actual prices and predictions
        plt.plot(x, prices, 'o-', color=self.colors['actual'], label='Actual', alpha=0.7, markersize=3)
        plt.plot(x, pred_means, '-', color=self.colors['prediction'], label='Prediction', linewidth=2)
        
        # Set title and labels
        if title:
            plt.title(title)
        else:
            plt.title(f'Price Predictions{" - " + horizon_name if horizon_name else ""}')
        
        plt.xlabel('Time' if timestamps is None else 'Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis for timestamps
        if timestamps is not None and isinstance(timestamps[0], (pd.Timestamp, datetime)):
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots and save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.output_dir,
                f'price_predictions_{horizon_name}_{timestamp}.png'
            )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_multi_horizon_predictions(
        self,
        prices: np.ndarray,
        horizon_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
        timestamps: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = False,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot predictions for multiple horizons
        
        Args:
            prices: Actual prices
            horizon_predictions: Dictionary of (mean, std) tuples for each horizon
            timestamps: Timestamps for x-axis (optional)
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
        """
        n_horizons = len(horizon_predictions)
        fig, axs = plt.subplots(n_horizons, 1, figsize=figsize, sharex=True)
        
        # Create x-axis
        x = timestamps if timestamps is not None else np.arange(len(prices))
        
        # Plot for each horizon
        for i, (horizon_name, (pred_means, pred_stds)) in enumerate(horizon_predictions.items()):
            ax = axs[i] if n_horizons > 1 else axs
            
            # Plot uncertainty band for each confidence level
            for conf_level in reversed(self.confidence_levels):
                z_score = np.sqrt(2) * np.abs(np.arctanh(conf_level)) if conf_level < 1 else 3
                ax.fill_between(
                    x,
                    pred_means - z_score * pred_stds,
                    pred_means + z_score * pred_stds,
                    alpha=0.2,
                    color=self.colors.get(horizon_name, self.colors['uncertainty']),
                    label=f'{conf_level:.0%} CI' if conf_level == self.confidence_levels[0] else None
                )
            
            # Plot actual prices and predictions
            ax.plot(x, prices, 'o-', color=self.colors['actual'], label='Actual', alpha=0.7, markersize=3)
            ax.plot(x, pred_means, '-', color=self.colors.get(horizon_name, self.colors['prediction']), 
                    label='Prediction', linewidth=2)
            
            # Set title and labels
            ax.set_title(f'{horizon_name.capitalize()} Horizon')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Set overall title
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle('Multi-Horizon Price Predictions', fontsize=16)
        
        # Set x-axis label for bottom subplot
        axs[-1].set_xlabel('Time' if timestamps is None else 'Date')
        
        # Format x-axis for timestamps
        if timestamps is not None and isinstance(timestamps[0], (pd.Timestamp, datetime)):
            axs[-1].xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save plot if requested
        if self.save_plots and save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.output_dir,
                f'multi_horizon_predictions_{timestamp}.png'
            )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_confidence_metrics(
        self,
        confidence_values: Dict[str, np.ndarray],
        timestamps: Optional[np.ndarray] = None,
        threshold: Optional[float] = 0.6,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = False,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot confidence metrics over time
        
        Args:
            confidence_values: Dictionary of confidence values for each horizon
            timestamps: Timestamps for x-axis (optional)
            threshold: Confidence threshold line (optional)
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Create x-axis
        x = timestamps if timestamps is not None else np.arange(len(next(iter(confidence_values.values()))))
        
        # Plot confidence values for each horizon
        for horizon_name, conf_values in confidence_values.items():
            plt.plot(
                x, conf_values, '-', 
                label=f'{horizon_name.capitalize()}',
                color=self.colors.get(horizon_name, None),
                linewidth=2
            )
        
        # Add threshold line if provided
        if threshold is not None:
            plt.axhline(
                y=threshold, color='gray', linestyle='--', 
                alpha=0.7, label=f'Threshold ({threshold:.1f})'
            )
        
        # Set title and labels
        if title:
            plt.title(title)
        else:
            plt.title('Prediction Confidence Over Time')
        
        plt.xlabel('Time' if timestamps is None else 'Date')
        plt.ylabel('Confidence')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis for timestamps
        if timestamps is not None and isinstance(timestamps[0], (pd.Timestamp, datetime)):
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        # Set y-axis limits
        plt.ylim(0, 1.05)
        
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots and save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.output_dir,
                f'confidence_metrics_{timestamp}.png'
            )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_prediction_samples(
        self,
        prices: np.ndarray,
        samples: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        n_samples: int = 20,
        horizon_name: str = '',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = False,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot individual prediction samples
        
        Args:
            prices: Actual prices
            samples: Prediction samples (shape: [n_timesteps, n_samples])
            timestamps: Timestamps for x-axis (optional)
            n_samples: Number of samples to plot
            horizon_name: Name of prediction horizon (optional)
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Create x-axis
        x = timestamps if timestamps is not None else np.arange(len(prices))
        
        # Plot actual prices
        plt.plot(x, prices, 'o-', color=self.colors['actual'], label='Actual', alpha=0.8, markersize=4, linewidth=2)
        
        # Limit the number of samples
        n_plot = min(n_samples, samples.shape[1])
        
        # Plot samples
        for i in range(n_plot):
            plt.plot(x, samples[:, i], '-', alpha=0.2, color=self.colors['prediction'])
        
        # Add a label for the samples
        plt.plot([], [], '-', color=self.colors['prediction'], alpha=0.2, label='Samples')
        
        # Set title and labels
        if title:
            plt.title(title)
        else:
            plt.title(f'Prediction Samples{" - " + horizon_name if horizon_name else ""}')
        
        plt.xlabel('Time' if timestamps is None else 'Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis for timestamps
        if timestamps is not None and isinstance(timestamps[0], (pd.Timestamp, datetime)):
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots and save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.output_dir,
                f'prediction_samples_{horizon_name}_{timestamp}.png'
            )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_calibration_curve(
        self,
        targets: np.ndarray,
        pred_means: np.ndarray,
        pred_stds: np.ndarray,
        horizon_name: str = '',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = False,
        figsize: Tuple[int, int] = (8, 8)
    ):
        """
        Plot calibration curve
        
        Args:
            targets: Actual target values
            pred_means: Predicted mean values
            pred_stds: Predicted standard deviations
            horizon_name: Name of prediction horizon (optional)
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Generate confidence levels
        confidence_levels = np.linspace(0.01, 0.99, 50)
        observed_frequencies = []
        
        # Calculate observed frequencies
        for conf_level in confidence_levels:
            z_score = np.sqrt(2) * np.abs(np.arctanh(conf_level)) if conf_level < 1 else 3
            lower = pred_means - z_score * pred_stds
            upper = pred_means + z_score * pred_stds
            
            # Calculate frequency of targets within prediction interval
            within_interval = np.logical_and(targets >= lower, targets <= upper)
            frequency = np.mean(within_interval)
            observed_frequencies.append(frequency)
        
        # Plot calibration curve
        plt.plot(confidence_levels, observed_frequencies, '-', color='blue', linewidth=2, label='Model')
        plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5, label='Perfect Calibration')
        
        # Add shaded area for over/under confidence
        plt.fill_between(
            confidence_levels, confidence_levels, observed_frequencies,
            where=(observed_frequencies <= confidence_levels),
            color='red', alpha=0.2, label='Under-confident'
        )
        plt.fill_between(
            confidence_levels, confidence_levels, observed_frequencies,
            where=(observed_frequencies >= confidence_levels),
            color='green', alpha=0.2, label='Over-confident'
        )
        
        # Set title and labels
        if title:
            plt.title(title)
        else:
            plt.title(f'Calibration Curve{" - " + horizon_name if horizon_name else ""}')
        
        plt.xlabel('Expected Fraction')
        plt.ylabel('Observed Fraction')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Set axis limits
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Add 45-degree diagonal grid lines
        plt.plot([0.2, 0.2], [0, 0.2], ':', color='gray', alpha=0.5)
        plt.plot([0.4, 0.4], [0, 0.4], ':', color='gray', alpha=0.5)
        plt.plot([0.6, 0.6], [0, 0.6], ':', color='gray', alpha=0.5)
        plt.plot([0.8, 0.8], [0, 0.8], ':', color='gray', alpha=0.5)
        
        plt.plot([0, 0.2], [0.2, 0.2], ':', color='gray', alpha=0.5)
        plt.plot([0, 0.4], [0.4, 0.4], ':', color='gray', alpha=0.5)
        plt.plot([0, 0.6], [0.6, 0.6], ':', color='gray', alpha=0.5)
        plt.plot([0, 0.8], [0.8, 0.8], ':', color='gray', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots and save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.output_dir,
                f'calibration_curve_{horizon_name}_{timestamp}.png'
            )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_interval_coverage(
        self,
        targets: np.ndarray,
        pred_means: np.ndarray,
        pred_stds: np.ndarray,
        horizon_names: List[str],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = False,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot interval coverage for multiple horizons
        
        Args:
            targets: Dictionary or list of actual target values
            pred_means: Dictionary or list of predicted mean values
            pred_stds: Dictionary or list of predicted standard deviations
            horizon_names: Names of prediction horizons
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Fixed confidence levels to evaluate
        eval_levels = [0.5, 0.8, 0.9, 0.95]
        
        # Set up bar positions
        n_horizons = len(horizon_names)
        n_levels = len(eval_levels)
        bar_width = 0.8 / n_horizons
        positions = np.arange(n_levels)
        
        # Calculate coverage for each horizon
        for i, horizon in enumerate(horizon_names):
            coverages = []
            
            # Get targets and predictions for this horizon
            if isinstance(targets, dict):
                h_targets = targets[horizon]
                h_pred_means = pred_means[horizon]
                h_pred_stds = pred_stds[horizon]
            else:
                h_targets = targets
                h_pred_means = pred_means
                h_pred_stds = pred_stds
            
            # Calculate coverage for each confidence level
            for conf_level in eval_levels:
                z_score = np.sqrt(2) * np.abs(np.arctanh(conf_level)) if conf_level < 1 else 3
                lower = h_pred_means - z_score * h_pred_stds
                upper = h_pred_means + z_score * h_pred_stds
                
                # Calculate frequency of targets within prediction interval
                within_interval = np.logical_and(h_targets >= lower, h_targets <= upper)
                coverage = np.mean(within_interval)
                coverages.append(coverage)
            
            # Plot bars for this horizon
            offset = bar_width * (i - n_horizons / 2 + 0.5)
            plt.bar(
                positions + offset, coverages, width=bar_width,
                color=self.colors.get(horizon, None),
                label=horizon.capitalize()
            )
        
        # Add expected coverage line
        for i, level in enumerate(eval_levels):
            plt.plot(
                [i - 0.4, i + 0.4], [level, level],
                '--', color='black', alpha=0.7, linewidth=1.5
            )
        
        # Set title and labels
        if title:
            plt.title(title)
        else:
            plt.title('Prediction Interval Coverage')
        
        plt.xlabel('Confidence Level')
        plt.ylabel('Observed Coverage')
        plt.xticks(positions, [f'{level:.0%}' for level in eval_levels])
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
        
        # Set y-axis limits
        plt.ylim(0, 1.05)
        
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots and save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.output_dir,
                f'interval_coverage_{timestamp}.png'
            )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_trading_signals(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        positions: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = False,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot trading signals and positions
        
        Args:
            prices: Price data
            signals: Trading signals (1 for buy, 0 for hold, -1 for sell)
            positions: Position sizes (optional)
            timestamps: Timestamps for x-axis (optional)
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        
        # Create x-axis
        x = timestamps if timestamps is not None else np.arange(len(prices))
        
        # Setup subplots
        if positions is not None:
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1], sharex=ax1)
        else:
            ax1 = plt.gca()
        
        # Plot price chart
        ax1.plot(x, prices, color='black', linewidth=1.5, label='Price')
        
        # Find signal indices
        buy_indices = np.where(signals == 1)[0]
        sell_indices = np.where(signals == -1)[0]
        
        # Plot buy signals
        if len(buy_indices) > 0:
            ax1.scatter(
                x[buy_indices], prices[buy_indices],
                marker='^', color=self.colors['buy'], s=100, 
                label='Buy Signal', zorder=5
            )
        
        # Plot sell signals
        if len(sell_indices) > 0:
            ax1.scatter(
                x[sell_indices], prices[sell_indices],
                marker='v', color=self.colors['sell'], s=100, 
                label='Sell Signal', zorder=5
            )
        
        # Set title and labels for price chart
        if title:
            ax1.set_title(title)
        else:
            ax1.set_title('Trading Signals')
        
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot positions if provided
        if positions is not None:
            # Plot position sizes
            ax2.bar(x, positions, color=np.where(positions > 0, self.colors['buy'], self.colors['sell']), 
                   alpha=0.7, label='Position')
            
            # Set labels for position chart
            ax2.set_ylabel('Position Size')
            ax2.set_xlabel('Time' if timestamps is None else 'Date')
            ax2.grid(True, alpha=0.3)
            
            # Add zero line
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        else:
            ax1.set_xlabel('Time' if timestamps is None else 'Date')
        
        # Format x-axis for timestamps
        if timestamps is not None and isinstance(timestamps[0], (pd.Timestamp, datetime)):
            if positions is not None:
                ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
            else:
                ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                ax1.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots and save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.output_dir,
                f'trading_signals_{timestamp}.png'
            )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close() 