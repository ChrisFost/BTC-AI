"""
Model evaluation script for probabilistic trading models.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime
import os
import json

class ModelEvaluator:
    """Class for evaluating probabilistic model performance"""
    
    def __init__(
        self,
        log_dir: str = 'evaluation_logs',
        confidence_levels: List[float] = [0.5, 0.8, 0.9, 0.95, 0.99],
        save_plots: bool = True
    ):
        self.log_dir = log_dir
        self.confidence_levels = confidence_levels
        self.save_plots = save_plots
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_file = os.path.join(
            self.log_dir,
            f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def evaluate_point_predictions(
        self,
        targets: np.ndarray,
        pred_means: np.ndarray,
        horizon_name: str = ''
    ) -> Dict[str, float]:
        """
        Evaluate point prediction metrics
        
        Args:
            targets: True values
            pred_means: Predicted means
            horizon_name: Name of prediction horizon
            
        Returns:
            Dictionary of metrics
        """
        prefix = f"{horizon_name}_" if horizon_name else ""
        
        metrics = {
            f"{prefix}mse": mean_squared_error(targets, pred_means),
            f"{prefix}rmse": np.sqrt(mean_squared_error(targets, pred_means)),
            f"{prefix}mae": mean_absolute_error(targets, pred_means),
            f"{prefix}r2": r2_score(targets, pred_means)
        }
        
        return metrics
    
    def evaluate_probabilistic_metrics(
        self,
        targets: np.ndarray,
        pred_means: np.ndarray,
        pred_stds: np.ndarray,
        samples: np.ndarray,
        horizon_name: str = ''
    ) -> Dict[str, float]:
        """
        Evaluate probabilistic prediction metrics
        
        Args:
            targets: True values
            pred_means: Predicted means
            pred_stds: Predicted standard deviations
            samples: Sampled predictions
            horizon_name: Name of prediction horizon
            
        Returns:
            Dictionary of metrics
        """
        prefix = f"{horizon_name}_" if horizon_name else ""
        
        # Negative log likelihood
        nll = -np.mean(
            -0.5 * np.log(2 * np.pi * pred_stds**2) - 
            0.5 * ((targets - pred_means) / pred_stds)**2
        )
        
        # Continuous Ranked Probability Score (CRPS)
        crps = np.mean(np.abs(samples - targets[:, np.newaxis]))
        
        # Calibration error
        calibration_error = self.compute_calibration_error(targets, pred_means, pred_stds)
        
        metrics = {
            f"{prefix}nll": nll,
            f"{prefix}crps": crps,
            f"{prefix}calibration_error": calibration_error
        }
        
        return metrics
    
    def compute_calibration_error(
        self,
        targets: np.ndarray,
        pred_means: np.ndarray,
        pred_stds: np.ndarray
    ) -> float:
        """
        Compute calibration error
        
        Args:
            targets: True values
            pred_means: Predicted means
            pred_stds: Predicted standard deviations
            
        Returns:
            Calibration error
        """
        errors = []
        
        for conf_level in self.confidence_levels:
            z_score = np.sqrt(2) * torch.erfinv(torch.tensor(conf_level)).item()
            intervals = pred_means + z_score * pred_stds
            coverage = np.mean(targets <= intervals)
            error = np.abs(coverage - conf_level)
            errors.append(error)
        
        return np.mean(errors)
    
    def evaluate_interval_coverage(
        self,
        targets: np.ndarray,
        pred_means: np.ndarray,
        pred_stds: np.ndarray,
        horizon_name: str = ''
    ) -> Dict[str, float]:
        """
        Evaluate prediction interval coverage
        
        Args:
            targets: True values
            pred_means: Predicted means
            pred_stds: Predicted standard deviations
            horizon_name: Name of prediction horizon
            
        Returns:
            Dictionary of coverage metrics
        """
        prefix = f"{horizon_name}_" if horizon_name else ""
        metrics = {}
        
        for conf_level in self.confidence_levels:
            z_score = np.sqrt(2) * torch.erfinv(torch.tensor(conf_level)).item()
            lower = pred_means - z_score * pred_stds
            upper = pred_means + z_score * pred_stds
            
            coverage = np.mean((targets >= lower) & (targets <= upper))
            metrics[f"{prefix}coverage_{conf_level:.2f}"] = coverage
        
        return metrics
    
    def plot_calibration_curve(
        self,
        targets: np.ndarray,
        pred_means: np.ndarray,
        pred_stds: np.ndarray,
        horizon_name: str = ''
    ):
        """
        Plot calibration curve
        
        Args:
            targets: True values
            pred_means: Predicted means
            pred_stds: Predicted standard deviations
            horizon_name: Name of prediction horizon
        """
        confidence_levels = np.linspace(0.01, 0.99, 50)
        observed_frequencies = []
        
        for conf_level in confidence_levels:
            z_score = np.sqrt(2) * torch.erfinv(torch.tensor(conf_level)).item()
            intervals = pred_means + z_score * pred_stds
            coverage = np.mean(targets <= intervals)
            observed_frequencies.append(coverage)
        
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.plot(confidence_levels, observed_frequencies, label='Model')
        plt.xlabel('Predicted probability')
        plt.ylabel('Observed frequency')
        plt.title(f'Calibration Curve{" - " + horizon_name if horizon_name else ""}')
        plt.legend()
        plt.grid(True)
        
        if self.save_plots:
            plt.savefig(os.path.join(
                self.log_dir,
                f'calibration_curve_{horizon_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            ))
            plt.close()
        else:
            plt.show()
    
    def plot_prediction_intervals(
        self,
        targets: np.ndarray,
        pred_means: np.ndarray,
        pred_stds: np.ndarray,
        horizon_name: str = '',
        num_points: int = 100
    ):
        """
        Plot prediction intervals
        
        Args:
            targets: True values
            pred_means: Predicted means
            pred_stds: Predicted standard deviations
            horizon_name: Name of prediction horizon
            num_points: Number of points to plot
        """
        # Sample points if too many
        if len(targets) > num_points:
            idx = np.linspace(0, len(targets)-1, num_points, dtype=int)
            targets = targets[idx]
            pred_means = pred_means[idx]
            pred_stds = pred_stds[idx]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(targets))
        
        # Plot prediction intervals
        for conf_level in self.confidence_levels:
            z_score = np.sqrt(2) * torch.erfinv(torch.tensor(conf_level)).item()
            plt.fill_between(
                x,
                pred_means - z_score * pred_stds,
                pred_means + z_score * pred_stds,
                alpha=0.1,
                label=f'{conf_level:.0%} CI'
            )
        
        plt.plot(x, targets, 'k.', label='Actual', markersize=10)
        plt.plot(x, pred_means, 'r-', label='Prediction', linewidth=2)
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Prediction Intervals{" - " + horizon_name if horizon_name else ""}')
        plt.legend()
        plt.grid(True)
        
        if self.save_plots:
            plt.savefig(os.path.join(
                self.log_dir,
                f'prediction_intervals_{horizon_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            ))
            plt.close()
        else:
            plt.show()
    
    def evaluate_model(
        self,
        targets: Dict[str, np.ndarray],
        pred_means: Dict[str, np.ndarray],
        pred_stds: Dict[str, np.ndarray],
        samples: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance for all horizons
        
        Args:
            targets: Dictionary of true values for each horizon
            pred_means: Dictionary of predicted means for each horizon
            pred_stds: Dictionary of predicted standard deviations for each horizon
            samples: Dictionary of sampled predictions for each horizon
            
        Returns:
            Dictionary of metrics for each horizon
        """
        all_metrics = {}
        
        for horizon in targets.keys():
            # Point prediction metrics
            metrics = self.evaluate_point_predictions(
                targets[horizon],
                pred_means[horizon],
                horizon
            )
            
            # Probabilistic metrics
            prob_metrics = self.evaluate_probabilistic_metrics(
                targets[horizon],
                pred_means[horizon],
                pred_stds[horizon],
                samples[horizon],
                horizon
            )
            metrics.update(prob_metrics)
            
            # Interval coverage
            coverage_metrics = self.evaluate_interval_coverage(
                targets[horizon],
                pred_means[horizon],
                pred_stds[horizon],
                horizon
            )
            metrics.update(coverage_metrics)
            
            # Plot calibration curve and prediction intervals
            self.plot_calibration_curve(
                targets[horizon],
                pred_means[horizon],
                pred_stds[horizon],
                horizon
            )
            
            self.plot_prediction_intervals(
                targets[horizon],
                pred_means[horizon],
                pred_stds[horizon],
                horizon
            )
            
            all_metrics[horizon] = metrics
            
            # Log metrics
            logging.info(f"Metrics for horizon {horizon}:")
            for metric, value in metrics.items():
                logging.info(f"  {metric}: {value:.4f}")
        
        # Save metrics to file
        metrics_file = os.path.join(
            self.log_dir,
            f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        return all_metrics 