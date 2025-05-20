#!/usr/bin/env python
"""
Evaluation Metrics for Probabilistic Predictions

This module provides functions for evaluating probabilistic predictions,
including both point prediction metrics and probabilistic scoring rules.
"""

import numpy as np
from scipy import stats
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluator:
    """
    Class for evaluating probabilistic predictions with various metrics.
    
    Features:
    - Point prediction metrics (MSE, RMSE, MAE, R^2)
    - Probabilistic metrics (NLL, CRPS, calibration error)
    - Interval coverage analysis
    - Horizon-specific metrics
    """
    
    def __init__(self):
        """Initialize model evaluator."""
        pass
        
    def evaluate_point_predictions(self, targets, predictions):
        """
        Evaluate point predictions with standard regression metrics.
        
        Args:
            targets (array-like): Actual observed values
            predictions (array-like): Predicted values (point estimates)
            
        Returns:
            dict: Dictionary of point prediction metrics
        """
        # Ensure arrays
        targets = np.array(targets)
        predictions = np.array(predictions)
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # Direction accuracy
        directions_actual = np.sign(np.diff(targets))
        directions_pred = np.sign(np.diff(predictions))
        direction_accuracy = np.mean(directions_actual == directions_pred)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "direction_accuracy": direction_accuracy
        }
    
    def negative_log_likelihood(self, targets, pred_means, pred_stds):
        """
        Calculate negative log likelihood for Gaussian predictions.
        
        Args:
            targets (array-like): Actual observed values
            pred_means (array-like): Predicted mean values
            pred_stds (array-like): Predicted standard deviations
            
        Returns:
            float: Mean negative log likelihood
        """
        # Ensure arrays
        targets = np.array(targets)
        pred_means = np.array(pred_means)
        pred_stds = np.array(pred_stds)
        
        # Ensure positive std
        pred_stds = np.maximum(pred_stds, 1e-8)
        
        # Calculate negative log likelihood for each prediction
        nll = -stats.norm.logpdf(targets, loc=pred_means, scale=pred_stds)
        
        # Return mean NLL
        return np.mean(nll)
    
    def continuous_ranked_probability_score(self, targets, pred_means, pred_stds):
        """
        Calculate Continuous Ranked Probability Score (CRPS) for Gaussian predictions.
        
        Args:
            targets (array-like): Actual observed values
            pred_means (array-like): Predicted mean values
            pred_stds (array-like): Predicted standard deviations
            
        Returns:
            float: Mean CRPS
        """
        # Ensure arrays
        targets = np.array(targets)
        pred_means = np.array(pred_means)
        pred_stds = np.array(pred_stds)
        
        # Ensure positive std
        pred_stds = np.maximum(pred_stds, 1e-8)
        
        # Calculate standardized error
        z = (targets - pred_means) / pred_stds
        
        # Calculate CRPS for Gaussian distribution using the closed-form solution
        crps = pred_stds * (z * (2 * stats.norm.cdf(z) - 1) + 
                            2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
        
        # Return mean CRPS
        return np.mean(crps)
    
    def calibration_error(self, targets, pred_means, pred_stds, bins=10):
        """
        Calculate calibration error for probabilistic predictions.
        
        Args:
            targets (array-like): Actual observed values
            pred_means (array-like): Predicted mean values
            pred_stds (array-like): Predicted standard deviations
            bins (int, optional): Number of bins for calibration. Defaults to 10.
            
        Returns:
            float: Mean calibration error
        """
        # Ensure arrays
        targets = np.array(targets)
        pred_means = np.array(pred_means)
        pred_stds = np.array(pred_stds)
        
        # Calculate predicted quantiles
        pred_quantiles = []
        
        for target, mean, std in zip(targets, pred_means, pred_stds):
            # Convert to standard normal CDF
            std = max(std, 1e-8)  # Ensure positive std
            z_score = (target - mean) / std
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
        
        return calibration_error
    
    def interval_coverage(self, targets, pred_means, pred_stds):
        """
        Calculate interval coverage for different confidence levels.
        
        Args:
            targets (array-like): Actual observed values
            pred_means (array-like): Predicted mean values
            pred_stds (array-like): Predicted standard deviations
            
        Returns:
            dict: Coverage statistics for different intervals
        """
        # Ensure arrays
        targets = np.array(targets)
        pred_means = np.array(pred_means)
        pred_stds = np.array(pred_stds)
        
        # Calculate coverage for different confidence intervals
        intervals = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
        z_values = [stats.norm.ppf((1 + interval) / 2) for interval in intervals]
        
        coverage_stats = {}
        
        for interval, z in zip(intervals, z_values):
            # Calculate bounds
            upper_bounds = pred_means + z * pred_stds
            lower_bounds = pred_means - z * pred_stds
            
            # Count targets within bounds
            within_bounds = np.sum((targets >= lower_bounds) & (targets <= upper_bounds))
            coverage = within_bounds / len(targets)
            coverage_stats[f"{interval*100:.0f}%"] = coverage
            
            # Calculate error ratio (1.0 is perfect calibration)
            ratio = coverage / interval
            coverage_stats[f"{interval*100:.0f}%_ratio"] = ratio
        
        return coverage_stats
    
    def evaluate_horizon_metrics(self, all_targets, all_pred_means, all_pred_stds, horizons):
        """
        Calculate metrics for different prediction horizons.
        
        Args:
            all_targets (dict): Dictionary of targets for each horizon
            all_pred_means (dict): Dictionary of predicted means for each horizon
            all_pred_stds (dict): Dictionary of predicted standard deviations for each horizon
            horizons (list): List of horizon values to evaluate
            
        Returns:
            dict: Dictionary of metrics for each horizon
        """
        horizon_metrics = {}
        
        for horizon in horizons:
            if horizon in all_targets and horizon in all_pred_means and horizon in all_pred_stds:
                targets = all_targets[horizon]
                pred_means = all_pred_means[horizon]
                pred_stds = all_pred_stds[horizon]
                
                # Calculate point metrics
                point_metrics = self.evaluate_point_predictions(targets, pred_means)
                
                # Calculate probabilistic metrics
                nll = self.negative_log_likelihood(targets, pred_means, pred_stds)
                crps = self.continuous_ranked_probability_score(targets, pred_means, pred_stds)
                cal_error = self.calibration_error(targets, pred_means, pred_stds)
                intervals = self.interval_coverage(targets, pred_means, pred_stds)
                
                # Store metrics
                horizon_metrics[horizon] = {
                    **point_metrics,
                    "nll": nll,
                    "crps": crps,
                    "calibration_error": cal_error,
                    "interval_coverage": intervals
                }
                
        return horizon_metrics
    
    def evaluate_model(self, targets, pred_means, pred_stds, prediction_samples=None, horizons=None):
        """
        Comprehensive model evaluation with all metrics.
        
        Args:
            targets (array-like or dict): Actual observed values. If dict, contains values for each horizon.
            pred_means (array-like or dict): Predicted mean values. If dict, contains values for each horizon.
            pred_stds (array-like or dict): Predicted standard deviations. If dict, contains values for each horizon.
            prediction_samples (array-like, optional): Samples from the predictive distribution. Defaults to None.
            horizons (list, optional): List of horizons to evaluate. Required if inputs are dictionaries.
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        # Check if inputs are dictionaries (horizon-specific)
        is_dict_input = isinstance(targets, dict) and isinstance(pred_means, dict) and isinstance(pred_stds, dict)
        
        if is_dict_input:
            if horizons is None:
                # Try to infer horizons from keys
                horizons = sorted(list(set(targets.keys()) & set(pred_means.keys()) & set(pred_stds.keys())))
                
            if not horizons:
                raise ValueError("No matching horizons found in inputs")
                
            # Evaluate each horizon
            horizon_metrics = self.evaluate_horizon_metrics(targets, pred_means, pred_stds, horizons)
            
            # Calculate aggregate metrics across all horizons
            all_targets = []
            all_pred_means = []
            all_pred_stds = []
            
            for horizon in horizons:
                if horizon in targets and horizon in pred_means and horizon in pred_stds:
                    all_targets.extend(targets[horizon])
                    all_pred_means.extend(pred_means[horizon])
                    all_pred_stds.extend(pred_stds[horizon])
                    
            # Calculate overall metrics
            overall_point_metrics = self.evaluate_point_predictions(all_targets, all_pred_means)
            overall_nll = self.negative_log_likelihood(all_targets, all_pred_means, all_pred_stds)
            overall_crps = self.continuous_ranked_probability_score(all_targets, all_pred_means, all_pred_stds)
            overall_cal_error = self.calibration_error(all_targets, all_pred_means, all_pred_stds)
            overall_intervals = self.interval_coverage(all_targets, all_pred_means, all_pred_stds)
            
            return {
                "overall": {
                    **overall_point_metrics,
                    "nll": overall_nll,
                    "crps": overall_crps,
                    "calibration_error": overall_cal_error,
                    "interval_coverage": overall_intervals
                },
                "horizons": horizon_metrics
            }
        else:
            # Single set of predictions (no horizon differentiation)
            point_metrics = self.evaluate_point_predictions(targets, pred_means)
            nll = self.negative_log_likelihood(targets, pred_means, pred_stds)
            crps = self.continuous_ranked_probability_score(targets, pred_means, pred_stds)
            cal_error = self.calibration_error(targets, pred_means, pred_stds)
            intervals = self.interval_coverage(targets, pred_means, pred_stds)
            
            return {
                **point_metrics,
                "nll": nll,
                "crps": crps,
                "calibration_error": cal_error,
                "interval_coverage": intervals
            }
    
    def format_metrics(self, metrics, include_horizons=True, precision=4):
        """
        Format metrics as a pretty string for printing or logging.
        
        Args:
            metrics (dict): Metrics dictionary from evaluate_model
            include_horizons (bool, optional): Whether to include horizon-specific metrics. Defaults to True.
            precision (int, optional): Decimal precision for formatting. Defaults to 4.
            
        Returns:
            str: Formatted metrics string
        """
        lines = []
        
        # Format number with given precision
        def fmt(x):
            return f"{x:.{precision}f}"
        
        if "overall" in metrics:
            # We have horizon-specific metrics
            lines.append("=== OVERALL METRICS ===")
            overall = metrics["overall"]
            
            # Point metrics
            lines.append(f"RMSE: {fmt(overall['rmse'])}, MAE: {fmt(overall['mae'])}, R²: {fmt(overall['r2'])}")
            lines.append(f"MAPE: {fmt(overall['mape'])}%, Direction Accuracy: {fmt(overall['direction_accuracy'] * 100)}%")
            
            # Probabilistic metrics
            lines.append(f"NLL: {fmt(overall['nll'])}, CRPS: {fmt(overall['crps'])}, Calibration Error: {fmt(overall['calibration_error'])}")
            
            # Interval coverage
            lines.append("Interval Coverage:")
            for interval, coverage in overall['interval_coverage'].items():
                if not interval.endswith("_ratio"):
                    ratio = overall['interval_coverage'].get(f"{interval}_ratio", 0)
                    lines.append(f"  {interval}: {fmt(coverage * 100)}% (Ratio: {fmt(ratio)})")
            
            if include_horizons and "horizons" in metrics:
                # Add horizon-specific metrics
                lines.append("\n=== HORIZON-SPECIFIC METRICS ===")
                
                for horizon, horizon_metrics in metrics["horizons"].items():
                    lines.append(f"\nHorizon: {horizon}")
                    lines.append(f"RMSE: {fmt(horizon_metrics['rmse'])}, MAE: {fmt(horizon_metrics['mae'])}, R²: {fmt(horizon_metrics['r2'])}")
                    lines.append(f"NLL: {fmt(horizon_metrics['nll'])}, CRPS: {fmt(horizon_metrics['crps'])}")
                    lines.append(f"Calibration Error: {fmt(horizon_metrics['calibration_error'])}")
                    
                    # Key interval coverages
                    for key in ["95%", "68%"]:
                        if key in horizon_metrics['interval_coverage']:
                            coverage = horizon_metrics['interval_coverage'][key]
                            ratio = horizon_metrics['interval_coverage'].get(f"{key}_ratio", 0)
                            lines.append(f"{key} Coverage: {fmt(coverage * 100)}% (Ratio: {fmt(ratio)})")
        else:
            # Single set of metrics
            lines.append("=== MODEL METRICS ===")
            
            # Point metrics
            lines.append(f"RMSE: {fmt(metrics['rmse'])}, MAE: {fmt(metrics['mae'])}, R²: {fmt(metrics['r2'])}")
            lines.append(f"MAPE: {fmt(metrics['mape'])}%, Direction Accuracy: {fmt(metrics['direction_accuracy'] * 100)}%")
            
            # Probabilistic metrics
            lines.append(f"NLL: {fmt(metrics['nll'])}, CRPS: {fmt(metrics['crps'])}, Calibration Error: {fmt(metrics['calibration_error'])}")
            
            # Interval coverage
            lines.append("Interval Coverage:")
            for interval, coverage in metrics['interval_coverage'].items():
                if not interval.endswith("_ratio"):
                    ratio = metrics['interval_coverage'].get(f"{interval}_ratio", 0)
                    lines.append(f"  {interval}: {fmt(coverage * 100)}% (Ratio: {fmt(ratio)})")
        
        return "\n".join(lines) 