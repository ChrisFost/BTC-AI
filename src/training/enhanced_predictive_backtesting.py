#!/usr/bin/env python
"""
Enhanced Predictive Agent Backtesting System

This module provides advanced backtesting specifically for predictive agents with:
- Range-based predictions instead of single point predictions
- Bucket-specific time horizon optimization
- 1/2 step recalculation evaluation
- Positive reinforcement for appropriate horizon selection
- Quality-based evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
import torch

from .predictive_agent_evaluator import PredictiveAgentEvaluator

logger = logging.getLogger(__name__)

class EnhancedPredictiveBacktester:
    """
    Advanced backtesting system for predictive agents with range predictions
    and bucket-specific performance optimization.
    """
    
    def __init__(self, 
                 bucket_type: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced predictive backtester.
        
        Args:
            bucket_type: Bucket type (Scalping, Short, Medium, Long)
            config: Configuration parameters
        """
        self.bucket_type = bucket_type
        self.config = config or {}
        
        # Initialize evaluator
        self.evaluator = PredictiveAgentEvaluator(bucket_type, config)
        
        # Tracking for 1/2 step recalculations
        self.half_step_predictions = {}
        self.half_step_evaluations = []
        
        # Results storage
        self.backtest_results = {
            "predictions_made": [],
            "evaluations": [],
            "half_step_evaluations": [],
            "horizon_performance": defaultdict(list),
            "range_accuracy_history": [],
            "reward_adjustments": []
        }
        
        # Define target horizons for each bucket (in 5-minute bars)
        self.target_horizons = self._get_target_horizons()
        
    def _get_target_horizons(self) -> Dict[str, List[int]]:
        """
        Get target prediction horizons based on your specified goals.
        
        Returns:
            Dictionary mapping horizon categories to time steps
        """
        if self.bucket_type == "Scalping":
            # 30 minutes to 1 week, ideally 1-12 hours
            return {
                "short": [6, 12, 24],          # 30min - 2hours
                "medium": [36, 72, 144],       # 3hours - 12hours  
                "long": [288, 576, 1008],      # 1day - 3.5days
                "extended": [2016]              # 1 week
            }
        elif self.bucket_type == "Short":
            # 1 week to 3 months, ideally 1-4 weeks
            return {
                "short": [2016, 4032],         # 1-2 weeks
                "medium": [6048, 8064],        # 3-4 weeks
                "long": [12096, 17280],        # 6weeks - 2months
                "extended": [25920]            # 3 months
            }
        elif self.bucket_type == "Medium":
            # 1-6 months, ideally 2-4 months
            return {
                "short": [8640, 17280],        # 1-2 months
                "medium": [25920, 34560],      # 3-4 months
                "long": [43200, 51840],        # 5-6 months
                "extended": []                 # No extended for medium
            }
        elif self.bucket_type == "Long":
            # 6 months to 2 years, ideally 6-18 months
            return {
                "short": [51840, 69120],       # 6-8 months
                "medium": [86400, 129600],     # 10-15 months
                "long": [172800, 207360],      # 20-24 months
                "extended": []                 # No extended for long
            }
        else:
            # Default fallback
            return {
                "short": [12, 36, 72],
                "medium": [144, 288, 576],
                "long": [1008, 2016],
                "extended": []
            }
    
    def run_enhanced_backtest(self,
                            data: pd.DataFrame,
                            predictive_agent,
                            num_predictions: int = 1000,
                            evaluation_frequency: int = 10) -> Dict[str, Any]:
        """
        Run enhanced backtesting focused on prediction quality.
        
        Args:
            data: Market data DataFrame
            predictive_agent: The predictive agent to test
            num_predictions: Number of predictions to make and evaluate
            evaluation_frequency: How often to evaluate predictions
            
        Returns:
            Dictionary containing comprehensive backtest results
        """
        logger.info(f"Starting enhanced predictive backtest for {self.bucket_type} bucket")
        
        # Prepare data
        data = data.reset_index(drop=True)
        start_idx = max(100, predictive_agent.model.config.get("WINDOW_SIZE", 288))
        
        predictions_made = 0
        current_idx = start_idx
        
        while predictions_made < num_predictions and current_idx < len(data) - max(self.target_horizons.get("extended", [2016])):
            # Get current market state
            current_price = data.iloc[current_idx]["close"]
            current_time = data.iloc[current_idx].get("timestamp", current_idx)
            
            # Prepare observation for agent
            obs = self._prepare_observation(data, current_idx, predictive_agent)
            
            # Generate predictions with ranges
            predictions = self._generate_range_predictions(
                predictive_agent, obs, current_price, current_time
            )
            
            # Calculate target times for 1/2 step evaluation
            half_step_targets = self._calculate_half_step_targets(predictions, current_idx)
            
            # Store prediction for later evaluation
            prediction_record = {
                "prediction_id": predictions_made,
                "timestamp": current_time,
                "current_idx": current_idx,
                "current_price": current_price,
                "predictions": predictions,
                "half_step_targets": half_step_targets,
                "evaluated": False
            }
            
            self.backtest_results["predictions_made"].append(prediction_record)
            
            # Check if we can evaluate previous predictions
            if predictions_made > 0 and predictions_made % evaluation_frequency == 0:
                self._evaluate_ready_predictions(data, current_idx)
            
            predictions_made += 1
            current_idx += self._get_prediction_step_size()
        
        # Final evaluation of remaining predictions
        self._evaluate_ready_predictions(data, len(data) - 1, final_evaluation=True)
        
        # Calculate comprehensive results
        final_results = self._calculate_final_results()
        
        logger.info(f"Enhanced predictive backtest completed. Made {predictions_made} predictions.")
        
        return final_results
    
    def _prepare_observation(self, data: pd.DataFrame, idx: int, agent) -> torch.Tensor:
        """
        Prepare observation tensor for the agent.
        
        Args:
            data: Market data DataFrame
            idx: Current index
            agent: The predictive agent
            
        Returns:
            Observation tensor
        """
        window_size = agent.model.config.get("WINDOW_SIZE", 288)
        start_idx = max(0, idx - window_size)
        
        # Get feature columns (assuming all numeric columns except timestamp)
        feature_cols = [col for col in data.columns if col not in ['timestamp'] and data[col].dtype in ['float64', 'int64']]
        
        # Extract window of data
        window_data = data.iloc[start_idx:idx][feature_cols].values
        
        # Pad if necessary
        if len(window_data) < window_size:
            padding = np.zeros((window_size - len(window_data), len(feature_cols)))
            window_data = np.vstack([padding, window_data])
        
        # Convert to tensor
        obs_tensor = torch.FloatTensor(window_data).unsqueeze(0)  # Add batch dimension
        
        return obs_tensor
    
    def _generate_range_predictions(self,
                                  agent,
                                  obs: torch.Tensor,
                                  current_price: float,
                                  current_time) -> Dict[str, Any]:
        """
        Generate range-based predictions for all target horizons.
        
        Args:
            agent: The predictive agent
            obs: Observation tensor
            current_price: Current market price
            current_time: Current timestamp
            
        Returns:
            Dictionary containing range predictions for each horizon
        """
        predictions = {}
        
        with torch.no_grad():
            # Get model outputs
            if hasattr(agent, 'predict_with_uncertainty'):
                model_outputs = agent.predict_with_uncertainty(obs)
            else:
                # Fallback to standard prediction
                model_outputs = agent.model(obs)
        
        # Generate predictions for each target horizon category
        for category, horizons in self.target_horizons.items():
            for horizon_steps in horizons:
                horizon_name = f"h{horizon_steps}"
                
                # Extract prediction components
                if isinstance(model_outputs, dict):
                    pred_mean = model_outputs.get('prediction_means', {}).get(horizon_name, 0.0)
                    pred_std = model_outputs.get('prediction_stds', {}).get(horizon_name, 0.01)
                    confidence = model_outputs.get('confidence', {}).get(horizon_name, 0.5)
                else:
                    # Fallback for simple outputs
                    pred_mean = 0.0
                    pred_std = 0.01
                    confidence = 0.5
                
                # Convert relative predictions to absolute prices
                if isinstance(pred_mean, torch.Tensor):
                    pred_mean = pred_mean.item()
                if isinstance(pred_std, torch.Tensor):
                    pred_std = pred_std.item()
                if isinstance(confidence, torch.Tensor):
                    confidence = confidence.item()
                
                # Calculate price predictions (assuming model outputs returns)
                predicted_price = current_price * (1 + pred_mean)
                price_std = current_price * abs(pred_std)
                
                # Calculate confidence intervals (multiple levels)
                predictions[horizon_name] = {
                    "horizon_steps": horizon_steps,
                    "category": category,
                    "mean_return": pred_mean,
                    "mean_price": predicted_price,
                    "std": pred_std,
                    "price_std": price_std,
                    "confidence": confidence,
                    
                    # 68% confidence interval (1 sigma)
                    "lower_bound_68": predicted_price - price_std,
                    "upper_bound_68": predicted_price + price_std,
                    
                    # 95% confidence interval (2 sigma)
                    "lower_bound_95": predicted_price - 2 * price_std,
                    "upper_bound_95": predicted_price + 2 * price_std,
                    
                    # Conservative range (3 sigma)
                    "lower_bound": predicted_price - 3 * price_std,
                    "upper_bound": predicted_price + 3 * price_std,
                    
                    "prediction_time": current_time,
                    "target_time": f"t+{horizon_steps}_steps"  # Use string representation to avoid timestamp arithmetic
                }
        
        return predictions
    
    def _calculate_half_step_targets(self, predictions: Dict[str, Any], current_idx: int) -> Dict[str, int]:
        """
        Calculate target indices for 1/2 step recalculations.
        
        Args:
            predictions: Current predictions
            current_idx: Current data index
            
        Returns:
            Dictionary mapping horizon names to half-step target indices
        """
        half_step_targets = {}
        
        for horizon_name, pred_data in predictions.items():
            horizon_steps = pred_data["horizon_steps"]
            half_step_idx = current_idx + max(1, horizon_steps // 2)
            half_step_targets[horizon_name] = half_step_idx
            
        return half_step_targets
    
    def _evaluate_ready_predictions(self, data: pd.DataFrame, current_idx: int, final_evaluation: bool = False):
        """
        Evaluate predictions that are ready for assessment.
        
        Args:
            data: Market data DataFrame
            current_idx: Current data index
            final_evaluation: Whether this is the final evaluation
        """
        for prediction_record in self.backtest_results["predictions_made"]:
            if prediction_record["evaluated"]:
                continue
            
            prediction_idx = prediction_record["current_idx"]
            predictions = prediction_record["predictions"]
            
            # Check which predictions can be evaluated
            actual_outcomes = {}
            half_step_outcomes = {}
            
            for horizon_name, pred_data in predictions.items():
                horizon_steps = pred_data["horizon_steps"]
                target_idx = prediction_idx + horizon_steps
                half_step_idx = prediction_record["half_step_targets"][horizon_name]
                
                # Evaluate full prediction if target time reached
                if target_idx <= current_idx or final_evaluation:
                    if target_idx < len(data):
                        actual_outcomes[horizon_name] = {
                            "price": data.iloc[target_idx]["close"],
                            "idx": target_idx
                        }
                
                # Evaluate half-step if ready
                if half_step_idx <= current_idx and horizon_name not in self.half_step_predictions:
                    if half_step_idx < len(data):
                        half_step_outcomes[horizon_name] = {
                            "price": data.iloc[half_step_idx]["close"],
                            "idx": half_step_idx
                        }
                        self.half_step_predictions[horizon_name] = half_step_outcomes[horizon_name]
            
            # Perform evaluations if we have outcomes
            if actual_outcomes:
                evaluation = self.evaluator.evaluate_prediction_quality(
                    predictions, actual_outcomes, prediction_record["current_price"]
                )
                evaluation["prediction_id"] = prediction_record["prediction_id"]
                self.backtest_results["evaluations"].append(evaluation)
                
                # Calculate reward adjustment
                reward_adjustment = self.evaluator.calculate_reward_adjustment(evaluation)
                self.backtest_results["reward_adjustments"].append(reward_adjustment)
                
                prediction_record["evaluated"] = True
            
            # Store half-step evaluations
            if half_step_outcomes:
                half_step_eval = self._evaluate_half_step_accuracy(
                    predictions, half_step_outcomes, prediction_record["current_price"]
                )
                half_step_eval["prediction_id"] = prediction_record["prediction_id"]
                self.backtest_results["half_step_evaluations"].append(half_step_eval)
    
    def _evaluate_half_step_accuracy(self, 
                                   original_predictions: Dict[str, Any],
                                   half_step_outcomes: Dict[str, Any],
                                   original_price: float) -> Dict[str, Any]:
        """
        Evaluate accuracy at the halfway point of predictions.
        
        Args:
            original_predictions: Original full-horizon predictions
            half_step_outcomes: Actual outcomes at half-step points
            original_price: Price when original prediction was made
            
        Returns:
            Dictionary containing half-step evaluation results
        """
        half_step_evaluation = {
            "timestamp": datetime.now(),
            "half_step_metrics": {}
        }
        
        for horizon_name, outcome in half_step_outcomes.items():
            if horizon_name in original_predictions:
                pred_data = original_predictions[horizon_name]
                actual_half_price = outcome["price"]
                
                # Calculate expected price at half-step (linear interpolation)
                full_predicted_price = pred_data["mean_price"]
                expected_half_price = (original_price + full_predicted_price) / 2
                
                # Calculate half-step accuracy
                half_step_error = abs(actual_half_price - expected_half_price) / original_price
                
                # Check if within prediction range (scaled for half-step)
                half_range_low = (pred_data["lower_bound"] + original_price) / 2
                half_range_high = (pred_data["upper_bound"] + original_price) / 2
                within_half_range = half_range_low <= actual_half_price <= half_range_high
                
                half_step_evaluation["half_step_metrics"][horizon_name] = {
                    "half_step_error_pct": half_step_error,
                    "within_half_range": within_half_range,
                    "expected_half_price": expected_half_price,
                    "actual_half_price": actual_half_price,
                    "original_prediction_on_track": half_step_error < 0.1  # Within 10%
                }
        
        return half_step_evaluation
    
    def _get_prediction_step_size(self) -> int:
        """
        Get step size between predictions based on bucket type.
        
        Returns:
            Number of steps to advance between predictions
        """
        step_sizes = {
            "Scalping": 6,    # Every 30 minutes
            "Short": 144,     # Every 12 hours  
            "Medium": 288,    # Every day
            "Long": 1440      # Every 5 days
        }
        return step_sizes.get(self.bucket_type, 144)
    
    def _calculate_final_results(self) -> Dict[str, Any]:
        """
        Calculate comprehensive final results from all evaluations.
        
        Returns:
            Dictionary containing final backtest results
        """
        if not self.backtest_results["evaluations"]:
            return {"error": "No evaluations completed"}
        
        # Aggregate evaluation metrics
        overall_scores = []
        range_hit_rates = []
        horizon_performance = defaultdict(list)
        accuracy_distributions = defaultdict(int)
        
        for evaluation in self.backtest_results["evaluations"]:
            quality_metrics = evaluation.get("quality_metrics", {})
            overall_scores.append(quality_metrics.get("overall_score", 0.0))
            range_hit_rates.append(quality_metrics.get("range_hit_rate", 0.0))
            
            # Collect per-horizon performance
            for horizon_name, metrics in evaluation.get("horizon_metrics", {}).items():
                horizon_performance[horizon_name].append({
                    "range_accuracy": metrics["range_accuracy"]["within_range"],
                    "point_error": metrics["range_accuracy"]["point_error_pct"],
                    "horizon_score": metrics["horizon_score"]["appropriateness_score"],
                    "category": metrics["horizon_score"]["category"]
                })
                
                # Count accuracy categories
                accuracy_distributions[metrics["range_accuracy"]["accuracy_category"]] += 1
        
        # Calculate summary statistics
        avg_overall_score = np.mean(overall_scores)
        avg_range_hit_rate = np.mean(range_hit_rates)
        avg_reward_adjustment = np.mean(self.backtest_results["reward_adjustments"]) if self.backtest_results["reward_adjustments"] else 0.0
        
        # Calculate horizon-specific summaries
        horizon_summaries = {}
        for horizon_name, performance_list in horizon_performance.items():
            if performance_list:
                horizon_summaries[horizon_name] = {
                    "avg_range_accuracy": np.mean([p["range_accuracy"] for p in performance_list]),
                    "avg_point_error": np.mean([p["point_error"] for p in performance_list]),
                    "avg_horizon_score": np.mean([p["horizon_score"] for p in performance_list]),
                    "num_predictions": len(performance_list),
                    "most_common_category": max(set([p["category"] for p in performance_list]), 
                                              key=[p["category"] for p in performance_list].count)
                }
        
        # Half-step analysis
        half_step_summary = self._analyze_half_step_performance()
        
        # Get recommendations
        recommendations = self.evaluator.get_prediction_recommendations()
        
        final_results = {
            "bucket_type": self.bucket_type,
            "summary": {
                "total_predictions": len(self.backtest_results["predictions_made"]),
                "total_evaluations": len(self.backtest_results["evaluations"]),
                "avg_overall_score": avg_overall_score,
                "avg_range_hit_rate": avg_range_hit_rate,
                "avg_reward_adjustment": avg_reward_adjustment,
                "accuracy_distribution": dict(accuracy_distributions)
            },
            "horizon_performance": horizon_summaries,
            "half_step_analysis": half_step_summary,
            "recommendations": recommendations,
            "target_horizons_used": self.target_horizons,
            "detailed_evaluations": self.backtest_results["evaluations"],
            "half_step_evaluations": self.backtest_results["half_step_evaluations"]
        }
        
        return final_results
    
    def _analyze_half_step_performance(self) -> Dict[str, Any]:
        """
        Analyze performance of half-step recalculations.
        
        Returns:
            Dictionary containing half-step analysis
        """
        if not self.backtest_results["half_step_evaluations"]:
            return {"message": "No half-step evaluations available"}
        
        on_track_rates = defaultdict(list)
        half_step_errors = defaultdict(list)
        
        for half_eval in self.backtest_results["half_step_evaluations"]:
            for horizon_name, metrics in half_eval.get("half_step_metrics", {}).items():
                on_track_rates[horizon_name].append(metrics["original_prediction_on_track"])
                half_step_errors[horizon_name].append(metrics["half_step_error_pct"])
        
        summary = {}
        for horizon_name in on_track_rates:
            summary[horizon_name] = {
                "on_track_rate": np.mean(on_track_rates[horizon_name]),
                "avg_half_step_error": np.mean(half_step_errors[horizon_name]),
                "num_half_step_evaluations": len(on_track_rates[horizon_name])
            }
        
        return summary
    
    def save_backtest_report(self, filepath: str):
        """
        Save comprehensive backtest report to file.
        
        Args:
            filepath: Path to save the report
        """
        final_results = self._calculate_final_results()
        
        # Convert datetime objects to strings for JSON serialization
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj
        
        serializable_results = convert_datetime(final_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Enhanced predictive backtest report saved to {filepath}")


def run_enhanced_predictive_backtest(data: pd.DataFrame,
                                   predictive_agent,
                                   bucket_type: str,
                                   config: Optional[Dict[str, Any]] = None,
                                   output_dir: str = "enhanced_backtest_results") -> Dict[str, Any]:
    """
    Run enhanced predictive backtesting for a specific bucket.
    
    Args:
        data: Market data DataFrame
        predictive_agent: The predictive agent to test
        bucket_type: Bucket type (Scalping, Short, Medium, Long)
        config: Configuration parameters
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing backtest results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize backtester
    backtester = EnhancedPredictiveBacktester(bucket_type, config)
    
    # Run backtest
    results = backtester.run_enhanced_backtest(data, predictive_agent)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"enhanced_predictive_backtest_{bucket_type}_{timestamp}.json")
    backtester.save_backtest_report(report_path)
    
    return results 