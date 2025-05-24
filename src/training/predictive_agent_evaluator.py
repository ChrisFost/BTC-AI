#!/usr/bin/env python
"""
Enhanced Predictive Agent Evaluator

This module provides comprehensive evaluation for predictive agents focusing on:
- Price range predictions instead of single point predictions
- Bucket-specific time horizon goals
- Prediction accuracy within tolerance ranges
- Positive reinforcement for appropriate horizon selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

class PredictiveAgentEvaluator:
    """
    Enhanced evaluator for predictive agents with focus on range predictions
    and bucket-specific performance metrics.
    """
    
    def __init__(self, bucket_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the predictive agent evaluator.
        
        Args:
            bucket_type: Bucket type (Scalping, Short, Medium, Long)
            config: Configuration parameters
        """
        self.bucket_type = bucket_type
        self.config = config or {}
        
        # Define bucket-specific time horizon preferences
        self.bucket_horizons = {
            "Scalping": {
                "min_horizon": 6,  # 30 minutes (6 * 5min bars)
                "max_horizon": 2016,  # 1 week (7*24*12 bars)
                "preferred_range": (6, 288),  # 30min to 1 day
                "ideal_range": (12, 144),  # 1 hour to 12 hours
            },
            "Short": {
                "min_horizon": 144,  # 12 hours (144 * 5min bars)
                "max_horizon": 8640,  # 1 month (30*24*12 bars)
                "preferred_range": (288, 4320),  # 1 day to 15 days
                "ideal_range": (864, 2592),  # 3 days to 9 days
            },
            "Medium": {
                "min_horizon": 288,  # 1 day
                "max_horizon": 51840,  # 6 months (180*24*12 bars)
                "preferred_range": (1440, 25920),  # 5 days to 3 months
                "ideal_range": (4320, 17280),  # 15 days to 2 months
            },
            "Long": {
                "min_horizon": 8640,  # 1 month
                "max_horizon": 207360,  # 2 years (730*24*12 bars)
                "preferred_range": (17280, 103680),  # 2 months to 1 year
                "ideal_range": (34560, 69120),  # 4 months to 8 months
            }
        }
        
        # Prediction accuracy tolerance levels
        self.accuracy_tolerances = {
            "excellent": 0.015,  # Within 1.5% of actual price
            "good": 0.05,        # Within 5% of actual price
            "acceptable": 0.10,  # Within 10% of actual price
            "poor": 0.15,        # Within 15% of actual price
        }
        
        # Initialize storage for evaluation results
        self.evaluation_history = []
        self.prediction_records = []
        
    def evaluate_prediction_quality(self, 
                                  predictions: Dict[str, Any], 
                                  actual_outcomes: Dict[str, Any],
                                  current_price: float) -> Dict[str, Any]:
        """
        Evaluate the quality of price range predictions.
        
        Args:
            predictions: Dictionary containing prediction data with ranges
            actual_outcomes: Dictionary containing actual market outcomes
            current_price: Current market price when prediction was made
            
        Returns:
            Dictionary containing detailed evaluation metrics
        """
        evaluation = {
            "timestamp": datetime.now(),
            "bucket_type": self.bucket_type,
            "current_price": current_price,
            "predictions": predictions,
            "actual_outcomes": actual_outcomes,
            "quality_metrics": {},
            "horizon_metrics": {},
            "range_accuracy": {},
            "confidence_calibration": {}
        }
        
        # Evaluate each horizon prediction
        for horizon_name, pred_data in predictions.items():
            if horizon_name not in actual_outcomes:
                continue
                
            actual_price = actual_outcomes[horizon_name].get("price", current_price)
            horizon_steps = int(horizon_name.replace('h', ''))
            
            # Extract prediction range data
            pred_mean = pred_data.get("mean_price", current_price)
            pred_low = pred_data.get("lower_bound", pred_mean * 0.95)
            pred_high = pred_data.get("upper_bound", pred_mean * 1.05)
            confidence = pred_data.get("confidence", 0.5)
            
            # Calculate range accuracy
            range_accuracy = self._calculate_range_accuracy(
                actual_price, pred_low, pred_high, pred_mean, current_price
            )
            
            # Calculate horizon appropriateness
            horizon_score = self._evaluate_horizon_appropriateness(horizon_steps)
            
            # Store horizon-specific metrics
            evaluation["horizon_metrics"][horizon_name] = {
                "horizon_steps": horizon_steps,
                "range_accuracy": range_accuracy,
                "horizon_score": horizon_score,
                "confidence": confidence,
                "prediction_range_width": abs(pred_high - pred_low),
                "actual_vs_predicted_error": abs(actual_price - pred_mean) / current_price
            }
        
        # Calculate overall quality metrics
        evaluation["quality_metrics"] = self._calculate_overall_quality(
            evaluation["horizon_metrics"]
        )
        
        # Store evaluation record
        self.evaluation_history.append(evaluation)
        
        return evaluation
    
    def _calculate_range_accuracy(self, actual_price: float, pred_low: float, 
                                pred_high: float, pred_mean: float, 
                                base_price: float) -> Dict[str, Any]:
        """
        Calculate how accurate the prediction range was.
        
        Args:
            actual_price: Actual market price at horizon
            pred_low: Predicted lower bound
            pred_high: Predicted upper bound  
            pred_mean: Predicted mean price
            base_price: Price when prediction was made
            
        Returns:
            Dictionary with range accuracy metrics
        """
        # Check if actual price falls within predicted range
        within_range = pred_low <= actual_price <= pred_high
        
        # Calculate distance from range if outside
        if actual_price < pred_low:
            range_miss_distance = (pred_low - actual_price) / base_price
        elif actual_price > pred_high:
            range_miss_distance = (actual_price - pred_high) / base_price
        else:
            range_miss_distance = 0.0
        
        # Calculate point prediction accuracy
        point_error = abs(actual_price - pred_mean) / base_price
        
        # Determine accuracy category
        accuracy_category = "poor"
        for category, threshold in self.accuracy_tolerances.items():
            if point_error <= threshold:
                accuracy_category = category
                break
        
        return {
            "within_range": within_range,
            "range_miss_distance": range_miss_distance,
            "point_error_pct": point_error,
            "accuracy_category": accuracy_category,
            "range_width_pct": abs(pred_high - pred_low) / base_price
        }
    
    def _evaluate_horizon_appropriateness(self, horizon_steps: int) -> Dict[str, Any]:
        """
        Evaluate how appropriate the horizon length is for the bucket type.
        
        Args:
            horizon_steps: Number of time steps in the horizon
            
        Returns:
            Dictionary with horizon appropriateness metrics
        """
        bucket_config = self.bucket_horizons.get(self.bucket_type, {})
        ideal_range = bucket_config.get("ideal_range", (12, 144))
        preferred_range = bucket_config.get("preferred_range", (6, 288))
        
        # Calculate appropriateness score
        if ideal_range[0] <= horizon_steps <= ideal_range[1]:
            appropriateness_score = 1.0
            category = "ideal"
        elif preferred_range[0] <= horizon_steps <= preferred_range[1]:
            # Linear scaling within preferred range
            if horizon_steps < ideal_range[0]:
                appropriateness_score = 0.7 + 0.3 * (horizon_steps - preferred_range[0]) / (ideal_range[0] - preferred_range[0])
            else:
                appropriateness_score = 0.7 + 0.3 * (preferred_range[1] - horizon_steps) / (preferred_range[1] - ideal_range[1])
            category = "preferred"
        else:
            # Outside preferred range - lower score
            if horizon_steps < preferred_range[0]:
                appropriateness_score = max(0.1, 0.5 * horizon_steps / preferred_range[0])
                category = "too_short"
            else:
                appropriateness_score = max(0.1, 0.5 * preferred_range[1] / horizon_steps)
                category = "too_long"
        
        return {
            "appropriateness_score": appropriateness_score,
            "category": category,
            "ideal_range": ideal_range,
            "preferred_range": preferred_range
        }
    
    def _calculate_overall_quality(self, horizon_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall prediction quality metrics across all horizons.
        
        Args:
            horizon_metrics: Metrics for each individual horizon
            
        Returns:
            Dictionary with overall quality metrics
        """
        if not horizon_metrics:
            return {"overall_score": 0.0, "num_predictions": 0}
        
        # Collect metrics
        range_accuracies = []
        horizon_scores = []
        within_range_count = 0
        accuracy_categories = defaultdict(int)
        
        for horizon_name, metrics in horizon_metrics.items():
            range_acc = metrics["range_accuracy"]
            horizon_score = metrics["horizon_score"]["appropriateness_score"]
            
            range_accuracies.append(1.0 if range_acc["within_range"] else max(0.0, 1.0 - range_acc["range_miss_distance"]))
            horizon_scores.append(horizon_score)
            
            if range_acc["within_range"]:
                within_range_count += 1
            
            accuracy_categories[range_acc["accuracy_category"]] += 1
        
        # Calculate overall scores
        avg_range_accuracy = np.mean(range_accuracies)
        avg_horizon_appropriateness = np.mean(horizon_scores)
        range_hit_rate = within_range_count / len(horizon_metrics)
        
        # Combined score (weighted)
        overall_score = (
            0.4 * avg_range_accuracy +
            0.3 * range_hit_rate +
            0.3 * avg_horizon_appropriateness
        )
        
        return {
            "overall_score": overall_score,
            "avg_range_accuracy": avg_range_accuracy,
            "avg_horizon_appropriateness": avg_horizon_appropriateness,
            "range_hit_rate": range_hit_rate,
            "num_predictions": len(horizon_metrics),
            "accuracy_distribution": dict(accuracy_categories)
        }
    
    def calculate_reward_adjustment(self, evaluation_results: Dict[str, Any]) -> float:
        """
        Calculate reward adjustment based on predictive performance.
        
        Args:
            evaluation_results: Results from evaluate_prediction_quality
            
        Returns:
            Float reward adjustment (positive for good predictions, negative for poor)
        """
        quality_metrics = evaluation_results.get("quality_metrics", {})
        overall_score = quality_metrics.get("overall_score", 0.0)
        
        # Positive reinforcement scaling
        if overall_score >= 0.8:
            reward_adjustment = 2.0 * (overall_score - 0.8) / 0.2  # Scale 0.8-1.0 to 0-2.0
        elif overall_score >= 0.6:
            reward_adjustment = 1.0 * (overall_score - 0.6) / 0.2  # Scale 0.6-0.8 to 0-1.0
        elif overall_score >= 0.4:
            reward_adjustment = 0.5 * (overall_score - 0.4) / 0.2  # Scale 0.4-0.6 to 0-0.5
        else:
            reward_adjustment = -0.5 * (0.4 - overall_score) / 0.4  # Negative for poor performance
        
        return reward_adjustment
    
    def get_prediction_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for improving prediction performance.
        
        Returns:
            Dictionary with actionable recommendations
        """
        if not self.evaluation_history:
            return {"recommendations": ["No evaluation history available"]}
        
        # Analyze recent performance (last 20 evaluations)
        recent_evaluations = self.evaluation_history[-20:]
        
        # Collect performance patterns
        horizon_performance = defaultdict(list)
        accuracy_trends = []
        
        for eval_result in recent_evaluations:
            quality_metrics = eval_result.get("quality_metrics", {})
            accuracy_trends.append(quality_metrics.get("overall_score", 0.0))
            
            for horizon_name, metrics in eval_result.get("horizon_metrics", {}).items():
                horizon_performance[horizon_name].append({
                    "accuracy": metrics["range_accuracy"]["within_range"],
                    "appropriateness": metrics["horizon_score"]["appropriateness_score"]
                })
        
        recommendations = []
        
        # Analyze horizon performance
        best_horizons = []
        problematic_horizons = []
        
        for horizon_name, performance_list in horizon_performance.items():
            if len(performance_list) >= 3:  # Need at least 3 samples
                avg_accuracy = np.mean([p["accuracy"] for p in performance_list])
                avg_appropriateness = np.mean([p["appropriateness"] for p in performance_list])
                
                if avg_accuracy >= 0.7 and avg_appropriateness >= 0.8:
                    best_horizons.append(horizon_name)
                elif avg_accuracy < 0.4 or avg_appropriateness < 0.5:
                    problematic_horizons.append(horizon_name)
        
        # Generate recommendations
        if best_horizons:
            recommendations.append(f"Focus more on horizons {best_horizons} - they show consistently good performance")
        
        if problematic_horizons:
            recommendations.append(f"Consider reducing emphasis on horizons {problematic_horizons} - they show poor performance")
        
        # Trend analysis
        if len(accuracy_trends) >= 10:
            recent_trend = np.polyfit(range(len(accuracy_trends)), accuracy_trends, 1)[0]
            if recent_trend > 0.01:
                recommendations.append("Prediction quality is improving - continue current approach")
            elif recent_trend < -0.01:
                recommendations.append("Prediction quality is declining - consider adjusting prediction strategy")
        
        # Overall performance recommendations
        avg_recent_score = np.mean(accuracy_trends) if accuracy_trends else 0.0
        if avg_recent_score < 0.5:
            recommendations.append("Overall prediction quality is low - consider expanding prediction ranges")
        elif avg_recent_score > 0.8:
            recommendations.append("Excellent prediction quality - consider tightening ranges for more precise predictions")
        
        return {
            "recommendations": recommendations,
            "best_performing_horizons": best_horizons,
            "problematic_horizons": problematic_horizons,
            "avg_recent_score": avg_recent_score,
            "trend": "improving" if len(accuracy_trends) >= 10 and np.polyfit(range(len(accuracy_trends)), accuracy_trends, 1)[0] > 0.01 else "stable"
        }
    
    def save_evaluation_report(self, filepath: str):
        """
        Save comprehensive evaluation report to file.
        
        Args:
            filepath: Path to save the report
        """
        report = {
            "bucket_type": self.bucket_type,
            "evaluation_summary": self.get_prediction_recommendations(),
            "evaluation_history": [
                {
                    **eval_result,
                    "timestamp": eval_result["timestamp"].isoformat()
                }
                for eval_result in self.evaluation_history
            ],
            "bucket_configuration": self.bucket_horizons.get(self.bucket_type, {}),
            "accuracy_tolerances": self.accuracy_tolerances
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {filepath}") 