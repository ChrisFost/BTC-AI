#!/usr/bin/env python
"""
Predictive Agent Interface

This module provides utilities for main trading agents to access
predictions and recommendations from their bucket's dedicated predictive agent.
"""

import os
import json
import time
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class PredictiveAgentInterface:
    """
    Interface for accessing predictions from bucket-specific predictive agents.
    
    This class allows main trading agents to easily retrieve predictions,
    recommendations, and market insights from their dedicated predictive agent.
    """
    
    def __init__(self, bucket_type: str, models_dir: str = None):
        """
        Initialize the predictive agent interface.
        
        Args:
            bucket_type: The bucket type (Scalping, Short, Medium, Long)
            models_dir: Directory where models are stored
        """
        self.bucket_type = bucket_type
        self.models_dir = models_dir or os.path.join(os.path.dirname(__file__), "..", "..", "Models")
        self.predictive_agent_dir = os.path.join(self.models_dir, bucket_type, "predictive_agent")
        self.predictions_file = os.path.join(self.predictive_agent_dir, f"{bucket_type.lower()}_predictions.json")
        self.summary_file = os.path.join(self.predictive_agent_dir, f"{bucket_type.lower()}_final_summary.json")
        
        # Cache for predictions to avoid frequent file reads
        self._predictions_cache = None
        self._cache_timestamp = 0
        self._cache_ttl = 30  # Cache for 30 seconds
        
    def is_predictive_agent_available(self) -> bool:
        """
        Check if a predictive agent exists for this bucket.
        
        Returns:
            bool: True if predictive agent is available, False otherwise
        """
        agent_file = os.path.join(self.predictive_agent_dir, f"{self.bucket_type.lower()}_predictive_agent.pth")
        return os.path.exists(agent_file) and os.path.exists(self.predictive_agent_dir)
    
    def get_latest_predictions(self, max_age_seconds: int = 300) -> Optional[Dict[str, Any]]:
        """
        Get the latest predictions from the predictive agent.
        
        Args:
            max_age_seconds: Maximum age of predictions to consider valid (default 5 minutes)
            
        Returns:
            Dict containing predictions or None if no valid predictions found
        """
        try:
            # Check cache first
            current_time = time.time()
            if (self._predictions_cache and 
                current_time - self._cache_timestamp < self._cache_ttl):
                return self._predictions_cache
            
            # Try to load from file
            if not os.path.exists(self.predictions_file):
                logger.debug(f"No predictions file found for {self.bucket_type} bucket")
                return None
            
            with open(self.predictions_file, 'r') as f:
                predictions = json.load(f)
            
            # Check if predictions are recent enough
            prediction_time = predictions.get('timestamp', 0)
            if current_time - prediction_time > max_age_seconds:
                logger.warning(f"Predictions for {self.bucket_type} are stale ({current_time - prediction_time:.0f}s old)")
                return None
            
            # Update cache
            self._predictions_cache = predictions
            self._cache_timestamp = current_time
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error loading predictions for {self.bucket_type}: {str(e)}")
            return None
    
    def get_recommendation(self) -> str:
        """
        Get the current recommendation from the predictive agent.
        
        Returns:
            str: Recommendation ('hold', 'caution', 'bullish', 'bearish', 'unknown')
        """
        predictions = self.get_latest_predictions()
        if not predictions:
            return 'unknown'
        
        recommendations = predictions.get('recommendations', {})
        return recommendations.get('suggested_action', 'unknown')
    
    def get_confidence_level(self) -> float:
        """
        Get the confidence level of the current predictions.
        
        Returns:
            float: Confidence level between 0.0 and 1.0, or 0.5 if unavailable
        """
        predictions = self.get_latest_predictions()
        if not predictions:
            return 0.5
        
        recommendations = predictions.get('recommendations', {})
        return recommendations.get('confidence_level', 0.5)
    
    def get_predicted_performance(self) -> float:
        """
        Get the predicted performance metric.
        
        Returns:
            float: Predicted performance value, or 0.0 if unavailable
        """
        predictions = self.get_latest_predictions()
        if not predictions:
            return 0.0
        
        return predictions.get('predicted_performance', 0.0)
    
    def get_market_sentiment(self) -> str:
        """
        Get the current market sentiment assessment.
        
        Returns:
            str: Market sentiment ('bullish', 'bearish', 'neutral', 'unknown')
        """
        predictions = self.get_latest_predictions()
        if not predictions:
            return 'unknown'
        
        return predictions.get('market_sentiment', 'neutral')
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get the training status of the predictive agent.
        
        Returns:
            Dict containing training status information
        """
        try:
            if os.path.exists(self.summary_file):
                with open(self.summary_file, 'r') as f:
                    summary = json.load(f)
                return summary
            else:
                return {"status": "not_found", "training_completed": False}
        except Exception as e:
            logger.error(f"Error reading training status for {self.bucket_type}: {str(e)}")
            return {"status": "error", "training_completed": False}
    
    def should_trust_predictions(self, min_confidence: float = 0.6) -> bool:
        """
        Determine if the predictions are reliable enough to use.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            bool: True if predictions should be trusted, False otherwise
        """
        # Check if predictive agent is available
        if not self.is_predictive_agent_available():
            return False
        
        # Check if training is completed
        status = self.get_training_status()
        if not status.get('training_completed', False):
            return False
        
        # Check prediction confidence
        confidence = self.get_confidence_level()
        if confidence < min_confidence:
            return False
        
        # Check if predictions are recent
        predictions = self.get_latest_predictions(max_age_seconds=600)  # 10 minutes
        if not predictions:
            return False
        
        return True
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all available predictions.
        
        Returns:
            Dict containing all prediction information
        """
        predictions = self.get_latest_predictions()
        training_status = self.get_training_status()
        
        return {
            "bucket_type": self.bucket_type,
            "available": self.is_predictive_agent_available(),
            "trustworthy": self.should_trust_predictions(),
            "recommendation": self.get_recommendation(),
            "confidence": self.get_confidence_level(),
            "predicted_performance": self.get_predicted_performance(),
            "market_sentiment": self.get_market_sentiment(),
            "training_completed": training_status.get('training_completed', False),
            "raw_predictions": predictions,
            "training_status": training_status
        }

def get_bucket_predictions(bucket_type: str, models_dir: str = None) -> Dict[str, Any]:
    """
    Convenience function to get predictions for a specific bucket.
    
    Args:
        bucket_type: The bucket type (Scalping, Short, Medium, Long)
        models_dir: Directory where models are stored
        
    Returns:
        Dict containing prediction summary
    """
    interface = PredictiveAgentInterface(bucket_type, models_dir)
    return interface.get_prediction_summary()

def list_available_predictive_agents(models_dir: str = None) -> List[str]:
    """
    List all bucket types that have predictive agents available.
    
    Args:
        models_dir: Directory where models are stored
        
    Returns:
        List of bucket types with available predictive agents
    """
    available_buckets = []
    buckets = ["Scalping", "Short", "Medium", "Long"]
    
    for bucket in buckets:
        interface = PredictiveAgentInterface(bucket, models_dir)
        if interface.is_predictive_agent_available():
            available_buckets.append(bucket)
    
    return available_buckets

# Example usage
if __name__ == "__main__":
    # Test the interface
    print("Testing Predictive Agent Interface...")
    
    # Check all buckets
    available = list_available_predictive_agents()
    print(f"Available predictive agents: {available}")
    
    # Test a specific bucket
    if available:
        bucket = available[0]
        predictions = get_bucket_predictions(bucket)
        print(f"\nPredictions for {bucket}:")
        for key, value in predictions.items():
            if key != 'raw_predictions':  # Skip raw data for cleaner output
                print(f"  {key}: {value}") 