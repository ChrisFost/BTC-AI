"""Helper for bucket forecasting using DynamicHorizonPredictor."""

import torch
from datetime import datetime
from typing import Dict, Optional, List
import json
import os

try:
    from src.models.dynamic_horizon_predictor import DynamicHorizonPredictor
except Exception as e:
    # Provide stub if import fails
    class DynamicHorizonPredictor(torch.nn.Module):
        def __init__(self, feature_size: int, config: Dict = None):
            super().__init__()

        def get_forecast(
            self,
            features: torch.Tensor,
            requested_horizon: int = None,
            confidence: float = 0.68,
            timestamp: Optional[datetime] = None,
            history_list: Optional[List[Dict]] = None,
        ):
            return {"mean": None, "low": None, "high": None, "horizon": 0}


class ForecastHelperManager:
    """Manage one helper predictor per bucket with enhanced timestamp recording."""

    def __init__(self):
        self.helpers: Dict[str, DynamicHorizonPredictor] = {}
        # Store a history of all forecasts produced across all buckets
        self.forecast_history = []
        # Track forecasts by bucket for analysis
        self.bucket_forecasts = {}

    def get_helper(
        self, bucket: str, feature_size: int, config: Optional[Dict] = None
    ) -> DynamicHorizonPredictor:
        """Return existing helper for bucket or create a new one."""
        if bucket not in self.helpers:
            self.helpers[bucket] = DynamicHorizonPredictor(
                feature_size=feature_size, config=config or {}
            )
            self.bucket_forecasts[bucket] = []
        return self.helpers[bucket]

    def forecast(
        self,
        bucket: str,
        features: torch.Tensor,
        requested_horizon: int = None,
        confidence: float = 0.68,
        timestamp: Optional[datetime] = None,
        backtester: Optional[object] = None,
    ) -> Dict:
        """Get a forecast for the given bucket and record the result with proper timestamps."""
        helper = self.helpers.get(bucket)
        if helper is None:
            raise ValueError(f"Helper for bucket {bucket} not initialized")
        
        # Ensure timestamp is provided for consistent recording
        forecast_timestamp = timestamp or datetime.utcnow()
        
        # Get forecast history list from backtester if available
        history_list = backtester.forecast_history if backtester is not None else None
        
        # Get forecast with timestamp
        result = helper.get_forecast(
            features,
            requested_horizon,
            confidence,
            timestamp=forecast_timestamp,
            history_list=history_list,
        )

        # Record forecast with timestamp and bucket info for later evaluation
        forecast_record = {
            "timestamp": forecast_timestamp,
            "bucket": bucket,
            "requested_horizon": requested_horizon,
            "confidence": confidence,
            **result,
        }
        
        # Store in manager's history
        self.forecast_history.append(forecast_record)
        
        # Store in bucket-specific history
        if bucket not in self.bucket_forecasts:
            self.bucket_forecasts[bucket] = []
        self.bucket_forecasts[bucket].append(forecast_record)

        return result
    
    def get_bucket_forecast_history(self, bucket: str, since: Optional[datetime] = None) -> List[Dict]:
        """
        Get forecast history for a specific bucket.
        
        Args:
            bucket: Bucket name to get forecasts for
            since: Optional datetime to filter forecasts after this time
            
        Returns:
            List of forecast records for the bucket
        """
        bucket_forecasts = self.bucket_forecasts.get(bucket, [])
        
        if since is None:
            return bucket_forecasts.copy()
        
        return [
            record for record in bucket_forecasts 
            if record["timestamp"] >= since
        ]
    
    def get_all_forecast_history(self, since: Optional[datetime] = None) -> List[Dict]:
        """
        Get forecast history for all buckets.
        
        Args:
            since: Optional datetime to filter forecasts after this time
            
        Returns:
            List of forecast records from all buckets
        """
        if since is None:
            return self.forecast_history.copy()
        
        return [
            record for record in self.forecast_history 
            if record["timestamp"] >= since
        ]
    
    def save_forecast_history(self, filepath: str, bucket: Optional[str] = None):
        """
        Save forecast history to file.
        
        Args:
            filepath: Path to save the forecast history
            bucket: Optional bucket name to save only that bucket's history
        """
        if bucket is not None:
            history_to_save = self.get_bucket_forecast_history(bucket)
        else:
            history_to_save = self.forecast_history
        
        # Convert datetime objects to ISO strings for JSON serialization
        serializable_history = []
        for record in history_to_save:
            serialized_record = record.copy()
            if isinstance(serialized_record["timestamp"], datetime):
                serialized_record["timestamp"] = serialized_record["timestamp"].isoformat()
            serializable_history.append(serialized_record)
        
        # Ensure directory exists
        dir_path = os.path.dirname(filepath)
        if dir_path:  # Only create directory if there is a directory path
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def clear_forecast_history(self, bucket: Optional[str] = None):
        """
        Clear forecast history.
        
        Args:
            bucket: Optional bucket name to clear only that bucket's history
        """
        if bucket is not None:
            if bucket in self.bucket_forecasts:
                self.bucket_forecasts[bucket].clear()
            if bucket in self.helpers:
                self.helpers[bucket].clear_forecast_history()
        else:
            self.forecast_history.clear()
            self.bucket_forecasts.clear()
            for helper in self.helpers.values():
                helper.clear_forecast_history()
    
    def get_forecast_statistics(self, bucket: Optional[str] = None) -> Dict:
        """
        Get statistics about forecast history.
        
        Args:
            bucket: Optional bucket name to get stats for specific bucket
            
        Returns:
            Dictionary with forecast statistics
        """
        if bucket is not None:
            forecasts = self.get_bucket_forecast_history(bucket)
        else:
            forecasts = self.forecast_history
        
        if not forecasts:
            return {"total_forecasts": 0, "buckets": [], "time_range": None}
        
        buckets = list(set(f.get("bucket", "unknown") for f in forecasts))
        timestamps = [f["timestamp"] for f in forecasts if "timestamp" in f]
        
        time_range = None
        if timestamps:
            # Handle both datetime objects and ISO strings
            parsed_timestamps = []
            for ts in timestamps:
                if isinstance(ts, str):
                    try:
                        parsed_timestamps.append(datetime.fromisoformat(ts))
                    except:
                        continue
                elif isinstance(ts, datetime):
                    parsed_timestamps.append(ts)
            
            if parsed_timestamps:
                time_range = {
                    "earliest": min(parsed_timestamps).isoformat(),
                    "latest": max(parsed_timestamps).isoformat()
                }
        
        return {
            "total_forecasts": len(forecasts),
            "buckets": buckets,
            "time_range": time_range,
            "forecast_count_by_bucket": {
                bucket_name: len(self.get_bucket_forecast_history(bucket_name))
                for bucket_name in buckets
            }
        }
