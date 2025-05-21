"""Helper for bucket forecasting using DynamicHorizonPredictor."""

import torch
from datetime import datetime
from typing import Dict, Optional

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
        ):
            return {"mean": None, "low": None, "high": None, "horizon": 0}


class ForecastHelperManager:
    """Manage one helper predictor per bucket."""

    def __init__(self):
        self.helpers: Dict[str, DynamicHorizonPredictor] = {}
        # Store a history of all forecasts produced
        self.forecast_history = []

    def get_helper(
        self, bucket: str, feature_size: int, config: Optional[Dict] = None
    ) -> DynamicHorizonPredictor:
        """Return existing helper for bucket or create a new one."""
        if bucket not in self.helpers:
            self.helpers[bucket] = DynamicHorizonPredictor(
                feature_size=feature_size, config=config or {}
            )
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
        """Get a forecast for the given bucket and record the result."""
        helper = self.helpers.get(bucket)
        if helper is None:
            raise ValueError(f"Helper for bucket {bucket} not initialized")
        history_list = backtester.forecast_history if backtester is not None else None
        result = helper.get_forecast(
            features,
            requested_horizon,
            confidence,
            timestamp=timestamp,
            history_list=history_list,
        )

        # Record forecast with timestamp for later evaluation
        self.forecast_history.append(
            {
                "timestamp": timestamp or datetime.utcnow(),
                "bucket": bucket,
                **result,
            }
        )

        return result
