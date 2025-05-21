"""Helper for bucket forecasting using DynamicHorizonPredictor."""

import torch
from typing import Dict, Optional, List

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
            history_list: List[Dict] | None = None,
        ):
            forecast = {"mean": None, "low": None, "high": None, "horizon": 0}
            if history_list is not None:
                history_list.append(forecast)
            return forecast


class ForecastHelperManager:
    """Manage one helper predictor per bucket.

    The optional ``config`` passed to :py:meth:`get_helper` may contain UI
    parameters such as ``horizon_range``, ``frequency`` and ``capital_allocation``
    which will be forwarded to :class:`DynamicHorizonPredictor`.
    """

    def __init__(self, backtester=None):
        self.helpers: Dict[str, DynamicHorizonPredictor] = {}
        self.backtester = backtester
        # Local history of forecasts for optional analysis
        self.forecast_history = []

    def get_helper(self, bucket: str, feature_size: int, config: Optional[Dict] = None) -> DynamicHorizonPredictor:
        """Return existing helper for bucket or create a new one."""
        if bucket not in self.helpers:
            self.helpers[bucket] = DynamicHorizonPredictor(feature_size=feature_size, config=config or {})
        return self.helpers[bucket]

    def forecast(
        self,
        bucket: str,
        features: torch.Tensor,
        requested_horizon: int = None,
        confidence: float = 0.68,
        timestamp=None,
        history_list: Optional[List[Dict]] = None,
    ) -> Dict:
        """Get a forecast for the given bucket."""
        helper = self.helpers.get(bucket)
        if helper is None:
            raise ValueError(f"Helper for bucket {bucket} not initialized")
        forecast = helper.get_forecast(
            features,
            requested_horizon,
            confidence,
            history_list=history_list,
        )
        entry = {
            "timestamp": timestamp,
            "bucket": bucket,
            "forecast": forecast,
        }
        # Store locally
        self.forecast_history.append(entry)
        if self.backtester is not None:
            self.backtester.forecast_history.append(entry)
        if history_list is not None and history_list is not self.forecast_history:
            history_list.append(entry)
        return forecast
