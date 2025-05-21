"""Helper for bucket forecasting using DynamicHorizonPredictor."""

import torch
from typing import Dict, Optional

try:
    from src.models.dynamic_horizon_predictor import DynamicHorizonPredictor
except Exception as e:
    # Provide stub if import fails
    class DynamicHorizonPredictor(torch.nn.Module):
        def __init__(self, feature_size: int, config: Dict = None):
            super().__init__()
        def get_forecast(self, features: torch.Tensor, requested_horizon: int = None, confidence: float = 0.68):
            return {"mean": None, "low": None, "high": None, "horizon": 0}


class ForecastHelperManager:
    """Manage one helper predictor per bucket.

    The optional ``config`` passed to :py:meth:`get_helper` may contain UI
    parameters such as ``horizon_range``, ``frequency`` and ``capital_allocation``
    which will be forwarded to :class:`DynamicHorizonPredictor`.
    """

    def __init__(self):
        self.helpers: Dict[str, DynamicHorizonPredictor] = {}

    def get_helper(self, bucket: str, feature_size: int, config: Optional[Dict] = None) -> DynamicHorizonPredictor:
        """Return existing helper for bucket or create a new one."""
        if bucket not in self.helpers:
            self.helpers[bucket] = DynamicHorizonPredictor(feature_size=feature_size, config=config or {})
        return self.helpers[bucket]

    def forecast(self, bucket: str, features: torch.Tensor, requested_horizon: int = None, confidence: float = 0.68) -> Dict:
        """Get a forecast for the given bucket."""
        helper = self.helpers.get(bucket)
        if helper is None:
            raise ValueError(f"Helper for bucket {bucket} not initialized")
        return helper.get_forecast(features, requested_horizon, confidence)
