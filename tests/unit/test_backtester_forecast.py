import os
import sys
import torch
from unittest.mock import MagicMock

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

from src.training.forecast_helper import ForecastHelperManager
from src.training.backtesting import Backtester


def test_forecast_history_updates_prediction_quality():
    backtester = Backtester()
    manager = ForecastHelperManager(backtester=backtester)

    # Stub helper to return deterministic forecast
    helper = MagicMock()
    helper.get_forecast.return_value = {
        "mean": 101.0,
        "low": 99.0,
        "high": 103.0,
        "horizon": 1,
    }
    manager.helpers["bucket"] = helper

    features = torch.tensor([[1.0]])
    ts = "2024-01-01"
    forecast = manager.forecast("bucket", features, timestamp=ts)

    assert backtester.forecast_history[0]["forecast"] == forecast

    # Update metrics using stored forecast (no prediction passed)
    backtester.update_metrics(price=100.0, timestamp=ts, target=104.0)

    assert len(backtester.prediction_quality) == 1
    record = backtester.prediction_quality[0]
    assert record["prediction"] == forecast["mean"]
    assert record["target"] == 104.0
