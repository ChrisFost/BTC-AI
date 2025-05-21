import unittest
from datetime import datetime
import torch

from src.training.forecast_helper import ForecastHelperManager
from src.training.backtesting import Backtester


class ForecastHistoryTest(unittest.TestCase):
    def test_forecast_records_used_in_backtester(self):
        helper_mgr = ForecastHelperManager()
        backtester = Backtester()

        helper_mgr.get_helper("Scalping", feature_size=1)
        ts = datetime.utcnow()
        features = torch.zeros(1, 1)
        helper_mgr.forecast(
            "Scalping",
            features,
            requested_horizon=1,
            timestamp=ts,
            backtester=backtester,
        )

        backtester.update_metrics(price=1.0, timestamp=ts, target=1.0)

        self.assertTrue(
            len(backtester.prediction_quality) > 0,
            "prediction_quality should record entry when forecast history exists",
        )


if __name__ == "__main__":
    unittest.main()
