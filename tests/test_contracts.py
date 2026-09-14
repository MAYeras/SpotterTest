import unittest
import numpy as np
import pandas as pd
from freight import features, metrics, matrix, CAT
from score import validate_predictions, validate_december


class Contracts(unittest.TestCase):
    def test_features_accept_missing_values_without_mutating_input(self):
        d = pd.DataFrame(
            dict(
                pickup=[None],
                delivery=["B"],
                equipment=["Van"],
                date=["2025-01-01"],
                distance=[10],
                weight=[np.nan],
                quote_signal=[2],
            )
        )
        result = features(d)
        self.assertEqual(result.route.iloc[0], "Missing -> B")
        self.assertEqual(result.quote_total.iloc[0], 20)
        self.assertNotIn("route", d)
        self.assertTrue(pd.isna(result.weight.iloc[0]))

    def test_unknown_categories_do_not_acquire_training_codes(self):
        d = pd.DataFrame({c: ["unseen"] for c in CAT})
        result = matrix(d, CAT, "lightgbm", {c: ["known"] for c in CAT})
        self.assertTrue(result.isna().all().all())

    def test_metrics_known_answer(self):
        result = metrics(np.array([100.0, 200.0]), np.array([90.0, 220.0]))
        self.assertEqual(result["mae"], 15.0)
        self.assertEqual(result["wape"], 0.1)
        self.assertEqual(result["bias"], 5.0)

    def test_predictions_reject_duplicate_ids_and_infinity(self):
        d = pd.DataFrame(
            {
                "load_id": [f"TE-{i:06d}" for i in range(1, 12001)],
                "predicted_rate": 100.0,
            }
        )
        validate_predictions(d)
        d.loc[0, "predicted_rate"] = np.inf
        with self.assertRaises(SystemExit):
            validate_predictions(d)
        d.loc[0, "predicted_rate"] = 100.0
        d.loc[0, "load_id"] = d.loc[1, "load_id"]
        with self.assertRaises(SystemExit):
            validate_predictions(d)

    def test_december_rejects_changed_scenario(self):
        d = pd.DataFrame(
            {
                "pickup": "Lexington",
                "delivery": "Fort Wayne",
                "distance": 360.0,
                "equipment": "Dry Van",
                "weight": 32000.0,
                "date": pd.date_range("2025-12-01", periods=31),
                "predicted_rate": 800.0,
            }
        )
        validate_december(d)
        d.loc[0, "distance"] = 100.0
        with self.assertRaises(SystemExit):
            validate_december(d)


if __name__ == "__main__":
    unittest.main()
