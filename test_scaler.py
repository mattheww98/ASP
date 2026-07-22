import unittest

import torch

from utils import Scaler


class ScalerTest(unittest.TestCase):
    def setUp(self):
        self.data = torch.tensor(
            [
                [0.0, 2.0, 4.0],
                [1.0, 4.0, 8.0],
                [2.0, 6.0, 12.0],
                [3.0, 8.0, 16.0],
            ]
        )

    def test_all_methods_round_trip(self):
        for method in Scaler.METHODS:
            with self.subTest(method=method):
                scaler = Scaler(self.data, method=method)
                restored = scaler.unscale(scaler.scale(self.data))
                torch.testing.assert_close(restored, self.data)

    def test_minmax_scales_each_label_column(self):
        scaled = Scaler(self.data, method="min-max").scale(self.data)
        torch.testing.assert_close(torch.amin(scaled, dim=0), torch.zeros(3))
        torch.testing.assert_close(torch.amax(scaled, dim=0), torch.ones(3))

    def test_median_uses_median_of_spectrum_maxima(self):
        scaler = Scaler(self.data, method="median")
        self.assertEqual(scaler.median_max.item(), 10.0)
        torch.testing.assert_close(scaler.scale(self.data), self.data / 10.0)

    def test_log_rejects_negative_labels(self):
        with self.assertRaises(ValueError):
            Scaler(torch.tensor([[1.0, -1.0]]), method="log")

    def test_state_round_trip(self):
        for method in Scaler.METHODS:
            with self.subTest(method=method):
                scaler = Scaler(self.data, method=method)
                loaded = Scaler(torch.zeros_like(self.data))
                loaded.load_state_dict(scaler.state_dict())
                torch.testing.assert_close(loaded.scale(self.data), scaler.scale(self.data))

    def test_legacy_state_defaults_to_z_scaling(self):
        scaler = Scaler(self.data)
        legacy_state = {"means": scaler.means, "stds": scaler.stds}
        loaded = Scaler(torch.zeros_like(self.data))
        loaded.load_state_dict(legacy_state)
        self.assertEqual(loaded.method, "z")
        torch.testing.assert_close(loaded.scale(self.data), scaler.scale(self.data))


if __name__ == "__main__":
    unittest.main()
