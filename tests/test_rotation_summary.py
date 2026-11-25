"""Validate experiment aggregation without requiring NumPy or Matplotlib."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_rotation_error import aggregate


class RotationSummary(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for mode in ("direct", "rotated"):
            for seed, value in enumerate((1, 100, 4)):
                self.rows.append(dict(mode=mode, kernel="blocked", tile="32", np_metrics_agree="1",
                    y_match_scalar="1", scale_precision="float", K="256", N="256", M="8", a="10",
                    data_seed=str(seed), rel_frob_err=str(value), max_abs_err=str(10 * value)))

    def test_median_excludes_other_kernels(self):
        extra = [dict(self.rows[0], kernel="scalar", rel_frob_err="999"),
                 dict(self.rows[0], tile="16", rel_frob_err="999")]
        medians = aggregate(self.rows + extra)
        self.assertEqual(len(medians), 2)
        self.assertEqual(medians[0]["rel_frob_err"], 4)
        self.assertEqual(medians[0]["max_abs_err"], 40)
        self.assertEqual(medians[0]["data_seeds"], "0;1;2")

    def test_missing_duplicate_and_unverified_data(self):
        for bad in (self.rows[:-1], self.rows + [self.rows[0]],
                    [dict(row, np_metrics_agree="0") for row in self.rows],
                    [dict(row, rel_frob_err="nan") for row in self.rows]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    aggregate(bad)


if __name__ == "__main__":
    unittest.main()
