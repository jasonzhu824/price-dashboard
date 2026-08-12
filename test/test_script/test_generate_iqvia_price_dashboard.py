"""
test_generate_iqvia_price_dashboard.py
======================================
Unit tests for generate_iqvia_price_dashboard.py.

=== Run
    python test/test_script/test_generate_iqvia_price_dashboard.py

=== Coverage
- Column rename map (standardized display names)
- Price = Value / Volume calculation (normal, zero volume, negative volume)
- Monthly summary aggregation with weighted price
- Generated flat file integrity (existence, shape, renames applied)
"""

# === Imports
import os
import sys
import unittest

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, SRC_DIR)

import generate_iqvia_price_dashboard as module  # noqa: E402


class TestRenameMap(unittest.TestCase):
    """Column standardization must map original names to display names."""

    def test_rename_columns(self):
        raw_df = pd.DataFrame({
            "VBP Type 1": ["GD+20"],
            "Comp sum CN": ["纽迪希亚"],
            "SKU CN": ["康全甘500ml"],
            "Sub Category3": ["General"],
            "Tube_ONS": ["Tube"],
            "Value": [100.0],
            "Volume": [10.0],
        })
        renamed_df = raw_df.rename(columns=module.RENAME_MAP)
        expected = ["VBP", "Company", "Product", "Product Category", "Tube-ONS"]
        for col in expected:
            self.assertIn(col, renamed_df.columns)
        self.assertNotIn("VBP Type 1", renamed_df.columns)


class TestCalcPrice(unittest.TestCase):
    """Price must equal Value / Volume with a zero-volume guard."""

    def test_normal_price(self):
        df = pd.DataFrame({"Value": [100.0, 300.0], "Volume": [10.0, 20.0]})
        result_df = module.calc_price(df)
        np.testing.assert_allclose(result_df["Price"], [10.0, 15.0])

    def test_zero_volume_gives_nan(self):
        df = pd.DataFrame({"Value": [50.0], "Volume": [0.0]})
        result_df = module.calc_price(df)
        self.assertTrue(np.isnan(result_df.loc[0, "Price"]))

    def test_negative_volume_kept_as_is(self):
        df = pd.DataFrame({"Value": [100.0], "Volume": [-5.0]})
        result_df = module.calc_price(df)
        self.assertEqual(result_df.loc[0, "Price"], -20.0)


class TestSummaryAggregation(unittest.TestCase):
    """Monthly summary must use weighted price (total Value / total Volume)."""

    def setUp(self):
        self.data_df = pd.DataFrame({
            "YM": [202301, 202301, 202302],
            "VBP": ["GD+20", "GD+20", "Independent"],
            "Province": ["广东", "广东", "北京"],
            "City": ["深圳", "深圳", "北京"],
            "Channel": ["Hospital", "Hospital", "Hospital"],
            "Company": ["纽迪希亚", "纽迪希亚", "费森尤斯卡比"],
            "Tube-ONS": ["Tube", "Tube", "Tube"],
            "Product": ["康全甘500ml", "康全甘500ml", "瑞先"],
            "Product Category": ["General", "General", "General"],
            "Value": [100.0, 200.0, 50.0],
            "Volume": [10.0, 30.0, 5.0],
        })
        # selections: dict[col] -> list; empty list means "all values"
        self.filters = {col: [] for col in module.FILTER_ORDER}

    def test_monthly_aggregation_and_weighted_price(self):
        summary_df = module.make_summary_df(self.data_df, self.filters)
        self.assertEqual(len(summary_df), 2)
        row_jan = summary_df[summary_df["YM"] == 202301].iloc[0]
        self.assertEqual(row_jan["Value"], 300.0)
        self.assertEqual(row_jan["Volume"], 40.0)
        self.assertAlmostEqual(row_jan["Price"], 300.0 / 40.0)

    def test_filter_restricts_summary(self):
        self.filters["Province"] = ["广东"]
        summary_df = module.make_summary_df(self.data_df, self.filters)
        # Full month range is kept; months without records are zero-filled
        self.assertEqual(len(summary_df), 2)
        row_jan = summary_df[summary_df["YM"] == 202301].iloc[0]
        self.assertEqual(row_jan["Value"], 300.0)
        row_feb = summary_df[summary_df["YM"] == 202302].iloc[0]
        self.assertEqual(row_feb["Value"], 0.0)
        self.assertTrue(np.isnan(row_feb["Price"]))

    def test_full_month_range_is_kept(self):
        summary_df = module.make_summary_df(self.data_df, self.filters)
        self.assertEqual(len(summary_df), 2)
        self.assertEqual(list(summary_df["YM"]), [202301, 202302])
        self.assertEqual(list(summary_df["YM_label"]), ["2023-01", "2023-02"])


class TestFlatFileIntegrity(unittest.TestCase):
    """The generated flat file must exist with expected shape and names."""

    def test_flat_file_output(self):
        self.assertTrue(os.path.exists(module.PROCESSED_DATA_PATH))
        flat_df = pd.read_csv(module.PROCESSED_DATA_PATH)
        self.assertEqual(len(flat_df), 65084)
        self.assertIn("Price", flat_df.columns)
        for col in ["VBP", "Company", "Product", "Product Category", "Tube-ONS"]:
            self.assertIn(col, flat_df.columns)


class TestPriceSpikeMarking(unittest.TestCase):
    """Price chart must mark |MoM change| above threshold in red."""

    def setUp(self):
        self.summary_df = pd.DataFrame({
            "YM": [202301, 202302, 202303],
            "YM_label": ["2023-01", "2023-02", "2023-03"],
            "Value": [100.0, 100.0, 100.0],
            "Volume": [10.0, 10.0, 10.0],
            "Price": [10.0, 10.6, 11.0],  # +6% spike, then +3.8% normal
        })

    def test_spike_points_marked_red(self):
        fig = module.make_figure_price(self.summary_df)
        self.assertEqual(len(fig.data), 2)
        spike_trace = fig.data[1]
        self.assertEqual(spike_trace.line.color, module.COLOR_DOWN)
        spike_y = list(spike_trace.y)
        self.assertTrue(np.isnan(spike_y[0]))      # first month: no baseline
        self.assertAlmostEqual(spike_y[1], 10.6)   # +6% change -> marked
        self.assertTrue(np.isnan(spike_y[2]))      # +3.8% change -> not marked

    def test_no_spike_all_flat(self):
        flat_df = self.summary_df.assign(Price=[10.0, 10.4, 10.8])  # all < 5%
        fig = module.make_figure_price(flat_df)
        spike_y = list(fig.data[1].y)
        self.assertTrue(all(np.isnan(v) for v in spike_y))


if __name__ == "__main__":
    unittest.main(verbosity=2)
