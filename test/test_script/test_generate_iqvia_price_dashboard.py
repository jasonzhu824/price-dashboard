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


class TestPriceLines(unittest.TestCase):
    """Monthly Price must split into one line per Product/Company when multi-selected."""

    def setUp(self):
        self.data_df = pd.DataFrame({
            "YM": [202301, 202301, 202302, 202302],
            "VBP": ["GD+20"] * 4,
            "Province": ["广东"] * 4,
            "City": ["深圳"] * 4,
            "Channel": ["Hospital"] * 4,
            "Company": ["纽迪希亚", "纽迪希亚", "费森", "费森"],
            "Tube-ONS": ["Tube"] * 4,
            "Product": ["百普素", "乐赋", "百普素", "乐赋"],
            "Product Category": ["General"] * 4,
            "Value": [100.0, 200.0, 150.0, 250.0],
            "Volume": [10.0, 20.0, 15.0, 25.0],
        })
        self.filters = {col: [] for col in module.FILTER_ORDER}

    def test_grouped_by_product_when_multiple_products(self):
        self.filters["Product"] = ["百普素", "乐赋"]
        lines_df = module.make_price_lines_df(self.data_df, self.filters)
        self.assertEqual(sorted(lines_df["Line"].unique()), ["乐赋", "百普素"])
        self.assertEqual(len(lines_df), 4)  # 2 lines x full month range

    def test_grouped_by_company_when_only_companies_selected(self):
        self.filters["Company"] = ["纽迪希亚", "费森"]
        lines_df = module.make_price_lines_df(self.data_df, self.filters)
        self.assertEqual(sorted(lines_df["Line"].unique()), ["纽迪希亚", "费森"])

    def test_single_total_line_without_grouping(self):
        lines_df = module.make_price_lines_df(self.data_df, self.filters)
        self.assertEqual(list(lines_df["Line"].unique()), ["Total"])
        self.assertEqual(len(lines_df), 2)

    def test_multi_line_figure_draws_main_and_spike_traces(self):
        self.filters["Product"] = ["百普素", "乐赋"]
        lines_df = module.make_price_lines_df(self.data_df, self.filters)
        fig = module.make_figure_price(lines_df)
        # 2 main lines + 2 per-line spike traces (hidden from legend)
        self.assertEqual(len(fig.data), 4)
        legend_names = [t.name for t in fig.data if t.showlegend is not False]
        self.assertEqual(sorted(legend_names), ["乐赋", "百普素"])

    def test_spike_trace_links_to_main_line_via_legendgroup(self):
        """Clicking a legend item must hide that line's spike markers too."""
        self.filters["Product"] = ["百普素", "乐赋"]
        lines_df = module.make_price_lines_df(self.data_df, self.filters)
        fig = module.make_figure_price(lines_df)
        # main trace i and its spike trace i+1 share the same legendgroup
        for i in range(0, len(fig.data), 2):
            self.assertEqual(fig.data[i].legendgroup, fig.data[i + 1].legendgroup)
            self.assertIsNotNone(fig.data[i].legendgroup)


class TestProvincePriceTable(unittest.TestCase):
    """Province Price Table: YOY/MoM first, optional filter, all rows shown."""

    def setUp(self):
        self.data_df = pd.DataFrame({
            "YM": [202301, 202301, 202302, 202302, 202303, 202303],
            "Province": ["广东", "浙江", "广东", "浙江", "广东", "浙江"],
            "Company": ["纽迪希亚"] * 6,
            "Product": ["百普素"] * 6,
            "VBP": ["GD+20"] * 6,
            "City": ["深圳", "杭州", "深圳", "杭州", "深圳", "杭州"],
            "Channel": ["Hospital"] * 6,
            "Tube-ONS": ["Tube"] * 6,
            "Product Category": ["General"] * 6,
            "Value": [100.0, 50.0, 120.0, 55.0, 140.0, 45.0],
            "Volume": [10.0, 5.0, 12.0, 5.5, 14.0, 4.5],
        })

    def test_excluded_provinces_removed(self):
        tbl = module.make_province_price_table(
            self.data_df, "纽迪希亚", "百普素"
        )
        self.assertNotIn("其他", tbl.index)
        self.assertNotIn("EC+Pharmacy", tbl.index)

    def test_yoy_mom_are_first_two_columns(self):
        tbl = module.make_province_price_table(
            self.data_df, "纽迪希亚", "百普素"
        )
        self.assertEqual(tbl.columns[0], "CM YOY Change %")
        self.assertEqual(tbl.columns[1], "CM MoM Change %")

    def test_all_rows_shown_without_filter(self):
        # Without company/product filters, all provinces are included
        tbl = module.make_province_price_table(self.data_df)
        self.assertIn("广东", tbl.index)
        self.assertIn("浙江", tbl.index)
        self.assertTrue(len(tbl) >= 2)

    def test_no_scientific_notation_in_price_cols(self):
        tbl = module.make_province_price_table(
            self.data_df, "纽迪希亚", "百普素"
        )
        for col in tbl.columns[2:]:  # skip YOY/MoM
            for val in tbl[col]:
                if val not in ("N/A", ""):
                    self.assertNotIn("e-", str(val))


class TestPctRedStyle(unittest.TestCase):
    """YOY/MoM values >= 5% in absolute terms must be highlighted red."""

    def test_highlight_at_or_above_threshold(self):
        self.assertTrue(module._pct_red_style("+5.20%"))
        self.assertTrue(module._pct_red_style("-5.00%"))
        self.assertTrue(module._pct_red_style("+26.73%"))

    def test_no_highlight_below_threshold(self):
        self.assertEqual(module._pct_red_style("+4.99%"), "")
        self.assertEqual(module._pct_red_style("-4.99%"), "")

    def test_no_highlight_for_placeholders(self):
        self.assertEqual(module._pct_red_style("<0.01%"), "")
        self.assertEqual(module._pct_red_style("N/A"), "")


class TestFilterOptionSorting(unittest.TestCase):
    """Company options follow business order; other columns sort by name."""

    def test_company_fixed_order(self):
        values = ["雀巢", "其他", "雅培", "纽迪希亚", "费卡华瑞"]
        result = module._sort_filter_options("Company", values)
        self.assertEqual(
            result, ["纽迪希亚", "费卡华瑞", "雅培", "其他", "雀巢"]
        )

    def test_unknown_company_appended_alphabetically(self):
        values = ["雀巢", "雅培"]
        result = module._sort_filter_options("Company", values)
        self.assertEqual(result, ["雅培", "雀巢"])

    def test_other_columns_sorted_by_name(self):
        values = ["b", "a", "c"]
        self.assertEqual(
            module._sort_filter_options("Product", values), ["a", "b", "c"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
