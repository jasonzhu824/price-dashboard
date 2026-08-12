"""
generate_iqvia_price_dashboard.py
=================================
Generate an IQVIA price flat file (Price = Value / Volume) and build an
interactive Danone-styled dashboard with Streamlit + plotly.

=== Usage
- Generate flat file only:
    python src/generate_iqvia_price_dashboard.py --generate
- Launch dashboard (streamlit):
    streamlit run src/generate_iqvia_price_dashboard.py

=== Data Source
- Raw:      data/raw/MS Tracking DB_2301-2605.xlsx (sheet: DB)
- Output:   data/processed/iqvia_price_flat_file.csv

=== Dashboard Filters (in order, cascade)
VBP -> Province -> City -> Channel -> Company -> Tube-ONS -> Product -> Product Category
"""

# === Imports
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# === Global Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "MS Tracking DB_2301-2605.xlsx")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "iqvia_price_flat_file.csv")
SHEET_NAME = "DB"

# Column standardization map (original name -> display name)
RENAME_MAP = {
    "VBP Type 1": "VBP",
    "Comp sum CN": "Company",
    "SKU CN": "Product",
    "Sub Category3": "Product Category",
    "Tube_ONS": "Tube-ONS",
}

# Dashboard filter order (cascade dependency order)
FILTER_ORDER = [
    "VBP",
    "Province",
    "City",
    "Channel",
    "Company",
    "Tube-ONS",
    "Product",
    "Product Category",
]

# Danone AMN SFE Data Visualization Style Guide palette
COLOR_PRIMARY = "#002577"    # headers / Value bars
COLOR_SECONDARY = "#005eb8"  # Volume bars
COLOR_TERTIARY = "#00acec"   # Price line
COLOR_QUATERNARY = "#bddfff"
COLOR_UP = "#53ac17"
COLOR_DOWN = "#ee3340"
COLOR_ACCENT = "#f88706"
COLOR_ACCENT2 = "#56b9b7"
COLOR_BLACK = "#000000"
FONT_FAMILY = "Danone One Condensed, Arial, sans-serif"

# Color palette for multiple price lines (cycles when more lines than colors)
LINE_COLORS = [
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY,
    COLOR_UP, COLOR_ACCENT, COLOR_ACCENT2, COLOR_QUATERNARY,
]

# Price spike marking threshold (month-over-month change, in percent)
PRICE_CHANGE_THRESHOLD_PCT = 5.0

# Data cache (module-level global)
_data_cache = None


# === Data Loading & Caching
def load_data(use_cache=True):
    """Load raw tracking DB from Excel and standardize column names.

    Args:
        use_cache: If True, reuse the module-level cached DataFrame.

    Returns:
        DataFrame with renamed columns.
    """
    global _data_cache
    if use_cache and _data_cache is not None:
        print("Using cached data for raw tracking DB")
        return _data_cache
    df = (
        pd.read_excel(RAW_DATA_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
        .rename(columns=RENAME_MAP)
    )
    _data_cache = df
    print(f"Loaded {os.path.basename(RAW_DATA_PATH)}")
    return df


def clear_cache():
    """Reset the module-level cached DataFrame."""
    global _data_cache
    _data_cache = None


# === Price Calculation & Flat File Generation
def calc_price(df):
    """Calculate unit price as Value / Volume.

    Price is NaN when Volume is zero (division by zero guard).
    Negative Volume yields negative Price and is kept as-is for traceability.

    Args:
        df: DataFrame containing Value and Volume columns.

    Returns:
        DataFrame with an extra Price column.
    """
    return df.assign(
        Price=lambda x: np.where(x["Volume"] != 0, x["Value"] / x["Volume"], np.nan)
    )


def generate_flat_file():
    """Compute Price and write the full-granularity flat file as CSV."""
    flat_df = calc_price(load_data())
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    # utf-8-sig keeps Chinese characters readable in Excel
    flat_df.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")
    print("=" * 50)
    print(f"Generated {os.path.basename(PROCESSED_DATA_PATH)}")
    print(f"Rows: {len(flat_df)} | Columns: {len(flat_df.columns)}")
    print(f"Price range: {flat_df['Price'].min():.4f} ~ {flat_df['Price'].max():.4f}")


def load_flat_file():
    """Load the flat file; fall back to computing from raw data if missing."""
    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"Loaded {os.path.basename(PROCESSED_DATA_PATH)}")
        return pd.read_csv(PROCESSED_DATA_PATH)
    print("Flat file not found, computing from raw data...")
    return calc_price(load_data())


# === Aggregation (shared by any frontend)
def build_selection_mask(data_df, selections, columns):
    """Build a boolean mask from the selected values of the given columns.

    An empty selection means "all values" for that column.

    Args:
        data_df: Full DataFrame.
        selections: Dict mapping column name to list/tuple of selected values.
        columns: Filter columns to apply, in cascade order.

    Returns:
        Boolean Series aligned to data_df index.
    """
    mask = pd.Series(True, index=data_df.index)
    for col in columns:
        selected = selections.get(col, [])
        if selected:
            mask &= data_df[col].isin(selected)
    return mask


def _full_ym_range(data_df):
    """Generate the continuous YM series covering the global data range."""
    start_ym = str(int(data_df["YM"].min()))
    end_ym = str(int(data_df["YM"].max()))
    return pd.period_range(start=start_ym, end=end_ym, freq="M").strftime("%Y%m").astype(int)


def make_summary_df(data_df, selections):
    """Aggregate monthly Value / Volume and weighted Price for the selection.

    The result covers every month of the global data range; months without
    records get Value/Volume = 0 and Price = NaN (shown as gaps).

    Args:
        data_df: Full DataFrame.
        selections: Dict mapping column name to list/tuple of selected values.

    Returns:
        DataFrame with columns YM, YM_label, Value, Volume, Price.
    """
    mask = build_selection_mask(data_df, selections, FILTER_ORDER)
    sub_df = data_df[mask]
    full_ym = _full_ym_range(data_df)
    summary_df = (
        sub_df.groupby("YM", as_index=False)
        .agg(Value=("Value", "sum"), Volume=("Volume", "sum"))
        .set_index("YM")
        .reindex(full_ym, fill_value=0)
        .rename_axis("YM")
        .reset_index()
        .assign(
            Price=lambda x: np.where(
                x["Volume"] != 0,
                np.round(x["Value"] / x["Volume"], 4),
                np.nan,
            )
        )
        .assign(YM_label=lambda x: x["YM"].astype(str).str.replace(
            r"(\d{4})(\d{2})", r"\1-\2", regex=True
        ))
        .reset_index(drop=True)
    )
    return summary_df


def make_price_lines_df(data_df, selections):
    """Build per-line monthly Price series for the current selection.

    Line grouping follows the selection so multiple Products or Companies
    can be compared at once on the Price chart:
    - >=2 Products explicitly selected -> one line per Product
    - >=2 Companies explicitly selected -> one line per Company
    - otherwise a single aggregated line (Line = "Total")

    Args:
        data_df: Full DataFrame.
        selections: Dict mapping column name to list/tuple of selected values.

    Returns:
        DataFrame with columns YM, YM_label, Value, Volume, Price, Line.
        Price is NaN for months without records (shown as gaps).
    """
    mask = build_selection_mask(data_df, selections, FILTER_ORDER)
    sub_df = data_df[mask]
    full_ym = _full_ym_range(data_df)
    sel_products = selections.get("Product", [])
    sel_companies = selections.get("Company", [])
    if len(sel_products) >= 2:
        group_key = "Product"
    elif len(sel_companies) >= 2:
        group_key = "Company"
    else:
        group_key = None

    # Single aggregated line: reuse the plain monthly summary
    if group_key is None:
        return make_summary_df(data_df, selections).assign(Line="Total")

    line_frames = []
    for key, grp in sub_df.groupby(group_key):
        monthly = (
            grp.groupby("YM", as_index=False)
            .agg(Value=("Value", "sum"), Volume=("Volume", "sum"))
            .set_index("YM")
            .reindex(full_ym, fill_value=0)
            .rename_axis("YM")
            .reset_index()
            .assign(
                Price=lambda x: np.where(
                    x["Volume"] != 0,
                    np.round(x["Value"] / x["Volume"], 4),
                    np.nan,
                )
            )
            .assign(YM_label=lambda x: x["YM"].astype(str).str.replace(
                r"(\d{4})(\d{2})", r"\1-\2", regex=True
            ))
            .assign(Line=key)
        )
        line_frames.append(monthly)
    return pd.concat(line_frames, ignore_index=True)


EXCLUDED_PROVINCES = ["其他", "EC+Pharmacy"]


def make_province_price_table(data_df, company, product):
    """Build a province × month price table with CM YOY / MoM change.

    Each row is a province ("其他" and "EC+Pharmacy" excluded). Monthly
    price columns cover the full data range.  The last two columns are:
    - CM YOY Change % (latest month vs same month last year)
    - CM MoM Change % (latest month vs previous month)
    The table is sorted by CM YOY Change % descending.

    Args:
        data_df: Full DataFrame.
        company: Selected company name.
        product: Selected product name.

    Returns:
        DataFrame indexed by province with monthly price columns and the
        two computed metric columns.
    """
    all_provinces = sorted(
        p for p in data_df["Province"].dropna().unique()
        if p not in EXCLUDED_PROVINCES
    )
    yms = sorted(data_df["YM"].unique())
    latest_ym = yms[-1]
    # YOY reference: same month, previous year (fall back to earliest month)
    if len(yms) >= 13:
        yoy_ym = yms[-13]
    else:
        yoy_ym = yms[0]
    # MoM reference: previous month
    mom_idx = yms.index(latest_ym) - 1
    mom_ym = yms[mom_idx] if mom_idx >= 0 else latest_ym

    selections = {col: [] for col in FILTER_ORDER}
    selections["Company"] = [company]
    selections["Product"] = [product]

    records = {}
    for province in all_provinces:
        sel = dict(selections)
        sel["Province"] = [province]
        s = make_summary_df(data_df, sel)
        records[province] = s.set_index("YM")["Price"]

    price_df = pd.DataFrame(records)  # index=YM (int), columns=province
    price_df = price_df.T  # provinces as rows, months as columns
    price_df = price_df[sorted(price_df.columns)]
    # Format column labels as YYYY-MM
    price_df.columns = [
        f"{str(c)[:4]}-{str(c)[4:]}" for c in price_df.columns
    ]

    latest_label = f"{str(latest_ym)[:4]}-{str(latest_ym)[4:]}"
    yoy_label = f"{str(yoy_ym)[:4]}-{str(yoy_ym)[4:]}"
    mom_label = f"{str(mom_ym)[:4]}-{str(mom_ym)[4:]}"

    if yoy_label in price_df.columns:
        price_df["CM YOY Change %"] = (
            (price_df[latest_label].astype(float)
             / price_df[yoy_label].astype(float) - 1) * 100
        )
    else:
        price_df["CM YOY Change %"] = np.nan

    if mom_label in price_df.columns:
        price_df["CM MoM Change %"] = (
            (price_df[latest_label].astype(float)
             / price_df[mom_label].astype(float) - 1) * 100
        )
    else:
        price_df["CM MoM Change %"] = np.nan

    price_df = price_df.sort_values(
        "CM YOY Change %", ascending=False, na_position="last"
    )
    # Format price columns as strings after sorting, avoids scientific notation in UI
    price_cols = [c for c in price_df.columns
                  if c not in ("CM YOY Change %", "CM MoM Change %")]
    for col in price_cols:
        price_df[col] = price_df[col].apply(
            lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
        )
    # Format for display after sorting (string formatting breaks ordering)
    price_df["CM YOY Change %"] = price_df["CM YOY Change %"].apply(_fmt_pct)
    price_df["CM MoM Change %"] = price_df["CM MoM Change %"].apply(_fmt_pct)
    return price_df


def _fmt_pct(val):
    """Format a percentage value for the province price table.

    Values below 0.01% in absolute terms are shown as "<0.01%" to avoid
    cluttering the table with near-zero figures.  NaN/None becomes "N/A".
    """
    if pd.isna(val):
        return "N/A"
    if abs(val) < 0.01:
        return "<0.01%"
    return f"{val:+.2f}%"


# === Chart & Table (Danone AMN SFE style)
def _make_empty_figure():
    """Build an empty figure with a 'no data' annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data for current selection",
        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        font=dict(family=FONT_FAMILY, size=15, color=COLOR_BLACK),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def make_figure_price(summary_df):
    """Build the monthly Price line chart with spike marking.

    Supports multiple price lines: when the input contains a "Line" column
    (see make_price_lines_df) every group is drawn as its own line with its
    own |MoM change| > PRICE_CHANGE_THRESHOLD_PCT red marking. Without a
    "Line" column a single aggregated line is drawn (backwards compatible).
    The legend sits horizontally below the plot so long series names
    (e.g. "MoM Change > 5%") are never truncated.
    Styling follows the Danone AMN SFE style guide (colors / font sizes).
    """
    if summary_df.empty:
        return _make_empty_figure()

    fig = go.Figure()
    has_lines = "Line" in summary_df.columns
    if has_lines:
        # Full month sequence for the shared x-axis
        x = (
            summary_df[["YM", "YM_label"]]
            .drop_duplicates()
            .sort_values("YM")["YM_label"]
        )
        for idx, (line, grp) in enumerate(summary_df.groupby("Line")):
            color = LINE_COLORS[idx % len(LINE_COLORS)]
            grp = grp.sort_values("YM")
            price = grp["Price"]
            fig.add_trace(go.Scatter(
                x=grp["YM_label"], y=price, name=line,
                mode="lines+markers",
                line=dict(color=color, width=3),
                marker=dict(size=6, color=color),
            ))
            # Mark |MoM change| above the threshold in red per line
            pct_change = price.pct_change() * 100
            spike_y = price.where(pct_change.abs() > PRICE_CHANGE_THRESHOLD_PCT)
            fig.add_trace(go.Scatter(
                x=grp["YM_label"], y=spike_y,
                name=f"{line} | MoM > {PRICE_CHANGE_THRESHOLD_PCT:.0f}%",
                mode="lines+markers",
                line=dict(color=COLOR_DOWN, width=3),
                marker=dict(size=8, color=COLOR_DOWN),
                connectgaps=False,
                showlegend=False,
            ))
    else:
        x = summary_df["YM_label"]
        price = summary_df["Price"]
        fig.add_trace(go.Scatter(
            x=x, y=price, name="Price",
            mode="lines+markers",
            line=dict(color=COLOR_TERTIARY, width=3),
            marker=dict(size=6, color=COLOR_TERTIARY),
        ))
        # Mark months with |MoM change| above the threshold in red
        pct_change = price.pct_change() * 100
        spike_y = price.where(pct_change.abs() > PRICE_CHANGE_THRESHOLD_PCT)
        fig.add_trace(go.Scatter(
            x=x, y=spike_y, name=f"MoM Change > {PRICE_CHANGE_THRESHOLD_PCT:.0f}%",
            mode="lines+markers",
            line=dict(color=COLOR_DOWN, width=3),
            marker=dict(size=8, color=COLOR_DOWN),
            connectgaps=False,
        ))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=15, color=COLOR_BLACK),
        title=dict(
            text="Monthly Price",
            font=dict(family=FONT_FAMILY, size=24, color=COLOR_PRIMARY),
        ),
        xaxis=dict(title=None, tickfont=dict(size=11), gridcolor="#e6e6e6",
                   tickangle=-45, tickmode="array",
                   tickvals=list(x)[::3], ticktext=list(x)[::3]),
        yaxis=dict(
            title=dict(text="Price (CNY/Unit)", font=dict(size=15, color=COLOR_TERTIARY)),
            gridcolor="#e6e6e6",
        ),
        # Horizontal legend below the plot avoids right-edge truncation
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.28,
            xanchor="center", x=0.5,
            font=dict(size=13)
        ),
        hovermode="x unified",
        margin=dict(l=60, r=60, t=90, b=90),
    )
    return fig




# === Dashboard (Streamlit)
def _apply_danone_style():
    """Inject Danone AMN SFE styling with card-based layout.

    Plain white page background with solid white rounded cards around
    every chart; plotly figures use transparent paper so the cards show.
    """
    st.markdown(
        f"""
        <style>
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        /* White cards wrapping charts and table */
        [data-testid="stPlotlyChart"] {{
            background: #ffffff;
            border-radius: 18px;
            border: 1px solid #e8edf5;
            box-shadow: 0 8px 32px rgba(0, 37, 119, 0.10);
            padding: 14px 18px 6px 18px;
            margin-bottom: 20px;
        }}
        /* Center content with a max width for 16:9 screens */
        .block-container {{
            max-width: 1560px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}
        /* Headings in Danone primary blue */
        h1, h2, h3, h5 {{
            color: {COLOR_PRIMARY};
            font-family: {FONT_FAMILY};
            font-weight: bold;
        }}
        h5 {{
            font-size: 17px;
        }}
        /* Filter labels keep the Danone accent orange */
        .stMultiSelect > label {{
            color: {COLOR_ACCENT};
            font-weight: bold;
            font-family: {FONT_FAMILY};
            font-size: 15px;
        }}
        /* Frosted select widgets */
        [data-baseweb="select"] > div {{
            background: rgba(255, 255, 255, 0.85);
            border-radius: 12px;
            border: 1px solid rgba(0, 37, 119, 0.15);
        }}
        .streamlit-expanderHeader {{
            font-family: {FONT_FAMILY};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_dashboard():
    """Build and display the interactive dashboard (streamlit + plotly)."""
    st.set_page_config(
        page_title="IQVIA Price Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_danone_style()
    st.markdown(
        f'<h1 style="color:{COLOR_PRIMARY};font-family:{FONT_FAMILY};'
        f'font-size:30pt;font-weight:bold;margin:0;">IQVIA Price Dashboard</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-size:12pt;font-style:italic;color:#5a6b8c;">'
        f'Source: MS Tracking DB_2301-2605 | Price = Value / Volume | 2023 - 2026</p>',
        unsafe_allow_html=True,
    )

    # Cached data load (copy to avoid mutating the cached object)
    @st.cache_data
    def _load_flat():
        return load_flat_file()
    df = _load_flat().copy()
    df["YM"] = df["YM"].astype(int)

    # Cascade filters in strict order; empty selection = all values
    st.markdown("##### Filters")
    selections = {}
    mask = pd.Series(True, index=df.index)
    cols = st.columns(4)
    for idx, col_name in enumerate(FILTER_ORDER):
        with cols[idx % 4]:
            available = sorted(df.loc[mask, col_name].dropna().unique().tolist())
            selected = st.multiselect(
                col_name, available, key=f"filter_{col_name}",
                help="Leave empty to select all",
            )
            selections[col_name] = selected
            if selected:
                mask &= df[col_name].isin(selected)

    # Price chart on top (full width) — one line per selected Product/Company
    price_lines_df = make_price_lines_df(df, selections)
    st.plotly_chart(make_figure_price(price_lines_df), use_container_width=True)

    # Province Price Break Down: province × month price table
    st.markdown(
        f'<h3 style="font-family:{FONT_FAMILY};font-size:24px;font-weight:bold;'
        f'color:{COLOR_PRIMARY};">Province Price Break Down</h3>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Select Company and Product to view province-level monthly price "
        "table (\"其他\" and \"EC+Pharmacy\" excluded)"
    )
    bd_company = st.selectbox(
        "Company", sorted(df["Company"].dropna().unique()),
        key="bd_company", index=None,
        placeholder="Select Company",
    )
    bd_products = []
    if bd_company:
        bd_products = sorted(
            df.loc[df["Company"] == bd_company, "Product"]
            .dropna().unique()
        )
    bd_product = st.selectbox(
        "Product", bd_products,
        key=f"bd_product_{bd_company}", index=None,
        placeholder="Select Product",
        disabled=not bd_products,
    )
    if bd_company and bd_product:
        tbl = make_province_price_table(df, bd_company, bd_product)
        price_cols = [
            c for c in tbl.columns
            if c not in ("CM YOY Change %", "CM MoM Change %")
        ]
        col_config = {
            c: st.column_config.TextColumn(c)
            for c in price_cols
        }
        col_config["CM YOY Change %"] = st.column_config.TextColumn("CM YOY Change %")
        col_config["CM MoM Change %"] = st.column_config.TextColumn("CM MoM Change %")
        st.dataframe(
            tbl, use_container_width=True,
            column_config=col_config,
        )
    else:
        st.info(
            "Select Company and Product to show the province price table"
        )


# === CLI Entry
if __name__ == "__main__":
    if "--generate" in sys.argv:
        generate_flat_file()
    else:
        build_dashboard()
