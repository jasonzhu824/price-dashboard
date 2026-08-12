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


# === Chart & Table (Danone AMN SFE style)
def _make_empty_figure():
    """Build an empty figure with a 'no data' annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data for current selection",
        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        font=dict(family=FONT_FAMILY, size=14, color=COLOR_BLACK),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def make_figure_value(summary_df):
    """Build the monthly Value bar chart.

    Styling follows the Danone AMN SFE style guide (colors / font sizes).
    """
    if summary_df.empty:
        return _make_empty_figure()

    fig = go.Figure()
    x = summary_df["YM_label"]
    fig.add_trace(go.Bar(
        x=x, y=summary_df["Value"], name="Value",
        marker_color=COLOR_PRIMARY,
    ))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=14, color=COLOR_BLACK),
        title=dict(
            text="Monthly Value",
            font=dict(family=FONT_FAMILY, size=20, color=COLOR_PRIMARY),
        ),
        xaxis=dict(title=None, tickfont=dict(size=10), gridcolor="#e6e6e6",
                   tickangle=-45, tickmode="array",
                   tickvals=list(x)[::3], ticktext=list(x)[::3]),
        yaxis=dict(
            title=dict(text="Value (CNY)", font=dict(size=14, color=COLOR_PRIMARY)),
            gridcolor="#e6e6e6",
        ),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=80, b=40),
    )
    return fig


def make_figure_volume(summary_df):
    """Build the monthly Volume bar chart.

    Styling follows the Danone AMN SFE style guide (colors / font sizes).
    """
    if summary_df.empty:
        return _make_empty_figure()

    fig = go.Figure()
    x = summary_df["YM_label"]
    fig.add_trace(go.Bar(
        x=x, y=summary_df["Volume"], name="Volume",
        marker_color=COLOR_SECONDARY,
    ))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=14, color=COLOR_BLACK),
        title=dict(
            text="Monthly Volume",
            font=dict(family=FONT_FAMILY, size=20, color=COLOR_PRIMARY),
        ),
        xaxis=dict(title=None, tickfont=dict(size=10), gridcolor="#e6e6e6",
                   tickangle=-45, tickmode="array",
                   tickvals=list(x)[::3], ticktext=list(x)[::3]),
        yaxis=dict(
            title=dict(text="Volume", font=dict(size=14, color=COLOR_SECONDARY)),
            gridcolor="#e6e6e6",
        ),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=80, b=40),
    )
    return fig


def make_figure_price(summary_df):
    """Build the monthly Price line chart with spike marking.

    Months whose month-over-month Price change exceeds
    PRICE_CHANGE_THRESHOLD_PCT (absolute value) are marked in red.
    The legend sits horizontally below the plot so long series names
    (e.g. "MoM Change > 5%") are never truncated.
    Styling follows the Danone AMN SFE style guide (colors / font sizes).
    """
    if summary_df.empty:
        return _make_empty_figure()

    fig = go.Figure()
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
        font=dict(family=FONT_FAMILY, size=14, color=COLOR_BLACK),
        title=dict(
            text="Monthly Price",
            font=dict(family=FONT_FAMILY, size=20, color=COLOR_PRIMARY),
        ),
        xaxis=dict(title=None, tickfont=dict(size=10), gridcolor="#e6e6e6",
                   tickangle=-45, tickmode="array",
                   tickvals=list(x)[::3], ticktext=list(x)[::3]),
        yaxis=dict(
            title=dict(text="Price (CNY/Unit)", font=dict(size=14, color=COLOR_TERTIARY)),
            gridcolor="#e6e6e6",
        ),
        # Horizontal legend below the plot avoids right-edge truncation
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.28,
            xanchor="center", x=0.5,
            font=dict(size=12),
        ),
        hovermode="x unified",
        margin=dict(l=60, r=60, t=80, b=90),
    )
    return fig


def make_table(summary_df):
    """Build the monthly summary table styled per the style guide."""
    header = dict(
        values=["Month", "Value", "Volume", "Price"],
        fill_color=COLOR_PRIMARY,
        font=dict(color="white", size=14, family=FONT_FAMILY),
        align="center",
    )
    cells = dict(
        values=[
            summary_df["YM_label"],
            summary_df["Value"],
            summary_df["Volume"],
            summary_df["Price"],
        ],
        fill_color="white",
        font=dict(size=14, family=FONT_FAMILY, color=COLOR_BLACK),
        align="center",
        format=[None, ".2f", ".2f", ".2f"],
    )
    return go.Figure(
        data=[go.Table(header=header, cells=cells)],
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=60 + 24 * max(len(summary_df), 1),
        ),
    )


# === Dashboard (Streamlit)
def _apply_danone_style():
    """Inject Danone AMN SFE styling with a frosted-glass UI.

    Light gradient background with translucent frosted cards behind every
    chart; plotly figures use transparent paper so the glass shows.
    """
    st.markdown(
        f"""
        <style>
        /* Light gradient background */
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(135deg, #ffffff 0%, #f0f4fa 60%, #e3ecf7 100%);
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        /* Frosted glass cards wrapping charts and table */
        [data-testid="stPlotlyChart"] {{
            background: rgba(255, 255, 255, 0.65);
            -webkit-backdrop-filter: blur(16px) saturate(150%);
            backdrop-filter: blur(16px) saturate(150%);
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.9);
            box-shadow: 0 8px 32px rgba(0, 37, 119, 0.12);
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
        /* Filter labels keep the Danone accent orange */
        .stMultiSelect > label {{
            color: {COLOR_ACCENT};
            font-weight: bold;
            font-family: {FONT_FAMILY};
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
        f'font-size:20pt;font-weight:bold;margin:0;">IQVIA Price Dashboard</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-size:10pt;font-style:italic;color:#5a6b8c;">'
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

    # Summary, charts and table for the current selection
    summary_df = make_summary_df(df, selections)

    # Price chart on top (full width)
    st.plotly_chart(make_figure_price(summary_df), use_container_width=True)

    # Value and Volume bar charts stacked (Value on top, Volume below)
    st.plotly_chart(make_figure_value(summary_df), use_container_width=True)
    st.plotly_chart(make_figure_volume(summary_df), use_container_width=True)

    st.markdown(
        f'<h3 style="font-family:{FONT_FAMILY};font-size:20px;font-weight:bold;'
        f'color:{COLOR_PRIMARY};">Monthly Summary</h3>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_table(summary_df), use_container_width=True)


# === CLI Entry
if __name__ == "__main__":
    if "--generate" in sys.argv:
        generate_flat_file()
    else:
        build_dashboard()
