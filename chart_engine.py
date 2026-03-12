# =============================================================================
# chart_engine.py
# =============================================================================
# Generates interactive Plotly charts from pandas DataFrames.
#
# The executor calls generate_chart() with the chart specification from
# the planner's JSON plan.  The function returns a Plotly figure object
# that Streamlit can render with st.plotly_chart().
#
# Supported chart types:
#   bar, line, scatter, pie, histogram, heatmap, box, area,
#   stacked_bar, grouped_bar
# =============================================================================

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import Config


# ---------------------------------------------------------------------------
# Shared layout template applied to every chart for a consistent look.
# ---------------------------------------------------------------------------
_LAYOUT_TEMPLATE = dict(
    font=dict(family="Inter, sans-serif", size=12, color="#333333"),
    paper_bgcolor="#FAFAFA",
    plot_bgcolor="#FAFAFA",
    title_font_size=16,
    title_font_color="#1a1a2e",
    legend=dict(
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#CCCCCC",
        borderwidth=1,
    ),
    margin=dict(l=60, r=30, t=60, b=60),
)


def _apply_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the shared layout template and an optional title."""
    fig.update_layout(
        **_LAYOUT_TEMPLATE,
        title_text=title,
        height=Config.DEFAULT_CHART_HEIGHT,
        width=Config.DEFAULT_CHART_WIDTH,
    )
    return fig


def _prepare_data_for_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: Optional[str],
    agg_func: str = "count",
    groupby: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate the DataFrame as needed before charting.

    If y_column is None and agg_func is "count", we compute value counts
    on x_column.  If y_column is specified, we group by x_column (and any
    extra groupby columns) and apply the aggregation function.
    """
    if y_column is None or agg_func == "count":
        # Count occurrences of each value in x_column
        counts = df[x_column].value_counts().reset_index()
        counts.columns = [x_column, "count"]
        return counts

    # Determine the groupby columns
    gb_cols = [x_column]
    if groupby:
        gb_cols = gb_cols + [c for c in groupby if c != x_column and c in df.columns]

    # Ensure the y column is numeric
    numeric_y = pd.to_numeric(df[y_column], errors="coerce")
    temp_df = df.copy()
    temp_df[y_column] = numeric_y

    try:
        result = temp_df.groupby(gb_cols, dropna=False)[y_column].agg(agg_func).reset_index()
    except Exception:
        result = temp_df.groupby(gb_cols, dropna=False)[y_column].sum().reset_index()

    return result


# =============================================================================
# Individual chart builders
# =============================================================================

def _bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: Optional[str] = None,
) -> go.Figure:
    """Standard vertical bar chart."""
    fig = px.bar(
        df, x=x, y=y, color=color, title=title,
        color_discrete_sequence=Config.CHART_COLOR_PALETTE,
    )
    return _apply_layout(fig, title)


def _line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: Optional[str] = None,
) -> go.Figure:
    """Line chart, typically for time-series data."""
    fig = px.line(
        df, x=x, y=y, color=color, title=title,
        color_discrete_sequence=Config.CHART_COLOR_PALETTE,
        markers=True,
    )
    return _apply_layout(fig, title)


def _scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: Optional[str] = None,
) -> go.Figure:
    """Scatter plot."""
    fig = px.scatter(
        df, x=x, y=y, color=color, title=title,
        color_discrete_sequence=Config.CHART_COLOR_PALETTE,
    )
    return _apply_layout(fig, title)


def _pie_chart(
    df: pd.DataFrame,
    names_col: str,
    values_col: str,
    title: str,
) -> go.Figure:
    """Pie / donut chart."""
    fig = px.pie(
        df, names=names_col, values=values_col, title=title,
        color_discrete_sequence=Config.CHART_COLOR_PALETTE,
        hole=0.3,  # donut style
    )
    return _apply_layout(fig, title)


def _histogram_chart(
    df: pd.DataFrame,
    x: str,
    title: str,
    color: Optional[str] = None,
) -> go.Figure:
    """Histogram for distribution analysis."""
    fig = px.histogram(
        df, x=x, color=color, title=title,
        color_discrete_sequence=Config.CHART_COLOR_PALETTE,
    )
    return _apply_layout(fig, title)


def _box_chart(
    df: pd.DataFrame,
    x: Optional[str],
    y: str,
    title: str,
    color: Optional[str] = None,
) -> go.Figure:
    """Box plot for distribution comparison."""
    fig = px.box(
        df, x=x, y=y, color=color, title=title,
        color_discrete_sequence=Config.CHART_COLOR_PALETTE,
    )
    return _apply_layout(fig, title)


def _area_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: Optional[str] = None,
) -> go.Figure:
    """Filled area chart."""
    fig = px.area(
        df, x=x, y=y, color=color, title=title,
        color_discrete_sequence=Config.CHART_COLOR_PALETTE,
    )
    return _apply_layout(fig, title)


def _heatmap_chart(
    df: pd.DataFrame,
    title: str,
) -> go.Figure:
    """
    Heatmap from a numeric DataFrame.
    If the DataFrame has non-numeric columns they are used as index labels.
    """
    # Attempt to pivot or use as-is
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        # Fall back to showing value counts as a heatmap
        numeric_df = pd.DataFrame({"value": [0]})

    fig = px.imshow(
        numeric_df,
        title=title,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    return _apply_layout(fig, title)


def _stacked_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
) -> go.Figure:
    """Stacked bar chart using a colour/group column."""
    fig = px.bar(
        df, x=x, y=y, color=color, title=title,
        color_discrete_sequence=Config.CHART_COLOR_PALETTE,
        barmode="stack",
    )
    return _apply_layout(fig, title)


def _grouped_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
) -> go.Figure:
    """Grouped (side-by-side) bar chart."""
    fig = px.bar(
        df, x=x, y=y, color=color, title=title,
        color_discrete_sequence=Config.CHART_COLOR_PALETTE,
        barmode="group",
    )
    return _apply_layout(fig, title)


# =============================================================================
# Public interface
# =============================================================================

def generate_chart(
    df: pd.DataFrame,
    chart_spec: Dict[str, Any],
) -> Optional[go.Figure]:
    """
    Generate a Plotly figure from a DataFrame and a chart specification.

    Parameters
    ----------
    df : pd.DataFrame
        The (possibly filtered) data to chart.
    chart_spec : dict
        Keys expected:
            chart_type : str   -- one of the supported chart types
            x_column   : str   -- column for the x-axis (or names for pie)
            y_column   : str   -- column for the y-axis (optional for some types)
            groupby    : list  -- additional groupby columns (optional)
            title      : str   -- chart title (optional)
            agg_func   : str   -- aggregation function (default "count")

    Returns
    -------
    plotly.graph_objects.Figure or None
        None if the chart could not be generated.
    """
    chart_type = chart_spec.get("chart_type", "bar").lower()
    x_column = chart_spec.get("x_column")
    y_column = chart_spec.get("y_column")
    groupby = chart_spec.get("groupby") or []
    title = chart_spec.get("title", "Chart")
    agg_func = chart_spec.get("agg_func", "count")

    # Validate that the required x column exists
    if not x_column or x_column not in df.columns:
        return None

    # Validate y column if specified
    if y_column and y_column not in df.columns:
        y_column = None

    # Validate groupby columns
    groupby = [c for c in groupby if c in df.columns]

    try:
        # -- Pie chart (special handling) --------------------------------
        if chart_type == "pie":
            chart_data = _prepare_data_for_chart(
                df, x_column, y_column, agg_func, groupby
            )
            values_col = y_column if y_column and y_column in chart_data.columns else "count"
            if values_col not in chart_data.columns:
                values_col = chart_data.columns[-1]
            return _pie_chart(chart_data, x_column, values_col, title)

        # -- Histogram (works on raw data) -------------------------------
        if chart_type == "histogram":
            return _histogram_chart(df, x_column, title)

        # -- Box plot (works on raw data) --------------------------------
        if chart_type == "box":
            y_col = y_column if y_column else x_column
            x_col = x_column if y_column else None
            return _box_chart(df, x_col, y_col, title)

        # -- Heatmap (special handling) ----------------------------------
        if chart_type == "heatmap":
            return _heatmap_chart(df, title)

        # -- All other chart types need aggregated data ------------------
        chart_data = _prepare_data_for_chart(
            df, x_column, y_column, agg_func, groupby
        )
        y_col = y_column if y_column and y_column in chart_data.columns else "count"
        if y_col not in chart_data.columns:
            y_col = chart_data.columns[-1]

        # Determine the colour column for grouped/stacked charts
        color_col = groupby[0] if groupby else None

        if chart_type == "bar":
            return _bar_chart(chart_data, x_column, y_col, title, color_col)
        if chart_type == "line":
            return _line_chart(chart_data, x_column, y_col, title, color_col)
        if chart_type == "scatter":
            return _scatter_chart(chart_data, x_column, y_col, title, color_col)
        if chart_type == "area":
            return _area_chart(chart_data, x_column, y_col, title, color_col)
        if chart_type == "stacked_bar":
            if not color_col:
                return _bar_chart(chart_data, x_column, y_col, title)
            return _stacked_bar_chart(chart_data, x_column, y_col, color_col, title)
        if chart_type == "grouped_bar":
            if not color_col:
                return _bar_chart(chart_data, x_column, y_col, title)
            return _grouped_bar_chart(chart_data, x_column, y_col, color_col, title)

        # Default fallback: bar chart
        return _bar_chart(chart_data, x_column, y_col, title, color_col)

    except Exception:
        # If anything goes wrong, return None so the caller can show
        # a graceful error message instead of crashing.
        return None