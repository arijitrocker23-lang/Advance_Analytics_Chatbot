# =============================================================================
# utils.py
# =============================================================================
# Shared utility functions used by multiple modules:
#   - JSON serialisation helpers (handles numpy / pandas types)
#   - DataFrame schema extraction
#   - Relative-date parsing and calculation
#   - General-question detection and answering
#   - Filter application on DataFrames
# =============================================================================

import json
import re
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from data_models import CustomInstructions

warnings.filterwarnings("ignore")


# =============================================================================
# JSON helpers
# =============================================================================

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Serialise *obj* to a JSON string, gracefully handling types that the
    standard json module cannot serialise (numpy scalars, pandas
    Timestamps, NaT, sets, etc.).
    """

    def _default(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if o is pd.NaT:
            return None
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, set):
            return list(o)
        return str(o)

    return json.dumps(obj, default=_default, **kwargs)


# =============================================================================
# Schema extraction
# =============================================================================

def df_schema_summary(
    df: pd.DataFrame,
    max_examples: int = 5,
) -> Dict[str, Any]:
    """
    Build a lightweight JSON-friendly schema summary of *df*.

    The summary includes column names, dtypes, null counts, unique counts,
    a handful of example values, and flags for numeric / date columns.
    This is the primary input to the planner prompt.
    """
    date_cols = detect_date_columns(df)

    schema: Dict[str, Any] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "date_columns": date_cols,
        "columns": [],
    }

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        schema["columns"].append(
            {
                "name": col,
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "unique_count": int(series.nunique()),
                "example_values": (
                    [str(v) for v in non_null.unique()[:max_examples].tolist()]
                    if not non_null.empty
                    else []
                ),
                "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
                "is_date_column": col in date_cols,
            }
        )
    return schema


# =============================================================================
# Date helpers
# =============================================================================

def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """
    Heuristically detect columns in *df* that contain date / datetime values.
    Checks the dtype first, then falls back to attempting to parse a sample
    of string values.
    """
    date_columns: List[str] = []
    for col in df.columns:
        # Already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue
        # Try parsing a sample of string values
        if df[col].dtype == "object":
            sample = df[col].dropna().head(200)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample, errors="raise")
                    date_columns.append(col)
                except Exception:
                    pass
    return date_columns


def get_date_context_string() -> str:
    """
    Return a formatted string with the current date, time, and day of week.
    This is injected into the planner prompt so that relative-date filters
    (e.g. "last 30 days") can be resolved correctly.
    """
    now = datetime.now()
    return (
        f"CURRENT DATE/TIME CONTEXT:\n"
        f"- Current Date: {now.strftime('%Y-%m-%d')}\n"
        f"- Current Time: {now.strftime('%H:%M:%S')}\n"
        f"- Day of Week: {now.strftime('%A')}\n"
    )


def calculate_date_from_relative(
    relative_value: Dict[str, Any],
) -> Tuple[datetime, datetime]:
    """
    Given a relative-date specification (e.g. {"value": 30, "unit": "days",
    "direction": "ago"}), return a (start, end) datetime pair.
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    value = int(relative_value.get("value", 0))
    unit = str(relative_value.get("unit", "days")).rstrip("s")
    direction = str(relative_value.get("direction", "before"))

    def _subtract(date: datetime, val: int, u: str) -> datetime:
        if u == "day":
            return date - timedelta(days=val)
        if u == "week":
            return date - timedelta(weeks=val)
        if u == "month":
            return date - relativedelta(months=val)
        if u == "year":
            return date - relativedelta(years=val)
        if u == "quarter":
            return date - relativedelta(months=val * 3)
        return date

    if direction in ("before", "ago"):
        start = _subtract(today, value, unit)
        end = today + timedelta(days=1)
    elif direction == "after":
        start = today
        end = _subtract(today, -value, unit)
    else:
        start = today
        end = today + timedelta(days=1)

    return start, end


# =============================================================================
# General-question detection
# =============================================================================

# Regex patterns that match non-data questions.
_GENERAL_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(what|tell me|show)\s+(is\s+)?(the\s+)?(current\s+)?time\b", "time"),
    (r"\bwhat\s+time\s+is\s+it\b", "time"),
    (r"\bcurrent\s+time\b", "time"),
    (r"\b(what|tell me|show)\s+(is\s+)?(the\s+)?(today\'?s?\s+)?date\b", "date"),
    (r"\bwhat\s+date\s+is\s+it\b", "date"),
    (r"\btoday\'?s?\s+date\b", "date"),
    (r"\bcurrent\s+date\b", "date"),
    (r"\b(what|which)\s+(day|day of the week)\s+(is\s+)?(it|today)\b", "day"),
    (r"\bwhat\s+day\s+is\s+it\b", "day"),
    (r"\btoday\'?s?\s+day\b", "day"),
    (r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))[\s\!\?\.]?$", "greeting"),
    (r"^(hi|hello|hey)\s+(there|bot|chatbot)[\s\!\?\.]?$", "greeting"),
    (r"\b(who|what)\s+are\s+you\b", "about"),
    (r"\bwhat\s+can\s+you\s+do\b", "about"),
    (r"\bhelp\s*me\b", "about"),
    (r"^help[\s\!\?\.]?$", "about"),
    (r"\bhow\s+are\s+you\b", "how_are_you"),
]


def detect_general_question(question: str) -> Optional[str]:
    """
    If *question* matches a known general-question pattern, return the
    question type (e.g. "time", "date", "greeting").  Otherwise return None.
    """
    q_lower = question.strip().lower()
    for pattern, q_type in _GENERAL_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return q_type
    return None


def answer_general_question(
    question_type: str,
    original_question: str = "",
) -> str:
    """
    Produce a canned answer for a general (non-data) question.
    """
    now = datetime.now()

    if question_type == "time":
        return (
            f"The current time is {now.strftime('%I:%M:%S %p')} "
            f"({now.strftime('%H:%M:%S')} in 24-hour format)."
        )
    if question_type == "date":
        return (
            f"Today's date is {now.strftime('%A, %B %d, %Y')} "
            f"({now.strftime('%Y-%m-%d')})."
        )
    if question_type == "day":
        return f"Today is {now.strftime('%A')}."
    if question_type == "greeting":
        hour = now.hour
        greeting = (
            "Good morning" if hour < 12
            else ("Good afternoon" if hour < 17 else "Good evening")
        )
        return (
            f"{greeting}! I am your Analytical Chatbot. "
            "How can I help you analyse your data today?"
        )
    if question_type == "about":
        return (
            "About Me\n\n"
            "I am a general-purpose analytical chatbot that helps you explore "
            "and understand your data.  Here is what I can do:\n\n"
            "- Load Data: Upload CSV files or load from S3\n"
            "- Query Data: Ask questions in natural language\n"
            "- Analyse: Count, group, filter, aggregate, pivot, and describe\n"
            "- Visualise: Generate bar, line, scatter, pie, histogram, box, "
            "and other charts\n"
            "- Date Filtering: Handle relative dates like 'last 30 days'\n"
            "- Follow Custom Rules: Apply your organisation's business rules\n"
            "- Knowledge Base: Store and use domain knowledge\n\n"
            "Load your data and start asking questions!"
        )
    if question_type == "how_are_you":
        return (
            "I am doing great, thank you for asking! "
            "I am ready to help you analyse your data."
        )
    return (
        f"I received your message: '{original_question}'. "
        "Is there something specific about your data you would like to analyse?"
    )


# =============================================================================
# Filter application
# =============================================================================

def apply_filters(
    df: pd.DataFrame,
    filters: List[Dict[str, Any]],
    custom_instructions: Optional[CustomInstructions] = None,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Apply a list of filter specifications to *df* and return the filtered
    DataFrame together with optional date-range metadata.

    Each filter is a dict with keys: column, op, value.
    Supported ops: eq, ne, gt, gte, lt, lte, contains, startswith, endswith,
    isin, notnull, isnull, between, relative_date.
    """
    if not filters:
        return df, None

    mask = pd.Series(True, index=df.index)
    date_range_info: Optional[Dict[str, Any]] = None

    for f in filters:
        col = f.get("column")
        op = f.get("op")
        val = f.get("value")

        # Resolve column aliases if configured
        if custom_instructions and custom_instructions.column_aliases:
            col = custom_instructions.column_aliases.get(col, col)

        if not col or col not in df.columns:
            continue

        series = df[col]

        # Resolve value mappings if configured
        if custom_instructions and custom_instructions.value_mappings:
            col_mappings = custom_instructions.value_mappings.get(col, {})
            if isinstance(val, str):
                for term, actual_val in col_mappings.items():
                    if term.lower() == val.lower():
                        val = actual_val
                        break
            elif isinstance(val, list):
                new_val = []
                for v in val:
                    mapped = False
                    for term, actual_val in col_mappings.items():
                        if term.lower() == str(v).lower():
                            new_val.append(actual_val)
                            mapped = True
                            break
                    if not mapped:
                        new_val.append(v)
                val = new_val

        try:
            # -- relative_date -----------------------------------------------
            if op == "relative_date" and isinstance(val, dict):
                start_dt, end_dt = calculate_date_from_relative(val)
                date_series = pd.to_datetime(series, errors="coerce")
                mask &= (date_series >= start_dt) & (date_series < end_dt)
                date_range_info = {
                    "start_date": start_dt.isoformat(),
                    "end_date": end_dt.isoformat(),
                    "column": col,
                    "description": f"Last {val.get('value')} {val.get('unit', 'days')}",
                }
                continue

            # -- null checks --------------------------------------------------
            if op == "isnull":
                mask &= series.isna()
                continue
            if op == "notnull":
                mask &= series.notna()
                continue

            # -- between (dates or numerics) ----------------------------------
            if op == "between" and isinstance(val, list) and len(val) == 2:
                try:
                    date_series = pd.to_datetime(series, errors="coerce")
                    if date_series.notna().sum() > len(series) * 0.5:
                        start = pd.to_datetime(val[0])
                        end = pd.to_datetime(val[1])
                        mask &= (date_series >= start) & (date_series <= end)
                        date_range_info = {
                            "start_date": start.isoformat(),
                            "end_date": end.isoformat(),
                            "column": col,
                            "description": f"{val[0]} to {val[1]}",
                        }
                        continue
                except Exception:
                    pass
                num_series = pd.to_numeric(series, errors="coerce")
                mask &= (num_series >= float(val[0])) & (num_series <= float(val[1]))
                continue

            # -- isin ---------------------------------------------------------
            if op == "isin" and isinstance(val, list):
                s_lower = series.astype(str).str.lower()
                mask &= s_lower.isin([str(v).lower() for v in val])
                continue

            # -- numeric / date comparisons -----------------------------------
            if op in ("gt", "gte", "lt", "lte"):
                # Try date comparison first
                try:
                    date_series = pd.to_datetime(series, errors="coerce")
                    if date_series.notna().sum() > len(series) * 0.5:
                        date_val = pd.to_datetime(val)
                        if op == "gt":
                            mask &= date_series > date_val
                        elif op == "gte":
                            mask &= date_series >= date_val
                        elif op == "lt":
                            mask &= date_series < date_val
                        elif op == "lte":
                            mask &= date_series <= date_val
                        continue
                except Exception:
                    pass
                # Fall back to numeric comparison
                num_series = pd.to_numeric(series, errors="coerce")
                num_val = float(val)
                if op == "gt":
                    mask &= num_series > num_val
                elif op == "gte":
                    mask &= num_series >= num_val
                elif op == "lt":
                    mask &= num_series < num_val
                elif op == "lte":
                    mask &= num_series <= num_val
                continue

            # -- string operations --------------------------------------------
            s_str = series.astype(str)
            if op == "eq":
                mask &= s_str.str.lower() == str(val).lower()
            elif op == "ne":
                mask &= s_str.str.lower() != str(val).lower()
            elif op == "contains":
                mask &= s_str.str.contains(str(val), case=False, na=False)
            elif op == "startswith":
                mask &= s_str.str.lower().str.startswith(str(val).lower(), na=False)
            elif op == "endswith":
                mask &= s_str.str.lower().str.endswith(str(val).lower(), na=False)

        except Exception:
            # If a single filter fails, skip it and continue with the rest
            continue

    return df[mask], date_range_info


def result_to_dataframe(
    structured_result: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """
    Extract a list-of-dicts result from a structured result dictionary
    and convert it to a pandas DataFrame for display.
    """
    if not structured_result:
        return None
    for key in ("result", "result_preview", "describe"):
        val = structured_result.get(key)
        if isinstance(val, list) and val and isinstance(val[0], dict):
            return pd.DataFrame(val)
    return None