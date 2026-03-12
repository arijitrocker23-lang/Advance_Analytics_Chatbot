# =============================================================================
# executor.py
# =============================================================================
# The executor takes a JSON plan produced by the planner and runs it against
# the pandas DataFrame.  It returns:
#   - A human-readable markdown summary string
#   - A structured result dictionary (for the explainer and table display)
#   - Optional date-range metadata
#   - An optional Plotly figure (for chart operations)
# =============================================================================

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from chart_engine import generate_chart
from config import Config
from data_models import CustomInstructions
from utils import (
    answer_general_question,
    apply_filters,
    safe_json_dumps,
)


class Executor:
    """
    Executes a planner-generated JSON plan against a DataFrame.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(
        self,
        plan,
        question,
        df,
        schema,
        custom_instructions,
    ):
        """
        Execute the plan and return (markdown, structured_result, date_range, figure).
        """
        op = plan.get("operation")

        # -- General questions -------------------------------------------
        if op == "general_question":
            q_type = plan.get("question_type", "other")
            original = plan.get("original_question", question)
            answer = answer_general_question(q_type, original)
            result = {"operation": "general_question", "question_type": q_type}
            return answer, result, None, None

        # -- No data loaded ----------------------------------------------
        if df is None:
            return (
                "No dataset loaded. Please upload a CSV file first.",
                {"operation": "unsupported"},
                None,
                None,
            )

        # -- Unsupported operation ---------------------------------------
        if op == "unsupported":
            reason = plan.get("reason", "Unknown reason.")
            return (
                "Unsupported query: {}".format(reason),
                {"operation": "unsupported"},
                None,
                None,
            )

        # -- Apply filters -----------------------------------------------
        filters = plan.get("filters") or []
        resolved_filters = self._resolve_filters(filters, custom_instructions)
        filtered, date_range_info = apply_filters(
            df, resolved_filters, custom_instructions
        )

        has_ci = not custom_instructions.is_empty()

        # -- Dispatch to the correct handler -----------------------------
        try:
            if op == "count_rows":
                return self._handle_count_rows(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "groupby_count":
                return self._handle_groupby_count(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "groupby_agg":
                return self._handle_groupby_agg(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "filter_show":
                return self._handle_filter_show(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "value_counts":
                return self._handle_value_counts(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "describe":
                return self._handle_describe(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "correlation":
                return self._handle_correlation(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "crosstab":
                return self._handle_crosstab(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "pivot_table":
                return self._handle_pivot_table(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "duplicate_check":
                return self._handle_duplicate_check(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "null_analysis":
                return self._handle_null_analysis(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "unique_values":
                return self._handle_unique_values(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "top_bottom":
                return self._handle_top_bottom(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "percentage":
                return self._handle_percentage(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "chart":
                return self._handle_chart(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "rolling_window":
                return self._handle_rolling_window(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "cumulative":
                return self._handle_cumulative(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "rank":
                return self._handle_rank(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            elif op == "date_range_analysis":
                return self._handle_date_range_analysis(
                    plan, question, df, filtered, date_range_info, custom_instructions, has_ci, schema
                )
            else:
                return (
                    "Operation '{}' is not implemented.".format(op),
                    {"operation": op, "plan": plan},
                    date_range_info,
                    None,
                )
        except Exception as exc:
            # Catch any execution errors and return them as a message
            return (
                "Error executing operation '{}': {}".format(op, str(exc)),
                {"operation": op, "error": str(exc)},
                date_range_info,
                None,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_column(self, col_name, custom_instructions):
        """Resolve a column name through the alias mapping."""
        if not col_name:
            return col_name
        if custom_instructions.column_aliases:
            for alias, actual in custom_instructions.column_aliases.items():
                if alias.lower() == col_name.lower():
                    return actual
        return col_name

    def _resolve_filters(self, filters, custom_instructions):
        """Resolve column names in filters through alias mappings."""
        resolved = []
        for f in filters:
            rf = f.copy()
            if "column" in rf:
                rf["column"] = self._resolve_column(rf["column"], custom_instructions)
            resolved.append(rf)
        return resolved

    def _infer_groupby(self, question, schema, df, custom_instructions):
        """
        Attempt to infer the groupby column from the question text
        when the planner did not provide one.
        """
        if not schema:
            return []

        cols = [c["name"] for c in schema.get("columns", [])]
        cols_lower = {c.lower(): c for c in cols}

        # Include aliases in the lookup
        if custom_instructions.column_aliases:
            for alias, actual in custom_instructions.column_aliases.items():
                cols_lower[alias.lower()] = actual

        q = question.strip().lower()
        # Look for "by <column_name>" at the end of the question
        match = re.search(r"\bby\s+([a-z0-9_ \-]+)\s*$", q)
        if match:
            candidate = match.group(1).strip()
            if candidate in cols_lower:
                return [cols_lower[candidate]]

        return []

    # ==================================================================
    # Operation handlers
    # ==================================================================

    def _handle_count_rows(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the count_rows operation."""
        result = {
            "operation": "count_rows",
            "count": int(len(filtered)),
            "total_rows": int(len(df)),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info
        md = "**Count:** {:,} (Total in dataset: {:,})".format(len(filtered), len(df))
        return md, result, date_range_info, None

    def _handle_groupby_count(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the groupby_count operation."""
        groupby_cols = plan.get("groupby") or []
        groupby_cols = [self._resolve_column(c, ci) for c in groupby_cols]

        if not groupby_cols:
            groupby_cols = self._infer_groupby(question, schema, filtered, ci)

        if not groupby_cols:
            return (
                "Could not determine the group-by column. "
                "Try: 'count by <exact column name>'.",
                {"operation": "groupby_count", "plan": plan},
                date_range_info,
                None,
            )

        valid = [c for c in groupby_cols if c in filtered.columns]
        if not valid:
            return (
                "Columns {} not found in dataset. Available: {}".format(
                    groupby_cols, list(filtered.columns)[:10]
                ),
                {"operation": "groupby_count", "plan": plan},
                date_range_info,
                None,
            )

        out = (
            filtered.groupby(valid, dropna=False)
            .size()
            .reset_index(name="count")
        )
        ascending = bool(plan.get("ascending", False))
        out = out.sort_values("count", ascending=ascending)

        top_n = plan.get("top_n")
        if top_n:
            out = out.head(int(top_n))

        # Add total row
        total_count = int(out["count"].sum())
        total_row = {
            col: ("**TOTAL**" if i == 0 else "") for i, col in enumerate(valid)
        }
        total_row["count"] = total_count
        out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)

        result = {
            "operation": "groupby_count",
            "groupby": valid,
            "result": out.to_dict(orient="records"),
            "total_count": total_count,
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Count by:** {} | **Total:** {:,}".format(", ".join(valid), total_count)
        return md, result, date_range_info, None

    def _handle_groupby_agg(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the groupby_agg operation."""
        groupby_cols = [
            self._resolve_column(c, ci) for c in (plan.get("groupby") or [])
        ]
        agg_col = self._resolve_column(plan.get("agg_column"), ci)
        agg_func = plan.get("agg_func", "sum")

        if not groupby_cols or not agg_col:
            return (
                "Group-by columns and aggregation column required.",
                {"operation": "groupby_agg"},
                date_range_info,
                None,
            )

        if agg_col not in filtered.columns:
            return (
                "Aggregation column '{}' not found.".format(agg_col),
                {"operation": "groupby_agg"},
                date_range_info,
                None,
            )

        valid_gb = [c for c in groupby_cols if c in filtered.columns]
        if not valid_gb:
            return (
                "Group-by columns not found in dataset.",
                {"operation": "groupby_agg"},
                date_range_info,
                None,
            )

        out = (
            filtered.groupby(valid_gb, dropna=False)[agg_col]
            .agg(agg_func)
            .reset_index()
        )
        agg_col_name = "{}_{}".format(agg_func, agg_col)
        out.columns = valid_gb + [agg_col_name]
        out = out.sort_values(agg_col_name, ascending=False)

        top_n = plan.get("top_n")
        if top_n:
            out = out.head(int(top_n))

        total_val = out[agg_col_name].sum()
        total_row = {
            col: ("**TOTAL**" if i == 0 else "")
            for i, col in enumerate(valid_gb)
        }
        total_row[agg_col_name] = total_val
        out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)

        result = {
            "operation": "groupby_agg",
            "groupby": valid_gb,
            "agg_column": agg_col,
            "agg_func": agg_func,
            "result": out.to_dict(orient="records"),
            "total": total_val,
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**{} of {} by:** {} | **Total:** {:,.2f}".format(
            agg_func.title(), agg_col, ", ".join(valid_gb), total_val
        )
        return md, result, date_range_info, None

    def _handle_filter_show(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the filter_show operation."""
        limit = plan.get("limit")
        cols = plan.get("columns")
        sort_by = plan.get("sort_by")
        ascending = bool(plan.get("ascending", True))

        tmp = filtered

        if sort_by:
            sort_by = self._resolve_column(sort_by, ci)
        if sort_by and sort_by in tmp.columns:
            tmp = tmp.sort_values(sort_by, ascending=ascending)

        if cols:
            cols_resolved = [self._resolve_column(c, ci) for c in cols]
            cols_ok = [c for c in cols_resolved if c in tmp.columns]
            if cols_ok:
                tmp = tmp[cols_ok]

        if limit:
            tmp = tmp.head(int(limit))

        result = {
            "operation": "filter_show",
            "total_matches": int(len(filtered)),
            "result_preview": tmp.head(Config.MAX_DISPLAY_ROWS).to_dict(orient="records"),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Matches:** {:,} | **Showing:** {}".format(
            len(filtered), min(len(tmp), Config.MAX_DISPLAY_ROWS)
        )
        return md, result, date_range_info, None

    def _handle_value_counts(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the value_counts operation."""
        col = self._resolve_column(plan.get("column"), ci)

        if not col or col not in filtered.columns:
            return (
                "Column '{}' not found.".format(col),
                {"operation": "value_counts"},
                date_range_info,
                None,
            )

        normalize = bool(plan.get("normalize", False))
        vc = filtered[col].value_counts(dropna=False, normalize=normalize)
        out = vc.reset_index()
        out.columns = [col, "percentage" if normalize else "count"]
        if normalize:
            out["percentage"] = (out["percentage"] * 100).round(2)

        top_n = plan.get("top_n")
        if top_n:
            out = out.head(int(top_n))

        if not normalize:
            total_count = int(out["count"].sum())
            total_row = {col: "**TOTAL**", "count": total_count}
            out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)

        result = {
            "operation": "value_counts",
            "column": col,
            "result": out.to_dict(orient="records"),
            "total_unique": int(filtered[col].nunique()),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Unique values in {}:** {:,}".format(col, filtered[col].nunique())
        return md, result, date_range_info, None

    def _handle_describe(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the describe operation."""
        cols = plan.get("columns")
        if cols:
            cols_resolved = [self._resolve_column(c, ci) for c in cols]
            cols_ok = [c for c in cols_resolved if c in filtered.columns]
            if cols_ok:
                desc = filtered[cols_ok].describe(include="all")
            else:
                desc = filtered.describe(include="all")
        else:
            desc = filtered.describe(include="all")

        desc_df = desc.transpose().reset_index().rename(columns={"index": "column"})

        result = {
            "operation": "describe",
            "describe": desc_df.to_dict(orient="records"),
            "total_rows": int(len(filtered)),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Describe computed.** | **Total rows:** {:,}".format(len(filtered))
        return md, result, date_range_info, None

    def _handle_correlation(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the correlation operation."""
        cols = plan.get("columns")
        method = plan.get("method", "pearson")

        if cols:
            cols_resolved = [self._resolve_column(c, ci) for c in cols]
            cols_ok = [c for c in cols_resolved if c in filtered.columns]
            num = filtered[cols_ok].select_dtypes(include=[np.number])
        else:
            num = filtered.select_dtypes(include=[np.number])

        if num.shape[1] < 2:
            return (
                "Not enough numeric columns for correlation.",
                {"operation": "correlation"},
                date_range_info,
                None,
            )

        corr = num.corr(method=method)
        result = {
            "operation": "correlation",
            "method": method,
            "result": corr.reset_index().to_dict(orient="records"),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Correlation ({})**".format(method)
        return md, result, date_range_info, None

    def _handle_crosstab(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the crosstab operation."""
        row_col = self._resolve_column(plan.get("row_column"), ci)
        col_col = self._resolve_column(plan.get("col_column"), ci)

        if not row_col or not col_col:
            return (
                "Row and column must be specified for crosstab.",
                {"operation": "crosstab"},
                date_range_info,
                None,
            )
        if row_col not in filtered.columns or col_col not in filtered.columns:
            return (
                "Crosstab columns not found in dataset.",
                {"operation": "crosstab"},
                date_range_info,
                None,
            )

        normalize = plan.get("normalize")
        ct = pd.crosstab(
            filtered[row_col], filtered[col_col],
            margins=True, normalize=normalize,
        )
        out = ct.reset_index()

        result = {
            "operation": "crosstab",
            "result": out.to_dict(orient="records"),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Crosstab:** {} x {}".format(row_col, col_col)
        return md, result, date_range_info, None

    def _handle_pivot_table(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the pivot_table operation."""
        index = plan.get("index") or []
        columns = plan.get("columns") or []
        values = plan.get("values")
        agg_func = plan.get("agg_func", "mean")

        index = [self._resolve_column(c, ci) for c in index]
        columns = [self._resolve_column(c, ci) for c in columns]
        values = self._resolve_column(values, ci)

        index = [c for c in index if c and c in filtered.columns]
        columns = [c for c in columns if c and c in filtered.columns]

        if not index:
            return (
                "Index columns required for pivot table.",
                {"operation": "pivot_table"},
                date_range_info,
                None,
            )

        pivot = pd.pivot_table(
            filtered,
            index=index,
            columns=columns if columns else None,
            values=values if values and values in filtered.columns else None,
            aggfunc=agg_func,
            margins=True,
        )
        out = pivot.reset_index()
        # Flatten multi-level column names
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [
                "_".join(str(c) for c in col).strip("_")
                for col in out.columns.values
            ]

        result = {
            "operation": "pivot_table",
            "result": out.to_dict(orient="records"),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Pivot table:** index={}, columns={}".format(index, columns)
        return md, result, date_range_info, None

    def _handle_duplicate_check(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the duplicate_check operation."""
        cols = plan.get("columns")
        if cols:
            cols = [self._resolve_column(c, ci) for c in cols]
            cols = [c for c in cols if c in filtered.columns]
        else:
            cols = None

        dupes = filtered[filtered.duplicated(subset=cols, keep=False)]
        dupe_count = len(dupes)

        preview = dupes.head(Config.MAX_DISPLAY_ROWS)
        result = {
            "operation": "duplicate_check",
            "duplicate_count": dupe_count,
            "total_rows": int(len(filtered)),
            "result_preview": preview.to_dict(orient="records"),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Duplicate rows:** {:,} out of {:,} total rows".format(
            dupe_count, len(filtered)
        )
        return md, result, date_range_info, None

    def _handle_null_analysis(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the null_analysis operation."""
        cols = plan.get("columns")
        if cols:
            cols = [self._resolve_column(c, ci) for c in cols]
            cols = [c for c in cols if c in filtered.columns]
            analysis_df = filtered[cols] if cols else filtered
        else:
            analysis_df = filtered

        null_counts = analysis_df.isnull().sum()
        total = len(analysis_df)
        null_pct = (null_counts / total * 100).round(2)

        out = pd.DataFrame({
            "column": null_counts.index,
            "null_count": null_counts.values,
            "null_percentage": null_pct.values,
            "non_null_count": (total - null_counts).values,
        })
        out = out.sort_values("null_count", ascending=False)

        result = {
            "operation": "null_analysis",
            "result": out.to_dict(orient="records"),
            "total_rows": total,
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Null analysis** | **Total rows:** {:,}".format(total)
        return md, result, date_range_info, None

    def _handle_unique_values(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the unique_values operation."""
        col = self._resolve_column(plan.get("column"), ci)
        if not col or col not in filtered.columns:
            return (
                "Column '{}' not found.".format(col),
                {"operation": "unique_values"},
                date_range_info,
                None,
            )

        unique_vals = filtered[col].dropna().unique().tolist()
        result = {
            "operation": "unique_values",
            "column": col,
            "unique_count": len(unique_vals),
            "values": [str(v) for v in unique_vals[:500]],
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Unique values in {}:** {:,}".format(col, len(unique_vals))
        return md, result, date_range_info, None

    def _handle_top_bottom(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the top_bottom operation."""
        col = self._resolve_column(plan.get("column"), ci)
        n = int(plan.get("n", 10))
        direction = plan.get("direction", "top")

        if not col or col not in filtered.columns:
            return (
                "Column '{}' not found.".format(col),
                {"operation": "top_bottom"},
                date_range_info,
                None,
            )

        numeric_col = pd.to_numeric(filtered[col], errors="coerce")
        temp = filtered.copy()
        temp["_sort_col"] = numeric_col

        if direction == "top":
            out = temp.nlargest(n, "_sort_col")
        else:
            out = temp.nsmallest(n, "_sort_col")

        out = out.drop(columns=["_sort_col"])

        result = {
            "operation": "top_bottom",
            "direction": direction,
            "n": n,
            "column": col,
            "result_preview": out.to_dict(orient="records"),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**{} {} by {}**".format(direction.title(), n, col)
        return md, result, date_range_info, None

    def _handle_percentage(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the percentage operation."""
        col = self._resolve_column(plan.get("column"), ci)

        if not col or col not in filtered.columns:
            return (
                "Column '{}' not found.".format(col),
                {"operation": "percentage"},
                date_range_info,
                None,
            )

        vc = filtered[col].value_counts(dropna=False)
        total = vc.sum()
        out = pd.DataFrame({
            col: vc.index,
            "count": vc.values,
            "percentage": (vc.values / total * 100).round(2),
        })

        result = {
            "operation": "percentage",
            "column": col,
            "result": out.to_dict(orient="records"),
            "total": int(total),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Percentage distribution of {}** | **Total:** {:,}".format(col, total)
        return md, result, date_range_info, None

    def _handle_chart(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the chart operation by delegating to the chart engine."""
        chart_spec = {
            "chart_type": plan.get("chart_type", "bar"),
            "x_column": self._resolve_column(plan.get("x_column"), ci),
            "y_column": self._resolve_column(plan.get("y_column"), ci),
            "groupby": [
                self._resolve_column(c, ci)
                for c in (plan.get("groupby") or [])
            ],
            "title": plan.get("title", "Chart"),
            "agg_func": plan.get("agg_func", "count"),
        }

        figure = generate_chart(filtered, chart_spec)

        if figure is None:
            return (
                "Could not generate the requested chart. "
                "Please check column names and data types.",
                {"operation": "chart", "plan": plan},
                date_range_info,
                None,
            )

        result = {
            "operation": "chart",
            "chart_type": chart_spec["chart_type"],
            "x_column": chart_spec["x_column"],
            "y_column": chart_spec["y_column"],
            "title": chart_spec["title"],
            "rows_used": int(len(filtered)),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**{} Chart:** {} | **Data points:** {:,}".format(
            chart_spec["chart_type"].replace("_", " ").title(),
            chart_spec["title"],
            len(filtered),
        )
        return md, result, date_range_info, figure

    def _handle_rolling_window(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the rolling_window operation."""
        col = self._resolve_column(plan.get("column"), ci)
        window = int(plan.get("window_size", 7))
        operation = plan.get("operation", "mean")

        if not col or col not in filtered.columns:
            return (
                "Column '{}' not found.".format(col),
                {"operation": "rolling_window"},
                date_range_info,
                None,
            )

        numeric_col = pd.to_numeric(filtered[col], errors="coerce")
        rolled = numeric_col.rolling(window=window).agg(operation)

        out = filtered.copy()
        out["rolling_{}_{}".format(operation, window)] = rolled

        result = {
            "operation": "rolling_window",
            "column": col,
            "window_size": window,
            "roll_operation": operation,
            "result_preview": out.head(Config.MAX_DISPLAY_ROWS).to_dict(orient="records"),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Rolling {} (window={}) on {}**".format(operation, window, col)
        return md, result, date_range_info, None

    def _handle_cumulative(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the cumulative operation."""
        col = self._resolve_column(plan.get("column"), ci)
        operation = plan.get("operation", "sum")

        if not col or col not in filtered.columns:
            return (
                "Column '{}' not found.".format(col),
                {"operation": "cumulative"},
                date_range_info,
                None,
            )

        numeric_col = pd.to_numeric(filtered[col], errors="coerce")

        if operation == "sum":
            cumulative = numeric_col.cumsum()
        elif operation == "count":
            cumulative = (~numeric_col.isna()).cumsum()
        elif operation == "mean":
            cumulative = numeric_col.expanding().mean()
        else:
            cumulative = numeric_col.cumsum()

        out = filtered.copy()
        out["cumulative_{}".format(operation)] = cumulative

        result = {
            "operation": "cumulative",
            "column": col,
            "cum_operation": operation,
            "result_preview": out.head(Config.MAX_DISPLAY_ROWS).to_dict(orient="records"),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Cumulative {} on {}**".format(operation, col)
        return md, result, date_range_info, None

    def _handle_rank(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the rank operation."""
        col = self._resolve_column(plan.get("column"), ci)
        ascending = bool(plan.get("ascending", True))
        method = plan.get("method", "min")

        if not col or col not in filtered.columns:
            return (
                "Column '{}' not found.".format(col),
                {"operation": "rank"},
                date_range_info,
                None,
            )

        numeric_col = pd.to_numeric(filtered[col], errors="coerce")
        out = filtered.copy()
        out["rank"] = numeric_col.rank(ascending=ascending, method=method).astype("Int64")
        out = out.sort_values("rank")

        result = {
            "operation": "rank",
            "column": col,
            "result_preview": out.head(Config.MAX_DISPLAY_ROWS).to_dict(orient="records"),
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Ranking by {}** ({})".format(
            col, "ascending" if ascending else "descending"
        )
        return md, result, date_range_info, None

    def _handle_date_range_analysis(self, plan, question, df, filtered, date_range_info, ci, has_ci, schema):
        """Handle the date_range_analysis operation."""
        date_col = self._resolve_column(plan.get("date_column"), ci)
        period = plan.get("groupby_period", "month")
        agg_col = self._resolve_column(plan.get("agg_column"), ci)
        agg_func = plan.get("agg_func", "count")

        if not date_col or date_col not in filtered.columns:
            return (
                "Date column '{}' not found.".format(date_col),
                {"operation": "date_range_analysis"},
                date_range_info,
                None,
            )

        temp = filtered.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=[date_col])

        # Create period column
        period_map = {"day": "D", "week": "W", "month": "M", "year": "Y"}
        freq = period_map.get(period, "M")
        temp["_period"] = temp[date_col].dt.to_period(freq).astype(str)

        if agg_col and agg_col in temp.columns and agg_func != "count":
            out = temp.groupby("_period")[agg_col].agg(agg_func).reset_index()
            out.columns = ["period", "{}_{}".format(agg_func, agg_col)]
        else:
            out = temp.groupby("_period").size().reset_index(name="count")
            out.columns = ["period", "count"]

        result = {
            "operation": "date_range_analysis",
            "result": out.to_dict(orient="records"),
            "period": period,
            "custom_instructions_applied": has_ci,
        }
        if date_range_info:
            result["date_range_used"] = date_range_info

        md = "**Date range analysis by {}** on {}".format(period, date_col)
        return md, result, date_range_info, None