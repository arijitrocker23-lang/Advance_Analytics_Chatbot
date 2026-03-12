# =============================================================================
# config.py
# =============================================================================
# Central configuration file for the General-Purpose Analytical Chatbot.
# Contains all constants, model definitions, prompt templates, and default
# settings used across the application.
#
# NO PROXY SETTINGS -- calls go directly to AWS Bedrock.
# =============================================================================

import os
from typing import Dict, List

# Set AWS region
os.environ["AWS_REGION"] = "us-east-1"

# IMPORTANT: Remove any proxy environment variables that might interfere
# with direct AWS API calls.
for _proxy_key in ["http_proxy", "https_proxy", "HTTP_proxy", "HTTPS_proxy"]:
    os.environ.pop(_proxy_key, None)


class Config:
    """
    Application-wide configuration.
    All tunable parameters, model identifiers, prompt templates, and
    display limits live here.
    """

    # ------------------------------------------------------------------
    # AWS
    # ------------------------------------------------------------------
    AWS_REGION = "us-east-1"

    # ------------------------------------------------------------------
    # Available Claude models
    #
    # TESTED AND CONFIRMED WORKING model IDs:
    # - us.anthropic.claude-sonnet-4-5-20250929-v1:0  (works)
    # - us.anthropic.claude-opus-4-5-20251101-v1:0    (works)
    #
    # NOTE: These models require temperature OR top_p, NOT both.
    #       The bedrock_client.py handles this by only sending temperature.
    # ------------------------------------------------------------------
    AVAILABLE_MODELS = {
        "Claude Sonnet 4.5": {
            "inference_profile": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "description": "Fast and capable -- best balance of speed and quality",
        },
        "Claude Opus 4.5": {
            "inference_profile": "us.anthropic.claude-opus-4-5-20251101-v1:0",
            "model_id": "us.anthropic.claude-opus-4-5-20251101-v1:0",
            "description": "Most capable -- best for complex analysis and reasoning",
        },
    }

    # Default model
    DEFAULT_MODEL_NAME = "Claude Sonnet 4.5"

    # ------------------------------------------------------------------
    # File / display limits
    # ------------------------------------------------------------------
    MAX_FILE_SIZE_MB = 500.0
    MAX_DISPLAY_ROWS = 200
    CONVERSATION_CONTEXT_LIMIT = 10

    # ------------------------------------------------------------------
    # Chart settings
    # ------------------------------------------------------------------
    DEFAULT_CHART_HEIGHT = 500
    DEFAULT_CHART_WIDTH = 800
    CHART_COLOR_PALETTE = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
        "#3B1F2B", "#44BBA4", "#E94F37", "#393E41",
        "#8ECDDD", "#F4D35E",
    ]

    # ------------------------------------------------------------------
    # System prompts
    # ------------------------------------------------------------------

    PLANNER_SYSTEM_PROMPT = """You are a data analysis planning assistant.
Return ONLY valid JSON. No extra text, no markdown fencing, no explanation.

You receive:
- A JSON schema describing a pandas DataFrame.
- A user question.
- Previous conversation context (for follow-up questions).
- Current date/time context.
- USER-PROVIDED CONTEXT AND INSTRUCTIONS (if any) -- you MUST follow these.
- FEW-SHOT EXAMPLES showing correct plan formats.
- KNOWLEDGE BASE entries with domain knowledge.

STEP 1 -- Determine question type:
Check if the question is a GENERAL QUESTION (not data-related):
- Time / date / day of week
- Greetings (hello, hi)
- About the chatbot / help
- How are you

If general, return:
{"operation": "general_question", "question_type": "<type>", "original_question": "<question>"}
where type is one of: "time", "date", "day", "greeting", "about", "other"

STEP 2 -- For DATA ANALYSIS questions, produce a JSON plan with these fields:

Required:
  "operation": one of the operations listed below

Operations and their required fields:
  count_rows           -- filters (optional)
  groupby_count        -- groupby (list of column names), filters, top_n, ascending
  groupby_agg          -- groupby, agg_column, agg_func (sum/mean/min/max/median/std), filters, top_n
  filter_show          -- filters, columns (list, optional), limit, sort_by, ascending
  describe             -- filters, columns (optional)
  value_counts         -- column, filters, top_n, normalize (bool)
  correlation          -- columns (optional), method (pearson/spearman/kendall)
  crosstab             -- row_column, col_column, normalize (optional)
  date_range_analysis  -- date_column, groupby_period (day/week/month/year), agg_column, agg_func
  pivot_table          -- index, columns, values, agg_func
  rank                 -- column, groupby (optional), ascending, method
  cumulative           -- column, groupby (optional), operation (sum/count/mean)
  percentage           -- column, groupby (optional)
  rolling_window       -- column, window_size, operation (mean/sum/min/max)
  duplicate_check      -- columns (optional)
  null_analysis        -- columns (optional)
  unique_values        -- column
  top_bottom           -- column, n, direction (top/bottom), groupby (optional)

  chart                -- chart_type, x_column, y_column (optional), groupby (optional),
                          title (optional), filters (optional), agg_func (optional)
                          chart_type is one of: bar, line, scatter, pie, histogram,
                          heatmap, box, area, stacked_bar, grouped_bar

Filter format:
{
  "column": "<column_name>",
  "op": "eq"|"ne"|"gt"|"gte"|"lt"|"lte"|"contains"|"startswith"|"endswith"|"isin"|"notnull"|"isnull"|"between"|"relative_date",
  "value": ...
}

For relative_date:
{
  "column": "<date_column>",
  "op": "relative_date",
  "value": {"value": <number>, "unit": "days"|"weeks"|"months"|"years", "direction": "before"|"after"|"ago"}
}

Rules:
- Use the ENTIRE dataset (no sampling).
- Use conversation context for follow-up questions.
- Apply all user-provided instructions and column/value mappings.
- For relative date phrases ("last 30 days"), use relative_date filter.
- When the user asks for a chart/graph/plot/visualization, set operation to "chart".
- If the user asks "X by Y", default to groupby_count with groupby=["Y"].
- Always try to include totals where relevant.
- Return ONLY the JSON object. No other text.
"""

    EXPLAINER_SYSTEM_PROMPT = """You are a data explanation assistant.
Return a short, crisp answer.

Rules:
- 3-6 bullet points maximum.
- If a table is shown, do NOT repeat the whole table -- highlight key numbers only.
- If a chart is shown, describe the main trend or insight.
- If this is a follow-up question, connect to previous context in one line.
- If the user provided custom context or instructions, incorporate them.
- Use the user's terminology and formatting preferences when explaining.
- No extra sections or padding.
"""

    # ------------------------------------------------------------------
    # Helper class methods
    # ------------------------------------------------------------------

    @classmethod
    def get_model_inference_profile(cls, model_name):
        """Return the inference-profile ID for the given friendly model name."""
        info = cls.AVAILABLE_MODELS.get(model_name)
        if info:
            return info["inference_profile"]
        return cls.AVAILABLE_MODELS[cls.DEFAULT_MODEL_NAME]["inference_profile"]

    @classmethod
    def get_model_names(cls):
        """Return a list of all available model display names."""
        return list(cls.AVAILABLE_MODELS.keys())

    @classmethod
    def get_model_description(cls, model_name):
        """Return the human-readable description for a model."""
        return cls.AVAILABLE_MODELS.get(model_name, {}).get("description", "")