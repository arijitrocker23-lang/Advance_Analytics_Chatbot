# =============================================================================
# few_shot.py
# =============================================================================
# Manages few-shot examples and meta-prompt instructions that are prepended
# to the planner prompt.  This implements in-context learning: by showing
# the model several (question, plan) pairs, it learns the expected output
# format and analytical patterns without any fine-tuning.
#
# "Meta-prompt" instructions are high-level directives that shape the
# model's reasoning style (e.g. "always prefer groupby over filter when
# the user asks 'by X'").
# =============================================================================

from typing import Any, Dict, List

from data_models import FewShotExample


# ---------------------------------------------------------------------------
# Built-in few-shot examples.  These cover a variety of operation types
# so that the model sees at least one example of each before it has to
# generate a plan for a novel question.
# ---------------------------------------------------------------------------
BUILTIN_FEW_SHOT_EXAMPLES: List[FewShotExample] = [
    # -- groupby_count example -------------------------------------------
    FewShotExample(
        question="How many records are there by department?",
        expected_plan=(
            '{"operation": "groupby_count", "groupby": ["department"], '
            '"filters": [], "top_n": null, "ascending": false}'
        ),
        description="Simple group-by count",
        category="aggregation",
    ),
    # -- filter_show example ---------------------------------------------
    FewShotExample(
        question="Show me all records where status is active",
        expected_plan=(
            '{"operation": "filter_show", '
            '"filters": [{"column": "status", "op": "eq", "value": "active"}], '
            '"columns": null, "limit": 100, "sort_by": null, "ascending": true}'
        ),
        description="Simple equality filter",
        category="filter",
    ),
    # -- groupby_agg example ---------------------------------------------
    FewShotExample(
        question="What is the total revenue by region?",
        expected_plan=(
            '{"operation": "groupby_agg", "groupby": ["region"], '
            '"agg_column": "revenue", "agg_func": "sum", '
            '"filters": [], "top_n": null}'
        ),
        description="Aggregation with sum",
        category="aggregation",
    ),
    # -- chart example ---------------------------------------------------
    FewShotExample(
        question="Show me a bar chart of sales by category",
        expected_plan=(
            '{"operation": "chart", "chart_type": "bar", '
            '"x_column": "category", "y_column": "sales", '
            '"groupby": null, "title": "Sales by Category", '
            '"filters": [], "agg_func": "sum"}'
        ),
        description="Bar chart generation",
        category="chart",
    ),
    # -- relative_date filter example ------------------------------------
    FewShotExample(
        question="How many orders were placed in the last 30 days?",
        expected_plan=(
            '{"operation": "count_rows", '
            '"filters": [{"column": "order_date", "op": "relative_date", '
            '"value": {"value": 30, "unit": "days", "direction": "ago"}}]}'
        ),
        description="Relative date filtering",
        category="date",
    ),
    # -- value_counts example --------------------------------------------
    FewShotExample(
        question="What are the top 10 most common values in the city column?",
        expected_plan=(
            '{"operation": "value_counts", "column": "city", '
            '"filters": [], "top_n": 10, "normalize": false}'
        ),
        description="Value counts with top-n",
        category="exploration",
    ),
    # -- pie chart example -----------------------------------------------
    FewShotExample(
        question="Show a pie chart of the distribution of device types",
        expected_plan=(
            '{"operation": "chart", "chart_type": "pie", '
            '"x_column": "device_type", "y_column": null, '
            '"groupby": null, "title": "Device Type Distribution", '
            '"filters": [], "agg_func": "count"}'
        ),
        description="Pie chart generation",
        category="chart",
    ),
    # -- pivot table example ---------------------------------------------
    FewShotExample(
        question="Create a pivot table of average salary by department and level",
        expected_plan=(
            '{"operation": "pivot_table", "index": ["department"], '
            '"columns": ["level"], "values": "salary", "agg_func": "mean"}'
        ),
        description="Pivot table with mean aggregation",
        category="aggregation",
    ),
]


# ---------------------------------------------------------------------------
# Built-in meta-prompt instructions.  These are general reasoning
# directives that improve the quality and consistency of generated plans.
# ---------------------------------------------------------------------------
BUILTIN_META_PROMPTS: List[str] = [
    (
        "When the user asks 'X by Y', always use groupby_count or groupby_agg "
        "with groupby=['Y'] unless a specific aggregation function is mentioned."
    ),
    (
        "When the user asks for a chart, graph, plot, or visualisation, "
        "always set operation to 'chart' and choose the most appropriate chart_type."
    ),
    (
        "When generating groupby results, always sort by the count or aggregate "
        "value in descending order unless the user requests otherwise."
    ),
    (
        "For filter_show operations, default to showing the first 100 rows "
        "unless the user specifies a different limit."
    ),
    (
        "When the user mentions 'last N days/weeks/months', use relative_date "
        "filter on the most likely date column from the schema."
    ),
    (
        "Always include totals in groupby results by adding a TOTAL row at the end."
    ),
    (
        "If the question is ambiguous, prefer the interpretation that provides "
        "the most useful analytical insight."
    ),
    (
        "For percentage or proportion questions, use value_counts with "
        "normalize=true or the percentage operation."
    ),
]


class FewShotManager:
    """
    Manages few-shot examples and meta-prompt instructions.

    Built-in examples are always included.  Users can add additional
    custom examples and meta-prompts through the UI, which are stored
    in separate lists so they can be managed independently.
    """

    def __init__(self) -> None:
        # Custom examples added by the user at runtime
        self._custom_examples: List[FewShotExample] = []
        # Custom meta-prompts added by the user
        self._custom_meta_prompts: List[str] = []

    # ------------------------------------------------------------------
    # Example management
    # ------------------------------------------------------------------

    def add_example(self, example: FewShotExample) -> None:
        """Add a user-supplied few-shot example."""
        self._custom_examples.append(example)

    def remove_example(self, index: int) -> None:
        """Remove a custom example by index."""
        if 0 <= index < len(self._custom_examples):
            self._custom_examples.pop(index)

    def get_custom_examples(self) -> List[FewShotExample]:
        return list(self._custom_examples)

    def get_all_examples(self) -> List[FewShotExample]:
        """Return built-in examples followed by custom examples."""
        return BUILTIN_FEW_SHOT_EXAMPLES + self._custom_examples

    # ------------------------------------------------------------------
    # Meta-prompt management
    # ------------------------------------------------------------------

    def add_meta_prompt(self, prompt: str) -> None:
        """Add a user-supplied meta-prompt instruction."""
        self._custom_meta_prompts.append(prompt)

    def remove_meta_prompt(self, index: int) -> None:
        if 0 <= index < len(self._custom_meta_prompts):
            self._custom_meta_prompts.pop(index)

    def get_custom_meta_prompts(self) -> List[str]:
        return list(self._custom_meta_prompts)

    def get_all_meta_prompts(self) -> List[str]:
        """Return built-in meta-prompts followed by custom meta-prompts."""
        return BUILTIN_META_PROMPTS + self._custom_meta_prompts

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def get_few_shot_prompt(self) -> str:
        """
        Build the few-shot section of the planner prompt.

        This includes all meta-prompt instructions followed by all
        few-shot examples formatted as (Question -> Expected Plan) pairs.
        """
        parts: List[str] = []

        # -- Meta-prompt instructions ------------------------------------
        all_meta = self.get_all_meta_prompts()
        if all_meta:
            parts.append("\n=== META-PROMPT INSTRUCTIONS ===")
            for idx, mp in enumerate(all_meta, 1):
                parts.append(f"  {idx}. {mp}")
            parts.append("=== END META-PROMPT INSTRUCTIONS ===\n")

        # -- Few-shot examples -------------------------------------------
        all_examples = self.get_all_examples()
        if all_examples:
            parts.append("\n=== FEW-SHOT EXAMPLES ===")
            parts.append(
                "Below are example (question -> plan) pairs.  "
                "Follow the same JSON structure in your response.\n"
            )
            for idx, ex in enumerate(all_examples, 1):
                parts.append(f"Example {idx} [{ex.category}]:")
                if ex.description:
                    parts.append(f"  Description: {ex.description}")
                parts.append(f"  Question: {ex.question}")
                parts.append(f"  Plan: {ex.expected_plan}")
                parts.append("")
            parts.append("=== END FEW-SHOT EXAMPLES ===\n")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise custom state to a plain dictionary."""
        return {
            "custom_examples": [e.to_dict() for e in self._custom_examples],
            "custom_meta_prompts": self._custom_meta_prompts,
        }

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Restore custom state from a dictionary."""
        self._custom_examples = [
            FewShotExample.from_dict(d)
            for d in data.get("custom_examples", [])
        ]
        self._custom_meta_prompts = data.get("custom_meta_prompts", [])

    def clear_custom(self) -> None:
        """Remove all custom examples and meta-prompts (built-ins remain)."""
        self._custom_examples.clear()
        self._custom_meta_prompts.clear()