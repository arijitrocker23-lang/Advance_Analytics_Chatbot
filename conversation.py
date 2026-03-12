# =============================================================================
# conversation.py
# =============================================================================
# Manages the running conversation context so that the planner can resolve
# follow-up questions ("show me the same thing but for last month") by
# referencing previous exchanges.
# =============================================================================

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from config import Config
from utils import safe_json_dumps


class ConversationContext:
    """
    A rolling window of recent question-answer exchanges.

    Each exchange stores the user question, the JSON plan that was
    generated, a summary of the result, and any date-range or filter
    metadata so that subsequent questions can reference prior context.
    """

    def __init__(self, max_history: int = Config.CONVERSATION_CONTEXT_LIMIT):
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []

        # Quick-access fields for the most recent exchange
        self.last_operation: Optional[Dict[str, Any]] = None
        self.last_date_range: Optional[Dict[str, Any]] = None
        self.last_filters: List[Dict[str, Any]] = []
        self.last_columns_used: Set[str] = set()

    # ------------------------------------------------------------------
    # Recording exchanges
    # ------------------------------------------------------------------

    def add_exchange(
        self,
        question: str,
        plan: Dict[str, Any],
        result_summary: Dict[str, Any],
        date_range: Optional[Dict[str, Any]],
    ) -> None:
        """Append a new exchange to the history, trimming if necessary."""
        # Collect all columns referenced by the plan
        columns_used: Set[str] = set()
        filters = plan.get("filters") or []
        for f in filters:
            if f.get("column"):
                columns_used.add(f["column"])
        groupby = plan.get("groupby") or []
        columns_used.update(groupby)
        if plan.get("column"):
            columns_used.add(plan["column"])

        self.history.append(
            {
                "question": question,
                "plan": plan,
                "result_summary": result_summary,
                "timestamp": datetime.now().isoformat(),
                "date_range": date_range,
                "filters_used": filters,
                "columns_used": list(columns_used),
            }
        )

        # Trim to the most recent entries
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Update quick-access fields
        self.last_operation = plan
        self.last_date_range = date_range
        self.last_filters = filters
        self.last_columns_used = columns_used

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def get_context_for_prompt(self) -> str:
        """
        Build a formatted string summarising the last few exchanges.
        This is injected into the planner prompt for follow-up resolution.
        """
        if not self.history:
            return "No previous conversation context."

        parts = ["Previous conversation context:"]
        # Include only the last 5 exchanges to keep the prompt compact
        for idx, exchange in enumerate(self.history[-5:], 1):
            parts.append(f"\n{idx}. User asked: {exchange['question']}")
            parts.append(
                f"   Operation: {exchange['plan'].get('operation', 'unknown')}"
            )
            parts.append(
                f"   Result summary: "
                f"{safe_json_dumps(exchange.get('result_summary', {}))}"
            )
            if exchange.get("date_range"):
                parts.append(
                    f"   Date range used: "
                    f"{safe_json_dumps(exchange['date_range'])}"
                )
            if exchange.get("filters_used"):
                parts.append(
                    f"   Filters used: "
                    f"{safe_json_dumps(exchange['filters_used'])}"
                )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all conversation state."""
        self.history.clear()
        self.last_operation = None
        self.last_date_range = None
        self.last_filters = []
        self.last_columns_used = set()