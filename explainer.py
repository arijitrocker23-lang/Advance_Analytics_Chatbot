# =============================================================================
# explainer.py
# =============================================================================
# The explainer is the final stage of the analysis pipeline.  It takes the
# structured result from the executor and asks Claude to produce a short,
# human-readable narrative summary.
#
# The explainer prompt includes:
#   - The user question
#   - Conversation context (for follow-up questions)
#   - The structured result
#   - Custom instructions (terminology, formatting rules)
# =============================================================================

from typing import Any, Dict, Optional

from bedrock_client import BedrockClient
from config import Config
from conversation import ConversationContext
from data_models import CustomInstructions
from utils import safe_json_dumps


class Explainer:
    """
    Generates a natural-language explanation of a structured analysis result.
    """

    def __init__(self, bedrock_client, model_id):
        self._bedrock = bedrock_client
        self._model_id = model_id

    def set_model(self, model_id):
        """Switch the model used for explanations."""
        self._model_id = model_id

    def explain(self, question, structured_result, conversation, custom_instructions):
        """
        Generate a short explanation of the structured_result.
        Returns an empty string if the result is not suitable for explanation.
        """
        if not structured_result:
            return ""

        op = structured_result.get("operation")
        if op in (None, "unsupported", "general_question"):
            return ""

        has_data = any(
            k in structured_result
            for k in (
                "result", "result_preview", "describe", "count",
                "duplicate_count", "unique_count", "values",
            )
        )
        if not has_data:
            return ""

        custom_context = custom_instructions.get_context_for_explainer()

        payload = {
            "question": question,
            "conversation_context": conversation.get_context_for_prompt(),
            "result": structured_result,
        }

        user_text = safe_json_dumps(payload, indent=2)
        if custom_context:
            user_text += custom_context

        try:
            # NOTE: only temperature, no top_p (Claude 4.x constraint)
            return self._bedrock.call(
                model_id=self._model_id,
                system_prompt=Config.EXPLAINER_SYSTEM_PROMPT,
                user_text=user_text,
                max_tokens=512,
                temperature=0.2,
            )
        except Exception:
            return ""