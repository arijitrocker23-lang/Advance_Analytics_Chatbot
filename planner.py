# =============================================================================
# planner.py
# =============================================================================
# The planner is the first stage of the analysis pipeline.  It takes a
# natural-language question and produces a structured JSON plan that the
# executor can run against the DataFrame.
#
# The planner uses the Claude model via the Bedrock converse API.  The
# prompt is assembled from:
#   1. The system prompt (Config.PLANNER_SYSTEM_PROMPT)
#   2. Few-shot examples and meta-prompt instructions
#   3. Knowledge-base entries relevant to the question
#   4. User-provided custom instructions (aliases, rules, etc.)
#   5. Conversation context (for follow-up resolution)
#   6. The DataFrame schema
#   7. The user question
# =============================================================================

# =============================================================================
# planner.py
# =============================================================================
# Translates natural-language questions into structured JSON execution plans.
# =============================================================================

import json
import re
from typing import Any, Dict, Optional

from bedrock_client import BedrockClient
from config import Config
from conversation import ConversationContext
from few_shot import FewShotManager
from knowledge_base import KnowledgeBase
from data_models import CustomInstructions
from utils import (
    detect_general_question,
    get_date_context_string,
    safe_json_dumps,
)


class Planner:
    """
    Translates a user question into a JSON execution plan.
    """

    def __init__(self, bedrock_client, model_id, few_shot_manager, knowledge_base):
        self._bedrock = bedrock_client
        self._model_id = model_id
        self._few_shot = few_shot_manager
        self._kb = knowledge_base

    def set_model(self, model_id):
        """Allow the model to be switched at runtime."""
        self._model_id = model_id

    def plan(self, question, schema, conversation, custom_instructions):
        """
        Generate a JSON plan for the given question.

        If the question is a general (non-data) question, returns a plan
        with operation "general_question" without calling the model.
        """
        # -- Step 1: Check for general questions -------------------------
        general_type = detect_general_question(question)
        if general_type:
            return {
                "operation": "general_question",
                "question_type": general_type,
                "original_question": question,
            }

        # -- Step 2: Ensure data is loaded --------------------------------
        if schema is None:
            return {
                "operation": "unsupported",
                "reason": "No dataset loaded. Please upload a CSV file first.",
            }

        # -- Step 3: Assemble the user prompt ----------------------------
        # Start with the schema
        user_content = "SCHEMA:\n" + safe_json_dumps(schema, indent=2) + "\n\n"

        # Add current date/time context
        user_content += get_date_context_string() + "\n"

        # Add few-shot examples and meta-prompts
        user_content += self._few_shot.get_few_shot_prompt() + "\n"

        # Add knowledge-base entries relevant to this question
        kb_context = self._kb.get_context_for_prompt(question)
        if kb_context:
            user_content += kb_context + "\n"

        # Add custom instructions (aliases, rules, terminology)
        custom_context = custom_instructions.get_context_for_planner()
        if custom_context:
            user_content += custom_context + "\n"

        # Add conversation history for follow-up resolution
        user_content += conversation.get_context_for_prompt() + "\n\n"

        # Add the actual question
        user_content += "USER QUESTION:\n" + question + "\n\n"
        user_content += "Return ONLY the JSON plan."

        # -- Step 4: Call the model --------------------------------------
        try:
            plan_text = self._bedrock.call(
                model_id=self._model_id,
                system_prompt=Config.PLANNER_SYSTEM_PROMPT,
                user_text=user_content,
                max_tokens=2048,
                temperature=0.0,
            )
            # Strip markdown code fences if the model wraps the JSON
            plan_text = re.sub(r"^```json\s*", "", plan_text)
            plan_text = re.sub(r"\s*```$", "", plan_text)
            plan_text = plan_text.strip()
            return json.loads(plan_text)
        except json.JSONDecodeError:
            return {
                "operation": "unsupported",
                "reason": "The planner returned invalid JSON.",
            }
        except Exception as exc:
            return {
                "operation": "unsupported",
                "reason": "Planner error: {}".format(str(exc)),
            }