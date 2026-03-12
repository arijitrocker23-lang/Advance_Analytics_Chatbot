# =============================================================================
# analyzer.py
# =============================================================================
# The Analyzer is the top-level orchestrator.  It owns the DataFrame, schema,
# conversation context, custom instructions, knowledge base, and few-shot
# manager.  It wires together the planner, executor, and explainer into a
# single process_question() call that the UI invokes.
# =============================================================================

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from bedrock_client import BedrockClient
from config import Config
from conversation import ConversationContext
from data_loader import S3Helper, load_csv_from_bytes
from executor import Executor
from explainer import Explainer
from few_shot import FewShotManager
from knowledge_base import KnowledgeBase
from data_models import CustomInstructions
from planner import Planner
from utils import df_schema_summary


class Analyzer:
    """
    Top-level analysis engine.

    Responsibilities:
      - Manage the loaded DataFrame and its schema
      - Coordinate the planner -> executor -> explainer pipeline
      - Maintain conversation context across questions
      - Store custom instructions, few-shot examples, and KB entries
    """

    def __init__(self, model_name: str = Config.DEFAULT_MODEL_NAME) -> None:
        # -- Data state --------------------------------------------------
        self.df: Optional[pd.DataFrame] = None
        self.schema: Optional[Dict[str, Any]] = None
        self.date_columns: List[str] = []
        self.file_name: Optional[str] = None

        # -- Model state -------------------------------------------------
        self.model_name = model_name
        self.model_id = Config.get_model_inference_profile(model_name)

        # -- Shared components -------------------------------------------
        self._bedrock = BedrockClient()
        self.conversation = ConversationContext()
        self.custom_instructions = CustomInstructions()
        self.few_shot_manager = FewShotManager()
        self.knowledge_base = KnowledgeBase()

        # -- Pipeline stages ---------------------------------------------
        self._planner = Planner(
            self._bedrock, self.model_id,
            self.few_shot_manager, self.knowledge_base,
        )
        self._executor = Executor()
        self._explainer = Explainer(self._bedrock, self.model_id)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def set_model(self, model_name: str) -> None:
        """Switch the active model for all pipeline stages."""
        self.model_name = model_name
        self.model_id = Config.get_model_inference_profile(model_name)
        self._planner.set_model(self.model_id)
        self._explainer.set_model(self.model_id)

    def get_current_model_name(self) -> str:
        return self.model_name

    def get_current_model_id(self) -> str:
        return self.model_id

    # ------------------------------------------------------------------
    # Custom instructions
    # ------------------------------------------------------------------

    def set_custom_instructions(self, instructions: CustomInstructions) -> None:
        """Replace the current custom instructions."""
        self.custom_instructions = instructions

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_from_upload(self, file) -> Tuple[bool, str]:
        """Load a CSV from a Streamlit uploaded file object."""
        content = file.read()
        success, message, df = load_csv_from_bytes(content, file.name)
        if success:
            self._set_dataframe(df, file.name)
        return success, message

    def load_from_s3(self, s3_uri: str, s3_helper: S3Helper) -> Tuple[bool, str]:
        """Load a CSV from an S3 URI."""
        meta = s3_helper.get_metadata(s3_uri)
        if meta["size_mb"] > Config.MAX_FILE_SIZE_MB:
            return False, (
                f"S3 object is {meta['size_mb']:.2f} MB; "
                f"max allowed is {Config.MAX_FILE_SIZE_MB} MB."
            )
        content = s3_helper.download(s3_uri)
        file_name = f"{meta['bucket']}/{meta['key']}"
        success, message, df = load_csv_from_bytes(content, file_name)
        if success:
            self._set_dataframe(df, file_name)
        return success, message

    def _set_dataframe(self, df: pd.DataFrame, file_name: str) -> None:
        """Internal helper to update the DataFrame and derived state."""
        self.df = df
        self.file_name = file_name
        self.schema = df_schema_summary(df)
        self.date_columns = self.schema.get("date_columns", [])
        # Reset conversation when new data is loaded
        self.conversation.clear()

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process_question(
        self,
        question: str,
    ) -> Tuple[str, Dict[str, Any], str, Optional[go.Figure]]:
        """
        Process a user question through the full pipeline.

        Returns
        -------
        tuple of (markdown_text, structured_result, explanation, figure)
            - markdown_text: short summary for display
            - structured_result: full result dict (used for tables)
            - explanation: natural-language narrative from the explainer
            - figure: Plotly figure or None
        """
        # -- Step 1: Plan ------------------------------------------------
        plan = self._planner.plan(
            question=question,
            schema=self.schema,
            conversation=self.conversation,
            custom_instructions=self.custom_instructions,
        )

        # -- Step 2: Execute ---------------------------------------------
        result_md, structured_result, date_range_info, figure = (
            self._executor.execute(
                plan=plan,
                question=question,
                df=self.df,
                schema=self.schema,
                custom_instructions=self.custom_instructions,
            )
        )

        # -- Step 3: Skip explanation for general questions ---------------
        if plan.get("operation") == "general_question":
            return result_md, structured_result, "", figure

        # -- Step 4: Explain ---------------------------------------------
        explanation = self._explainer.explain(
            question=question,
            structured_result=structured_result,
            conversation=self.conversation,
            custom_instructions=self.custom_instructions,
        )

        # -- Step 5: Record the exchange in conversation history ----------
        summary: Dict[str, Any] = {
            "operation": structured_result.get("operation"),
        }
        for key in ("count", "total_matches", "total_count", "duplicate_count"):
            if key in structured_result:
                summary[key] = structured_result[key]

        self.conversation.add_exchange(
            question=question,
            plan=plan,
            result_summary=summary,
            date_range=date_range_info,
        )

        return result_md, structured_result, explanation, figure