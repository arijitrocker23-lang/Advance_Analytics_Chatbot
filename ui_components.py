# =============================================================================
# ui_components.py
# =============================================================================
# All Streamlit UI rendering logic lives here.  This module defines the
# functions that draw the sidebar (data loading), the instructions panel,
# the knowledge base panel, the few-shot training panel, and the chat
# interface.
#
# The main app.py calls these functions to compose the full page layout.
# THIS FILE MUST NOT IMPORT FROM ITSELF.
# =============================================================================

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# -- Project imports --
# These import from other files in the chatbot directory.
# None of them import from ui_components, so there is no circular dependency.
from analyzer import Analyzer
from config import Config
from data_loader import S3Helper
from few_shot import FewShotManager
from knowledge_base import KnowledgeBase
from data_models import CustomInstructions, FewShotExample, KnowledgeBaseEntry
from utils import result_to_dataframe


# =============================================================================
# Custom CSS
# =============================================================================
# This CSS string is injected into the page via st.markdown() in app.py.
# It provides consistent, attractive styling across the entire application
# including the sidebar, chat messages, buttons, tabs, tables, and custom
# card components.
# =============================================================================

CUSTOM_CSS = """
<style>
    /* ------------------------------------------------------------------ */
    /* Global font                                                          */
    /* ------------------------------------------------------------------ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ------------------------------------------------------------------ */
    /* Sidebar styling                                                      */
    /* ------------------------------------------------------------------ */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #cccccc;
    }

    /* ------------------------------------------------------------------ */
    /* Chat message styling                                                 */
    /* ------------------------------------------------------------------ */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* ------------------------------------------------------------------ */
    /* Expander styling                                                     */
    /* ------------------------------------------------------------------ */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 14px;
    }

    /* ------------------------------------------------------------------ */
    /* Button styling                                                       */
    /* ------------------------------------------------------------------ */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    /* ------------------------------------------------------------------ */
    /* Tab styling                                                          */
    /* ------------------------------------------------------------------ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }

    /* ------------------------------------------------------------------ */
    /* Data table styling                                                    */
    /* ------------------------------------------------------------------ */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ------------------------------------------------------------------ */
    /* Card-like containers used for status display                          */
    /* ------------------------------------------------------------------ */
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 16px 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 12px;
    }
    .status-card h4 {
        margin: 0 0 4px 0;
        font-size: 14px;
        font-weight: 500;
        opacity: 0.9;
    }
    .status-card p {
        margin: 0;
        font-size: 20px;
        font-weight: 700;
    }

    /* ------------------------------------------------------------------ */
    /* Info banner shown on the welcome screen                               */
    /* ------------------------------------------------------------------ */
    .info-banner {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }

    /* ------------------------------------------------------------------ */
    /* Metric cards used in the sidebar to show row/column counts            */
    /* ------------------------------------------------------------------ */
    .metric-row {
        display: flex;
        gap: 12px;
        margin-bottom: 16px;
    }
    .metric-card {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-card .label {
        font-size: 12px;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 24px;
        font-weight: 700;
        color: #1a1a2e;
        margin-top: 4px;
    }
</style>
"""


# =============================================================================
# Session state initialisation
# =============================================================================

def initialize_session_state():
    """
    Ensure all required keys exist in st.session_state.
    This function is called once at the top of every page render in app.py.
    It creates the Analyzer, S3Helper, and all instruction-related state
    variables if they do not already exist.
    """
    # -- Core objects --
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = Analyzer(Config.DEFAULT_MODEL_NAME)

    if "s3_helper" not in st.session_state:
        st.session_state.s3_helper = S3Helper()

    # -- Chat state --
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # -- Data loading state --
    if "file_loaded" not in st.session_state:
        st.session_state.file_loaded = False

    if "current_file_name" not in st.session_state:
        st.session_state.current_file_name = None

    # -- Model selection state --
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = Config.DEFAULT_MODEL_NAME

    # -- Custom instruction fields --
    # Each field is stored separately in session state so that Streamlit
    # widgets can bind to them directly without needing a callback.
    if "inst_data_context" not in st.session_state:
        st.session_state.inst_data_context = ""

    if "inst_org_instructions" not in st.session_state:
        st.session_state.inst_org_instructions = []

    if "inst_column_aliases" not in st.session_state:
        st.session_state.inst_column_aliases = {}

    if "inst_value_mappings" not in st.session_state:
        st.session_state.inst_value_mappings = {}

    if "inst_formatting_rules" not in st.session_state:
        st.session_state.inst_formatting_rules = {}

    if "inst_terminology" not in st.session_state:
        st.session_state.inst_terminology = {}

    if "inst_business_rules" not in st.session_state:
        st.session_state.inst_business_rules = []

    if "instructions_saved" not in st.session_state:
        st.session_state.instructions_saved = False


def build_custom_instructions():
    """
    Build a CustomInstructions object from the current session state values.
    This is called whenever we need to pass instructions to the analyzer
    or display instruction status in the UI.
    """
    return CustomInstructions(
        data_context=st.session_state.inst_data_context,
        org_instructions=st.session_state.inst_org_instructions.copy(),
        column_aliases=st.session_state.inst_column_aliases.copy(),
        value_mappings={
            k: v.copy() for k, v in st.session_state.inst_value_mappings.items()
        },
        formatting_rules=st.session_state.inst_formatting_rules.copy(),
        terminology=st.session_state.inst_terminology.copy(),
        business_rules=st.session_state.inst_business_rules.copy(),
    )


# =============================================================================
# Sidebar: Data loading and dataset information
# =============================================================================

def render_sidebar():
    """
    Render the sidebar containing:
      - Application header
      - Data loading tabs (Local CSV and S3 CSV)
      - Dataset information (rows, columns, column details)
      - Active model and instruction status
      - Utility buttons (Clear Chat, Unload Data)
      - Tips section
    """
    with st.sidebar:
        # -- Application header --
        st.markdown(
            '<div style="text-align:center; padding: 10px 0;">'
            '<h2 style="color:#ffffff; margin:0;">Analytical Chatbot</h2>'
            '<p style="color:#aaaaaa; font-size:12px; margin:4px 0 0 0;">'
            'General-Purpose Data Analysis'
            '</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # -- Data loading section --
        st.markdown("### Load Data")

        # Two tabs: one for local file upload, one for S3
        tab_local, tab_s3 = st.tabs(["Local CSV", "S3 CSV"])

        # -- Local CSV upload tab --
        with tab_local:
            file = st.file_uploader(
                "Upload a CSV file",
                type=["csv"],
                help="Maximum file size: {} MB".format(
                    int(Config.MAX_FILE_SIZE_MB)
                ),
            )
            if file is not None:
                # Show file size to the user
                size_mb = file.size / (1024 * 1024)
                st.caption("File size: {:.2f} MB".format(size_mb))

                # Check file size limit
                if size_mb > Config.MAX_FILE_SIZE_MB:
                    st.error(
                        "File too large. Maximum allowed: {} MB".format(
                            Config.MAX_FILE_SIZE_MB
                        )
                    )
                else:
                    # Load button
                    if st.button(
                        "Load CSV", type="primary", key="btn_load_local"
                    ):
                        with st.spinner("Loading file..."):
                            ok, msg = (
                                st.session_state.analyzer.load_from_upload(file)
                            )
                        if ok:
                            # Update session state to reflect loaded data
                            st.session_state.file_loaded = True
                            st.session_state.current_file_name = file.name
                            # Reset chat history with a load confirmation message
                            st.session_state.chat_history = [
                                {"role": "assistant", "content": msg}
                            ]
                            st.rerun()
                        else:
                            st.error(msg)

        # -- S3 CSV loading tab --
        with tab_s3:
            st.markdown("Enter an S3 URI to load a CSV:")
            s3_uri = st.text_input(
                "S3 URI",
                placeholder="s3://your-bucket/path/to/file.csv",
                value="",
                key="s3_uri_input",
            )
            if st.button("Load from S3", type="primary", key="btn_load_s3"):
                if s3_uri.strip():
                    with st.spinner("Loading from S3..."):
                        try:
                            ok, msg = st.session_state.analyzer.load_from_s3(
                                s3_uri.strip(), st.session_state.s3_helper
                            )
                            if ok:
                                st.session_state.file_loaded = True
                                st.session_state.current_file_name = (
                                    s3_uri.strip()
                                )
                                st.session_state.chat_history = [
                                    {"role": "assistant", "content": msg}
                                ]
                                st.rerun()
                            else:
                                st.error(msg)
                        except Exception as exc:
                            st.error(str(exc))
                else:
                    st.warning("Please enter an S3 URI.")

        st.markdown("---")

        # -- Dataset information section --
        # Only shown when data is loaded
        if (
            st.session_state.file_loaded
            and st.session_state.analyzer.df is not None
        ):
            df = st.session_state.analyzer.df

            st.markdown("### Dataset Info")

            # Metric cards rendered as HTML for consistent styling
            st.markdown(
                '<div class="metric-row">'
                '<div class="metric-card">'
                '<div class="label">Rows</div>'
                '<div class="value">{:,}</div>'
                "</div>"
                '<div class="metric-card">'
                '<div class="label">Columns</div>'
                '<div class="value">{}</div>'
                "</div>"
                "</div>".format(len(df), len(df.columns)),
                unsafe_allow_html=True,
            )

            # Show the source file name
            st.caption(
                "Source: {}".format(st.session_state.current_file_name)
            )

            # Column list in a collapsible expander
            with st.expander("Column Details", expanded=False):
                for col in df.columns:
                    dtype_str = str(df[col].dtype)
                    null_count = int(df[col].isna().sum())
                    st.caption(
                        "{} ({}) - {} nulls".format(col, dtype_str, null_count)
                    )

            # Show the currently active model
            active_model = (
                st.session_state.analyzer.get_current_model_name()
            )
            st.markdown("**Model:** {}".format(active_model))

            # Show custom instructions status
            ci = build_custom_instructions()
            if not ci.is_empty():
                st.markdown("**Instructions:** Active")

            # -- Utility buttons --
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Clear Chat",
                    use_container_width=True,
                    key="btn_clear_chat",
                ):
                    st.session_state.chat_history = []
                    st.session_state.analyzer.conversation.clear()
                    st.rerun()
            with col2:
                if st.button(
                    "Unload Data",
                    use_container_width=True,
                    key="btn_unload",
                ):
                    # Reset all data-related state
                    st.session_state.analyzer.df = None
                    st.session_state.analyzer.schema = None
                    st.session_state.analyzer.file_name = None
                    st.session_state.file_loaded = False
                    st.session_state.current_file_name = None
                    st.session_state.chat_history = []
                    st.session_state.analyzer.conversation.clear()
                    st.rerun()

        # -- Tips section --
        st.markdown("---")
        st.markdown("### Tips")
        st.markdown(
            "- Upload a CSV to start analysing\n"
            "- Ask questions in natural language\n"
            "- Request charts: 'show a bar chart of X by Y'\n"
            "- Use relative dates: 'last 30 days'\n"
            "- Add few-shot examples to improve accuracy\n"
            "- Build a knowledge base for domain context"
        )


# =============================================================================
# Configuration panel (left column in main area)
# =============================================================================

def render_configuration_panel():
    """
    Render the left-side configuration panel with four tabs:
      - Model: select which Claude model to use
      - Instructions: custom context, rules, aliases, terminology
      - Knowledge Base: domain knowledge entries
      - Training: few-shot examples and meta-prompt instructions
    """
    st.markdown("## Configuration")

    tab_model, tab_instr, tab_kb, tab_train = st.tabs(
        ["Model", "Instructions", "Knowledge Base", "Training"]
    )

    # ==================================================================
    # TAB: Model Selection
    # ==================================================================
    with tab_model:
        st.markdown("### Model Selection")
        st.markdown("Choose the Claude model to use for analysis.")

        # Get the list of available model names from config
        model_names = Config.get_model_names()

        # Find the index of the currently selected model
        current_idx = (
            model_names.index(st.session_state.selected_model)
            if st.session_state.selected_model in model_names
            else 0
        )

        # Model dropdown
        selected = st.selectbox(
            "Select Model",
            options=model_names,
            index=current_idx,
            key="model_select_widget",
        )

        # Show model description and inference profile
        desc = Config.get_model_description(selected)
        profile = Config.get_model_inference_profile(selected)
        st.caption(desc)
        st.caption("Profile: {}".format(profile))

        # Apply model change if user selected a different model
        if selected != st.session_state.selected_model:
            st.session_state.selected_model = selected
            st.session_state.analyzer.set_model(selected)
            st.success("Switched to {}".format(selected))

        # Show currently active model
        st.markdown("---")
        active = st.session_state.analyzer.get_current_model_name()
        st.markdown("**Currently Active:** {}".format(active))

    # ==================================================================
    # TAB: Custom Instructions
    # ==================================================================
    with tab_instr:
        _render_instructions_tab()

    # ==================================================================
    # TAB: Knowledge Base
    # ==================================================================
    with tab_kb:
        _render_knowledge_base_tab()

    # ==================================================================
    # TAB: Training
    # ==================================================================
    with tab_train:
        _render_training_tab()


# =============================================================================
# Instructions tab (private helper)
# =============================================================================

def _render_instructions_tab():
    """
    Render the custom instructions configuration tab.
    Includes sections for:
      - Data context
      - Organisation instructions
      - Column aliases
      - Value mappings
      - Business rules
      - Terminology / glossary
      - Formatting preferences
      - Save / Reset buttons
    """
    st.markdown("### Custom Instructions")
    st.caption(
        "Configure context and rules that the chatbot will follow "
        "when analysing your data."
    )

    # ------------------------------------------------------------------
    # Data Context section
    # ------------------------------------------------------------------
    with st.expander("Data Context", expanded=False):
        st.markdown("Describe what your dataset represents:")
        data_context = st.text_area(
            "Data Context",
            value=st.session_state.inst_data_context,
            placeholder=(
                "Example: This dataset contains quarterly sales data "
                "for all regions including product details and revenue."
            ),
            height=80,
            label_visibility="collapsed",
            key="input_data_context",
        )
        # Update session state immediately so the value persists
        st.session_state.inst_data_context = data_context

    # ------------------------------------------------------------------
    # Organisation Instructions section
    # ------------------------------------------------------------------
    with st.expander("Organisation Instructions", expanded=False):
        st.markdown("Instructions the chatbot must always follow:")

        # Display existing instructions with delete buttons
        for i, instr in enumerate(st.session_state.inst_org_instructions):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.text("  {}. {}".format(i + 1, instr))
            with col2:
                if st.button("X", key="del_oi_{}".format(i)):
                    st.session_state.inst_org_instructions.pop(i)
                    st.rerun()

        # Input for adding a new instruction
        new_instr = st.text_input(
            "Add instruction",
            placeholder="Example: Always show currency values in USD",
            key="input_new_org_instr",
        )
        if st.button("Add Instruction", key="btn_add_oi"):
            if new_instr.strip():
                st.session_state.inst_org_instructions.append(
                    new_instr.strip()
                )
                st.rerun()

    # ------------------------------------------------------------------
    # Column Aliases section
    # ------------------------------------------------------------------
    with st.expander("Column Aliases", expanded=False):
        st.markdown("Map friendly names to actual column names:")

        # Display existing aliases with delete buttons
        for alias, actual in list(
            st.session_state.inst_column_aliases.items()
        ):
            col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
            with col1:
                st.text(alias)
            with col2:
                st.text(actual)
            with col3:
                if st.button("X", key="del_ca_{}".format(alias)):
                    del st.session_state.inst_column_aliases[alias]
                    st.rerun()

        # Inputs for adding a new alias
        c1, c2 = st.columns(2)
        with c1:
            alias_term = st.text_input(
                "Friendly name", key="input_alias_term"
            )
        with c2:
            alias_col = st.text_input(
                "Actual column", key="input_alias_col"
            )
        if st.button("Add Alias", key="btn_add_ca"):
            if alias_term.strip() and alias_col.strip():
                st.session_state.inst_column_aliases[
                    alias_term.strip()
                ] = alias_col.strip()
                st.rerun()

    # ------------------------------------------------------------------
    # Value Mappings section
    # ------------------------------------------------------------------
    with st.expander("Value Mappings", expanded=False):
        st.markdown("Map user terms to actual data values:")

        # Display existing mappings grouped by column
        for column, mappings in list(
            st.session_state.inst_value_mappings.items()
        ):
            st.markdown("**Column: {}**".format(column))
            for term, value in list(mappings.items()):
                c1, c2, c3 = st.columns([0.4, 0.4, 0.2])
                with c1:
                    st.text(term)
                with c2:
                    st.text(value)
                with c3:
                    if st.button(
                        "X",
                        key="del_vm_{}_{}".format(column, term),
                    ):
                        del st.session_state.inst_value_mappings[column][
                            term
                        ]
                        # Remove the column key if no mappings remain
                        if not st.session_state.inst_value_mappings[column]:
                            del st.session_state.inst_value_mappings[column]
                        st.rerun()

        # Inputs for adding a new value mapping
        c1, c2, c3 = st.columns(3)
        with c1:
            vm_col = st.text_input("Column", key="input_vm_col")
        with c2:
            vm_term = st.text_input("User term", key="input_vm_term")
        with c3:
            vm_val = st.text_input("Actual value", key="input_vm_val")
        if st.button("Add Mapping", key="btn_add_vm"):
            if vm_col.strip() and vm_term.strip() and vm_val.strip():
                if (
                    vm_col.strip()
                    not in st.session_state.inst_value_mappings
                ):
                    st.session_state.inst_value_mappings[
                        vm_col.strip()
                    ] = {}
                st.session_state.inst_value_mappings[vm_col.strip()][
                    vm_term.strip()
                ] = vm_val.strip()
                st.rerun()

    # ------------------------------------------------------------------
    # Business Rules section
    # ------------------------------------------------------------------
    with st.expander("Business Rules", expanded=False):
        st.markdown("Business rules applied during analysis:")

        # Display existing rules with delete buttons
        for i, rule in enumerate(st.session_state.inst_business_rules):
            c1, c2 = st.columns([0.9, 0.1])
            with c1:
                # Truncate long rules for display
                display = (
                    rule[:120] + "..." if len(rule) > 120 else rule
                )
                st.text("  {}. {}".format(i + 1, display))
            with c2:
                if st.button("X", key="del_br_{}".format(i)):
                    st.session_state.inst_business_rules.pop(i)
                    st.rerun()

        # Input for adding a new business rule
        new_rule = st.text_area(
            "Add business rule",
            placeholder=(
                "Example: Revenue should always be calculated as "
                "quantity * unit_price"
            ),
            key="input_new_br",
            height=60,
        )
        if st.button("Add Rule", key="btn_add_br"):
            if new_rule.strip():
                st.session_state.inst_business_rules.append(
                    new_rule.strip()
                )
                st.rerun()

    # ------------------------------------------------------------------
    # Terminology / Glossary section
    # ------------------------------------------------------------------
    with st.expander("Terminology / Glossary", expanded=False):
        st.markdown("Define domain-specific terms:")

        # Display existing terms with delete buttons
        for term, defn in list(st.session_state.inst_terminology.items()):
            c1, c2, c3 = st.columns([0.25, 0.6, 0.15])
            with c1:
                st.text(term)
            with c2:
                display = (
                    defn[:100] + "..." if len(defn) > 100 else defn
                )
                st.text(display)
            with c3:
                if st.button("X", key="del_tm_{}".format(term)):
                    del st.session_state.inst_terminology[term]
                    st.rerun()

        # Inputs for adding a new term
        c1, c2 = st.columns([0.3, 0.7])
        with c1:
            new_term = st.text_input("Term", key="input_tm_term")
        with c2:
            new_defn = st.text_input("Definition", key="input_tm_defn")
        if st.button("Add Term", key="btn_add_tm"):
            if new_term.strip() and new_defn.strip():
                st.session_state.inst_terminology[
                    new_term.strip()
                ] = new_defn.strip()
                st.rerun()

    # ------------------------------------------------------------------
    # Formatting Preferences section
    # ------------------------------------------------------------------
    with st.expander("Formatting Preferences", expanded=False):
        date_fmt = st.text_input(
            "Date Format",
            value=st.session_state.inst_formatting_rules.get(
                "date_format", ""
            ),
            placeholder="e.g. DD-MMM-YYYY",
            key="input_fmt_date",
        )
        num_fmt = st.text_input(
            "Number Format",
            value=st.session_state.inst_formatting_rules.get(
                "number_format", ""
            ),
            placeholder="e.g. thousands with K",
            key="input_fmt_num",
        )
        currency = st.text_input(
            "Currency Symbol",
            value=st.session_state.inst_formatting_rules.get(
                "currency", ""
            ),
            placeholder="e.g. USD",
            key="input_fmt_currency",
        )

        # Build the formatting rules dict from the input values
        new_fmt = {}
        if date_fmt.strip():
            new_fmt["date_format"] = date_fmt.strip()
        if num_fmt.strip():
            new_fmt["number_format"] = num_fmt.strip()
        if currency.strip():
            new_fmt["currency"] = currency.strip()
        st.session_state.inst_formatting_rules = new_fmt

    # ------------------------------------------------------------------
    # Save / Reset buttons
    # ------------------------------------------------------------------
    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button(
            "Save",
            type="primary",
            use_container_width=True,
            key="btn_save_instr",
        ):
            # Build the instructions object and pass it to the analyzer
            ci = build_custom_instructions()
            st.session_state.analyzer.set_custom_instructions(ci)
            st.session_state.instructions_saved = True
            st.success("Instructions saved.")

    with c2:
        if st.button(
            "Reset", use_container_width=True, key="btn_reset_instr"
        ):
            # Clear all instruction fields in session state
            st.session_state.inst_data_context = ""
            st.session_state.inst_org_instructions = []
            st.session_state.inst_column_aliases = {}
            st.session_state.inst_value_mappings = {}
            st.session_state.inst_formatting_rules = {}
            st.session_state.inst_terminology = {}
            st.session_state.inst_business_rules = []
            # Also clear the instructions in the analyzer
            st.session_state.analyzer.set_custom_instructions(
                CustomInstructions()
            )
            st.session_state.instructions_saved = False
            st.rerun()

    with c3:
        # Show count of active instruction items
        ci = build_custom_instructions()
        if not ci.is_empty():
            count = (
                (1 if ci.data_context else 0)
                + len(ci.org_instructions)
                + len(ci.column_aliases)
                + sum(len(v) for v in ci.value_mappings.values())
                + len(ci.formatting_rules)
                + len(ci.terminology)
                + len(ci.business_rules)
            )
            st.markdown("**{}** item(s)".format(count))
        else:
            st.markdown("No instructions set")


# =============================================================================
# Knowledge base tab (private helper)
# =============================================================================

def _render_knowledge_base_tab():
    """
    Render the knowledge base management tab.
    Users can add, view, and remove domain knowledge entries that the
    chatbot uses when generating analysis plans.
    """
    st.markdown("### Knowledge Base")
    st.caption(
        "Store domain knowledge, definitions, and reference information "
        "that the chatbot can use when analysing your data."
    )

    # Get the knowledge base from the analyzer
    kb = st.session_state.analyzer.knowledge_base

    # -- Display existing entries --
    entries = kb.get_all_entries()
    if entries:
        st.markdown("**{} entries** in knowledge base".format(len(entries)))
        for i, entry in enumerate(entries):
            with st.expander(
                "[{}] {}".format(entry.category, entry.title),
                expanded=False,
            ):
                st.markdown(entry.content)
                if entry.tags:
                    st.caption(
                        "Tags: {}".format(", ".join(entry.tags))
                    )
                if st.button("Remove", key="del_kb_{}".format(i)):
                    kb.remove_entry(i)
                    st.rerun()
    else:
        st.markdown("No entries yet. Add domain knowledge below.")

    # -- Add new entry form --
    st.markdown("---")
    st.markdown("**Add New Entry**")

    kb_title = st.text_input(
        "Title",
        placeholder="e.g. Revenue Calculation Formula",
        key="input_kb_title",
    )
    kb_content = st.text_area(
        "Content",
        placeholder=(
            "e.g. Revenue is calculated as "
            "quantity * unit_price * (1 - discount_rate)"
        ),
        height=80,
        key="input_kb_content",
    )
    c1, c2 = st.columns(2)
    with c1:
        kb_category = st.selectbox(
            "Category",
            options=[
                "general",
                "definition",
                "formula",
                "rule",
                "reference",
            ],
            key="input_kb_category",
        )
    with c2:
        kb_tags = st.text_input(
            "Tags (comma-separated)",
            placeholder="revenue, calculation, finance",
            key="input_kb_tags",
        )

    if st.button(
        "Add to Knowledge Base", type="primary", key="btn_add_kb"
    ):
        if kb_title.strip() and kb_content.strip():
            # Parse the comma-separated tags
            tags = [t.strip() for t in kb_tags.split(",") if t.strip()]
            entry = KnowledgeBaseEntry(
                title=kb_title.strip(),
                content=kb_content.strip(),
                category=kb_category,
                tags=tags,
            )
            kb.add_entry(entry)
            st.success("Added: {}".format(kb_title.strip()))
            st.rerun()
        else:
            st.warning("Title and content are required.")

    # -- Clear all button --
    if entries:
        st.markdown("---")
        if st.button("Clear All Entries", key="btn_clear_kb"):
            kb.clear()
            st.rerun()


# =============================================================================
# Training tab (private helper)
# =============================================================================

def _render_training_tab():
    """
    Render the few-shot training and meta-prompt management tab.
    Users can:
      - View built-in meta-prompts and few-shot examples
      - Add custom meta-prompts (high-level reasoning directives)
      - Add custom few-shot examples (question -> plan pairs)
      - Clear all custom training data
    """
    st.markdown("### Training")
    st.caption(
        "Improve the chatbot's accuracy by adding few-shot examples "
        "and meta-prompt instructions. Built-in examples are always "
        "included; custom examples are added on top."
    )

    # Get the few-shot manager from the analyzer
    fsm = st.session_state.analyzer.few_shot_manager

    # ==================================================================
    # Section: Meta-Prompt Instructions
    # ==================================================================
    st.markdown("#### Meta-Prompt Instructions")
    st.caption(
        "High-level directives that shape how the model reasons about "
        "your questions. Built-in instructions are always active."
    )

    # Display custom meta-prompts with delete buttons
    custom_mps = fsm.get_custom_meta_prompts()
    if custom_mps:
        st.markdown(
            "**{} custom instruction(s):**".format(len(custom_mps))
        )
        for i, mp in enumerate(custom_mps):
            c1, c2 = st.columns([0.9, 0.1])
            with c1:
                display = mp[:150] + "..." if len(mp) > 150 else mp
                st.text("  {}. {}".format(i + 1, display))
            with c2:
                if st.button("X", key="del_mp_{}".format(i)):
                    fsm.remove_meta_prompt(i)
                    st.rerun()

    # Input for adding a new meta-prompt
    new_mp = st.text_area(
        "Add meta-prompt instruction",
        placeholder=(
            "Example: When the user mentions 'active devices', always "
            "filter by status='active' AND last_seen within the last "
            "90 days."
        ),
        height=60,
        key="input_new_mp",
    )
    if st.button("Add Meta-Prompt", key="btn_add_mp"):
        if new_mp.strip():
            fsm.add_meta_prompt(new_mp.strip())
            st.success("Meta-prompt added.")
            st.rerun()

    # Show built-in meta-prompts in a collapsed expander
    # Import here to avoid any circular issues (this is a data list, not a class)
    from few_shot import BUILTIN_META_PROMPTS

    with st.expander(
        "View Built-in Meta-Prompts ({})".format(
            len(BUILTIN_META_PROMPTS)
        ),
        expanded=False,
    ):
        for i, mp in enumerate(BUILTIN_META_PROMPTS, 1):
            st.caption("{}. {}".format(i, mp))

    st.markdown("---")

    # ==================================================================
    # Section: Few-Shot Examples
    # ==================================================================
    st.markdown("#### Few-Shot Examples")
    st.caption(
        "Teach the chatbot by example. Provide a question and the "
        "expected JSON plan. The model learns the pattern and applies "
        "it to similar questions."
    )

    # Display custom examples with delete buttons
    custom_exs = fsm.get_custom_examples()
    if custom_exs:
        st.markdown(
            "**{} custom example(s):**".format(len(custom_exs))
        )
        for i, ex in enumerate(custom_exs):
            with st.expander(
                "[{}] {}".format(ex.category, ex.question[:80]),
                expanded=False,
            ):
                st.markdown("**Question:** {}".format(ex.question))
                st.code(ex.expected_plan, language="json")
                if ex.description:
                    st.caption(
                        "Description: {}".format(ex.description)
                    )
                if st.button("Remove", key="del_fs_{}".format(i)):
                    fsm.remove_example(i)
                    st.rerun()

    # -- Add new example form --
    st.markdown("**Add New Example:**")

    fs_question = st.text_input(
        "Question",
        placeholder=(
            "e.g. Show me the top 5 departments by headcount"
        ),
        key="input_fs_question",
    )
    fs_plan = st.text_area(
        "Expected JSON Plan",
        placeholder=(
            '{"operation": "groupby_count", "groupby": ["department"], '
            '"filters": [], "top_n": 5, "ascending": false}'
        ),
        height=80,
        key="input_fs_plan",
    )
    c1, c2 = st.columns(2)
    with c1:
        fs_category = st.selectbox(
            "Category",
            options=[
                "general",
                "aggregation",
                "filter",
                "chart",
                "date",
                "exploration",
            ],
            key="input_fs_category",
        )
    with c2:
        fs_desc = st.text_input(
            "Description (optional)",
            placeholder="Brief note about this example",
            key="input_fs_desc",
        )

    if st.button("Add Example", type="primary", key="btn_add_fs"):
        if fs_question.strip() and fs_plan.strip():
            # Validate that the expected plan is valid JSON
            try:
                json.loads(fs_plan.strip())
            except json.JSONDecodeError:
                st.error("The expected plan must be valid JSON.")
                return

            example = FewShotExample(
                question=fs_question.strip(),
                expected_plan=fs_plan.strip(),
                description=fs_desc.strip(),
                category=fs_category,
            )
            fsm.add_example(example)
            st.success("Example added.")
            st.rerun()
        else:
            st.warning("Question and expected plan are required.")

    # Show built-in examples in a collapsed expander
    from few_shot import BUILTIN_FEW_SHOT_EXAMPLES

    with st.expander(
        "View Built-in Examples ({})".format(
            len(BUILTIN_FEW_SHOT_EXAMPLES)
        ),
        expanded=False,
    ):
        for i, ex in enumerate(BUILTIN_FEW_SHOT_EXAMPLES, 1):
            st.caption(
                "{}. [{}] {}".format(i, ex.category, ex.question)
            )
            st.code(ex.expected_plan, language="json")

    # -- Clear custom training data button --
    st.markdown("---")
    if custom_exs or custom_mps:
        if st.button(
            "Clear All Custom Training Data", key="btn_clear_train"
        ):
            fsm.clear_custom()
            st.success("Custom training data cleared.")
            st.rerun()


# =============================================================================
# Chat interface (right column in main area)
# =============================================================================

def render_chat():
    """
    Render the main chat interface including:
      - Header with active model badge
      - Active instructions summary
      - Welcome screen when no data is loaded
      - Chat message history (text, tables, charts)
      - Chat input box
      - Question processing with full error handling
    """
    # -- Header --
    st.markdown(
        '<h1 style="margin-bottom:0;">Analytical Chatbot</h1>',
        unsafe_allow_html=True,
    )

    # Show which model is currently active
    active_model = st.session_state.analyzer.get_current_model_name()
    st.caption("Model: {}".format(active_model))

    # -- Active instructions summary banner --
    ci = build_custom_instructions()
    if not ci.is_empty() and st.session_state.instructions_saved:
        with st.expander("Active Instructions Summary", expanded=False):
            if ci.data_context:
                st.markdown(
                    "**Context:** {}...".format(ci.data_context[:120])
                )
            if ci.business_rules:
                st.markdown(
                    "**Business Rules:** {} active".format(
                        len(ci.business_rules)
                    )
                )
            if ci.terminology:
                st.markdown(
                    "**Terminology:** {} term(s)".format(
                        len(ci.terminology)
                    )
                )

    # -- Welcome screen when no data is loaded --
    if not st.session_state.file_loaded:
        st.markdown(
            '<div class="info-banner">'
            "<strong>Welcome!</strong> Upload a CSV file from the sidebar "
            "to start analysing your data. You can also ask general "
            "questions like 'what time is it?' or 'hello'."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("### What can I help you with?")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                "**General**\n"
                "- What time is it?\n"
                "- What is today's date?\n"
                "- What can you do?"
            )
        with c2:
            st.markdown(
                "**Analysis**\n"
                "- Count rows by category\n"
                "- Show top 10 values\n"
                "- Describe the dataset"
            )
        with c3:
            st.markdown(
                "**Visualisation**\n"
                "- Bar chart of sales by region\n"
                "- Pie chart of status distribution\n"
                "- Line chart of revenue over time"
            )

    # -- Render chat message history --
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            # Always show the text content
            st.markdown(msg["content"])

            # Show which model produced this response
            if msg.get("model_used"):
                st.caption("Model: {}".format(msg["model_used"]))

            # Show whether custom instructions were applied
            if msg.get("instructions_applied"):
                st.caption("Custom instructions applied")

            # Show data table if present in this message
            if msg.get("table") is not None:
                st.dataframe(
                    msg["table"],
                    use_container_width=True,
                    hide_index=True,
                )

            # Show chart if present in this message
            if msg.get("figure") is not None:
                st.plotly_chart(
                    msg["figure"],
                    use_container_width=True,
                )

    # -- Chat input box --
    question = st.chat_input("Ask a question about your data...")
    if not question:
        # No input yet, stop rendering here
        return

    # -- Add user message to history --
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    # -- Process the question through the full pipeline --
    # Wrap in try/except so that any error is shown as an assistant
    # message rather than crashing the UI with a blank screen.
    analyzer = st.session_state.analyzer

    try:
        result_md, structured_result, explanation, figure = (
            analyzer.process_question(question)
        )
    except Exception as exc:
        # Build a user-friendly error message
        error_message = (
            "An error occurred while processing your question.\n\n"
            "**Error:** {}\n\n"
            "**Type:** {}\n\n"
            "Please try rephrasing your question or check the data."
        ).format(str(exc), type(exc).__name__)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": error_message,
            "table": None,
            "figure": None,
            "instructions_applied": False,
            "model_used": analyzer.get_current_model_name(),
        })
        st.rerun()
        return

    # -- Determine if this was a general question (no table/chart needed) --
    is_general = structured_result.get("operation") == "general_question"

    # -- Extract table data for display --
    table_df = None if is_general else result_to_dataframe(structured_result)
    preview_df = (
        table_df.head(Config.MAX_DISPLAY_ROWS)
        if table_df is not None
        else None
    )

    # -- Build the assistant response text --
    assistant_text = result_md

    # Append date range info if the query involved date filtering
    if structured_result.get("date_range_used"):
        desc = structured_result["date_range_used"].get("description", "")
        assistant_text += "\n\n**Date range:** {}".format(desc)

    # Append the natural-language explanation if one was generated
    if explanation:
        assistant_text += "\n\n" + explanation

    # Check whether custom instructions were applied
    instructions_applied = structured_result.get(
        "custom_instructions_applied", False
    )

    # -- Add assistant message to history --
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_text,
        "table": preview_df,
        "figure": figure,
        "instructions_applied": instructions_applied,
        "model_used": analyzer.get_current_model_name(),
    })

    # -- Rerun to display the new messages immediately --
    st.rerun()