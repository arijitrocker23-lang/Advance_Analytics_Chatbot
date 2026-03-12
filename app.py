# =============================================================================
# app.py
# =============================================================================
# Main entry point. Run with: streamlit run app.py
# =============================================================================

import warnings

import streamlit as st

from ui_components import (
    CUSTOM_CSS,
    initialize_session_state,
    render_sidebar,
    render_configuration_panel,
    render_chat,
)

warnings.filterwarnings("ignore")


def main():
    # Page configuration must be the first Streamlit call
    st.set_page_config(
        page_title="Analytical Chatbot",
        page_icon="A",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialise session state
    initialize_session_state()

    # Render sidebar for data loading
    render_sidebar()

    # Main content: configuration panel on left, chat on right
    col_config, col_chat = st.columns([1, 2])

    with col_config:
        render_configuration_panel()

    with col_chat:
        render_chat()


if __name__ == "__main__":
    main()