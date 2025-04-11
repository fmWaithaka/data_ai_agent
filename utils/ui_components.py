"""
Streamlit UI components and rendering utilities
"""

import logging
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import streamlit as st

from utils.db_utils import db_manager

logger = logging.getLogger(__name__)

def display_results(container, result_data, execute_query_sql=None) -> None:
    """Render query results with intelligent formatting"""
    try:
        if isinstance(result_data, list) and result_data:
            if isinstance(result_data[0], dict):
                if "error" in result_data[0]:
                    container.error(f"Database Error: {result_data[0]['error']}")
                    return

                # Display SQL if available
                if execute_query_sql:
                    with container.expander("View SQL Query", expanded=False):
                        st.code(execute_query_sql, language="sql")

                # Attempt DataFrame display
                try:
                    container.dataframe(result_data)
                except Exception as df_error:
                    container.warning(f"DataFrame Error: {df_error}")
                    container.write(result_data)

                # Single metric detection
                if len(result_data) == 1 and len(result_data[0]) == 1:
                    key, value = next(iter(result_data[0].items()))
                    if isinstance(value, (int, float)):
                        formatted = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                        container.metric(label=key.replace('_', ' ').title(), value=formatted)

            else:  # List of non-dicts
                container.write(result_data)
        elif isinstance(result_data, str):
            if "error" in result_data.lower():
                container.error(result_data)
            else:
                container.markdown(result_data)
        elif not result_data:
            container.info("No data returned")
        else:
            container.write(result_data)
    except Exception as e:
        logger.error("Result display error: %s", e)
        container.error(f"Display Error: {str(e)}")

def display_code_execution_results(container, result_part_dict: Dict[str, Any]) -> None:
    """Render code execution results with plot detection"""
    try:
        outcome = str(result_part_dict.get('outcome', 'UNKNOWN')).upper()
        output = str(result_part_dict.get('output', ''))

        if outcome == 'OK':
            container.success("✅ Code executed successfully")
            if output:
                if _display_matplotlib_plot(container):
                    return
                container.markdown("**Code Output:**")
                container.code(output, language='text')
        else:
            container.error(f"Execution Failed: {outcome}")
            if output:
                container.code(output, language='text')
    except Exception as e:
        logger.error("Code display error: %s", e)
        container.error(f"Code Display Error: {str(e)}")

def _display_matplotlib_plot(container) -> bool:
    """Attempt to display matplotlib plot returns success status"""
    try:
        fig = plt.gcf()
        if fig and fig.axes:
            container.pyplot(fig)
            plt.close(fig)
            return True
    except Exception as e:
        logger.debug("Plot display attempt failed: %s", e)
    return False

def render_chat_history() -> None:
    """Render chat message history with state management"""
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant_message(idx)
            else:
                st.markdown(message["content"])

def setup_sidebar() -> None:
    """Configure and render sidebar components"""
    with st.sidebar:
        st.header("Database Schema Explorer")
        _render_schema_explorer()

def _render_schema_explorer() -> None:
    """Render the schema exploration components"""
    try:
        tables = _get_cached_tables()
        
        if not tables:
            st.warning("No tables found in database")
            return

        selected_table = st.selectbox(
            "Choose Table", 
            options=[""] + tables, 
            index=0,
            help="Select a table to view its schema"
        )
        
        if selected_table:
            _display_table_columns(selected_table)
            
    except Exception as e:
        st.error(f"Schema loading error: {str(e)}")

def _get_cached_tables() -> List[str]:
    """Get cached table list with error handling"""
    @st.cache_data(ttl=3600)
    def cached_show_tables():
        try:
            return db_manager.show_tables()  # Updated call
        except Exception as e:
            logger.error("Table load error: %s", e)
            return []
    return cached_show_tables()

def _display_table_columns(table_name: str) -> None:
    """Display columns for a selected table"""
    @st.cache_data(ttl=3600)
    def cached_get_columns(table: str):
        try:
            return db_manager.get_table_columns(table)  # Updated call
        except Exception as e:
            logger.error("Column load error: %s", e)
            return None

    with st.spinner(f"Loading {table_name} schema..."):
        columns = cached_get_columns(table_name)

    if columns:
        st.subheader(f"Table: `{table_name}`")
        st.dataframe(columns, use_container_width=True)
    elif columns is None:
        st.error("Failed to load columns")
    else:
        st.info(f"No schema information available for {table_name}")

def _render_assistant_message(message_idx: int) -> None:
    """Render detailed assistant message components"""
    if message_idx not in st.session_state.assistant_details:
        return

    container = st.container()
    for part in st.session_state.assistant_details[message_idx]:
        part_type = part.get("type")
        content = part.get("content")
        sql = part.get("sql")

        match part_type:
            case "text":
                container.markdown(content)
            case "data" | "tool_result":
                display_results(container, content, sql)
            case "code":
                container.markdown("**Generated Code:**")
                container.code(content, language="python")
            case "code_result":
                display_code_execution_results(container, content)
            case "error" | "tool_error":
                container.error(str(content))
            case _:
                logger.warning("Unknown message part type: %s", part_type)