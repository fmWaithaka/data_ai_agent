"""
Retail DB Analyst Agent - Streamlit Application (Live API)

Main application module implementing the Streamlit interface for interacting with
the GenAI model and retail database tools.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

import streamlit as st
from google.genai import types

from config import get_settings
from services.ai_client import init_live_client
from services.tool_handler import ToolHandler
from utils.ui_components import (
    display_results,
    display_code_execution_results,
    render_chat_history,
    setup_sidebar
)
from utils.custom_types import StreamMessage

# --- Constants ---
MODEL_NAME = 'gemini-2.0-flash-exp'
STATUS_EXPANDED = True

# --- Configure Logging ---
logger = logging.getLogger(__name__)


def main() -> None:
    """Main application entry point configuring Streamlit and managing state."""
    configure_page()
    settings = get_settings()
    initialize_session_state()
    
    try:
        live_client = init_live_client(settings.google_api_key)
        if not live_client:
            st.stop()
            
        setup_sidebar()
        render_chat_interface(live_client)
        
    except Exception as e:
        logger.error("Critical application error: %s", e, exc_info=True)
        st.error("A critical error occurred. Please check the logs.")
        st.stop()


def configure_page() -> None:
    """Configures Streamlit page settings."""
    st.set_page_config(
        page_title="Retail DB Analyst Agent (Live API)",
        page_icon="🚀",
        layout="wide"
    )


def initialize_session_state() -> None:
    """Initializes Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "assistant_details" not in st.session_state:
        st.session_state.assistant_details = {}


def render_chat_interface(client: Any) -> None:
    """Renders the main chat interface and handles user interactions."""
    st.title("Retail DB Analyst Agent (Live API)")
    st.markdown("---")
    
    render_chat_history()
    
    if prompt := st.chat_input("Ask about data, request analysis or plots..."):
        handle_user_input(prompt, client)


def handle_user_input(prompt: str, client: Any) -> None:
    """Processes user input and coordinates AI response generation."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        process_ai_response(client, prompt)


def process_ai_response(client: Any, prompt: str) -> None:
    """Orchestrates AI response processing and output rendering."""
    response_container = st.container()
    status_container = st.status("Processing request...", expanded=STATUS_EXPANDED)
    
    current_turn = len(st.session_state.messages)
    
    try:
        tool_handler = ToolHandler(client)
        
        asyncio.run(
            process_response_stream(
                client,
                prompt,
                tool_handler,
                response_container,
                status_container
            )
        )
        
        update_chat_history(current_turn, tool_handler)
        
    except Exception as e:
        handle_processing_error(e, status_container, current_turn) 


async def process_response_stream(client: Any, prompt: str, tool_handler: ToolHandler,
                                response_container: Any, status_container: Any) -> None:
    """Processes the AI response stream asynchronously."""
    try:
        async for chunk in tool_handler.process_query(prompt):
            handle_stream_chunk(chunk, response_container, status_container)
            
    except RuntimeError as re:
        if "event loop" in str(re):
            await handle_nested_event_loop(client, prompt, tool_handler,
                                         response_container, status_container)
        else:
            raise


async def handle_nested_event_loop(client: Any, prompt: str, tool_handler: ToolHandler,
                                 response_container: Any, status_container: Any) -> None:
    """Handles nested event loop scenario for Streamlit compatibility."""
    import nest_asyncio
    nest_asyncio.apply()
    
    async for chunk in tool_handler.process_query(prompt):
        handle_stream_chunk(chunk, response_container, status_container)


def handle_stream_chunk(chunk: StreamMessage, response_container: Any,
                      status_container: Any) -> None:
    """Handles individual chunks from the response stream."""
    chunk_type = chunk.get("type")
    content = chunk.get("content")
    sql = chunk.get("sql")
    
    match chunk_type:
        case "status":
            status_container.update(label=content)
        case "text":
            response_container.markdown(content)
        case "tool_result" | "data":
            with status_container:
                display_results(st.container(), content, sql)
        case "code":
            with status_container:
                st.markdown("**Generated Code:**")
                st.code(content, language='python')
        case "code_result":
            with status_container:
                display_code_execution_results(st.container(), content)
        case "tool_error" | "error":
            with status_container:
                st.error(str(content))


def update_chat_history(current_turn: int, tool_handler: ToolHandler) -> None:
    """Updates chat history with the current interaction details."""
    st.session_state.assistant_details[current_turn] = tool_handler.stream_output
    st.session_state.messages.append({
        "role": "assistant",
        "content": tool_handler.final_text or "[AI Processed Request]"
    })


def handle_processing_error(error: Exception, status_container: Any,
                          current_turn: int) -> None:
    """Handles errors during response processing."""
    logger.error("Processing error: %s", error, exc_info=True)
    error_msg = f"Processing error: {str(error)}"
    
    status_container.update(label="Error", state="error", expanded=True)
    st.error(error_msg)
    
    st.session_state.assistant_details[current_turn] = [
        {"type": "error", "content": error_msg}
    ]
    st.session_state.messages.append({
        "role": "assistant",
        "content": "[Error Processing Request]"
    })


if __name__ == "__main__":
    main()