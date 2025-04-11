"""
GenAI client initialization and management
"""

import logging
from typing import Optional, Any

from google.genai import types
import streamlit as st

logger = logging.getLogger(__name__)

def init_live_client(api_key: str) -> Optional[Any]:
    """Initialize and validate GenAI client with error handling"""
    try:
        from google import genai
        logger.info("Initializing Live Client for v1alpha API...")
        
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version='v1alpha')
        )
    except ImportError as e:
        logger.critical("GenAI SDK not installed: %s", e)
        st.error("Missing required GenAI SDK dependencies")
    except AttributeError as e:
        logger.error("SDK version mismatch: %s", e)
        st.error("Incompatible GenAI SDK version - check installation")
    except Exception as e:
        logger.exception("Client initialization failed: %s", e)
        st.error("Failed to initialize AI client")
    return None