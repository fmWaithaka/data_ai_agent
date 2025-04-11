"""
Custom type definitions for type checking and documentation
"""

from typing import TypedDict, Optional, Any

class StreamMessage(TypedDict):
    """Standardized stream message format"""
    type: str
    content: Any
    sql: Optional[str]

class ToolResult(TypedDict):
    """Structure for tool execution results"""
    success: bool
    data: Any
    error: Optional[str]

class AIConfig(TypedDict):
    """Configuration for AI model interaction"""
    model: str
    temperature: float
    max_tokens: int