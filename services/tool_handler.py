"""
Handles tool declarations and execution logic
"""

import asyncio
import logging
from typing import Dict, Any, AsyncGenerator, List

from google.genai import types
from utils.custom_types import StreamMessage
from utils.db_utils import show_tables, get_table_columns, execute_query 

logger = logging.getLogger(__name__)
MODEL_NAME = 'gemini-2.0-flash-exp' 

class ToolHandler:
    """Manages tool declarations and execution flow"""
    
    def __init__(self, client):
        self.client = client
        self.stream_output = []
        self.final_text = None
        self.available_functions = {  
            "show_tables": show_tables,
            "get_table_columns": get_table_columns,
            "execute_query": execute_query
        }
        self.function_declarations = self._init_tool_declarations()

    def _init_tool_declarations(self) -> list:
        """Initialize supported tool declarations"""
        try:
            return [
                types.FunctionDeclaration.from_callable(
                    client=self.client,
                    callable=func
                ) for func in self.available_functions.values()
            ]
        except Exception as e:
            logger.error("Tool declaration failed: %s", e)
            raise

    def _build_system_instruction(self) -> str:
        """System instruction for Advanced Data Analysis & Dashboard Preparation Agent"""
        return  """**1. Role & Objective:**
        You are an expert, collaborative Data Analyst Assistant. Your primary objective is to help the user explore, profile, clean, transform, and analyze data from the `retail_db` database (MySQL 8.0.35). Your insights and actions should directly support the user's goal of understanding the data thoroughly to build effective Power BI dashboards efficiently.

        **2. Environment & Tools:**
        * **Database:** You are connected to a MySQL 8.0.35 database named `retail_db`. Assume standard relational structures (e.g., `orders.order_customer_id` links to `customers.customer_id`).
        * **SQL Execution Tool:**
            * `execute_query(sql: str)`: Use this to execute **SELECT** queries. You MUST formulate appropriate SQL to perform tasks requested by the user.
            * **Constraint:** This tool CANNOT perform data modifications (INSERT, UPDATE, DELETE, DROP, etc.). Politely decline such requests.
        * **Schema Tools:**
            * `show_tables()`: Lists available tables. Use when needed to understand scope.
            * `get_table_columns(table_name: str)`: Gets schema details. Use when needed to formulate correct queries.
        * **Code Execution Tool:**
            * Built-in Python environment (includes `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`).
            * Use this *after* fetching data via `execute_query` when SQL is insufficient for: complex statistics (outliers, correlation), advanced data reshaping (pivoting), or creating visualizations (plots).
            * **Process:** Fetch data with SQL -> Generate Python code (import libraries, process data, `plt.show()` for plots) -> Execute code.

        **3. Core Task Guidance (Perform these based on user requests):**
        * **Exploration & Profiling:** Provide table lists (`show_tables`), column details (`get_table_columns`), row counts (`COUNT(*)`), distinct value counts (`COUNT(DISTINCT col)`), Min/Max/Avg values, and data type summaries. Use code execution for distribution plots (histograms, box plots).
        * **Data Quality & Cleaning:** Identify potential issues using SQL (`WHERE col IS NULL`, `GROUP BY ... HAVING COUNT(*) > 1` for duplicates). Use SQL `CASE` statements for basic cleaning/recoding. Use code execution for statistical outlier detection (e.g., IQR method). Suggest validation checks (e.g., "Compare total order revenue with sum of order_item subtotals?").
        * **Relational Analysis:** Construct SQL queries using appropriate `JOIN` types based on table relationships to answer cross-table questions.
        * **Aggregation & Summarization:** Generate SQL using `GROUP BY` with `SUM`, `AVG`, `COUNT`, etc., to provide summary statistics as requested.
        * **Trend & Time-Series:** Use SQL date functions (`YEAR`, `MONTH`, `DATE_FORMAT`) combined with aggregation for time-based analysis. Use code execution for plotting time-series line charts.
        * **Segmentation & Calculated Fields:** Use SQL `GROUP BY` for segmentation and `CASE` statements or arithmetic operations for calculated fields within queries.
        * **Advanced SQL:** Employ Window Functions (`RANK`, `ROW_NUMBER`, `LAG`), CTEs, or subqueries when necessary for complex analytical questions.

        **4. Interaction Workflow & Style:**
        1.  **Analyze Request:** Understand the user's goal. Check if it requires SQL, code execution, or both.
        2.  **Plan & Announce:** Briefly state your plan (e.g., "Okay, I'll run a query to get sales data, then generate a plot.").
        3.  **Execute Tool(s):** Call functions or execute code.
        4.  **Present Results:** Display data (e.g., tables using `execute_query` results) and visualizations (plots from code execution) clearly. Provide a concise textual summary of the key findings from the results. **Avoid `print()` statements for raw data within generated Python code; summarize results textually.**
        5.  **Recommend for Power BI:** *Crucially*, follow up findings with specific, actionable recommendations tailored to Power BI:
            * Suggest **KPIs** relevant to the result (e.g., "Track 'Average Order Value'").
            * Recommend **Power BI Visuals** (e.g., "Use a Clustered Bar Chart for this comparison", "A Scatter Plot could show correlation between X and Y"). Mention potential Axes/Values/Legends.
            * Suggest **Filters/Slicers** (e.g., "Add a Date slicer", "Filter by Department").
            * Mention **Data Quality Notes** if issues were found (e.g., "Address NULLs in [column] before loading").
            * Propose **Next Analytical Steps**.

        **Constraint:** Focus solely on SELECT queries and Python analysis/visualization. Be accurate, concise, and directly helpful to the data analyst's pre-dashboarding workflow.
        """

    async def process_query(self, prompt: str) -> AsyncGenerator[StreamMessage, None]:
        """Process user query through Live API session"""
        system_instruction = self._build_system_instruction()
        
        async with self.client.aio.live.connect(
            model=MODEL_NAME,
            config={
                "response_modalities": ["TEXT"],
                "system_instruction": {"parts": [{"text": system_instruction}]},
                "tools": [
                    {"code_execution": {}},
                    {"function_declarations": [d.to_json_dict() for d in self.function_declarations]}
                ]
            }
        ) as session:
            async for msg in self._process_session(session, prompt):
                yield msg

    async def _process_session(self, session, prompt: str) -> AsyncGenerator[StreamMessage, None]:
        """Handle Live API session processing"""
        await session.send(input=prompt, end_of_turn=True)
        
        async for msg in session.receive():
            if msg.text:
                yield {"type": "text", "content": msg.text}
            elif msg.tool_call:
                async for tool_msg in self._handle_tool_call(msg.tool_call, session):
                    yield tool_msg
            elif msg.server_content and msg.server_content.model_turn:  # Added from original
                async for part_msg in self._handle_model_turn(msg.server_content.model_turn):
                    yield part_msg

    async def _handle_model_turn(self, model_turn) -> AsyncGenerator[StreamMessage, None]:
        """Handle code execution parts from original app.py"""
        for part in model_turn.parts:
            if code := part.executable_code:
                yield {"type": "code", "content": code.code}
            elif result := part.code_execution_result:
                yield {"type": "code_result", "content": {
                    "outcome": str(result.outcome),
                    "output": str(result.output)
                }}

    async def _handle_tool_call(self, tool_call, session) -> AsyncGenerator[StreamMessage, None]:
        """Process tool call requests with original error handling"""
        tool_responses_to_send = []
        
        for fc in tool_call.function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args)
            sql = tool_args.get('sql', None)  # Capture SQL for display
            
            if tool_name not in self.available_functions:
                error_msg = f"Tool '{tool_name}' not found"
                logger.error(error_msg)
                yield {"type": "tool_error", "content": error_msg}
                continue

            try:
                tool_impl = self.available_functions[tool_name]
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: tool_impl(**tool_args))
                
                yield {"type": "tool_result", "content": result, "sql": sql}
                
                # Build response for API
                tool_responses_to_send.append(
                    types.FunctionResponse(
                        name=fc.name,
                        id=fc.id,
                        response={'result': result}
                    )
                )

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Tool {tool_name} error: {error_msg}")
                yield {"type": "tool_error", "content": error_msg}
                tool_responses_to_send.append(
                    types.FunctionResponse(
                        name=fc.name,
                        id=fc.id,
                        response={'error': error_msg}
                    )
                )

        if tool_responses_to_send:
            await session.send(input=types.LiveClientToolResponse(
                function_responses=tool_responses_to_send
            ))

