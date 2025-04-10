# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from utils import show_tables, get_table_columns, execute_query
import time
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import asyncio 
# --- Page Configuration ---
st.set_page_config(
    page_title="Retail DB Analyst Agent (Live API)",
    page_icon="🚀",
    layout="wide"
)

# --- Load Environment Variables & Secrets ---
load_dotenv()

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
# --- Check API Key ---
if not GOOGLE_API_KEY:
    st.error("Error: GOOGLE_API_KEY not found.")
    st.stop()

# --- Tool Definition & Mapping ---
# Dictionary mapping names to the actual Python functions
available_functions = {
    "show_tables": show_tables,
    "get_table_columns": get_table_columns,
    "execute_query": execute_query,
}

# --- Live Client Initialization (Cached) ---
@st.cache_resource
def init_live_client():
    """Initializes the GenAI Client for the Live API (v1alpha)."""
    print("Initializing Live Client for v1alpha API...")
    try:
        # Client instance required for Live API
        live_client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options=types.HttpOptions(api_version='v1alpha') # Critical for Live API
        )
        print("Live Client initialized successfully.")
        return live_client
    except AttributeError as ae:
         st.error(f"SDK Error: 'genai' module has no attribute 'Client'. Found version {genai.__version__}. Please ensure the correct library version is installed and the environment is clean. Details: {ae}")
         print(f"AttributeError during Live Client init: {ae}")
         return None
    except Exception as e:
        st.error(f"Error initializing Live Client: {e}")
        print(f"Error details during Live Client init: {e}")
        return None

live_client = init_live_client()

if not live_client:
    st.stop()

def display_results(container, result_data, execute_query_sql=None):
    # ... (no changes needed from previous version) ...
    if isinstance(result_data, list) and result_data:
        if isinstance(result_data[0], dict):
            if "error" in result_data[0]:
                 container.error(f"Database/Tool Error: {result_data[0]['error']}")
                 return
            try:
                 if execute_query_sql:
                      with container.expander("View SQL Query", expanded=False): # Start collapsed
                           st.code(execute_query_sql, language="sql")
                 container.dataframe(result_data)
            except Exception as df_e:
                 container.warning(f"Could not display data as DataFrame ({df_e}), showing raw list:")
                 container.write(result_data)
            # Heuristic for single value metric (optional refinement)
            if len(result_data) == 1 and len(result_data[0]) == 1:
                 key = list(result_data[0].keys())[0]
                 value = result_data[0][key]
                 if isinstance(value, (int, float)):
                      formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                      container.metric(label=key.replace('_', ' ').title(), value=formatted_value)
                      # Don't return here anymore, allow dataframe to also show
                      # return
        else: # List of non-dicts
            container.write(result_data)
    elif isinstance(result_data, str): # String results/errors
         if "error" in result_data.lower(): container.error(result_data)
         else: container.markdown(result_data)
    elif result_data is None or (isinstance(result_data, list) and not result_data):
         container.info("Tool execution returned no data.")
    else: # Fallback
        container.write(result_data)


def display_code_execution_results(container, result_part_dict):
    """Displays code execution results dict, attempting to render plots."""
    # Check for common success outcomes (adjust if needed based on actual API response)
    # Use .get() with default to handle potential missing keys gracefully
    outcome_str = str(result_part_dict.get('outcome', 'UNKNOWN')).upper() # Normalize to upper
    output_str = str(result_part_dict.get('output', ''))

    # Explicitly check for the most likely success indicator 'OK'
    if outcome_str == 'OK': # **** CORRECTED CHECK ****
        container.success("✅ Code executed successfully.")
        if output_str:
            captured_plot = False
            # Try capturing matplotlib plot
            try:
                current_fig = plt.gcf()
                if current_fig and current_fig.axes and any(ax.has_data() for ax in current_fig.axes):
                    container.pyplot(current_fig)
                    plt.clf(); plt.close(current_fig) # Clear and close figure
                    print("Displayed plot using st.pyplot(gcf).")
                    captured_plot = True
            except Exception as plt_e:
                print(f"Plot capture with gcf failed (may be normal): {plt_e}")

            # If no plot was captured, display output as code
            if not captured_plot:
                container.markdown("**Code Output:**")
                container.code(output_str, language='text')

    else: # Handle ERROR or other non-OK outcomes
        # container.error(f"⚠️ Code execution failed: {outcome_str}")
        if output_str:
            container.code(output_str, language='text')


# --- Async Generator for Live API Interaction ---
async def process_live_session(client, model_name, config, initial_message, tool_implementations):
    """Connects to Live API, sends message, yields outputs as they arrive."""
    print("Starting Live API session stream...")
    yield {"type": "status", "content": "Connecting to AI..."} # Initial status

    try:
        async with client.aio.live.connect(model=model_name, config=config) as session:
            print("Live API Session connected. Sending initial message...")
            yield {"type": "status", "content": "Sending message..."}
            await session.send(input=initial_message, end_of_turn=True)

            async for msg in session.receive():
                print(f"Stream received msg part type: {type(msg)}") # Debug
                # --- Parse Message Parts and Yield ---
                if text := msg.text:
                    yield {"type": "text", "content": text}
                elif tool_call := msg.tool_call:
                    yield {"type": "status", "content": f"AI requested tool: `{tool_call.function_calls[0].name}`..."} # Show first call name
                    tool_responses_to_send = []
                    # Process all function calls received in this message part
                    for fc in tool_call.function_calls:
                        tool_name = fc.name
                        tool_args = dict(fc.args)
                        sql = tool_args.get('sql') # Get SQL if present
                        yield {"type": "tool_call", "content": {"name": tool_name, "args": tool_args}} # Yield the call info

                        if tool_name in tool_implementations:
                            tool_impl = tool_implementations[tool_name]
                            try:
                                print(f"Executing tool: {tool_name}")
                                # Run synchronous DB calls in executor
                                loop = asyncio.get_running_loop()
                                yield {"type": "status", "content": f"Executing tool `{tool_name}`..."}
                                if sql: yield {"type": "status", "content": f"Executing SQL:\n```sql\n{sql}\n```"}
                                result = await loop.run_in_executor(None, lambda: tool_impl(**tool_args))
                                print(f"Tool {tool_name} executed successfully.")
                                tool_response_part = types.FunctionResponse(name=fc.name, id=fc.id, response={'result': result})
                                # Yield the *result* for immediate display
                                yield {"type": "tool_result", "content": result, "sql": sql}
                            except Exception as e:
                                print(f"Error executing tool {tool_name}: {e}")
                                yield {"type": "status", "content": f"Error executing tool `{tool_name}`"}
                                result = {"error": str(e)}
                                tool_response_part = types.FunctionResponse(name=fc.name, id=fc.id, response=result)
                                yield {"type": "tool_error", "content": str(e)} # Yield error info
                        else: # Unknown function
                             print(f"Error: Unknown tool requested: {tool_name}")
                             yield {"type": "status", "content": f"Error: Unknown tool `{tool_name}` requested"}
                             result = {"error": f"Tool '{tool_name}' not found."}
                             tool_response_part = types.FunctionResponse(name=fc.name, id=fc.id, response=result)
                             yield {"type": "tool_error", "content": f"Unknown tool: {tool_name}"}
                        tool_responses_to_send.append(tool_response_part)

                    # Send tool responses back
                    if tool_responses_to_send:
                         print("Sending tool responses back to Live API...")
                         yield {"type": "status", "content": "Sending tool results back to AI..."}
                         tool_response_msg = types.LiveClientToolResponse(function_responses=tool_responses_to_send)
                         await session.send(input=tool_response_msg)
                         print("Tool responses sent.")

                elif msg.server_content and msg.server_content.model_turn:
                    # Handle code execution parts
                    for part in msg.server_content.model_turn.parts:
                        if code := part.executable_code:
                            print("Received executable code from AI.")
                            yield {"type": "status", "content": "AI generated code..."}
                            yield {"type": "code", "content": code.code}
                        elif result := part.code_execution_result:
                            print(f"Received code execution result: Outcome={result.outcome}")
                            yield {"type": "status", "content": "Processing code execution results..."}
                            result_info = {"outcome": str(result.outcome), "output": str(result.output)}
                            yield {"type": "code_result", "content": result_info}
                        else:
                             print(f"Received other server_content part: {type(part)}")
                else:
                     print(f"Received unhandled message type: {type(msg)}")

    except Exception as e:
        print(f"Error during Live API session processing: {e}")
        yield {"type": "error", "content": f"An error occurred during the session: {e}"}

    print("Live API session stream finished.")
    yield {"type": "status", "content": "Processing complete."}

# --- Streamlit UI ---
st.title("Retail DB Analyst Agent (Live API)")
st.write("""
Ask questions, request analysis, or generate plots about the `retail_db` database.
The AI uses the Live API with SQL tools and Python Code Execution.
""")
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    # (Sidebar code remains unchanged)
    st.header("Database Schema")
    st.markdown("Explore the tables and columns available.")
    @st.cache_data(ttl=3600)
    def cached_show_tables(): return show_tables()
    @st.cache_data(ttl=3600)
    def cached_get_table_columns(table_name): return get_table_columns(table_name)
    tables = []
    try:
        with st.spinner("Loading tables..."): tables = cached_show_tables()
    except Exception as e: st.error(f"Error loading tables: {e}")
    if tables:
        selected_table = st.selectbox("Select a table:", options=[""] + tables, index=0)
        if selected_table:
            st.subheader(f"Schema: `{selected_table}`")
            try:
                with st.spinner(f"Fetching columns..."): columns = cached_get_table_columns(selected_table)
                if columns: st.dataframe(columns, use_container_width=True)
                elif isinstance(columns, list): st.info(f"No schema info for `{selected_table}`.")
                else: st.warning(f"Could not retrieve columns for `{selected_table}`.")
            except Exception as e: st.error(f"Error fetching columns: {e}")
    elif isinstance(tables, list): st.warning("No tables found.")
    else: st.error("Failed to retrieve tables.")


# --- Main Chat Area ---
st.header("Chat Interaction")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [] # 
if "assistant_turn_details" not in st.session_state:
     st.session_state.assistant_turn_details = {} 

# --- Display Chat History ---
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and idx in st.session_state.assistant_turn_details:
            # Render detailed parts if available
            turn_container = st.container()
            for part_info in st.session_state.assistant_turn_details[idx]:
                content_type = part_info.get("type", "unknown")
                content = part_info.get("content", "")
                sql = part_info.get("sql")
                if content_type == "text": turn_container.markdown(content)
                elif content_type == "data" or content_type == "tool_result": display_results(turn_container, content, sql)
                elif content_type == "code": turn_container.markdown("**Generated Code:**"); turn_container.code(content, language='python')
                elif content_type == "code_result": display_code_execution_results(turn_container, content)
                elif content_type == "tool_error" or content_type == "error": turn_container.error(str(content))

        elif isinstance(message["content"], str): # Default for user messages
            st.markdown(message["content"])
        else: # Fallback
             st.write(message["content"])


# --- Handle New User Input ---
# --- Handle New User Input ---
if prompt := st.chat_input("Ask about data, request analysis or plots..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Process using Live API ---
    with st.chat_message("assistant"):
        final_response_container = st.container()
        status_container = st.status("Processing request...", expanded=True)

        try:
            final_text_response_content = None
            all_outputs_for_history = []
            current_turn_index = len(st.session_state.messages)
            model_name = 'gemini-2.0-flash-exp' 
            try:
                show_tables_tool_decl = types.FunctionDeclaration.from_callable(client=live_client, callable=show_tables)
                get_cols_tool_decl = types.FunctionDeclaration.from_callable(client=live_client, callable=get_table_columns)
                execute_query_tool_decl = types.FunctionDeclaration.from_callable(client=live_client, callable=execute_query)
                db_function_declarations = [ d.to_json_dict() for d in [show_tables_tool_decl, get_cols_tool_decl, execute_query_tool_decl]]
            except Exception as decl_e: st.error(f"FuncDecl Error: {decl_e}"); raise decl_e

            system_instruction = """You are an expert Data Analyst Assistant working with a retail MySQL database (version 8.0.35) named `retail_db`.
            Your primary objective is to help a data analyst explore, clean, transform, analyze data, and generate insights in preparation for building Power BI dashboards.

            **Core Capabilities & Tools:**

            1.  **Database Interaction (Use these Functions):**
                * `show_tables()`: Lists all tables.
                * `get_table_columns(table_name: str)`: Shows schema (columns, types) for a table.
                * `execute_query(sql: str)`: Executes **SELECT** SQL queries ONLY. You can and should construct complex queries involving:
                    * Filtering (`WHERE`) & Sorting (`ORDER BY`, `LIMIT`).
                    * Joins (`INNER JOIN`, `LEFT JOIN`) between related tables (e.g., orders to customers, order_items to products).
                    * Aggregations (`GROUP BY` with `SUM`, `AVG`, `COUNT`, `MIN`, `MAX`).
                    * Data Cleaning/Transformation within SQL (`DISTINCT`, `CASE` statements, arithmetic operations, date/string functions).
                    * Window Functions (`RANK`, `ROW_NUMBER`, `LAG`, `LEAD`, running totals, etc.).
                    * Subqueries and Common Table Expressions (CTEs).
                * **Constraint:** Never attempt `INSERT`, `UPDATE`, `DELETE`, `DROP`, or other modification SQL via `execute_query`. State you cannot perform such actions if asked.

            2.  **Code Execution (Use this Built-in Tool):**
                * You have a **Python Code Execution** environment with `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`.
                * Use this tool *after* retrieving necessary data with `execute_query`, for tasks like:
                    * **Visualization:** Generate plots (histograms, bar charts, line charts, scatter plots, box plots) using `matplotlib.pyplot` and `seaborn`. **Crucially, call `plt.show()` to ensure the plot is rendered and captured.**
                    * **Statistical Analysis:** Calculate correlations, identify outliers (e.g., using IQR method in pandas/numpy), perform basic statistical summaries beyond SQL aggregates.
                    * **Complex Data Manipulation:** Reshape data (e.g., pivot/unpivot using `pandas`) if needed for analysis or visualization.
                * **Process:** Import required libraries inside your code block. Assume the data fetched by `execute_query` is available (often implicitly passed or needs reference, be clear in your generated code's comments if necessary).

            **Mandatory Workflow & Response Structure:**

            1.  **Understand & Plan:** Analyze request, decide SQL/Python, ask clarifying questions if essential.
            2.  **Execute Tool(s):** Announce tool usage. Call DB functions or generate/execute Python code. **IMPORTANT: When generating Python code, do NOT use `print()` to output intermediate data structures (like lists or DataFrames) if the result is captured in a variable or is the final step before visualization. Rely on your final text response to summarize data findings.** Use `print()` only for necessary status updates or final textual results within the code block if appropriate.
            3.  **Analyze Results:** Interpret data from SQL or output/plot from code. Note quality issues.
            4.  **Present Findings:** Clearly display results. Use Markdown for explanations. Render data tables/metrics/plots as appropriate via the tool results or code execution output. Explicitly state when showing a plot.
            5.  **Provide Power BI Recommendations:** Based on findings, proactively suggest KPIs, visualizations, filters, quality notes, and next steps suitable for Power BI.
                * **Relevant KPIs:** What key metrics does this analysis suggest tracking?
                * **Visualization Suggestions:** What specific Power BI chart types (Bar, Line, Scatter, Map, Card, Treemap, etc.) would best represent *this specific data*? Mention potential axes or measures.
                * **Dashboard Filters/Slicers:** What interactive filters would be useful for exploring this data further in Power BI (e.g., Date slicer, Category filter)?
                * **Data Quality Notes:** Briefly mention any potential data quality concerns found that the analyst should address before loading into Power BI (e.g., "Consider handling the null values found in the price column.").
                * **Next Steps:** Suggest logical follow-up questions or analyses.
            **Focus on providing actionable data insights and relevant dashboarding recommendations.**

            **Example Interaction Snippet (Focus on Response):**
            *User:* "Show the trend of total monthly revenue for completed orders."
            *You:* "Okay, I will calculate the total monthly revenue for completed orders using an SQL query."
            *[Calls execute_query with appropriate SQL joining orders, order_items, filtering status='COMPLETE', grouping by YEAR/MONTH of order_date, summing order_item_subtotal]*
            *[Displays results table/dataframe]*
            "Here is the total revenue per month for completed orders. We can see a general upward trend over the period with seasonality peaking in [Month].
            **Power BI Recommendations:**
            * **KPI:** Track 'Total Monthly Revenue' and 'Month-over-Month Revenue Growth %'.
            * **Visualization:** A Line Chart in Power BI would effectively visualize this monthly revenue trend. Use Month/Year on the X-axis and Total Revenue on the Y-axis.
            * **Filters:** Add slicers for Year and potentially Product Category to explore trends within segments.
            * **Next Step:** You might want to investigate which product categories are driving the revenue peaks."

            **Your primary goal is to act as a helpful, insightful assistant, leveraging your tools to provide not just data, but context and actionable recommendations for the data analyst.**
            """
            config = {
                "response_modalities": ["TEXT"],
                "system_instruction": {"parts": [{"text": system_instruction}]},
                "tools": [ {"code_execution": {}}, {"function_declarations": db_function_declarations} ],
            }
            print(f"Using Live API Config: {config}")

            # --- Define ASYNC consumer function ---
            async def consume_and_process_stream():
                async for chunk in process_live_session(
                    client=live_client, 
                    model_name=model_name, 
                    config=config,
                    initial_message=prompt, 
                    tool_implementations=available_functions
                ):
                    all_outputs_for_history.append(chunk)
                    chunk_type = chunk.get("type")
                    content = chunk.get("content")
                    sql = chunk.get("sql")
                    if chunk_type == "status":
                        status_container.update(label=content) # Status updates go here
                    elif chunk_type == "text":
                        final_text_response_content = content
                    elif chunk_type == "tool_result" or chunk_type == "data":
                         with status_container:
                              display_results(st.container(), content, sql)
                    elif chunk_type == "code":
                         with status_container:
                              st.markdown("**Generated Code:**")
                              st.code(content, language='python')
                    elif chunk_type == "code_result":
                         with status_container:
                              display_code_execution_results(st.container(), content)
                    elif chunk_type == "tool_error" or chunk_type == "error":
                         with status_container:
                              st.error(str(content))

            # --- Run the ASYNC consumer ---
            status_container.update(label="Connecting and processing stream...") # Initial status
            try:
                 asyncio.run(consume_and_process_stream()) # Run the consumer coroutine
                 if final_text_response_content:
                     final_response_container.markdown(final_text_response_content)
                 else:
                      final_response_container.info("Processing finished, no final text summary provided by AI.")
                 status_container.update(label="Processing complete.", state="complete", expanded=False) # Collapse status

            except RuntimeError as re:
                if "cannot run event loop while another loop is running" in str(re):
                    st.warning("Asyncio loop conflict. Trying nest_asyncio.")
                    import nest_asyncio
                    nest_asyncio.apply()
                    asyncio.run(consume_and_process_stream()) # Try again
                    if final_text_response_content: final_response_container.markdown(final_text_response_content)
                    else: final_response_container.info("Processing finished (nested loop).")
                    status_container.update(label="Processing complete (nested).", state="complete", expanded=False)
                else: raise
            except Exception as async_e: raise async_e


            # --- Update History ---
            st.session_state.assistant_turn_details[current_turn_index] = all_outputs_for_history
            # Store only the final text summary in the basic history
            st.session_state.messages.append({"role": "assistant", "content": final_text_response_content or "[Assistant processed request with tool/code usage]"})

        except Exception as e:
            # General error handling for the entire block
            error_message = f" An error occurred: {e}"
            print(error_message)
            status_container.update(label="Error", state="error", expanded=True)
            st.error(f"Sorry, I encountered an error: {e}")
            # Update history with error state
            current_turn_index = len(st.session_state.messages)
            st.session_state.assistant_turn_details[current_turn_index] = [{"type": "error", "content": str(e)}]
            st.session_state.messages.append({"role": "assistant", "content": "[Error occurred processing request]"})

    st.rerun()
   