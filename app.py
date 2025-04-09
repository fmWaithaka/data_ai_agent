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
    st.error("🔴 Error: GOOGLE_API_KEY not found.")
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
         st.error(f"🔴 SDK Error: 'genai' module has no attribute 'Client'. Found version {genai.__version__}. Please ensure the correct library version is installed and the environment is clean. Details: {ae}")
         print(f"AttributeError during Live Client init: {ae}")
         return None
    except Exception as e:
        st.error(f"🔴 Error initializing Live Client: {e}")
        print(f"Error details during Live Client init: {e}")
        return None

live_client = init_live_client()

if not live_client:
    st.stop()

def display_results(container, result_data, execute_query_sql=None):
    if isinstance(result_data, list) and result_data:
        if isinstance(result_data[0], dict):
            if "error" in result_data[0]:
                 container.error(f"Database/Tool Error: {result_data[0]['error']}")
                 return
            try:
                 if execute_query_sql:
                      with container.expander("View SQL Query"):
                           st.code(execute_query_sql, language="sql")
                 container.dataframe(result_data)
            except Exception as df_e:
                 container.warning(f"Could not display data as DataFrame ({df_e}), showing raw list:")
                 container.write(result_data)
            if len(result_data) == 1 and len(result_data[0]) == 1:
                 key = list(result_data[0].keys())[0]
                 value = result_data[0][key]
                 if isinstance(value, (int, float)):
                      formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                      col1, col2 = container.columns([1, 3])
                      col1.metric(label=key.replace('_', ' ').title(), value=formatted_value)
                      return
        else:
            container.write(result_data)
    elif isinstance(result_data, str):
         if "error" in result_data.lower(): container.error(result_data)
         else: container.markdown(result_data)
    elif result_data is None or (isinstance(result_data, list) and not result_data):
         container.info("Tool execution returned no data.")
    else: container.write(result_data)

def display_code_execution_results(container, result_part_dict):
    """Displays code execution results dict, attempting to render plots."""
    outcome_str = str(result_part_dict.get('outcome', 'UNKNOWN')).lower()
    output_str = str(result_part_dict.get('output', ''))
    if outcome_str == 'ok':
        container.success("✅ Code executed successfully.")
        if output_str:
            captured_plot = False
            try:
                fig = plt.gcf()
                if fig and fig.axes and any(ax.has_data() for ax in fig.axes):
                    container.pyplot(fig)
                    plt.clf(); plt.close(fig) # Clear and close figure
                    print("Displayed plot using st.pyplot(gcf).")
                    captured_plot = True
            except Exception as plt_e: print(f"Plot capture with gcf failed: {plt_e}")
            if not captured_plot:
                 container.markdown("**Code Output:**")
                 container.code(output_str, language='text')
    else:
        container.error(f"⚠️ Code execution failed: {outcome_str}")
        if output_str: container.code(output_str, language='text')


# --- Async Handler for Live API Interaction ---
async def process_live_session(client, model_name, config, initial_message, tool_implementations):
    """Connects to Live API, sends message, handles stream, returns collected outputs."""
    collected_outputs = [] 
    final_text_summary = None # Store the textual summary if received

    print("Starting Live API session...")
    try:
        async with client.aio.live.connect(model=model_name, config=config) as session:
            print("Live API Session connected. Sending initial message...")
            await session.send(input=initial_message, end_of_turn=True)

            async for msg in session.receive():
                print(f"Received msg part type: {type(msg)}") # Debug msg types
                # --- Parse Message Parts ---
                if text := msg.text:
                    print(f"Received text chunk: {text[:100]}...")
                    collected_outputs.append({"type": "text", "content": text})
                    final_text_summary = text # Overwrite with the last text received
                elif tool_call := msg.tool_call:
                    print(f"Received tool call request: {tool_call.function_calls}")
                    tool_responses_to_send = []
                    for fc in tool_call.function_calls:
                        tool_name = fc.name
                        tool_args = dict(fc.args)
                        sql = tool_args.get('sql') # Get SQL if present
                        collected_outputs.append({"type": "tool_call", "content": {"name": tool_name, "args": tool_args}})

                        if tool_name in tool_implementations:
                            tool_impl = tool_implementations[tool_name]
                            try:
                                print(f"Executing tool: {tool_name}")
                                # IMPORTANT: Run synchronous DB calls in executor to avoid blocking asyncio loop
                                loop = asyncio.get_running_loop()
                                result = await loop.run_in_executor(None, lambda: tool_impl(**tool_args))
                                print(f"Tool {tool_name} executed successfully.")
                                tool_response_part = types.FunctionResponse(name=fc.name, id=fc.id, response={'result': result})
                                collected_outputs.append({"type": "tool_result", "content": result, "sql": sql})
                            except Exception as e:
                                print(f"Error executing tool {tool_name}: {e}")
                                result = {"error": str(e)}
                                tool_response_part = types.FunctionResponse(name=fc.name, id=fc.id, response=result)
                                collected_outputs.append({"type": "tool_error", "content": str(e)})
                        else: 
                             print(f"Error: Unknown tool requested: {tool_name}")
                             result = {"error": f"Tool '{tool_name}' not found."}
                             tool_response_part = types.FunctionResponse(name=fc.name, id=fc.id, response=result)
                             collected_outputs.append({"type": "tool_error", "content": f"Unknown tool: {tool_name}"})
                        tool_responses_to_send.append(tool_response_part)

                    # Send tool responses back
                    if tool_responses_to_send:
                         print("Sending tool responses back to Live API...")
                         tool_response_msg = types.LiveClientToolResponse(function_responses=tool_responses_to_send)
                         await session.send(input=tool_response_msg)
                         print("Tool responses sent.")

                elif msg.server_content and msg.server_content.model_turn:
                    # Handle code execution parts
                    for part in msg.server_content.model_turn.parts:
                        if code := part.executable_code:
                            print("Received executable code from AI.")
                            collected_outputs.append({"type": "code", "content": code.code})
                        elif result := part.code_execution_result:
                            print(f"Received code execution result: Outcome={result.outcome}")
                            result_info = {"outcome": str(result.outcome), "output": str(result.output)}
                            collected_outputs.append({"type": "code_result", "content": result_info})
                        else:
                             print(f"Received other server_content part: {type(part)}")
                else:
                     print(f"Received unhandled message type: {type(msg)}")


    except Exception as e:
        print(f"Error during Live API session processing: {e}")
        collected_outputs.append({"type": "error", "content": f"An error occurred during the session: {e}"})

    print("Live API session processing finished.")
    # Return all collected parts for display and the final text summary
    return collected_outputs, final_text_summary


# --- Streamlit UI ---
st.title("🚀 Retail DB Analyst Agent (Live API) 🚀")
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
if prompt := st.chat_input("Ask about data, request analysis or plots..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Process using Live API ---
    with st.chat_message("assistant"):
        status_placeholder = st.status("Processing via Live API...", expanded=True)
        # Create container for results within the chat message context
        results_container = st.container()

        try:
            # --- Prepare Live API Config ---
            model_name = 'gemini-2.0-flash-exp' # Requires capable model

            # Create FunctionDeclarations for DB tools, passing the initialized client
            try:
                show_tables_tool_decl = types.FunctionDeclaration.from_callable(client=live_client, callable=show_tables)
                get_cols_tool_decl = types.FunctionDeclaration.from_callable(client=live_client, callable=get_table_columns)
                execute_query_tool_decl = types.FunctionDeclaration.from_callable(client=live_client, callable=execute_query)

                db_function_declarations = [
                    show_tables_tool_decl.to_json_dict(), 
                    get_cols_tool_decl.to_json_dict(),
                    execute_query_tool_decl.to_json_dict()
                ]

                print("Created FunctionDeclarations for Live API.")
            except Exception as decl_e:
                 st.error(f"Failed to create FunctionDeclarations for Live API: {decl_e}")
                 raise decl_e # Stop if declarations fail

            # System prompt (reuse the detailed one)
            system_instruction = """You are an expert Data Analyst Assistant working with a retail MySQL database (version 8.0.35).
                    Your goal is to help the user explore, clean, transform, analyze data, and get recommendations for Power BI dashboards.

                    **Core Capabilities:**
                    1.  **Database Interaction (Tools: `show_tables`, `get_table_columns`, `execute_query`):**
                        - Use these tools via function calls to interact with the database using SQL (SELECT queries only for `execute_query`). Construct complex SQL as needed.
                        - **Prohibited SQL:** Do NOT execute UPDATE, DELETE, INSERT, etc.
                    2.  **Code Execution (Built-in Tool):**
                        - You have a built-in **Python Code Execution tool**. Use it for: Visualizations (Matplotlib, Seaborn - ensure plots render), Advanced Analysis (Pandas, Scipy, Numpy), Complex Transformations.
                        - **Process:** Typically, fetch data using `execute_query`, then write and execute Python code. Import libraries within the generated code.

                    **Workflow:**
                    1.  **Understand & Plan:** Analyze request. Decide SQL vs Python code execution.
                    2.  **Execute Tools:** Call DB functions or generate+execute Python code.
                    3.  **Analyze Results:** Interpret SQL or code output. Check errors.
                    4.  **Respond & Visualize:** Present results clearly (Markdown, summaries). **If you generate a plot, state that you are showing the plot.**
                    5.  **Recommend (Power BI):** Provide actionable insights, follow-up questions, quality checks, or relevant Power BI visualization types based on the results.

                    **Focus on clear execution, actionable results, and relevant recommendations.**
                    """
            
            config = {
                "response_modalities": ["TEXT"], # Match working example
                "system_instruction": {"parts": [{"text": system_instruction}]},
                "tools": [
                    {"code_execution": {}}, # Enable code execution
                    {"function_declarations": db_function_declarations},
                ],
            }

            # --- Run the Async Task using asyncio.run ---
            status_placeholder.update(label="Connecting and processing...")
            # This executes the async function and waits for it to complete
            collected_outputs, final_text_summary = asyncio.run(
                process_live_session(
                    client=live_client,
                    model_name=model_name,
                    config=config,
                    initial_message=prompt,
                    tool_implementations=available_functions # Pass Python functions here
                )
            )
            status_placeholder.update(label="Processing complete.", state="complete", expanded=False)

            # --- Display Collected Outputs in the container ---
            assistant_turn_idx = len(st.session_state.messages) # Index for this new turn
            st.session_state.assistant_turn_details[assistant_turn_idx] = collected_outputs

            for part_info in collected_outputs:
                 content_type = part_info.get("type", "unknown")
                 content = part_info.get("content", "")
                 sql = part_info.get("sql")
                 if content_type == "text": results_container.markdown(content)
                 elif content_type == "data" or content_type == "tool_result": display_results(results_container, content, sql)
                 elif content_type == "code": results_container.markdown("**Generated Code:**"); results_container.code(content, language='python')
                 elif content_type == "code_result": display_code_execution_results(results_container, content)
                 elif content_type == "tool_error" or content_type == "error": results_container.error(str(content))
                 # Optionally hide tool_call details

            # Add a consolidated message to basic history
            final_summary_for_history = final_text_summary if final_text_summary else "[Assistant processed request with tool/code usage]"
            st.session_state.messages.append({"role": "assistant", "content": final_summary_for_history})


        except Exception as e:
            error_message = f"🔴 An error occurred: {e}"
            print(error_message)
            status_placeholder.update(label="Error", state="error", expanded=True)
            st.error(f"Sorry, I encountered an error: {e}") # Show error in main area too
            # Add error message to history
            st.session_state.messages.append({"role": "assistant", "content": "[Error occurred processing request]"})

]    st.rerun()