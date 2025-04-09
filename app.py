# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import google.generativeai.types as types
# Ensure utils.py or db_utils.py exists with your functions
from utils import show_tables, get_table_columns, execute_query
import time # Can be useful for debugging or adding small delays

# --- Page Configuration ---
st.set_page_config(
    page_title="Retail DB Analyst Assistant",
    page_icon="📊",
    layout="wide"
)

# --- Load Environment Variables & Secrets ---
load_dotenv()
# Use Streamlit secrets first, then fallback to environment variables
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Check API Key ---
if not GOOGLE_API_KEY:
    st.error("🔴 Error: GOOGLE_API_KEY not found. Please set it in secrets or .env.")
    st.stop()

# --- Configure GenAI Client ---
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("GenAI Configured successfully.")
except Exception as e:
    st.error(f"🔴 Error configuring Google AI SDK: {e}")
    st.stop()

# --- Tool Definition & Mapping ---
# Your imported Python functions that the AI can call
available_functions = {
    "show_tables": show_tables,
    "get_table_columns": get_table_columns,
    "execute_query": execute_query,
}
# Define tools for the model using the functions
db_tools = [show_tables, get_table_columns, execute_query]

# --- Gemini Model Initialization (Cached) ---
@st.cache_resource # Cache the model and chat session resource for efficiency
def init_gemini_chat():
    print("Attempting to initialize Gemini Model and Chat...")
    try:
        # --- Enhanced System Instruction ---
        system_instruction = """You are an expert Data Analyst Assistant working with a retail MySQL database (version 8.0.35).
        Your goal is to help the user explore, clean, transform, and understand the data to facilitate building Power BI dashboards.

        **Core Capabilities (using provided tools):**
        - Use `show_tables` to list available tables.
        - Use `get_table_columns` to understand table schemas (column names, data types).
        - Use `execute_query` to run **SELECT** SQL queries. Construct queries involving:
            - Filtering (WHERE), Joins (INNER, LEFT), Aggregations (GROUP BY, SUM, AVG, COUNT, MIN, MAX),
            - Data Cleaning/Transformation (DISTINCT, CASE), Window Functions (RANK, ROW_NUMBER), Date/Time functions, CTEs.
        - **Strictly Prohibited:** You CANNOT execute SQL commands like UPDATE, DELETE, INSERT, DROP, etc. If asked, state you cannot perform modifications.

        **Workflow:**
        1.  **Understand & Plan:** Analyze the user's request. Ask clarifying questions ONLY if the request is too ambiguous. Determine the SQL query or sequence needed.
        2.  **Execute:** Use the appropriate tool (`show_tables`, `get_table_columns`, or `execute_query`).
        3.  **Analyze Results:** Briefly interpret the results from the tool.
        4.  **Respond:** Present the results clearly. Use Markdown for text. If `execute_query` returned data, summarize the findings based on that data.
        5.  **Recommend (Power BI Context):** Based on the query results or analysis, proactively suggest:
            - Relevant KPIs to track.
            - Interesting follow-up questions the analyst might ask.
            - Potential data quality issues observed (e.g., based on NULL counts or unexpected values if queried).
            - Suitable visualization types for Power BI (e.g., "This trend data could be shown effectively with a Line Chart in Power BI.").

        **Focus on providing actionable data insights and relevant dashboarding recommendations.**
        """

        model = genai.GenerativeModel(
            "gemini-1.5-flash", # Good balance of capability and speed
            system_instruction=system_instruction,
            tools=db_tools) # Provide the list of function objects

        # Start chat WITHOUT automatic function calling - we will handle the loop
        chat = model.start_chat()
        print("Gemini Model and Chat initialized successfully (Manual Function Calling).")
        return chat
    except Exception as e:
        st.error(f"🔴 Error initializing Gemini Model: {e}")
        print(f"Error details during init: {e}")
        return None

chat_session = init_gemini_chat()

# Stop execution if chat session failed to initialize
if not chat_session:
    st.error("🔴 Failed to initialize chat session. Please check API key and configuration.")
    st.stop()

# --- Streamlit UI ---
st.title("📊 Retail Database Analyst Assistant")
st.write("Ask questions in natural language to explore the `retail_db` database and get Power BI recommendations.")
st.markdown("---")

# --- Sidebar (Keep as is from Phase 2) ---
with st.sidebar:
    st.header("Database Schema")
    st.markdown("Explore the tables and columns available.")

    @st.cache_data(ttl=3600)
    def cached_show_tables():
        return show_tables()

    @st.cache_data(ttl=3600)
    def cached_get_table_columns(table_name):
        return get_table_columns(table_name)

    tables = []
    try:
        with st.spinner("Loading tables..."):
            tables = cached_show_tables()
    except Exception as e:
        st.error(f"Error loading tables: {e}")

    if tables:
        selected_table = st.selectbox(
            "Select a table to view its schema:",
            options=[""] + tables, index=0,
        )
        if selected_table:
            st.subheader(f"Schema for `{selected_table}`")
            try:
                with st.spinner(f"Fetching columns for {selected_table}..."):
                    columns = cached_get_table_columns(selected_table)
                if columns:
                    st.dataframe(columns, use_container_width=True)
                elif isinstance(columns, list):
                     st.info(f"Table `{selected_table}` schema information might be unavailable or empty.")
                else:
                    st.warning(f"Could not retrieve columns for `{selected_table}`.")
            except Exception as e:
                st.error(f"Error fetching columns for {selected_table}: {e}")
    elif isinstance(tables, list):
         st.warning("No tables found in the database.")
    else:
        st.error("Failed to retrieve table list from the database.")


# --- Main Chat Area ---
st.header("Chat with your Data")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper function to display results intelligently ---
def display_results(container, result_data, execute_query_sql=None):
    """Displays tool results (dataframes, metrics, lists, errors)"""
    if isinstance(result_data, list) and result_data:
        if isinstance(result_data[0], dict):
            # Check for specific error structure from db_utils
            if "error" in result_data[0]:
                 container.error(f"Database/Tool Error: {result_data[0]['error']}")
                 return

            # Display as DataFrame
            try:
                 container.dataframe(result_data)
            except Exception as df_e:
                 container.warning(f"Could not display data as DataFrame ({df_e}), showing raw list:")
                 container.write(result_data) # Fallback

            # Basic heuristic for single value metrics (enhance as needed)
            if len(result_data) == 1 and len(result_data[0]) == 1:
                 key = list(result_data[0].keys())[0]
                 value = result_data[0][key]
                 if isinstance(value, (int, float)):
                      formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                      # Use columns for side-by-side layout
                      col1, col2 = container.columns([1, 3])
                      col1.metric(label=key.replace('_', ' ').title(), value=formatted_value)
                      # col2 can be used for comparison or context later
                      return # Stop if displayed as metric

        else: # Handle list of non-dicts (e.g., from show_tables)
            container.write(result_data)
    elif isinstance(result_data, str):
         if "error" in result_data.lower(): # Check for string errors
              container.error(result_data)
         else:
              container.markdown(result_data)
    elif result_data is None or (isinstance(result_data, list) and not result_data):
         container.info("Tool execution returned no data.")
    else: # Fallback for other types
        container.write(result_data)


# --- Display Chat History ---
# Store results temporarily to display them within the corresponding assistant turn
assistant_outputs = {} # turn_index -> list of outputs (text, data)

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        content = message["content"]
        if message["role"] == "assistant" and idx in assistant_outputs:
             # Display multiple parts (data + text) for this assistant turn
             for output in assistant_outputs[idx]:
                  if output["type"] == "data":
                       display_results(st.container(), output["data"], output.get("sql"))
                  elif output["type"] == "text":
                       st.markdown(output["data"])
                  else: # Fallback
                       st.write(output["data"])
        elif isinstance(content, str): # Default display for user messages or simple text
            st.markdown(content)
        # If content needs special handling but isn't pre-processed, add logic here
        # else:
        #    st.write(content)


# --- Handle New User Input ---
if prompt := st.chat_input("Ask about the data (e.g., What are the top 5 products by revenue?)"):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Manual Function Calling Loop ---
    processing_complete = False
    current_prompt_content = prompt
    turn_index = len(st.session_state.messages) # Index for the upcoming assistant message
    assistant_outputs[turn_index] = [] # Initialize output list for this turn

    with st.chat_message("assistant"):
        # Use a status indicator for ongoing processing steps
        with st.status("Processing request...", expanded=True) as status:
            try:
                response = None
                # Loop until the AI gives a final text response for this turn
                while not processing_complete:
                    status.update(label="Sending request to AI...")
                    response = chat_session.send_message(current_prompt_content)

                    if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
                        status.error("Received an empty or invalid response from the AI.")
                        assistant_outputs[turn_index].append({"type": "text", "data": "Sorry, I received an invalid response. Please try again."})
                        processing_complete = True
                        break

                    # --- Process Response Parts ---
                    next_prompt_parts = [] # Parts to send back in the next loop iteration
                    found_final_text_in_part = False

                    for part in response.candidates[0].content.parts:
                        if func_call := part.function_call:
                            func_name = func_call.name
                            args = dict(func_call.args)
                            status.update(label=f"AI requested: `{func_name}`")

                            if func_name in available_functions:
                                tool_function = available_functions[func_name]
                                try:
                                    sql_executed = args.get('sql', None) # Keep track of SQL if it's execute_query
                                    status.write(f"Executing: `{func_name}({args if args else ''})`")
                                    if sql_executed:
                                         status.write(f"SQL: ```sql\n{sql_executed}\n```")

                                    result_data = tool_function(**args) # Call the actual function

                                    status.write("Tool executed. Preparing results...")
                                    # Add result data to output list for immediate display *after* loop
                                    assistant_outputs[turn_index].append({"type": "data", "data": result_data, "sql": sql_executed})
                                    # Prepare response part for next AI turn
                                    next_prompt_parts.append(part) # Original call
                                    next_prompt_parts.append(types.Part(function_response=types.FunctionResponse(name=func_name, response={"result": result_data})))

                                except Exception as tool_e:
                                    status.error(f"Error calling tool '{func_name}': {tool_e}")
                                    error_response = {"error": str(tool_e)}
                                    assistant_outputs[turn_index].append({"type": "text", "data": f"Error executing `{func_name}`: {tool_e}"})
                                    next_prompt_parts.append(part)
                                    next_prompt_parts.append(types.Part(function_response=types.FunctionResponse(name=func_name, response=error_response)))
                            else:
                                status.error(f"AI requested unknown function '{func_name}'")
                                error_response = {"error": f"Function '{func_name}' not available."}
                                assistant_outputs[turn_index].append({"type": "text", "data": f"Error: Unknown function `{func_name}` requested."})
                                next_prompt_parts.append(part)
                                next_prompt_parts.append(types.Part(function_response=types.FunctionResponse(name=func_name, response=error_response)))

                        elif response_text := part.text:
                            # Append final text to the output list for this turn
                            assistant_outputs[turn_index].append({"type": "text", "data": response_text})
                            found_final_text_in_part = True # Mark that we have the text

                        else:
                             status.write(f"Received unprocessed part type: {type(part)}")
                             print("Unprocessed part:", part) # Log unknown parts

                    # --- End of Processing Parts for this AI Response ---

                    if found_final_text_in_part:
                        # If we got text, assume the AI is done for this turn
                        status.update(label="Processing complete.", state="complete", expanded=False)
                        processing_complete = True
                    elif next_prompt_parts:
                         # If we only got function calls/responses, loop again sending results back
                         current_prompt_content = next_prompt_parts
                         status.update(label="Sending tool results back to AI...")
                    else:
                         # No text, no function call responses to send back? Unusual state.
                         status.warning("AI turn finished without text or further actions needed. Ending turn.")
                         processing_complete = True # Exit loop

            except Exception as e:
                error_message = f"🔴 An error occurred during chat processing: {e}"
                print(error_message)
                status.update(label="Error", state="error", expanded=True)
                assistant_outputs[turn_index].append({"type": "text", "data": "Sorry, I encountered an error processing your request."})
                # Ensure processing_complete is set to exit loop on error
                processing_complete = True
            finally:
                 # Update session state AFTER the loop finishes for this turn
                 # Consolidate outputs for storage in session state (optional, could just keep text)
                 final_assistant_content_for_history = "\n\n".join([str(out["data"]) for out in assistant_outputs[turn_index]]) # Basic string concatenation for now
                 st.session_state.messages.append({"role": "assistant", "content": final_assistant_content_for_history})

    # Force Streamlit to rerun from top to display the new messages correctly
    # Use with caution, ensure state is managed properly to avoid infinite loops
    st.rerun()