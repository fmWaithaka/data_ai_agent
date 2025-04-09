# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import google.generativeai.types as types # Keep this for potential future config use
from utils import show_tables, get_table_columns # Import necessary functions

# --- Page Configuration (Optional but Recommended) ---
st.set_page_config(
    page_title="Retail DB Analyst Assistant",
    page_icon="📊",
    layout="wide"
)

# --- Load Environment Variables ---
load_dotenv()
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# --- Check API Key ---
if not GOOGLE_API_KEY:
    st.error("🔴 Error: GOOGLE_API_KEY not found. Please set it in .env or Streamlit secrets.")
    st.stop() # Halt execution if no key

# --- Configure GenAI Client ---
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Optional: Add client initialization if needed later
    # client = genai.Client(api_key=GOOGLE_API_KEY)
    print("GenAI Configured successfully.")
except Exception as e:
    st.error(f"🔴 Error configuring Google AI SDK: {e}")
    st.stop()

# --- Caching Functions for DB Schema ---
# Cache data fetching to avoid hitting the DB repeatedly
@st.cache_data(ttl=3600) # Cache for 1 hour
def cached_show_tables():
    print("Cache miss: Running show_tables()")
    return show_tables()

@st.cache_data(ttl=3600) # Cache for 1 hour
def cached_get_table_columns(table_name):
    print(f"Cache miss: Running get_table_columns({table_name})")
    return get_table_columns(table_name)

# --- Streamlit UI ---
st.title("📊 Retail Database Analyst Assistant")
st.write("Interact with the `retail_db` using natural language.")
st.markdown("---") # Divider

# --- Sidebar for Schema Exploration ---
with st.sidebar:
    st.header("Database Schema")
    st.markdown("Explore the tables and columns available.")

    tables = []
    try:
        with st.spinner("Loading tables..."):
            tables = cached_show_tables()
    except Exception as e:
        st.error(f"Error loading tables: {e}")

    if tables:
        selected_table = st.selectbox(
            "Select a table to view its schema:",
            options=tables,
            index=None, # No default selection
            placeholder="Choose a table..."
        )

        if selected_table:
            st.subheader(f"Schema for `{selected_table}`")
            try:
                with st.spinner(f"Fetching columns for {selected_table}..."):
                    columns = cached_get_table_columns(selected_table)
                if columns:
                    # Displaying as a more readable dataframe
                    st.dataframe(columns, use_container_width=True)
                else:
                    st.warning(f"Could not retrieve columns for {selected_table}.")
            except Exception as e:
                st.error(f"Error fetching columns for {selected_table}: {e}")
    else:
        st.warning("No tables found or unable to connect to the database.")

# --- Main Chat Area (Placeholder for next phase) ---
st.header("Chat with your Data")
st.write("*(Chat interface will be implemented in the next phase)*")

# Initialize chat history (essential for chat apps)
if "messages" not in st.session_state:
    st.session_state.messages = []
    print("Initialized Session State: messages")

# Display existing messages (will be empty initially)
# for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.markdown(message["content"]) # Basic display for now

# Chat input (will be processed in next phase)
# if prompt := st.chat_input("Ask about the data..."):
#    st.write(f"You asked: {prompt}") # Placeholder action