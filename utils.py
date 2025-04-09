# db_utils.py
import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv() # Load variables from .env

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_AUTH_PLUGIN = os.getenv("DB_AUTH_PLUGIN") # Will be None if not set

def create_db_connection():
    """Creates and returns a MySQL database connection."""
    connection = None
    try:
        conn_args = {
            'host': DB_HOST,
            'port': int(DB_PORT),
            'user': DB_USER,
            'password': DB_PASSWORD,
            'database': DB_NAME,
        }
        # Only add auth_plugin if it's set
        if DB_AUTH_PLUGIN:
            conn_args['auth_plugin'] = DB_AUTH_PLUGIN

        connection = mysql.connector.connect(**conn_args)
        # print("MySQL Database connection successful") # Keep this for debug if needed
    except Error as e:
        print(f"Error connecting to MySQL Database: '{e}'")
    return connection

def show_tables() -> list[str]:
    """Retrieve the names of all tables in the database."""
    tables = []
    conn = create_db_connection()
    if conn and conn.is_connected():
        try:
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES;")
            tables = [table[0] for table in cursor.fetchall()]
            cursor.close()
        except Error as e:
            print(f"Error showing tables: {e}")
        finally:
            conn.close()
    return tables

def get_table_columns(table_name: str) -> list[dict]:
    """Get column information for a specific table.

    Returns:
        List of dictionaries {'COLUMN_NAME': ..., 'DATA_TYPE': ..., 'IS_NULLABLE': ...}
    """
    columns_info = []
    conn = create_db_connection()
    if conn and conn.is_connected():
        try:
            cursor = conn.cursor(dictionary=True) # Fetch as dicts
            # Use prepared statement placeholder style for mysql.connector
            query = """
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = %s
                AND TABLE_NAME = %s
                ORDER BY ORDINAL_POSITION;
            """
            cursor.execute(query, (DB_NAME, table_name)) # Pass DB_NAME and table_name
            columns_info = cursor.fetchall()
            cursor.close()
        except Error as e:
            print(f"Error getting columns for table {table_name}: {e}")
        finally:
            conn.close()
    return columns_info


def execute_query(sql: str) -> list[dict]:
    """Execute an SQL query and return results as a list of dictionaries."""
    results = []
    conn = create_db_connection()
    # Basic validation: Deny potentially harmful commands if needed
    # This is a VERY basic check, consider more robust validation/sandboxing
    disallowed_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE']
    if any(keyword in sql.upper() for keyword in disallowed_keywords):
         print(f"Error: Query contains disallowed keyword: {sql}")
         # Optionally raise an error or return specific error message
         return [{"error": "Query contains disallowed keywords (e.g., DROP, DELETE, UPDATE). Only SELECT is allowed."}]

    if conn and conn.is_connected():
        try:
            # Use dictionary=True to get results directly as dicts
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
        except Error as e:
            print(f"Error executing query '{sql}': {e}")
            # Return error message in a structured way
            return [{"error": f"Database error: {e}"}]
        except Exception as e:
             print(f"General Error executing query '{sql}': {e}")
             return [{"error": f"An unexpected error occurred: {e}"}]
        finally:
            conn.close()
    else:
         return [{"error": "Failed to connect to the database."}]
    return results

# --- Test function ---
# if __name__ == '__main__':
#     print("Testing DB Utilities...")
#     # Test connection implicitly via functions
#     print("Tables:", show_tables())
#     if show_tables():
#          print("Columns for 'orders':", get_table_columns('orders'))
#          print("Sample Query (First 2 orders):", execute_query("SELECT * FROM orders LIMIT 2;"))
#          print("Sample Query (Non-SELECT):", execute_query("UPDATE orders SET order_status = 'TEST' WHERE order_id = 1;")) # Should fail/return error
#     else:
#          print("Could not connect to DB to run tests.")