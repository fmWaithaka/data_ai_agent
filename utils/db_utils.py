"""
Database utilities with enhanced security and integration with app configuration
"""

import logging
from contextlib import contextmanager
from typing import Iterator, List, Dict, Any, Optional
import mysql.connector
from mysql.connector import Error, pooling

from config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Get application settings
settings = get_settings()

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.pool = self._create_connection_pool()
        
    def _create_connection_pool(self) -> pooling.MySQLConnectionPool:
        """Create connection pool using application settings"""
        return pooling.MySQLConnectionPool(
            pool_name="retail_analytics_pool",
            pool_size=5,
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            database=settings.db_name,
            auth_plugin='mysql_native_password' if settings.db_type == 'mysql' else None
        )

    @contextmanager
    def get_connection(self) -> Iterator[mysql.connector.connection.MySQLConnection]:
        """Context manager for safe connection handling"""
        conn = self.pool.get_connection()
        try:
            yield conn
        except Error as e:
            logger.error("Database connection error: %s", e)
            raise
        finally:
            conn.close()

    def show_tables(self) -> List[str]:
        """Retrieve all tables in the database"""
        try:
            with self.get_connection() as conn, conn.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                return [table[0] for table in cursor.fetchall()]
        except Error as e:
            logger.error("Failed to retrieve tables: %s", e)
            return []

    def get_table_columns(self, table_name: str) -> List[Dict[str, str]]:
        """Get full column metadata for a table"""
        query = """
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY, EXTRA
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
        """
        try:
            with self.get_connection() as conn, conn.cursor(dictionary=True) as cursor:
                cursor.execute(query, (settings.db_name, table_name))
                return cursor.fetchall()
        except Error as e:
            logger.error("Failed to get columns for %s: %s", table_name, e)
            return []

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute a read-only query with security validation"""
        if not self._validate_query(sql):
            logger.warning("Blocked potentially unsafe query: %s", sql)
            return [{"error": "Query contains unauthorized operations"}]

        try:
            with self.get_connection() as conn, conn.cursor(dictionary=True) as cursor:
                cursor.execute(sql)
                return cursor.fetchall()[:settings.max_query_rows]
        except Error as e:
            logger.error("Query execution failed: %s", e)
            return [{"error": f"Database error: {e}"}]
        except Exception as e:
            logger.exception("Unexpected error executing query")
            return [{"error": f"Unexpected error: {e}"}]
        
    def _validate_query(self, query: str) -> bool:
        """Validate SQL query against security policies"""
        clean_query = query.strip().upper()
        allowed_keywords = {"SELECT", "SHOW", "DESCRIBE", "EXPLAIN"}
        forbidden_keywords = {
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
            "CREATE", "TRUNCATE", "GRANT", "REVOKE"
        }

        if any(kw in clean_query for kw in forbidden_keywords):
            return False
        return any(kw in clean_query for kw in allowed_keywords)

    def get_foreign_keys(self) -> Dict[str, List[Dict[str, str]]]:
        """Retrieve foreign key relationships for data modeling"""
        query = """
            SELECT 
                TABLE_NAME, COLUMN_NAME, 
                REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s 
            AND REFERENCED_TABLE_NAME IS NOT NULL
        """
        try:
            with self.get_connection() as conn, conn.cursor(dictionary=True) as cursor:
                cursor.execute(query, (settings.db_name,))
                relationships = {}
                for row in cursor:
                    table = row["TABLE_NAME"]
                    relationships.setdefault(table, []).append({
                        "source_column": row["COLUMN_NAME"],
                        "target_table": row["REFERENCED_TABLE_NAME"],
                        "target_column": row["REFERENCED_COLUMN_NAME"]
                    })
                return relationships
        except Error as e:
            logger.error("Failed to retrieve foreign keys: %s", e)
            return {}

# Initialize database manager instance
db_manager = DatabaseManager()