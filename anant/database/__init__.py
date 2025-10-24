"""
Anant Database Module
====================

Database connectivity, schema management, and database server components.
"""

try:
    from .database_server import DatabaseServer
    from .create_schema import (
        DatabaseConfig,
        create_database_schema,
        insert_sample_data
    )
    from .setup_database_schema import setup_database
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

__all__ = []
if DATABASE_AVAILABLE:
    __all__.extend([
        "DatabaseServer",
        "DatabaseConfig",
        "create_database_schema",
        "insert_sample_data",
        "setup_database"
    ])