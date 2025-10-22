#!/usr/bin/env python3
"""
Database Schema Setup
===================

Create the initial database schema using Alembic migrations.
"""

import sys
import asyncio
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment to avoid requiring all config
os.environ.setdefault("ANANT_ENV", "development")

# Mock the logger and core modules to avoid full initialization
class SimpleLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

sys.modules['anant_integration.core'] = type('MockModule', (), {'logger': SimpleLogger()})()

# Now import our database modules
from anant_integration.database.config import DatabaseConfig
from anant_integration.database.models import Base
from anant_integration.database.migrations import MigrationManager

# Import SQLAlchemy for manual table creation
from sqlalchemy import create_engine
import alembic
from alembic import command
from alembic.config import Config


def create_alembic_migration():
    """Create the initial Alembic migration"""
    try:
        print("🔧 Setting up Alembic migrations...")
        
        # Initialize configuration
        config = DatabaseConfig()
        
        # Create migration manager
        migration_manager = MigrationManager(config)
        
        # Initialize migrations directory structure
        migration_manager.initialize_migrations()
        print("✅ Migration structure initialized")
        
        # Check if we already have migrations
        revisions = migration_manager.get_revision_history()
        if revisions:
            print(f"📋 Found {len(revisions)} existing migrations")
            return True
        
        # Create initial migration
        print("📝 Creating initial migration...")
        revision = migration_manager.create_initial_migration()
        print(f"✅ Created initial migration: {revision}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create migration: {e}")
        return False


def apply_migrations():
    """Apply migrations to database"""
    try:
        print("\n🚀 Applying migrations to database...")
        
        config = DatabaseConfig()
        migration_manager = MigrationManager(config)
        
        # Get current revision
        current_rev = migration_manager.get_current_revision()
        print(f"📊 Current database revision: {current_rev or 'None'}")
        
        # Apply migrations
        migration_manager.upgrade_database("head")
        
        # Check new revision
        new_rev = migration_manager.get_current_revision()
        print(f"✅ Database upgraded to revision: {new_rev}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to apply migrations: {e}")
        return False


def verify_schema():
    """Verify the database schema was created"""
    try:
        print("\n🔍 Verifying database schema...")
        
        config = DatabaseConfig()
        
        # Create sync engine for inspection
        engine = create_engine(config.postgres_url.replace("+asyncpg", ""))
        
        # Check if tables exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        expected_tables = [
            'users', 'concepts', 'graphs', 'mappings', 'security_policies',
            'concept_relationships', 'graph_metadata', 'mapping_metadata',
            'security_audit_logs', 'ontology_mappings', 'system_configurations',
            'cache_entries', 'alembic_version'
        ]
        
        missing_tables = []
        for table in expected_tables:
            if table in tables:
                print(f"  ✅ Table '{table}' exists")
            else:
                missing_tables.append(table)
                print(f"  ❌ Table '{table}' missing")
        
        engine.dispose()
        
        if not missing_tables:
            print("✅ All expected tables exist!")
            return True
        else:
            print(f"⚠️ Missing {len(missing_tables)} tables: {missing_tables}")
            return False
            
    except Exception as e:
        print(f"❌ Schema verification failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 Anant Integration - Database Schema Setup")
    print("=" * 50)
    
    success = True
    
    # Step 1: Create migrations
    if not create_alembic_migration():
        success = False
    
    # Step 2: Apply migrations
    if success and not apply_migrations():
        success = False
    
    # Step 3: Verify schema
    if success and not verify_schema():
        success = False
    
    if success:
        print("\n🎉 Database schema setup completed successfully!")
        print("\n📋 Next steps:")
        print("  1. The database is ready for use")
        print("  2. You can now run the anant_integration services")
        print("  3. Use repositories to interact with the database")
        return True
    else:
        print("\n❌ Database schema setup failed!")
        return False


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n💥 Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)