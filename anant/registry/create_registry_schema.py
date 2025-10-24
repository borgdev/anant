#!/usr/bin/env python3
"""
Graph Registry Schema for Anant
===============================

Lightweight PostgreSQL schema optimized for graph registry/catalog functionality.
Aligns with Anant's native Parquet+Polars storage architecture.

PostgreSQL serves as:
- Graph registry/catalog (metadata only)
- User management
- Access control
- Audit trails
- System configuration

Actual graph data stored in Parquet files via Anant's native storage.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Database imports
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, 
    Boolean, DateTime, Text, UUID, JSON, ForeignKey, Index,
    ARRAY, DECIMAL, func, BigInteger, Float
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from typing import Optional


class DatabaseConfig(BaseSettings):
    """Database configuration for Graph Registry"""
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT") 
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="anant_registry", env="POSTGRES_DB")
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


def create_graph_registry_schema():
    """Create the Anant Graph Registry schema optimized for catalog functionality"""
    print("🏗️ Creating Anant Graph Registry schema...")
    
    config = DatabaseConfig()
    engine = create_engine(config.postgres_url)
    metadata = MetaData()
    
    # Users table - User management and authentication
    users = Table(
        'users',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('username', String(100), unique=True, nullable=False),
        Column('email', String(255), unique=True, nullable=False),
        Column('password_hash', String(255), nullable=False),
        Column('full_name', String(255)),
        Column('is_active', Boolean, default=True),
        Column('is_admin', Boolean, default=False),
        Column('preferences', JSONB, default={}),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Column('last_login', DateTime(timezone=True)),
        Index('idx_users_username', 'username'),
        Index('idx_users_email', 'email'),
        Index('idx_users_active', 'is_active')
    )
    
    # Graph Registry - Catalog of graph instances (metadata only, data in Parquet)
    graph_registry = Table(
        'graph_registry',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('name', String(255), nullable=False),
        Column('display_name', String(255)),
        Column('description', Text),
        Column('graph_type', String(100), nullable=False),  # hypergraph, metagraph, knowledge_graph, etc.
        Column('owner_id', PG_UUID(as_uuid=True), ForeignKey('users.id'), nullable=False),
        Column('status', String(50), default='active'),  # active, archived, deprecated
        Column('version', String(50), default='1.0.0'),
        Column('is_public', Boolean, default=False),
        
        # Storage locations (Anant's native Parquet storage)
        Column('storage_path', String(1000), nullable=False),  # Path to Parquet files
        Column('metadata_path', String(1000)),  # Path to metadata files
        Column('backup_paths', ARRAY(String)),  # Backup locations
        
        # Graph statistics (cached for UI display)
        Column('node_count', BigInteger, default=0),
        Column('edge_count', BigInteger, default=0),
        Column('hyperedge_count', BigInteger, default=0),
        Column('file_size_bytes', BigInteger, default=0),
        Column('last_computed_stats', DateTime(timezone=True)),
        
        # Registry metadata
        Column('tags', ARRAY(String), default=[]),
        Column('categories', ARRAY(String), default=[]),
        Column('properties', JSONB, default={}),
        Column('schema_info', JSONB, default={}),  # Schema.org or custom schema info
        
        # Lifecycle
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Column('last_accessed', DateTime(timezone=True)),
        Column('access_count', Integer, default=0),
        
        # Constraints
        Index('idx_graph_registry_name', 'name'),
        Index('idx_graph_registry_owner', 'owner_id'),
        Index('idx_graph_registry_type', 'graph_type'),
        Index('idx_graph_registry_status', 'status'),
        Index('idx_graph_registry_public', 'is_public'),
        Index('idx_graph_registry_tags', 'tags', postgresql_using='gin'),
        Index('idx_graph_registry_storage_path', 'storage_path')
    )
    
    # Access Control - Permissions for graph access
    graph_permissions = Table(
        'graph_permissions',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('graph_id', PG_UUID(as_uuid=True), ForeignKey('graph_registry.id', ondelete='CASCADE'), nullable=False),
        Column('user_id', PG_UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE')),
        Column('role', String(100)),  # For role-based access
        Column('permission_type', String(50), nullable=False),  # read, write, admin, execute
        Column('granted_by', PG_UUID(as_uuid=True), ForeignKey('users.id')),
        Column('granted_at', DateTime(timezone=True), server_default=func.now()),
        Column('expires_at', DateTime(timezone=True)),
        Column('is_active', Boolean, default=True),
        
        Index('idx_graph_permissions_graph', 'graph_id'),
        Index('idx_graph_permissions_user', 'user_id'),
        Index('idx_graph_permissions_type', 'permission_type'),
        Index('idx_graph_permissions_active', 'is_active')
    )
    
    # Query History - Track graph usage for analytics
    query_history = Table(
        'query_history',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('graph_id', PG_UUID(as_uuid=True), ForeignKey('graph_registry.id'), nullable=False),
        Column('user_id', PG_UUID(as_uuid=True), ForeignKey('users.id')),
        Column('operation_type', String(100), nullable=False),  # read, query, update, etc.
        Column('query_info', JSONB, default={}),  # Query details, filters, etc.
        Column('execution_time_ms', Float),
        Column('result_count', Integer),
        Column('success', Boolean, default=True),
        Column('error_message', Text),
        Column('timestamp', DateTime(timezone=True), server_default=func.now()),
        Column('ip_address', String(50)),
        Column('user_agent', String(500)),
        
        Index('idx_query_history_graph', 'graph_id'),
        Index('idx_query_history_user', 'user_id'),
        Index('idx_query_history_timestamp', 'timestamp'),
        Index('idx_query_history_operation', 'operation_type'),
        Index('idx_query_history_success', 'success')
    )
    
    # Graph Relationships - Links between graphs in the registry
    graph_relationships = Table(
        'graph_relationships',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('source_graph_id', PG_UUID(as_uuid=True), ForeignKey('graph_registry.id', ondelete='CASCADE'), nullable=False),
        Column('target_graph_id', PG_UUID(as_uuid=True), ForeignKey('graph_registry.id', ondelete='CASCADE'), nullable=False),
        Column('relationship_type', String(100), nullable=False),  # derived_from, merged_with, subset_of, etc.
        Column('properties', JSONB, default={}),
        Column('created_by', PG_UUID(as_uuid=True), ForeignKey('users.id')),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('is_active', Boolean, default=True),
        
        Index('idx_graph_relationships_source', 'source_graph_id'),
        Index('idx_graph_relationships_target', 'target_graph_id'),
        Index('idx_graph_relationships_type', 'relationship_type'),
        Index('idx_graph_relationships_active', 'is_active')
    )
    
    # System Configuration - Registry settings and metadata
    system_config = Table(
        'system_config',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('key', String(255), unique=True, nullable=False),
        Column('value', Text),
        Column('value_type', String(50), default='string'),  # string, json, boolean, integer
        Column('category', String(100), default='general'),
        Column('description', Text),
        Column('is_sensitive', Boolean, default=False),
        Column('updated_by', PG_UUID(as_uuid=True), ForeignKey('users.id')),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        
        Index('idx_system_config_key', 'key'),
        Index('idx_system_config_category', 'category')
    )
    
    # Storage Locations - Registry of storage backends and paths
    storage_locations = Table(
        'storage_locations',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('name', String(255), unique=True, nullable=False),
        Column('storage_type', String(100), nullable=False),  # local, s3, gcs, etc.
        Column('base_path', String(1000), nullable=False),
        Column('configuration', JSONB, default={}),  # Storage-specific config
        Column('is_active', Boolean, default=True),
        Column('is_default', Boolean, default=False),
        Column('capacity_bytes', BigInteger),
        Column('used_bytes', BigInteger, default=0),
        Column('last_checked', DateTime(timezone=True)),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        
        Index('idx_storage_locations_name', 'name'),
        Index('idx_storage_locations_type', 'storage_type'),
        Index('idx_storage_locations_active', 'is_active'),
        Index('idx_storage_locations_default', 'is_default')
    )
    
    # Audit Logs - Security and compliance tracking
    audit_logs = Table(
        'audit_logs',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('user_id', PG_UUID(as_uuid=True), ForeignKey('users.id')),
        Column('graph_id', PG_UUID(as_uuid=True), ForeignKey('graph_registry.id')),
        Column('action', String(255), nullable=False),
        Column('resource_type', String(100), nullable=False),
        Column('resource_id', String(255)),
        Column('success', Boolean, nullable=False),
        Column('details', JSONB, default={}),
        Column('ip_address', String(50)),
        Column('user_agent', String(500)),
        Column('session_id', String(255)),
        Column('timestamp', DateTime(timezone=True), server_default=func.now()),
        
        Index('idx_audit_logs_user', 'user_id'),
        Index('idx_audit_logs_graph', 'graph_id'),
        Index('idx_audit_logs_action', 'action'),
        Index('idx_audit_logs_resource_type', 'resource_type'),
        Index('idx_audit_logs_timestamp', 'timestamp'),
        Index('idx_audit_logs_success', 'success')
    )
    
    try:
        # Create all tables
        print("📋 Creating graph registry tables...")
        metadata.create_all(engine)
        
        # Verify tables were created
        from sqlalchemy import inspect
        inspector = inspect(engine)
        created_tables = inspector.get_table_names()
        
        expected_tables = [
            'users', 'graph_registry', 'graph_permissions', 'query_history',
            'graph_relationships', 'system_config', 'storage_locations', 'audit_logs'
        ]
        
        print(f"✅ Created {len(created_tables)} registry tables:")
        for table in sorted(created_tables):
            if table in expected_tables:
                print(f"  ✅ {table}")
            else:
                print(f"  ℹ️ {table} (existing)")
        
        missing = set(expected_tables) - set(created_tables)
        if missing:
            print(f"⚠️ Missing tables: {missing}")
            return False
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Failed to create registry schema: {e}")
        return False


def insert_initial_data():
    """Insert initial data for the graph registry"""
    print("\n📝 Inserting initial registry data...")
    
    try:
        config = DatabaseConfig()
        engine = create_engine(config.postgres_url)
        
        with engine.connect() as conn:
            # Admin user
            admin_id = uuid4()
            conn.execute(
                """INSERT INTO users (id, username, email, password_hash, full_name, is_admin) 
                   VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (username) DO NOTHING""",
                (admin_id, 'admin', 'admin@anant.ai', 'hashed_password_here', 'Registry Admin', True)
            )
            
            # Default storage location (local Parquet files)
            storage_id = uuid4()
            conn.execute(
                """INSERT INTO storage_locations (id, name, storage_type, base_path, is_active, is_default, configuration) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (storage_id, 'local_parquet', 'local', './anant_graphs', True, True, 
                 '{"format": "parquet", "compression": "snappy", "engine": "polars"}')
            )
            
            # System configuration
            configs = [
                ('registry_version', '1.0.0', 'system', 'Graph registry schema version'),
                ('default_graph_type', 'hypergraph', 'graphs', 'Default graph type for new graphs'),
                ('max_file_size_mb', '1000', 'storage', 'Maximum file size in MB'),
                ('enable_query_history', 'true', 'analytics', 'Enable query history tracking'),
                ('parquet_compression', 'snappy', 'storage', 'Default Parquet compression'),
                ('polars_lazy_evaluation', 'true', 'performance', 'Enable Polars lazy evaluation')
            ]
            
            for key, value, category, description in configs:
                conn.execute(
                    """INSERT INTO system_config (id, key, value, category, description) 
                       VALUES (%s, %s, %s, %s, %s) ON CONFLICT (key) DO NOTHING""",
                    (uuid4(), key, value, category, description)
                )
            
            # Sample graph registration (demonstrating Parquet storage)
            sample_graph_id = uuid4()
            conn.execute(
                """INSERT INTO graph_registry (
                    id, name, display_name, description, graph_type, owner_id, 
                    storage_path, node_count, edge_count, tags, properties
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    sample_graph_id, 'sample_hypergraph', 'Sample Hypergraph',
                    'Demonstration hypergraph using Parquet storage',
                    'hypergraph', admin_id, './anant_graphs/sample_hypergraph.parquet',
                    0, 0, ['demo', 'sample'], 
                    '{"storage_format": "parquet", "engine": "polars", "created_via": "registry"}'
                )
            )
            
            conn.commit()
            
        engine.dispose()
        print("✅ Initial registry data inserted successfully")
        return True
        
    except Exception as e:
        print(f"⚠️ Initial data insertion failed: {e}")
        return False


def main():
    """Main registry setup function"""
    print("🚀 Anant Graph Registry - Lightweight Schema Setup")
    print("=" * 60)
    print("📋 PostgreSQL as Graph Registry/Catalog (metadata only)")
    print("📁 Actual graph data stored in Parquet files via Anant native storage")
    print("⚡ Optimized for Polars+Parquet architecture")
    print("=" * 60)
    
    success = True
    
    # Create schema
    if not create_graph_registry_schema():
        success = False
    
    # Insert initial data
    if success:
        insert_initial_data()
    
    if success:
        print("\n🎉 Anant Graph Registry setup completed successfully!")
        print("\n📋 Registry includes:")
        print("  • 📊 Graph catalog and metadata")
        print("  • 👥 User management and authentication")
        print("  • 🔐 Access control and permissions")
        print("  • 📈 Query history and analytics")
        print("  • 🔗 Graph relationship tracking")
        print("  • 💾 Storage location management")
        print("  • 📝 Comprehensive audit trails")
        print("\n🏗️ Architecture:")
        print("  • PostgreSQL: Registry/catalog metadata only")
        print("  • Parquet files: Actual graph data (Anant native)")
        print("  • Polars: High-performance data operations")
        print("\n🚀 Registry is ready for Anant integration!")
        return True
    else:
        print("\n❌ Graph registry setup failed!")
        return False


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n💥 Registry setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)