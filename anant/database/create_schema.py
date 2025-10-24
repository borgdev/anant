#!/usr/bin/env python3
"""
Standalone Database Schema Creation
=================================

Creates the database schema directly using SQLAlchemy without Alembic for now.
This gets us up and running quickly.
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
    ARRAY, DECIMAL, func
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from typing import Optional


class DatabaseConfig(BaseSettings):
    """Minimal database configuration"""
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT") 
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="anant_meta", env="POSTGRES_DB")
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


def create_database_schema():
    """Create the complete database schema"""
    print("🏗️ Creating database schema...")
    
    config = DatabaseConfig()
    engine = create_engine(config.postgres_url)
    metadata = MetaData()
    
    # Users table
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
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Column('last_login', DateTime(timezone=True)),
        Column('preferences', JSONB, default={}),
        Index('idx_users_username', 'username'),
        Index('idx_users_email', 'email'),
        Index('idx_users_active', 'is_active')
    )
    
    # Concepts table
    concepts = Table(
        'concepts',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('user_id', PG_UUID(as_uuid=True), ForeignKey('users.id'), nullable=False),
        Column('name', String(255), nullable=False),
        Column('namespace', String(255)),
        Column('type', String(100)),
        Column('description', Text),
        Column('properties', JSONB, default={}),
        Column('synonyms', ARRAY(String)),
        Column('tags', ARRAY(String)),
        Column('version', Integer, default=1),
        Column('is_active', Boolean, default=True),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Index('idx_concepts_name', 'name'),
        Index('idx_concepts_namespace', 'namespace'),
        Index('idx_concepts_type', 'type'),
        Index('idx_concepts_user', 'user_id'),
        Index('idx_concepts_active', 'is_active')
    )
    
    # Graphs table
    graphs = Table(
        'graphs',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('user_id', PG_UUID(as_uuid=True), ForeignKey('users.id'), nullable=False),
        Column('name', String(255), nullable=False),
        Column('description', Text),
        Column('graph_type', String(100), default='hypergraph'),
        Column('status', String(50), default='draft'),
        Column('version', Integer, default=1),
        Column('entity_id', PG_UUID(as_uuid=True)),  # For versioning
        Column('is_public', Boolean, default=False),
        Column('data_path', String(500)),  # File storage path
        Column('metadata', JSONB, default={}),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Index('idx_graphs_name', 'name'),
        Index('idx_graphs_user', 'user_id'),
        Index('idx_graphs_type', 'graph_type'),
        Index('idx_graphs_status', 'status'),
        Index('idx_graphs_public', 'is_public')
    )
    
    # Mappings table
    mappings = Table(
        'mappings',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('user_id', PG_UUID(as_uuid=True), ForeignKey('users.id'), nullable=False),
        Column('source_concept_id', PG_UUID(as_uuid=True), ForeignKey('concepts.id'), nullable=False),
        Column('target_concept_id', PG_UUID(as_uuid=True), ForeignKey('concepts.id'), nullable=False),
        Column('mapping_type', String(100), nullable=False),
        Column('confidence_score', DECIMAL(5, 4)),
        Column('context', JSONB, default={}),
        Column('version', Integer, default=1),
        Column('entity_id', PG_UUID(as_uuid=True)),  # For versioning
        Column('is_active', Boolean, default=True),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Index('idx_mappings_source', 'source_concept_id'),
        Index('idx_mappings_target', 'target_concept_id'),
        Index('idx_mappings_type', 'mapping_type'),
        Index('idx_mappings_user', 'user_id'),
        Index('idx_mappings_active', 'is_active')
    )
    
    # Security Policies table
    security_policies = Table(
        'security_policies',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('name', String(255), nullable=False),
        Column('resource_type', String(100), nullable=False),
        Column('action', String(100), nullable=False),
        Column('effect', String(20), nullable=False),  # ALLOW/DENY
        Column('policy_data', JSONB, nullable=False),
        Column('is_active', Boolean, default=True),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Index('idx_security_policies_resource', 'resource_type'),
        Index('idx_security_policies_action', 'action'),
        Index('idx_security_policies_active', 'is_active')
    )
    
    # Concept Relationships table
    concept_relationships = Table(
        'concept_relationships',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('source_concept_id', PG_UUID(as_uuid=True), ForeignKey('concepts.id'), nullable=False),
        Column('target_concept_id', PG_UUID(as_uuid=True), ForeignKey('concepts.id'), nullable=False),
        Column('relationship_type', String(100), nullable=False),
        Column('properties', JSONB, default={}),
        Column('weight', DECIMAL(10, 6)),
        Column('is_active', Boolean, default=True),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Index('idx_concept_relationships_source', 'source_concept_id'),
        Index('idx_concept_relationships_target', 'target_concept_id'),
        Index('idx_concept_relationships_type', 'relationship_type')
    )
    
    # Graph Metadata table
    graph_metadata = Table(
        'graph_metadata',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('graph_id', PG_UUID(as_uuid=True), ForeignKey('graphs.id'), nullable=False),
        Column('key', String(255), nullable=False),
        Column('value', Text),
        Column('value_type', String(50), default='string'),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Index('idx_graph_metadata_graph', 'graph_id'),
        Index('idx_graph_metadata_key', 'key')
    )
    
    # Mapping Metadata table
    mapping_metadata = Table(
        'mapping_metadata',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('mapping_id', PG_UUID(as_uuid=True), ForeignKey('mappings.id'), nullable=False),
        Column('key', String(255), nullable=False),
        Column('value', Text),
        Column('value_type', String(50), default='string'),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Index('idx_mapping_metadata_mapping', 'mapping_id'),
        Index('idx_mapping_metadata_key', 'key')
    )
    
    # Security Audit Logs table
    security_audit_logs = Table(
        'security_audit_logs',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('user_id', PG_UUID(as_uuid=True), ForeignKey('users.id')),
        Column('action', String(255), nullable=False),
        Column('resource_type', String(100)),
        Column('resource_id', String(255)),
        Column('success', Boolean, nullable=False),
        Column('details', JSONB, default={}),
        Column('ip_address', String(50)),
        Column('user_agent', String(500)),
        Column('timestamp', DateTime(timezone=True), server_default=func.now()),
        Index('idx_security_audit_logs_user', 'user_id'),
        Index('idx_security_audit_logs_action', 'action'),
        Index('idx_security_audit_logs_timestamp', 'timestamp'),
        Index('idx_security_audit_logs_success', 'success')
    )
    
    # Ontology Mappings table
    ontology_mappings = Table(
        'ontology_mappings',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('source_ontology', String(255), nullable=False),
        Column('target_ontology', String(255), nullable=False),
        Column('source_term', String(255), nullable=False),
        Column('target_term', String(255), nullable=False),
        Column('mapping_type', String(100), nullable=False),
        Column('confidence', DECIMAL(5, 4)),
        Column('context', JSONB, default={}),
        Column('is_active', Boolean, default=True),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Index('idx_ontology_mappings_source', 'source_ontology'),
        Index('idx_ontology_mappings_target', 'target_ontology'),
        Index('idx_ontology_mappings_type', 'mapping_type')
    )
    
    # System Configurations table
    system_configurations = Table(
        'system_configurations',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('key', String(255), unique=True, nullable=False),
        Column('value', Text),
        Column('value_type', String(50), default='string'),
        Column('category', String(100), default='general'),
        Column('description', Text),
        Column('is_sensitive', Boolean, default=False),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Index('idx_system_configurations_key', 'key'),
        Index('idx_system_configurations_category', 'category')
    )
    
    # Cache Entries table
    cache_entries = Table(
        'cache_entries',
        metadata,
        Column('id', PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
        Column('key', String(500), unique=True, nullable=False),
        Column('value', JSONB, nullable=False),
        Column('expires_at', DateTime(timezone=True)),
        Column('created_at', DateTime(timezone=True), server_default=func.now()),
        Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Index('idx_cache_entries_key', 'key'),
        Index('idx_cache_entries_expires', 'expires_at')
    )
    
    try:
        # Create all tables
        print("📋 Creating database tables...")
        metadata.create_all(engine)
        
        # Verify tables were created
        from sqlalchemy import inspect
        inspector = inspect(engine)
        created_tables = inspector.get_table_names()
        
        expected_tables = [
            'users', 'concepts', 'graphs', 'mappings', 'security_policies',
            'concept_relationships', 'graph_metadata', 'mapping_metadata',
            'security_audit_logs', 'ontology_mappings', 'system_configurations',
            'cache_entries'
        ]
        
        print(f"✅ Created {len(created_tables)} tables:")
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
        print(f"❌ Failed to create schema: {e}")
        return False


def insert_sample_data():
    """Insert some sample data for testing"""
    print("\n📝 Inserting sample data...")
    
    try:
        config = DatabaseConfig()
        engine = create_engine(config.postgres_url)
        
        with engine.connect() as conn:
            # Sample admin user
            user_id = uuid4()
            conn.execute(
                "INSERT INTO users (id, username, email, password_hash, full_name, is_admin) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (username) DO NOTHING",
                (user_id, 'admin', 'admin@anant.ai', 'hashed_password', 'Admin User', True)
            )
            
            # Sample configuration
            conn.execute(
                "INSERT INTO system_configurations (id, key, value, category, description) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (key) DO NOTHING",
                (uuid4(), 'database_version', '1.0.0', 'system', 'Database schema version')
            )
            
            # Sample security policy
            conn.execute(
                "INSERT INTO security_policies (id, name, resource_type, action, effect, policy_data) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (uuid4(), 'Admin Full Access', 'all', 'all', 'ALLOW', '{"roles": ["admin"]}')
            )
            
            conn.commit()
            
        engine.dispose()
        print("✅ Sample data inserted successfully")
        return True
        
    except Exception as e:
        print(f"⚠️ Sample data insertion failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 Anant Integration - Standalone Database Schema Setup")
    print("=" * 60)
    
    success = True
    
    # Create schema
    if not create_database_schema():
        success = False
    
    # Insert sample data
    if success:
        insert_sample_data()  # Non-critical, so don't fail on error
    
    if success:
        print("\n🎉 Database schema setup completed successfully!")
        print("\n📋 Database is ready with:")
        print("  • 12 core tables with proper relationships")
        print("  • Indexes for performance")
        print("  • UUID primary keys") 
        print("  • JSONB columns for flexible data")
        print("  • Audit trails and metadata")
        print("\n🚀 You can now use the anant_integration platform!")
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