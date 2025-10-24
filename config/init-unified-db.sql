-- Anant Enterprise Platform - Unified Database Initialization
-- This script creates both the enterprise and registry databases in a single PostgreSQL instance

-- Create the registry database for graph metadata
CREATE DATABASE anant_registry;

-- Create the enterprise user (keeping postgres as superuser)
CREATE USER anant WITH ENCRYPTED PASSWORD 'anant_secure_2024';

-- Grant privileges to the anant user on both databases
GRANT ALL PRIVILEGES ON DATABASE anant_enterprise TO anant;
GRANT ALL PRIVILEGES ON DATABASE anant_registry TO anant;
GRANT ALL PRIVILEGES ON DATABASE anant_registry TO postgres;

-- Connect to registry database and set up schema
\c anant_registry;

-- Create registry schema for graph metadata
CREATE SCHEMA IF NOT EXISTS registry;

-- Create tables for graph registry
CREATE TABLE IF NOT EXISTS registry.graphs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    version VARCHAR(50) DEFAULT '1.0.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    parquet_path VARCHAR(500),
    status VARCHAR(50) DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS registry.graph_schemas (
    id SERIAL PRIMARY KEY,
    graph_id INTEGER REFERENCES registry.graphs(id) ON DELETE CASCADE,
    schema_name VARCHAR(255) NOT NULL,
    schema_definition JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS registry.graph_lineage (
    id SERIAL PRIMARY KEY,
    source_graph_id INTEGER REFERENCES registry.graphs(id),
    target_graph_id INTEGER REFERENCES registry.graphs(id),
    relationship_type VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_graphs_name ON registry.graphs(name);
CREATE INDEX IF NOT EXISTS idx_graphs_status ON registry.graphs(status);
CREATE INDEX IF NOT EXISTS idx_schemas_graph_id ON registry.graph_schemas(graph_id);
CREATE INDEX IF NOT EXISTS idx_lineage_source ON registry.graph_lineage(source_graph_id);
CREATE INDEX IF NOT EXISTS idx_lineage_target ON registry.graph_lineage(target_graph_id);

-- Grant permissions on registry schema
GRANT USAGE ON SCHEMA registry TO anant;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA registry TO anant;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA registry TO anant;

-- Connect to enterprise database and set up schema
\c anant_enterprise;

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enterprise schemas
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS security;
CREATE SCHEMA IF NOT EXISTS auth;

-- Authentication tables
CREATE TABLE IF NOT EXISTS auth.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API Keys for service authentication
CREATE TABLE IF NOT EXISTS auth.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    permissions JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP WITH TIME ZONE
);

-- Create enterprise tables
CREATE TABLE IF NOT EXISTS analytics.query_history (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    execution_time_ms INTEGER,
    user_id UUID REFERENCES auth.users(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    result_count INTEGER,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tags JSONB DEFAULT '{}',
    source VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS security.access_logs (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    action VARCHAR(100),
    resource VARCHAR(200),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT true
);

-- Ray cluster information
CREATE TABLE IF NOT EXISTS monitoring.ray_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cluster_name VARCHAR(255) UNIQUE NOT NULL,
    head_node_address VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'initializing',
    configuration JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Ray tasks tracking
CREATE TABLE IF NOT EXISTS monitoring.ray_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) UNIQUE NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    cluster_id UUID REFERENCES monitoring.ray_clusters(id),
    user_id UUID REFERENCES auth.users(id),
    status VARCHAR(50) DEFAULT 'pending',
    parameters JSONB DEFAULT '{}',
    result JSONB,
    execution_time_ms INTEGER,
    node_id VARCHAR(255),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for enterprise tables
CREATE INDEX IF NOT EXISTS idx_users_username ON auth.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON auth.users(email);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON auth.api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON auth.api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_query_history_timestamp ON analytics.query_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_history_user ON analytics.query_history(user_id);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON monitoring.system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON monitoring.system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_access_logs_timestamp ON security.access_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_access_logs_user ON security.access_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_ray_tasks_status ON monitoring.ray_tasks(status);
CREATE INDEX IF NOT EXISTS idx_ray_tasks_type ON monitoring.ray_tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_ray_tasks_created_at ON monitoring.ray_tasks(created_at);

-- Grant permissions on enterprise schemas
GRANT USAGE ON SCHEMA analytics TO anant;
GRANT USAGE ON SCHEMA monitoring TO anant;
GRANT USAGE ON SCHEMA security TO anant;
GRANT USAGE ON SCHEMA auth TO anant;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO anant;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO anant;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA security TO anant;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA auth TO anant;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO anant;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO anant;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA security TO anant;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA auth TO anant;

-- Insert default data
\c anant_registry;
INSERT INTO registry.graphs (name, description, version, metadata) 
VALUES 
    ('default', 'Default Anant graph', '1.0.0', '{"type": "default", "created_by": "system"}'),
    ('schema_org', 'Schema.org knowledge graph', '1.0.0', '{"type": "ontology", "source": "schema.org"}')
ON CONFLICT (name) DO NOTHING;

-- Connect back to default database
\c anant_enterprise;

-- Insert default admin user (password: admin123 - should be changed in production)
INSERT INTO auth.users (username, email, full_name, password_hash, is_admin) 
VALUES ('admin', 'admin@anant.ai', 'System Administrator', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewAYTyGq1Nc1a.K.', true)
ON CONFLICT (username) DO NOTHING;

-- Insert default API key for testing (key: test_api_key_123)
INSERT INTO auth.api_keys (key_hash, name, user_id, permissions)
SELECT 
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewAYTyGq1Nc1a.K.',
    'Default Test API Key',
    u.id,
    '{"read": true, "write": true, "admin": true}'::jsonb
FROM auth.users u WHERE u.username = 'admin'
ON CONFLICT (key_hash) DO NOTHING;

-- Final message
SELECT 'Unified database initialization completed successfully' as status;