-- Development Database Initialization
-- Creates basic auth schema and tables for development

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create auth schema
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

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON auth.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON auth.users(email);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON auth.api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON auth.api_keys(user_id);

-- Insert default admin user (password: admin123)
INSERT INTO auth.users (username, email, full_name, password_hash, is_admin) 
VALUES ('admin', 'admin@anant.dev', 'Dev Admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewAYTyGq1Nc1a.K.', true)
ON CONFLICT (username) DO NOTHING;

-- Insert default API key for testing (key: dev_test_key_123)
INSERT INTO auth.api_keys (key_hash, name, user_id, permissions)
SELECT 
    'da39a3ee5e6b4b0d3255bfef95601890afd80709',  -- sha1 hash of "dev_test_key_123"
    'Development Test API Key',
    u.id,
    '{"read": true, "write": true, "admin": true}'::jsonb
FROM auth.users u WHERE u.username = 'admin'
ON CONFLICT (key_hash) DO NOTHING;

-- Success message
SELECT 'Development database initialized successfully' as status;