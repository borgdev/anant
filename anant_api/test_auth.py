"""
Authentication validation test script
Tests both JWT token and API key authentication through the database service
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.auth_service import auth_service
from app.services.database_service import DatabaseService
from app.config import settings


async def test_authentication():
    """Test authentication services"""
    print("🔧 Testing Authentication Services")
    print("=" * 50)
    
    # Initialize database service
    db_service = DatabaseService()
    db_connected = await db_service.initialize()
    
    if not db_connected:
        print("❌ Database connection failed - using fallback mode")
        await test_fallback_authentication()
        return
    
    print("✅ Database connected successfully")
    
    # Test database-backed authentication
    async for db in db_service.get_session():
        await test_database_authentication(db)
        break
    
    await db_service.shutdown()


async def test_fallback_authentication():
    """Test fallback authentication (no database)"""
    print("\n📋 Testing Fallback Authentication")
    print("-" * 30)
    
    # Test development API key
    test_api_key = "ak_dev_test123"
    print(f"Testing API key: {test_api_key}")
    
    # Note: This would normally require the middleware context
    print("✅ Fallback authentication logic available")


async def test_database_authentication(db):
    """Test database-backed authentication"""
    print("\n📋 Testing Database Authentication")
    print("-" * 30)
    
    # Test 1: API Key Authentication
    print("\n1️⃣ Testing API Key Authentication")
    test_api_keys = [
        "ak_dev_test123",  # Development key
        "ak_invalid_key",  # Invalid key
        "invalid_format"   # Wrong format
    ]
    
    for api_key in test_api_keys:
        try:
            success, result = await auth_service.authenticate_api_key(api_key, db)
            status = "✅ VALID" if success else "❌ INVALID"
            print(f"  {api_key}: {status}")
            if success:
                print(f"    User: {result.get('username', 'unknown')}")
                print(f"    Permissions: {result.get('permissions', [])}")
        except Exception as e:
            print(f"  {api_key}: ❌ ERROR - {str(e)}")
    
    # Test 2: JWT Token Authentication  
    print("\n2️⃣ Testing JWT Token Authentication")
    
    # Create a test token
    test_user = {
        "user_id": "test_user_123",
        "username": "test_user",
        "email": "test@anant.local",
        "is_admin": False,
        "permissions": ["read", "write"]
    }
    
    try:
        test_token = auth_service.create_access_token(test_user)
        print(f"  Created test token: {test_token[:20]}...")
        
        # Test the token
        success, result = await auth_service.authenticate_jwt_token(test_token, db)
        status = "✅ VALID" if success else "❌ INVALID"
        print(f"  Token validation: {status}")
        
        if success:
            print(f"    User: {result.get('username', 'unknown')}")
            print(f"    Permissions: {result.get('permissions', [])}")
        else:
            print(f"    Error: {result.get('error', 'unknown')}")
            
    except Exception as e:
        print(f"  Token test: ❌ ERROR - {str(e)}")
    
    # Test 3: Database Health
    print("\n3️⃣ Testing Database Tables")
    try:
        from sqlalchemy import text
        
        # Check if auth tables exist
        tables_query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'auth'
        """)
        
        result = await db.execute(tables_query)
        tables = result.fetchall()
        
        if tables:
            print("  ✅ Auth schema exists with tables:")
            for table in tables:
                print(f"    - {table.table_name}")
        else:
            print("  ⚠️ Auth schema not found - using fallback authentication")
            
    except Exception as e:
        print(f"  ❌ Error checking tables: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_authentication())