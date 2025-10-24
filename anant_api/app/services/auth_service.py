"""
Authentication service
Handles user authentication, API key validation, and JWT token management
"""

import hashlib
import secrets
import time
import json
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
import jwt
import bcrypt

from ..config import settings
from ..services.database_service import get_db


class AuthenticationService:
    """Service for handling authentication operations"""
    
    def __init__(self):
        self.jwt_algorithm = settings.ALGORITHM
        self.secret_key = settings.SECRET_KEY
        self.token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    
    async def authenticate_api_key(self, api_key: str, db: AsyncSession) -> Tuple[bool, Dict[str, Any]]:
        """
        Authenticate using API key against database
        """
        try:
            # Validate API key format first
            if not api_key or len(api_key) < 8:
                return (False, {"error": "Invalid API key format"})
            
            # Hash the API key for lookup (assuming keys are stored hashed)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Query the database for the API key
            query = text("""
                SELECT ak.id, ak.name, ak.permissions, ak.is_active, ak.expires_at,
                       u.id as user_id, u.username, u.email, u.is_admin, u.is_active as user_active
                FROM auth.api_keys ak
                LEFT JOIN auth.users u ON ak.user_id = u.id
                WHERE ak.key_hash = :key_hash 
                AND ak.is_active = true 
                AND (ak.expires_at IS NULL OR ak.expires_at > NOW())
                AND (u.is_active = true OR u.is_active IS NULL)
            """)
            
            result = await db.execute(query, {"key_hash": key_hash})
            api_key_record = result.fetchone()
            
            if api_key_record:
                # Update last used timestamp
                update_query = text("""
                    UPDATE auth.api_keys 
                    SET last_used_at = NOW()
                    WHERE key_hash = :key_hash
                """)
                await db.execute(update_query, {"key_hash": key_hash})
                await db.commit()
                
                # Parse permissions from JSONB
                permissions = []
                if api_key_record.permissions:
                    if isinstance(api_key_record.permissions, dict):
                        # Extract permissions from JSONB dict
                        permissions = [k for k, v in api_key_record.permissions.items() if v]
                    else:
                        # Handle string format if needed
                        permissions = [p.strip() for p in str(api_key_record.permissions).split(",") if p.strip()]
                
                # Return user info
                user_info = {
                    "user_id": api_key_record.user_id or f"api_user_{api_key_record.id}",
                    "username": api_key_record.username or f"api_user_{api_key_record.id}",
                    "email": api_key_record.email or "api@anant.local",
                    "is_admin": api_key_record.is_admin or False,
                    "is_api_key": True,
                    "api_key_id": api_key_record.id,
                    "api_key_name": api_key_record.name,
                    "permissions": permissions or ["read"],
                    "api_key": api_key[:8] + "..." + api_key[-4:],  # Show first 8 and last 4 chars
                    "auth_source": "database"
                }
                
                return (True, user_info)
            else:
                # Check for development/fallback API keys
                return await self._authenticate_api_key_fallback(api_key)
                
        except Exception as e:
            # Fallback to development keys
            return await self._authenticate_api_key_fallback(api_key, str(e))
    
    async def _authenticate_api_key_fallback(self, api_key: str, error_reason: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Fallback API key authentication for development"""
        try:
            # Development API keys
            if api_key.startswith("ak_dev_"):
                return (True, {
                    "user_id": "dev_user",
                    "username": "dev_user", 
                    "email": "dev@anant.local",
                    "is_admin": True,
                    "is_api_key": True,
                    "permissions": ["read", "write", "admin"],
                    "api_key": api_key[:8] + "...",
                    "auth_source": "fallback",
                    "fallback_reason": error_reason or "database_unavailable"
                })
            
            # Admin override key for emergencies
            if api_key == "ak_admin_override_emergency":
                return (True, {
                    "user_id": "emergency_admin",
                    "username": "emergency_admin",
                    "email": "admin@anant.local", 
                    "is_admin": True,
                    "is_api_key": True,
                    "permissions": ["read", "write", "admin", "emergency"],
                    "api_key": "ak_admin_...",
                    "auth_source": "emergency_override"
                })
            
            return (False, {"error": "Invalid API key", "auth_source": "fallback"})
            
        except Exception as e:
            return (False, {"error": f"API key validation failed: {str(e)}", "auth_source": "fallback"})
    
    async def authenticate_jwt_token(self, token: str, db: AsyncSession) -> Tuple[bool, Dict[str, Any]]:
        """
        Authenticate using JWT token
        """
        try:
            # Decode JWT token
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.jwt_algorithm]
            )
            
            # Check token expiration
            exp = payload.get("exp")
            if exp and exp < time.time():
                return (False, {"error": "Token has expired"})
            
            user_id = payload.get("sub")
            if not user_id:
                return (False, {"error": "Invalid token: missing user ID"})
            
            # Query database for user info
            query = text("""
                SELECT id, username, email, is_admin, is_active, created_at
                FROM auth.users 
                WHERE id = :user_id AND is_active = true
            """)
            
            result = await db.execute(query, {"user_id": user_id})
            user_record = result.fetchone()
            
            if user_record:
                # Update last activity (using updated_at since we don't have last_activity_at)
                update_query = text("""
                    UPDATE auth.users 
                    SET updated_at = NOW()
                    WHERE id = :user_id
                """)
                await db.execute(update_query, {"user_id": user_id})
                await db.commit()
                
                # Extract user information
                user_info = {
                    "user_id": user_record.id,
                    "username": user_record.username,
                    "email": user_record.email,
                    "is_admin": user_record.is_admin,
                    "is_api_key": False,
                    "permissions": payload.get("permissions", ["read"]),
                    "token_type": "jwt",
                    "created_at": user_record.created_at.isoformat() if user_record.created_at else None,
                    "auth_source": "database"
                }
                
                return (True, user_info)
            else:
                # User not found in database, but token is valid - use fallback
                return await self._authenticate_jwt_token_fallback(token, payload)
                
        except jwt.ExpiredSignatureError:
            return (False, {"error": "Token has expired"})
        
        except jwt.InvalidTokenError:
            return (False, {"error": "Invalid token"})
        
        except Exception as e:
            # Fallback for development
            return await self._authenticate_jwt_token_fallback(token, None, str(e))
    
    async def _authenticate_jwt_token_fallback(self, token: str, payload: Optional[dict] = None, error_reason: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Fallback JWT token authentication"""
        try:
            if not payload:
                # Try to decode without verification for fallback
                payload = jwt.decode(
                    token, 
                    self.secret_key, 
                    algorithms=[self.jwt_algorithm],
                    options={"verify_exp": False}  # Skip expiration for fallback
                )
            
            if payload:
                return (True, {
                    "user_id": payload.get("sub", "dev_user"),
                    "username": payload.get("username", "dev_user"),
                    "email": payload.get("email", "dev@anant.local"),
                    "is_admin": payload.get("is_admin", True),
                    "is_api_key": False,
                    "permissions": payload.get("permissions", ["read", "write", "admin"]),
                    "token_type": "jwt",
                    "auth_source": "fallback",
                    "fallback_reason": error_reason or "database_unavailable"
                })
            else:
                return (False, {"error": "Failed to decode token", "auth_source": "fallback"})
                
        except Exception as e:
            return (False, {"error": f"Token validation failed: {str(e)}", "auth_source": "fallback"})
    
    def create_access_token(self, user_data: dict) -> str:
        """Create JWT access token"""
        expire = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
        
        to_encode = {
            "sub": str(user_data["user_id"]),
            "username": user_data["username"],
            "email": user_data.get("email"),
            "is_admin": user_data.get("is_admin", False),
            "permissions": user_data.get("permissions", ["read"]),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access_token"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.jwt_algorithm)
        return encoded_jwt
    
    async def create_api_key(self, user_id: str, key_name: str, permissions: list, db: AsyncSession) -> str:
        """Create new API key for user"""
        try:
            # Generate random API key
            api_key = f"ak_{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Convert permissions list to JSONB format
            permissions_dict = {perm: True for perm in permissions}
            
            # Insert into database
            query = text("""
                INSERT INTO auth.api_keys (user_id, name, key_hash, permissions, is_active, created_at)
                VALUES (:user_id, :key_name, :key_hash, :permissions, true, NOW())
                RETURNING id
            """)
            
            result = await db.execute(query, {
                "user_id": user_id,
                "key_name": key_name,
                "key_hash": key_hash,
                "permissions": permissions_dict
            })
            await db.commit()
            
            return api_key
            
        except Exception as e:
            await db.rollback()
            raise Exception(f"Failed to create API key: {str(e)}")
    
    async def validate_user_credentials(self, username: str, password: str, db: AsyncSession) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate user login credentials"""
        try:
            query = text("""
                SELECT id, username, email, password_hash, is_admin, is_active
                FROM auth.users 
                WHERE username = :username AND is_active = true
            """)
            
            result = await db.execute(query, {"username": username})
            user_record = result.fetchone()
            
            if not user_record:
                return (False, None)
            
            # Verify password
            if bcrypt.checkpw(password.encode('utf-8'), user_record.password_hash.encode('utf-8')):
                # Update updated_at timestamp (since we don't have last_login_at in the schema)
                update_query = text("""
                    UPDATE auth.users 
                    SET updated_at = NOW()
                    WHERE id = :user_id
                """)
                await db.execute(update_query, {"user_id": user_record.id})
                await db.commit()
                
                user_info = {
                    "user_id": user_record.id,
                    "username": user_record.username,
                    "email": user_record.email,
                    "is_admin": user_record.is_admin,
                    "is_active": user_record.is_active
                }
                
                return (True, user_info)
            
            return (False, None)
            
        except Exception as e:
            return (False, None)
    
    def hash_password(self, password: str) -> str:
        """Hash password for storage"""
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8')


# Global authentication service instance
auth_service = AuthenticationService()