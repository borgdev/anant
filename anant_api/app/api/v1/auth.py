"""
Authentication endpoints
Login, logout, token management, and API key operations
"""

from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...core.lifecycle import app_lifecycle
from ...services.database_service import get_db
from ...services.auth_service import auth_service
from ...middleware.auth import get_current_user, get_current_admin_user

router = APIRouter()
security = HTTPBearer()


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: Dict[str, Any]


class ApiKeyRequest(BaseModel):
    key_name: str
    permissions: List[str] = ["read"]


class ApiKeyResponse(BaseModel):
    api_key: str
    key_name: str
    permissions: List[str]
    created_at: str


@router.post("/login", response_model=LoginResponse)
async def login(
    credentials: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """Authenticate user and return JWT token"""
    try:
        # Validate credentials
        success, user_info = await auth_service.validate_user_credentials(
            credentials.username, 
            credentials.password, 
            db
        )
        
        if not success or not user_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Create access token
        access_token = auth_service.create_access_token(user_info)
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=auth_service.token_expire_minutes * 60,
            user_info={
                "user_id": user_info["user_id"],
                "username": user_info["username"],
                "email": user_info["email"],
                "is_admin": user_info["is_admin"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.post("/logout")
async def logout(request: Request):
    """Logout user (client-side token invalidation)"""
    return {
        "message": "Logged out successfully",
        "note": "Please remove the token from client storage"
    }


@router.get("/me")
async def get_current_user_info(
    current_user: dict = Depends(get_current_user)
):
    """Get current authenticated user information"""
    return {
        "user_info": current_user,
        "authenticated": True,
        "auth_method": current_user.get("token_type", "unknown")
    }


@router.post("/api-keys", response_model=ApiKeyResponse)
async def create_api_key(
    key_request: ApiKeyRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create new API key for authenticated user"""
    try:
        # Validate permissions
        valid_permissions = ["read", "write", "admin"]
        for perm in key_request.permissions:
            if perm not in valid_permissions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid permission: {perm}"
                )
        
        # Check if user can create API keys with requested permissions
        user_permissions = current_user.get("permissions", [])
        if not current_user.get("is_admin", False):
            for perm in key_request.permissions:
                if perm not in user_permissions:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Cannot create API key with permission '{perm}'"
                    )
        
        # Create API key
        api_key = await auth_service.create_api_key(
            current_user["user_id"],
            key_request.key_name,
            key_request.permissions,
            db
        )
        
        return ApiKeyResponse(
            api_key=api_key,
            key_name=key_request.key_name,
            permissions=key_request.permissions,
            created_at=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}"
        )


@router.get("/api-keys")
async def list_api_keys(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List user's API keys"""
    try:
        from sqlalchemy import text
        
        query = text("""
            SELECT id, key_name, permissions, is_active, created_at, last_used_at, usage_count
            FROM auth.api_keys
            WHERE user_id = :user_id
            ORDER BY created_at DESC
        """)
        
        result = await db.execute(query, {"user_id": current_user["user_id"]})
        api_keys = result.fetchall()
        
        return {
            "api_keys": [
                {
                    "id": key.id,
                    "key_name": key.key_name,
                    "permissions": key.permissions.split(",") if key.permissions else [],
                    "is_active": key.is_active,
                    "created_at": key.created_at.isoformat() if key.created_at else None,
                    "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
                    "usage_count": key.usage_count or 0
                }
                for key in api_keys
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list API keys: {str(e)}"
        )


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Revoke (deactivate) an API key"""
    try:
        from sqlalchemy import text
        
        # Check if user owns the key or is admin
        if not current_user.get("is_admin", False):
            check_query = text("""
                SELECT user_id FROM auth.api_keys WHERE id = :key_id
            """)
            result = await db.execute(check_query, {"key_id": key_id})
            key_owner = result.scalar()
            
            if key_owner != current_user["user_id"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot revoke API key that doesn't belong to you"
                )
        
        # Deactivate the key
        update_query = text("""
            UPDATE auth.api_keys 
            SET is_active = false, revoked_at = NOW()
            WHERE id = :key_id
            RETURNING key_name
        """)
        
        result = await db.execute(update_query, {"key_id": key_id})
        key_name = result.scalar()
        
        if not key_name:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        await db.commit()
        
        return {
            "message": f"API key '{key_name}' has been revoked",
            "revoked_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke API key: {str(e)}"
        )


@router.get("/admin/users")
async def list_users(
    admin_user: dict = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """List all users (admin only)"""
    try:
        from sqlalchemy import text
        
        query = text("""
            SELECT id, username, email, is_admin, is_active, created_at, last_login_at
            FROM auth.users
            ORDER BY created_at DESC
        """)
        
        result = await db.execute(query)
        users = result.fetchall()
        
        return {
            "users": [
                {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "is_admin": user.is_admin,
                    "is_active": user.is_active,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None
                }
                for user in users
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {str(e)}"
        )