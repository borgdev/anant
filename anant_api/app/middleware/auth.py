"""
Authentication Middleware
========================

JWT and API key authentication middleware for Ray-deployed FastAPI app.
"""

import jwt
import time
from typing import Optional, Tuple
from fastapi import HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..config import settings
from ..services.auth_service import auth_service
from ..services.database_service import get_db


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware supporting JWT tokens and API keys"""
    
    def __init__(self, app, excluded_paths: Optional[list] = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/",
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/api/v1/health",
            "/api/v1/database/health",
            "/api/v1/database/status"
        ]
        self.security = HTTPBearer(auto_error=False)
    
    async def dispatch(self, request: Request, call_next):
        """Process authentication for each request"""
        
        # Skip authentication for excluded paths
        if self._is_excluded_path(request.url.path):
            return await call_next(request)
        
        # Check for authentication
        auth_result = await self._authenticate_request(request)
        
        if not auth_result[0]:  # Authentication failed
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Authentication required",
                    "detail": auth_result[1],
                    "timestamp": time.time()
                }
            )
        
        # Add user info to request state
        request.state.user = auth_result[1]
        request.state.auth_method = auth_result[2]
        
        response = await call_next(request)
        return response
    
    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from authentication"""
        for excluded in self.excluded_paths:
            if path.startswith(excluded):
                return True
        return False
    
    async def _authenticate_request(self, request: Request) -> Tuple[bool, dict, str]:
        """
        Authenticate request using JWT token or API key with database validation
        Returns: (success, user_info/error_message, auth_method)
        """
        
        # Get database session
        try:
            from ..core.lifecycle import app_lifecycle
            if not app_lifecycle.database_service.is_connected():
                # Fallback to basic validation if database is not available
                return await self._fallback_authentication(request)
            
            async for db in app_lifecycle.database_service.get_session():
                # Try API key authentication first
                api_key = self._extract_api_key(request)
                if api_key:
                    success, result = await auth_service.authenticate_api_key(api_key, db)
                    if success:
                        return (True, result, "api_key")
                
                # Try JWT token authentication
                token = self._extract_jwt_token(request)
                if token:
                    success, result = await auth_service.authenticate_jwt_token(token, db)
                    if success:
                        return (True, result, "jwt_token")
                
                return (False, {"error": "No valid authentication provided"}, "none")
                
        except Exception as e:
            # Fallback to basic validation on error
            return await self._fallback_authentication(request)
    
    async def _fallback_authentication(self, request: Request) -> Tuple[bool, dict, str]:
        """Fallback authentication when database is unavailable"""
        # Try API key authentication first
        api_key = self._extract_api_key(request)
        if api_key:
            result = await self._authenticate_api_key_fallback(api_key)
            if result[0]:
                return (True, result[1], "api_key")
        
        # Try JWT token authentication
        token = self._extract_jwt_token(request)
        if token:
            result = await self._authenticate_jwt_token_fallback(token)
            if result[0]:
                return (True, result[1], "jwt_token")
        
        return (False, {"error": "No valid authentication provided"}, "none")
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request headers"""
        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
        
        # Check query parameter
        api_key = request.query_params.get("api_key")
        if api_key:
            return api_key
        
        return None
    
    def _extract_jwt_token(self, request: Request) -> Optional[str]:
        """Extract JWT token from Authorization header"""
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            return authorization[7:]  # Remove "Bearer " prefix
        return None
    
    async def _authenticate_api_key_fallback(self, api_key: str) -> Tuple[bool, dict]:
        """
        Authenticate using API key
        In production, this would check against database
        """
        try:
            # Placeholder API key validation
            # In real implementation, query database for API key
            if api_key.startswith("ak_"):  # Valid API key format
                return (True, {
                    "user_id": "api_user",
                    "username": "api_user",
                    "is_api_key": True,
                    "permissions": ["read", "write"],
                    "api_key": api_key[:8] + "..."  # Truncated for security
                })
            else:
                return (False, {"error": "Invalid API key format"})
        
        except Exception as e:
            return (False, {"error": f"API key validation failed: {str(e)}"})
    
    async def _authenticate_jwt_token_fallback(self, token: str) -> Tuple[bool, dict]:
        """
        Authenticate using JWT token
        """
        try:
            # Decode JWT token
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=[settings.ALGORITHM]
            )
            
            # Check token expiration
            exp = payload.get("exp")
            if exp and exp < time.time():
                return (False, {"error": "Token has expired"})
            
            # Extract user information
            user_info = {
                "user_id": payload.get("sub"),
                "username": payload.get("username"),
                "email": payload.get("email"),
                "is_admin": payload.get("is_admin", False),
                "is_api_key": False,
                "permissions": payload.get("permissions", ["read"]),
                "token_type": "jwt"
            }
            
            return (True, user_info)
        
        except jwt.ExpiredSignatureError:
            return (False, {"error": "Token has expired"})
        
        except jwt.InvalidTokenError:
            return (False, {"error": "Invalid token"})
        
        except Exception as e:
            return (False, {"error": f"Token validation failed: {str(e)}"})


def get_current_user(request: Request) -> dict:
    """Dependency to get current authenticated user"""
    if not hasattr(request.state, 'user'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return request.state.user


def get_current_admin_user(request: Request) -> dict:
    """Dependency to get current admin user"""
    user = get_current_user(request)
    if not user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


def require_permissions(required_permissions: list):
    """Dependency factory to require specific permissions"""
    def permission_checker(request: Request) -> dict:
        user = get_current_user(request)
        user_permissions = user.get("permissions", [])
        
        for permission in required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
        
        return user
    
    return permission_checker