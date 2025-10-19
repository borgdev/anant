"""
ANANT Access Control System

Role-based access control (RBAC) and permission management
for hypergraph data governance and security.
"""

import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum
from dataclasses import dataclass, field
import json
import logging
import hashlib
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class PermissionType(Enum):
    """Types of permissions in the system"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    EXPORT = "export"
    IMPORT = "import"
    SHARE = "share"

class ResourceType(Enum):
    """Types of resources that can be protected"""
    HYPERGRAPH = "hypergraph"
    NODE = "node"
    EDGE = "edge"
    DATASET = "dataset"
    POLICY = "policy"
    REPORT = "report"
    SYSTEM = "system"

class AccessDecision(Enum):
    """Access control decision results"""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"

@dataclass
class Permission:
    """Individual permission definition"""
    id: str
    name: str
    description: str
    permission_type: PermissionType
    resource_type: ResourceType
    resource_pattern: str = "*"  # Wildcard or specific resource ID
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Role:
    """Role definition with permissions"""
    id: str
    name: str
    description: str
    permissions: List[str] = field(default_factory=list)  # Permission IDs
    parent_roles: List[str] = field(default_factory=list)  # Inherit from parent roles
    
    # Role properties
    is_system_role: bool = False
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Constraints
    max_users: Optional[int] = None
    session_timeout_minutes: Optional[int] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class User:
    """User account with roles and permissions"""
    id: str
    username: str
    email: str
    roles: List[str] = field(default_factory=list)  # Role IDs
    direct_permissions: List[str] = field(default_factory=list)  # Direct permission IDs
    
    # User properties
    is_active: bool = True
    is_system_user: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    
    # Security properties
    password_hash: Optional[str] = None
    mfa_enabled: bool = False
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    # Session management
    active_sessions: List[str] = field(default_factory=list)
    max_concurrent_sessions: int = 5
    
    # Metadata
    groups: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessSession:
    """User access session"""
    id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: Optional[datetime] = None
    
    # Session context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    source: str = "unknown"
    
    # Session state
    is_active: bool = True
    permissions_cache: Dict[str, bool] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        if self.expires_at and datetime.now() > self.expires_at:
            return True
        return False

@dataclass
class AccessRequest:
    """Access control request"""
    user_id: str
    session_id: Optional[str]
    resource_type: ResourceType
    resource_id: str
    permission_type: PermissionType
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AccessResult:
    """Access control decision result"""
    decision: AccessDecision
    reason: str
    permissions_checked: List[str] = field(default_factory=list)
    conditions_met: List[str] = field(default_factory=list)
    conditions_failed: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AccessController:
    """Role-based access control system"""
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        
        # Core data structures
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.permissions: Dict[str, Permission] = {}
        self.sessions: Dict[str, AccessSession] = {}
        
        # Caching for performance
        self.user_permissions_cache: Dict[str, Set[str]] = {}
        self.role_hierarchy_cache: Dict[str, Set[str]] = {}
        
        # Configuration
        self.default_session_timeout = timedelta(hours=8)
        self.max_login_attempts = 3
        self.lockout_duration = timedelta(minutes=30)
        self.cache_ttl = timedelta(minutes=15)
        
        # Statistics
        self.stats = {
            'total_access_requests': 0,
            'access_granted': 0,
            'access_denied': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'active_sessions': 0,
            'failed_logins': 0
        }
        
        # Create built-in permissions and roles
        self._create_built_in_permissions()
        self._create_built_in_roles()
        
        logger.info("Access control system initialized")
    
    def _create_built_in_permissions(self):
        """Create built-in system permissions"""
        built_in_permissions = [
            # Hypergraph permissions
            Permission(
                id="hypergraph_read",
                name="Read Hypergraph",
                description="Read access to hypergraph data",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.HYPERGRAPH
            ),
            Permission(
                id="hypergraph_write",
                name="Write Hypergraph",
                description="Write access to hypergraph data",
                permission_type=PermissionType.WRITE,
                resource_type=ResourceType.HYPERGRAPH
            ),
            Permission(
                id="hypergraph_delete",
                name="Delete Hypergraph",
                description="Delete access to hypergraph data",
                permission_type=PermissionType.DELETE,
                resource_type=ResourceType.HYPERGRAPH
            ),
            
            # Node permissions
            Permission(
                id="node_read",
                name="Read Nodes",
                description="Read access to individual nodes",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.NODE
            ),
            Permission(
                id="node_write",
                name="Write Nodes",
                description="Write access to individual nodes",
                permission_type=PermissionType.WRITE,
                resource_type=ResourceType.NODE
            ),
            
            # Dataset permissions
            Permission(
                id="dataset_export",
                name="Export Dataset",
                description="Export dataset to external formats",
                permission_type=PermissionType.EXPORT,
                resource_type=ResourceType.DATASET
            ),
            Permission(
                id="dataset_import",
                name="Import Dataset",
                description="Import dataset from external sources",
                permission_type=PermissionType.IMPORT,
                resource_type=ResourceType.DATASET
            ),
            
            # Policy permissions
            Permission(
                id="policy_read",
                name="Read Policies",
                description="Read governance policies",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.POLICY
            ),
            Permission(
                id="policy_write",
                name="Write Policies",
                description="Create and modify governance policies",
                permission_type=PermissionType.WRITE,
                resource_type=ResourceType.POLICY
            ),
            
            # System permissions
            Permission(
                id="system_admin",
                name="System Administration",
                description="Full system administration access",
                permission_type=PermissionType.ADMIN,
                resource_type=ResourceType.SYSTEM
            )
        ]
        
        for permission in built_in_permissions:
            self.add_permission(permission)
    
    def _create_built_in_roles(self):
        """Create built-in system roles"""
        
        # Super Administrator Role
        super_admin = Role(
            id="super_admin",
            name="Super Administrator",
            description="Full system access with all permissions",
            permissions=list(self.permissions.keys()),  # All permissions
            is_system_role=True,
            tags=["system", "admin"]
        )
        
        # Data Administrator Role
        data_admin = Role(
            id="data_admin",
            name="Data Administrator",
            description="Full data management access",
            permissions=[
                "hypergraph_read", "hypergraph_write", "hypergraph_delete",
                "node_read", "node_write",
                "dataset_export", "dataset_import"
            ],
            is_system_role=True,
            tags=["data", "admin"]
        )
        
        # Policy Administrator Role
        policy_admin = Role(
            id="policy_admin",
            name="Policy Administrator",
            description="Governance policy management access",
            permissions=[
                "policy_read", "policy_write",
                "hypergraph_read"
            ],
            is_system_role=True,
            tags=["policy", "governance"]
        )
        
        # Data Analyst Role
        analyst = Role(
            id="data_analyst",
            name="Data Analyst",
            description="Read-only data analysis access",
            permissions=[
                "hypergraph_read", "node_read",
                "dataset_export"
            ],
            is_system_role=True,
            session_timeout_minutes=480,  # 8 hours
            tags=["analyst", "readonly"]
        )
        
        # Basic User Role
        basic_user = Role(
            id="basic_user",
            name="Basic User",
            description="Basic read access to approved data",
            permissions=["hypergraph_read", "node_read"],
            is_system_role=True,
            session_timeout_minutes=240,  # 4 hours
            tags=["user", "basic"]
        )
        
        for role in [super_admin, data_admin, policy_admin, analyst, basic_user]:
            self.add_role(role)
    
    def add_permission(self, permission: Permission) -> None:
        """Add a new permission to the system"""
        self.permissions[permission.id] = permission
        self._invalidate_caches()
        logger.info(f"Added permission: {permission.name} ({permission.id})")
    
    def add_role(self, role: Role) -> None:
        """Add a new role to the system"""
        self.roles[role.id] = role
        role.updated_at = datetime.now()
        self._invalidate_caches()
        logger.info(f"Added role: {role.name} ({role.id})")
    
    def add_user(self, user: User) -> None:
        """Add a new user to the system"""
        self.users[user.id] = user
        self._invalidate_caches()
        logger.info(f"Added user: {user.username} ({user.id})")
    
    def create_user(self, 
                   username: str,
                   email: str,
                   roles: Optional[List[str]] = None,
                   password: Optional[str] = None) -> str:
        """Create a new user with specified roles"""
        user_id = str(uuid.uuid4())
        
        # Validate roles exist
        roles = roles or ["basic_user"]
        for role_id in roles:
            if role_id not in self.roles:
                raise ValueError(f"Role not found: {role_id}")
        
        # Hash password if provided
        password_hash = None
        if password:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            roles=roles,
            password_hash=password_hash
        )
        
        self.add_user(user)
        return user_id
    
    def authenticate_user(self, 
                         username: str,
                         password: str,
                         context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Authenticate user and create session"""
        context = context or {}
        
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            self.stats['failed_logins'] += 1
            logger.warning(f"Authentication failed: user not found - {username}")
            return None
        
        # Check if user is locked
        if user.locked_until and datetime.now() < user.locked_until:
            logger.warning(f"Authentication failed: user locked - {username}")
            return None
        
        # Check if user is active
        if not user.is_active:
            logger.warning(f"Authentication failed: user inactive - {username}")
            return None
        
        # Verify password
        if user.password_hash:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if user.password_hash != password_hash:
                user.login_attempts += 1
                
                # Lock user if max attempts exceeded
                if user.login_attempts >= self.max_login_attempts:
                    user.locked_until = datetime.now() + self.lockout_duration
                    logger.warning(f"User locked due to failed login attempts: {username}")
                
                self.stats['failed_logins'] += 1
                logger.warning(f"Authentication failed: invalid password - {username}")
                return None
        
        # Reset login attempts on successful authentication
        user.login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        
        # Create new session
        session_id = self.create_session(user.id, context)
        
        logger.info(f"User authenticated successfully: {username}")
        return session_id
    
    def create_session(self, user_id: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create a new user session"""
        context = context or {}
        
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        # Check concurrent session limit
        active_sessions = [s for s in self.sessions.values() 
                          if s.user_id == user_id and s.is_active and not s.is_expired()]
        
        if len(active_sessions) >= user.max_concurrent_sessions:
            # Terminate oldest session
            oldest_session = min(active_sessions, key=lambda s: s.created_at)
            self.terminate_session(oldest_session.id)
        
        # Create new session
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Determine session timeout
        timeout = self.default_session_timeout
        for role_id in user.roles:
            role = self.roles.get(role_id)
            if role and role.session_timeout_minutes:
                role_timeout = timedelta(minutes=role.session_timeout_minutes)
                timeout = min(timeout, role_timeout)
        
        session = AccessSession(
            id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            expires_at=now + timeout,
            ip_address=context.get('ip_address'),
            user_agent=context.get('user_agent'),
            source=context.get('source', 'web')
        )
        
        self.sessions[session_id] = session
        user.active_sessions.append(session_id)
        
        self.stats['active_sessions'] += 1
        
        logger.info(f"Created session for user {user_id}: {session_id}")
        return session_id
    
    def terminate_session(self, session_id: str) -> bool:
        """Terminate a user session"""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        session.is_active = False
        
        # Remove from user's active sessions
        user = self.users.get(session.user_id)
        if user and session_id in user.active_sessions:
            user.active_sessions.remove(session_id)
        
        self.stats['active_sessions'] = max(0, self.stats['active_sessions'] - 1)
        
        logger.info(f"Terminated session: {session_id}")
        return True
    
    def check_access(self, request: AccessRequest) -> AccessResult:
        """Check if user has access to requested resource"""
        self.stats['total_access_requests'] += 1
        
        # Get user
        user = self.users.get(request.user_id)
        if not user:
            result = AccessResult(
                decision=AccessDecision.DENY,
                reason=f"User not found: {request.user_id}"
            )
            self.stats['access_denied'] += 1
            return result
        
        # Check if user is active
        if not user.is_active:
            result = AccessResult(
                decision=AccessDecision.DENY,
                reason="User account is inactive"
            )
            self.stats['access_denied'] += 1
            return result
        
        # Validate session if provided
        if request.session_id:
            session = self.sessions.get(request.session_id)
            if not session or not session.is_active or session.is_expired():
                result = AccessResult(
                    decision=AccessDecision.DENY,
                    reason="Invalid or expired session"
                )
                self.stats['access_denied'] += 1
                return result
            
            # Update session activity
            session.last_activity = datetime.now()
        
        # Get user's effective permissions
        user_permissions = self._get_user_permissions(request.user_id)
        
        # Check for matching permissions
        matching_permissions = []
        for perm_id in user_permissions:
            permission = self.permissions.get(perm_id)
            if not permission:
                continue
            
            # Check permission type and resource type match
            if (permission.permission_type == request.permission_type and
                permission.resource_type == request.resource_type):
                
                # Check resource pattern match
                if self._matches_resource_pattern(
                    request.resource_id, 
                    permission.resource_pattern
                ):
                    matching_permissions.append(perm_id)
        
        if matching_permissions:
            # Check conditions for all matching permissions
            conditions_met = []
            conditions_failed = []
            
            for perm_id in matching_permissions:
                permission = self.permissions[perm_id]
                if permission.conditions:
                    condition_result = self._evaluate_conditions(
                        permission.conditions, 
                        request.context
                    )
                    if condition_result:
                        conditions_met.append(perm_id)
                    else:
                        conditions_failed.append(perm_id)
                else:
                    conditions_met.append(perm_id)
            
            # Grant access if at least one permission's conditions are met
            if conditions_met:
                result = AccessResult(
                    decision=AccessDecision.ALLOW,
                    reason="Access granted based on user permissions",
                    permissions_checked=matching_permissions,
                    conditions_met=conditions_met,
                    conditions_failed=conditions_failed
                )
                self.stats['access_granted'] += 1
            else:
                result = AccessResult(
                    decision=AccessDecision.DENY,
                    reason="Permission conditions not met",
                    permissions_checked=matching_permissions,
                    conditions_failed=conditions_failed
                )
                self.stats['access_denied'] += 1
        else:
            result = AccessResult(
                decision=AccessDecision.DENY,
                reason="No matching permissions found",
                permissions_checked=[]
            )
            self.stats['access_denied'] += 1
        
        # Log access attempt if audit system is available
        if self.audit_system:
            from .audit_system import AuditEvent, AuditEventType, AuditCategory, AuditLevel
            
            self.audit_system.log_event(AuditEvent(
                id="",
                timestamp=datetime.now(),
                event_type=AuditEventType.DATA_READ if request.permission_type == PermissionType.READ else AuditEventType.DATA_WRITE,
                category=AuditCategory.DATA_ACCESS,
                level=AuditLevel.INFO if result.decision == AccessDecision.ALLOW else AuditLevel.WARN,
                source="access_controller",
                user_id=request.user_id,
                session_id=request.session_id,
                resource_id=request.resource_id,
                resource_type=request.resource_type.value,
                action=request.permission_type.value,
                outcome="success" if result.decision == AccessDecision.ALLOW else "failure",
                description=result.reason,
                metadata={
                    'permissions_checked': result.permissions_checked,
                    'conditions_met': result.conditions_met,
                    'context': request.context
                }
            ))
        
        return result
    
    def _get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all effective permissions for a user (including inherited from roles)"""
        # Check cache first
        cache_key = user_id
        if cache_key in self.user_permissions_cache:
            self.stats['cache_hits'] += 1
            return self.user_permissions_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        user = self.users.get(user_id)
        if not user:
            return set()
        
        permissions = set()
        
        # Add direct permissions
        permissions.update(user.direct_permissions)
        
        # Add permissions from roles (including inherited roles)
        for role_id in user.roles:
            role_permissions = self._get_role_permissions(role_id)
            permissions.update(role_permissions)
        
        # Cache the result
        self.user_permissions_cache[cache_key] = permissions
        
        return permissions
    
    def _get_role_permissions(self, role_id: str) -> Set[str]:
        """Get all permissions for a role (including inherited from parent roles)"""
        # Check cache first
        if role_id in self.role_hierarchy_cache:
            return self.role_hierarchy_cache[role_id]
        
        role = self.roles.get(role_id)
        if not role:
            return set()
        
        permissions = set()
        
        # Add direct permissions
        permissions.update(role.permissions)
        
        # Add permissions from parent roles (recursive)
        for parent_role_id in role.parent_roles:
            parent_permissions = self._get_role_permissions(parent_role_id)
            permissions.update(parent_permissions)
        
        # Cache the result
        self.role_hierarchy_cache[role_id] = permissions
        
        return permissions
    
    def _matches_resource_pattern(self, resource_id: str, pattern: str) -> bool:
        """Check if resource ID matches the permission pattern"""
        if pattern == "*":
            return True
        
        if pattern == resource_id:
            return True
        
        # Simple wildcard matching (can be enhanced with regex)
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return resource_id.startswith(prefix)
        
        return False
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate permission conditions against request context"""
        for key, expected_value in conditions.items():
            if key not in context:
                return False
            
            context_value = context[key]
            
            # Simple equality check (can be enhanced with more operators)
            if isinstance(expected_value, list):
                if context_value not in expected_value:
                    return False
            elif context_value != expected_value:
                return False
        
        return True
    
    def _invalidate_caches(self):
        """Invalidate permission caches"""
        self.user_permissions_cache.clear()
        self.role_hierarchy_cache.clear()
    
    def assign_role_to_user(self, user_id: str, role_id: str) -> bool:
        """Assign a role to a user"""
        user = self.users.get(user_id)
        role = self.roles.get(role_id)
        
        if not user or not role:
            return False
        
        if role_id not in user.roles:
            user.roles.append(role_id)
            self._invalidate_caches()
            logger.info(f"Assigned role {role_id} to user {user_id}")
            return True
        
        return False
    
    def remove_role_from_user(self, user_id: str, role_id: str) -> bool:
        """Remove a role from a user"""
        user = self.users.get(user_id)
        
        if not user or role_id not in user.roles:
            return False
        
        user.roles.remove(role_id)
        self._invalidate_caches()
        logger.info(f"Removed role {role_id} from user {user_id}")
        return True
    
    def grant_permission_to_user(self, user_id: str, permission_id: str) -> bool:
        """Grant a direct permission to a user"""
        user = self.users.get(user_id)
        permission = self.permissions.get(permission_id)
        
        if not user or not permission:
            return False
        
        if permission_id not in user.direct_permissions:
            user.direct_permissions.append(permission_id)
            self._invalidate_caches()
            logger.info(f"Granted permission {permission_id} to user {user_id}")
            return True
        
        return False
    
    def revoke_permission_from_user(self, user_id: str, permission_id: str) -> bool:
        """Revoke a direct permission from a user"""
        user = self.users.get(user_id)
        
        if not user or permission_id not in user.direct_permissions:
            return False
        
        user.direct_permissions.remove(permission_id)
        self._invalidate_caches()
        logger.info(f"Revoked permission {permission_id} from user {user_id}")
        return True
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.is_expired() or not session.is_active:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.terminate_session(session_id)
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information (excluding sensitive data)"""
        user = self.users.get(user_id)
        if not user:
            return None
        
        return {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'roles': user.roles,
            'is_active': user.is_active,
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'mfa_enabled': user.mfa_enabled,
            'active_sessions': len([s for s in self.sessions.values() 
                                  if s.user_id == user_id and s.is_active]),
            'groups': user.groups,
            'effective_permissions': list(self._get_user_permissions(user_id))
        }
    
    def get_role_info(self, role_id: str) -> Optional[Dict[str, Any]]:
        """Get role information"""
        role = self.roles.get(role_id)
        if not role:
            return None
        
        return {
            'id': role.id,
            'name': role.name,
            'description': role.description,
            'permissions': role.permissions,
            'parent_roles': role.parent_roles,
            'is_system_role': role.is_system_role,
            'is_active': role.is_active,
            'created_at': role.created_at.isoformat(),
            'updated_at': role.updated_at.isoformat(),
            'max_users': role.max_users,
            'session_timeout_minutes': role.session_timeout_minutes,
            'tags': role.tags,
            'effective_permissions': list(self._get_role_permissions(role_id))
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get access control statistics"""
        stats = self.stats.copy()
        
        # Add additional statistics
        stats['total_users'] = len(self.users)
        stats['active_users'] = len([u for u in self.users.values() if u.is_active])
        stats['total_roles'] = len(self.roles)
        stats['total_permissions'] = len(self.permissions)
        stats['total_sessions'] = len(self.sessions)
        
        # Calculate success rate
        total_requests = self.stats['total_access_requests']
        if total_requests > 0:
            access_success_rate = (self.stats['access_granted'] / total_requests) * 100
        else:
            access_success_rate = 0.0
        
        # Cache statistics
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            cache_hit_rate = (
                self.stats['cache_hits'] / 
                (self.stats['cache_hits'] + self.stats['cache_misses'])
            ) * 100
        else:
            cache_hit_rate = 0.0
        
        # Create result dictionary with all statistics
        result: Dict[str, Any] = {}
        result.update(stats)
        result['access_success_rate'] = access_success_rate
        result['cache_hit_rate'] = cache_hit_rate
        
        return result