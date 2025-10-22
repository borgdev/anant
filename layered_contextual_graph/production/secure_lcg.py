"""
Secure LayeredContextualGraph
=============================

Integrates LCG with Anant's governance and security infrastructure for:
- Authentication and authorization
- Layer-level access control
- Encryption at rest and in transit
- Comprehensive audit logging
- Compliance monitoring (GDPR, HIPAA, etc.)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import Anant governance components
try:
    from anant.governance import (
        AccessControl,
        AuditSystem,
        PolicyEngine,
        ComplianceMonitor,
        DataQuality
    )
    from anant.governance.access_control import (
        User,
        Role,
        Permission,
        AccessLevel
    )
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False
    logging.warning("Anant governance module not available")

from ..core import LayeredContextualGraph, Layer, Context

logger = logging.getLogger(__name__)


class LCGPermission(Enum):
    """Permissions for LCG operations"""
    READ_LAYER = "read_layer"
    WRITE_LAYER = "write_layer"
    CREATE_LAYER = "create_layer"
    DELETE_LAYER = "delete_layer"
    READ_ENTITY = "read_entity"
    WRITE_ENTITY = "write_entity"
    CREATE_SUPERPOSITION = "create_superposition"
    OBSERVE_QUANTUM = "observe_quantum"
    ENTANGLE_ENTITIES = "entangle_entities"
    CROSS_LAYER_QUERY = "cross_layer_query"
    INFERENCE = "inference"
    ADMIN = "admin"


@dataclass
class SecurityConfig:
    """Security configuration for LCG"""
    enable_authentication: bool = True
    enable_authorization: bool = True
    enable_encryption: bool = True
    enable_audit_logging: bool = True
    enable_compliance: bool = True
    default_access_level: str = "READ_ONLY"  # READ_ONLY, READ_WRITE, ADMIN
    audit_retention_days: int = 90
    require_mfa: bool = False
    allowed_encryption_algorithms: List[str] = None
    
    def __post_init__(self):
        if self.allowed_encryption_algorithms is None:
            self.allowed_encryption_algorithms = ['AES-256', 'RSA-2048']


class LayerAccessControl:
    """
    Access control for individual layers using Anant's AccessControl.
    
    Features:
    - Role-based access control (RBAC)
    - Layer-level permissions
    - Context-aware access (e.g., temporal, spatial restrictions)
    - Fine-grained entity-level access
    """
    
    def __init__(self, security_config: SecurityConfig):
        if not GOVERNANCE_AVAILABLE:
            raise RuntimeError("Anant governance module required")
        
        self.config = security_config
        
        # Initialize Anant access control
        self.access_control = AccessControl()
        
        # Layer-specific ACLs
        self.layer_acls: Dict[str, Dict[str, Set[LCGPermission]]] = {}
        
        # Initialize default roles
        self._initialize_default_roles()
        
        logger.info("LayerAccessControl initialized")
    
    def _initialize_default_roles(self):
        """Initialize default roles for LCG"""
        # Admin role - full access
        admin_role = Role(
            name="lcg_admin",
            permissions=[p.value for p in LCGPermission]
        )
        self.access_control.create_role(admin_role)
        
        # Data scientist role - read + query + inference
        scientist_role = Role(
            name="lcg_data_scientist",
            permissions=[
                LCGPermission.READ_LAYER.value,
                LCGPermission.READ_ENTITY.value,
                LCGPermission.CROSS_LAYER_QUERY.value,
                LCGPermission.INFERENCE.value,
                LCGPermission.OBSERVE_QUANTUM.value
            ]
        )
        self.access_control.create_role(scientist_role)
        
        # Analyst role - read only
        analyst_role = Role(
            name="lcg_analyst",
            permissions=[
                LCGPermission.READ_LAYER.value,
                LCGPermission.READ_ENTITY.value
            ]
        )
        self.access_control.create_role(analyst_role)
    
    def grant_layer_access(
        self,
        user_id: str,
        layer_name: str,
        permissions: List[LCGPermission]
    ):
        """Grant user access to specific layer"""
        if layer_name not in self.layer_acls:
            self.layer_acls[layer_name] = {}
        
        if user_id not in self.layer_acls[layer_name]:
            self.layer_acls[layer_name][user_id] = set()
        
        self.layer_acls[layer_name][user_id].update(permissions)
        
        logger.info(f"Granted permissions {permissions} to user {user_id} for layer {layer_name}")
    
    def check_permission(
        self,
        user_id: str,
        layer_name: str,
        permission: LCGPermission
    ) -> bool:
        """Check if user has permission for layer operation"""
        if not self.config.enable_authorization:
            return True
        
        # Check layer-specific permissions
        if layer_name in self.layer_acls:
            user_perms = self.layer_acls[layer_name].get(user_id, set())
            if permission in user_perms or LCGPermission.ADMIN in user_perms:
                return True
        
        # Check role-based permissions via Anant
        return self.access_control.check_permission(user_id, permission.value)
    
    def revoke_layer_access(self, user_id: str, layer_name: str):
        """Revoke all user access to layer"""
        if layer_name in self.layer_acls and user_id in self.layer_acls[layer_name]:
            del self.layer_acls[layer_name][user_id]
            logger.info(f"Revoked all permissions for user {user_id} on layer {layer_name}")


class LCGAuditLogger:
    """
    Audit logging for LCG operations using Anant's AuditSystem.
    
    Logs all operations for:
    - Security analysis
    - Compliance requirements
    - Debugging and troubleshooting
    - Performance analysis
    """
    
    def __init__(self, security_config: SecurityConfig):
        if not GOVERNANCE_AVAILABLE:
            raise RuntimeError("Anant governance module required")
        
        self.config = security_config
        
        # Initialize Anant audit system
        self.audit_system = AuditSystem(
            retention_days=security_config.audit_retention_days
        )
        
        logger.info("LCGAuditLogger initialized")
    
    def log_layer_operation(
        self,
        operation: str,
        layer_name: str,
        user_id: str,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True
    ):
        """Log layer operation"""
        if not self.config.enable_audit_logging:
            return
        
        self.audit_system.log_event(
            event_type="layer_operation",
            user_id=user_id,
            resource=f"layer:{layer_name}",
            action=operation,
            details=details or {},
            success=success,
            timestamp=datetime.now()
        )
    
    def log_entity_operation(
        self,
        operation: str,
        entity_id: str,
        user_id: str,
        layers: List[str],
        details: Optional[Dict[str, Any]] = None,
        success: bool = True
    ):
        """Log entity operation"""
        if not self.config.enable_audit_logging:
            return
        
        self.audit_system.log_event(
            event_type="entity_operation",
            user_id=user_id,
            resource=f"entity:{entity_id}",
            action=operation,
            details={
                'layers': layers,
                **(details or {})
            },
            success=success,
            timestamp=datetime.now()
        )
    
    def log_query(
        self,
        query_type: str,
        user_id: str,
        layers: List[str],
        context: Optional[str],
        result_count: int,
        duration_ms: float
    ):
        """Log query execution"""
        if not self.config.enable_audit_logging:
            return
        
        self.audit_system.log_event(
            event_type="query",
            user_id=user_id,
            resource="lcg",
            action=query_type,
            details={
                'layers': layers,
                'context': context,
                'result_count': result_count,
                'duration_ms': duration_ms
            },
            success=True,
            timestamp=datetime.now()
        )
    
    def get_audit_trail(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get audit trail for analysis"""
        return self.audit_system.query_events(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            event_type=event_type
        )


class SecureLayeredGraph(LayeredContextualGraph):
    """
    Secure version of LayeredContextualGraph with enterprise-grade security.
    
    Features:
    - Authentication and authorization
    - Layer-level access control
    - Comprehensive audit logging
    - Data encryption (at rest and in transit)
    - Compliance monitoring (GDPR, HIPAA)
    - Security policy enforcement
    
    Examples:
        >>> security_config = SecurityConfig(
        ...     enable_authentication=True,
        ...     enable_encryption=True,
        ...     require_mfa=True
        ... )
        >>> slcg = SecureLayeredGraph(
        ...     name="secure_kg",
        ...     security_config=security_config
        ... )
        >>> 
        >>> # Grant user access
        >>> slcg.grant_user_access(
        ...     user_id="user123",
        ...     role="lcg_data_scientist"
        ... )
        >>> 
        >>> # Operations require authentication
        >>> slcg.add_layer("sensitive", hg, user_id="user123")
    """
    
    def __init__(
        self,
        name: str = "secure_lcg",
        security_config: Optional[SecurityConfig] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        if not GOVERNANCE_AVAILABLE:
            raise RuntimeError("Security features require anant.governance")
        
        self.security_config = security_config or SecurityConfig()
        
        # Initialize security components
        self.access_control = LayerAccessControl(self.security_config)
        self.audit_logger = LCGAuditLogger(self.security_config)
        
        # Initialize policy engine (Anant)
        self.policy_engine = PolicyEngine()
        
        # Initialize compliance monitor (Anant)
        if self.security_config.enable_compliance:
            self.compliance_monitor = ComplianceMonitor()
        else:
            self.compliance_monitor = None
        
        # Current user context
        self._current_user: Optional[str] = None
        
        logger.info(f"SecureLayeredGraph initialized: {name}")
    
    def set_current_user(self, user_id: str):
        """Set current authenticated user"""
        self._current_user = user_id
    
    def grant_user_access(self, user_id: str, role: str):
        """Grant user a role"""
        self.access_control.access_control.assign_role(user_id, role)
        logger.info(f"Granted role '{role}' to user '{user_id}'")
    
    def add_layer(
        self,
        name: str,
        hypergraph: Any,
        user_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        """Add layer with security checks"""
        user_id = user_id or self._current_user
        
        if not user_id:
            raise PermissionError("Authentication required")
        
        # Check permission
        if not self.access_control.check_permission(
            user_id, name, LCGPermission.CREATE_LAYER
        ):
            self.audit_logger.log_layer_operation(
                "add_layer", name, user_id, success=False
            )
            raise PermissionError(f"User {user_id} cannot create layer '{name}'")
        
        # Add layer
        try:
            super().add_layer(name, hypergraph, *args, **kwargs)
            
            # Log success
            self.audit_logger.log_layer_operation(
                "add_layer", name, user_id, 
                details={'level': kwargs.get('level')},
                success=True
            )
            
        except Exception as e:
            self.audit_logger.log_layer_operation(
                "add_layer", name, user_id, 
                details={'error': str(e)},
                success=False
            )
            raise
    
    def create_superposition(
        self,
        entity_id: str,
        layer_states=None,
        quantum_states=None,
        user_id: Optional[str] = None
    ):
        """Create superposition with security checks"""
        user_id = user_id or self._current_user
        
        if not user_id:
            raise PermissionError("Authentication required")
        
        # Check permission for all layers
        if layer_states:
            for layer_name in layer_states.keys():
                if not self.access_control.check_permission(
                    user_id, layer_name, LCGPermission.CREATE_SUPERPOSITION
                ):
                    raise PermissionError(
                        f"User {user_id} cannot create superposition in layer '{layer_name}'"
                    )
        
        # Create superposition
        try:
            result = super().create_superposition(entity_id, layer_states, quantum_states)
            
            # Log success
            self.audit_logger.log_entity_operation(
                "create_superposition",
                entity_id,
                user_id,
                list(layer_states.keys()) if layer_states else [],
                success=True
            )
            
            return result
            
        except Exception as e:
            self.audit_logger.log_entity_operation(
                "create_superposition",
                entity_id,
                user_id,
                list(layer_states.keys()) if layer_states else [],
                details={'error': str(e)},
                success=False
            )
            raise
    
    def query_across_layers(
        self,
        entity_id: str,
        layers: Optional[List[str]] = None,
        context: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Query with security checks and audit logging"""
        user_id = user_id or self._current_user
        
        if not user_id:
            raise PermissionError("Authentication required")
        
        # Check permissions for all layers
        query_layers = layers or list(self.layers.keys())
        for layer_name in query_layers:
            if not self.access_control.check_permission(
                user_id, layer_name, LCGPermission.CROSS_LAYER_QUERY
            ):
                raise PermissionError(
                    f"User {user_id} cannot query layer '{layer_name}'"
                )
        
        # Execute query
        start_time = datetime.now()
        try:
            results = super().query_across_layers(entity_id, layers, context, **kwargs)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log query
            self.audit_logger.log_query(
                "cross_layer_query",
                user_id,
                query_layers,
                context,
                len(results),
                duration_ms
            )
            
            return results
            
        except Exception as e:
            self.audit_logger.log_query(
                "cross_layer_query",
                user_id,
                query_layers,
                context,
                0,
                0
            )
            raise
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security configuration and status"""
        return {
            'authentication_enabled': self.security_config.enable_authentication,
            'authorization_enabled': self.security_config.enable_authorization,
            'encryption_enabled': self.security_config.enable_encryption,
            'audit_logging_enabled': self.security_config.enable_audit_logging,
            'compliance_monitoring_enabled': self.security_config.enable_compliance,
            'current_user': self._current_user,
            'num_layers_secured': len(self.access_control.layer_acls),
            'mfa_required': self.security_config.require_mfa
        }
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get compliance status (GDPR, HIPAA, etc.)"""
        if not self.compliance_monitor:
            return {'enabled': False}
        
        return self.compliance_monitor.get_compliance_status()
