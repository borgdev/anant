"""
Policy Engine for Governance Layer
==================================

Implements enterprise governance policies, access control, and compliance
management for metagraph knowledge systems.
"""

import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Literal
from datetime import datetime
from dataclasses import dataclass
import uuid


@dataclass
class PolicyRule:
    """Represents a governance policy rule"""
    rule_id: str
    rule_name: str
    rule_type: str  # "access", "retention", "classification", "compliance"
    conditions: Dict[str, Any]
    actions: Dict[str, Any]
    priority: int = 0
    active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class AccessControl:
    """Access control configuration"""
    user_id: str
    resource_id: str
    permissions: List[str]  # ["read", "write", "delete", "admin"]
    classification_level: str
    granted_at: datetime = None
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.granted_at is None:
            self.granted_at = datetime.now()


class PolicyEngine:
    """
    Enterprise policy engine for governance and compliance.
    
    Features:
    - Access control and permissions
    - Data classification and retention
    - Compliance rule enforcement
    - Audit logging and monitoring
    """
    
    def __init__(self, 
                 metadata_store,
                 storage_path: Path):
        """Initialize policy engine with storage."""
        self.metadata_store = metadata_store
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize policy storage
        self.policies_file = self.storage_path / "policies.parquet"
        self.access_control_file = self.storage_path / "access_control.parquet"
        self.audit_log_file = self.storage_path / "audit_log.parquet"
        
        # Load existing policies
        self._load_policies()
    
    def _load_policies(self):
        """Load existing policies from storage."""
        try:
            if self.policies_file.exists():
                self.policies_df = pl.read_parquet(self.policies_file)
            else:
                self.policies_df = pl.DataFrame({
                    "rule_id": [],
                    "rule_name": [],
                    "rule_type": [],
                    "conditions": [],
                    "actions": [],
                    "priority": [],
                    "active": [],
                    "created_at": []
                }, schema={
                    "rule_id": pl.Utf8,
                    "rule_name": pl.Utf8,
                    "rule_type": pl.Utf8,
                    "conditions": pl.Utf8,  # JSON serialized
                    "actions": pl.Utf8,     # JSON serialized
                    "priority": pl.Int32,
                    "active": pl.Boolean,
                    "created_at": pl.Datetime
                })
            
            if self.access_control_file.exists():
                self.access_df = pl.read_parquet(self.access_control_file)
            else:
                self.access_df = pl.DataFrame({
                    "user_id": [],
                    "resource_id": [],
                    "permissions": [],
                    "classification_level": [],
                    "granted_at": [],
                    "expires_at": []
                }, schema={
                    "user_id": pl.Utf8,
                    "resource_id": pl.Utf8,
                    "permissions": pl.Utf8,  # JSON serialized
                    "classification_level": pl.Utf8,
                    "granted_at": pl.Datetime,
                    "expires_at": pl.Datetime
                })
                
        except Exception as e:
            print(f"Warning: Could not load policies: {e}")
            # Initialize empty DataFrames as fallback
            self.policies_df = pl.DataFrame()
            self.access_df = pl.DataFrame()
    
    def add_policy_rule(self, policy_rule: PolicyRule) -> bool:
        """Add a new policy rule."""
        try:
            import orjson
            
            # Convert to DataFrame row
            new_row = pl.DataFrame({
                "rule_id": [policy_rule.rule_id],
                "rule_name": [policy_rule.rule_name],
                "rule_type": [policy_rule.rule_type],
                "conditions": [orjson.dumps(policy_rule.conditions).decode()],
                "actions": [orjson.dumps(policy_rule.actions).decode()],
                "priority": [policy_rule.priority],
                "active": [policy_rule.active],
                "created_at": [policy_rule.created_at]
            })
            
            # Add to policies
            self.policies_df = pl.concat([self.policies_df, new_row])
            self._save_policies()
            return True
            
        except Exception as e:
            print(f"Error adding policy rule: {e}")
            return False
    
    def add_access_control(self, access_control: AccessControl) -> bool:
        """Add access control entry."""
        try:
            import orjson
            
            # Convert to DataFrame row
            new_row = pl.DataFrame({
                "user_id": [access_control.user_id],
                "resource_id": [access_control.resource_id],
                "permissions": [orjson.dumps(access_control.permissions).decode()],
                "classification_level": [access_control.classification_level],
                "granted_at": [access_control.granted_at],
                "expires_at": [access_control.expires_at]
            })
            
            # Add to access control
            self.access_df = pl.concat([self.access_df, new_row])
            self._save_policies()
            return True
            
        except Exception as e:
            print(f"Error adding access control: {e}")
            return False
    
    def check_access(self, user_id: str, resource_id: str, permission: str) -> bool:
        """Check if user has permission for resource."""
        try:
            if self.access_df.height == 0:
                return False  # No access rules defined
                
            # Filter for user and resource
            user_access = self.access_df.filter(
                (pl.col("user_id") == user_id) &
                (pl.col("resource_id") == resource_id)
            )
            
            if user_access.height == 0:
                return False  # No access defined
            
            # Check permissions (simplified - in reality would parse JSON)
            for row in user_access.iter_rows(named=True):
                try:
                    import orjson
                    permissions = orjson.loads(row["permissions"])
                    if permission in permissions or "admin" in permissions:
                        # Check expiration
                        if row["expires_at"] is None or row["expires_at"] > datetime.now():
                            return True
                except Exception:
                    continue
            
            return False
            
        except Exception as e:
            print(f"Error checking access: {e}")
            return False  # Fail secure
    
    def get_applicable_policies(self, entity_type: str, context: Dict[str, Any]) -> List[PolicyRule]:
        """Get policies applicable to an entity type and context."""
        applicable_policies = []
        
        try:
            if self.policies_df.height == 0:
                return applicable_policies
                
            # Filter active policies
            active_policies = self.policies_df.filter(pl.col("active") == True)
            
            for row in active_policies.iter_rows(named=True):
                try:
                    import orjson
                    conditions = orjson.loads(row["conditions"])
                    
                    # Simple condition matching (can be enhanced)
                    if "entity_type" in conditions and conditions["entity_type"] == entity_type:
                        policy_rule = PolicyRule(
                            rule_id=row["rule_id"],
                            rule_name=row["rule_name"],
                            rule_type=row["rule_type"],
                            conditions=conditions,
                            actions=orjson.loads(row["actions"]),
                            priority=row["priority"],
                            active=row["active"],
                            created_at=row["created_at"]
                        )
                        applicable_policies.append(policy_rule)
                        
                except Exception as e:
                    print(f"Warning: Error processing policy {row.get('rule_id')}: {e}")
                    continue
            
            # Sort by priority (higher priority first)
            applicable_policies.sort(key=lambda p: p.priority, reverse=True)
            
        except Exception as e:
            print(f"Error getting applicable policies: {e}")
        
        return applicable_policies
    
    def log_audit_event(self, 
                       user_id: str, 
                       action: str, 
                       resource_id: str, 
                       details: Dict[str, Any] = None) -> None:
        """Log an audit event."""
        try:
            import orjson
            
            audit_entry = pl.DataFrame({
                "event_id": [str(uuid.uuid4())],
                "timestamp": [datetime.now()],
                "user_id": [user_id],
                "action": [action],
                "resource_id": [resource_id],
                "details": [orjson.dumps(details or {}).decode()],
                "success": [True]  # Assume success unless specified
            })
            
            # Append to audit log (simplified implementation)
            if self.audit_log_file.exists():
                existing_audit = pl.read_parquet(self.audit_log_file)
                combined_audit = pl.concat([existing_audit, audit_entry])
            else:
                combined_audit = audit_entry
            
            combined_audit.write_parquet(self.audit_log_file)
            
        except Exception as e:
            print(f"Warning: Could not log audit event: {e}")
    
    def _save_policies(self):
        """Save policies to storage."""
        try:
            if self.policies_df.height > 0:
                self.policies_df.write_parquet(self.policies_file)
            if self.access_df.height > 0:
                self.access_df.write_parquet(self.access_control_file)
        except Exception as e:
            print(f"Warning: Could not save policies: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get governance statistics."""
        return {
            "total_policies": self.policies_df.height if hasattr(self, 'policies_df') else 0,
            "active_policies": self.policies_df.filter(pl.col("active") == True).height if hasattr(self, 'policies_df') and self.policies_df.height > 0 else 0,
            "access_control_entries": self.access_df.height if hasattr(self, 'access_df') else 0,
            "policy_types": self.policies_df.select("rule_type").unique().to_series().to_list() if hasattr(self, 'policies_df') and self.policies_df.height > 0 else []
        }
    
    @property
    def policies(self) -> Dict[str, Dict[str, Any]]:
        """Get all policies as a dictionary."""
        try:
            if self.policies_df.is_empty():
                return {}
            
            import orjson
            policies = {}
            for row in self.policies_df.to_dicts():
                policies[row["policy_id"]] = {
                    "name": row["name"],
                    "description": row["description"],
                    "policy_type": row["policy_type"],
                    "rules": orjson.loads(row["rules"]) if row["rules"] else {},
                    "status": row["status"],
                    "created_at": row["created_at"]
                }
            return policies
        except Exception:
            return {}
    
    def create_policy(self, name: str, description: str, policy_type: str, rules: Dict[str, Any]) -> str:
        """Create a new policy."""
        try:
            import orjson
            policy_id = str(uuid.uuid4())
            
            new_policy = pl.DataFrame({
                "policy_id": [policy_id],
                "name": [name],
                "description": [description],
                "policy_type": [policy_type],
                "rules": [orjson.dumps(rules).decode()],
                "status": ["active"],
                "created_at": [datetime.now()],
                "updated_at": [datetime.now()],
                "created_by": ["system"]
            })
            
            self.policies_df = pl.concat([self.policies_df, new_policy])
            return policy_id
        except Exception:
            return ""
    
    def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific policy by ID."""
        try:
            import orjson
            policy = self.policies_df.filter(pl.col("policy_id") == policy_id)
            if policy.is_empty():
                return None
            
            row = policy.to_dicts()[0]
            return {
                "policy_id": row["policy_id"],
                "name": row["name"],
                "description": row["description"],
                "policy_type": row["policy_type"],
                "rules": orjson.loads(row["rules"]) if row["rules"] else {},
                "status": row["status"],
                "created_at": row["created_at"]
            }
        except Exception:
            return None
    
    def save(self) -> None:
        """Save policy data."""
        self._save_policies()