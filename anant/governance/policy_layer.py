"""
Policy Layer for Metagraph - Phase 1
====================================

Manages governance policies, compliance tracking, and access control
for enterprise knowledge management with audit trails.
"""

import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Literal, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import orjson
import uuid
import hashlib

# Type aliases
PolicyType = Literal["access", "data_quality", "retention", "classification", "lineage", "compliance"]
AccessLevel = Literal["read", "write", "delete", "admin", "none"]
ComplianceStatus = Literal["compliant", "non_compliant", "unknown", "exempt"]
DataClassification = Literal["public", "internal", "confidential", "restricted", "top_secret"]
AuditEventType = Literal["policy_created", "policy_updated", "policy_deleted", "access_granted", "access_denied", "compliance_check", "violation_detected"]
ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"]


@dataclass
class Policy:
    """Represents a governance policy."""
    policy_id: str
    name: str
    policy_type: PolicyType
    description: str
    rules: Dict[str, Any]
    enabled: bool
    created_by: str
    created_at: datetime
    updated_at: datetime


@dataclass
class AccessRequest:
    """Represents an access request."""
    request_id: str
    user_id: str
    resource_id: str
    access_level: AccessLevel
    justification: str
    status: Literal["pending", "approved", "denied", "expired"]
    requested_at: datetime
    reviewed_by: Optional[str]
    reviewed_at: Optional[datetime]


@dataclass
class AuditEvent:
    """Represents an audit event."""
    event_id: str
    event_type: AuditEventType
    user_id: str
    resource_id: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]


class PolicyEngine:
    """
    Enterprise policy engine for governance and compliance.
    
    Manages access control, data quality policies, compliance tracking,
    and audit trails with high-performance Polars+Parquet backend.
    """
    
    def __init__(self, 
                 storage_path: str = "./metagraph_governance",
                 compression: ParquetCompression = "zstd",
                 audit_retention_days: int = 2555):  # 7 years
        """
        Initialize policy engine.
        
        Args:
            storage_path: Path for storing governance data
            compression: Parquet compression algorithm
            audit_retention_days: Days to retain audit logs
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.compression: ParquetCompression = compression
        self.audit_retention_days = audit_retention_days
        
        # File paths
        self.policies_file = self.storage_path / "policies.parquet"
        self.access_requests_file = self.storage_path / "access_requests.parquet"
        self.permissions_file = self.storage_path / "permissions.parquet"
        self.audit_log_file = self.storage_path / "audit_log.parquet"
        self.compliance_file = self.storage_path / "compliance.parquet"
        self.classifications_file = self.storage_path / "classifications.parquet"
        
        # Initialize data structures
        self._init_policies()
        self._init_access_requests()
        self._init_permissions()
        self._init_audit_log()
        self._init_compliance()
        self._init_classifications()
        
        # Policy evaluation cache
        self._policy_cache: Dict[str, bool] = {}
        self._permission_cache: Dict[str, Set[str]] = {}
        
        # Built-in policy validators
        self._policy_validators: Dict[PolicyType, Callable] = {
            "data_quality": self._validate_data_quality_policy,
            "access": self._validate_access_policy,
            "retention": self._validate_retention_policy,
            "classification": self._validate_classification_policy,
            "lineage": self._validate_lineage_policy,
            "compliance": self._validate_compliance_policy
        }
    
    def _init_policies(self):
        """Initialize policies DataFrame."""
        if self.policies_file.exists():
            self.policies_df = pl.read_parquet(self.policies_file)
        else:
            self.policies_df = pl.DataFrame({
                "policy_id": pl.Series([], dtype=pl.Utf8),
                "name": pl.Series([], dtype=pl.Utf8),
                "policy_type": pl.Series([], dtype=pl.Utf8),
                "description": pl.Series([], dtype=pl.Utf8),
                "rules": pl.Series([], dtype=pl.Utf8),  # JSON
                "enabled": pl.Series([], dtype=pl.Boolean),
                "created_by": pl.Series([], dtype=pl.Utf8),
                "created_at": pl.Series([], dtype=pl.Datetime),
                "updated_at": pl.Series([], dtype=pl.Datetime),
                "version": pl.Series([], dtype=pl.Int64),
                "tags": pl.Series([], dtype=pl.List(pl.Utf8)),
                "priority": pl.Series([], dtype=pl.Int32),
                "scope": pl.Series([], dtype=pl.Utf8),  # JSON - entities/resources this applies to
                "metadata": pl.Series([], dtype=pl.Utf8)  # JSON
            })
    
    def _init_access_requests(self):
        """Initialize access requests DataFrame."""
        if self.access_requests_file.exists():
            self.access_requests_df = pl.read_parquet(self.access_requests_file)
        else:
            self.access_requests_df = pl.DataFrame({
                "request_id": pl.Series([], dtype=pl.Utf8),
                "user_id": pl.Series([], dtype=pl.Utf8),
                "resource_id": pl.Series([], dtype=pl.Utf8),
                "access_level": pl.Series([], dtype=pl.Utf8),
                "justification": pl.Series([], dtype=pl.Utf8),
                "status": pl.Series([], dtype=pl.Utf8),
                "requested_at": pl.Series([], dtype=pl.Datetime),
                "reviewed_by": pl.Series([], dtype=pl.Utf8),
                "reviewed_at": pl.Series([], dtype=pl.Datetime),
                "expires_at": pl.Series([], dtype=pl.Datetime),
                "approval_conditions": pl.Series([], dtype=pl.Utf8),  # JSON
                "metadata": pl.Series([], dtype=pl.Utf8)  # JSON
            })
    
    def _init_permissions(self):
        """Initialize permissions DataFrame."""
        if self.permissions_file.exists():
            self.permissions_df = pl.read_parquet(self.permissions_file)
        else:
            self.permissions_df = pl.DataFrame({
                "permission_id": pl.Series([], dtype=pl.Utf8),
                "user_id": pl.Series([], dtype=pl.Utf8),
                "resource_id": pl.Series([], dtype=pl.Utf8),
                "access_level": pl.Series([], dtype=pl.Utf8),
                "granted_by": pl.Series([], dtype=pl.Utf8),
                "granted_at": pl.Series([], dtype=pl.Datetime),
                "expires_at": pl.Series([], dtype=pl.Datetime),
                "conditions": pl.Series([], dtype=pl.Utf8),  # JSON
                "source_request_id": pl.Series([], dtype=pl.Utf8),
                "revoked": pl.Series([], dtype=pl.Boolean),
                "revoked_at": pl.Series([], dtype=pl.Datetime),
                "revoked_by": pl.Series([], dtype=pl.Utf8),
                "metadata": pl.Series([], dtype=pl.Utf8)  # JSON
            })
    
    def _init_audit_log(self):
        """Initialize audit log DataFrame."""
        if self.audit_log_file.exists():
            self.audit_log_df = pl.read_parquet(self.audit_log_file)
        else:
            self.audit_log_df = pl.DataFrame({
                "event_id": pl.Series([], dtype=pl.Utf8),
                "event_type": pl.Series([], dtype=pl.Utf8),
                "user_id": pl.Series([], dtype=pl.Utf8),
                "resource_id": pl.Series([], dtype=pl.Utf8),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "details": pl.Series([], dtype=pl.Utf8),  # JSON
                "ip_address": pl.Series([], dtype=pl.Utf8),
                "user_agent": pl.Series([], dtype=pl.Utf8),
                "session_id": pl.Series([], dtype=pl.Utf8),
                "outcome": pl.Series([], dtype=pl.Utf8),
                "risk_score": pl.Series([], dtype=pl.Float64),
                "geolocation": pl.Series([], dtype=pl.Utf8),  # JSON
                "metadata": pl.Series([], dtype=pl.Utf8)  # JSON
            })
    
    def _init_compliance(self):
        """Initialize compliance DataFrame."""
        if self.compliance_file.exists():
            self.compliance_df = pl.read_parquet(self.compliance_file)
        else:
            self.compliance_df = pl.DataFrame({
                "compliance_id": pl.Series([], dtype=pl.Utf8),
                "resource_id": pl.Series([], dtype=pl.Utf8),
                "framework": pl.Series([], dtype=pl.Utf8),  # GDPR, HIPAA, SOX, etc.
                "requirement": pl.Series([], dtype=pl.Utf8),
                "status": pl.Series([], dtype=pl.Utf8),
                "last_checked": pl.Series([], dtype=pl.Datetime),
                "next_check": pl.Series([], dtype=pl.Datetime),
                "evidence": pl.Series([], dtype=pl.Utf8),  # JSON
                "violations": pl.Series([], dtype=pl.Utf8),  # JSON
                "remediation_plan": pl.Series([], dtype=pl.Utf8),  # JSON
                "responsible_party": pl.Series([], dtype=pl.Utf8),
                "metadata": pl.Series([], dtype=pl.Utf8)  # JSON
            })
    
    def _init_classifications(self):
        """Initialize data classifications DataFrame."""
        if self.classifications_file.exists():
            self.classifications_df = pl.read_parquet(self.classifications_file)
        else:
            self.classifications_df = pl.DataFrame({
                "classification_id": pl.Series([], dtype=pl.Utf8),
                "resource_id": pl.Series([], dtype=pl.Utf8),
                "classification": pl.Series([], dtype=pl.Utf8),
                "confidence": pl.Series([], dtype=pl.Float64),
                "classified_by": pl.Series([], dtype=pl.Utf8),
                "classified_at": pl.Series([], dtype=pl.Datetime),
                "classification_method": pl.Series([], dtype=pl.Utf8),  # manual, automatic, inherited
                "sensitivity_labels": pl.Series([], dtype=pl.List(pl.Utf8)),
                "handling_instructions": pl.Series([], dtype=pl.Utf8),  # JSON
                "retention_period": pl.Series([], dtype=pl.Int64),  # days
                "metadata": pl.Series([], dtype=pl.Utf8)  # JSON
            })
    
    def create_policy(self,
                     name: str,
                     policy_type: PolicyType,
                     description: str,
                     rules: Dict[str, Any],
                     created_by: str,
                     enabled: bool = True,
                     tags: Optional[List[str]] = None,
                     priority: int = 100,
                     scope: Optional[Dict[str, Any]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new governance policy.
        
        Args:
            name: Policy name
            policy_type: Type of policy
            description: Policy description
            rules: Policy rules and conditions
            created_by: Policy creator
            enabled: Whether policy is active
            tags: Classification tags
            priority: Policy priority (lower = higher priority)
            scope: Resources this policy applies to
            metadata: Additional metadata
            
        Returns:
            policy_id: Unique identifier for the policy
        """
        # Validate policy rules
        if policy_type in self._policy_validators:
            is_valid, error_msg = self._policy_validators[policy_type](rules)
            if not is_valid:
                raise ValueError(f"Invalid policy rules: {error_msg}")
        
        policy_id = str(uuid.uuid4())
        now = datetime.now()
        
        new_policy = pl.DataFrame({
            "policy_id": [policy_id],
            "name": [name],
            "policy_type": [policy_type],
            "description": [description],
            "rules": [orjson.dumps(rules).decode()],
            "enabled": [enabled],
            "created_by": [created_by],
            "created_at": [now],
            "updated_at": [now],
            "version": [1],
            "tags": [tags or []],
            "priority": [priority],
            "scope": [orjson.dumps(scope or {}).decode()],
            "metadata": [orjson.dumps(metadata or {}).decode()]
        })
        
        self.policies_df = pl.concat([self.policies_df, new_policy])
        
        # Clear cache
        self._policy_cache.clear()
        
        # Audit log
        self._log_audit_event(
            event_type="policy_created",
            user_id=created_by,
            details={"policy_id": policy_id, "policy_type": policy_type, "name": name}
        )
        
        return policy_id
    
    def evaluate_policy(self,
                       policy_id: str,
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a policy against provided context.
        
        Args:
            policy_id: Policy to evaluate
            context: Evaluation context (user, resource, action, etc.)
            
        Returns:
            Evaluation result with decision, reasons, and metadata
        """
        cache_key = f"{policy_id}_{hashlib.md5(orjson.dumps(context, sort_keys=True)).hexdigest()}"
        
        policy = self.policies_df.filter(pl.col("policy_id") == policy_id)
        if policy.height == 0:
            return {"decision": "error", "reason": "Policy not found"}
        
        policy_row = policy.row(0, named=True)
        
        if not policy_row["enabled"]:
            return {"decision": "skip", "reason": "Policy disabled"}
        
        rules = orjson.loads(policy_row["rules"])
        scope = orjson.loads(policy_row["scope"])
        
        # Check if context is within policy scope
        if not self._check_scope(scope, context):
            return {"decision": "skip", "reason": "Outside policy scope"}
        
        # Evaluate policy based on type
        result = self._evaluate_policy_rules(policy_row["policy_type"], rules, context)
        
        # Add metadata
        result.update({
            "policy_id": policy_id,
            "policy_name": policy_row["name"],
            "policy_type": policy_row["policy_type"],
            "evaluated_at": datetime.now().isoformat()
        })
        
        return result
    
    def check_access(self,
                    user_id: str,
                    resource_id: str,
                    access_level: AccessLevel,
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check if user has access to resource at specified level.
        
        Args:
            user_id: User identifier
            resource_id: Resource identifier
            access_level: Required access level
            context: Additional context for decision
            
        Returns:
            Access decision with reasons and conditions
        """
        now = datetime.now()
        access_context = {
            "user_id": user_id,
            "resource_id": resource_id,
            "access_level": access_level,
            "timestamp": now,
            **(context or {})
        }
        
        # Check existing permissions
        permissions = self.permissions_df.filter(
            (pl.col("user_id") == user_id) &
            (pl.col("resource_id") == resource_id) &
            (pl.col("revoked") == False) &
            (pl.col("expires_at") > now)
        )
        
        has_permission = False
        permission_level = "none"
        
        if permissions.height > 0:
            # Get highest permission level
            access_levels = ["none", "read", "write", "delete", "admin"]
            for perm in permissions.iter_rows(named=True):
                perm_level = perm["access_level"]
                if access_levels.index(perm_level) >= access_levels.index(access_level):
                    has_permission = True
                    permission_level = perm_level
                    break
        
        # Evaluate access policies
        access_policies = self.policies_df.filter(
            (pl.col("policy_type") == "access") &
            (pl.col("enabled") == True)
        ).sort("priority")
        
        policy_decisions = []
        final_decision = "denied" if not has_permission else "granted"
        
        for policy in access_policies.iter_rows(named=True):
            policy_result = self.evaluate_policy(policy["policy_id"], access_context)
            policy_decisions.append(policy_result)
            
            if policy_result["decision"] == "deny":
                final_decision = "denied"
                break
            elif policy_result["decision"] == "allow" and not has_permission:
                final_decision = "granted"
                has_permission = True
        
        # Log access attempt
        self._log_audit_event(
            event_type="access_granted" if final_decision == "granted" else "access_denied",
            user_id=user_id,
            resource_id=resource_id,
            details={
                "access_level": access_level,
                "decision": final_decision,
                "has_permission": has_permission,
                "permission_level": permission_level,
                "policy_evaluations": len(policy_decisions)
            }
        )
        
        return {
            "decision": final_decision,
            "has_permission": has_permission,
            "permission_level": permission_level,
            "policy_decisions": policy_decisions,
            "evaluated_at": now.isoformat(),
            "context": access_context
        }
    
    def request_access(self,
                      user_id: str,
                      resource_id: str,
                      access_level: AccessLevel,
                      justification: str,
                      expires_in_days: int = 30,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit access request for approval.
        
        Args:
            user_id: User requesting access
            resource_id: Resource to access
            access_level: Requested access level
            justification: Business justification
            expires_in_days: Request expiration in days
            metadata: Additional metadata
            
        Returns:
            request_id: Unique identifier for the request
        """
        request_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + timedelta(days=expires_in_days)
        
        new_request = pl.DataFrame({
            "request_id": [request_id],
            "user_id": [user_id],
            "resource_id": [resource_id],
            "access_level": [access_level],
            "justification": [justification],
            "status": ["pending"],
            "requested_at": [now],
            "reviewed_by": [None],
            "reviewed_at": [None],
            "expires_at": [expires_at],
            "approval_conditions": [orjson.dumps({}).decode()],
            "metadata": [orjson.dumps(metadata or {}).decode()]
        })
        
        self.access_requests_df = pl.concat([self.access_requests_df, new_request])
        
        return request_id
    
    def classify_resource(self,
                         resource_id: str,
                         classification: DataClassification,
                         classified_by: str,
                         confidence: float = 1.0,
                         classification_method: str = "manual",
                         sensitivity_labels: Optional[List[str]] = None,
                         handling_instructions: Optional[Dict[str, Any]] = None,
                         retention_period_days: Optional[int] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Classify a resource with sensitivity level.
        
        Args:
            resource_id: Resource to classify
            classification: Data classification level
            classified_by: User performing classification
            confidence: Classification confidence (0.0 to 1.0)
            classification_method: How classification was determined
            sensitivity_labels: Additional sensitivity labels
            handling_instructions: Special handling requirements
            retention_period_days: Data retention period
            metadata: Additional metadata
            
        Returns:
            classification_id: Unique identifier for the classification
        """
        classification_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Remove existing classification for this resource
        self.classifications_df = self.classifications_df.filter(
            pl.col("resource_id") != resource_id
        )
        
        new_classification = pl.DataFrame({
            "classification_id": [classification_id],
            "resource_id": [resource_id],
            "classification": [classification],
            "confidence": [max(0.0, min(1.0, confidence))],
            "classified_by": [classified_by],
            "classified_at": [now],
            "classification_method": [classification_method],
            "sensitivity_labels": [sensitivity_labels or []],
            "handling_instructions": [orjson.dumps(handling_instructions or {}).decode()],
            "retention_period": [retention_period_days],
            "metadata": [orjson.dumps(metadata or {}).decode()]
        })
        
        self.classifications_df = pl.concat([self.classifications_df, new_classification])
        
        return classification_id
    
    def check_compliance(self,
                        resource_id: str,
                        framework: str,
                        requirement: str) -> Dict[str, Any]:
        """
        Check compliance status for a resource against specific requirement.
        
        Args:
            resource_id: Resource to check
            framework: Compliance framework (GDPR, HIPAA, etc.)
            requirement: Specific requirement to check
            
        Returns:
            Compliance status and details
        """
        # Get resource classification
        classification = self.classifications_df.filter(
            pl.col("resource_id") == resource_id
        )
        
        classification_level = "unknown"
        if classification.height > 0:
            classification_level = classification.row(0, named=True)["classification"]
        
        # Get compliance policies for this framework
        compliance_policies = self.policies_df.filter(
            (pl.col("policy_type") == "compliance") &
            (pl.col("enabled") == True)
        )
        
        compliance_result = {
            "resource_id": resource_id,
            "framework": framework,
            "requirement": requirement,
            "classification": classification_level,
            "status": "unknown",
            "violations": [],
            "recommendations": [],
            "last_checked": datetime.now()
        }
        
        # Evaluate compliance policies
        context = {
            "resource_id": resource_id,
            "framework": framework,
            "requirement": requirement,
            "classification": classification_level
        }
        
        for policy in compliance_policies.iter_rows(named=True):
            result = self.evaluate_policy(policy["policy_id"], context)
            
            if result["decision"] == "violation":
                compliance_result["status"] = "non_compliant"
                compliance_result["violations"].append(result)
            elif result["decision"] == "compliant":
                compliance_result["status"] = "compliant"
        
        return compliance_result
    
    def _check_scope(self, scope: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if context matches policy scope."""
        if not scope:
            return True
        
        for key, expected_value in scope.items():
            context_value = context.get(key)
            
            if isinstance(expected_value, list):
                if context_value not in expected_value:
                    return False
            elif isinstance(expected_value, dict):
                # Pattern matching
                if "pattern" in expected_value:
                    import re
                    if not re.match(expected_value["pattern"], str(context_value)):
                        return False
            else:
                if context_value != expected_value:
                    return False
        
        return True
    
    def _evaluate_policy_rules(self, policy_type: str, rules: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate policy rules against context."""
        if policy_type == "access":
            return self._evaluate_access_rules(rules, context)
        elif policy_type == "data_quality":
            return self._evaluate_data_quality_rules(rules, context)
        elif policy_type == "retention":
            return self._evaluate_retention_rules(rules, context)
        elif policy_type == "classification":
            return self._evaluate_classification_rules(rules, context)
        elif policy_type == "compliance":
            return self._evaluate_compliance_rules(rules, context)
        else:
            return {"decision": "error", "reason": f"Unknown policy type: {policy_type}"}
    
    def _evaluate_access_rules(self, rules: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate access control rules."""
        # Simple rule evaluation - can be extended
        if "allowed_users" in rules:
            if context.get("user_id") not in rules["allowed_users"]:
                return {"decision": "deny", "reason": "User not in allowed list"}
        
        if "blocked_users" in rules:
            if context.get("user_id") in rules["blocked_users"]:
                return {"decision": "deny", "reason": "User is blocked"}
        
        if "time_restrictions" in rules:
            # Check time-based access restrictions
            now = datetime.now()
            restrictions = rules["time_restrictions"]
            
            if "business_hours_only" in restrictions and restrictions["business_hours_only"]:
                if now.hour < 8 or now.hour > 18 or now.weekday() >= 5:
                    return {"decision": "deny", "reason": "Outside business hours"}
        
        if "classification_requirements" in rules:
            # Check if user has appropriate clearance for data classification
            required_clearance = rules["classification_requirements"]
            user_clearance = context.get("user_clearance", "public")
            
            clearance_levels = ["public", "internal", "confidential", "restricted", "top_secret"]
            if clearance_levels.index(user_clearance) < clearance_levels.index(required_clearance):
                return {"decision": "deny", "reason": "Insufficient clearance level"}
        
        return {"decision": "allow", "reason": "Access granted"}
    
    def _evaluate_data_quality_rules(self, rules: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate data quality rules."""
        # Placeholder for data quality rule evaluation
        return {"decision": "pass", "reason": "Data quality check passed"}
    
    def _evaluate_retention_rules(self, rules: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate data retention rules."""
        # Placeholder for retention rule evaluation
        return {"decision": "retain", "reason": "Within retention period"}
    
    def _evaluate_classification_rules(self, rules: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate classification rules."""
        # Placeholder for classification rule evaluation
        return {"decision": "classified", "reason": "Classification applied"}
    
    def _evaluate_compliance_rules(self, rules: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate compliance rules."""
        # Placeholder for compliance rule evaluation
        framework = context.get("framework", "")
        
        if framework == "GDPR":
            # Example GDPR compliance check
            if "personal_data" in rules and rules["personal_data"]:
                classification = context.get("classification", "")
                if classification not in ["confidential", "restricted"]:
                    return {"decision": "violation", "reason": "Personal data must be classified as confidential or restricted"}
        
        return {"decision": "compliant", "reason": "Compliance requirements met"}
    
    def _validate_data_quality_policy(self, rules: Dict[str, Any]) -> tuple[bool, str]:
        """Validate data quality policy rules."""
        required_fields = ["checks"]
        for field in required_fields:
            if field not in rules:
                return False, f"Missing required field: {field}"
        return True, ""
    
    def _validate_access_policy(self, rules: Dict[str, Any]) -> tuple[bool, str]:
        """Validate access policy rules."""
        # Basic validation
        if not isinstance(rules, dict):
            return False, "Rules must be a dictionary"
        return True, ""
    
    def _validate_retention_policy(self, rules: Dict[str, Any]) -> tuple[bool, str]:
        """Validate retention policy rules."""
        if "retention_period_days" not in rules:
            return False, "Missing retention_period_days"
        if not isinstance(rules["retention_period_days"], int) or rules["retention_period_days"] <= 0:
            return False, "retention_period_days must be a positive integer"
        return True, ""
    
    def _validate_classification_policy(self, rules: Dict[str, Any]) -> tuple[bool, str]:
        """Validate classification policy rules."""
        if "classification_mapping" not in rules:
            return False, "Missing classification_mapping"
        return True, ""
    
    def _validate_lineage_policy(self, rules: Dict[str, Any]) -> tuple[bool, str]:
        """Validate lineage policy rules."""
        if "tracking_enabled" not in rules:
            return False, "Missing tracking_enabled"
        return True, ""
    
    def _validate_compliance_policy(self, rules: Dict[str, Any]) -> tuple[bool, str]:
        """Validate compliance policy rules."""
        if "framework" not in rules:
            return False, "Missing framework"
        return True, ""
    
    def _log_audit_event(self,
                        event_type: AuditEventType,
                        user_id: str,
                        resource_id: Optional[str] = None,
                        details: Optional[Dict[str, Any]] = None,
                        ip_address: Optional[str] = None,
                        user_agent: Optional[str] = None,
                        session_id: Optional[str] = None,
                        outcome: str = "success",
                        risk_score: float = 0.0) -> str:
        """Log an audit event."""
        event_id = str(uuid.uuid4())
        now = datetime.now()
        
        new_event = pl.DataFrame({
            "event_id": [event_id],
            "event_type": [event_type],
            "user_id": [user_id],
            "resource_id": [resource_id],
            "timestamp": [now],
            "details": [orjson.dumps(details or {}).decode()],
            "ip_address": [ip_address],
            "user_agent": [user_agent],
            "session_id": [session_id],
            "outcome": [outcome],
            "risk_score": [risk_score],
            "geolocation": [orjson.dumps({}).decode()],
            "metadata": [orjson.dumps({}).decode()]
        })
        
        self.audit_log_df = pl.concat([self.audit_log_df, new_event])
        return event_id
    
    def save(self):
        """Save all governance data to Parquet files."""
        self.policies_df.write_parquet(self.policies_file, compression=self.compression)
        self.access_requests_df.write_parquet(self.access_requests_file, compression=self.compression)
        self.permissions_df.write_parquet(self.permissions_file, compression=self.compression)
        self.audit_log_df.write_parquet(self.audit_log_file, compression=self.compression)
        self.compliance_df.write_parquet(self.compliance_file, compression=self.compression)
        self.classifications_df.write_parquet(self.classifications_file, compression=self.compression)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get governance statistics."""
        return {
            "policies_count": self.policies_df.height,
            "active_policies_count": self.policies_df.filter(pl.col("enabled") == True).height,
            "access_requests_count": self.access_requests_df.height,
            "pending_requests_count": self.access_requests_df.filter(pl.col("status") == "pending").height,
            "permissions_count": self.permissions_df.height,
            "active_permissions_count": self.permissions_df.filter(pl.col("revoked") == False).height,
            "audit_events_count": self.audit_log_df.height,
            "classified_resources_count": self.classifications_df.height,
            "compliance_records_count": self.compliance_df.height,
            "policy_types": self.policies_df["policy_type"].unique().to_list(),
            "classification_levels": self.classifications_df["classification"].unique().to_list()
        }