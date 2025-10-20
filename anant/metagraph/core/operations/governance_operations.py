"""
Governance Operations Module
===========================

Enterprise governance and compliance operations including:
- Policy management and enforcement
- Access control and security
- Compliance monitoring and reporting
- Audit trail management
- Data quality and governance
"""

import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Literal
from datetime import datetime, timedelta
import uuid
import logging
from collections import defaultdict

from ....exceptions import (
    GovernanceError, ValidationError, handle_exception,
    require_not_none, require_valid_string, require_valid_dict
)

logger = logging.getLogger(__name__)


class GovernanceOperations:
    """
    Handles governance, compliance, and policy operations for the Metagraph.
    
    Provides policy management, access control, compliance monitoring,
    audit trails, and data quality governance with proper error handling.
    """
    
    def __init__(self, storage_path: str, metadata_store, policy_engine):
        """
        Initialize governance operations.
        
        Args:
            storage_path: Path to store governance data
            metadata_store: Reference to metadata storage system
            policy_engine: Reference to policy enforcement engine
        """
        self.storage_path = Path(storage_path)
        self.metadata_store = metadata_store
        self.policy_engine = policy_engine
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Initialize governance data structures
        self.policies = {}
        self.audit_logs = []
        self.access_control_rules = {}
        
    def create_policy(self,
                     policy_name: str,
                     policy_rules: Dict[str, Any],
                     policy_type: str = "data_governance",
                     enforcement_level: str = "strict") -> str:
        """
        Create a new governance policy.
        
        Args:
            policy_name: Name of the policy
            policy_rules: Rules and conditions for the policy
            policy_type: Type of policy (data_governance, access_control, etc.)
            enforcement_level: How strictly to enforce (strict, warn, monitor)
            
        Returns:
            Policy ID
            
        Raises:
            GovernanceError: If policy creation fails
        """
        try:
            # Validate inputs
            policy_name = require_valid_string(policy_name, "policy_name")
            policy_rules = require_valid_dict(policy_rules, "policy_rules")
            
            if enforcement_level not in ["strict", "warn", "monitor"]:
                raise ValidationError(
                    f"Invalid enforcement level: {enforcement_level}",
                    error_code="INVALID_ENFORCEMENT_LEVEL",
                    context={"enforcement_level": enforcement_level}
                )
            
            # Check if policy already exists
            if policy_name in self.policies:
                raise GovernanceError(
                    f"Policy '{policy_name}' already exists",
                    error_code="POLICY_ALREADY_EXISTS",
                    context={"policy_name": policy_name}
                )
            
            # Create policy
            policy_id = str(uuid.uuid4())
            policy_data = {
                "policy_id": policy_id,
                "policy_name": policy_name,
                "policy_type": policy_type,
                "policy_rules": policy_rules,
                "enforcement_level": enforcement_level,
                "created_at": datetime.now().isoformat(),
                "active": True,
                "version": 1
            }
            
            # Store policy
            self.policies[policy_name] = policy_data
            
            # Register with policy engine
            if not self.policy_engine.register_policy(policy_data):
                del self.policies[policy_name]
                raise GovernanceError(
                    f"Failed to register policy with enforcement engine",
                    error_code="POLICY_REGISTRATION_FAILED",
                    context={"policy_name": policy_name, "policy_id": policy_id}
                )
            
            # Log policy creation
            self._create_audit_entry(
                action="create_policy",
                entity_type="policy",
                entity_id=policy_id,
                details={
                    "policy_name": policy_name,
                    "policy_type": policy_type,
                    "enforcement_level": enforcement_level
                }
            )
            
            self.logger.info(
                "Policy created successfully",
                extra={
                    "policy_id": policy_id,
                    "policy_name": policy_name,
                    "policy_type": policy_type,
                    "enforcement_level": enforcement_level
                }
            )
            
            return policy_id
            
        except (GovernanceError, ValidationError):
            raise
        except Exception as e:
            raise handle_exception(f"creating policy '{policy_name}'", e, {
                "policy_name": policy_name,
                "policy_type": policy_type
            })
    
    def check_compliance(self, policy_id: str) -> Dict[str, Any]:
        """
        Check compliance against a specific policy.
        
        Args:
            policy_id: ID of policy to check against
            
        Returns:
            Compliance check results
            
        Raises:
            GovernanceError: If compliance check fails
        """
        try:
            # Find policy
            policy = self._find_policy_by_id(policy_id)
            if not policy:
                raise GovernanceError(
                    f"Policy '{policy_id}' not found",
                    error_code="POLICY_NOT_FOUND",
                    context={"policy_id": policy_id}
                )
            
            # Get all entities for compliance checking
            entities = self.metadata_store.get_all_entities()
            
            compliance_results = {
                "policy_id": policy_id,
                "policy_name": policy["policy_name"],
                "check_timestamp": datetime.now().isoformat(),
                "total_entities_checked": len(entities),
                "compliant_entities": 0,
                "non_compliant_entities": 0,
                "violations": [],
                "compliance_rate": 0.0,
                "risk_level": "low"
            }
            
            # Check each entity against policy
            for entity in entities:
                violations = self._check_entity_compliance(entity, policy)
                if violations:
                    compliance_results["non_compliant_entities"] += 1
                    compliance_results["violations"].extend(violations)
                else:
                    compliance_results["compliant_entities"] += 1
            
            # Calculate compliance rate
            total_entities = compliance_results["total_entities_checked"]
            if total_entities > 0:
                compliance_results["compliance_rate"] = (
                    compliance_results["compliant_entities"] / total_entities
                )
            
            # Determine risk level
            compliance_rate = compliance_results["compliance_rate"]
            if compliance_rate < 0.7:
                compliance_results["risk_level"] = "high"
            elif compliance_rate < 0.9:
                compliance_results["risk_level"] = "medium"
            else:
                compliance_results["risk_level"] = "low"
            
            # Log compliance check
            self._create_audit_entry(
                action="compliance_check",
                entity_type="policy",
                entity_id=policy_id,
                details={
                    "compliance_rate": compliance_rate,
                    "violations_count": len(compliance_results["violations"]),
                    "risk_level": compliance_results["risk_level"]
                }
            )
            
            return compliance_results
            
        except (GovernanceError, ValidationError):
            raise
        except Exception as e:
            raise handle_exception(f"checking compliance for policy '{policy_id}'", e, {
                "policy_id": policy_id
            })
    
    def check_entity_access(self,
                           entity_id: str,
                           user_id: str,
                           action: str,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check if a user has access to perform an action on an entity.
        
        Args:
            entity_id: ID of entity to access
            user_id: ID of user requesting access
            action: Action to perform (read, write, delete, etc.)
            context: Additional context for access decision
            
        Returns:
            Access decision and reasoning
            
        Raises:
            GovernanceError: If access check fails
        """
        try:
            # Validate inputs
            entity_id = require_valid_string(entity_id, "entity_id")
            user_id = require_valid_string(user_id, "user_id")
            action = require_valid_string(action, "action")
            
            # Get entity
            entity = self.metadata_store.get_entity(entity_id)
            if not entity:
                raise GovernanceError(
                    f"Entity '{entity_id}' not found",
                    error_code="ENTITY_NOT_FOUND",
                    context={"entity_id": entity_id}
                )
            
            # Check access rules
            access_result = {
                "entity_id": entity_id,
                "user_id": user_id,
                "action": action,
                "access_granted": False,
                "reason": "default_deny",
                "policies_applied": [],
                "check_timestamp": datetime.now().isoformat()
            }
            
            # Apply access control policies
            applicable_policies = self._get_applicable_access_policies(entity, user_id, action)
            
            for policy in applicable_policies:
                policy_result = self._evaluate_access_policy(entity, user_id, action, policy, context)
                access_result["policies_applied"].append({
                    "policy_name": policy["policy_name"],
                    "result": policy_result["granted"],
                    "reason": policy_result["reason"]
                })
                
                # Grant access if any policy allows it
                if policy_result["granted"]:
                    access_result["access_granted"] = True
                    access_result["reason"] = f"granted_by_policy:{policy['policy_name']}"
                    break
            
            # Log access attempt
            self._create_audit_entry(
                action="access_check",
                entity_type="entity",
                entity_id=entity_id,
                details={
                    "user_id": user_id,
                    "requested_action": action,
                    "access_granted": access_result["access_granted"],
                    "reason": access_result["reason"]
                }
            )
            
            return access_result
            
        except (GovernanceError, ValidationError):
            raise
        except Exception as e:
            raise handle_exception(f"checking access for entity '{entity_id}'", e, {
                "entity_id": entity_id,
                "user_id": user_id,
                "action": action
            })
    
    def audit_trail(self, entity_id: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        Get audit trail for entities or system-wide.
        
        Args:
            entity_id: Optional specific entity to audit
            days: Number of days to include in audit
            
        Returns:
            Audit trail information
            
        Raises:
            GovernanceError: If audit retrieval fails
        """
        try:
            if days <= 0:
                raise ValidationError(
                    "Days must be positive",
                    error_code="INVALID_DAYS",
                    context={"days": days}
                )
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Filter audit logs
            relevant_logs = []
            for log_entry in self.audit_logs:
                entry_time = datetime.fromisoformat(log_entry["timestamp"])
                if start_time <= entry_time <= end_time:
                    if entity_id is None or log_entry.get("entity_id") == entity_id:
                        relevant_logs.append(log_entry)
            
            # Analyze audit data
            audit_summary = {
                "audit_period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "days": days
                },
                "entity_id": entity_id,
                "total_events": len(relevant_logs),
                "events_by_action": defaultdict(int),
                "events_by_user": defaultdict(int),
                "events_by_day": defaultdict(int),
                "security_events": [],
                "compliance_events": [],
                "events": relevant_logs[-100:]  # Last 100 events
            }
            
            # Analyze events
            for event in relevant_logs:
                action = event.get("action", "unknown")
                user_id = event.get("user_id", "system")
                event_date = datetime.fromisoformat(event["timestamp"]).date().isoformat()
                
                audit_summary["events_by_action"][action] += 1
                audit_summary["events_by_user"][user_id] += 1
                audit_summary["events_by_day"][event_date] += 1
                
                # Identify security and compliance events
                if action in ["access_denied", "security_violation", "unauthorized_access"]:
                    audit_summary["security_events"].append(event)
                elif action in ["compliance_check", "policy_violation", "data_quality_issue"]:
                    audit_summary["compliance_events"].append(event)
            
            # Convert defaultdicts to regular dicts
            audit_summary["events_by_action"] = dict(audit_summary["events_by_action"])
            audit_summary["events_by_user"] = dict(audit_summary["events_by_user"])
            audit_summary["events_by_day"] = dict(audit_summary["events_by_day"])
            
            return audit_summary
            
        except ValidationError:
            raise
        except Exception as e:
            raise handle_exception("retrieving audit trail", e, {
                "entity_id": entity_id,
                "days": days
            })
    
    def data_quality_rules(self, entity_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get and apply data quality rules.
        
        Args:
            entity_type: Optional filter for specific entity type
            
        Returns:
            Data quality assessment and rules
            
        Raises:
            GovernanceError: If quality assessment fails
        """
        try:
            entities = self.metadata_store.get_all_entities()
            if entity_type:
                entities = [e for e in entities if e.get("entity_type") == entity_type]
            
            quality_assessment = {
                "assessment_timestamp": datetime.now().isoformat(),
                "entity_type_filter": entity_type,
                "total_entities_assessed": len(entities),
                "quality_rules": self._get_data_quality_rules(),
                "quality_violations": [],
                "overall_quality_score": 0.0,
                "quality_by_rule": {},
                "recommendations": []
            }
            
            # Apply quality rules
            total_violations = 0
            rule_violations = defaultdict(int)
            
            for entity in entities:
                entity_violations = self._assess_entity_quality(entity, quality_assessment["quality_rules"])
                quality_assessment["quality_violations"].extend(entity_violations)
                
                for violation in entity_violations:
                    rule_name = violation.get("rule_name")
                    if rule_name:
                        rule_violations[rule_name] += 1
                
                total_violations += len(entity_violations)
            
            # Calculate quality metrics
            if entities:
                quality_assessment["overall_quality_score"] = max(0.0, 1.0 - (total_violations / (len(entities) * len(quality_assessment["quality_rules"]))))
            
            # Quality by rule
            for rule_name in quality_assessment["quality_rules"]:
                violations = rule_violations.get(rule_name, 0)
                quality_assessment["quality_by_rule"][rule_name] = {
                    "violations": violations,
                    "compliance_rate": 1.0 - (violations / len(entities)) if entities else 1.0
                }
            
            # Generate recommendations
            quality_assessment["recommendations"] = self._generate_quality_recommendations(
                quality_assessment["overall_quality_score"],
                rule_violations
            )
            
            return quality_assessment
            
        except Exception as e:
            raise handle_exception("assessing data quality", e, {
                "entity_type": entity_type
            })
    
    def _find_policy_by_id(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Find policy by ID."""
        for policy in self.policies.values():
            if policy["policy_id"] == policy_id:
                return policy
        return None
    
    def _check_entity_compliance(self, entity: Dict[str, Any], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check entity compliance against a policy."""
        violations = []
        policy_rules = policy.get("policy_rules", {})
        
        # Check required properties
        required_properties = policy_rules.get("required_properties", [])
        entity_properties = entity.get("properties", {})
        
        for required_prop in required_properties:
            if required_prop not in entity_properties:
                violations.append({
                    "entity_id": entity["entity_id"],
                    "violation_type": "missing_required_property",
                    "rule": f"required_property:{required_prop}",
                    "severity": "high",
                    "details": f"Entity missing required property: {required_prop}"
                })
        
        # Check property value constraints
        property_constraints = policy_rules.get("property_constraints", {})
        for prop_name, constraints in property_constraints.items():
            if prop_name in entity_properties:
                prop_value = entity_properties[prop_name]
                
                # Check data type
                expected_type = constraints.get("type")
                if expected_type and not isinstance(prop_value, eval(expected_type)):
                    violations.append({
                        "entity_id": entity["entity_id"],
                        "violation_type": "invalid_property_type",
                        "rule": f"property_type:{prop_name}",
                        "severity": "medium",
                        "details": f"Property {prop_name} should be {expected_type}, got {type(prop_value).__name__}"
                    })
                
                # Check value ranges
                min_value = constraints.get("min_value")
                max_value = constraints.get("max_value")
                if isinstance(prop_value, (int, float)):
                    if min_value is not None and prop_value < min_value:
                        violations.append({
                            "entity_id": entity["entity_id"],
                            "violation_type": "value_below_minimum",
                            "rule": f"min_value:{prop_name}",
                            "severity": "medium",
                            "details": f"Property {prop_name} value {prop_value} is below minimum {min_value}"
                        })
                    if max_value is not None and prop_value > max_value:
                        violations.append({
                            "entity_id": entity["entity_id"],
                            "violation_type": "value_above_maximum",
                            "rule": f"max_value:{prop_name}",
                            "severity": "medium",
                            "details": f"Property {prop_value} value {prop_value} is above maximum {max_value}"
                        })
        
        return violations
    
    def _get_applicable_access_policies(self, entity: Dict[str, Any], user_id: str, action: str) -> List[Dict[str, Any]]:
        """Get access policies applicable to the request."""
        applicable_policies = []
        
        for policy in self.policies.values():
            if policy.get("policy_type") == "access_control" and policy.get("active", True):
                # Check if policy applies to this entity type, user, and action
                policy_rules = policy.get("policy_rules", {})
                entity_types = policy_rules.get("entity_types", [])
                actions = policy_rules.get("actions", [])
                
                if (not entity_types or entity.get("entity_type") in entity_types) and \
                   (not actions or action in actions):
                    applicable_policies.append(policy)
        
        return applicable_policies
    
    def _evaluate_access_policy(self, entity: Dict[str, Any], user_id: str, action: str, 
                               policy: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate access policy for specific request."""
        policy_rules = policy.get("policy_rules", {})
        
        # Default deny
        result = {"granted": False, "reason": "policy_deny"}
        
        # Check user permissions
        user_permissions = policy_rules.get("user_permissions", {})
        if user_id in user_permissions:
            allowed_actions = user_permissions[user_id]
            if action in allowed_actions:
                result = {"granted": True, "reason": "user_permission"}
        
        # Check role-based permissions
        user_roles = self._get_user_roles(user_id)
        role_permissions = policy_rules.get("role_permissions", {})
        for role in user_roles:
            if role in role_permissions:
                allowed_actions = role_permissions[role]
                if action in allowed_actions:
                    result = {"granted": True, "reason": f"role_permission:{role}"}
                    break
        
        # Check entity-specific rules
        entity_rules = policy_rules.get("entity_rules", {})
        entity_type = entity.get("entity_type")
        if entity_type in entity_rules:
            type_rules = entity_rules[entity_type]
            if action in type_rules.get("allowed_actions", []):
                result = {"granted": True, "reason": f"entity_type_permission:{entity_type}"}
        
        return result
    
    def _get_user_roles(self, user_id: str) -> List[str]:
        """Get roles for a user."""
        # Simple implementation - in real system, this would query user management system
        user_roles_map = {
            "admin": ["administrator", "data_steward"],
            "analyst": ["data_analyst", "viewer"],
            "viewer": ["viewer"]
        }
        return user_roles_map.get(user_id, ["viewer"])
    
    def _create_audit_entry(self, action: str, entity_type: str, entity_id: str, 
                           details: Dict[str, Any], user_id: str = "system"):
        """Create an audit log entry."""
        audit_entry = {
            "audit_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "user_id": user_id,
            "details": details,
            "source": "governance_operations"
        }
        
        self.audit_logs.append(audit_entry)
        
        # Keep only recent audit logs (last 10000 entries)
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-10000:]
    
    def _get_data_quality_rules(self) -> List[str]:
        """Get list of data quality rules."""
        return [
            "completeness",
            "accuracy",
            "consistency",
            "validity",
            "uniqueness",
            "timeliness",
            "relevance"
        ]
    
    def _assess_entity_quality(self, entity: Dict[str, Any], quality_rules: List[str]) -> List[Dict[str, Any]]:
        """Assess entity against quality rules."""
        violations = []
        entity_properties = entity.get("properties", {})
        
        for rule in quality_rules:
            if rule == "completeness":
                # Check for missing or empty properties
                required_props = ["name", "description"]  # Example required properties
                for prop in required_props:
                    if prop not in entity_properties or not entity_properties[prop]:
                        violations.append({
                            "entity_id": entity["entity_id"],
                            "rule_name": rule,
                            "violation_type": "missing_required_property",
                            "severity": "high",
                            "details": f"Missing or empty required property: {prop}"
                        })
            
            elif rule == "consistency":
                # Check for data type consistency
                for prop_name, prop_value in entity_properties.items():
                    if prop_name.endswith("_id") and not isinstance(prop_value, str):
                        violations.append({
                            "entity_id": entity["entity_id"],
                            "rule_name": rule,
                            "violation_type": "type_inconsistency",
                            "severity": "medium",
                            "details": f"Property {prop_name} should be string for ID field"
                        })
            
            elif rule == "validity":
                # Check for valid values
                created_at = entity.get("created_at")
                if created_at:
                    try:
                        datetime.fromisoformat(created_at)
                    except ValueError:
                        violations.append({
                            "entity_id": entity["entity_id"],
                            "rule_name": rule,
                            "violation_type": "invalid_timestamp",
                            "severity": "high",
                            "details": f"Invalid timestamp format in created_at: {created_at}"
                        })
        
        return violations
    
    def _generate_quality_recommendations(self, overall_score: float, rule_violations: Dict[str, int]) -> List[str]:
        """Generate recommendations for improving data quality."""
        recommendations = []
        
        if overall_score < 0.7:
            recommendations.append("Implement comprehensive data validation rules")
            recommendations.append("Establish data quality monitoring processes")
        
        if rule_violations.get("completeness", 0) > 0:
            recommendations.append("Review and enforce required property standards")
            recommendations.append("Implement automated completeness checks")
        
        if rule_violations.get("consistency", 0) > 0:
            recommendations.append("Standardize data types and formats across entities")
            recommendations.append("Implement schema validation")
        
        if rule_violations.get("validity", 0) > 0:
            recommendations.append("Implement input validation and sanitization")
            recommendations.append("Regular data format audits")
        
        return recommendations