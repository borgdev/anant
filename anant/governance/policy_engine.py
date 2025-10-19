"""
ANANT Policy Engine

Enterprise policy definition, evaluation, and enforcement system
for hypergraph data governance.
"""

import polars as pl
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PolicyType(Enum):
    """Types of governance policies"""
    DATA_PRIVACY = "data_privacy"
    DATA_RETENTION = "data_retention"
    ACCESS_CONTROL = "access_control"
    DATA_QUALITY = "data_quality"
    OPERATIONAL = "operational"
    SECURITY = "security"
    COMPLIANCE = "compliance"

class PolicySeverity(Enum):
    """Policy violation severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class PolicyCondition:
    """Condition that triggers policy evaluation"""
    field: str
    operator: str  # eq, ne, gt, lt, ge, le, in, not_in, contains, regex
    value: Any
    description: str = ""

@dataclass
class PolicyAction:
    """Action to take when policy is violated"""
    action_type: str  # block, warn, log, remediate, notify
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

@dataclass
class Policy:
    """Governance policy definition"""
    id: str
    name: str
    description: str
    policy_type: PolicyType
    severity: PolicySeverity
    conditions: List[PolicyCondition]
    actions: List[PolicyAction]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'policy_type': self.policy_type.value,
            'severity': self.severity.value,
            'conditions': [
                {
                    'field': c.field,
                    'operator': c.operator,
                    'value': c.value,
                    'description': c.description
                } for c in self.conditions
            ],
            'actions': [
                {
                    'action_type': a.action_type,
                    'parameters': a.parameters,
                    'description': a.description
                } for a in self.actions
            ],
            'enabled': self.enabled,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': self.tags,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Policy':
        """Create policy from dictionary"""
        conditions = [
            PolicyCondition(**c) for c in data.get('conditions', [])
        ]
        actions = [
            PolicyAction(**a) for a in data.get('actions', [])
        ]
        
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            policy_type=PolicyType(data['policy_type']),
            severity=PolicySeverity(data['severity']),
            conditions=conditions,
            actions=actions,
            enabled=data.get('enabled', True),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat())),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )

class PolicyEngine:
    """Advanced policy engine for hypergraph governance"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.policies: Dict[str, Policy] = {}
        self.policy_evaluators: Dict[str, Callable] = {}
        self.config_path = config_path
        self.evaluation_stats = {
            'total_evaluations': 0,
            'violations_detected': 0,
            'policies_triggered': {},
            'performance_metrics': {}
        }
        
        # Built-in policy evaluators
        self._register_built_in_evaluators()
        
        # Load policies from config if provided
        if config_path:
            self.load_policies_from_file(config_path)
    
    def _register_built_in_evaluators(self):
        """Register built-in policy evaluation functions"""
        
        def evaluate_condition(data: pl.DataFrame, condition: PolicyCondition) -> pl.Series:
            """Evaluate a single condition against data"""
            field = condition.field
            operator = condition.operator
            value = condition.value
            
            if field not in data.columns:
                return pl.Series([False] * len(data))
            
            col = data[field]
            
            if operator == 'eq':
                return col == value
            elif operator == 'ne':
                return col != value
            elif operator == 'gt':
                return col > value
            elif operator == 'lt':
                return col < value
            elif operator == 'ge':
                return col >= value
            elif operator == 'le':
                return col <= value
            elif operator == 'in':
                return col.is_in(value if isinstance(value, list) else [value])
            elif operator == 'not_in':
                return ~col.is_in(value if isinstance(value, list) else [value])
            elif operator == 'contains':
                return col.str.contains(str(value))
            elif operator == 'regex':
                return col.str.contains(str(value), literal=False)
            else:
                raise ValueError(f"Unsupported operator: {operator}")
        
        self.policy_evaluators['default'] = evaluate_condition
    
    def add_policy(self, policy: Policy) -> None:
        """Add a new policy to the engine"""
        policy.updated_at = datetime.now()
        self.policies[policy.id] = policy
        self.policy_evaluators.setdefault(policy.id, {})
        logger.info(f"Added policy: {policy.name} ({policy.id})")
    
    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy from the engine"""
        if policy_id in self.policies:
            del self.policies[policy_id]
            if policy_id in self.policy_evaluators:
                del self.policy_evaluators[policy_id]
            logger.info(f"Removed policy: {policy_id}")
            return True
        return False
    
    def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing policy"""
        if policy_id not in self.policies:
            return False
        
        policy = self.policies[policy_id]
        for key, value in updates.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        
        policy.updated_at = datetime.now()
        logger.info(f"Updated policy: {policy_id}")
        return True
    
    def enable_policy(self, policy_id: str) -> bool:
        """Enable a policy"""
        return self.update_policy(policy_id, {'enabled': True})
    
    def disable_policy(self, policy_id: str) -> bool:
        """Disable a policy"""
        return self.update_policy(policy_id, {'enabled': False})
    
    def evaluate_policies(self, data: pl.DataFrame, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate all enabled policies against data"""
        start_time = datetime.now()
        violations = []
        policy_results = {}
        
        context = context or {}
        
        for policy_id, policy in self.policies.items():
            if not policy.enabled:
                continue
            
            try:
                policy_start = datetime.now()
                
                # Evaluate all conditions (AND logic)
                violation_mask = None
                for condition in policy.conditions:
                    condition_result = self.policy_evaluators['default'](data, condition)
                    if violation_mask is None:
                        violation_mask = condition_result
                    else:
                        violation_mask = violation_mask & condition_result
                
                if violation_mask is not None and violation_mask.any():
                    violating_indices = data.filter(violation_mask)
                    
                    violation = {
                        'policy_id': policy.id,
                        'policy_name': policy.name,
                        'policy_type': policy.policy_type.value,
                        'severity': policy.severity.value,
                        'violation_count': len(violating_indices),
                        'violating_data': violating_indices.to_dicts(),
                        'timestamp': datetime.now().isoformat(),
                        'context': context
                    }
                    
                    violations.append(violation)
                    
                    # Update stats
                    self.evaluation_stats['violations_detected'] += len(violating_indices)
                    if policy.id not in self.evaluation_stats['policies_triggered']:
                        self.evaluation_stats['policies_triggered'][policy.id] = 0
                    self.evaluation_stats['policies_triggered'][policy.id] += 1
                
                policy_end = datetime.now()
                policy_results[policy_id] = {
                    'evaluated': True,
                    'violations_found': len(violating_indices) if 'violating_indices' in locals() else 0,
                    'evaluation_time_ms': (policy_end - policy_start).total_seconds() * 1000
                }
                
            except Exception as e:
                logger.error(f"Error evaluating policy {policy_id}: {str(e)}")
                policy_results[policy_id] = {
                    'evaluated': False,
                    'error': str(e),
                    'evaluation_time_ms': 0
                }
        
        end_time = datetime.now()
        total_time_ms = (end_time - start_time).total_seconds() * 1000
        
        self.evaluation_stats['total_evaluations'] += 1
        self.evaluation_stats['performance_metrics'][datetime.now().isoformat()] = {
            'total_time_ms': total_time_ms,
            'policies_evaluated': len([p for p in policy_results.values() if p.get('evaluated', False)]),
            'violations_found': len(violations)
        }
        
        return {
            'violations': violations,
            'policy_results': policy_results,
            'evaluation_time_ms': total_time_ms,
            'timestamp': end_time.isoformat()
        }
    
    def get_policies_by_type(self, policy_type: PolicyType) -> List[Policy]:
        """Get all policies of a specific type"""
        return [p for p in self.policies.values() if p.policy_type == policy_type]
    
    def get_policies_by_severity(self, severity: PolicySeverity) -> List[Policy]:
        """Get all policies of a specific severity"""
        return [p for p in self.policies.values() if p.severity == severity]
    
    def search_policies(self, query: str, fields: List[str] = None) -> List[Policy]:
        """Search policies by name, description, or tags"""
        fields = fields or ['name', 'description', 'tags']
        query = query.lower()
        
        results = []
        for policy in self.policies.values():
            match = False
            
            if 'name' in fields and query in policy.name.lower():
                match = True
            if 'description' in fields and query in policy.description.lower():
                match = True
            if 'tags' in fields and any(query in tag.lower() for tag in policy.tags):
                match = True
            
            if match:
                results.append(policy)
        
        return results
    
    def save_policies_to_file(self, file_path: str) -> bool:
        """Save all policies to a JSON file"""
        try:
            policies_data = {
                'policies': [p.to_dict() for p in self.policies.values()],
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_policies': len(self.policies),
                    'engine_stats': self.evaluation_stats
                }
            }
            
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(policies_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.policies)} policies to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving policies: {str(e)}")
            return False
    
    def load_policies_from_file(self, file_path: str) -> bool:
        """Load policies from a JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            loaded_count = 0
            for policy_data in data.get('policies', []):
                policy = Policy.from_dict(policy_data)
                self.add_policy(policy)
                loaded_count += 1
            
            logger.info(f"Loaded {loaded_count} policies from {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading policies: {str(e)}")
            return False
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get policy evaluation statistics"""
        return self.evaluation_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset evaluation statistics"""
        self.evaluation_stats = {
            'total_evaluations': 0,
            'violations_detected': 0,
            'policies_triggered': {},
            'performance_metrics': {}
        }
    
    def create_default_policies(self) -> List[Policy]:
        """Create a set of default governance policies"""
        default_policies = []
        
        # Data Privacy Policy
        privacy_policy = Policy(
            id="privacy_001",
            name="PII Data Protection",
            description="Detect and protect personally identifiable information",
            policy_type=PolicyType.DATA_PRIVACY,
            severity=PolicySeverity.CRITICAL,
            conditions=[
                PolicyCondition(
                    field="data_type",
                    operator="in",
                    value=["email", "phone", "ssn", "credit_card"],
                    description="Contains PII data types"
                )
            ],
            actions=[
                PolicyAction(
                    action_type="block",
                    parameters={"reason": "PII data access not authorized"},
                    description="Block access to PII data"
                )
            ],
            tags=["privacy", "pii", "gdpr"]
        )
        
        # Data Quality Policy
        quality_policy = Policy(
            id="quality_001",
            name="Data Completeness Check",
            description="Ensure data completeness requirements are met",
            policy_type=PolicyType.DATA_QUALITY,
            severity=PolicySeverity.HIGH,
            conditions=[
                PolicyCondition(
                    field="completeness_score",
                    operator="lt",
                    value=0.95,
                    description="Data completeness below 95%"
                )
            ],
            actions=[
                PolicyAction(
                    action_type="warn",
                    parameters={"threshold": 0.95},
                    description="Warn about data quality issues"
                )
            ],
            tags=["quality", "completeness"]
        )
        
        # Data Retention Policy
        retention_policy = Policy(
            id="retention_001",
            name="Data Retention Limit",
            description="Enforce data retention time limits",
            policy_type=PolicyType.DATA_RETENTION,
            severity=PolicySeverity.MEDIUM,
            conditions=[
                PolicyCondition(
                    field="age_days",
                    operator="gt",
                    value=365,
                    description="Data older than 1 year"
                )
            ],
            actions=[
                PolicyAction(
                    action_type="remediate",
                    parameters={"action": "archive"},
                    description="Archive old data"
                )
            ],
            tags=["retention", "archive"]
        )
        
        default_policies.extend([privacy_policy, quality_policy, retention_policy])
        
        # Add all default policies to engine
        for policy in default_policies:
            self.add_policy(policy)
        
        return default_policies