"""
ANANT Remediation Engine

Automated policy violation remediation and corrective action system
for hypergraph governance and compliance enforcement.
"""

import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .policy_engine import Policy, PolicySeverity, PolicyType
from .audit_system import AuditSystem, AuditEvent, AuditEventType, AuditCategory, AuditLevel

logger = logging.getLogger(__name__)

class RemediationStatus(Enum):
    """Status of remediation actions"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

class RemediationType(Enum):
    """Types of remediation actions"""
    DATA_QUARANTINE = "data_quarantine"
    DATA_ANONYMIZATION = "data_anonymization"
    DATA_DELETION = "data_deletion"
    DATA_ENCRYPTION = "data_encryption"
    ACCESS_REVOCATION = "access_revocation"
    USER_NOTIFICATION = "user_notification"
    WORKFLOW_TRIGGER = "workflow_trigger"
    SYSTEM_ALERT = "system_alert"
    COMPLIANCE_REPORT = "compliance_report"
    CUSTOM_ACTION = "custom_action"

class RemediationPriority(Enum):
    """Priority levels for remediation actions"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class RemediationAction:
    """Definition of a remediation action"""
    id: str
    name: str
    description: str
    remediation_type: RemediationType
    priority: RemediationPriority
    
    # Execution parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300  # 5 minute default timeout
    retry_attempts: int = 3
    
    # Conditions
    applicable_policies: List[str] = field(default_factory=list)
    applicable_severity: List[PolicySeverity] = field(default_factory=list)
    
    # Approval requirements
    requires_approval: bool = False
    auto_approve_threshold: Optional[PolicySeverity] = None
    
    # Execution function
    executor: Optional[Callable] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True
    tags: List[str] = field(default_factory=list)

@dataclass
class RemediationExecution:
    """Record of a remediation action execution"""
    id: str
    action_id: str
    violation_id: str
    policy_id: str
    
    # Execution details
    status: RemediationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Results
    success: bool = False
    error_message: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    triggered_by: str = "system"  # system, user, schedule
    approval_status: str = "not_required"  # not_required, pending, approved, rejected
    approved_by: Optional[str] = None
    
    # Impact tracking
    records_affected: int = 0
    data_modified: bool = False
    
    # Audit trail
    audit_events: List[str] = field(default_factory=list)  # Audit event IDs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'action_id': self.action_id,
            'violation_id': self.violation_id,
            'policy_id': self.policy_id,
            'status': self.status.value,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'success': self.success,
            'error_message': self.error_message,
            'results': self.results,
            'triggered_by': self.triggered_by,
            'approval_status': self.approval_status,
            'approved_by': self.approved_by,
            'records_affected': self.records_affected,
            'data_modified': self.data_modified,
            'audit_events': self.audit_events
        }

class RemediationExecutor(ABC):
    """Abstract base class for remediation executors"""
    
    @abstractmethod
    async def execute(self, 
                     violation_data: Dict[str, Any],
                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute remediation action"""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for this executor"""
        pass

class DataQuarantineExecutor(RemediationExecutor):
    """Executor for quarantining data"""
    
    async def execute(self, violation_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Quarantine violating data by moving it to isolated storage"""
        quarantine_location = parameters.get('quarantine_location', './quarantine')
        retention_days = parameters.get('retention_days', 30)
        
        # Simulate data quarantine process
        affected_records = violation_data.get('violation_count', 0)
        
        # Create quarantine directory
        Path(quarantine_location).mkdir(parents=True, exist_ok=True)
        
        # Generate quarantine metadata
        quarantine_id = f"quarantine_{int(datetime.now().timestamp())}"
        metadata = {
            'quarantine_id': quarantine_id,
            'timestamp': datetime.now().isoformat(),
            'violation_data': violation_data,
            'retention_until': (datetime.now() + timedelta(days=retention_days)).isoformat(),
            'records_quarantined': affected_records
        }
        
        # Save quarantine metadata
        metadata_file = Path(quarantine_location) / f"{quarantine_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'quarantine_id': quarantine_id,
            'location': quarantine_location,
            'records_quarantined': affected_records,
            'retention_until': metadata['retention_until']
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate quarantine parameters"""
        required_params = ['quarantine_location']
        return all(param in parameters for param in required_params)

class DataAnonymizationExecutor(RemediationExecutor):
    """Executor for anonymizing data"""
    
    async def execute(self, violation_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize sensitive data fields"""
        fields_to_anonymize = parameters.get('fields', [])
        anonymization_method = parameters.get('method', 'mask')
        
        affected_records = violation_data.get('violation_count', 0)
        
        # Simulate anonymization process
        if anonymization_method == 'mask':
            result_method = "Masked with asterisks"
        elif anonymization_method == 'hash':
            result_method = "Replaced with SHA-256 hash"
        elif anonymization_method == 'remove':
            result_method = "Field values removed"
        else:
            result_method = "Custom anonymization applied"
        
        return {
            'method_used': result_method,
            'fields_anonymized': fields_to_anonymize,
            'records_affected': affected_records,
            'anonymization_timestamp': datetime.now().isoformat()
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate anonymization parameters"""
        required_params = ['fields', 'method']
        return all(param in parameters for param in required_params)

class AccessRevocationExecutor(RemediationExecutor):
    """Executor for revoking access rights"""
    
    async def execute(self, violation_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Revoke user access to violated resources"""
        user_id = parameters.get('user_id')
        resources = parameters.get('resources', [])
        temporary = parameters.get('temporary', True)
        duration_hours = parameters.get('duration_hours', 24)
        
        # Simulate access revocation
        revocation_id = f"revocation_{int(datetime.now().timestamp())}"
        
        result = {
            'revocation_id': revocation_id,
            'user_affected': user_id,
            'resources_affected': resources,
            'temporary_revocation': temporary,
            'revocation_timestamp': datetime.now().isoformat()
        }
        
        if temporary:
            result['restore_at'] = (datetime.now() + timedelta(hours=duration_hours)).isoformat()
        
        return result
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate access revocation parameters"""
        return 'user_id' in parameters or 'resources' in parameters

class NotificationExecutor(RemediationExecutor):
    """Executor for sending notifications"""
    
    async def execute(self, violation_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification about policy violation"""
        recipients = parameters.get('recipients', [])
        notification_type = parameters.get('type', 'email')
        template = parameters.get('template', 'default')
        
        # Simulate notification sending
        notification_id = f"notification_{int(datetime.now().timestamp())}"
        
        # Generate notification content
        message = f"""
        Policy Violation Alert
        
        Policy: {violation_data.get('policy_name', 'Unknown')}
        Severity: {violation_data.get('severity', 'Unknown')}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Violations: {violation_data.get('violation_count', 0)}
        
        This is an automated notification from the ANANT Governance System.
        """
        
        return {
            'notification_id': notification_id,
            'recipients': recipients,
            'type': notification_type,
            'template': template,
            'message_preview': message[:100] + "...",
            'sent_at': datetime.now().isoformat()
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate notification parameters"""
        return 'recipients' in parameters and len(parameters['recipients']) > 0

class RemediationEngine:
    """Advanced automated remediation engine"""
    
    def __init__(self, audit_system: Optional[AuditSystem] = None):
        self.audit_system = audit_system
        self.actions: Dict[str, RemediationAction] = {}
        self.executions: Dict[str, RemediationExecution] = {}
        self.executors: Dict[RemediationType, RemediationExecutor] = {}
        
        # Configuration
        self.auto_remediation_enabled = True
        self.max_concurrent_remediations = 10
        self.approval_timeout_hours = 24
        
        # Execution management
        self.executor_pool = ThreadPoolExecutor(max_workers=self.max_concurrent_remediations)
        self.pending_approvals: Dict[str, RemediationExecution] = {}
        
        # Statistics
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'auto_approved': 0,
            'manual_approved': 0,
            'execution_times': []
        }
        
        # Register built-in executors
        self._register_built_in_executors()
        
        # Create default remediation actions
        self._create_default_actions()
        
        logger.info("Remediation engine initialized")
    
    def _register_built_in_executors(self):
        """Register built-in remediation executors"""
        self.executors[RemediationType.DATA_QUARANTINE] = DataQuarantineExecutor()
        self.executors[RemediationType.DATA_ANONYMIZATION] = DataAnonymizationExecutor()
        self.executors[RemediationType.ACCESS_REVOCATION] = AccessRevocationExecutor()
        self.executors[RemediationType.USER_NOTIFICATION] = NotificationExecutor()
    
    def _create_default_actions(self):
        """Create default remediation actions"""
        
        # Critical data protection action
        quarantine_action = RemediationAction(
            id="quarantine_pii",
            name="Quarantine PII Data",
            description="Quarantine personally identifiable information that violates privacy policies",
            remediation_type=RemediationType.DATA_QUARANTINE,
            priority=RemediationPriority.CRITICAL,
            parameters={
                'quarantine_location': './quarantine/pii',
                'retention_days': 30
            },
            applicable_severity=[PolicySeverity.CRITICAL, PolicySeverity.HIGH],
            auto_approve_threshold=PolicySeverity.CRITICAL,
            tags=['pii', 'privacy', 'gdpr']
        )
        
        # Data anonymization action
        anonymize_action = RemediationAction(
            id="anonymize_sensitive",
            name="Anonymize Sensitive Data",
            description="Anonymize sensitive data fields that violate privacy policies",
            remediation_type=RemediationType.DATA_ANONYMIZATION,
            priority=RemediationPriority.HIGH,
            parameters={
                'fields': ['email', 'phone', 'ssn'],
                'method': 'hash'
            },
            applicable_severity=[PolicySeverity.HIGH, PolicySeverity.MEDIUM],
            requires_approval=True,
            tags=['anonymization', 'privacy']
        )
        
        # Access revocation action
        revoke_access_action = RemediationAction(
            id="revoke_unauthorized_access",
            name="Revoke Unauthorized Access",
            description="Temporarily revoke access for users with policy violations",
            remediation_type=RemediationType.ACCESS_REVOCATION,
            priority=RemediationPriority.HIGH,
            parameters={
                'temporary': True,
                'duration_hours': 24
            },
            applicable_severity=[PolicySeverity.CRITICAL, PolicySeverity.HIGH],
            auto_approve_threshold=PolicySeverity.HIGH,
            tags=['access_control', 'security']
        )
        
        # Notification action
        notification_action = RemediationAction(
            id="notify_violation",
            name="Send Violation Notification",
            description="Send notification to administrators about policy violations",
            remediation_type=RemediationType.USER_NOTIFICATION,
            priority=RemediationPriority.MEDIUM,
            parameters={
                'recipients': ['admin@example.com'],
                'type': 'email',
                'template': 'violation_alert'
            },
            applicable_severity=[PolicySeverity.CRITICAL, PolicySeverity.HIGH, PolicySeverity.MEDIUM],
            auto_approve_threshold=PolicySeverity.LOW,
            tags=['notification', 'alerting']
        )
        
        # Add all default actions
        for action in [quarantine_action, anonymize_action, revoke_access_action, notification_action]:
            self.add_action(action)
    
    def add_action(self, action: RemediationAction) -> None:
        """Add a remediation action to the engine"""
        self.actions[action.id] = action
        logger.info(f"Added remediation action: {action.name} ({action.id})")
    
    def remove_action(self, action_id: str) -> bool:
        """Remove a remediation action"""
        if action_id in self.actions:
            del self.actions[action_id]
            logger.info(f"Removed remediation action: {action_id}")
            return True
        return False
    
    def register_executor(self, remediation_type: RemediationType, executor: RemediationExecutor) -> None:
        """Register a custom remediation executor"""
        self.executors[remediation_type] = executor
        logger.info(f"Registered executor for {remediation_type.value}")
    
    async def trigger_remediation(self, 
                                violation_data: Dict[str, Any],
                                policy: Policy,
                                context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Trigger remediation actions for a policy violation"""
        context = context or {}
        
        # Find applicable remediation actions
        applicable_actions = self._find_applicable_actions(policy, violation_data)
        
        if not applicable_actions:
            logger.info(f"No applicable remediation actions for policy {policy.id}")
            return []
        
        execution_ids = []
        
        for action in applicable_actions:
            if not action.enabled:
                continue
                
            # Create execution record
            execution = RemediationExecution(
                id=f"exec_{int(datetime.now().timestamp())}_{action.id}",
                action_id=action.id,
                violation_id=violation_data.get('violation_id', ''),
                policy_id=policy.id,
                status=RemediationStatus.PENDING,
                started_at=datetime.now(),
                triggered_by=context.get('triggered_by', 'system')
            )
            
            self.executions[execution.id] = execution
            execution_ids.append(execution.id)
            
            # Check if approval is required
            if self._requires_approval(action, policy.severity):
                execution.approval_status = "pending"
                self.pending_approvals[execution.id] = execution
                logger.info(f"Remediation action {action.id} requires approval for execution {execution.id}")
                
                # Log audit event for approval request
                if self.audit_system:
                    self.audit_system.log_event(AuditEvent(
                        id="",
                        timestamp=datetime.now(),
                        event_type=AuditEventType.SYSTEM_START,
                        category=AuditCategory.COMPLIANCE_EVENT,
                        level=AuditLevel.INFO,
                        source="remediation_engine",
                        action="approval_required",
                        description=f"Remediation action {action.name} requires approval",
                        metadata={'execution_id': execution.id, 'action_id': action.id}
                    ))
            else:
                # Auto-approve and execute
                execution.approval_status = "approved"
                self.stats['auto_approved'] += 1
                await self._execute_remediation(execution, action, violation_data)
        
        return execution_ids
    
    def _find_applicable_actions(self, policy: Policy, violation_data: Dict[str, Any]) -> List[RemediationAction]:
        """Find remediation actions applicable to a policy violation"""
        applicable_actions = []
        
        for action in self.actions.values():
            # Check policy ID match
            if action.applicable_policies and policy.id not in action.applicable_policies:
                continue
            
            # Check severity match
            if action.applicable_severity and policy.severity not in action.applicable_severity:
                continue
            
            # Check if executor exists
            if action.remediation_type not in self.executors:
                logger.warning(f"No executor registered for {action.remediation_type.value}")
                continue
            
            applicable_actions.append(action)
        
        # Sort by priority (critical first)
        priority_order = {
            RemediationPriority.CRITICAL: 0,
            RemediationPriority.HIGH: 1,
            RemediationPriority.MEDIUM: 2,
            RemediationPriority.LOW: 3
        }
        
        applicable_actions.sort(key=lambda a: priority_order.get(a.priority, 99))
        
        return applicable_actions
    
    def _requires_approval(self, action: RemediationAction, severity: PolicySeverity) -> bool:
        """Check if remediation action requires manual approval"""
        if not action.requires_approval:
            return False
        
        if action.auto_approve_threshold and severity.value in [s.value for s in [action.auto_approve_threshold]]:
            return False
        
        return True
    
    async def _execute_remediation(self, 
                                 execution: RemediationExecution,
                                 action: RemediationAction,
                                 violation_data: Dict[str, Any]) -> None:
        """Execute a remediation action"""
        execution.status = RemediationStatus.IN_PROGRESS
        start_time = datetime.now()
        
        try:
            # Get executor for this action type
            executor = self.executors.get(action.remediation_type)
            if not executor:
                raise Exception(f"No executor available for {action.remediation_type.value}")
            
            # Validate parameters
            if not executor.validate_parameters(action.parameters):
                raise Exception("Invalid parameters for remediation action")
            
            # Log audit event for execution start
            if self.audit_system:
                audit_id = self.audit_system.log_event(AuditEvent(
                    id="",
                    timestamp=datetime.now(),
                    event_type=AuditEventType.SYSTEM_START,
                    category=AuditCategory.COMPLIANCE_EVENT,
                    level=AuditLevel.INFO,
                    source="remediation_engine",
                    action="remediation_start",
                    description=f"Starting remediation action: {action.name}",
                    metadata={'execution_id': execution.id, 'action_id': action.id}
                ))
                execution.audit_events.append(audit_id)
            
            # Execute with timeout
            try:
                results = await asyncio.wait_for(
                    executor.execute(violation_data, action.parameters),
                    timeout=action.timeout_seconds
                )
                
                execution.results = results
                execution.success = True
                execution.status = RemediationStatus.COMPLETED
                execution.records_affected = results.get('records_affected', 0)
                execution.data_modified = results.get('data_modified', False)
                
                self.stats['successful_executions'] += 1
                
            except asyncio.TimeoutError:
                raise Exception(f"Remediation action timed out after {action.timeout_seconds} seconds")
        
        except Exception as e:
            execution.success = False
            execution.status = RemediationStatus.FAILED
            execution.error_message = str(e)
            
            self.stats['failed_executions'] += 1
            
            logger.error(f"Remediation execution failed: {execution.id} - {str(e)}")
        
        finally:
            execution.completed_at = datetime.now()
            execution.duration_seconds = (execution.completed_at - start_time).total_seconds()
            
            self.stats['total_executions'] += 1
            self.stats['execution_times'].append(execution.duration_seconds)
            
            # Keep only last 1000 execution times for statistics
            if len(self.stats['execution_times']) > 1000:
                self.stats['execution_times'] = self.stats['execution_times'][-1000:]
            
            # Log audit event for execution completion
            if self.audit_system:
                audit_id = self.audit_system.log_event(AuditEvent(
                    id="",
                    timestamp=datetime.now(),
                    event_type=AuditEventType.SYSTEM_STOP,
                    category=AuditCategory.COMPLIANCE_EVENT,
                    level=AuditLevel.INFO if execution.success else AuditLevel.ERROR,
                    source="remediation_engine",
                    action="remediation_complete",
                    description=f"Remediation action completed: {action.name}",
                    outcome="success" if execution.success else "failure",
                    duration_ms=execution.duration_seconds * 1000,
                    metadata={
                        'execution_id': execution.id,
                        'action_id': action.id,
                        'success': execution.success,
                        'error': execution.error_message
                    }
                ))
                execution.audit_events.append(audit_id)
    
    def approve_remediation(self, execution_id: str, approver: str) -> bool:
        """Manually approve a pending remediation action"""
        if execution_id not in self.pending_approvals:
            logger.warning(f"No pending approval found for execution {execution_id}")
            return False
        
        execution = self.pending_approvals[execution_id]
        action = self.actions.get(execution.action_id)
        
        if not action:
            logger.error(f"Action {execution.action_id} not found for execution {execution_id}")
            return False
        
        # Update approval status
        execution.approval_status = "approved"
        execution.approved_by = approver
        
        # Remove from pending approvals
        del self.pending_approvals[execution_id]
        
        self.stats['manual_approved'] += 1
        
        # Execute the remediation asynchronously
        asyncio.create_task(self._execute_remediation(execution, action, {}))
        
        logger.info(f"Remediation execution {execution_id} approved by {approver}")
        return True
    
    def reject_remediation(self, execution_id: str, approver: str, reason: str = "") -> bool:
        """Reject a pending remediation action"""
        if execution_id not in self.pending_approvals:
            logger.warning(f"No pending approval found for execution {execution_id}")
            return False
        
        execution = self.pending_approvals[execution_id]
        
        # Update status
        execution.approval_status = "rejected"
        execution.approved_by = approver
        execution.status = RemediationStatus.CANCELLED
        execution.error_message = f"Rejected by {approver}: {reason}"
        execution.completed_at = datetime.now()
        
        # Remove from pending approvals
        del self.pending_approvals[execution_id]
        
        logger.info(f"Remediation execution {execution_id} rejected by {approver}")
        return True
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a remediation execution"""
        if execution_id not in self.executions:
            return None
        
        execution = self.executions[execution_id]
        return execution.to_dict()
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests"""
        return [execution.to_dict() for execution in self.pending_approvals.values()]
    
    def get_execution_history(self, 
                            limit: int = 100,
                            status_filter: Optional[RemediationStatus] = None) -> List[Dict[str, Any]]:
        """Get remediation execution history"""
        executions = list(self.executions.values())
        
        if status_filter:
            executions = [e for e in executions if e.status == status_filter]
        
        # Sort by start time, most recent first
        executions.sort(key=lambda e: e.started_at, reverse=True)
        
        return [e.to_dict() for e in executions[:limit]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get remediation engine statistics"""
        stats = self.stats.copy()
        
        # Calculate additional statistics
        if self.stats['execution_times']:
            stats['avg_execution_time'] = sum(self.stats['execution_times']) / len(self.stats['execution_times'])
            stats['max_execution_time'] = max(self.stats['execution_times'])
            stats['min_execution_time'] = min(self.stats['execution_times'])
        else:
            stats['avg_execution_time'] = 0.0
            stats['max_execution_time'] = 0.0
            stats['min_execution_time'] = 0.0
        
        stats['success_rate'] = (
            (self.stats['successful_executions'] / self.stats['total_executions'] * 100)
            if self.stats['total_executions'] > 0 else 0.0
        )
        
        stats['pending_approvals'] = len(self.pending_approvals)
        stats['total_actions'] = len(self.actions)
        stats['enabled_actions'] = len([a for a in self.actions.values() if a.enabled])
        
        return stats
    
    def cleanup_old_executions(self, days: int = 90) -> int:
        """Clean up old execution records"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        old_executions = [
            eid for eid, execution in self.executions.items()
            if execution.started_at < cutoff_date
        ]
        
        for eid in old_executions:
            del self.executions[eid]
        
        logger.info(f"Cleaned up {len(old_executions)} old remediation executions")
        return len(old_executions)