"""
ANANT Audit System

Comprehensive audit logging, analysis, and forensic investigation
system for hypergraph governance and compliance.
"""

import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
import json
import logging
import hashlib
import uuid
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

class AuditLevel(Enum):
    """Audit event severity levels"""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    CRITICAL = "critical"

class AuditCategory(Enum):
    """Categories of audit events"""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    POLICY_CHANGE = "policy_change"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_EVENT = "compliance_event"
    PERFORMANCE_EVENT = "performance_event"

class AuditEventType(Enum):
    """Specific types of audit events"""
    # Data events
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    
    # Policy events
    POLICY_CREATE = "policy_create"
    POLICY_UPDATE = "policy_update"
    POLICY_DELETE = "policy_delete"
    POLICY_VIOLATION = "policy_violation"
    
    # User events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_PERMISSION_CHANGE = "user_permission_change"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    BACKUP_CREATE = "backup_create"
    BACKUP_RESTORE = "backup_restore"
    
    # Security events
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

@dataclass
class AuditEvent:
    """Individual audit event record"""
    id: str
    timestamp: datetime
    event_type: AuditEventType
    category: AuditCategory
    level: AuditLevel
    source: str  # Source system/component
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    action: str = ""
    description: str = ""
    outcome: str = "success"  # success, failure, partial
    
    # Technical details
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    
    # Data context
    data_before: Optional[Dict[str, Any]] = None
    data_after: Optional[Dict[str, Any]] = None
    affected_records: int = 0
    
    # Performance metrics
    duration_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    
    # Security fields
    risk_score: float = 0.0
    threat_indicators: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate ID and checksum after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Calculate event hash for integrity
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for event integrity"""
        event_data = {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'source': self.source,
            'user_id': self.user_id,
            'action': self.action,
            'outcome': self.outcome
        }
        
        event_string = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'category': self.category.value,
            'level': self.level.value,
            'source': self.source,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'resource_id': self.resource_id,
            'resource_type': self.resource_type,
            'action': self.action,
            'description': self.description,
            'outcome': self.outcome,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'request_id': self.request_id,
            'data_before': self.data_before,
            'data_after': self.data_after,
            'affected_records': self.affected_records,
            'duration_ms': self.duration_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'tags': self.tags,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id,
            'risk_score': self.risk_score,
            'threat_indicators': self.threat_indicators,
            'checksum': getattr(self, 'checksum', '')
        }

class AuditQuery:
    """Query builder for audit log searches"""
    
    def __init__(self):
        self.filters = {}
        self.date_range = {}
        self.sort_by = 'timestamp'
        self.sort_desc = True
        self.limit = None
        
    def filter_by_user(self, user_id: str) -> 'AuditQuery':
        self.filters['user_id'] = user_id
        return self
    
    def filter_by_event_type(self, event_type: AuditEventType) -> 'AuditQuery':
        self.filters['event_type'] = event_type.value
        return self
    
    def filter_by_category(self, category: AuditCategory) -> 'AuditQuery':
        self.filters['category'] = category.value
        return self
    
    def filter_by_level(self, level: AuditLevel) -> 'AuditQuery':
        self.filters['level'] = level.value
        return self
    
    def filter_by_source(self, source: str) -> 'AuditQuery':
        self.filters['source'] = source
        return self
    
    def filter_by_outcome(self, outcome: str) -> 'AuditQuery':
        self.filters['outcome'] = outcome
        return self
    
    def date_range_filter(self, start: datetime, end: datetime) -> 'AuditQuery':
        self.date_range = {'start': start, 'end': end}
        return self
    
    def last_hours(self, hours: int) -> 'AuditQuery':
        end = datetime.now()
        start = end - timedelta(hours=hours)
        return self.date_range_filter(start, end)
    
    def last_days(self, days: int) -> 'AuditQuery':
        end = datetime.now()
        start = end - timedelta(days=days)
        return self.date_range_filter(start, end)
    
    def sort(self, field: str, descending: bool = True) -> 'AuditQuery':
        self.sort_by = field
        self.sort_desc = descending
        return self
    
    def limit_results(self, limit: int) -> 'AuditQuery':
        self.limit = limit
        return self

class AuditSystem:
    """Comprehensive audit logging and analysis system"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "./audit_logs"
        self.events: List[AuditEvent] = []
        self.event_handlers: Dict[AuditEventType, List[Callable]] = defaultdict(list)
        
        # Configuration
        self.max_memory_events = 10000  # Max events to keep in memory
        self.auto_flush_interval = timedelta(minutes=5)
        self.retention_days = 365  # Keep audit logs for 1 year
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'events_by_level': defaultdict(int),
            'events_by_user': defaultdict(int),
            'last_flush_time': None,
            'integrity_checks': 0,
            'integrity_failures': 0
        }
        
        # Security monitoring
        self.risk_thresholds = {
            AuditLevel.CRITICAL: 90.0,
            AuditLevel.ERROR: 70.0,
            AuditLevel.WARN: 50.0
        }
        
        # Ensure storage directory exists
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Audit system initialized with storage: {self.storage_path}")
    
    def log_event(self, event: AuditEvent) -> str:
        """Log an audit event"""
        # Add to memory storage
        self.events.append(event)
        
        # Update statistics
        self.stats['total_events'] += 1
        self.stats['events_by_type'][event.event_type.value] += 1
        self.stats['events_by_level'][event.level.value] += 1
        if event.user_id:
            self.stats['events_by_user'][event.user_id] += 1
        
        # Trigger event handlers
        for handler in self.event_handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in audit event handler: {str(e)}")
        
        # Check if we need to flush to disk
        if len(self.events) >= self.max_memory_events:
            self.flush_to_disk()
        
        # Security monitoring
        if event.risk_score > 80.0 or event.level in [AuditLevel.CRITICAL, AuditLevel.ERROR]:
            self._trigger_security_alert(event)
        
        logger.debug(f"Logged audit event: {event.id} ({event.event_type.value})")
        return event.id
    
    def log_data_access(self, 
                       user_id: str, 
                       resource_id: str, 
                       action: str,
                       outcome: str = "success",
                       **kwargs) -> str:
        """Convenience method for logging data access events"""
        event = AuditEvent(
            id="",
            timestamp=datetime.now(),
            event_type=AuditEventType.DATA_READ,
            category=AuditCategory.DATA_ACCESS,
            level=AuditLevel.INFO,
            source="anant_hypergraph",
            user_id=user_id,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            **kwargs
        )
        return self.log_event(event)
    
    def log_policy_violation(self,
                           policy_id: str,
                           user_id: Optional[str] = None,
                           description: str = "",
                           **kwargs) -> str:
        """Log policy violation event"""
        event = AuditEvent(
            id="",
            timestamp=datetime.now(),
            event_type=AuditEventType.POLICY_VIOLATION,
            category=AuditCategory.COMPLIANCE_EVENT,
            level=AuditLevel.WARN,
            source="policy_engine",
            user_id=user_id,
            resource_id=policy_id,
            resource_type="policy",
            action="policy_violation",
            description=description,
            risk_score=75.0,
            **kwargs
        )
        return self.log_event(event)
    
    def log_security_event(self,
                          event_type: AuditEventType,
                          description: str,
                          risk_score: float = 50.0,
                          **kwargs) -> str:
        """Log security-related event"""
        event = AuditEvent(
            id="",
            timestamp=datetime.now(),
            event_type=event_type,
            category=AuditCategory.SECURITY_EVENT,
            level=AuditLevel.WARN if risk_score < 70 else AuditLevel.ERROR,
            source="security_monitor",
            description=description,
            risk_score=risk_score,
            **kwargs
        )
        return self.log_event(event)
    
    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events based on criteria"""
        # Load events from disk if needed
        all_events = self._get_all_events()
        
        # Convert to DataFrame for efficient filtering
        events_data = [event.to_dict() for event in all_events]
        if not events_data:
            return []
        
        df = pl.DataFrame(events_data)
        
        # Apply filters
        for field, value in query.filters.items():
            if field in df.columns:
                df = df.filter(pl.col(field) == value)
        
        # Apply date range filter
        if query.date_range:
            start_str = query.date_range['start'].isoformat()
            end_str = query.date_range['end'].isoformat()
            df = df.filter(
                (pl.col('timestamp') >= start_str) & 
                (pl.col('timestamp') <= end_str)
            )
        
        # Sort results
        df = df.sort(query.sort_by, descending=query.sort_desc)
        
        # Apply limit
        if query.limit:
            df = df.head(query.limit)
        
        # Convert back to AuditEvent objects
        results = []
        for row in df.to_dicts():
            # Reconstruct AuditEvent from dictionary
            event = self._dict_to_audit_event(row)
            results.append(event)
        
        return results
    
    def _dict_to_audit_event(self, data: Dict[str, Any]) -> AuditEvent:
        """Convert dictionary back to AuditEvent"""
        return AuditEvent(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=AuditEventType(data['event_type']),
            category=AuditCategory(data['category']),
            level=AuditLevel(data['level']),
            source=data['source'],
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            resource_id=data.get('resource_id'),
            resource_type=data.get('resource_type'),
            action=data.get('action', ''),
            description=data.get('description', ''),
            outcome=data.get('outcome', 'success'),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent'),
            request_id=data.get('request_id'),
            data_before=data.get('data_before'),
            data_after=data.get('data_after'),
            affected_records=data.get('affected_records', 0),
            duration_ms=data.get('duration_ms', 0.0),
            memory_usage_mb=data.get('memory_usage_mb', 0.0),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
            correlation_id=data.get('correlation_id'),
            risk_score=data.get('risk_score', 0.0),
            threat_indicators=data.get('threat_indicators', [])
        )
    
    def _get_all_events(self) -> List[AuditEvent]:
        """Get all events from memory and disk"""
        all_events = list(self.events)  # Events in memory
        
        # Load events from disk files
        storage_path = Path(self.storage_path)
        for log_file in storage_path.glob("audit_*.json"):
            try:
                with open(log_file, 'r') as f:
                    file_events = json.load(f)
                    for event_data in file_events:
                        event = self._dict_to_audit_event(event_data)
                        all_events.append(event)
            except Exception as e:
                logger.error(f"Error loading audit file {log_file}: {str(e)}")
        
        return all_events
    
    def add_event_handler(self, event_type: AuditEventType, handler: Callable[[AuditEvent], None]) -> None:
        """Add event handler for specific event type"""
        self.event_handlers[event_type].append(handler)
        logger.info(f"Added event handler for {event_type.value}")
    
    def remove_event_handler(self, event_type: AuditEventType, handler: Callable) -> bool:
        """Remove event handler"""
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
            return True
        return False
    
    def flush_to_disk(self) -> None:
        """Flush in-memory events to disk"""
        if not self.events:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_{timestamp}.json"
        filepath = Path(self.storage_path) / filename
        
        try:
            events_data = [event.to_dict() for event in self.events]
            with open(filepath, 'w') as f:
                json.dump(events_data, f, indent=2, default=str)
            
            logger.info(f"Flushed {len(self.events)} audit events to {filepath}")
            
            # Clear memory events
            self.events.clear()
            self.stats['last_flush_time'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error flushing audit events to disk: {str(e)}")
    
    def _trigger_security_alert(self, event: AuditEvent) -> None:
        """Trigger security alert for high-risk events"""
        alert_message = (
            f"Security Alert: High-risk audit event detected\n"
            f"Event ID: {event.id}\n"
            f"Type: {event.event_type.value}\n"
            f"Risk Score: {event.risk_score}\n"
            f"User: {event.user_id or 'Unknown'}\n"
            f"Description: {event.description}\n"
            f"Timestamp: {event.timestamp.isoformat()}"
        )
        
        logger.warning(alert_message)
        
        # Here you could add integrations with alerting systems
        # like email, Slack, PagerDuty, etc.
    
    def generate_audit_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive audit report for time period"""
        query = AuditQuery().date_range_filter(start_date, end_date)
        events = self.query_events(query)
        
        if not events:
            return {
                'period': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
                'summary': {'total_events': 0},
                'message': 'No events found in specified period'
            }
        
        # Event statistics
        events_by_type = defaultdict(int)
        events_by_level = defaultdict(int)
        events_by_user = defaultdict(int)
        events_by_outcome = defaultdict(int)
        
        total_duration = 0.0
        high_risk_events = []
        
        for event in events:
            events_by_type[event.event_type.value] += 1
            events_by_level[event.level.value] += 1
            events_by_outcome[event.outcome] += 1
            
            if event.user_id:
                events_by_user[event.user_id] += 1
            
            total_duration += event.duration_ms
            
            if event.risk_score > 70.0:
                high_risk_events.append({
                    'id': event.id,
                    'type': event.event_type.value,
                    'user': event.user_id,
                    'risk_score': event.risk_score,
                    'description': event.description,
                    'timestamp': event.timestamp.isoformat()
                })
        
        # Top users by activity
        top_users = sorted(events_by_user.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Most common event types
        top_event_types = sorted(events_by_type.items(), key=lambda x: x[1], reverse=True)[:10]
        
        report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_events': len(events),
                'unique_users': len(events_by_user),
                'avg_events_per_day': len(events) / max((end_date - start_date).days, 1),
                'total_duration_ms': total_duration,
                'avg_duration_ms': total_duration / len(events) if events else 0
            },
            'breakdown': {
                'by_type': dict(events_by_type),
                'by_level': dict(events_by_level),
                'by_outcome': dict(events_by_outcome)
            },
            'top_users': top_users,
            'top_event_types': top_event_types,
            'high_risk_events': high_risk_events,
            'security_summary': {
                'high_risk_count': len(high_risk_events),
                'failed_events': events_by_outcome.get('failure', 0),
                'security_events': events_by_type.get('security_event', 0)
            }
        }
        
        return report
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Verify integrity of audit logs"""
        all_events = self._get_all_events()
        
        total_events = len(all_events)
        integrity_failures = 0
        corrupted_events = []
        
        for event in all_events:
            expected_checksum = event._calculate_checksum()
            if hasattr(event, 'checksum') and event.checksum != expected_checksum:
                integrity_failures += 1
                corrupted_events.append(event.id)
        
        self.stats['integrity_checks'] += 1
        self.stats['integrity_failures'] += integrity_failures
        
        result = {
            'total_events_checked': total_events,
            'integrity_failures': integrity_failures,
            'corrupted_events': corrupted_events,
            'integrity_score': ((total_events - integrity_failures) / total_events * 100) if total_events > 0 else 100,
            'check_timestamp': datetime.now().isoformat()
        }
        
        if integrity_failures > 0:
            logger.warning(f"Audit integrity check failed: {integrity_failures} corrupted events found")
        else:
            logger.info(f"Audit integrity check passed: {total_events} events verified")
        
        return result
    
    def cleanup_old_logs(self) -> Dict[str, Any]:
        """Clean up audit logs older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        storage_path = Path(self.storage_path)
        
        deleted_files = []
        total_size_freed = 0
        
        for log_file in storage_path.glob("audit_*.json"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    file_size = log_file.stat().st_size
                    log_file.unlink()
                    deleted_files.append(str(log_file))
                    total_size_freed += file_size
                except Exception as e:
                    logger.error(f"Error deleting old log file {log_file}: {str(e)}")
        
        result = {
            'cutoff_date': cutoff_date.isoformat(),
            'deleted_files': len(deleted_files),
            'size_freed_bytes': total_size_freed,
            'size_freed_mb': total_size_freed / (1024 * 1024),
            'cleanup_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Audit log cleanup: deleted {len(deleted_files)} files, freed {result['size_freed_mb']:.2f} MB")
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit system statistics"""
        stats = self.stats.copy()
        stats['memory_events'] = len(self.events)
        stats['storage_path'] = str(self.storage_path)
        stats['retention_days'] = self.retention_days
        
        # Convert defaultdicts to regular dicts
        stats['events_by_type'] = dict(stats['events_by_type'])
        stats['events_by_level'] = dict(stats['events_by_level'])
        stats['events_by_user'] = dict(stats['events_by_user'])
        
        return stats