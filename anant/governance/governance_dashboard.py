"""
ANANT Governance Dashboard

Comprehensive dashboard for monitoring, managing, and visualizing
hypergraph governance, compliance, and data quality metrics.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import logging
from dataclasses import dataclass, field

from .policy_engine import PolicyEngine, Policy, PolicyType, PolicySeverity
from .compliance_monitor import ComplianceMonitor, ComplianceReport, ComplianceFramework
from .audit_system import AuditSystem, AuditQuery, AuditLevel
from .remediation_engine import RemediationEngine, RemediationStatus
from .access_control import AccessController
from .data_quality import DataQualityManager, QualityDimension

logger = logging.getLogger(__name__)

@dataclass
class DashboardMetric:
    """Individual dashboard metric"""
    name: str
    value: Any
    unit: str = ""
    description: str = ""
    trend: str = "stable"  # up, down, stable
    alert_level: str = "none"  # none, info, warning, critical
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    id: str
    title: str
    widget_type: str  # metric, chart, table, alert
    data_source: str
    refresh_interval: int = 300  # seconds
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)  # x, y, width, height

class GovernanceDashboard:
    """Comprehensive governance monitoring dashboard"""
    
    def __init__(self,
                 policy_engine: Optional[PolicyEngine] = None,
                 compliance_monitor: Optional[ComplianceMonitor] = None,
                 audit_system: Optional[AuditSystem] = None,
                 remediation_engine: Optional[RemediationEngine] = None,
                 access_controller: Optional[AccessController] = None,
                 data_quality_manager: Optional[DataQualityManager] = None):
        
        self.policy_engine = policy_engine
        self.compliance_monitor = compliance_monitor
        self.audit_system = audit_system
        self.remediation_engine = remediation_engine
        self.access_controller = access_controller
        self.data_quality_manager = data_quality_manager
        
        # Dashboard configuration
        self.widgets: Dict[str, DashboardWidget] = {}
        self.metrics_cache: Dict[str, DashboardMetric] = {}
        self.cache_ttl = timedelta(minutes=5)
        self.last_refresh = datetime.now()
        
        # Alert thresholds
        self.alert_thresholds = {
            'policy_violations_per_hour': 10,
            'compliance_score_minimum': 80.0,
            'data_quality_score_minimum': 85.0,
            'failed_remediations_percentage': 20.0,
            'unauthorized_access_attempts': 5
        }
        
        # Initialize default dashboard layout
        self._create_default_widgets()
        
        logger.info("Governance Dashboard initialized")
    
    def _create_default_widgets(self):
        """Create default dashboard widgets"""
        
        # Policy Overview Widget
        policy_widget = DashboardWidget(
            id="policy_overview",
            title="Policy Overview",
            widget_type="metric",
            data_source="policy_engine",
            config={
                'metrics': ['total_policies', 'enabled_policies', 'violations_today'],
                'display_type': 'card_grid'
            },
            position={'x': 0, 'y': 0, 'width': 4, 'height': 2}
        )
        
        # Compliance Status Widget
        compliance_widget = DashboardWidget(
            id="compliance_status",
            title="Compliance Status",
            widget_type="chart",
            data_source="compliance_monitor",
            config={
                'chart_type': 'gauge',
                'metric': 'overall_compliance_score',
                'thresholds': {'red': 70, 'yellow': 85, 'green': 95}
            },
            position={'x': 4, 'y': 0, 'width': 4, 'height': 2}
        )
        
        # Data Quality Widget
        quality_widget = DashboardWidget(
            id="data_quality",
            title="Data Quality Score",
            widget_type="chart",
            data_source="data_quality_manager",
            config={
                'chart_type': 'bar',
                'metric': 'dimension_scores',
                'show_trends': True
            },
            position={'x': 8, 'y': 0, 'width': 4, 'height': 2}
        )
        
        # Recent Alerts Widget
        alerts_widget = DashboardWidget(
            id="recent_alerts",
            title="Recent Alerts",
            widget_type="table",
            data_source="audit_system",
            config={
                'query_params': {'level': 'WARN,ERROR,CRITICAL', 'limit': 10},
                'columns': ['timestamp', 'event_type', 'description', 'severity']
            },
            position={'x': 0, 'y': 2, 'width': 6, 'height': 3}
        )
        
        # Remediation Status Widget
        remediation_widget = DashboardWidget(
            id="remediation_status",
            title="Remediation Actions",
            widget_type="metric",
            data_source="remediation_engine",
            config={
                'metrics': ['pending_actions', 'successful_actions', 'failed_actions'],
                'time_range': 'last_24h'
            },
            position={'x': 6, 'y': 2, 'width': 6, 'height': 3}
        )
        
        # Access Control Widget
        access_widget = DashboardWidget(
            id="access_control",
            title="Access Control",
            widget_type="metric",
            data_source="access_controller",
            config={
                'metrics': ['active_sessions', 'access_success_rate', 'failed_logins'],
                'display_type': 'status_indicators'
            },
            position={'x': 0, 'y': 5, 'width': 12, 'height': 2}
        )
        
        widgets = [policy_widget, compliance_widget, quality_widget, 
                  alerts_widget, remediation_widget, access_widget]
        
        for widget in widgets:
            self.widgets[widget.id] = widget
    
    def get_dashboard_data(self, refresh_cache: bool = False) -> Dict[str, Any]:
        """Get complete dashboard data"""
        
        # Check if cache refresh is needed
        if refresh_cache or (datetime.now() - self.last_refresh) > self.cache_ttl:
            self._refresh_metrics_cache()
        
        dashboard_data = {
            'last_updated': self.last_refresh.isoformat(),
            'metrics': {metric_id: metric.__dict__ for metric_id, metric in self.metrics_cache.items()},
            'widgets': {widget_id: widget.__dict__ for widget_id, widget in self.widgets.items()},
            'alerts': self._get_active_alerts(),
            'summary': self._generate_summary()
        }
        
        return dashboard_data
    
    def _refresh_metrics_cache(self):
        """Refresh all dashboard metrics"""
        self.metrics_cache.clear()
        
        # Policy Engine Metrics
        if self.policy_engine:
            policy_stats = self.policy_engine.get_evaluation_stats()
            
            self.metrics_cache['total_policies'] = DashboardMetric(
                name="Total Policies",
                value=len(self.policy_engine.policies),
                description="Total number of governance policies"
            )
            
            enabled_policies = len([p for p in self.policy_engine.policies.values() if p.enabled])
            self.metrics_cache['enabled_policies'] = DashboardMetric(
                name="Enabled Policies",
                value=enabled_policies,
                description="Number of active governance policies"
            )
            
            self.metrics_cache['policy_evaluations'] = DashboardMetric(
                name="Policy Evaluations",
                value=policy_stats.get('total_evaluations', 0),
                description="Total policy evaluations performed"
            )
            
            self.metrics_cache['policy_violations'] = DashboardMetric(
                name="Policy Violations",
                value=policy_stats.get('violations_detected', 0),
                description="Total policy violations detected"
            )
        
        # Compliance Monitor Metrics
        if self.compliance_monitor:
            compliance_stats = self.compliance_monitor.get_monitoring_stats()
            
            self.metrics_cache['compliance_scans'] = DashboardMetric(
                name="Compliance Scans",
                value=compliance_stats.get('total_scans', 0),
                description="Total compliance scans performed"
            )
            
            self.metrics_cache['compliance_violations'] = DashboardMetric(
                name="Compliance Violations",
                value=compliance_stats.get('violations_detected', 0),
                description="Total compliance violations detected"
            )
            
            # Calculate overall compliance score from recent reports
            recent_reports = list(self.compliance_monitor.reports.values())[-5:]  # Last 5 reports
            if recent_reports:
                avg_score = sum(r.compliance_score for r in recent_reports) / len(recent_reports)
                alert_level = "none"
                if avg_score < self.alert_thresholds['compliance_score_minimum']:
                    alert_level = "critical" if avg_score < 70 else "warning"
                
                self.metrics_cache['overall_compliance_score'] = DashboardMetric(
                    name="Compliance Score",
                    value=round(avg_score, 1),
                    unit="%",
                    description="Average compliance score from recent assessments",
                    alert_level=alert_level
                )
        
        # Audit System Metrics
        if self.audit_system:
            audit_stats = self.audit_system.get_statistics()
            
            self.metrics_cache['total_audit_events'] = DashboardMetric(
                name="Audit Events",
                value=audit_stats.get('total_events', 0),
                description="Total audit events logged"
            )
            
            self.metrics_cache['memory_audit_events'] = DashboardMetric(
                name="Recent Events",
                value=audit_stats.get('memory_events', 0),
                description="Recent audit events in memory"
            )
            
            # Get recent critical/error events
            query = AuditQuery().last_hours(24).filter_by_level(AuditLevel.ERROR)
            recent_errors = self.audit_system.query_events(query)
            
            alert_level = "none"
            if len(recent_errors) > 10:
                alert_level = "critical" if len(recent_errors) > 20 else "warning"
            
            self.metrics_cache['recent_errors'] = DashboardMetric(
                name="Recent Errors",
                value=len(recent_errors),
                description="Error-level events in last 24 hours",
                alert_level=alert_level
            )
        
        # Remediation Engine Metrics
        if self.remediation_engine:
            remediation_stats = self.remediation_engine.get_statistics()
            
            self.metrics_cache['total_remediations'] = DashboardMetric(
                name="Total Remediations",
                value=remediation_stats.get('total_executions', 0),
                description="Total remediation actions executed"
            )
            
            self.metrics_cache['successful_remediations'] = DashboardMetric(
                name="Successful Remediations",
                value=remediation_stats.get('successful_executions', 0),
                description="Successfully completed remediation actions"
            )
            
            success_rate = remediation_stats.get('success_rate', 0.0)
            alert_level = "none"
            if success_rate < 80.0:
                alert_level = "critical" if success_rate < 60.0 else "warning"
            
            self.metrics_cache['remediation_success_rate'] = DashboardMetric(
                name="Remediation Success Rate",
                value=round(success_rate, 1),
                unit="%",
                description="Percentage of successful remediation actions",
                alert_level=alert_level
            )
            
            pending_approvals = len(self.remediation_engine.get_pending_approvals())
            self.metrics_cache['pending_approvals'] = DashboardMetric(
                name="Pending Approvals",
                value=pending_approvals,
                description="Remediation actions awaiting approval",
                alert_level="warning" if pending_approvals > 5 else "none"
            )
        
        # Access Control Metrics
        if self.access_controller:
            access_stats = self.access_controller.get_statistics()
            
            self.metrics_cache['total_users'] = DashboardMetric(
                name="Total Users",
                value=access_stats.get('total_users', 0),
                description="Total registered users"
            )
            
            self.metrics_cache['active_sessions'] = DashboardMetric(
                name="Active Sessions",
                value=access_stats.get('active_sessions', 0),
                description="Currently active user sessions"
            )
            
            success_rate = access_stats.get('access_success_rate', 0.0)
            self.metrics_cache['access_success_rate'] = DashboardMetric(
                name="Access Success Rate",
                value=round(success_rate, 1),
                unit="%",
                description="Percentage of successful access requests"
            )
            
            failed_logins = access_stats.get('failed_logins', 0)
            alert_level = "none"
            if failed_logins > self.alert_thresholds['unauthorized_access_attempts']:
                alert_level = "warning"
            
            self.metrics_cache['failed_logins'] = DashboardMetric(
                name="Failed Logins",
                value=failed_logins,
                description="Total failed login attempts",
                alert_level=alert_level
            )
        
        # Data Quality Metrics
        if self.data_quality_manager:
            quality_stats = self.data_quality_manager.get_statistics()
            
            self.metrics_cache['quality_assessments'] = DashboardMetric(
                name="Quality Assessments",
                value=quality_stats.get('total_assessments', 0),
                description="Total data quality assessments performed"
            )
            
            avg_quality_score = quality_stats.get('avg_quality_score', 0.0) * 100
            alert_level = "none"
            if avg_quality_score < self.alert_thresholds['data_quality_score_minimum']:
                alert_level = "critical" if avg_quality_score < 70 else "warning"
            
            self.metrics_cache['avg_quality_score'] = DashboardMetric(
                name="Average Quality Score",
                value=round(avg_quality_score, 1),
                unit="%",
                description="Average data quality score across assessments",
                alert_level=alert_level
            )
            
            rule_success_rate = quality_stats.get('rule_success_rate', 0.0)
            self.metrics_cache['quality_rule_success'] = DashboardMetric(
                name="Quality Rule Success",
                value=round(rule_success_rate, 1),
                unit="%",
                description="Percentage of quality rules that passed"
            )
        
        self.last_refresh = datetime.now()
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts based on current metrics"""
        alerts = []
        
        for metric_id, metric in self.metrics_cache.items():
            if metric.alert_level != "none":
                alerts.append({
                    'id': f"alert_{metric_id}_{int(datetime.now().timestamp())}",
                    'metric_id': metric_id,
                    'metric_name': metric.name,
                    'level': metric.alert_level,
                    'message': f"{metric.name}: {metric.value}{metric.unit}",
                    'description': metric.description,
                    'timestamp': metric.timestamp.isoformat()
                })
        
        return sorted(alerts, key=lambda a: a['level'], reverse=True)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate dashboard summary"""
        alerts = self._get_active_alerts()
        critical_alerts = [a for a in alerts if a['level'] == 'critical']
        warning_alerts = [a for a in alerts if a['level'] == 'warning']
        
        # Calculate overall health score
        health_factors = []
        
        if 'overall_compliance_score' in self.metrics_cache:
            compliance_score = self.metrics_cache['overall_compliance_score'].value
            health_factors.append(compliance_score / 100.0)
        
        if 'avg_quality_score' in self.metrics_cache:
            quality_score = self.metrics_cache['avg_quality_score'].value
            health_factors.append(quality_score / 100.0)
        
        if 'remediation_success_rate' in self.metrics_cache:
            remediation_rate = self.metrics_cache['remediation_success_rate'].value
            health_factors.append(remediation_rate / 100.0)
        
        if 'access_success_rate' in self.metrics_cache:
            access_rate = self.metrics_cache['access_success_rate'].value
            health_factors.append(access_rate / 100.0)
        
        overall_health = sum(health_factors) / len(health_factors) * 100 if health_factors else 0
        
        # Determine health status
        if overall_health >= 90:
            health_status = "excellent"
        elif overall_health >= 80:
            health_status = "good"
        elif overall_health >= 70:
            health_status = "fair"
        else:
            health_status = "poor"
        
        return {
            'overall_health_score': round(overall_health, 1),
            'health_status': health_status,
            'total_alerts': len(alerts),
            'critical_alerts': len(critical_alerts),
            'warning_alerts': len(warning_alerts),
            'components_monitored': len([x for x in [
                self.policy_engine, self.compliance_monitor, self.audit_system,
                self.remediation_engine, self.access_controller, self.data_quality_manager
            ] if x is not None]),
            'last_updated': self.last_refresh.isoformat()
        }
    
    def get_widget_data(self, widget_id: str) -> Dict[str, Any]:
        """Get data for a specific widget"""
        if widget_id not in self.widgets:
            return {'error': f'Widget {widget_id} not found'}
        
        widget = self.widgets[widget_id]
        
        try:
            if widget.data_source == "policy_engine" and self.policy_engine:
                return self._get_policy_widget_data(widget)
            elif widget.data_source == "compliance_monitor" and self.compliance_monitor:
                return self._get_compliance_widget_data(widget)
            elif widget.data_source == "audit_system" and self.audit_system:
                return self._get_audit_widget_data(widget)
            elif widget.data_source == "remediation_engine" and self.remediation_engine:
                return self._get_remediation_widget_data(widget)
            elif widget.data_source == "access_controller" and self.access_controller:
                return self._get_access_widget_data(widget)
            elif widget.data_source == "data_quality_manager" and self.data_quality_manager:
                return self._get_quality_widget_data(widget)
            else:
                return {'error': f'Data source {widget.data_source} not available'}
        
        except Exception as e:
            logger.error(f"Error getting widget data for {widget_id}: {str(e)}")
            return {'error': str(e)}
    
    def _get_policy_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get policy engine widget data"""
        if not self.policy_engine:
            return {'error': 'Policy engine not available'}
            
        stats = self.policy_engine.get_evaluation_stats()
        
        return {
            'widget_id': widget.id,
            'title': widget.title,
            'data': {
                'total_policies': len(self.policy_engine.policies),
                'enabled_policies': len([p for p in self.policy_engine.policies.values() if p.enabled]),
                'total_evaluations': stats.get('total_evaluations', 0),
                'violations_detected': stats.get('violations_detected', 0),
                'policies_by_type': {
                    ptype.value: len([p for p in self.policy_engine.policies.values() if p.policy_type == ptype])
                    for ptype in PolicyType
                }
            }
        }
    
    def _get_compliance_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get compliance monitor widget data"""
        if not self.compliance_monitor:
            return {'error': 'Compliance monitor not available'}
            
        stats = self.compliance_monitor.get_monitoring_stats()
        
        # Get recent compliance scores
        recent_reports = list(self.compliance_monitor.reports.values())[-10:]
        scores = [r.compliance_score for r in recent_reports] if recent_reports else []
        
        return {
            'widget_id': widget.id,
            'title': widget.title,
            'data': {
                'monitoring_active': stats.get('monitoring_active', False),
                'total_scans': stats.get('total_scans', 0),
                'violations_detected': stats.get('violations_detected', 0),
                'reports_generated': stats.get('reports_generated', 0),
                'recent_scores': scores,
                'average_score': sum(scores) / len(scores) if scores else 0,
                'frameworks_monitored': stats.get('frameworks_monitored', [])
            }
        }
    
    def _get_audit_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get audit system widget data"""
        if not self.audit_system:
            return {'error': 'Audit system not available'}
            
        stats = self.audit_system.get_statistics()
        
        # Get recent events based on widget config
        config = widget.config
        query_params = config.get('query_params', {})
        
        query = AuditQuery().last_hours(24)
        if 'level' in query_params:
            levels = query_params['level'].split(',')
            # Note: This is simplified - would need proper level filtering in real implementation
        
        limit = query_params.get('limit', 10)
        query = query.limit_results(limit)
        
        recent_events = self.audit_system.query_events(query)
        
        return {
            'widget_id': widget.id,
            'title': widget.title,
            'data': {
                'total_events': stats.get('total_events', 0),
                'memory_events': stats.get('memory_events', 0),
                'recent_events': [
                    {
                        'timestamp': event.timestamp.isoformat(),
                        'event_type': event.event_type.value,
                        'description': event.description,
                        'level': event.level.value,
                        'user_id': event.user_id
                    } for event in recent_events
                ],
                'events_by_type': stats.get('events_by_type', {}),
                'events_by_level': stats.get('events_by_level', {})
            }
        }
    
    def _get_remediation_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get remediation engine widget data"""
        if not self.remediation_engine:
            return {'error': 'Remediation engine not available'}
            
        stats = self.remediation_engine.get_statistics()
        
        # Get recent executions
        recent_executions = self.remediation_engine.get_execution_history(limit=20)
        
        return {
            'widget_id': widget.id,
            'title': widget.title,
            'data': {
                'total_executions': stats.get('total_executions', 0),
                'successful_executions': stats.get('successful_executions', 0),
                'failed_executions': stats.get('failed_executions', 0),
                'success_rate': stats.get('success_rate', 0.0),
                'pending_approvals': stats.get('pending_approvals', 0),
                'total_actions': stats.get('total_actions', 0),
                'recent_executions': recent_executions[:10]  # Last 10
            }
        }
    
    def _get_access_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get access controller widget data"""
        if not self.access_controller:
            return {'error': 'Access controller not available'}
            
        stats = self.access_controller.get_statistics()
        
        return {
            'widget_id': widget.id,
            'title': widget.title,
            'data': {
                'total_users': stats.get('total_users', 0),
                'active_users': stats.get('active_users', 0),
                'active_sessions': stats.get('active_sessions', 0),
                'total_access_requests': stats.get('total_access_requests', 0),
                'access_granted': stats.get('access_granted', 0),
                'access_denied': stats.get('access_denied', 0),
                'access_success_rate': stats.get('access_success_rate', 0.0),
                'failed_logins': stats.get('failed_logins', 0),
                'cache_hit_rate': stats.get('cache_hit_rate', 0.0)
            }
        }
    
    def _get_quality_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data quality manager widget data"""
        if not self.data_quality_manager:
            return {'error': 'Data quality manager not available'}
            
        stats = self.data_quality_manager.get_statistics()
        
        # Get recent quality scores by dimension
        recent_reports = list(self.data_quality_manager.reports.values())[-5:]
        dimension_scores = {}
        
        for dimension in QualityDimension:
            scores = []
            for report in recent_reports:
                if dimension in report.dimension_scores:
                    scores.append(report.dimension_scores[dimension] * 100)
            if scores:
                dimension_scores[dimension.value] = {
                    'current': scores[-1],
                    'average': sum(scores) / len(scores),
                    'trend': scores
                }
        
        return {
            'widget_id': widget.id,
            'title': widget.title,
            'data': {
                'total_assessments': stats.get('total_assessments', 0),
                'total_rules': stats.get('total_rules', 0),
                'enabled_rules': stats.get('enabled_rules', 0),
                'rules_passed': stats.get('rules_passed', 0),
                'rules_failed': stats.get('rules_failed', 0),
                'rule_success_rate': stats.get('rule_success_rate', 0.0),
                'avg_quality_score': stats.get('avg_quality_score', 0.0) * 100,
                'dimension_scores': dimension_scores
            }
        }
    
    def add_widget(self, widget: DashboardWidget) -> None:
        """Add a custom widget to the dashboard"""
        self.widgets[widget.id] = widget
        logger.info(f"Added dashboard widget: {widget.title} ({widget.id})")
    
    def remove_widget(self, widget_id: str) -> bool:
        """Remove a widget from the dashboard"""
        if widget_id in self.widgets:
            del self.widgets[widget_id]
            logger.info(f"Removed dashboard widget: {widget_id}")
            return True
        return False
    
    def update_alert_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update alert thresholds"""
        self.alert_thresholds.update(thresholds)
        logger.info("Updated alert thresholds")
    
    def export_dashboard_config(self) -> Dict[str, Any]:
        """Export dashboard configuration"""
        return {
            'widgets': {wid: widget.__dict__ for wid, widget in self.widgets.items()},
            'alert_thresholds': self.alert_thresholds,
            'cache_ttl_minutes': self.cache_ttl.total_seconds() / 60,
            'export_timestamp': datetime.now().isoformat()
        }
    
    def import_dashboard_config(self, config: Dict[str, Any]) -> bool:
        """Import dashboard configuration"""
        try:
            # Import widgets
            if 'widgets' in config:
                self.widgets.clear()
                for wid, widget_data in config['widgets'].items():
                    # Convert position dict if needed
                    if 'position' not in widget_data:
                        widget_data['position'] = {'x': 0, 'y': 0, 'width': 4, 'height': 2}
                    
                    widget = DashboardWidget(**widget_data)
                    self.widgets[wid] = widget
            
            # Import alert thresholds
            if 'alert_thresholds' in config:
                self.alert_thresholds.update(config['alert_thresholds'])
            
            # Import cache TTL
            if 'cache_ttl_minutes' in config:
                self.cache_ttl = timedelta(minutes=config['cache_ttl_minutes'])
            
            logger.info("Successfully imported dashboard configuration")
            return True
        
        except Exception as e:
            logger.error(f"Error importing dashboard config: {str(e)}")
            return False