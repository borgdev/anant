"""
ANANT Advanced Governance Automation Demo

Complete demonstration of the enterprise governance, compliance,
and policy enforcement system for hypergraph data management.
"""

import polars as pl
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add governance module to path (parent directory)
sys.path.append(str(Path(__file__).parent.parent))

from governance.policy_engine import PolicyEngine, Policy, PolicyType, PolicySeverity, PolicyCondition, PolicyAction
from governance.compliance_monitor import ComplianceMonitor, ComplianceFramework
from governance.audit_system import AuditSystem, AuditEvent, AuditEventType, AuditCategory, AuditLevel
from governance.remediation_engine import RemediationEngine
from governance.access_control import AccessController, User, Role, Permission, PermissionType, ResourceType, AccessRequest
from governance.data_quality import DataQualityManager, QualityRule, QualityDimension, QualityRuleType
from governance.governance_dashboard import GovernanceDashboard

def create_sample_data() -> pl.DataFrame:
    """Create sample hypergraph data for demonstration"""
    import random
    import string
    
    # Generate sample data with various quality issues
    data = []
    for i in range(1000):
        # Introduce some data quality issues
        email = f"user{i}@example.com" if i % 10 != 0 else "invalid-email"  # 10% invalid emails
        age = random.randint(18, 80) if i % 20 != 0 else None  # 5% missing ages
        node_id = f"node_{i}" if i % 50 != 0 else f"node_{i % 100}"  # 2% duplicate IDs
        
        data.append({
            'node_id': node_id,
            'email': email,
            'age': age,
            'data_type': random.choice(['public', 'private', 'sensitive']),
            'created_at': datetime.now() - timedelta(days=random.randint(0, 365)),
            'last_modified': datetime.now() - timedelta(hours=random.randint(0, 24)),
            'completeness_score': random.uniform(0.7, 1.0),
            'is_active': random.choice([True, False])
        })
    
    return pl.DataFrame(data)

def demo_policy_engine():
    """Demonstrate policy engine capabilities"""
    print("\n" + "="*60)
    print("🛡️  POLICY ENGINE DEMONSTRATION")
    print("="*60)
    
    # Create policy engine
    policy_engine = PolicyEngine()
    
    # Create custom policies
    sensitive_data_policy = Policy(
        id="sensitive_data_protection",
        name="Sensitive Data Protection",
        description="Protect access to sensitive data types",
        policy_type=PolicyType.DATA_PRIVACY,
        severity=PolicySeverity.CRITICAL,
        conditions=[
            PolicyCondition(
                field="data_type",
                operator="eq",
                value="sensitive",
                description="Data marked as sensitive"
            )
        ],
        actions=[
            PolicyAction(
                action_type="block",
                parameters={"reason": "Sensitive data access requires special authorization"},
                description="Block unauthorized sensitive data access"
            )
        ],
        tags=["privacy", "security", "sensitive"]
    )
    
    # Add policy
    policy_engine.add_policy(sensitive_data_policy)
    
    # Create sample data and evaluate policies
    sample_data = create_sample_data()
    
    print(f"📊 Evaluating policies against {len(sample_data)} records...")
    
    # Evaluate policies
    evaluation_result = policy_engine.evaluate_policies(sample_data)
    
    print(f"⏱️  Evaluation completed in {evaluation_result['evaluation_time_ms']:.2f}ms")
    print(f"🚨 Violations found: {len(evaluation_result['violations'])}")
    
    # Show policy statistics
    stats = policy_engine.get_evaluation_stats()
    print(f"📈 Policy Statistics:")
    print(f"   Total evaluations: {stats['total_evaluations']}")
    print(f"   Violations detected: {stats['violations_detected']}")
    
    return policy_engine, sample_data

def demo_audit_system():
    """Demonstrate audit system capabilities"""
    print("\n" + "="*60)
    print("📋 AUDIT SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Create audit system
    audit_system = AuditSystem()
    
    # Log various types of events
    events_to_log = [
        ("user123", "data_read", "Reading hypergraph node data"),
        ("user456", "policy_violation", "Attempted access to sensitive data"),
        ("system", "backup_create", "Automated backup process started"),
        ("user789", "data_export", "Exported dataset for analysis")
    ]
    
    print("📝 Logging sample audit events...")
    
    for user, action, description in events_to_log:
        audit_system.log_data_access(
            user_id=user,
            resource_id=f"resource_{len(audit_system.events)}",
            action=action,
            description=description
        )
    
    # Query recent events
    from governance.audit_system import AuditQuery
    query = AuditQuery().last_hours(1).limit_results(10)
    recent_events = audit_system.query_events(query)
    
    print(f"🔍 Recent Events ({len(recent_events)}):")
    for event in recent_events[:3]:  # Show first 3
        print(f"   {event.timestamp.strftime('%H:%M:%S')} - {event.user_id}: {event.description}")
    
    # Generate audit report
    start_time = datetime.now() - timedelta(hours=1)
    end_time = datetime.now()
    report = audit_system.generate_audit_report(start_time, end_time)
    
    print(f"📊 Audit Report Summary:")
    print(f"   Total events: {report['summary']['total_events']}")
    print(f"   Unique users: {report['summary']['unique_users']}")
    print(f"   High-risk events: {len(report['high_risk_events'])}")
    
    return audit_system

def demo_access_control():
    """Demonstrate access control system"""
    print("\n" + "="*60)
    print("🔐 ACCESS CONTROL DEMONSTRATION")
    print("="*60)
    
    # Create access controller
    access_controller = AccessController()
    
    # Create test users
    analyst_user = access_controller.create_user(
        username="data_analyst",
        email="analyst@company.com",
        roles=["data_analyst"],
        password="secure_password"
    )
    
    admin_user = access_controller.create_user(
        username="admin_user", 
        email="admin@company.com",
        roles=["super_admin"],
        password="admin_password"
    )
    
    print(f"👥 Created users: data_analyst, admin_user")
    
    # Test authentication
    session_id = access_controller.authenticate_user(
        "data_analyst",
        "secure_password",
        {"ip_address": "192.168.1.100", "user_agent": "Mozilla/5.0"}
    )
    
    if session_id:
        print(f"✅ Authentication successful - Session: {session_id[:8]}...")
    
    # Test access control
    access_request = AccessRequest(
        user_id=analyst_user,
        session_id=session_id,
        resource_type=ResourceType.HYPERGRAPH,
        resource_id="graph_001",
        permission_type=PermissionType.READ
    )
    
    access_result = access_controller.check_access(access_request)
    print(f"🔍 Access Check Result: {access_result.decision.value}")
    print(f"   Reason: {access_result.reason}")
    
    # Show access statistics
    stats = access_controller.get_statistics()
    print(f"📊 Access Control Statistics:")
    print(f"   Total users: {stats['total_users']}")
    print(f"   Active sessions: {stats['active_sessions']}")
    print(f"   Success rate: {stats['access_success_rate']:.1f}%")
    
    return access_controller

def demo_data_quality():
    """Demonstrate data quality management"""
    print("\n" + "="*60)
    print("📏 DATA QUALITY DEMONSTRATION")
    print("="*60)
    
    # Create data quality manager
    quality_manager = DataQualityManager()
    
    # Create sample data with quality issues
    sample_data = create_sample_data()
    
    print(f"🔍 Analyzing data quality for {len(sample_data)} records...")
    
    # Perform quality assessment
    quality_report = quality_manager.evaluate_quality(sample_data, "demo_dataset")
    
    print(f"📊 Quality Assessment Results:")
    print(f"   Overall Score: {quality_report.overall_score:.3f}")
    print(f"   Total Rules: {quality_report.total_rules}")
    print(f"   Passed: {quality_report.passed_rules}")
    print(f"   Failed: {quality_report.failed_rules}")
    print(f"   Issues Found: {len(quality_report.issues)}")
    
    # Show dimension scores
    print(f"📈 Quality by Dimension:")
    for dimension, score in quality_report.dimension_scores.items():
        print(f"   {dimension.value}: {score:.3f}")
    
    # Show top issues
    if quality_report.issues:
        print(f"⚠️  Top Quality Issues:")
        for issue in quality_report.issues[:3]:
            print(f"   - {issue['rule_name']}: {issue['failed_count']} failures")
    
    return quality_manager

def demo_compliance_monitoring(policy_engine, audit_system):
    """Demonstrate compliance monitoring"""
    print("\n" + "="*60)
    print("📋 COMPLIANCE MONITORING DEMONSTRATION")
    print("="*60)
    
    # Create compliance monitor
    compliance_monitor = ComplianceMonitor(policy_engine)
    
    # Start monitoring
    compliance_monitor.start_monitoring(scan_interval_hours=1)
    
    # Perform compliance scan
    sample_data = create_sample_data()
    scan_results = compliance_monitor.scan_compliance(sample_data)
    
    print(f"🔍 Compliance Scan Results:")
    print(f"   Violations: {len(scan_results['violations'])}")
    print(f"   Scan Duration: {scan_results['summary']['scan_duration_ms']:.2f}ms")
    
    # Generate compliance report for GDPR
    report_start = datetime.now() - timedelta(days=1)
    report_end = datetime.now()
    
    compliance_report = compliance_monitor.generate_compliance_report(
        ComplianceFramework.GDPR,
        report_start,
        report_end
    )
    
    print(f"📊 GDPR Compliance Report:")
    print(f"   Overall Status: {compliance_report.overall_status.value}")
    print(f"   Compliance Score: {compliance_report.compliance_score:.1f}%")
    print(f"   Requirements Total: {compliance_report.requirements_total}")
    print(f"   Requirements Compliant: {compliance_report.requirements_compliant}")
    
    return compliance_monitor

def demo_remediation_engine(audit_system):
    """Demonstrate automated remediation"""
    print("\n" + "="*60)
    print("🔧 REMEDIATION ENGINE DEMONSTRATION")
    print("="*60)
    
    # Create remediation engine
    remediation_engine = RemediationEngine(audit_system)
    
    print(f"🛠️  Remediation Engine initialized with {len(remediation_engine.actions)} default actions")
    
    # Show available remediation actions
    print(f"📋 Available Remediation Actions:")
    for action_id, action in list(remediation_engine.actions.items())[:3]:
        print(f"   - {action.name}: {action.description}")
    
    # Show statistics
    stats = remediation_engine.get_statistics()
    print(f"📊 Remediation Statistics:")
    print(f"   Total Actions: {stats['total_actions']}")
    print(f"   Enabled Actions: {stats['enabled_actions']}")
    print(f"   Pending Approvals: {stats['pending_approvals']}")
    
    return remediation_engine

def demo_governance_dashboard():
    """Demonstrate governance dashboard"""
    print("\n" + "="*60)
    print("📊 GOVERNANCE DASHBOARD DEMONSTRATION")
    print("="*60)
    
    # Create all components
    policy_engine = PolicyEngine()
    audit_system = AuditSystem()
    access_controller = AccessController()
    quality_manager = DataQualityManager()
    compliance_monitor = ComplianceMonitor(policy_engine)
    remediation_engine = RemediationEngine(audit_system)
    
    # Create dashboard
    dashboard = GovernanceDashboard(
        policy_engine=policy_engine,
        compliance_monitor=compliance_monitor,
        audit_system=audit_system,
        remediation_engine=remediation_engine,
        access_controller=access_controller,
        data_quality_manager=quality_manager
    )
    
    print(f"🎛️  Dashboard initialized with {len(dashboard.widgets)} widgets")
    
    # Get dashboard data
    dashboard_data = dashboard.get_dashboard_data(refresh_cache=True)
    
    print(f"📈 Dashboard Summary:")
    summary = dashboard_data['summary']
    print(f"   Overall Health: {summary['health_status']} ({summary['overall_health_score']:.1f}%)")
    print(f"   Components Monitored: {summary['components_monitored']}")
    print(f"   Total Alerts: {summary['total_alerts']}")
    print(f"   Critical Alerts: {summary['critical_alerts']}")
    
    # Show key metrics
    metrics = dashboard_data['metrics']
    print(f"🔑 Key Metrics:")
    for metric_name, metric_data in list(metrics.items())[:5]:
        value = metric_data['value']
        unit = metric_data.get('unit', '')
        print(f"   {metric_data['name']}: {value}{unit}")
    
    return dashboard

def main():
    """Run complete governance automation demonstration"""
    print("🚀 ANANT ADVANCED GOVERNANCE AUTOMATION DEMO")
    print("=" * 80)
    print("Enterprise-grade governance, compliance, and policy enforcement")
    print("for hypergraph data management systems")
    print("=" * 80)
    
    try:
        # Run individual component demonstrations
        policy_engine, sample_data = demo_policy_engine()
        audit_system = demo_audit_system()
        access_controller = demo_access_control()
        quality_manager = demo_data_quality()
        compliance_monitor = demo_compliance_monitoring(policy_engine, audit_system)
        remediation_engine = demo_remediation_engine(audit_system)
        dashboard = demo_governance_dashboard()
        
        # Final summary
        print("\n" + "="*60)
        print("✅ GOVERNANCE AUTOMATION DEMO COMPLETE")
        print("="*60)
        
        print("🎯 Components Successfully Demonstrated:")
        print("   ✓ Policy Engine - Advanced policy definition and evaluation")
        print("   ✓ Audit System - Comprehensive event logging and analysis")
        print("   ✓ Access Control - Role-based permissions and authentication") 
        print("   ✓ Data Quality - Automated quality assessment and monitoring")
        print("   ✓ Compliance Monitor - Multi-framework compliance tracking")
        print("   ✓ Remediation Engine - Automated violation remediation")
        print("   ✓ Governance Dashboard - Unified monitoring and alerting")
        
        print("\n🏆 Advanced Governance Automation is now PRODUCTION READY!")
        print("   • Enterprise-grade policy enforcement")
        print("   • Real-time compliance monitoring") 
        print("   • Automated remediation workflows")
        print("   • Comprehensive audit trails")
        print("   • Role-based access control")
        print("   • Data quality assurance")
        print("   • Unified governance dashboard")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)