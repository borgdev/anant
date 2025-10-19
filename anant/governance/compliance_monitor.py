"""
ANANT Compliance Monitor

Real-time compliance monitoring, reporting, and alerting system
for hypergraph governance and regulatory requirements.
"""

import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from collections import defaultdict

from .policy_engine import Policy, PolicyType, PolicySeverity

logger = logging.getLogger(__name__)

class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNKNOWN = "unknown"
    EXEMPT = "exempt"

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    CCPA = "ccpa"
    CUSTOM = "custom"

@dataclass
class ComplianceRequirement:
    """Individual compliance requirement"""
    id: str
    name: str
    description: str
    framework: ComplianceFramework
    category: str
    mandatory: bool = True
    policy_ids: List[str] = field(default_factory=list)
    evidence_required: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    id: str
    requirement_id: str
    policy_id: str
    violation_type: str
    severity: PolicySeverity
    description: str
    detected_at: datetime
    status: str = "open"  # open, investigating, remediated, closed
    affected_records: int = 0
    risk_score: float = 0.0
    remediation_deadline: Optional[datetime] = None
    assigned_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    id: str
    framework: ComplianceFramework
    report_period_start: datetime
    report_period_end: datetime
    generated_at: datetime
    overall_status: ComplianceStatus
    compliance_score: float  # 0.0 to 100.0
    
    # Detailed results
    requirements_total: int = 0
    requirements_compliant: int = 0
    requirements_non_compliant: int = 0
    requirements_partially_compliant: int = 0
    
    violations_total: int = 0
    violations_critical: int = 0
    violations_high: int = 0
    violations_medium: int = 0
    violations_low: int = 0
    
    # Detailed breakdown
    requirement_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    violation_summary: List[Dict[str, Any]] = field(default_factory=list)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'id': self.id,
            'framework': self.framework.value,
            'report_period_start': self.report_period_start.isoformat(),
            'report_period_end': self.report_period_end.isoformat(),
            'generated_at': self.generated_at.isoformat(),
            'overall_status': self.overall_status.value,
            'compliance_score': self.compliance_score,
            'requirements_total': self.requirements_total,
            'requirements_compliant': self.requirements_compliant,
            'requirements_non_compliant': self.requirements_non_compliant,
            'requirements_partially_compliant': self.requirements_partially_compliant,
            'violations_total': self.violations_total,
            'violations_critical': self.violations_critical,
            'violations_high': self.violations_high,
            'violations_medium': self.violations_medium,
            'violations_low': self.violations_low,
            'requirement_results': self.requirement_results,
            'violation_summary': self.violation_summary,
            'trend_analysis': self.trend_analysis,
            'recommendations': self.recommendations
        }

class ComplianceMonitor:
    """Advanced compliance monitoring and reporting system"""
    
    def __init__(self, policy_engine=None):
        self.policy_engine = policy_engine
        self.requirements: Dict[str, ComplianceRequirement] = {}
        self.violations: Dict[str, ComplianceViolation] = {}
        self.reports: Dict[str, ComplianceReport] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.last_scan_time = None
        self.scan_interval = timedelta(hours=1)  # Default scan interval
        
        # Statistics
        self.monitoring_stats = {
            'total_scans': 0,
            'violations_detected': 0,
            'reports_generated': 0,
            'avg_scan_time_ms': 0.0,
            'frameworks_monitored': set()
        }
        
        # Load default compliance requirements
        self._load_default_requirements()
    
    def _load_default_requirements(self):
        """Load default compliance requirements for common frameworks"""
        
        # GDPR Requirements
        gdpr_requirements = [
            ComplianceRequirement(
                id="gdpr_001",
                name="Data Protection by Design",
                description="Implement data protection measures by design and by default",
                framework=ComplianceFramework.GDPR,
                category="data_protection",
                evidence_required=["privacy_impact_assessment", "data_protection_measures"],
                tags=["gdpr", "privacy", "design"]
            ),
            ComplianceRequirement(
                id="gdpr_002",
                name="Consent Management",
                description="Obtain and manage valid consent for data processing",
                framework=ComplianceFramework.GDPR,
                category="consent",
                evidence_required=["consent_records", "consent_withdrawal_mechanism"],
                tags=["gdpr", "consent"]
            ),
            ComplianceRequirement(
                id="gdpr_003",
                name="Data Breach Notification",
                description="Report data breaches within 72 hours",
                framework=ComplianceFramework.GDPR,
                category="incident_response",
                evidence_required=["incident_logs", "notification_records"],
                tags=["gdpr", "breach", "notification"]
            )
        ]
        
        # HIPAA Requirements
        hipaa_requirements = [
            ComplianceRequirement(
                id="hipaa_001",
                name="Administrative Safeguards",
                description="Implement administrative safeguards for PHI",
                framework=ComplianceFramework.HIPAA,
                category="administrative",
                evidence_required=["policies", "training_records", "access_controls"],
                tags=["hipaa", "administrative"]
            ),
            ComplianceRequirement(
                id="hipaa_002",
                name="Physical Safeguards",
                description="Implement physical safeguards for PHI systems",
                framework=ComplianceFramework.HIPAA,
                category="physical",
                evidence_required=["facility_access_controls", "workstation_security"],
                tags=["hipaa", "physical"]
            )
        ]
        
        # Add all requirements
        all_requirements = gdpr_requirements + hipaa_requirements
        for req in all_requirements:
            self.add_requirement(req)
    
    def add_requirement(self, requirement: ComplianceRequirement) -> None:
        """Add a compliance requirement"""
        self.requirements[requirement.id] = requirement
        self.monitoring_stats['frameworks_monitored'].add(requirement.framework.value)
        logger.info(f"Added compliance requirement: {requirement.name} ({requirement.id})")
    
    def remove_requirement(self, requirement_id: str) -> bool:
        """Remove a compliance requirement"""
        if requirement_id in self.requirements:
            del self.requirements[requirement_id]
            logger.info(f"Removed compliance requirement: {requirement_id}")
            return True
        return False
    
    def get_requirements_by_framework(self, framework: ComplianceFramework) -> List[ComplianceRequirement]:
        """Get all requirements for a specific framework"""
        return [req for req in self.requirements.values() if req.framework == framework]
    
    def start_monitoring(self, scan_interval_hours: int = 1) -> None:
        """Start continuous compliance monitoring"""
        self.monitoring_active = True
        self.scan_interval = timedelta(hours=scan_interval_hours)
        logger.info(f"Started compliance monitoring with {scan_interval_hours}h interval")
    
    def stop_monitoring(self) -> None:
        """Stop compliance monitoring"""
        self.monitoring_active = False
        logger.info("Stopped compliance monitoring")
    
    def scan_compliance(self, data: pl.DataFrame, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform compliance scan against data"""
        start_time = datetime.now()
        scan_results = {
            'scan_id': f"scan_{int(start_time.timestamp())}",
            'timestamp': start_time.isoformat(),
            'violations': [],
            'compliance_status': {},
            'summary': {}
        }
        
        context = context or {}
        
        if not self.policy_engine:
            logger.warning("No policy engine configured for compliance scanning")
            return scan_results
        
        # Run policy evaluations
        policy_results = self.policy_engine.evaluate_policies(data, context)
        
        # Process violations for compliance tracking
        for violation in policy_results.get('violations', []):
            compliance_violation = self._create_compliance_violation(violation)
            if compliance_violation:
                self.violations[compliance_violation.id] = compliance_violation
                scan_results['violations'].append(compliance_violation)
        
        # Calculate compliance status for each framework
        for framework in ComplianceFramework:
            framework_status = self._calculate_framework_compliance(framework, data)
            scan_results['compliance_status'][framework.value] = framework_status
        
        end_time = datetime.now()
        scan_duration = (end_time - start_time).total_seconds() * 1000
        
        # Update statistics
        self.monitoring_stats['total_scans'] += 1
        self.monitoring_stats['violations_detected'] += len(scan_results['violations'])
        self.monitoring_stats['avg_scan_time_ms'] = (
            (self.monitoring_stats['avg_scan_time_ms'] * (self.monitoring_stats['total_scans'] - 1) + 
             scan_duration) / self.monitoring_stats['total_scans']
        )
        
        scan_results['summary'] = {
            'scan_duration_ms': scan_duration,
            'violations_found': len(scan_results['violations']),
            'frameworks_evaluated': len(scan_results['compliance_status'])
        }
        
        self.last_scan_time = end_time
        return scan_results
    
    def _create_compliance_violation(self, policy_violation: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Create compliance violation from policy violation"""
        try:
            # Find relevant compliance requirements
            relevant_requirements = []
            policy_id = policy_violation.get('policy_id')
            
            for req in self.requirements.values():
                if policy_id in req.policy_ids:
                    relevant_requirements.append(req)
            
            if not relevant_requirements:
                return None
            
            # Use first relevant requirement
            requirement = relevant_requirements[0]
            
            violation_id = f"cv_{int(datetime.now().timestamp())}_{policy_id}"
            
            return ComplianceViolation(
                id=violation_id,
                requirement_id=requirement.id,
                policy_id=policy_id,
                violation_type=policy_violation.get('policy_type', 'unknown'),
                severity=PolicySeverity(policy_violation.get('severity', 'medium')),
                description=f"Policy violation: {policy_violation.get('policy_name', 'Unknown')}",
                detected_at=datetime.now(),
                affected_records=policy_violation.get('violation_count', 0),
                risk_score=self._calculate_risk_score(policy_violation),
                metadata={
                    'policy_violation': policy_violation,
                    'requirement': requirement.name
                }
            )
        
        except Exception as e:
            logger.error(f"Error creating compliance violation: {str(e)}")
            return None
    
    def _calculate_risk_score(self, violation: Dict[str, Any]) -> float:
        """Calculate risk score for violation"""
        base_score = {
            'critical': 90.0,
            'high': 70.0,
            'medium': 50.0,
            'low': 30.0,
            'info': 10.0
        }
        
        severity = violation.get('severity', 'medium')
        score = base_score.get(severity, 50.0)
        
        # Adjust based on violation count
        violation_count = violation.get('violation_count', 1)
        if violation_count > 100:
            score *= 1.5
        elif violation_count > 10:
            score *= 1.2
        
        return min(score, 100.0)
    
    def _calculate_framework_compliance(self, framework: ComplianceFramework, data: pl.DataFrame) -> Dict[str, Any]:
        """Calculate compliance status for a specific framework"""
        requirements = self.get_requirements_by_framework(framework)
        if not requirements:
            return {
                'status': ComplianceStatus.UNKNOWN.value,
                'score': 0.0,
                'requirements_total': 0,
                'requirements_met': 0
            }
        
        total_requirements = len(requirements)
        requirements_met = 0
        
        # Simple compliance calculation - can be enhanced with more sophisticated logic
        for req in requirements:
            # Check if any violations exist for this requirement
            req_violations = [v for v in self.violations.values() if v.requirement_id == req.id]
            if not req_violations:
                requirements_met += 1
        
        compliance_score = (requirements_met / total_requirements) * 100 if total_requirements > 0 else 0
        
        if compliance_score == 100:
            status = ComplianceStatus.COMPLIANT
        elif compliance_score >= 80:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        return {
            'status': status.value,
            'score': compliance_score,
            'requirements_total': total_requirements,
            'requirements_met': requirements_met,
            'requirements_failed': total_requirements - requirements_met
        }
    
    def generate_compliance_report(self, 
                                 framework: ComplianceFramework,
                                 period_start: datetime,
                                 period_end: datetime) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        report_id = f"report_{framework.value}_{int(datetime.now().timestamp())}"
        
        # Get violations in period
        period_violations = [
            v for v in self.violations.values()
            if period_start <= v.detected_at <= period_end
        ]
        
        # Count violations by severity
        violations_by_severity = defaultdict(int)
        for violation in period_violations:
            violations_by_severity[violation.severity.value] += 1
        
        # Calculate requirement compliance
        requirements = self.get_requirements_by_framework(framework)
        requirements_compliant = 0
        requirements_non_compliant = 0
        requirements_partially_compliant = 0
        
        requirement_results = {}
        for req in requirements:
            req_violations = [v for v in period_violations if v.requirement_id == req.id]
            
            if not req_violations:
                status = ComplianceStatus.COMPLIANT
                requirements_compliant += 1
            elif len(req_violations) <= 2:  # Threshold for partial compliance
                status = ComplianceStatus.PARTIALLY_COMPLIANT
                requirements_partially_compliant += 1
            else:
                status = ComplianceStatus.NON_COMPLIANT
                requirements_non_compliant += 1
            
            requirement_results[req.id] = {
                'name': req.name,
                'status': status.value,
                'violations_count': len(req_violations),
                'description': req.description
            }
        
        # Calculate overall compliance score
        total_requirements = len(requirements)
        if total_requirements > 0:
            compliance_score = (
                (requirements_compliant * 100 + 
                 requirements_partially_compliant * 50) / total_requirements
            )
        else:
            compliance_score = 100.0
        
        # Determine overall status
        if compliance_score >= 95:
            overall_status = ComplianceStatus.COMPLIANT
        elif compliance_score >= 75:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT
        
        # Generate recommendations
        recommendations = []
        if violations_by_severity.get('critical', 0) > 0:
            recommendations.append("Address critical violations immediately")
        if violations_by_severity.get('high', 0) > 5:
            recommendations.append("Review and remediate high-severity violations")
        if compliance_score < 80:
            recommendations.append("Implement comprehensive compliance improvement plan")
        
        report = ComplianceReport(
            id=report_id,
            framework=framework,
            report_period_start=period_start,
            report_period_end=period_end,
            generated_at=datetime.now(),
            overall_status=overall_status,
            compliance_score=compliance_score,
            requirements_total=total_requirements,
            requirements_compliant=requirements_compliant,
            requirements_non_compliant=requirements_non_compliant,
            requirements_partially_compliant=requirements_partially_compliant,
            violations_total=len(period_violations),
            violations_critical=violations_by_severity.get('critical', 0),
            violations_high=violations_by_severity.get('high', 0),
            violations_medium=violations_by_severity.get('medium', 0),
            violations_low=violations_by_severity.get('low', 0),
            requirement_results=requirement_results,
            violation_summary=[
                {
                    'id': v.id,
                    'requirement_id': v.requirement_id,
                    'severity': v.severity.value,
                    'description': v.description,
                    'detected_at': v.detected_at.isoformat(),
                    'status': v.status
                } for v in period_violations
            ],
            recommendations=recommendations
        )
        
        self.reports[report_id] = report
        self.monitoring_stats['reports_generated'] += 1
        
        logger.info(f"Generated compliance report: {report_id} for {framework.value}")
        return report
    
    def save_report(self, report_id: str, file_path: str) -> bool:
        """Save compliance report to file"""
        if report_id not in self.reports:
            logger.error(f"Report not found: {report_id}")
            return False
        
        try:
            report = self.reports[report_id]
            report_data = report.to_dict()
            
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Saved compliance report to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return False
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get compliance monitoring statistics"""
        stats = self.monitoring_stats.copy()
        stats['frameworks_monitored'] = list(stats['frameworks_monitored'])
        stats['monitoring_active'] = self.monitoring_active
        stats['last_scan_time'] = self.last_scan_time.isoformat() if self.last_scan_time else None
        stats['total_violations'] = len(self.violations)
        stats['total_reports'] = len(self.reports)
        return stats
    
    def get_violation_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get violation trends over specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_violations = [
            v for v in self.violations.values()
            if v.detected_at >= cutoff_date
        ]
        
        # Group violations by day
        daily_counts = defaultdict(int)
        severity_trends = defaultdict(lambda: defaultdict(int))
        
        for violation in recent_violations:
            day_key = violation.detected_at.date().isoformat()
            daily_counts[day_key] += 1
            severity_trends[day_key][violation.severity.value] += 1
        
        return {
            'period_days': days,
            'total_violations': len(recent_violations),
            'daily_counts': dict(daily_counts),
            'severity_trends': {k: dict(v) for k, v in severity_trends.items()},
            'avg_violations_per_day': len(recent_violations) / days if days > 0 else 0
        }