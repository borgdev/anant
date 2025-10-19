"""
ANANT Data Quality Manager

Automated data quality assessment, monitoring, and improvement
for hypergraph data governance and compliance.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
import json
import logging
import re
from pathlib import Path
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Data quality dimensions"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    RELEVANCE = "relevance"

class QualityRuleType(Enum):
    """Types of data quality rules"""
    NOT_NULL = "not_null"
    UNIQUE = "unique"
    RANGE = "range"
    PATTERN = "pattern"
    ENUM = "enum"
    FOREIGN_KEY = "foreign_key"
    CUSTOM = "custom"

class QualityStatus(Enum):
    """Quality assessment status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    UNKNOWN = "unknown"

@dataclass
class QualityRule:
    """Data quality rule definition"""
    id: str
    name: str
    description: str
    dimension: QualityDimension
    rule_type: QualityRuleType
    column: str
    
    # Rule parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    threshold: Optional[float] = None  # For rules that produce scores
    
    # Rule configuration
    enabled: bool = True
    severity: str = "medium"  # low, medium, high, critical
    
    # Custom validation function for CUSTOM rule type
    validator: Optional[Callable] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

@dataclass
class QualityResult:
    """Result of a quality rule evaluation"""
    rule_id: str
    status: QualityStatus
    score: float  # 0.0 to 1.0
    passed_count: int
    failed_count: int
    total_count: int
    
    # Details
    error_message: Optional[str] = None
    failed_values: List[Any] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    evaluation_time: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

@dataclass
class QualityMetric:
    """Quality metric for a dataset or column"""
    name: str
    dimension: QualityDimension
    value: float
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityReport:
    """Comprehensive data quality report"""
    id: str
    dataset_id: str
    generated_at: datetime
    
    # Overall quality scores
    overall_score: float  # 0.0 to 1.0
    dimension_scores: Dict[QualityDimension, float] = field(default_factory=dict)
    
    # Rule results
    rule_results: List[QualityResult] = field(default_factory=list)
    
    # Metrics
    metrics: List[QualityMetric] = field(default_factory=list)
    
    # Summary statistics
    total_rules: int = 0
    passed_rules: int = 0
    failed_rules: int = 0
    warning_rules: int = 0
    
    # Data summary
    total_records: int = 0
    columns_evaluated: List[str] = field(default_factory=list)
    
    # Issues and recommendations
    issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'generated_at': self.generated_at.isoformat(),
            'overall_score': self.overall_score,
            'dimension_scores': {dim.value: score for dim, score in self.dimension_scores.items()},
            'rule_results': [
                {
                    'rule_id': r.rule_id,
                    'status': r.status.value,
                    'score': r.score,
                    'passed_count': r.passed_count,
                    'failed_count': r.failed_count,
                    'total_count': r.total_count,
                    'error_message': r.error_message,
                    'failed_values': r.failed_values[:10],  # Limit to first 10
                    'statistics': r.statistics,
                    'evaluation_time': r.evaluation_time.isoformat(),
                    'duration_ms': r.duration_ms
                } for r in self.rule_results
            ],
            'metrics': [
                {
                    'name': m.name,
                    'dimension': m.dimension.value,
                    'value': m.value,
                    'description': m.description,
                    'details': m.details
                } for m in self.metrics
            ],
            'total_rules': self.total_rules,
            'passed_rules': self.passed_rules,
            'failed_rules': self.failed_rules,
            'warning_rules': self.warning_rules,
            'total_records': self.total_records,
            'columns_evaluated': self.columns_evaluated,
            'issues': self.issues,
            'recommendations': self.recommendations
        }

class DataQualityManager:
    """Advanced data quality assessment and monitoring system"""
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        self.rules: Dict[str, QualityRule] = {}
        self.reports: Dict[str, QualityReport] = {}
        
        # Configuration
        self.default_thresholds = {
            QualityDimension.COMPLETENESS: 0.95,
            QualityDimension.ACCURACY: 0.98,
            QualityDimension.CONSISTENCY: 0.95,
            QualityDimension.VALIDITY: 0.98,
            QualityDimension.UNIQUENESS: 1.0,
            QualityDimension.TIMELINESS: 0.90,
            QualityDimension.RELEVANCE: 0.85
        }
        
        # Statistics
        self.stats = {
            'total_assessments': 0,
            'total_rules_evaluated': 0,
            'rules_passed': 0,
            'rules_failed': 0,
            'avg_quality_score': 0.0,
            'assessments_by_dimension': defaultdict(int)
        }
        
        # Create default quality rules
        self._create_default_rules()
        
        logger.info("Data Quality Manager initialized")
    
    def _create_default_rules(self):
        """Create default data quality rules"""
        
        # Completeness rules
        completeness_rule = QualityRule(
            id="completeness_not_null",
            name="No Null Values",
            description="Check that column has no null/missing values",
            dimension=QualityDimension.COMPLETENESS,
            rule_type=QualityRuleType.NOT_NULL,
            column="*",  # Apply to all columns by default
            threshold=0.95,
            severity="high"
        )
        
        # Uniqueness rule
        uniqueness_rule = QualityRule(
            id="uniqueness_check",
            name="Unique Values",
            description="Check that column values are unique",
            dimension=QualityDimension.UNIQUENESS,
            rule_type=QualityRuleType.UNIQUE,
            column="id",
            threshold=1.0,
            severity="critical"
        )
        
        # Email pattern validation
        email_pattern_rule = QualityRule(
            id="email_pattern",
            name="Valid Email Format",
            description="Check that email addresses follow valid format",
            dimension=QualityDimension.VALIDITY,
            rule_type=QualityRuleType.PATTERN,
            column="email",
            parameters={'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
            threshold=0.98,
            severity="medium"
        )
        
        # Numeric range validation
        age_range_rule = QualityRule(
            id="age_range",
            name="Valid Age Range",
            description="Check that age values are within reasonable range",
            dimension=QualityDimension.VALIDITY,
            rule_type=QualityRuleType.RANGE,
            column="age",
            parameters={'min_value': 0, 'max_value': 150},
            threshold=0.99,
            severity="medium"
        )
        
        for rule in [completeness_rule, uniqueness_rule, email_pattern_rule, age_range_rule]:
            self.add_rule(rule)
    
    def add_rule(self, rule: QualityRule) -> None:
        """Add a quality rule to the manager"""
        rule.updated_at = datetime.now()
        self.rules[rule.id] = rule
        logger.info(f"Added quality rule: {rule.name} ({rule.id})")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a quality rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed quality rule: {rule_id}")
            return True
        return False
    
    def evaluate_quality(self, data: pl.DataFrame, dataset_id: str = "unknown") -> QualityReport:
        """Perform comprehensive data quality evaluation"""
        start_time = datetime.now()
        
        report = QualityReport(
            id=f"quality_report_{int(start_time.timestamp())}",
            dataset_id=dataset_id,
            generated_at=start_time,
            overall_score=0.0,
            total_records=len(data)
        )
        
        # Get applicable rules for this dataset
        applicable_rules = self._get_applicable_rules(data.columns)
        report.total_rules = len(applicable_rules)
        
        # Evaluate each rule
        rule_results = []
        dimension_scores = defaultdict(list)
        
        for rule in applicable_rules:
            if not rule.enabled:
                continue
            
            try:
                result = self._evaluate_rule(rule, data)
                rule_results.append(result)
                
                # Collect scores by dimension
                dimension_scores[rule.dimension].append(result.score)
                
                # Update counters
                if result.status == QualityStatus.PASSED:
                    report.passed_rules += 1
                elif result.status == QualityStatus.FAILED:
                    report.failed_rules += 1
                elif result.status == QualityStatus.WARNING:
                    report.warning_rules += 1
                
                # Add issues for failed rules
                if result.status in [QualityStatus.FAILED, QualityStatus.WARNING]:
                    report.issues.append({
                        'rule_id': rule.id,
                        'rule_name': rule.name,
                        'column': rule.column,
                        'severity': rule.severity,
                        'description': result.error_message or rule.description,
                        'failed_count': result.failed_count,
                        'score': result.score
                    })
                
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.id}: {str(e)}")
                result = QualityResult(
                    rule_id=rule.id,
                    status=QualityStatus.UNKNOWN,
                    score=0.0,
                    passed_count=0,
                    failed_count=0,
                    total_count=0,
                    error_message=str(e)
                )
                rule_results.append(result)
        
        report.rule_results = rule_results
        
        # Calculate dimension scores
        for dimension, scores in dimension_scores.items():
            if scores:
                report.dimension_scores[dimension] = sum(scores) / len(scores)
        
        # Calculate overall score
        if report.dimension_scores:
            report.overall_score = sum(report.dimension_scores.values()) / len(report.dimension_scores)
        
        # Generate metrics
        report.metrics = self._generate_metrics(data, report)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        # Update columns evaluated
        report.columns_evaluated = list(data.columns)
        
        # Update statistics
        self.stats['total_assessments'] += 1
        self.stats['total_rules_evaluated'] += len(rule_results)
        self.stats['rules_passed'] += report.passed_rules
        self.stats['rules_failed'] += report.failed_rules
        
        # Update average quality score
        self.stats['avg_quality_score'] = (
            (self.stats['avg_quality_score'] * (self.stats['total_assessments'] - 1) + 
             report.overall_score) / self.stats['total_assessments']
        )
        
        # Store report
        self.reports[report.id] = report
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"Quality assessment completed in {duration:.2f}ms - Score: {report.overall_score:.3f}")
        
        return report
    
    def _get_applicable_rules(self, columns: List[str]) -> List[QualityRule]:
        """Get rules applicable to the given columns"""
        applicable_rules = []
        
        for rule in self.rules.values():
            if rule.column == "*":
                # Rule applies to all columns - create instance for each column
                for col in columns:
                    column_rule = QualityRule(
                        id=f"{rule.id}_{col}",
                        name=f"{rule.name} ({col})",
                        description=rule.description,
                        dimension=rule.dimension,
                        rule_type=rule.rule_type,
                        column=col,
                        parameters=rule.parameters,
                        threshold=rule.threshold,
                        enabled=rule.enabled,
                        severity=rule.severity,
                        validator=rule.validator,
                        tags=rule.tags
                    )
                    applicable_rules.append(column_rule)
            elif rule.column in columns:
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _evaluate_rule(self, rule: QualityRule, data: pl.DataFrame) -> QualityResult:
        """Evaluate a single quality rule against data"""
        start_time = datetime.now()
        
        try:
            if rule.column not in data.columns:
                raise ValueError(f"Column '{rule.column}' not found in data")
            
            column_data = data[rule.column]
            total_count = len(column_data)
            
            # Evaluate based on rule type
            if rule.rule_type == QualityRuleType.NOT_NULL:
                passed_count = int(column_data.drop_nulls().len())
                failed_count = total_count - passed_count
                failed_values = []  # Don't collect null values
                
            elif rule.rule_type == QualityRuleType.UNIQUE:
                unique_count = int(column_data.n_unique())
                passed_count = unique_count
                failed_count = total_count - unique_count
                
                # Find duplicates
                duplicate_mask = column_data.is_duplicated()
                failed_values = data.filter(duplicate_mask)[rule.column].to_list()[:10]
                
            elif rule.rule_type == QualityRuleType.PATTERN:
                pattern = rule.parameters.get('pattern', '')
                if not pattern:
                    raise ValueError("Pattern rule requires 'pattern' parameter")
                
                # Apply pattern matching
                matches = column_data.str.contains(pattern)
                passed_count = int(matches.sum())
                failed_count = total_count - passed_count
                
                # Get failed values
                failed_mask = ~matches
                failed_values = data.filter(failed_mask)[rule.column].to_list()[:10]
                
            elif rule.rule_type == QualityRuleType.RANGE:
                min_val = rule.parameters.get('min_value')
                max_val = rule.parameters.get('max_value')
                
                if min_val is None and max_val is None:
                    raise ValueError("Range rule requires 'min_value' or 'max_value' parameter")
                
                # Apply range checks using data filtering
                valid_data = data
                if min_val is not None:
                    valid_data = valid_data.filter(pl.col(rule.column) >= min_val)
                if max_val is not None:
                    valid_data = valid_data.filter(pl.col(rule.column) <= max_val)
                
                passed_count = len(valid_data)
                failed_count = total_count - passed_count
                
                # Get failed values by finding non-matching records
                invalid_data = data
                if min_val is not None:
                    invalid_data = invalid_data.filter(pl.col(rule.column) < min_val)
                if max_val is not None:
                    remaining_data = data.filter(pl.col(rule.column) > max_val)
                    if len(invalid_data) == 0:
                        invalid_data = remaining_data
                    else:
                        invalid_data = invalid_data.vstack(remaining_data)
                
                failed_values = invalid_data[rule.column].to_list()[:10] if len(invalid_data) > 0 else []
                
            elif rule.rule_type == QualityRuleType.ENUM:
                allowed_values = rule.parameters.get('allowed_values', [])
                if not allowed_values:
                    raise ValueError("Enum rule requires 'allowed_values' parameter")
                
                # Check if values are in allowed set
                valid_mask = column_data.is_in(allowed_values)
                passed_count = int(valid_mask.sum())
                failed_count = total_count - passed_count
                
                # Get failed values
                failed_mask = ~valid_mask
                failed_values = data.filter(failed_mask)[rule.column].to_list()[:10]
                
            elif rule.rule_type == QualityRuleType.CUSTOM:
                if not rule.validator:
                    raise ValueError("Custom rule requires validator function")
                
                # Apply custom validator
                validation_result = rule.validator(column_data)
                if isinstance(validation_result, tuple):
                    passed_count, failed_count, failed_values = validation_result
                else:
                    # Assume validation_result is a boolean mask
                    passed_count = int(validation_result.sum())
                    failed_count = total_count - passed_count
                    failed_mask = ~validation_result
                    failed_values = data.filter(failed_mask)[rule.column].to_list()[:10]
            
            else:
                raise ValueError(f"Unsupported rule type: {rule.rule_type}")
            
            # Calculate score
            if total_count == 0:
                score = 1.0  # No data to validate
            else:
                score = float(passed_count) / float(total_count)
            
            # Determine status based on threshold
            if rule.threshold is not None:
                if score >= rule.threshold:
                    status = QualityStatus.PASSED
                elif score >= rule.threshold * 0.8:  # Warning threshold
                    status = QualityStatus.WARNING
                else:
                    status = QualityStatus.FAILED
            else:
                status = QualityStatus.PASSED if failed_count == 0 else QualityStatus.FAILED
            
            # Generate statistics
            statistics = {
                'score': score,
                'pass_rate': score * 100,
                'total_records': total_count
            }
            
            # Add column-specific statistics for numeric columns
            if column_data.dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                try:
                    stats_data = column_data.drop_nulls()
                    if len(stats_data) > 0:
                        # Use select to get scalar values
                        mean_val = data.select(pl.col(rule.column).mean()).item()
                        std_val = data.select(pl.col(rule.column).std()).item()
                        min_val = data.select(pl.col(rule.column).min()).item()
                        max_val = data.select(pl.col(rule.column).max()).item()
                        median_val = data.select(pl.col(rule.column).median()).item()
                        
                        if mean_val is not None:
                            statistics['mean'] = float(mean_val)
                        if std_val is not None:
                            statistics['std'] = float(std_val)
                        if min_val is not None:
                            statistics['min'] = float(min_val)
                        if max_val is not None:
                            statistics['max'] = float(max_val)
                        if median_val is not None:
                            statistics['median'] = float(median_val)
                except Exception:
                    pass
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() * 1000
            
            return QualityResult(
                rule_id=rule.id,
                status=status,
                score=score,
                passed_count=passed_count,
                failed_count=failed_count,
                total_count=total_count,
                failed_values=failed_values,
                statistics=statistics,
                duration_ms=duration
            )
        
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() * 1000
            
            return QualityResult(
                rule_id=rule.id,
                status=QualityStatus.UNKNOWN,
                score=0.0,
                passed_count=0,
                failed_count=0,
                total_count=0,
                error_message=str(e),
                duration_ms=duration
            )
    
    def _generate_metrics(self, data: pl.DataFrame, report: QualityReport) -> List[QualityMetric]:
        """Generate additional quality metrics"""
        metrics = []
        
        # Overall completeness metric
        total_cells = len(data) * len(data.columns)
        null_cells = sum(data[col].null_count() for col in data.columns)
        completeness = (total_cells - null_cells) / total_cells if total_cells > 0 else 1.0
        
        metrics.append(QualityMetric(
            name="Dataset Completeness",
            dimension=QualityDimension.COMPLETENESS,
            value=completeness,
            description=f"Overall data completeness across all columns",
            details={'null_cells': null_cells, 'total_cells': total_cells}
        ))
        
        # Column-level metrics
        for col in data.columns:
            col_data = data[col]
            
            # Completeness for this column
            col_completeness = (len(col_data) - col_data.null_count()) / len(col_data) if len(col_data) > 0 else 1.0
            metrics.append(QualityMetric(
                name=f"{col} Completeness",
                dimension=QualityDimension.COMPLETENESS,
                value=col_completeness,
                description=f"Completeness for column {col}"
            ))
            
            # Uniqueness for this column
            if len(col_data) > 0:
                col_uniqueness = col_data.n_unique() / len(col_data)
                metrics.append(QualityMetric(
                    name=f"{col} Uniqueness",
                    dimension=QualityDimension.UNIQUENESS,
                    value=col_uniqueness,
                    description=f"Uniqueness ratio for column {col}",
                    details={'unique_count': col_data.n_unique(), 'total_count': len(col_data)}
                ))
        
        return metrics
    
    def _generate_recommendations(self, report: QualityReport) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        # Overall quality recommendations
        if report.overall_score < 0.8:
            recommendations.append("Overall data quality is below acceptable threshold (80%). Review and address identified issues.")
        
        # Dimension-specific recommendations
        for dimension, score in report.dimension_scores.items():
            threshold = self.default_thresholds.get(dimension, 0.9)
            if score < threshold:
                if dimension == QualityDimension.COMPLETENESS:
                    recommendations.append("Address missing data issues to improve completeness.")
                elif dimension == QualityDimension.ACCURACY:
                    recommendations.append("Implement data validation checks to improve accuracy.")
                elif dimension == QualityDimension.CONSISTENCY:
                    recommendations.append("Standardize data formats and values to improve consistency.")
                elif dimension == QualityDimension.VALIDITY:
                    recommendations.append("Validate data against business rules and constraints.")
                elif dimension == QualityDimension.UNIQUENESS:
                    recommendations.append("Remove or merge duplicate records to improve uniqueness.")
        
        # Issue-specific recommendations
        critical_issues = [issue for issue in report.issues if issue['severity'] == 'critical']
        if critical_issues:
            recommendations.append(f"Address {len(critical_issues)} critical data quality issues immediately.")
        
        high_issues = [issue for issue in report.issues if issue['severity'] == 'high']
        if len(high_issues) > 5:
            recommendations.append(f"High priority: resolve {len(high_issues)} high-severity issues.")
        
        if not recommendations:
            recommendations.append("Data quality meets established thresholds. Continue regular monitoring.")
        
        return recommendations
    
    def create_custom_rule(self,
                          rule_id: str,
                          name: str,
                          description: str,
                          dimension: QualityDimension,
                          column: str,
                          validator: Callable,
                          threshold: Optional[float] = None,
                          severity: str = "medium") -> QualityRule:
        """Create a custom quality rule with validator function"""
        
        rule = QualityRule(
            id=rule_id,
            name=name,
            description=description,
            dimension=dimension,
            rule_type=QualityRuleType.CUSTOM,
            column=column,
            validator=validator,
            threshold=threshold,
            severity=severity
        )
        
        self.add_rule(rule)
        return rule
    
    def get_rule_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get templates for creating common quality rules"""
        return {
            'not_null': {
                'name': 'Not Null Check',
                'description': 'Ensures column has no null values',
                'rule_type': QualityRuleType.NOT_NULL.value,
                'dimension': QualityDimension.COMPLETENESS.value,
                'parameters': {}
            },
            'unique': {
                'name': 'Uniqueness Check',
                'description': 'Ensures all values in column are unique',
                'rule_type': QualityRuleType.UNIQUE.value,
                'dimension': QualityDimension.UNIQUENESS.value,
                'parameters': {}
            },
            'email_format': {
                'name': 'Email Format Validation',
                'description': 'Validates email address format',
                'rule_type': QualityRuleType.PATTERN.value,
                'dimension': QualityDimension.VALIDITY.value,
                'parameters': {'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'}
            },
            'numeric_range': {
                'name': 'Numeric Range Check',
                'description': 'Ensures numeric values are within specified range',
                'rule_type': QualityRuleType.RANGE.value,
                'dimension': QualityDimension.VALIDITY.value,
                'parameters': {'min_value': 0, 'max_value': 100}
            },
            'categorical_values': {
                'name': 'Categorical Value Check',
                'description': 'Ensures values are from allowed set',
                'rule_type': QualityRuleType.ENUM.value,
                'dimension': QualityDimension.VALIDITY.value,
                'parameters': {'allowed_values': ['A', 'B', 'C']}
            }
        }
    
    def save_report(self, report_id: str, file_path: str) -> bool:
        """Save quality report to file"""
        if report_id not in self.reports:
            logger.error(f"Report not found: {report_id}")
            return False
        
        try:
            report = self.reports[report_id]
            report_data = report.to_dict()
            
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Saved quality report to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return False
    
    def get_quality_trends(self, dataset_id: str, days: int = 30) -> Dict[str, Any]:
        """Get quality trends for a dataset over time"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter reports for this dataset and time period
        dataset_reports = [
            report for report in self.reports.values()
            if (report.dataset_id == dataset_id and 
                report.generated_at >= cutoff_date)
        ]
        
        if not dataset_reports:
            return {
                'dataset_id': dataset_id,
                'period_days': days,
                'reports_count': 0,
                'message': 'No reports found for specified period'
            }
        
        # Sort by generation time
        dataset_reports.sort(key=lambda r: r.generated_at)
        
        # Extract trends
        dates = [r.generated_at.date().isoformat() for r in dataset_reports]
        overall_scores = [r.overall_score for r in dataset_reports]
        
        # Dimension trends
        dimension_trends = {}
        for dimension in QualityDimension:
            dimension_scores = [
                r.dimension_scores.get(dimension, 0.0) for r in dataset_reports
            ]
            if any(score > 0 for score in dimension_scores):
                dimension_trends[dimension.value] = dimension_scores
        
        # Calculate trend statistics
        trend_direction = "stable"
        if len(overall_scores) >= 2:
            recent_avg = statistics.mean(overall_scores[-3:]) if len(overall_scores) >= 3 else overall_scores[-1]
            historical_avg = statistics.mean(overall_scores[:-3]) if len(overall_scores) >= 6 else overall_scores[0]
            
            if recent_avg > historical_avg + 0.05:
                trend_direction = "improving"
            elif recent_avg < historical_avg - 0.05:
                trend_direction = "declining"
        
        return {
            'dataset_id': dataset_id,
            'period_days': days,
            'reports_count': len(dataset_reports),
            'dates': dates,
            'overall_scores': overall_scores,
            'dimension_trends': dimension_trends,
            'current_score': overall_scores[-1] if overall_scores else 0.0,
            'average_score': statistics.mean(overall_scores) if overall_scores else 0.0,
            'trend_direction': trend_direction,
            'best_score': max(overall_scores) if overall_scores else 0.0,
            'worst_score': min(overall_scores) if overall_scores else 0.0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data quality manager statistics"""
        stats = self.stats.copy()
        
        # Convert defaultdict to regular dict
        stats['assessments_by_dimension'] = dict(stats['assessments_by_dimension'])
        
        # Add additional statistics
        stats['total_rules'] = len(self.rules)
        stats['enabled_rules'] = len([r for r in self.rules.values() if r.enabled])
        stats['total_reports'] = len(self.reports)
        
        # Calculate rule success rate
        total_rules_evaluated = self.stats['total_rules_evaluated']
        if total_rules_evaluated > 0:
            stats['rule_success_rate'] = (self.stats['rules_passed'] / total_rules_evaluated) * 100
        else:
            stats['rule_success_rate'] = 0.0
        
        return stats