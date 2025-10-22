"""
Anant Exception Hierarchy
=========================

Comprehensive exception classes for all Anant modules with proper error handling,
logging integration, and structured error information.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AnantError(Exception):
    """
    Base exception for all Anant library errors.
    
    Provides structured error information, logging integration,
    and context preservation for debugging.
    """
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None):
        """
        Initialize Anant error.
        
        Args:
            message: Human-readable error description
            error_code: Machine-readable error identifier
            context: Additional context information
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now()
        
        # Log the error
        logger.error(
            f"Anant Error: {message}",
            extra={
                "error_code": self.error_code,
                "error_message": message,  # Changed from 'message' to 'error_message'
                "context": self.context,
                "timestamp": self.timestamp.isoformat(),
                "cause": str(cause) if cause else None
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None
        }


class GraphError(AnantError):
    """Base exception for graph-related errors."""
    pass


class HypergraphError(GraphError):
    """Exceptions specific to hypergraph operations."""
    pass


class NodeError(HypergraphError):
    """Errors related to node operations."""
    pass


class EdgeError(HypergraphError):
    """Errors related to edge operations."""
    pass


class IncidenceError(HypergraphError):
    """Errors related to incidence operations."""
    pass


class MetagraphError(AnantError):
    """Base exception for metagraph-related errors."""
    pass


class EntityError(MetagraphError):
    """Errors related to entity operations."""
    pass


class RelationshipError(MetagraphError):
    """Errors related to relationship operations."""
    pass


class MetadataError(MetagraphError):
    """Errors related to metadata operations."""
    pass


class GovernanceError(MetagraphError):
    """Errors related to governance and policy operations."""
    pass


class TemporalError(MetagraphError):
    """Errors related to temporal operations."""
    pass


class KnowledgeGraphError(AnantError):
    """Base exception for knowledge graph errors."""
    pass


class SemanticError(KnowledgeGraphError):
    """Errors related to semantic operations."""
    pass


class OntologyError(KnowledgeGraphError):
    """Errors related to ontology operations."""
    pass


class ReasoningError(KnowledgeGraphError):
    """Errors related to reasoning and inference."""
    pass


class QueryError(KnowledgeGraphError):
    """Errors related to query operations."""
    pass


class StreamingError(AnantError):
    """Base exception for streaming-related errors."""
    pass


class EventError(StreamingError):
    """Errors related to event processing."""
    pass


class TemporalGraphError(StreamingError):
    """Errors related to temporal graph operations."""
    pass


class EventStoreError(StreamingError):
    """Errors related to event store operations."""
    pass


class DistributedError(AnantError):
    """Base exception for distributed computing errors."""
    pass


class PartitioningError(DistributedError):
    """Errors related to graph partitioning."""
    pass


class ClusterError(DistributedError):
    """Errors related to cluster operations."""
    pass


class ReplicationError(DistributedError):
    """Errors related to data replication."""
    pass


class IoError(AnantError):
    """Base exception for I/O operations."""
    pass


class FileFormatError(IoError):
    """Errors related to file format operations."""
    pass


class SerializationError(IoError):
    """Errors related to data serialization."""
    pass


class ValidationError(AnantError):
    """Base exception for validation errors."""
    pass


class SchemaValidationError(ValidationError):
    """Errors related to schema validation."""
    pass


class DataValidationError(ValidationError):
    """Errors related to data validation."""
    pass


class ConfigurationError(AnantError):
    """Errors related to configuration."""
    pass


class CachingError(AnantError):
    """Errors related to caching operations."""
    pass


class OptimizationError(AnantError):
    """Errors related to optimization operations."""
    pass


class GpuError(AnantError):
    """Errors related to GPU operations."""
    pass


# Convenience functions for common error patterns
def require_not_none(value: Any, name: str, context: Optional[Dict[str, Any]] = None) -> Any:
    """
    Validate that a value is not None.
    
    Args:
        value: Value to check
        name: Name of the value for error message
        context: Additional context for debugging
        
    Returns:
        The value if not None
        
    Raises:
        ValidationError: If value is None
    """
    if value is None:
        raise ValidationError(
            f"Required value '{name}' cannot be None",
            error_code="REQUIRED_VALUE_NONE",
            context=context or {}
        )
    return value


def require_valid_string(value: str, name: str, min_length: int = 1, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Validate that a string value is valid.
    
    Args:
        value: String to validate
        name: Name of the string for error message
        min_length: Minimum required length
        context: Additional context for debugging
        
    Returns:
        The string if valid
        
    Raises:
        ValidationError: If string is invalid
    """
    require_not_none(value, name, context)
    
    if not isinstance(value, str):
        raise ValidationError(
            f"Value '{name}' must be a string, got {type(value).__name__}",
            error_code="INVALID_STRING_TYPE",
            context=context or {}
        )
    
    if len(value.strip()) < min_length:
        raise ValidationError(
            f"Value '{name}' must be at least {min_length} characters long",
            error_code="STRING_TOO_SHORT",
            context=context or {}
        )
    
    return value


def require_valid_dict(value: Dict[str, Any], name: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate that a dictionary value is valid.
    
    Args:
        value: Dictionary to validate
        name: Name of the dictionary for error message
        context: Additional context for debugging
        
    Returns:
        The dictionary if valid
        
    Raises:
        ValidationError: If dictionary is invalid
    """
    require_not_none(value, name, context)
    
    if not isinstance(value, dict):
        raise ValidationError(
            f"Value '{name}' must be a dictionary, got {type(value).__name__}",
            error_code="INVALID_DICT_TYPE",
            context=context or {}
        )
    
    return value


def handle_exception(operation: str, exception: Exception, context: Optional[Dict[str, Any]] = None) -> AnantError:
    """
    Convert a generic exception to an appropriate Anant exception.
    
    Args:
        operation: Description of the operation that failed
        exception: Original exception
        context: Additional context for debugging
        
    Returns:
        Appropriate Anant exception
    """
    error_context = context or {}
    error_context["operation"] = operation
    error_context["original_exception"] = str(exception)
    
    # Map common exception types to Anant exceptions
    if isinstance(exception, (FileNotFoundError, PermissionError)):
        return IoError(
            f"I/O error during {operation}: {exception}",
            error_code="IO_OPERATION_FAILED",
            context=error_context,
            cause=exception
        )
    elif isinstance(exception, (ValueError, TypeError)):
        return ValidationError(
            f"Validation error during {operation}: {exception}",
            error_code="VALIDATION_FAILED",
            context=error_context,
            cause=exception
        )
    elif isinstance(exception, KeyError):
        return DataValidationError(
            f"Missing required data during {operation}: {exception}",
            error_code="MISSING_REQUIRED_DATA",
            context=error_context,
            cause=exception
        )
    elif isinstance(exception, ImportError):
        return ConfigurationError(
            f"Missing dependency for {operation}: {exception}",
            error_code="MISSING_DEPENDENCY",
            context=error_context,
            cause=exception
        )
    else:
        return AnantError(
            f"Unexpected error during {operation}: {exception}",
            error_code="UNEXPECTED_ERROR",
            context=error_context,
            cause=exception
        )