"""
Natural Language Interface - Type Definitions

Shared type definitions and enums used across the natural language processing modules.
This file contains the core data structures that are imported by operation modules.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime


class QueryType(Enum):
    """Types of queries supported by the natural language interface"""
    ENTITY_SEARCH = "entity_search"
    RELATIONSHIP_SEARCH = "relationship_search"
    AGGREGATION = "aggregation"
    BOOLEAN = "boolean"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


class Intent(Enum):
    """User intent classification"""
    FIND = "find"
    COUNT = "count"
    DESCRIBE = "describe"
    LIST = "list"
    COMPARE = "compare"
    ASK = "ask"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    HELP = "help"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for interpretations"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class EntityMention:
    """Represents an entity mention in natural language text"""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float
    normalized_form: Optional[str] = None
    entity_uri: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationMention:
    """Represents a relation mention in natural language text"""
    text: str
    relation_type: str
    start_pos: int
    end_pos: int
    confidence: float
    normalized_form: Optional[str] = None
    relation_uri: Optional[str] = None
    subject: Optional[EntityMention] = None
    object: Optional[EntityMention] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryInterpretation:
    """Complete interpretation of a natural language query"""
    original_query: str
    intent: Intent
    query_type: QueryType
    confidence: float
    entities: List[EntityMention] = field(default_factory=list)
    relations: List[RelationMention] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    temporal_info: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ConversationContext:
    """Context for multi-turn conversations"""
    conversation_id: str
    user_id: Optional[str]
    created_at: datetime
    last_updated: datetime
    previous_queries: List[str] = field(default_factory=list)
    entity_context: Dict[str, Any] = field(default_factory=dict)
    conversation_state: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class QueryResult:
    """Result of processing a natural language query"""
    query: str
    interpretation: QueryInterpretation
    formal_query: Optional[str]
    execution_result: Optional[Any]
    response_text: str
    suggestions: List[str] = field(default_factory=list)
    clarifications: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    success: bool = True
    error: Optional[str] = None