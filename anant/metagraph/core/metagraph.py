"""
Core Metagraph Module (Refactored)
==================================

Clean, modular Metagraph class that delegates operations to specialized modules.
Reduced from 5,311 lines to ~500 lines through proper separation of concerns.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from datetime import datetime
import uuid
import logging

from ...exceptions import (
    MetagraphError, ValidationError, handle_exception,
    require_not_none, require_valid_string, require_valid_dict
)

# Import operation modules
from .operations.entity_operations import EntityOperations
from .operations.analytics_operations import AnalyticsOperations  
from .operations.governance_operations import GovernanceOperations
from .operations.export_import_operations import ExportImportOperations
from .operations.graph_operations import GraphOperations, GraphMetrics, PathResult

logger = logging.getLogger(__name__)


class Metagraph:
    """
    Refactored Metagraph class - delegates to specialized operation modules.
    
    This class maintains the same API as the original but with proper 
    separation of concerns and modular architecture.
    """
    
    def __init__(self, 
                 storage_path: str = "./metagraph_data",
                 metadata_store=None,
                 hierarchical_store=None,
                 policy_engine=None,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the Metagraph with modular operation handlers."""
        self.storage_path = Path(storage_path)
        self.metadata_store = metadata_store
        self.hierarchical_store = hierarchical_store
        self.policy_engine = policy_engine
        self.config = config or {}
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Initialize operation modules
        self._initialize_operations()
        
        # Graph state
        self.graph_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        
        self.logger.info(f"Metagraph initialized (refactored): {self.graph_id[:8]}")
    
    def _initialize_operations(self):
        """Initialize all operation modules."""
        try:
            self.entity_ops = EntityOperations(
                storage_path=str(self.storage_path / "entities"),
                metadata_store=self.metadata_store,
                hierarchical_store=self.hierarchical_store
            )
            
            self.analytics_ops = AnalyticsOperations(
                storage_path=str(self.storage_path / "analytics"),
                metadata_store=self.metadata_store,
                config=self.config.get("analytics", {})
            )
            
            self.governance_ops = GovernanceOperations(
                storage_path=str(self.storage_path / "governance"),
                metadata_store=self.metadata_store,
                policy_engine=self.policy_engine
            )
            
            self.export_import_ops = ExportImportOperations(
                storage_path=str(self.storage_path / "export_import"),
                metadata_store=self.metadata_store,
                chunk_size=self.config.get("chunk_size", 10000)
            )
            
            self.graph_ops = GraphOperations(
                metadata_store=self.metadata_store,
                performance_config=self.config.get("performance", {})
            )
            
        except Exception as e:
            raise handle_exception("initializing operation modules", e)
    
    # Entity Operations (delegated)
    def create_entity(self, entity_data: Dict[str, Any], validate: bool = True) -> str:
        return self.entity_ops.create_entity(entity_data, validate)
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.entity_ops.get_entity(entity_id)
    
    def update_entity(self, entity_id: str, updates: Dict[str, Any], validate: bool = True) -> bool:
        return self.entity_ops.update_entity(entity_id, updates, validate)
    
    def delete_entity(self, entity_id: str, cascade: bool = False) -> bool:
        return self.entity_ops.delete_entity(entity_id, cascade)
    
    def search_entities(self, query: Optional[str] = None, filters: Optional[Dict[str, Any]] = None,
                       limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        return self.entity_ops.search_entities(query, filters, limit, offset)
    
    def create_relationship(self, relationship_data: Dict[str, Any]) -> str:
        return self.entity_ops.create_relationship(relationship_data)
    
    def get_entity_relationships(self, entity_id: str, relationship_type: Optional[str] = None,
                               direction: str = "both") -> List[Dict[str, Any]]:
        return self.entity_ops.get_entity_relationships(entity_id, relationship_type, direction)
    
    # Analytics Operations (delegated)
    def calculate_centrality(self, algorithm: str = "betweenness", 
                           entity_filter: Optional[Dict[str, Any]] = None,
                           normalize: bool = True) -> Dict[str, float]:
        return self.analytics_ops.calculate_centrality(algorithm, entity_filter, normalize)
    
    def detect_communities(self, algorithm: str = "louvain", resolution: float = 1.0,
                          min_community_size: int = 3) -> Dict[str, Any]:
        return self.analytics_ops.detect_communities(algorithm, resolution, min_community_size)
    
    def analyze_temporal_patterns(self, time_property: str = "created_at",
                                granularity: str = "day", 
                                entity_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.analytics_ops.analyze_temporal_patterns(time_property, granularity, entity_filter)
    
    def detect_anomalies(self, features: List[str], algorithm: str = "isolation_forest",
                        contamination: float = 0.1) -> Dict[str, Any]:
        return self.analytics_ops.detect_anomalies(features, algorithm, contamination)
    
    def calculate_similarity(self, entity_id1: str, entity_id2: str,
                           algorithm: str = "structural", 
                           features: Optional[List[str]] = None) -> float:
        return self.analytics_ops.calculate_similarity(entity_id1, entity_id2, algorithm, features)
    
    # Governance Operations (delegated)
    def create_policy(self, policy_name: str, policy_rules: Dict[str, Any],
                     policy_type: str = "data_governance", 
                     enforcement_level: str = "strict") -> str:
        return self.governance_ops.create_policy(policy_name, policy_rules, policy_type, enforcement_level)
    
    def check_compliance(self, policy_id: str) -> Dict[str, Any]:
        return self.governance_ops.check_compliance(policy_id)
    
    def check_entity_access(self, entity_id: str, user_id: str, action: str,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.governance_ops.check_entity_access(entity_id, user_id, action, context)
    
    def audit_trail(self, entity_id: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        return self.governance_ops.audit_trail(entity_id, days)
    
    def data_quality_rules(self, entity_type: Optional[str] = None) -> Dict[str, Any]:
        return self.governance_ops.data_quality_rules(entity_type)
    
    # Export/Import Operations (delegated)
    def export_entities(self, output_path: str, format_type: str = "json",
                       entity_types: Optional[List[str]] = None,
                       filters: Optional[Dict[str, Any]] = None,
                       include_relationships: bool = True, compress: bool = False) -> Dict[str, Any]:
        return self.export_import_ops.export_entities(
            output_path, format_type, entity_types, filters, include_relationships, compress
        )
    
    def import_entities(self, input_path: str, format_type: Optional[str] = None,
                       validation_mode: str = "strict", update_existing: bool = False,
                       batch_size: Optional[int] = None) -> Dict[str, Any]:
        return self.export_import_ops.import_entities(
            input_path, format_type, validation_mode, update_existing, batch_size
        )
    
    # Graph Operations (delegated)
    def build_graph_from_entities(self, entity_filter: Optional[Dict[str, Any]] = None,
                                 include_properties: bool = True, directed: bool = True) -> Dict[str, Any]:
        return self.graph_ops.build_graph_from_entities(entity_filter, include_properties, directed)
    
    def find_shortest_path(self, source_id: str, target_id: str, algorithm: str = "dijkstra",
                          weight_property: Optional[str] = None, 
                          max_depth: Optional[int] = None) -> PathResult:
        return self.graph_ops.find_shortest_path(source_id, target_id, algorithm, weight_property, max_depth)
    
    def calculate_graph_metrics(self, entity_filter: Optional[Dict[str, Any]] = None,
                               include_advanced: bool = False) -> GraphMetrics:
        return self.graph_ops.calculate_graph_metrics(entity_filter, include_advanced)
    
    def extract_subgraph(self, node_ids: List[str], include_neighbors: bool = False,
                        neighbor_depth: int = 1, preserve_connectivity: bool = True) -> Dict[str, Any]:
        return self.graph_ops.extract_subgraph(node_ids, include_neighbors, neighbor_depth, preserve_connectivity)
    
    # High-level utility methods
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        try:
            entity_stats = self.entity_ops.get_entity_statistics()
            graph_metrics = self.graph_ops.calculate_graph_metrics()
            
            return {
                "graph_id": self.graph_id,
                "created_at": self.created_at.isoformat(),
                "entities": entity_stats,
                "graph_metrics": {
                    "node_count": graph_metrics.node_count,
                    "edge_count": graph_metrics.edge_count,
                    "density": graph_metrics.density,
                    "connected_components": graph_metrics.connected_components
                }
            }
        except Exception as e:
            raise handle_exception("getting statistics", e)
    
    def __str__(self) -> str:
        """String representation."""
        try:
            stats = self.get_statistics()
            return (
                f"Metagraph(id={self.graph_id[:8]}..., "
                f"entities={stats['entities'].get('total_entities', 0)})"
            )
        except Exception:
            return f"Metagraph(id={self.graph_id[:8]}...)"
    
    def __repr__(self) -> str:
        return self.__str__()
