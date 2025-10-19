"""
Entity Resolution and Linking Module
===================================

Advanced entity resolution, duplicate detection, and cross-dataset linking
for knowledge graphs. Completely domain-agnostic and extensible.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from collections import defaultdict, Counter
from dataclasses import dataclass
import difflib
import hashlib

from ..utils.performance import performance_monitor, PerformanceProfiler
from ..algorithms.sampling import SmartSampler
from ..utils.extras import safe_import

# Optional dependencies for advanced similarity
nltk = safe_import('nltk')
sklearn = safe_import('sklearn')
numpy = safe_import('numpy')

logger = logging.getLogger(__name__)


@dataclass
class EntityMatch:
    """Represents a potential entity match"""
    entity1: str
    entity2: str
    confidence: float
    similarity_type: str
    evidence: Dict[str, Any]
    match_method: str


@dataclass
class EntityCluster:
    """Represents a cluster of duplicate entities"""
    cluster_id: str
    entities: List[str]
    canonical_entity: str
    confidence: float
    cluster_size: int


@dataclass
class LinkingResult:
    """Result of cross-dataset entity linking"""
    source_entity: str
    target_entities: List[str]
    confidence_scores: List[float]
    linking_evidence: List[Dict[str, Any]]


class EntityResolver:
    """
    Comprehensive entity resolution and duplicate detection
    
    Features:
    - Multiple similarity algorithms (string, semantic, structural)
    - Configurable matching thresholds
    - Clustering of duplicate entities
    - Cross-dataset entity linking
    - Performance optimization for large graphs
    """
    
    def __init__(self, knowledge_graph):
        """
        Initialize entity resolver
        
        Args:
            knowledge_graph: KnowledgeGraph instance to process
        """
        self.kg = knowledge_graph
        
        # Configuration
        self.config = {
            'string_similarity_threshold': 0.8,
            'semantic_similarity_threshold': 0.7,
            'structural_similarity_threshold': 0.6,
            'cluster_threshold': 0.75,
            'max_candidates': 1000,
            'use_sampling': True,
            'sampling_threshold': 10000
        }
        
        # Similarity algorithms
        self.similarity_algorithms = {
            'levenshtein': self._levenshtein_similarity,
            'jaccard': self._jaccard_similarity,
            'cosine': self._cosine_similarity,
            'soundex': self._soundex_similarity,
            'structural': self._structural_similarity
        }
        
        # Caches for performance
        self._similarity_cache = {}
        self._entity_features_cache = {}
        self._neighbor_cache = {}
        
        logger.info("Entity Resolver initialized")
    
    @performance_monitor("entity_duplicate_detection")
    def find_duplicates(self, 
                       entity_types: Optional[List[str]] = None,
                       similarity_methods: Optional[List[str]] = None,
                       min_confidence: float = 0.7) -> List[EntityMatch]:
        """
        Find duplicate entities in the knowledge graph
        
        Args:
            entity_types: Filter by specific entity types
            similarity_methods: Similarity algorithms to use
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of potential duplicate matches
        """
        
        if similarity_methods is None:
            similarity_methods = ['levenshtein', 'jaccard', 'structural']
        
        logger.info(f"Finding duplicates with methods: {similarity_methods}")
        
        # Get candidate entities
        candidates = self._get_duplicate_candidates(entity_types)
        
        # Use sampling for large datasets
        if self.config['use_sampling'] and len(candidates) > self.config['sampling_threshold']:
            logger.info(f"Sampling {self.config['max_candidates']} candidates from {len(candidates)}")
            candidates = candidates[:self.config['max_candidates']]
        
        matches = []
        
        with PerformanceProfiler("duplicate_detection") as profiler:
            
            profiler.checkpoint("start_comparison")
            
            # Compare all pairs of candidates
            for i, entity1 in enumerate(candidates):
                for j, entity2 in enumerate(candidates[i+1:], i+1):
                    
                    # Calculate combined similarity
                    combined_confidence = self._calculate_combined_similarity(
                        entity1, entity2, similarity_methods
                    )
                    
                    if combined_confidence >= min_confidence:
                        match = EntityMatch(
                            entity1=entity1,
                            entity2=entity2,
                            confidence=combined_confidence,
                            similarity_type='combined',
                            evidence=self._get_match_evidence(entity1, entity2),
                            match_method='+'.join(similarity_methods)
                        )
                        matches.append(match)
            
            profiler.checkpoint("comparison_complete")
        
        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        report = profiler.get_report()
        logger.info(f"Found {len(matches)} duplicate matches in {report['total_execution_time']:.2f}s")
        
        return matches
    
    def _get_duplicate_candidates(self, entity_types: Optional[List[str]] = None) -> List[str]:
        """Get candidate entities for duplicate detection"""
        
        if entity_types:
            candidates = []
            for entity_type in entity_types:
                entities = self.kg.get_entities_by_type(entity_type)
                candidates.extend(list(entities))
        else:
            candidates = list(self.kg.nodes)
        
        return candidates
    
    def _calculate_combined_similarity(self, entity1: str, entity2: str, 
                                     methods: List[str]) -> float:
        """Calculate combined similarity score from multiple methods"""
        
        cache_key = f"{entity1}|{entity2}|{'|'.join(sorted(methods))}"
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        similarities = []
        
        for method in methods:
            if method in self.similarity_algorithms:
                similarity = self.similarity_algorithms[method](entity1, entity2)
                similarities.append(similarity)
        
        # Weighted average (could be made configurable)
        if similarities:
            combined = sum(similarities) / len(similarities)
        else:
            combined = 0.0
        
        self._similarity_cache[cache_key] = combined
        return combined
    
    def _levenshtein_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate Levenshtein string similarity"""
        
        # Normalize entities for comparison
        norm1 = self._normalize_entity_name(entity1)
        norm2 = self._normalize_entity_name(entity2)
        
        if norm1 == norm2:
            return 1.0
        
        # Use difflib for similarity ratio
        similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        
        return similarity
    
    def _jaccard_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate Jaccard similarity based on token sets"""
        
        tokens1 = set(self._tokenize_entity(entity1))
        tokens2 = set(self._tokenize_entity(entity2))
        
        if not tokens1 and not tokens2:
            return 1.0
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def _cosine_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate cosine similarity based on character n-grams"""
        
        if sklearn is None or numpy is None:
            # Fallback to Jaccard if sklearn not available
            return self._jaccard_similarity(entity1, entity2)
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create character n-grams
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
            
            norm1 = self._normalize_entity_name(entity1)
            norm2 = self._normalize_entity_name(entity2)
            
            if not norm1 or not norm2:
                return 0.0
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform([norm1, norm2])
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix[0, 1]
            
        except Exception:
            # Fallback to Jaccard on any error
            return self._jaccard_similarity(entity1, entity2)
    
    def _soundex_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate phonetic similarity using Soundex algorithm"""
        
        def soundex(name: str) -> str:
            """Simple Soundex implementation"""
            if not name:
                return "0000"
            
            # Keep first letter, convert to uppercase
            name = name.upper()
            soundex_code = name[0]
            
            # Mapping for consonants
            mapping = {
                'B': '1', 'F': '1', 'P': '1', 'V': '1',
                'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
                'D': '3', 'T': '3',
                'L': '4',
                'M': '5', 'N': '5',
                'R': '6'
            }
            
            # Convert consonants
            for char in name[1:]:
                if char in mapping:
                    code = mapping[char]
                    if soundex_code[-1] != code:  # Avoid consecutive duplicates
                        soundex_code += code
                        
                if len(soundex_code) >= 4:
                    break
            
            # Pad with zeros
            soundex_code = (soundex_code + "0000")[:4]
            
            return soundex_code
        
        # Extract meaningful parts from URIs
        name1 = self._extract_entity_name(entity1)
        name2 = self._extract_entity_name(entity2)
        
        soundex1 = soundex(name1)
        soundex2 = soundex(name2)
        
        return 1.0 if soundex1 == soundex2 else 0.0
    
    def _structural_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate structural similarity based on graph neighborhood"""
        
        # Get neighbors for both entities
        neighbors1 = self._get_entity_neighbors(entity1)
        neighbors2 = self._get_entity_neighbors(entity2)
        
        if not neighbors1 and not neighbors2:
            return 1.0
        
        if not neighbors1 or not neighbors2:
            return 0.0
        
        # Calculate Jaccard similarity of neighbor sets
        intersection = neighbors1.intersection(neighbors2)
        union = neighbors1.union(neighbors2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _normalize_entity_name(self, entity: str) -> str:
        """Normalize entity name for comparison"""
        
        # Extract meaningful part from URI
        name = self._extract_entity_name(entity)
        
        # Convert to lowercase
        name = name.lower()
        
        # Remove common prefixes/suffixes
        prefixes = ['has', 'is', 'get', 'set', 'the']
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):].lstrip('_')
                break
        
        # Remove special characters
        name = re.sub(r'[^\w\s]', '', name)
        
        # Normalize whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def _extract_entity_name(self, entity: str) -> str:
        """Extract meaningful name from entity URI or identifier"""
        
        # Handle URIs
        if '/' in entity or '#' in entity:
            separator = '#' if '#' in entity else '/'
            name = entity.split(separator)[-1]
        else:
            name = entity
        
        # Handle prefixed names
        if ':' in name and not name.startswith('http'):
            name = name.split(':', 1)[1]
        
        return name
    
    def _tokenize_entity(self, entity: str) -> List[str]:
        """Tokenize entity name into meaningful tokens"""
        
        name = self._normalize_entity_name(entity)
        
        # Split on common separators
        tokens = re.split(r'[\s_\-\.]+', name)
        
        # Remove empty tokens
        tokens = [token for token in tokens if token]
        
        # Add camelCase splitting
        expanded_tokens = []
        for token in tokens:
            # Split camelCase
            camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', token)
            expanded_tokens.extend(camel_parts if camel_parts else [token])
        
        return [token.lower() for token in expanded_tokens if len(token) > 1]
    
    def _get_entity_neighbors(self, entity: str) -> Set[str]:
        """Get neighboring entities in the graph"""
        
        if entity in self._neighbor_cache:
            return self._neighbor_cache[entity]
        
        neighbors = set()
        
        if entity in self.kg.nodes:
            # Get incident edges
            incident_edges = self.kg.incidences.get_node_edges(entity)
            
            for edge in incident_edges:
                edge_nodes = self.kg.incidences.get_edge_nodes(edge)
                for neighbor in edge_nodes:
                    if neighbor != entity:
                        neighbors.add(neighbor)
        
        self._neighbor_cache[entity] = neighbors
        return neighbors
    
    def _get_match_evidence(self, entity1: str, entity2: str) -> Dict[str, Any]:
        """Collect evidence for why two entities might be duplicates"""
        
        evidence = {
            'name_similarity': self._levenshtein_similarity(entity1, entity2),
            'token_overlap': self._jaccard_similarity(entity1, entity2),
            'structural_similarity': self._structural_similarity(entity1, entity2),
            'same_type': self.kg.get_entity_type(entity1) == self.kg.get_entity_type(entity2),
            'shared_neighbors': len(self._get_entity_neighbors(entity1).intersection(
                                   self._get_entity_neighbors(entity2)))
        }
        
        return evidence
    
    @performance_monitor("entity_clustering") 
    def cluster_duplicates(self, matches: List[EntityMatch], 
                          min_cluster_size: int = 2) -> List[EntityCluster]:
        """
        Cluster duplicate entities into groups
        
        Args:
            matches: List of entity matches
            min_cluster_size: Minimum entities per cluster
            
        Returns:
            List of entity clusters
        """
        
        logger.info(f"Clustering {len(matches)} matches")
        
        # Build similarity graph
        similarity_graph = defaultdict(list)
        
        for match in matches:
            if match.confidence >= self.config['cluster_threshold']:
                similarity_graph[match.entity1].append((match.entity2, match.confidence))
                similarity_graph[match.entity2].append((match.entity1, match.confidence))
        
        # Find connected components
        visited = set()
        clusters = []
        cluster_id = 0
        
        for entity in similarity_graph:
            if entity not in visited:
                cluster_entities = []
                self._dfs_cluster(entity, similarity_graph, visited, cluster_entities)
                
                if len(cluster_entities) >= min_cluster_size:
                    # Choose canonical entity (most connected)
                    canonical = max(cluster_entities, 
                                  key=lambda e: len(similarity_graph[e]))
                    
                    # Calculate average confidence
                    confidences = []
                    for ent in cluster_entities:
                        for neighbor, conf in similarity_graph[ent]:
                            if neighbor in cluster_entities:
                                confidences.append(conf)
                    
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    cluster = EntityCluster(
                        cluster_id=f"cluster_{cluster_id}",
                        entities=cluster_entities,
                        canonical_entity=canonical,
                        confidence=avg_confidence,
                        cluster_size=len(cluster_entities)
                    )
                    
                    clusters.append(cluster)
                    cluster_id += 1
        
        logger.info(f"Created {len(clusters)} entity clusters")
        
        return clusters
    
    def _dfs_cluster(self, entity: str, graph: Dict, visited: Set[str], cluster: List[str]):
        """Depth-first search to find cluster members"""
        
        if entity in visited:
            return
        
        visited.add(entity)
        cluster.append(entity)
        
        for neighbor, confidence in graph[entity]:
            if neighbor not in visited and confidence >= self.config['cluster_threshold']:
                self._dfs_cluster(neighbor, graph, visited, cluster)
    
    @performance_monitor("cross_dataset_linking")
    def link_entities_cross_dataset(self, 
                                   target_entities: List[str],
                                   similarity_threshold: float = 0.6) -> Dict[str, LinkingResult]:
        """
        Link entities from the knowledge graph to external entities
        
        Args:
            target_entities: External entities to link to
            similarity_threshold: Minimum similarity for linking
            
        Returns:
            Dictionary mapping source entities to linking results
        """
        
        logger.info(f"Linking to {len(target_entities)} external entities")
        
        linking_results = {}
        
        # Get source entities (sample if too large)
        source_entities = list(self.kg.nodes)
        if len(source_entities) > self.config['max_candidates']:
            source_entities = source_entities[:self.config['max_candidates']]
        
        for source_entity in source_entities:
            
            # Find best matches with target entities
            matches = []
            
            for target_entity in target_entities:
                similarity = self._calculate_combined_similarity(
                    source_entity, target_entity, 
                    ['levenshtein', 'jaccard']
                )
                
                if similarity >= similarity_threshold:
                    matches.append((target_entity, similarity))
            
            if matches:
                # Sort by similarity
                matches.sort(key=lambda x: x[1], reverse=True)
                
                target_entities_matched = [match[0] for match in matches]
                confidence_scores = [match[1] for match in matches]
                linking_evidence = [
                    self._get_match_evidence(source_entity, target)
                    for target, _ in matches
                ]
                
                linking_results[source_entity] = LinkingResult(
                    source_entity=source_entity,
                    target_entities=target_entities_matched,
                    confidence_scores=confidence_scores,
                    linking_evidence=linking_evidence
                )
        
        logger.info(f"Successfully linked {len(linking_results)} entities")
        
        return linking_results
    
    def get_resolution_statistics(self) -> Dict[str, Any]:
        """Get entity resolution statistics and metrics"""
        
        return {
            'total_entities': len(self.kg.nodes),
            'entity_types': len(self.kg._entity_types),
            'cache_statistics': {
                'similarity_cache_size': len(self._similarity_cache),
                'neighbor_cache_size': len(self._neighbor_cache),
                'features_cache_size': len(self._entity_features_cache)
            },
            'configuration': self.config,
            'available_algorithms': list(self.similarity_algorithms.keys())
        }


class EntityLinker:
    """
    Specialized entity linking for cross-dataset integration
    """
    
    def __init__(self, source_kg, target_kg=None):
        """
        Initialize entity linker
        
        Args:
            source_kg: Source knowledge graph
            target_kg: Target knowledge graph (optional)
        """
        self.source_kg = source_kg
        self.target_kg = target_kg
        
        # Initialize resolvers
        self.source_resolver = EntityResolver(source_kg)
        self.target_resolver = EntityResolver(target_kg) if target_kg else None
    
    def find_cross_graph_links(self, similarity_threshold: float = 0.7) -> List[EntityMatch]:
        """
        Find entity links between two knowledge graphs
        
        Args:
            similarity_threshold: Minimum similarity for linking
            
        Returns:
            List of cross-graph entity matches
        """
        
        if not self.target_kg:
            raise ValueError("Target knowledge graph required for cross-graph linking")
        
        matches = []
        
        # Sample entities for performance
        source_entities = list(self.source_kg.nodes)[:1000]
        target_entities = list(self.target_kg.nodes)[:1000]
        
        logger.info(f"Cross-linking {len(source_entities)} source entities with {len(target_entities)} target entities")
        
        for source_entity in source_entities:
            for target_entity in target_entities:
                
                # Calculate similarity using source resolver
                similarity = self.source_resolver._calculate_combined_similarity(
                    source_entity, target_entity,
                    ['levenshtein', 'jaccard']
                )
                
                if similarity >= similarity_threshold:
                    match = EntityMatch(
                        entity1=source_entity,
                        entity2=target_entity,
                        confidence=similarity,
                        similarity_type='cross_graph',
                        evidence=self.source_resolver._get_match_evidence(source_entity, target_entity),
                        match_method='cross_graph_linking'
                    )
                    matches.append(match)
        
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Found {len(matches)} cross-graph links")
        
        return matches