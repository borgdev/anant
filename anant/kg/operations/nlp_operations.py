"""
NLP Operations for Knowledge Graph

Handles natural language processing operations including:
- Entity extraction from text
- Relationship extraction
- Named entity recognition
- Entity linking and disambiguation
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import re
from collections import defaultdict, Counter

from ...exceptions import KnowledgeGraphError
from ...utils.performance import performance_monitor

logger = logging.getLogger(__name__)


class NLPOperations:
    """
    NLP operations for knowledge graph
    
    Provides natural language processing capabilities for extracting
    entities and relationships from text data.
    """
    
    def __init__(self, knowledge_graph):
        """
        Initialize NLPOperations
        
        Parameters
        ----------
        knowledge_graph : KnowledgeGraph
            Parent knowledge graph instance
        """
        if knowledge_graph is None:
            raise KnowledgeGraphError("Knowledge graph instance cannot be None")
        self.kg = knowledge_graph
        self.logger = logger.getChild(self.__class__.__name__)
    
    @performance_monitor("kg_entity_extraction")
    def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract entities from text
        
        Parameters
        ----------
        text : str
            Text to extract entities from
        entity_types : Optional[List[str]]
            Filter by specific entity types
            
        Returns
        -------
        List[Dict[str, Any]]
            List of extracted entities with metadata
        """
        try:
            # Try advanced NLP libraries first
            try:
                return self._advanced_entity_extraction(text, entity_types)
            except ImportError:
                self.logger.info("Advanced NLP libraries not available, using simple extraction")
                return self._simple_entity_extraction(text, entity_types)
                
        except Exception as e:
            raise KnowledgeGraphError(f"Entity extraction failed: {e}")
    
    def _advanced_entity_extraction(self, text: str, entity_types: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Extract entities using advanced NLP libraries"""
        import spacy
        
        # Load spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback to basic model
            nlp = spacy.load("en_core_web_md")
        
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Filter by entity types if specified
            if entity_types and ent.label_ not in entity_types:
                continue
            
            entity_data = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 0.9,  # spaCy doesn't provide confidence scores
                'description': spacy.explain(ent.label_) or ent.label_
            }
            entities.append(entity_data)
        
        return entities
    
    def _simple_entity_extraction(self, text: str, entity_types: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Extract entities using simple pattern matching"""
        entities = []
        
        # Simple patterns for common entity types
        patterns = {
            'PERSON': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # FirstName LastName
                r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.? [A-Z][a-z]+\b'  # Title Name
            ],
            'ORG': [
                r'\b[A-Z][a-zA-Z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b',
                r'\b[A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+\b(?= is| was| has)'  # Company context
            ],
            'LOCATION': [
                r'\b[A-Z][a-z]+, [A-Z][A-Z]\b',  # City, State
                r'\b[A-Z][a-z]+ [A-Z][a-z]+(?= city| state| country)\b'
            ],
            'DATE': [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b'
            ]
        }
        
        for entity_type, type_patterns in patterns.items():
            # Skip if filtering by entity types
            if entity_types and entity_type not in entity_types:
                continue
            
            for pattern in type_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity_data = {
                        'text': match.group(),
                        'label': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.7,  # Lower confidence for simple matching
                        'description': entity_type.lower().replace('_', ' ')
                    }
                    entities.append(entity_data)
        
        return entities
    
    @performance_monitor("kg_relation_extraction")  
    def extract_relations(self, text: str, entity_pairs: Optional[List[Tuple[str, str]]] = None) -> List[Dict[str, Any]]:
        """
        Extract relationships from text
        
        Parameters
        ----------
        text : str
            Text to extract relationships from
        entity_pairs : Optional[List[Tuple[str, str]]]
            Specific entity pairs to look for relationships between
            
        Returns
        -------
        List[Dict[str, Any]]
            List of extracted relationships
        """
        try:
            # Try advanced relation extraction
            try:
                return self._advanced_relation_extraction(text, entity_pairs)
            except ImportError:
                return self._simple_relation_extraction(text, entity_pairs)
                
        except Exception as e:
            raise KnowledgeGraphError(f"Relation extraction failed: {e}")
    
    def _advanced_relation_extraction(self, text: str, entity_pairs: Optional[List[Tuple[str, str]]]) -> List[Dict[str, Any]]:
        """Extract relationships using advanced NLP"""
        import spacy
        
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        relations = []
        
        # Extract subject-verb-object patterns
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    # Look for subject
                    subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                    # Look for objects
                    objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr")]
                    
                    for subj in subjects:
                        for obj in objects:
                            # Filter by entity pairs if specified
                            if entity_pairs:
                                subj_text = subj.text
                                obj_text = obj.text
                                if not any((subj_text == s and obj_text == o) or 
                                         (subj_text == o and obj_text == s) 
                                         for s, o in entity_pairs):
                                    continue
                            
                            relation_data = {
                                'subject': subj.text,
                                'predicate': token.lemma_,
                                'object': obj.text,
                                'confidence': 0.8,
                                'sentence': sent.text.strip(),
                                'subject_start': subj.idx,
                                'subject_end': subj.idx + len(subj.text),
                                'object_start': obj.idx,
                                'object_end': obj.idx + len(obj.text)
                            }
                            relations.append(relation_data)
        
        return relations
    
    def _simple_relation_extraction(self, text: str, entity_pairs: Optional[List[Tuple[str, str]]]) -> List[Dict[str, Any]]:
        """Extract relationships using simple patterns"""
        relations = []
        
        # Simple relation patterns
        relation_patterns = [
            r'(\w+)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*(\w+)',  # X is Y
            r'(\w+)\s+(?:has|have|had)\s+(?:a|an|the)?\s*(\w+)',     # X has Y
            r'(\w+)\s+(?:works|worked)\s+(?:at|for|with)\s+(\w+)',   # X works at Y
            r'(\w+)\s+(?:owns|owned)\s+(\w+)',                       # X owns Y
            r'(\w+)\s+(?:created|founded|established)\s+(\w+)'       # X created Y
        ]
        
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            for pattern in relation_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    subject, obj = match.groups()
                    
                    # Filter by entity pairs if specified
                    if entity_pairs:
                        if not any((subject.lower() == s.lower() and obj.lower() == o.lower()) or
                                 (subject.lower() == o.lower() and obj.lower() == s.lower())
                                 for s, o in entity_pairs):
                            continue
                    
                    # Extract predicate from pattern
                    predicate = "relates_to"  # Default relation
                    if "is" in match.group().lower():
                        predicate = "is_a"
                    elif "has" in match.group().lower():
                        predicate = "has"
                    elif "works" in match.group().lower():
                        predicate = "works_at"
                    elif "owns" in match.group().lower():
                        predicate = "owns"
                    elif "created" in match.group().lower():
                        predicate = "created"
                    
                    relation_data = {
                        'subject': subject,
                        'predicate': predicate,
                        'object': obj,
                        'confidence': 0.6,
                        'sentence': sentence,
                        'subject_start': match.start(1),
                        'subject_end': match.end(1),
                        'object_start': match.start(2),
                        'object_end': match.end(2)
                    }
                    relations.append(relation_data)
        
        return relations
    
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract subject-predicate-object triples from text
        
        Parameters
        ----------
        text : str
            Text to extract triples from
            
        Returns
        -------
        List[Tuple[str, str, str]]
            List of (subject, predicate, object) triples
        """
        try:
            relations = self.extract_relations(text)
            triples = []
            
            for rel in relations:
                if all(key in rel for key in ['subject', 'predicate', 'object']):
                    triple = (rel['subject'], rel['predicate'], rel['object'])
                    triples.append(triple)
            
            return triples
            
        except Exception as e:
            self.logger.warning(f"Error extracting triples: {e}")
            return []
    
    def extract_concepts(self, text: str, min_frequency: int = 2) -> List[Dict[str, Any]]:
        """
        Extract key concepts from text
        
        Parameters
        ----------
        text : str
            Text to extract concepts from
        min_frequency : int, default 2
            Minimum frequency for concept inclusion
            
        Returns
        -------
        List[Dict[str, Any]]
            List of extracted concepts with frequency and context
        """
        try:
            # Tokenize and clean text
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Remove common stop words
            stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'this', 'that', 'these', 'those', 'a', 'an', 'as', 'it', 'its', 'can', 'will'
            }
            
            filtered_words = [word for word in words if word not in stop_words]
            
            # Count word frequencies
            word_counts = Counter(filtered_words)
            
            # Extract concepts above minimum frequency
            concepts = []
            for word, count in word_counts.items():
                if count >= min_frequency:
                    # Find contexts where this concept appears
                    contexts = []
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if word in sentence.lower():
                            contexts.append(sentence.strip())
                    
                    concept_data = {
                        'concept': word,
                        'frequency': count,
                        'contexts': contexts[:3],  # Limit to first 3 contexts
                        'type': 'CONCEPT'
                    }
                    concepts.append(concept_data)
            
            # Sort by frequency
            concepts.sort(key=lambda x: x['frequency'], reverse=True)
            
            return concepts
            
        except Exception as e:
            self.logger.warning(f"Error extracting concepts: {e}")
            return []
    
    def named_entity_recognition(self, text: str) -> Dict[str, List[str]]:
        """
        Perform named entity recognition on text
        
        Parameters
        ----------
        text : str
            Text for NER processing
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping entity types to lists of entities
        """
        try:
            entities = self.extract_entities(text)
            
            # Group entities by type
            entity_groups = defaultdict(list)
            seen_entities = set()
            
            for entity in entities:
                entity_text = entity['text']
                entity_type = entity['label']
                
                # Avoid duplicates
                if (entity_text, entity_type) not in seen_entities:
                    entity_groups[entity_type].append(entity_text)
                    seen_entities.add((entity_text, entity_type))
            
            return dict(entity_groups)
            
        except Exception as e:
            self.logger.warning(f"Error in named entity recognition: {e}")
            return {}
    
    def entity_linking(self, entities: List[str], knowledge_base: Optional[Dict] = None) -> Dict[str, List[str]]:
        """
        Link extracted entities to knowledge base entries
        
        Parameters
        ----------
        entities : List[str]
            List of entity strings to link
        knowledge_base : Optional[Dict]
            External knowledge base for linking (if None, uses internal KG)
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping entities to possible KB identifiers
        """
        try:
            linked_entities = {}
            
            # Use internal knowledge graph if no external KB provided
            if knowledge_base is None:
                kg_entities = set(self.kg.nodes)
            else:
                kg_entities = set(knowledge_base.keys())
            
            for entity in entities:
                candidates = []
                entity_lower = entity.lower()
                
                # Exact match
                for kg_entity in kg_entities:
                    if entity_lower == kg_entity.lower():
                        candidates.append(kg_entity)
                
                # Partial match if no exact match
                if not candidates:
                    for kg_entity in kg_entities:
                        if (entity_lower in kg_entity.lower() or 
                            kg_entity.lower() in entity_lower):
                            candidates.append(kg_entity)
                
                # String similarity matching (simplified)
                if not candidates:
                    for kg_entity in kg_entities:
                        similarity = self._simple_string_similarity(entity_lower, kg_entity.lower())
                        if similarity > 0.7:
                            candidates.append(kg_entity)
                
                linked_entities[entity] = candidates[:5]  # Limit to top 5 candidates
            
            return linked_entities
            
        except Exception as e:
            self.logger.warning(f"Error in entity linking: {e}")
            return {}
    
    def _simple_string_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple string similarity"""
        # Jaccard similarity based on character n-grams
        def get_ngrams(s: str, n: int = 2) -> set:
            return set(s[i:i+n] for i in range(len(s) - n + 1))
        
        ngrams1 = get_ngrams(s1)
        ngrams2 = get_ngrams(s2)
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0