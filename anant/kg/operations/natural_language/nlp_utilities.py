"""
NLP Utilities Operations

This module is part of the modular refactoring of natural_language.py using the delegation pattern.
It handles NLP component initialization, utility functions, and system configuration.

Core Operations:
- NLP component initialization
- Utility functions for text processing
- Configuration validation
- Performance monitoring
"""

import logging
from typing import Dict, List, Optional, Any

from ...natural_language_types import (
    QueryType, Intent, ConfidenceLevel, EntityMention, RelationMention,
    QueryInterpretation, ConversationContext
)

logger = logging.getLogger(__name__)


class NLPUtilitiesOperations:
    """
    Handles NLP utilities operations including:
    - NLP component initialization
    - Text processing utilities
    - Configuration validation
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_transformers = config.get('use_transformers', False)
        self.use_nltk = config.get('use_nltk', True)
        
        # Initialize components
        self._initialize_components()
        
        logger.info("NLPUtilitiesOperations initialized")
    
    def _initialize_components(self):
        """Initialize NLP components"""
        
        self.initialization_results = {
            'spacy_available': False,
            'transformers_available': False,
            'nltk_available': False,
            'models_loaded': [],
            'errors': []
        }
        
        # Try to initialize spaCy
        try:
            import spacy
            self.initialization_results['spacy_available'] = True
            logger.info("spaCy available")
        except ImportError:
            self.initialization_results['errors'].append("spaCy not available")
        
        # Try to initialize transformers
        if self.use_transformers:
            try:
                import transformers
                self.initialization_results['transformers_available'] = True
                logger.info("Transformers available")
            except ImportError:
                self.initialization_results['errors'].append("Transformers not available")
        
        # Try to initialize NLTK
        if self.use_nltk:
            try:
                import nltk
                self.initialization_results['nltk_available'] = True
                logger.info("NLTK available")
            except ImportError:
                self.initialization_results['errors'].append("NLTK not available")
    
    def initialize_nlp_components(self) -> Dict[str, Any]:
        """Initialize NLP components and return status"""
        return self.initialization_results
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for processing"""
        
        # Basic normalization
        normalized = text.strip().lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        
        # Simple keyword extraction
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
        
        words = text.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Return top 10 keywords
    
    def calculate_confidence(self, 
                           intent_score: float,
                           entity_score: float,
                           pattern_score: float) -> float:
        """Calculate overall confidence score"""
        
        # Weighted average of different confidence factors
        weights = {
            'intent': 0.4,
            'entity': 0.3,
            'pattern': 0.3
        }
        
        confidence = (
            intent_score * weights['intent'] +
            entity_score * weights['entity'] +
            pattern_score * weights['pattern']
        )
        
        return min(1.0, max(0.0, confidence))
    
    def get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate NLP configuration"""
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check required components
        if not any([
            self.initialization_results['spacy_available'],
            self.initialization_results['transformers_available'],
            self.initialization_results['nltk_available']
        ]):
            validation_results['errors'].append("No NLP libraries available")
            validation_results['valid'] = False
        
        # Check configuration values
        confidence_threshold = self.config.get('confidence_threshold', 0.3)
        if confidence_threshold > 0.9:
            validation_results['warnings'].append("Confidence threshold is very high - may reject valid queries")
        elif confidence_threshold < 0.1:
            validation_results['warnings'].append("Confidence threshold is very low - may accept invalid queries")
        
        cache_size = self.config.get('max_cache_size', 1000)
        if cache_size < 100:
            validation_results['recommendations'].append("Consider increasing cache size for better performance")
        
        return validation_results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        
        return {
            'initialization_results': self.initialization_results,
            'configuration': {
                'use_transformers': self.use_transformers,
                'use_nltk': self.use_nltk,
                'confidence_threshold': self.config.get('confidence_threshold', 0.3),
                'cache_size': self.config.get('max_cache_size', 1000)
            },
            'validation': self.validate_configuration()
        }
    
    def cleanup_resources(self):
        """Clean up NLP resources"""
        
        # Cleanup would go here for any models or resources
        logger.info("NLP resources cleaned up")