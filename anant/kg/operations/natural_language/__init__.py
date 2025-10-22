"""
Natural Language Operations Module

Specialized operation modules for the Natural Language Interface system.
These modules implement the delegation pattern to achieve modular design
and improved maintainability.
"""

from .nlp_query_processing import NLPQueryProcessing
from .entity_recognition import EntityRecognitionOperations
from .query_translation import QueryTranslationOperations
from .response_generation import ResponseGenerationOperations
from .context_management import ContextManagementOperations
from .nlp_utilities import NLPUtilitiesOperations

__all__ = [
    'NLPQueryProcessing',
    'EntityRecognitionOperations', 
    'QueryTranslationOperations',
    'ResponseGenerationOperations',
    'ContextManagementOperations',
    'NLPUtilitiesOperations'
]