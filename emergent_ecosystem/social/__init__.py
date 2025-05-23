"""
Social dynamics and communication systems.

This module contains social networks, communication evolution, and cultural transmission systems.
"""

from .networks import SocialNetwork, SocialRelationship
from .communication import CommunicationSystem, LanguageEvolution, Signal
from .culture import CulturalEvolution, CulturalSystem, CulturalKnowledge

__all__ = [
    'SocialNetwork', 'SocialRelationship',
    'CommunicationSystem', 'LanguageEvolution', 'Signal',
    'CulturalEvolution', 'CulturalSystem', 'CulturalKnowledge'
]
