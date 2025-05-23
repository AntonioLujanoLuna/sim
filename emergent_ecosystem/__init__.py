"""
Emergent Intelligence Ecosystem - A Complex Adaptive System Simulation

This package implements a comprehensive simulation of emergent intelligence in
artificial life forms, featuring:

- Advanced cognitive architecture (attention, planning, metacognition)
- Dynamic social networks and communication evolution
- Cultural knowledge transmission and innovation
- Environmental co-evolution and memory
- Real-time emergence detection
- Comprehensive visualization and analysis

Main Components:
- EmergentIntelligenceSimulation: Core simulation engine
- Config: Configuration management system
- EnhancedIndividual: Advanced agent with cognitive capabilities
- MainVisualization: Comprehensive visualization system

Usage:
    from emergent_ecosystem import EmergentIntelligenceSimulation, Config
    
    config = Config()
    simulation = EmergentIntelligenceSimulation(config)
    
    # Run simulation
    for step in range(1000):
        simulation.update()
"""

__version__ = "1.0.0"
__author__ = "Emergent Intelligence Research Team"

# Core components
from .core.simulation import EmergentIntelligenceSimulation
from .core.individual import EnhancedIndividual
from .config import Config, DEMO_CONFIG, RESEARCH_CONFIG, PERFORMANCE_CONFIG

# Visualization
from .visualization.main_display import MainVisualization, create_visualization

# Analysis tools
from .analysis.emergence_detection import EmergenceDetector
from .analysis.statistics import StatisticsTracker

# Social systems
from .social.networks import SocialNetwork
from .social.communication import CommunicationSystem, LanguageEvolution
from .social.culture import CulturalEvolution

# Environment
from .environment.ecosystem import EnvironmentalMemory

# Cognitive modules
from .cognition.attention import AttentionModule
from .cognition.planning import PlanningModule
from .cognition.metacognition import MetacognitionModule

__all__ = [
    # Core
    'EmergentIntelligenceSimulation',
    'EnhancedIndividual',
    'Config',
    'DEMO_CONFIG',
    'RESEARCH_CONFIG', 
    'PERFORMANCE_CONFIG',
    
    # Visualization
    'MainVisualization',
    'create_visualization',
    
    # Analysis
    'EmergenceDetector',
    'StatisticsTracker',
    
    # Social
    'SocialNetwork',
    'CommunicationSystem',
    'LanguageEvolution',
    'CulturalEvolution',
    
    # Environment
    'EnvironmentalMemory',
    
    # Cognition
    'AttentionModule',
    'PlanningModule',
    'MetacognitionModule',
] 