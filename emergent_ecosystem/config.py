"""
Configuration management system for the Emergent Intelligence Ecosystem.

This module provides centralized configuration for all simulation parameters,
organized into logical groups for easy management and experimentation.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SimulationConfig:
    """Core simulation parameters"""
    width: int = 1200
    height: int = 900
    max_population: int = 300
    initial_population: int = 80
    time_step: float = 0.1
    max_frames: int = 5000
    animation_interval: int = 50


@dataclass
class EnvironmentConfig:
    """Environmental system parameters"""
    patch_density: float = 0.001
    resource_regeneration_rate: float = 0.1
    environmental_memory_length: int = 500
    co_evolution_strength: float = 0.2
    grid_size: int = 50


@dataclass
class PhysicsConfig:
    """Physical interaction parameters"""
    separation_radius: float = 35
    alignment_radius: float = 70
    cohesion_radius: float = 90
    communication_radius: float = 120
    separation_force: float = 2.5
    alignment_force: float = 1.0
    cohesion_force: float = 0.8
    social_force: float = 1.2
    environmental_force: float = 0.6


@dataclass
class CognitiveConfig:
    """Cognitive architecture parameters"""
    memory_length: int = 100
    planning_horizon: int = 20
    attention_span: int = 5
    learning_rate: float = 0.1
    intelligence_min: float = 0.2
    intelligence_max: float = 0.8
    max_speed_min: float = 1.5
    max_speed_max: float = 3.5


@dataclass
class SocialConfig:
    """Social dynamics and cultural parameters"""
    max_social_connections: int = 8
    relationship_decay: float = 0.02
    trust_threshold: float = 0.6
    communication_mutation_rate: float = 0.05
    cultural_inheritance_rate: float = 0.3
    innovation_tendency_min: float = 0.0
    innovation_tendency_max: float = 1.0
    teaching_ability_min: float = 0.0
    teaching_ability_max: float = 1.0


@dataclass
class EvolutionConfig:
    """Evolution and breeding parameters"""
    breeding_energy_threshold: float = 70
    mutation_rate: float = 0.12
    sexual_selection_strength: float = 0.4
    breeding_probability: float = 0.003
    breeding_cost: float = 30
    max_ages: Dict[str, int] = None
    
    def __post_init__(self):
        if self.max_ages is None:
            self.max_ages = {
                'predator': 1000,
                'herbivore': 800,
                'scavenger': 600,
                'mystic': 1200
            }


@dataclass
class VisualizationConfig:
    """Visualization and display parameters"""
    figure_size: tuple = (18, 12)
    background_color: str = 'black'
    text_color: str = 'white'
    trail_length: int = 40
    species_colors: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.species_colors is None:
            self.species_colors = {
                'predator': (0.8, 0.2, 0.2),
                'herbivore': (0.2, 0.8, 0.2),
                'scavenger': (0.8, 0.8, 0.2),
                'mystic': (0.6, 0.2, 0.8)
            }

@dataclass
class Config:
    """Main configuration container that combines all subsystem configs"""
    simulation: SimulationConfig = None
    environment: EnvironmentConfig = None
    physics: PhysicsConfig = None
    cognitive: CognitiveConfig = None
    social: SocialConfig = None
    evolution: EvolutionConfig = None
    visualization: VisualizationConfig = None
    
    def __post_init__(self):
        # Initialize all subconfigs if not provided
        if self.simulation is None:
            self.simulation = SimulationConfig()
        if self.environment is None:
            self.environment = EnvironmentConfig()
        if self.physics is None:
            self.physics = PhysicsConfig()
        if self.cognitive is None:
            self.cognitive = CognitiveConfig()
        if self.social is None:
            self.social = SocialConfig()
        if self.evolution is None:
            self.evolution = EvolutionConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
    
    @property
    def species_configs(self) -> Dict[str, Any]:
        """Species-specific configuration parameters"""
        return {
            'predator': {'max_age': 1000, 'base_aggression': 0.8},
            'herbivore': {'max_age': 800, 'base_aggression': 0.2},
            'scavenger': {'max_age': 600, 'base_aggression': 0.4},
            'mystic': {'max_age': 1200, 'base_aggression': 0.1}
        }
    
    # Legacy compatibility - provide direct access to commonly used parameters
    @property
    def width(self) -> int:
        return getattr(self.simulation, 'width', 1200)  # Fallback if not initialized
    
    @property
    def height(self) -> int:
        return getattr(self.simulation, 'height', 900)  # Fallback if not initialized
    
    @property
    def max_population(self) -> int:
        return getattr(self.simulation, 'max_population', 300)  # Fallback if not initialized
    
    @property
    def initial_population(self) -> int:
        return getattr(self.simulation, 'initial_population', 80)  # Fallback if not initialized
    
    @property
    def time_step(self) -> float:
        return getattr(self.simulation, 'time_step', 0.1)  # Fallback if not initialized
    
    @property
    def separation_radius(self) -> float:
        return getattr(self.physics, 'separation_radius', 35)  # Fallback if not initialized
    
    @property
    def alignment_radius(self) -> float:
        return getattr(self.physics, 'alignment_radius', 70)  # Fallback if not initialized
    
    @property
    def cohesion_radius(self) -> float:
        return getattr(self.physics, 'cohesion_radius', 90)  # Fallback if not initialized
    
    @property
    def communication_radius(self) -> float:
        return getattr(self.physics, 'communication_radius', 120)  # Fallback if not initialized
    
    @property
    def separation_force(self) -> float:
        return getattr(self.physics, 'separation_force', 2.5)  # Fallback if not initialized
    
    @property
    def alignment_force(self) -> float:
        return getattr(self.physics, 'alignment_force', 1.0)  # Fallback if not initialized
    
    @property
    def cohesion_force(self) -> float:
        return getattr(self.physics, 'cohesion_force', 0.8)  # Fallback if not initialized
    
    @property
    def social_force(self) -> float:
        return getattr(self.physics, 'social_force', 1.2)  # Fallback if not initialized
    
    @property
    def environmental_force(self) -> float:
        return getattr(self.physics, 'environmental_force', 0.6)  # Fallback if not initialized
    
    @property
    def memory_length(self) -> int:
        return getattr(self.cognitive, 'memory_length', 100)  # Fallback if not initialized
    
    @property
    def planning_horizon(self) -> int:
        return getattr(self.cognitive, 'planning_horizon', 20)  # Fallback if not initialized
    
    @property
    def attention_span(self) -> int:
        return getattr(self.cognitive, 'attention_span', 5)  # Fallback if not initialized
    
    @property
    def learning_rate(self) -> float:
        return getattr(self.cognitive, 'learning_rate', 0.1)  # Fallback if not initialized
    
    @property
    def max_social_connections(self) -> int:
        return getattr(self.social, 'max_social_connections', 8)  # Fallback if not initialized
    
    @property
    def relationship_decay(self) -> float:
        return getattr(self.social, 'relationship_decay', 0.02)  # Fallback if not initialized
    
    @property
    def trust_threshold(self) -> float:
        return getattr(self.social, 'trust_threshold', 0.6)  # Fallback if not initialized
    
    @property
    def communication_mutation_rate(self) -> float:
        return getattr(self.social, 'communication_mutation_rate', 0.05)  # Fallback if not initialized
    
    @property
    def breeding_energy_threshold(self) -> float:
        return getattr(self.evolution, 'breeding_energy_threshold', 70)  # Fallback if not initialized
    
    @property
    def mutation_rate(self) -> float:
        return getattr(self.evolution, 'mutation_rate', 0.12)  # Fallback if not initialized
    
    @property
    def sexual_selection_strength(self) -> float:
        return getattr(self.evolution, 'sexual_selection_strength', 0.4)  # Fallback if not initialized
    
    @property
    def cultural_inheritance_rate(self) -> float:
        return getattr(self.social, 'cultural_inheritance_rate', 0.3)  # Fallback if not initialized
    
    @property
    def resource_regeneration_rate(self) -> float:
        return getattr(self.environment, 'resource_regeneration_rate', 0.1)  # Fallback if not initialized
    
    @property
    def environmental_memory_length(self) -> int:
        return getattr(self.environment, 'environmental_memory_length', 500)  # Fallback if not initialized
    
    @property
    def co_evolution_strength(self) -> float:
        return getattr(self.environment, 'co_evolution_strength', 0.2)  # Fallback if not initialized


# Predefined configuration profiles
DEMO_CONFIG = Config(
    simulation=SimulationConfig(
        max_population=200,
        initial_population=60,
        max_frames=3000
    ),
    cognitive=CognitiveConfig(
        learning_rate=0.15,
        memory_length=80
    )
)

RESEARCH_CONFIG = Config(
    simulation=SimulationConfig(
        max_population=500,
        initial_population=120,
        max_frames=10000
    ),
    cognitive=CognitiveConfig(
        learning_rate=0.08,
        memory_length=150,
        planning_horizon=30
    ),
    social=SocialConfig(
        cultural_inheritance_rate=0.4,
        innovation_tendency_max=0.8
    )
)

PERFORMANCE_CONFIG = Config(
    simulation=SimulationConfig(
        max_population=150,
        initial_population=40,
        animation_interval=100
    ),
    visualization=VisualizationConfig(
        trail_length=20
    )
)
