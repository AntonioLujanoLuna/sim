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
    
    # Legacy compatibility - provide direct access to commonly used parameters
    @property
    def width(self) -> int:
        return self.simulation.width
    
    @property
    def height(self) -> int:
        return self.simulation.height
    
    @property
    def max_population(self) -> int:
        return self.simulation.max_population
    
    @property
    def initial_population(self) -> int:
        return self.simulation.initial_population
    
    @property
    def time_step(self) -> float:
        return self.simulation.time_step
    
    @property
    def separation_radius(self) -> float:
        return self.physics.separation_radius
    
    @property
    def alignment_radius(self) -> float:
        return self.physics.alignment_radius
    
    @property
    def cohesion_radius(self) -> float:
        return self.physics.cohesion_radius
    
    @property
    def communication_radius(self) -> float:
        return self.physics.communication_radius
    
    @property
    def separation_force(self) -> float:
        return self.physics.separation_force
    
    @property
    def alignment_force(self) -> float:
        return self.physics.alignment_force
    
    @property
    def cohesion_force(self) -> float:
        return self.physics.cohesion_force
    
    @property
    def social_force(self) -> float:
        return self.physics.social_force
    
    @property
    def environmental_force(self) -> float:
        return self.physics.environmental_force
    
    @property
    def memory_length(self) -> int:
        return self.cognitive.memory_length
    
    @property
    def planning_horizon(self) -> int:
        return self.cognitive.planning_horizon
    
    @property
    def attention_span(self) -> int:
        return self.cognitive.attention_span
    
    @property
    def learning_rate(self) -> float:
        return self.cognitive.learning_rate
    
    @property
    def max_social_connections(self) -> int:
        return self.social.max_social_connections
    
    @property
    def relationship_decay(self) -> float:
        return self.social.relationship_decay
    
    @property
    def trust_threshold(self) -> float:
        return self.social.trust_threshold
    
    @property
    def communication_mutation_rate(self) -> float:
        return self.social.communication_mutation_rate
    
    @property
    def breeding_energy_threshold(self) -> float:
        return self.evolution.breeding_energy_threshold
    
    @property
    def mutation_rate(self) -> float:
        return self.evolution.mutation_rate
    
    @property
    def sexual_selection_strength(self) -> float:
        return self.evolution.sexual_selection_strength
    
    @property
    def cultural_inheritance_rate(self) -> float:
        return self.social.cultural_inheritance_rate
    
    @property
    def resource_regeneration_rate(self) -> float:
        return self.environment.resource_regeneration_rate
    
    @property
    def environmental_memory_length(self) -> int:
        return self.environment.environmental_memory_length
    
    @property
    def co_evolution_strength(self) -> float:
        return self.environment.co_evolution_strength


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
