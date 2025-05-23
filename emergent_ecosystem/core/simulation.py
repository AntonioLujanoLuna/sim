"""
Main simulation engine for the Emergent Intelligence Ecosystem.

This module contains the core simulation class that orchestrates all subsystems,
manages the simulation loop, and coordinates interactions between components.
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional

from ..config import Config
from .individual import EnhancedIndividual
from ..social.networks import SocialNetwork
from ..environment.ecosystem import EnvironmentalMemory
from ..analysis.emergence_detection import EmergenceDetector
from ..analysis.statistics import StatisticsTracker


class EmergentIntelligenceSimulation:
    """Main simulation engine with all advanced systems integrated"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.time_step = 0
        self.individuals: List[EnhancedIndividual] = []
        
        # Core systems
        self.social_network = SocialNetwork()
        self.environment = EnvironmentalMemory(
            self.config.width, 
            self.config.height,
            self.config.environment.patch_density,
            self.config.environment.grid_size
        )
        
        # Analysis systems
        self.emergence_detector = EmergenceDetector()
        self.statistics_tracker = StatisticsTracker()
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize diverse population with different species"""
        species_distribution = {
            'herbivore': 0.4, 
            'predator': 0.15, 
            'scavenger': 0.35, 
            'mystic': 0.1
        }
        
        for i in range(self.config.initial_population):
            species = np.random.choice(
                list(species_distribution.keys()), 
                p=list(species_distribution.values())
            )
            x = random.uniform(0, self.config.width)
            y = random.uniform(0, self.config.height)
            
            individual = EnhancedIndividual(
                x, y, species, 
                individual_id=i, 
                config=self.config
            )
            
            self.individuals.append(individual)
            self.social_network.add_individual(individual.id)
    
    def update(self, frame: int = None) -> List:
        """Main simulation update step"""
        self.time_step += 1
        
        # Update environment
        self.environment.update(
            self.individuals, 
            self.time_step, 
            self.config.resource_regeneration_rate
        )
        
        # Update all individuals
        interactions = []
        for individual in self.individuals[:]:
            individual.update_physics(
                self.individuals, 
                self.environment, 
                self.social_network
            )
            
            # Record interactions for social network updates
            nearby = self._get_nearby_individuals(individual, 50)
            
            for other in nearby:
                # Communication interaction
                if individual.active_signals and other.active_signals:
                    success = random.random() < 0.7
                    interactions.append((
                        individual.id, other.id, 'communication', success
                    ))
                
                # Cooperation attempts
                if (individual.sociability > 0.6 and other.sociability > 0.6 and 
                    random.random() < 0.1):
                    success = random.random() < (
                        individual.sociability + other.sociability
                    ) / 2
                    interactions.append((
                        individual.id, other.id, 'cooperation', success
                    ))
        
        # Update social network based on interactions
        for id1, id2, interaction_type, success in interactions:
            self.social_network.update_relationship(
                id1, id2, interaction_type, success, self.time_step
            )
        
        # Update individuals with interaction learning
        for individual in self.individuals:
            individual_interactions = [
                (itype, self._get_individual_by_id(id2), success) 
                for id1, id2, itype, success in interactions 
                if id1 == individual.id
            ]
            individual.learn_from_interactions(
                individual_interactions, self.social_network
            )
        
        # Social network analysis
        self.social_network.decay_relationships(
            self.time_step, self.config.relationship_decay
        )
        self.social_network.detect_communities()
        self.social_network.identify_leaders()
        
        # Breeding and evolution
        self._handle_breeding()
        
        # Remove dead individuals
        self._remove_dead_individuals()
        
        # Population management
        self._manage_population()
        
        # Analysis and detection
        self.emergence_detector.update(
            self.individuals, self.social_network, 
            self.environment, self.time_step
        )
        
        self.statistics_tracker.update(
            self.individuals, self.social_network, 
            self.environment, self.time_step
        )
        
        return []  # For matplotlib animation compatibility
    
    def _get_nearby_individuals(self, individual: EnhancedIndividual, 
                              radius: float) -> List[EnhancedIndividual]:
        """Get individuals within radius of given individual"""
        nearby = []
        for other in self.individuals:
            if other.id != individual.id:
                dx = other.x - individual.x
                dy = other.y - individual.y
                
                # Handle wraparound
                if abs(dx) > self.config.width / 2:
                    dx = dx - np.sign(dx) * self.config.width
                if abs(dy) > self.config.height / 2:
                    dy = dy - np.sign(dy) * self.config.height
                
                distance = np.sqrt(dx**2 + dy**2)
                if distance <= radius:
                    nearby.append(other)
        
        return nearby
    
    def _get_individual_by_id(self, individual_id: int) -> Optional[EnhancedIndividual]:
        """Get individual by ID"""
        for individual in self.individuals:
            if individual.id == individual_id:
                return individual
        return None
    
    def _handle_breeding(self):
        """Handle breeding with enhanced mate selection"""
        new_offspring = []
        
        for i, individual in enumerate(self.individuals):
            if len(self.individuals) + len(new_offspring) >= self.config.max_population:
                break
            
            potential_mates = [
                other for other in self.individuals[i+1:] 
                if individual.can_breed(other)
            ]
            
            if potential_mates and random.random() < self.config.evolution.breeding_probability:
                # Choose mate based on relationship strength and compatibility
                mate_scores = []
                for mate in potential_mates:
                    relationship_strength = self.social_network.get_relationship_strength(
                        individual.id, mate.id
                    )
                    compatibility = (
                        individual.intelligence + mate.intelligence + 
                        individual.sociability + mate.sociability
                    ) / 4
                    score = relationship_strength + compatibility
                    mate_scores.append(score)
                
                if mate_scores:
                    # Probabilistic selection based on scores
                    total_score = sum(mate_scores)
                    if total_score > 0:
                        probabilities = [score / total_score for score in mate_scores]
                        chosen_mate = np.random.choice(potential_mates, p=probabilities)
                        
                        offspring = individual.breed(chosen_mate)
                        new_offspring.append(offspring)
                        self.social_network.add_individual(offspring.id)
        
        self.individuals.extend(new_offspring)
    
    def _remove_dead_individuals(self):
        """Remove dead individuals from simulation"""
        dead_individuals = [ind for ind in self.individuals if not ind.is_alive()]
        
        for dead_ind in dead_individuals:
            self.individuals.remove(dead_ind)
            self.social_network.remove_individual(dead_ind.id)
    
    def _manage_population(self):
        """Manage population size and diversity"""
        if len(self.individuals) > self.config.max_population:
            # Keep fittest individuals based on multiple criteria
            self.individuals.sort(
                key=lambda x: (
                    x.energy + 
                    x.intelligence * 20 + 
                    len(x.social_memory) * 5 +
                    sum(x.cultural_knowledge.values()) * 2
                ), 
                reverse=True
            )
            self.individuals = self.individuals[:self.config.max_population]
    
    def get_simulation_state(self) -> Dict[str, Any]:
        """Get comprehensive simulation state"""
        return {
            'time_step': self.time_step,
            'population_size': len(self.individuals),
            'species_counts': self._get_species_counts(),
            'emergence_events': self.emergence_detector.get_recent_events(),
            'statistics': self.statistics_tracker.get_latest_stats(),
            'social_network_metrics': self.social_network.get_network_metrics(),
            'environmental_summary': self.environment.get_environmental_summary()
        }
    
    def _get_species_counts(self) -> Dict[str, int]:
        """Get count of each species"""
        counts = defaultdict(int)
        for individual in self.individuals:
            counts[individual.species_name] += 1
        return dict(counts)
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.time_step = 0
        self.individuals.clear()
        self.social_network = SocialNetwork()
        self.environment = EnvironmentalMemory(
            self.config.width, 
            self.config.height,
            self.config.environment.patch_density,
            self.config.environment.grid_size
        )
        self.emergence_detector = EmergenceDetector()
        self.statistics_tracker = StatisticsTracker()
        self._initialize_population()
    
    def run_steps(self, num_steps: int) -> List[Dict[str, Any]]:
        """Run simulation for specified number of steps"""
        states = []
        for _ in range(num_steps):
            self.update()
            states.append(self.get_simulation_state())
        return states
