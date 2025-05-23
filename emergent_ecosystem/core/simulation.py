"""
Main simulation engine for the Emergent Intelligence Ecosystem.

This module contains the core simulation class that orchestrates all subsystems,
manages the simulation loop, and coordinates interactions between components.
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional
import time

from ..config import Config
from .individual import EnhancedIndividual
from .spatial_index import PerformanceOptimizer
from .error_handling import error_handler, safe_execute, log_performance, validate_configuration
from ..social.networks import SocialNetwork
from ..environment.ecosystem import EnvironmentalMemory
from ..analysis.emergence_detection import EmergenceDetector
from ..analysis.statistics import StatisticsTracker


class EmergentIntelligenceSimulation:
    """Main simulation engine with all advanced systems integrated"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Validate configuration
        if not validate_configuration(self.config):
            raise ValueError("Invalid configuration provided")
        
        self.time_step = 0
        self.individuals: List[EnhancedIndividual] = []
        
        # Performance optimization
        self.performance_optimizer = PerformanceOptimizer(self.config)
        
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
        
        # Memory management
        self.data_history = deque(maxlen=1000)  # Limit history size
        self.information_history = deque(maxlen=500)  # Add size limit
        
        # Performance tracking
        self.performance_metrics = {}
        self.last_cleanup_time = 0
        self.cleanup_interval = 100  # Cleanup every 100 steps
        
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
    
    @log_performance
    def update(self, frame: int = None) -> List:
        """Main simulation update step with performance optimizations"""
        self.time_step += 1
        update_start_time = time.time()
        
        # Update spatial index for efficient neighbor finding
        self.performance_optimizer.update_spatial_index(self.individuals)
        
        # Update environment
        safe_execute(
            self.environment.update,
            self.individuals, 
            self.time_step, 
            self.config.resource_regeneration_rate,
            error_message="Environment update failed"
        )
        
        # Update all individuals with optimized perception
        interactions = []
        for individual in self.individuals[:]:
            try:
                # Use optimized perception system
                perception = self.performance_optimizer.optimize_individual_perception(
                    individual, self.individuals, self.environment, self.social_network
                )
                
                # Update individual with optimized perception
                individual.make_decisions(perception)
                individual.update_physics(self.individuals, self.environment, self.social_network)
                
                # Record interactions for social network updates (optimized)
                nearby = perception.get('nearby_individuals', [])
                
                for other_data in nearby:
                    other = other_data['individual']
                    
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
                        
            except Exception as e:
                error_handler.handle_perception_error(individual.id, e)
                continue
        
        # Update social network based on interactions
        for id1, id2, interaction_type, success in interactions:
            safe_execute(
                self.social_network.update_relationship,
                id1, id2, interaction_type, success, self.time_step,
                error_message=f"Failed to update relationship {id1}-{id2}"
            )
        
        # Update individuals with interaction learning
        for individual in self.individuals:
            try:
                individual_interactions = [
                    (itype, self._get_individual_by_id(id2), success) 
                    for id1, id2, itype, success in interactions 
                    if id1 == individual.id
                ]
                individual.learn_from_interactions(
                    individual_interactions, self.social_network
                )
            except Exception as e:
                error_handler.handle_cognition_error(
                    individual.id, 'interaction_learning', e
                )
        
        # Social network analysis with error handling
        safe_execute(
            self.social_network.decay_relationships,
            self.time_step, self.config.relationship_decay,
            error_message="Relationship decay failed"
        )
        
        safe_execute(
            self.social_network.detect_communities,
            error_message="Community detection failed"
        )
        
        safe_execute(
            self.social_network.identify_leaders,
            error_message="Leader identification failed"
        )
        
        # Breeding and evolution
        self._handle_breeding()
        
        # Remove dead individuals
        self._remove_dead_individuals()
        
        # Population management
        self._manage_population()
        
        # Periodic memory cleanup
        if self.time_step - self.last_cleanup_time >= self.cleanup_interval:
            self.performance_optimizer.optimize_memory_usage(self.individuals)
            self.last_cleanup_time = self.time_step
        
        # Analysis and detection
        safe_execute(
            self.emergence_detector.update,
            self.individuals, self.social_network, 
            self.environment, self.time_step,
            error_message="Emergence detection failed"
        )
        
        safe_execute(
            self.statistics_tracker.update,
            self.individuals, self.social_network, 
            self.environment, self.time_step,
            error_message="Statistics tracking failed"
        )
        
        # Track performance
        update_time = time.time() - update_start_time
        self.performance_optimizer.update_times.append(update_time)
        if len(self.performance_optimizer.update_times) > self.performance_optimizer.max_history_length:
            self.performance_optimizer.update_times.pop(0)
        
        # Store limited simulation state
        state = self.get_simulation_state()
        self.data_history.append(state)
        
        return []  # For matplotlib animation compatibility
    
    def _get_nearby_individuals_optimized(self, individual: EnhancedIndividual, 
                                        radius: float) -> List[EnhancedIndividual]:
        """Get individuals within radius using spatial indexing"""
        individuals_dict = {ind.id: ind for ind in self.individuals}
        return self.performance_optimizer.spatial_grid.get_nearby_individuals(
            individual.x, individual.y, radius, individuals_dict
        )
    
    def _get_individual_by_id(self, individual_id: int) -> Optional[EnhancedIndividual]:
        """Get individual by ID with caching for performance"""
        for individual in self.individuals:
            if individual.id == individual_id:
                return individual
        return None
    
    def _handle_breeding(self):
        """Handle breeding with enhanced mate selection and error handling"""
        new_offspring = []
        
        for i, individual in enumerate(self.individuals):
            if len(self.individuals) + len(new_offspring) >= self.config.max_population:
                break
            
            try:
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
                        compatibility = 1.0 - abs(individual.sociability - mate.sociability)
                        score = relationship_strength * 0.6 + compatibility * 0.4
                        mate_scores.append((mate, score))
                    
                    if mate_scores:
                        # Select mate probabilistically based on scores
                        mate_scores.sort(key=lambda x: x[1], reverse=True)
                        top_mates = mate_scores[:3]  # Consider top 3 mates
                        
                        if top_mates:
                            chosen_mate = random.choices(
                                [mate for mate, score in top_mates],
                                weights=[score for mate, score in top_mates]
                            )[0]
                            
                            offspring = individual.breed(chosen_mate)
                            if offspring:
                                new_offspring.append(offspring)
                                self.social_network.add_individual(offspring.id)
                                
            except Exception as e:
                error_handler.handle_cognition_error(
                    individual.id, 'breeding', e
                )
        
        # Add new offspring to population
        self.individuals.extend(new_offspring)
    
    def _remove_dead_individuals(self):
        """Remove dead individuals and clean up references"""
        dead_individuals = [ind for ind in self.individuals if not ind.is_alive()]
        
        for individual in dead_individuals:
            # Remove from spatial index
            self.performance_optimizer.remove_from_spatial_index(individual.id)
            
            # Remove from social network
            safe_execute(
                self.social_network.remove_individual,
                individual.id,
                error_message=f"Failed to remove individual {individual.id} from social network"
            )
            
            # Remove from main list
            self.individuals.remove(individual)
    
    def _manage_population(self):
        """Manage population size with improved efficiency"""
        if len(self.individuals) > self.config.max_population:
            # Remove weakest individuals
            self.individuals.sort(key=lambda x: x.energy + x.lifetime_rewards)
            excess = len(self.individuals) - self.config.max_population
            
            for individual in self.individuals[:excess]:
                self.performance_optimizer.remove_from_spatial_index(individual.id)
                safe_execute(
                    self.social_network.remove_individual,
                    individual.id,
                    error_message=f"Failed to remove excess individual {individual.id}"
                )
            
            self.individuals = self.individuals[excess:]
    
    def get_simulation_state(self) -> Dict[str, Any]:
        """Get current simulation state with memory optimization"""
        # Only store essential state information
        state = {
            'time_step': self.time_step,
            'population_size': len(self.individuals),
            'species_counts': self._get_species_counts(),
            'avg_energy': np.mean([ind.energy for ind in self.individuals]) if self.individuals else 0,
            'avg_age': np.mean([ind.age for ind in self.individuals]) if self.individuals else 0,
            'network_density': safe_execute(
                lambda: self.social_network.get_network_density(),
                fallback_value=0.0
            ),
            'performance_metrics': self.performance_optimizer.get_performance_metrics()
        }
        
        return state
    
    def _get_species_counts(self) -> Dict[str, int]:
        """Get count of each species"""
        counts = defaultdict(int)
        for individual in self.individuals:
            counts[individual.species_name] += 1
        return dict(counts)
    
    def reset_simulation(self):
        """Reset simulation state with proper cleanup"""
        # Clear all data structures
        self.individuals.clear()
        self.social_network = SocialNetwork()
        self.performance_optimizer.spatial_grid.clear()
        self.data_history.clear()
        self.information_history.clear()
        
        # Reset counters
        self.time_step = 0
        self.last_cleanup_time = 0
        
        # Reset error tracking
        error_handler.reset_error_tracking()
        
        # Reinitialize population
        self._initialize_population()
    
    def run_steps(self, num_steps: int) -> List[Dict[str, Any]]:
        """Run simulation for specified number of steps"""
        states = []
        
        for step in range(num_steps):
            try:
                self.update()
                states.append(self.get_simulation_state())
                
                # Progress reporting for long runs
                if num_steps > 100 and step % (num_steps // 10) == 0:
                    print(f"Completed {step}/{num_steps} steps ({step/num_steps*100:.1f}%)")
                    
            except Exception as e:
                error_handler._log_error('simulation', 'run_steps', e)
                print(f"Simulation error at step {step}: {str(e)}")
                break
        
        return states
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'optimizer_metrics': self.performance_optimizer.get_performance_metrics(),
            'error_summary': error_handler.get_error_summary(),
            'memory_usage': {
                'individuals_count': len(self.individuals),
                'data_history_size': len(self.data_history),
                'information_history_size': len(self.information_history),
                'spatial_grid_cells': len(self.performance_optimizer.spatial_grid.grid)
            }
        }
