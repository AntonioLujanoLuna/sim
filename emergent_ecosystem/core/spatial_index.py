"""
Spatial indexing system for efficient neighbor finding and collision detection.

This module implements spatial hash grids and other optimization techniques
to reduce the O(n²) complexity of individual perception and interaction systems.
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import math

from .individual import EnhancedIndividual


class SpatialHashGrid:
    """Spatial hash grid for efficient neighbor queries"""
    
    def __init__(self, width: float, height: float, cell_size: float = 100.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.individual_positions: Dict[int, Tuple[float, float]] = {}
        
    def _get_cell_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Get grid cell coordinates for given position"""
        cell_x = int(x // self.cell_size)
        cell_y = int(y // self.cell_size)
        return (cell_x, cell_y)
    
    def _get_neighboring_cells(self, cell_x: int, cell_y: int) -> List[Tuple[int, int]]:
        """Get all neighboring cells including the current cell"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbors.append((cell_x + dx, cell_y + dy))
        return neighbors
    
    def update_individual(self, individual: EnhancedIndividual):
        """Update individual's position in the spatial grid"""
        individual_id = individual.id
        new_pos = (individual.x, individual.y)
        
        # Remove from old cell if exists
        if individual_id in self.individual_positions:
            old_pos = self.individual_positions[individual_id]
            old_cell = self._get_cell_coords(old_pos[0], old_pos[1])
            self.grid[old_cell].discard(individual_id)
            
            # Clean up empty cells
            if not self.grid[old_cell]:
                del self.grid[old_cell]
        
        # Add to new cell
        new_cell = self._get_cell_coords(new_pos[0], new_pos[1])
        self.grid[new_cell].add(individual_id)
        self.individual_positions[individual_id] = new_pos
    
    def remove_individual(self, individual_id: int):
        """Remove individual from spatial grid"""
        if individual_id in self.individual_positions:
            pos = self.individual_positions[individual_id]
            cell = self._get_cell_coords(pos[0], pos[1])
            self.grid[cell].discard(individual_id)
            
            # Clean up empty cells
            if not self.grid[cell]:
                del self.grid[cell]
            
            del self.individual_positions[individual_id]
    
    def get_nearby_individuals(self, x: float, y: float, radius: float, 
                             individuals_dict: Dict[int, EnhancedIndividual]) -> List[EnhancedIndividual]:
        """Get individuals within radius using spatial indexing"""
        cell = self._get_cell_coords(x, y)
        
        # Determine how many cells we need to check based on radius
        cells_to_check = max(1, int(math.ceil(radius / self.cell_size)))
        
        nearby_individuals = []
        checked_ids = set()
        
        # Check all cells within the required range
        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                check_cell = (cell[0] + dx, cell[1] + dy)
                
                if check_cell in self.grid:
                    for individual_id in self.grid[check_cell]:
                        if individual_id not in checked_ids and individual_id in individuals_dict:
                            individual = individuals_dict[individual_id]
                            
                            # Calculate actual distance
                            dx_actual = individual.x - x
                            dy_actual = individual.y - y
                            distance = math.sqrt(dx_actual**2 + dy_actual**2)
                            
                            if distance <= radius:
                                nearby_individuals.append(individual)
                            
                            checked_ids.add(individual_id)
        
        return nearby_individuals
    
    def clear(self):
        """Clear all data from the spatial grid"""
        self.grid.clear()
        self.individual_positions.clear()


class PerformanceOptimizer:
    """Performance optimization utilities for the simulation"""
    
    def __init__(self, config):
        self.config = config
        self.spatial_grid = SpatialHashGrid(
            config.width, 
            config.height, 
            cell_size=max(config.communication_radius, config.cohesion_radius) * 1.5
        )
        
        # Performance monitoring
        self.update_times = []
        self.perception_times = []
        self.max_history_length = 100
        
    def optimize_individual_perception(self, individual: EnhancedIndividual, 
                                     all_individuals: List[EnhancedIndividual],
                                     environment, social_network) -> Dict:
        """Optimized perception using spatial indexing"""
        import time
        start_time = time.time()
        
        # Create individuals dictionary for fast lookup
        individuals_dict = {ind.id: ind for ind in all_individuals}
        
        # Use spatial indexing for nearby individuals
        nearby_individuals = self.spatial_grid.get_nearby_individuals(
            individual.x, individual.y, 
            self.config.communication_radius,
            individuals_dict
        )
        
        # Build optimized perception data
        perception = {
            'nearby_individuals': [],
            'environmental_features': [],
            'social_information': [],
            'danger_signals': [],
            'opportunity_signals': []
        }
        
        # Process only nearby individuals (much smaller set)
        for other in nearby_individuals:
            if other.id != individual.id:
                dx = other.x - individual.x
                dy = other.y - individual.y
                distance = math.sqrt(dx**2 + dy**2)
                
                perception_data = {
                    'individual': other,
                    'distance': distance,
                    'relationship_strength': social_network.get_relationship_strength(individual.id, other.id),
                    'species': other.species_name,
                    'energy_level': other.energy / 100.0,
                    'active_signals': other.active_signals.copy(),
                    'relative_position': (dx, dy)
                }
                perception['nearby_individuals'].append(perception_data)
        
        # Environmental perception (optimized)
        if hasattr(environment, 'get_nearby_patches'):
            nearby_patches = environment.get_nearby_patches(individual.x, individual.y, 50)
            for patch in nearby_patches:
                perception_data = {
                    'patch': patch,
                    'distance': math.sqrt((individual.x - patch.x)**2 + (individual.y - patch.y)**2),
                    'type': patch.patch_type,
                    'quality': patch.quality,
                    'resource_level': patch.resource_level,
                    'signals': patch.get_environmental_signal()
                }
                perception['environmental_features'].append(perception_data)
        
        # Process communication signals
        for individual_perception in perception['nearby_individuals']:
            for signal_id in individual_perception['active_signals']:
                meaning = individual.communication.interpret_signal(signal_id, 'general')
                if meaning > 0.5:  # Significant signal
                    perception['social_information'].append({
                        'sender': individual_perception['individual'],
                        'signal': signal_id,
                        'meaning': meaning,
                        'trust_level': individual_perception['relationship_strength']
                    })
        
        # Track performance
        perception_time = time.time() - start_time
        self.perception_times.append(perception_time)
        if len(self.perception_times) > self.max_history_length:
            self.perception_times.pop(0)
        
        return perception
    
    def update_spatial_index(self, individuals: List[EnhancedIndividual]):
        """Update spatial index with current individual positions"""
        for individual in individuals:
            self.spatial_grid.update_individual(individual)
    
    def remove_from_spatial_index(self, individual_id: int):
        """Remove individual from spatial index"""
        self.spatial_grid.remove_individual(individual_id)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        metrics = {}
        
        if self.perception_times:
            metrics['avg_perception_time'] = sum(self.perception_times) / len(self.perception_times)
            metrics['max_perception_time'] = max(self.perception_times)
            metrics['min_perception_time'] = min(self.perception_times)
        
        if self.update_times:
            metrics['avg_update_time'] = sum(self.update_times) / len(self.update_times)
            metrics['max_update_time'] = max(self.update_times)
            metrics['min_update_time'] = min(self.update_times)
        
        # Spatial grid statistics
        metrics['spatial_grid_cells'] = len(self.spatial_grid.grid)
        metrics['individuals_in_grid'] = len(self.spatial_grid.individual_positions)
        
        return metrics
    
    def optimize_memory_usage(self, individuals: List[EnhancedIndividual]):
        """Optimize memory usage by cleaning up old data"""
        for individual in individuals:
            # Limit memory sizes
            if hasattr(individual, 'spatial_memory') and len(individual.spatial_memory) > self.config.memory_length:
                # Remove oldest memories
                while len(individual.spatial_memory) > self.config.memory_length:
                    individual.spatial_memory.popleft()
            
            # Clean up old social memories
            if hasattr(individual, 'social_memory'):
                current_time = getattr(individual, 'age', 0)
                memory_cutoff = current_time - self.config.memory_length
                
                # Remove old social memories
                old_ids = [
                    ind_id for ind_id, memories in individual.social_memory.items()
                    if all(memory.get('timestamp', 0) < memory_cutoff for memory in memories)
                ]
                
                for old_id in old_ids:
                    del individual.social_memory[old_id]
            
            # Limit cultural knowledge size
            if hasattr(individual, 'cultural_knowledge') and len(individual.cultural_knowledge) > 50:
                # Keep only the most valuable knowledge
                sorted_knowledge = sorted(
                    individual.cultural_knowledge.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                individual.cultural_knowledge = dict(sorted_knowledge[:50]) 