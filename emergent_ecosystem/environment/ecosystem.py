"""
Environmental memory and co-evolutionary dynamics.

This module implements environmental patches with memory, resource dynamics,
species-environment interactions, and co-evolutionary feedback loops.
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Any


class EnvironmentalPatch:
    """Environmental patch with memory and adaptation capabilities"""
    
    def __init__(self, x: float, y: float, patch_type: str = 'neutral', grid_id: Optional[Tuple[int, int]] = None):
        self.x = x
        self.y = y
        self.patch_type = patch_type  # 'food', 'danger', 'shelter', 'neutral'
        self.grid_id = grid_id
        
        # Resource properties
        self.resource_level = 1.0
        self.max_resource_level = 1.0
        self.quality = 1.0
        self.carrying_capacity = 10  # Maximum individuals it can support
        
        # Memory and adaptation
        self.visitation_history = deque(maxlen=200)
        self.species_preferences = defaultdict(float)
        self.modification_level = 0.0  # How much species have modified this patch
        self.adaptation_rate = 0.01
        
        # Environmental feedback
        self.stress_level = 0.0  # Environmental degradation
        self.recovery_rate = 0.005
        self.last_regeneration_time = 0
        
        # Historical tracking
        self.resource_history = deque(maxlen=100)
        self.visitor_count_history = deque(maxlen=100)
        
    def update_from_visitation(self, visitor_species: str, visitor_energy: float, 
                             visitor_id: int, time_step: int, interaction_type: str = 'forage'):
        """Update patch based on visitation from an individual"""
        self.visitation_history.append({
            'species': visitor_species,
            'energy': visitor_energy,
            'individual_id': visitor_id,
            'time': time_step,
            'interaction': interaction_type
        })
        
        # Species-specific effects
        if visitor_species == 'herbivore' and self.patch_type == 'food':
            # Herbivores consume resources
            consumption = min(0.05, self.resource_level)
            self.resource_level = max(0.0, self.resource_level - consumption)
            self.stress_level = min(1.0, self.stress_level + 0.02)
            
        elif visitor_species == 'scavenger':
            # Scavengers may find resources in any patch type
            if self.patch_type != 'danger':
                consumption = min(0.03, self.resource_level)
                self.resource_level = max(0.0, self.resource_level - consumption)
            
        elif visitor_species == 'predator':
            # Predators create disturbance, may scare away prey
            self.stress_level = min(1.0, self.stress_level + 0.01)
            if self.patch_type == 'food':
                self.quality = max(0.3, self.quality - 0.01)
                
        elif visitor_species == 'mystic':
            # Mystics may have unique environmental effects
            if interaction_type == 'ritual':
                # Positive environmental effects
                self.quality = min(1.0, self.quality + 0.02)
                self.stress_level = max(0.0, self.stress_level - 0.01)
        
        # Track species preferences (how much each species likes this patch)
        if interaction_type == 'forage' and visitor_energy > 50:
            self.species_preferences[visitor_species] += 0.1
        elif interaction_type == 'avoid':
            self.species_preferences[visitor_species] -= 0.05
        
        # Environmental modification
        modification_strength = 0.01 * (visitor_energy / 100.0)
        self.modification_level = min(1.0, self.modification_level + modification_strength)
        
        # Crowding effects
        recent_visitors = len([v for v in self.visitation_history 
                             if time_step - v['time'] < 10])
        if recent_visitors > self.carrying_capacity:
            self.stress_level = min(1.0, self.stress_level + 0.05)
            self.quality = max(0.1, self.quality - 0.02)
    
    def regenerate(self, base_rate: float, time_step: int):
        """Natural resource regeneration and recovery"""
        if time_step - self.last_regeneration_time >= 1:
            self.last_regeneration_time = time_step
            
            # Resource regeneration affected by modification and stress
            stress_factor = 1.0 - self.stress_level * 0.5
            modification_factor = 1.0 - self.modification_level * 0.3
            
            if self.patch_type == 'food':
                regen_rate = base_rate * stress_factor * modification_factor
                self.resource_level = min(self.max_resource_level, 
                                        self.resource_level + regen_rate)
            
            # Quality and stress recovery
            self.quality = min(1.0, self.quality + self.recovery_rate)
            self.stress_level = max(0.0, self.stress_level - self.recovery_rate * 2)
            
            # Modification level slowly decreases
            self.modification_level = max(0.0, self.modification_level - 0.001)
            
            # Record history
            self.resource_history.append(self.resource_level)
            recent_visitors = len([v for v in self.visitation_history 
                                 if time_step - v['time'] < 5])
            self.visitor_count_history.append(recent_visitors)
    
    def adapt_to_species(self, dominant_species: str):
        """Adapt patch characteristics based on dominant visiting species"""
        if dominant_species in self.species_preferences:
            preference = self.species_preferences[dominant_species]
            
            if preference > 0.5:
                # Positive co-evolution
                if dominant_species == 'herbivore' and self.patch_type == 'food':
                    self.max_resource_level = min(1.5, self.max_resource_level + 0.01)
                elif dominant_species == 'mystic':
                    self.quality = min(1.2, self.quality + 0.005)
            
            elif preference < -0.3:
                # Negative adaptation (patch becomes less suitable)
                self.quality = max(0.5, self.quality - 0.005)
    
    def get_patch_value(self, species: str = None) -> float:
        """Get overall value of patch for a given species"""
        base_value = self.resource_level * self.quality * (1.0 - self.stress_level)
        
        if species and species in self.species_preferences:
            species_modifier = 1.0 + self.species_preferences[species] * 0.5
            base_value *= species_modifier
        
        return np.clip(base_value, 0.0, 2.0)
    
    def get_environmental_signal(self) -> Dict[str, float]:
        """Get environmental signals that individuals can perceive"""
        return {
            'resource_availability': self.resource_level,
            'environmental_quality': self.quality,
            'crowding_level': min(1.0, len(self.visitation_history) / 20.0),
            'stress_level': self.stress_level,
            'species_friendliness': max(self.species_preferences.values()) if self.species_preferences else 0.0
        }


class EnvironmentalMemory:
    """Environment with memory and co-evolutionary dynamics"""
    
    def __init__(self, width: int, height: int, patch_density: float = 0.001, grid_size: int = 50):
        self.width = width
        self.height = height
        self.patch_density = patch_density
        self.grid_size = grid_size
        
        # Patch management
        self.patches: List[EnvironmentalPatch] = []
        self.spatial_grid: Dict[Tuple[int, int], List[EnvironmentalPatch]] = {}
        
        # Environmental history and evolution
        self.environmental_history = []
        self.co_evolution_events = []
        self.species_environment_relationships = defaultdict(lambda: defaultdict(float))
        
        # Global environmental state
        self.global_health = 1.0
        self.biodiversity_index = 0.0
        self.environmental_complexity = 0.0
        
        # Initialize patches
        self._create_initial_patches()
        
    def _create_initial_patches(self):
        """Create initial environmental patches"""
        num_patches = int(self.width * self.height * self.patch_density)
        
        # Patch type distribution
        patch_types = ['food', 'shelter', 'neutral', 'neutral']  # Bias toward neutral
        
        for _ in range(num_patches):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            patch_type = random.choice(patch_types)
            
            grid_x, grid_y = self._get_grid_coordinates(x, y)
            patch = EnvironmentalPatch(x, y, patch_type, (grid_x, grid_y))
            
            # Add some variation to patches
            patch.max_resource_level = random.uniform(0.5, 1.5)
            patch.resource_level = patch.max_resource_level * random.uniform(0.3, 1.0)
            patch.quality = random.uniform(0.6, 1.0)
            
            self.patches.append(patch)
            self._add_to_grid(patch)
    
    def _get_grid_coordinates(self, x: float, y: float) -> Tuple[int, int]:
        """Get grid coordinates for spatial indexing"""
        grid_x = int(x // self.grid_size)
        grid_y = int(y // self.grid_size)
        return (grid_x, grid_y)
    
    def _add_to_grid(self, patch: EnvironmentalPatch):
        """Add patch to spatial grid for efficient lookup"""
        if patch.grid_id not in self.spatial_grid:
            self.spatial_grid[patch.grid_id] = []
        self.spatial_grid[patch.grid_id].append(patch)
    
    def get_nearby_patches(self, x: float, y: float, radius: float) -> List[EnvironmentalPatch]:
        """Get patches within radius of position"""
        nearby = []
        grid_x, grid_y = self._get_grid_coordinates(x, y)
        
        # Check surrounding grid cells
        search_range = int(radius // self.grid_size) + 1
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                cell = (grid_x + dx, grid_y + dy)
                if cell in self.spatial_grid:
                    for patch in self.spatial_grid[cell]:
                        dist = np.sqrt((x - patch.x)**2 + (y - patch.y)**2)
                        if dist <= radius:
                            nearby.append(patch)
        
        return nearby
    
    def get_patch_at_location(self, x: float, y: float, max_distance: float = 10.0) -> Optional[EnvironmentalPatch]:
        """Get the closest patch to a specific location"""
        nearby_patches = self.get_nearby_patches(x, y, max_distance)
        
        if not nearby_patches:
            return None
        
        # Return closest patch
        closest_patch = min(nearby_patches, 
                          key=lambda p: np.sqrt((x - p.x)**2 + (y - p.y)**2))
        return closest_patch
    
    def update(self, individuals: List[Any], time_step: int, regeneration_rate: float):
        """Update environmental state based on individual interactions"""
        # Reset visitor tracking
        current_visitors = defaultdict(list)
        
        # Record visitations and interactions
        for individual in individuals:
            nearby_patches = self.get_nearby_patches(individual.x, individual.y, 20)
            
            for patch in nearby_patches:
                distance = np.sqrt((individual.x - patch.x)**2 + (individual.y - patch.y)**2)
                
                if distance < 15:  # Close interaction
                    interaction_type = self._determine_interaction_type(individual, patch)
                    patch.update_from_visitation(
                        individual.species_name, 
                        individual.energy, 
                        individual.id,
                        time_step, 
                        interaction_type
                    )
                    current_visitors[patch].append(individual)
        
        # Regenerate all patches
        for patch in self.patches:
            patch.regenerate(regeneration_rate, time_step)
            
            # Adapt to visiting species
            if patch.visitation_history:
                recent_visitors = [v for v in patch.visitation_history 
                                 if time_step - v['time'] < 50]
                if recent_visitors:
                    species_counts = defaultdict(int)
                    for visitor in recent_visitors:
                        species_counts[visitor['species']] += 1
                    
                    if species_counts:
                        dominant_species = max(species_counts.items(), key=lambda x: x[1])[0]
                        patch.adapt_to_species(dominant_species)
        
        # Update global environmental metrics
        self._update_global_metrics(time_step)
        
        # Detect co-evolution events
        self._detect_coevolution_events(time_step)
    
    def _determine_interaction_type(self, individual: Any, patch: EnvironmentalPatch) -> str:
        """Determine the type of interaction between individual and patch"""
        if hasattr(individual, 'current_goal'):
            if individual.current_goal == 'find_food' and patch.patch_type == 'food':
                return 'forage'
            elif individual.current_goal == 'avoid_danger':
                return 'avoid'
            elif individual.species_name == 'mystic' and patch.quality > 0.8:
                return 'ritual'
        
        # Default interaction types based on species and patch
        if individual.species_name in ['herbivore', 'scavenger'] and patch.patch_type == 'food':
            return 'forage'
        elif patch.patch_type == 'shelter':
            return 'shelter'
        else:
            return 'explore'
    
    def _update_global_metrics(self, time_step: int):
        """Update global environmental health and complexity metrics"""
        if not self.patches:
            return
        
        # Environmental health
        avg_quality = np.mean([patch.quality for patch in self.patches])
        avg_stress = np.mean([patch.stress_level for patch in self.patches])
        self.global_health = avg_quality * (1.0 - avg_stress)
        
        # Biodiversity index (based on species diversity across patches)
        all_species = set()
        for patch in self.patches:
            all_species.update(patch.species_preferences.keys())
        self.biodiversity_index = len(all_species)
        
        # Environmental complexity (variation in patch types and states)
        patch_types = [patch.patch_type for patch in self.patches]
        type_diversity = len(set(patch_types))
        resource_variation = np.std([patch.resource_level for patch in self.patches])
        self.environmental_complexity = type_diversity + resource_variation
        
        # Record environmental state
        self.environmental_history.append({
            'time_step': time_step,
            'global_health': self.global_health,
            'biodiversity_index': self.biodiversity_index,
            'environmental_complexity': self.environmental_complexity,
            'total_patches': len(self.patches),
            'avg_resource_level': np.mean([patch.resource_level for patch in self.patches]),
            'avg_modification': np.mean([patch.modification_level for patch in self.patches])
        })
    
    def _detect_coevolution_events(self, time_step: int):
        """Detect significant co-evolutionary events"""
        if len(self.environmental_history) < 100:
            return
        
        recent_history = self.environmental_history[-50:]
        older_history = self.environmental_history[-100:-50]
        
        # Check for significant changes
        recent_health = np.mean([entry['global_health'] for entry in recent_history])
        older_health = np.mean([entry['global_health'] for entry in older_history])
        
        recent_complexity = np.mean([entry['environmental_complexity'] for entry in recent_history])
        older_complexity = np.mean([entry['environmental_complexity'] for entry in older_history])
        
        # Detect health changes
        if abs(recent_health - older_health) > 0.2:
            event_type = 'environmental_improvement' if recent_health > older_health else 'environmental_degradation'
            self.co_evolution_events.append({
                'type': event_type,
                'time_step': time_step,
                'magnitude': abs(recent_health - older_health),
                'health_change': recent_health - older_health
            })
        
        # Detect complexity changes
        if abs(recent_complexity - older_complexity) > 1.0:
            event_type = 'complexity_increase' if recent_complexity > older_complexity else 'complexity_decrease'
            self.co_evolution_events.append({
                'type': event_type,
                'time_step': time_step,
                'magnitude': abs(recent_complexity - older_complexity),
                'complexity_change': recent_complexity - older_complexity
            })
    
    def create_new_patch(self, x: float, y: float, patch_type: str, creator_species: str = None):
        """Create a new environmental patch (niche construction)"""
        grid_coords = self._get_grid_coordinates(x, y)
        new_patch = EnvironmentalPatch(x, y, patch_type, grid_coords)
        
        # Patches created by individuals may have special properties
        if creator_species:
            if creator_species == 'mystic':
                new_patch.quality = 1.2  # High quality
                new_patch.recovery_rate = 0.01  # Fast recovery
            elif creator_species == 'herbivore':
                new_patch.max_resource_level = 1.3  # High capacity
        
        self.patches.append(new_patch)
        self._add_to_grid(new_patch)
        
        # Record niche construction event
        self.co_evolution_events.append({
            'type': 'niche_construction',
            'time_step': 0,  # Will be updated by caller
            'creator_species': creator_species,
            'patch_type': patch_type,
            'location': (x, y)
        })
    
    def get_environmental_summary(self) -> Dict[str, Any]:
        """Get comprehensive environmental summary"""
        if not self.patches:
            return {}
        
        patch_types = defaultdict(int)
        resource_levels = []
        quality_levels = []
        stress_levels = []
        modification_levels = []
        
        for patch in self.patches:
            patch_types[patch.patch_type] += 1
            resource_levels.append(patch.resource_level)
            quality_levels.append(patch.quality)
            stress_levels.append(patch.stress_level)
            modification_levels.append(patch.modification_level)
        
        return {
            'total_patches': len(self.patches),
            'patch_types': dict(patch_types),
            'avg_resource_level': np.mean(resource_levels),
            'avg_quality': np.mean(quality_levels),
            'avg_stress': np.mean(stress_levels),
            'avg_modification': np.mean(modification_levels),
            'global_health': self.global_health,
            'biodiversity_index': self.biodiversity_index,
            'environmental_complexity': self.environmental_complexity,
            'coevolution_events': len(self.co_evolution_events)
        }
    
    def get_species_environment_relationships(self) -> Dict[str, Dict[str, float]]:
        """Get species-environment relationship matrix"""
        relationships = defaultdict(lambda: defaultdict(float))
        
        for patch in self.patches:
            for species, preference in patch.species_preferences.items():
                relationships[species][patch.patch_type] += preference
        
        # Normalize by number of patches of each type
        patch_type_counts = defaultdict(int)
        for patch in self.patches:
            patch_type_counts[patch.patch_type] += 1
        
        for species in relationships:
            for patch_type in relationships[species]:
                if patch_type_counts[patch_type] > 0:
                    relationships[species][patch_type] /= patch_type_counts[patch_type]
        
        return dict(relationships)
