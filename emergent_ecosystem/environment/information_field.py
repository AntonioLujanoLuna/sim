"""
Information propagation and diffusion across space.

This module implements information diffusion using reaction-diffusion equations
for different types of information (danger, resources, social signals).
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import ndimage
from collections import defaultdict
import math


@dataclass
class InformationSource:
    """Source of information in the environment"""
    x: float
    y: float
    info_type: str
    intensity: float
    decay_rate: float
    spatial_spread: float
    active: bool = True
    creation_time: int = 0


class InformationType:
    """Different categories of information with unique properties"""
    
    DANGER = "danger"
    FOOD = "food"
    SOCIAL = "social"
    TERRITORIAL = "territorial"
    MATING = "mating"
    HELP = "help"
    
    @classmethod
    def get_default_properties(cls, info_type: str) -> Dict[str, float]:
        """Get default diffusion properties for information type"""
        properties = {
            cls.DANGER: {
                'diffusion_rate': 2.0,
                'decay_rate': 0.05,
                'reaction_strength': 0.8,
                'spatial_spread': 100.0,
                'urgency': 0.9
            },
            cls.FOOD: {
                'diffusion_rate': 1.0,
                'decay_rate': 0.02,
                'reaction_strength': 0.6,
                'spatial_spread': 80.0,
                'urgency': 0.6
            },
            cls.SOCIAL: {
                'diffusion_rate': 1.5,
                'decay_rate': 0.03,
                'reaction_strength': 0.7,
                'spatial_spread': 120.0,
                'urgency': 0.5
            },
            cls.TERRITORIAL: {
                'diffusion_rate': 0.8,
                'decay_rate': 0.01,
                'reaction_strength': 0.9,
                'spatial_spread': 150.0,
                'urgency': 0.7
            },
            cls.MATING: {
                'diffusion_rate': 1.2,
                'decay_rate': 0.04,
                'reaction_strength': 0.5,
                'spatial_spread': 200.0,
                'urgency': 0.4
            },
            cls.HELP: {
                'diffusion_rate': 2.5,
                'decay_rate': 0.06,
                'reaction_strength': 0.8,
                'spatial_spread': 90.0,
                'urgency': 0.8
            }
        }
        return properties.get(info_type, properties[cls.SOCIAL])


class InformationField:
    """Information diffusion system using reaction-diffusion equations"""
    
    def __init__(self, width: int, height: int, resolution: int = 50):
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # Grid spacing
        self.dx = width / resolution
        self.dy = height / resolution
        
        # Information grids for each type
        self.fields = {}
        self.sources = defaultdict(list)
        self.information_history = []
        
        # Initialize grids for each information type
        for info_type in [InformationType.DANGER, InformationType.FOOD, 
                         InformationType.SOCIAL, InformationType.TERRITORIAL,
                         InformationType.MATING, InformationType.HELP]:
            self.fields[info_type] = np.zeros((resolution, resolution))
        
        # Diffusion parameters
        self.dt = 0.1  # Time step for numerical integration
        self.time_step = 0
        
        # Gradient fields for information flow
        self.gradients = {}
        self._update_gradients()
        
    def add_information_source(self, x: float, y: float, info_type: str, 
                              intensity: float, decay_rate: Optional[float] = None,
                              spatial_spread: Optional[float] = None):
        """Add a source of information at specified location"""
        properties = InformationType.get_default_properties(info_type)
        
        source = InformationSource(
            x=x, y=y,
            info_type=info_type,
            intensity=intensity,
            decay_rate=decay_rate or properties['decay_rate'],
            spatial_spread=spatial_spread or properties['spatial_spread'],
            creation_time=self.time_step
        )
        
        self.sources[info_type].append(source)
        
        # Immediately add information to the field
        self._add_point_source(source)
    
    def _add_point_source(self, source: InformationSource):
        """Add point source to the information field"""
        # Convert world coordinates to grid coordinates
        grid_x = int((source.x / self.width) * self.resolution)
        grid_y = int((source.y / self.height) * self.resolution)
        
        # Clamp to grid boundaries
        grid_x = max(0, min(self.resolution - 1, grid_x))
        grid_y = max(0, min(self.resolution - 1, grid_y))
        
        # Add Gaussian distribution around source
        spread_cells = max(1, int(source.spatial_spread / self.dx))
        
        for i in range(max(0, grid_x - spread_cells), 
                      min(self.resolution, grid_x + spread_cells + 1)):
            for j in range(max(0, grid_y - spread_cells), 
                          min(self.resolution, grid_y + spread_cells + 1)):
                
                # Calculate distance from source
                world_x = (i / self.resolution) * self.width
                world_y = (j / self.resolution) * self.height
                distance = np.sqrt((world_x - source.x)**2 + (world_y - source.y)**2)
                
                # Gaussian distribution
                if distance < source.spatial_spread:
                    strength = source.intensity * np.exp(-(distance**2) / (2 * (source.spatial_spread/3)**2))
                    self.fields[source.info_type][i, j] += strength
    
    def update(self, dt: Optional[float] = None):
        """Update information diffusion using reaction-diffusion equations"""
        if dt is None:
            dt = self.dt
        
        self.time_step += 1
        
        # Update each information field
        for info_type, field in self.fields.items():
            self._update_field(info_type, field, dt)
        
        # Update sources (decay over time)
        self._update_sources()
        
        # Update gradients for information flow
        self._update_gradients()
        
        # Record information state
        self._record_information_state()
    
    def _update_field(self, info_type: str, field: np.ndarray, dt: float):
        """Update single information field using diffusion equation"""
        properties = InformationType.get_default_properties(info_type)
        diffusion_rate = properties['diffusion_rate']
        decay_rate = properties['decay_rate']
        reaction_strength = properties['reaction_strength']
        
        # Calculate Laplacian (spatial diffusion)
        laplacian = ndimage.laplace(field)
        
        # Reaction-diffusion equation: ∂u/∂t = D∇²u - ku + R(u)
        # Where D is diffusion rate, k is decay rate, R is reaction term
        
        # Diffusion term
        diffusion_term = diffusion_rate * laplacian / (self.dx**2)
        
        # Decay term
        decay_term = -decay_rate * field
        
        # Reaction term (nonlinear amplification for strong signals)
        reaction_term = reaction_strength * field * (1 - field) * (field > 0.1)
        
        # Update field
        field += dt * (diffusion_term + decay_term + reaction_term)
        
        # Ensure non-negative values
        field[field < 0] = 0
        
        # Prevent runaway growth
        field[field > 1] = 1
    
    def _update_sources(self):
        """Update and remove expired information sources"""
        for info_type in list(self.sources.keys()):
            active_sources = []
            
            for source in self.sources[info_type]:
                if source.active:
                    # Decay source intensity
                    source.intensity *= (1 - source.decay_rate)
                    
                    # Remove if too weak
                    if source.intensity > 0.01:
                        active_sources.append(source)
                        
                        # Continue adding information to field
                        self._add_point_source(source)
            
            self.sources[info_type] = active_sources
    
    def _update_gradients(self):
        """Calculate gradients for information flow"""
        for info_type, field in self.fields.items():
            # Calculate gradients using finite differences
            grad_x, grad_y = np.gradient(field, self.dx, self.dy)
            self.gradients[info_type] = (grad_x, grad_y)
    
    def sample_information(self, x: float, y: float, 
                          info_types: Optional[List[str]] = None) -> Dict[str, float]:
        """Sample information at a specific location"""
        if info_types is None:
            info_types = list(self.fields.keys())
        
        # Convert world coordinates to grid coordinates
        grid_x = (x / self.width) * self.resolution
        grid_y = (y / self.height) * self.resolution
        
        # Clamp to valid range
        grid_x = max(0, min(self.resolution - 1, grid_x))
        grid_y = max(0, min(self.resolution - 1, grid_y))
        
        # Bilinear interpolation for smooth sampling
        x1, x2 = int(grid_x), min(int(grid_x) + 1, self.resolution - 1)
        y1, y2 = int(grid_y), min(int(grid_y) + 1, self.resolution - 1)
        
        wx = grid_x - x1
        wy = grid_y - y1
        
        information = {}
        for info_type in info_types:
            if info_type in self.fields:
                field = self.fields[info_type]
                
                # Bilinear interpolation
                value = (field[x1, y1] * (1 - wx) * (1 - wy) +
                        field[x2, y1] * wx * (1 - wy) +
                        field[x1, y2] * (1 - wx) * wy +
                        field[x2, y2] * wx * wy)
                
                information[info_type] = value
        
        return information
    
    def get_information_gradient(self, x: float, y: float, info_type: str) -> Tuple[float, float]:
        """Get information gradient (direction of strongest information flow)"""
        if info_type not in self.gradients:
            return (0.0, 0.0)
        
        # Convert world coordinates to grid coordinates
        grid_x = (x / self.width) * self.resolution
        grid_y = (y / self.height) * self.resolution
        
        # Clamp to valid range
        grid_x = max(0, min(self.resolution - 1, grid_x))
        grid_y = max(0, min(self.resolution - 1, grid_y))
        
        x_idx = int(grid_x)
        y_idx = int(grid_y)
        
        grad_x, grad_y = self.gradients[info_type]
        return (grad_x[x_idx, y_idx], grad_y[x_idx, y_idx])
    
    def get_information_flow_direction(self, x: float, y: float, 
                                     info_type: str) -> Tuple[float, float]:
        """Get normalized direction of information flow"""
        grad_x, grad_y = self.get_information_gradient(x, y, info_type)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        if magnitude > 0:
            return (grad_x / magnitude, grad_y / magnitude)
        else:
            return (0.0, 0.0)
    
    def create_information_wave(self, center_x: float, center_y: float, 
                               info_type: str, intensity: float, radius: float):
        """Create a wave of information emanating from a point"""
        # Add multiple sources in a circle
        num_sources = 8
        for i in range(num_sources):
            angle = 2 * math.pi * i / num_sources
            source_x = center_x + radius * math.cos(angle)
            source_y = center_y + radius * math.sin(angle)
            
            self.add_information_source(source_x, source_y, info_type, 
                                      intensity * 0.8, decay_rate=0.1)
    
    def interfere_information(self, x: float, y: float, info_type1: str, 
                            info_type2: str, interference_strength: float = 0.5):
        """Create interference between two information types"""
        if info_type1 in self.fields and info_type2 in self.fields:
            # Convert coordinates
            grid_x = int((x / self.width) * self.resolution)
            grid_y = int((y / self.height) * self.resolution)
            
            # Clamp to boundaries
            grid_x = max(0, min(self.resolution - 1, grid_x))
            grid_y = max(0, min(self.resolution - 1, grid_y))
            
            # Create interference pattern
            spread = 5  # Grid cells
            for i in range(max(0, grid_x - spread), 
                          min(self.resolution, grid_x + spread + 1)):
                for j in range(max(0, grid_y - spread), 
                              min(self.resolution, grid_y + spread + 1)):
                    
                    # Interference effect
                    field1_val = self.fields[info_type1][i, j]
                    field2_val = self.fields[info_type2][i, j]
                    
                    interference = interference_strength * field1_val * field2_val
                    
                    # Reduce both fields due to interference
                    self.fields[info_type1][i, j] *= (1 - interference)
                    self.fields[info_type2][i, j] *= (1 - interference)
    
    def get_information_density(self, info_type: str) -> float:
        """Get total information density for a specific type"""
        if info_type in self.fields:
            return np.sum(self.fields[info_type])
        return 0.0
    
    def get_information_center_of_mass(self, info_type: str) -> Tuple[float, float]:
        """Get center of mass of information distribution"""
        if info_type not in self.fields:
            return (self.width / 2, self.height / 2)
        
        field = self.fields[info_type]
        total_mass = np.sum(field)
        
        if total_mass == 0:
            return (self.width / 2, self.height / 2)
        
        # Calculate center of mass in grid coordinates
        x_indices, y_indices = np.meshgrid(range(self.resolution), 
                                          range(self.resolution), indexing='ij')
        
        center_x = np.sum(field * x_indices) / total_mass
        center_y = np.sum(field * y_indices) / total_mass
        
        # Convert back to world coordinates
        world_x = (center_x / self.resolution) * self.width
        world_y = (center_y / self.resolution) * self.height
        
        return (world_x, world_y)
    
    def detect_information_clusters(self, info_type: str, 
                                   threshold: float = 0.3) -> List[Tuple[float, float, float]]:
        """Detect clusters of high information density"""
        if info_type not in self.fields:
            return []
        
        field = self.fields[info_type]
        clusters = []
        
        # Find local maxima above threshold
        from scipy.ndimage import maximum_filter
        
        # Local maxima detection
        local_maxima = maximum_filter(field, size=3) == field
        above_threshold = field > threshold
        peaks = local_maxima & above_threshold
        
        # Extract cluster information
        peak_coords = np.where(peaks)
        for i, j in zip(peak_coords[0], peak_coords[1]):
            world_x = (i / self.resolution) * self.width
            world_y = (j / self.resolution) * self.height
            intensity = field[i, j]
            clusters.append((world_x, world_y, intensity))
        
        return clusters
    
    def _record_information_state(self):
        """Record current information state for analysis"""
        state = {
            'time_step': self.time_step,
            'total_information': {},
            'max_information': {},
            'cluster_count': {}
        }
        
        for info_type, field in self.fields.items():
            state['total_information'][info_type] = np.sum(field)
            state['max_information'][info_type] = np.max(field)
            clusters = self.detect_information_clusters(info_type)
            state['cluster_count'][info_type] = len(clusters)
        
        self.information_history.append(state)
        
        # Limit history length
        if len(self.information_history) > 1000:
            self.information_history = self.information_history[-500:]
    
    def get_field_for_visualization(self, info_type: str) -> np.ndarray:
        """Get information field for visualization"""
        if info_type in self.fields:
            return self.fields[info_type].copy()
        else:
            return np.zeros((self.resolution, self.resolution))
    
    def clear_field(self, info_type: str):
        """Clear all information of a specific type"""
        if info_type in self.fields:
            self.fields[info_type].fill(0)
        if info_type in self.sources:
            self.sources[info_type].clear()
    
    def clear_all_fields(self):
        """Clear all information fields"""
        for field in self.fields.values():
            field.fill(0)
        self.sources.clear()
    
    def get_information_statistics(self) -> Dict[str, Any]:
        """Get comprehensive information field statistics"""
        stats = {}
        
        for info_type, field in self.fields.items():
            field_stats = {
                'total_information': np.sum(field),
                'max_intensity': np.max(field),
                'mean_intensity': np.mean(field),
                'std_intensity': np.std(field),
                'active_sources': len(self.sources.get(info_type, [])),
                'clusters': len(self.detect_information_clusters(info_type))
            }
            
            # Calculate spatial autocorrelation
            if np.max(field) > 0:
                from scipy.signal import correlate2d
                autocorr = correlate2d(field, field, mode='same')
                field_stats['spatial_correlation'] = np.max(autocorr) / np.sum(field**2)
            else:
                field_stats['spatial_correlation'] = 0
            
            stats[info_type] = field_stats
        
        return stats
    
    def simulate_information_propagation(self, steps: int = 100) -> List[Dict]:
        """Simulate information propagation for analysis"""
        simulation_history = []
        
        for step in range(steps):
            self.update()
            state = {
                'step': step,
                'statistics': self.get_information_statistics(),
                'field_snapshots': {info_type: field.copy() 
                                   for info_type, field in self.fields.items()}
            }
            simulation_history.append(state)
        
        return simulation_history