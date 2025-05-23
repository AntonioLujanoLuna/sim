"""
Chaotic attractor field systems that influence agent movement.

This module implements multiple chaotic systems (Lorenz, Rössler, Chua, etc.)
to create dynamic environmental flow fields and complex movement patterns.
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
from scipy.integrate import odeint
import math


@dataclass
class AttractorParameters:
    """Parameters for a chaotic attractor"""
    name: str
    parameters: Dict[str, float]
    scale: float = 1.0
    offset: Tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    active: bool = True


class LorenzAttractor:
    """Classic Lorenz chaotic attractor"""
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.state = np.array([1.0, 1.0, 1.0])  # Initial state
        self.dt = 0.01
        
    def update(self, dt: Optional[float] = None):
        """Update attractor state using Runge-Kutta integration"""
        if dt is None:
            dt = self.dt
            
        def lorenz_equations(state, t):
            x, y, z = state
            return [
                self.sigma * (y - x),
                x * (self.rho - z) - y,
                x * y - self.beta * z
            ]
        
        # Single step integration
        t = np.array([0, dt])
        solution = odeint(lorenz_equations, self.state, t)
        self.state = solution[1]
        
        return self.state
    
    def get_force_2d(self, scale: float = 1.0) -> Tuple[float, float]:
        """Get 2D force vector from 3D attractor state"""
        # Project 3D state to 2D force
        fx = (self.state[0] - self.state[2]) * scale * 0.1
        fy = (self.state[1] - self.state[0]) * scale * 0.1
        return (fx, fy)


class RosslerAttractor:
    """Rössler chaotic attractor"""
    
    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        self.a = a
        self.b = b
        self.c = c
        self.state = np.array([0.1, 0.1, 0.1])
        self.dt = 0.01
    
    def update(self, dt: Optional[float] = None):
        """Update attractor state"""
        if dt is None:
            dt = self.dt
            
        def rossler_equations(state, t):
            x, y, z = state
            return [
                -y - z,
                x + self.a * y,
                self.b + z * (x - self.c)
            ]
        
        t = np.array([0, dt])
        solution = odeint(rossler_equations, self.state, t)
        self.state = solution[1]
        
        return self.state
    
    def get_force_2d(self, scale: float = 1.0) -> Tuple[float, float]:
        """Get 2D force vector"""
        fx = self.state[0] * scale * 0.05
        fy = self.state[1] * scale * 0.05
        return (fx, fy)


class ChuaAttractor:
    """Chua's circuit chaotic attractor"""
    
    def __init__(self, alpha: float = 15.6, beta: float = 28.0, 
                 m0: float = -1.143, m1: float = -0.714):
        self.alpha = alpha
        self.beta = beta
        self.m0 = m0
        self.m1 = m1
        self.state = np.array([0.1, 0.1, 0.1])
        self.dt = 0.01
    
    def _chua_nonlinearity(self, x):
        """Chua's nonlinear function"""
        return self.m1 * x + 0.5 * (self.m0 - self.m1) * (abs(x + 1) - abs(x - 1))
    
    def update(self, dt: Optional[float] = None):
        """Update attractor state"""
        if dt is None:
            dt = self.dt
            
        def chua_equations(state, t):
            x, y, z = state
            return [
                self.alpha * (y - x - self._chua_nonlinearity(x)),
                x - y + z,
                -self.beta * y
            ]
        
        t = np.array([0, dt])
        solution = odeint(chua_equations, self.state, t)
        self.state = solution[1]
        
        return self.state
    
    def get_force_2d(self, scale: float = 1.0) -> Tuple[float, float]:
        """Get 2D force vector"""
        fx = self.state[0] * scale * 0.02
        fy = self.state[1] * scale * 0.02
        return (fx, fy)


class DuffingAttractor:
    """Duffing oscillator (can exhibit chaotic behavior)"""
    
    def __init__(self, alpha: float = 1.0, beta: float = -1.0, 
                 gamma: float = 0.3, omega: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.state = np.array([0.1, 0.1])  # [position, velocity]
        self.time = 0.0
        self.dt = 0.01
    
    def update(self, dt: Optional[float] = None):
        """Update attractor state"""
        if dt is None:
            dt = self.dt
            
        def duffing_equations(state, t):
            x, v = state
            return [
                v,
                -self.alpha * v - self.beta * x - x**3 + self.gamma * np.cos(self.omega * t)
            ]
        
        t = np.array([self.time, self.time + dt])
        solution = odeint(duffing_equations, self.state, t)
        self.state = solution[1]
        self.time += dt
        
        return self.state
    
    def get_force_2d(self, scale: float = 1.0) -> Tuple[float, float]:
        """Get 2D force vector"""
        fx = self.state[0] * scale * 0.1
        fy = self.state[1] * scale * 0.1
        return (fx, fy)


class VanDerPolAttractor:
    """Van der Pol oscillator"""
    
    def __init__(self, mu: float = 2.0):
        self.mu = mu
        self.state = np.array([0.1, 0.1])
        self.dt = 0.01
    
    def update(self, dt: Optional[float] = None):
        """Update attractor state"""
        if dt is None:
            dt = self.dt
            
        def vanderpol_equations(state, t):
            x, v = state
            return [
                v,
                self.mu * (1 - x**2) * v - x
            ]
        
        t = np.array([0, dt])
        solution = odeint(vanderpol_equations, self.state, t)
        self.state = solution[1]
        
        return self.state
    
    def get_force_2d(self, scale: float = 1.0) -> Tuple[float, float]:
        """Get 2D force vector"""
        fx = self.state[0] * scale * 0.1
        fy = self.state[1] * scale * 0.1
        return (fx, fy)


class AttractorField:
    """Manages multiple chaotic attractors to create complex environmental fields"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.attractors = {}
        self.field_cache = {}
        self.cache_valid = False
        self.time_step = 0
        
        # Initialize default attractors
        self._initialize_default_attractors()
        
    def _initialize_default_attractors(self):
        """Initialize a set of default chaotic attractors"""
        # Lorenz attractor in center
        self.add_attractor(
            'lorenz_center',
            LorenzAttractor(),
            position=(self.width // 2, self.height // 2),
            scale=0.5,
            influence_radius=200
        )
        
        # Rössler attractors in corners
        self.add_attractor(
            'rossler_nw',
            RosslerAttractor(),
            position=(self.width // 4, self.height // 4),
            scale=0.3,
            influence_radius=150
        )
        
        self.add_attractor(
            'rossler_se',
            RosslerAttractor(a=0.3, b=0.3, c=6.2),
            position=(3 * self.width // 4, 3 * self.height // 4),
            scale=0.3,
            influence_radius=150
        )
        
        # Chua attractor on the side
        self.add_attractor(
            'chua_east',
            ChuaAttractor(),
            position=(3 * self.width // 4, self.height // 2),
            scale=0.4,
            influence_radius=180
        )
        
        # Van der Pol attractor
        self.add_attractor(
            'vanderpol_west',
            VanDerPolAttractor(),
            position=(self.width // 4, self.height // 2),
            scale=0.2,
            influence_radius=120
        )
    
    def add_attractor(self, name: str, attractor, position: Tuple[float, float],
                     scale: float = 1.0, influence_radius: float = 100):
        """Add a chaotic attractor to the field"""
        self.attractors[name] = {
            'attractor': attractor,
            'position': position,
            'scale': scale,
            'influence_radius': influence_radius,
            'active': True
        }
        self.cache_valid = False
    
    def remove_attractor(self, name: str):
        """Remove an attractor from the field"""
        if name in self.attractors:
            del self.attractors[name]
            self.cache_valid = False
    
    def update(self):
        """Update all attractors and invalidate cache"""
        self.time_step += 1
        
        for name, attractor_data in self.attractors.items():
            if attractor_data['active']:
                attractor_data['attractor'].update()
        
        self.cache_valid = False
    
    def get_force_at_position(self, x: float, y: float) -> Tuple[float, float]:
        """Get combined force from all attractors at given position"""
        total_fx = 0.0
        total_fy = 0.0
        
        for name, attractor_data in self.attractors.items():
            if not attractor_data['active']:
                continue
                
            # Calculate distance to attractor
            ax, ay = attractor_data['position']
            distance = np.sqrt((x - ax)**2 + (y - ay)**2)
            
            # Check if within influence radius
            if distance < attractor_data['influence_radius']:
                # Get force from attractor
                attractor = attractor_data['attractor']
                fx, fy = attractor.get_force_2d(attractor_data['scale'])
                
                # Apply distance-based falloff
                influence = max(0, 1.0 - distance / attractor_data['influence_radius'])
                influence = influence**2  # Quadratic falloff
                
                # Add rotational component based on distance
                angle = math.atan2(y - ay, x - ax)
                rotational_strength = 0.1 * influence
                rot_fx = -math.sin(angle) * rotational_strength
                rot_fy = math.cos(angle) * rotational_strength
                
                total_fx += (fx + rot_fx) * influence
                total_fy += (fy + rot_fy) * influence
        
        return (total_fx, total_fy)
    
    def get_force_field_grid(self, resolution: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get force field as meshgrid for visualization"""
        x = np.linspace(0, self.width, resolution)
        y = np.linspace(0, self.height, resolution)
        X, Y = np.meshgrid(x, y)
        
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(resolution):
            for j in range(resolution):
                fx, fy = self.get_force_at_position(X[i, j], Y[i, j])
                U[i, j] = fx
                V[i, j] = fy
        
        return X, Y, U, V
    
    def add_turbulence(self, strength: float = 0.1):
        """Add random turbulence to the field"""
        # Modulate attractor parameters slightly
        for name, attractor_data in self.attractors.items():
            attractor = attractor_data['attractor']
            
            if hasattr(attractor, 'sigma'):  # Lorenz
                attractor.sigma += random.uniform(-strength, strength)
                attractor.sigma = max(1.0, min(20.0, attractor.sigma))
            
            elif hasattr(attractor, 'a') and hasattr(attractor, 'b'):  # Rössler
                attractor.a += random.uniform(-strength * 0.1, strength * 0.1)
                attractor.a = max(0.1, min(1.0, attractor.a))
            
            elif hasattr(attractor, 'mu'):  # Van der Pol
                attractor.mu += random.uniform(-strength, strength)
                attractor.mu = max(0.5, min(5.0, attractor.mu))
    
    def evolve_attractors(self, population_feedback: Dict[str, float]):
        """Evolve attractor parameters based on population feedback"""
        for name, attractor_data in self.attractors.items():
            # Get local population density near this attractor
            ax, ay = attractor_data['position']
            local_density = population_feedback.get(f'density_{name}', 0.5)
            
            # Evolve parameters based on density
            attractor = attractor_data['attractor']
            
            if hasattr(attractor, 'sigma'):  # Lorenz
                if local_density > 0.7:
                    attractor.sigma *= 1.01  # Increase chaos
                elif local_density < 0.3:
                    attractor.sigma *= 0.99  # Decrease chaos
            
            elif hasattr(attractor, 'c'):  # Rössler
                if local_density > 0.7:
                    attractor.c *= 1.005
                elif local_density < 0.3:
                    attractor.c *= 0.995
            
            # Adjust influence radius based on population
            current_radius = attractor_data['influence_radius']
            if local_density > 0.8:
                attractor_data['influence_radius'] = min(300, current_radius * 1.01)
            elif local_density < 0.2:
                attractor_data['influence_radius'] = max(50, current_radius * 0.99)
    
    def get_complexity_measure(self) -> float:
        """Calculate overall complexity of the attractor field"""
        # Sample force field at multiple points
        sample_points = 50
        forces = []
        
        for _ in range(sample_points):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            fx, fy = self.get_force_at_position(x, y)
            force_magnitude = np.sqrt(fx**2 + fy**2)
            forces.append(force_magnitude)
        
        # Complexity as standard deviation of force magnitudes
        return np.std(forces) if forces else 0.0
    
    def get_attractor_states(self) -> Dict[str, Dict]:
        """Get current states of all attractors"""
        states = {}
        for name, attractor_data in self.attractors.items():
            attractor = attractor_data['attractor']
            states[name] = {
                'state': attractor.state.copy(),
                'position': attractor_data['position'],
                'scale': attractor_data['scale'],
                'influence_radius': attractor_data['influence_radius'],
                'active': attractor_data['active']
            }
        return states
    
    def create_flow_lines(self, start_points: List[Tuple[float, float]], 
                         steps: int = 100, step_size: float = 0.5) -> List[List[Tuple[float, float]]]:
        """Generate flow lines showing attractor field dynamics"""
        flow_lines = []
        
        for start_x, start_y in start_points:
            line = [(start_x, start_y)]
            x, y = start_x, start_y
            
            for _ in range(steps):
                fx, fy = self.get_force_at_position(x, y)
                
                # Normalize force
                force_mag = np.sqrt(fx**2 + fy**2)
                if force_mag > 0:
                    fx = fx / force_mag * step_size
                    fy = fy / force_mag * step_size
                
                x += fx
                y += fy
                
                # Boundary conditions
                if x < 0 or x > self.width or y < 0 or y > self.height:
                    break
                
                line.append((x, y))
            
            flow_lines.append(line)
        
        return flow_lines
    
    def reset_attractors(self):
        """Reset all attractors to initial states"""
        for attractor_data in self.attractors.values():
            attractor = attractor_data['attractor']
            
            # Reset to random initial conditions
            if hasattr(attractor, 'state'):
                if len(attractor.state) == 3:
                    attractor.state = np.array([
                        random.uniform(-1, 1),
                        random.uniform(-1, 1),
                        random.uniform(-1, 1)
                    ])
                else:
                    attractor.state = np.array([
                        random.uniform(-1, 1),
                        random.uniform(-1, 1)
                    ])
            
            if hasattr(attractor, 'time'):
                attractor.time = 0.0