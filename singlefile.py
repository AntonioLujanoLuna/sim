# =============================================================================
# MODULAR EMERGENT INTELLIGENCE ECOSYSTEM
# =============================================================================
# A complex adaptive system with hierarchical social networks, communication
# evolution, environmental memory, and cognitive architecture
#
# File Structure (for future organization):
# ├── main.py                    # This file - main simulation runner
# ├── core/
# │   ├── simulation.py          # Core simulation engine
# │   ├── individual.py          # Individual agent class
# │   └── config.py              # Configuration management
# ├── environment/
# │   ├── attractors.py          # Chaotic attractor systems
# │   ├── information_field.py   # Information propagation
# │   └── ecosystem.py           # Environmental memory & co-evolution
# ├── social/
# │   ├── networks.py            # Social network dynamics
# │   ├── communication.py       # Language evolution system
# │   └── culture.py             # Cultural transmission
# ├── cognition/
# │   ├── perception.py          # Sensory systems
# │   ├── memory.py              # Memory architectures
# │   ├── planning.py            # Forward planning & metacognition
# │   └── learning.py            # Adaptive learning systems
# ├── visualization/
# │   ├── main_display.py        # Primary visualization
# │   ├── network_viz.py         # Social network visualization
# │   └── analytics.py           # Real-time analytics
# └── analysis/
#     ├── emergence_detection.py # Emergence & phase transition detection
#     ├── information_theory.py  # Information-theoretic measures
#     └── evolution_metrics.py   # Evolutionary dynamics analysis
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection, PatchCollection
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Any
import colorsys
from scipy.integrate import odeint
from collections import deque, defaultdict
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION SYSTEM
# =============================================================================

@dataclass
class Config:
    # Simulation parameters
    width: int = 1200
    height: int = 900
    max_population: int = 300
    initial_population: int = 80
    time_step: float = 0.1
    
    # Physical parameters
    separation_radius: float = 35
    alignment_radius: float = 70
    cohesion_radius: float = 90
    communication_radius: float = 120
    
    # Force parameters
    separation_force: float = 2.5
    alignment_force: float = 1.0
    cohesion_force: float = 0.8
    social_force: float = 1.2
    environmental_force: float = 0.6
    
    # Cognitive parameters
    memory_length: int = 100
    planning_horizon: int = 20
    attention_span: int = 5
    learning_rate: float = 0.1
    
    # Social parameters
    max_social_connections: int = 8
    relationship_decay: float = 0.02
    trust_threshold: float = 0.6
    communication_mutation_rate: float = 0.05
    
    # Evolution parameters
    breeding_energy_threshold: float = 70
    mutation_rate: float = 0.12
    sexual_selection_strength: float = 0.4
    cultural_inheritance_rate: float = 0.3
    
    # Environmental parameters
    resource_regeneration_rate: float = 0.1
    environmental_memory_length: int = 500
    co_evolution_strength: float = 0.2

# =============================================================================
# COMMUNICATION & LANGUAGE EVOLUTION SYSTEM
# =============================================================================

class Signal:
    """Individual communication signal with meaning evolution"""
    def __init__(self, signal_id: int, intensity: float = 1.0):
        self.id = signal_id
        self.intensity = intensity  # Signal strength
        self.meaning_vector = np.random.random(5)  # Abstract meaning space
        self.usage_count = 0
        self.success_rate = 0.5

class CommunicationSystem:
    """Evolving communication and proto-language system"""
    def __init__(self):
        self.signal_repertoire = {}  # id -> Signal
        self.meaning_associations = defaultdict(list)  # context -> [signal_ids]
        self.syntax_rules = []  # Emerging grammar rules
        self.next_signal_id = 0
        
        # Initialize basic signals
        self._initialize_basic_signals()
    
    def _initialize_basic_signals(self):
        """Create basic survival signals"""
        basic_meanings = ['danger', 'food', 'mating', 'help', 'territory']
        for meaning in basic_meanings:
            signal = Signal(self.next_signal_id)
            self.signal_repertoire[self.next_signal_id] = signal
            self.meaning_associations[meaning].append(self.next_signal_id)
            self.next_signal_id += 1
    
    def create_new_signal(self, parent_signals: List[int] = None):
        """Create new signal through combination or mutation"""
        signal = Signal(self.next_signal_id)
        
        if parent_signals and len(parent_signals) >= 2:
            # Compositional signal creation
            parent1 = self.signal_repertoire[parent_signals[0]]
            parent2 = self.signal_repertoire[parent_signals[1]]
            signal.meaning_vector = (parent1.meaning_vector + parent2.meaning_vector) / 2
            signal.meaning_vector += np.random.normal(0, 0.1, 5)  # Mutation
        
        self.signal_repertoire[self.next_signal_id] = signal
        self.next_signal_id += 1
        return signal.id
    
    def interpret_signal(self, signal_id: int, context: str) -> float:
        """Interpret signal meaning in given context"""
        if signal_id not in self.signal_repertoire:
            return 0.0
        
        signal = self.signal_repertoire[signal_id]
        # Simplified meaning interpretation
        context_hash = hash(context) % 5
        return signal.meaning_vector[context_hash] * signal.intensity
    
    def update_signal_success(self, signal_id: int, success: bool):
        """Update signal based on communication success"""
        if signal_id in self.signal_repertoire:
            signal = self.signal_repertoire[signal_id]
            signal.usage_count += 1
            signal.success_rate = (signal.success_rate * 0.9 + (1.0 if success else 0.0) * 0.1)

# =============================================================================
# SOCIAL NETWORK SYSTEM
# =============================================================================

class SocialRelationship:
    """Individual social relationship with history"""
    def __init__(self, individual_id: int):
        self.individual_id = individual_id
        self.strength = 0.5  # Relationship strength
        self.trust = 0.5     # Trust level
        self.interaction_history = deque(maxlen=50)
        self.shared_experiences = 0
        self.last_interaction_time = 0
        self.relationship_type = 'neutral'  # 'friend', 'rival', 'mate', 'kin'

class SocialNetwork:
    """Dynamic social network with emergent hierarchy"""
    def __init__(self):
        self.relationships = defaultdict(dict)  # individual_id -> {other_id: SocialRelationship}
        self.interaction_graph = nx.DiGraph()
        self.communities = []
        self.leaders = set()
        self.influence_network = defaultdict(float)
    
    def add_individual(self, individual_id: int):
        """Add individual to social network"""
        if individual_id not in self.relationships:
            self.relationships[individual_id] = {}
            self.interaction_graph.add_node(individual_id)
    
    def update_relationship(self, id1: int, id2: int, interaction_type: str, 
                          success: bool, time_step: int):
        """Update relationship based on interaction"""
        # Ensure both individuals exist in network
        self.add_individual(id1)
        self.add_individual(id2)
        
        # Update relationship from id1's perspective
        if id2 not in self.relationships[id1]:
            self.relationships[id1][id2] = SocialRelationship(id2)
        
        rel = self.relationships[id1][id2]
        rel.last_interaction_time = time_step
        rel.interaction_history.append((interaction_type, success, time_step))
        
        # Update trust and strength based on interaction
        if success:
            rel.trust = min(1.0, rel.trust + 0.1)
            rel.strength = min(1.0, rel.strength + 0.05)
        else:
            rel.trust = max(0.0, rel.trust - 0.05)
        
        # Update interaction graph
        if self.interaction_graph.has_edge(id1, id2):
            self.interaction_graph[id1][id2]['weight'] += 1
        else:
            self.interaction_graph.add_edge(id1, id2, weight=1)
    
    def get_relationship_strength(self, id1: int, id2: int) -> float:
        """Get relationship strength between two individuals"""
        if id1 in self.relationships and id2 in self.relationships[id1]:
            return self.relationships[id1][id2].strength
        return 0.0
    
    def decay_relationships(self, time_step: int, decay_rate: float = 0.02):
        """Decay unused relationships over time"""
        for individual_id, relationships in self.relationships.items():
            for other_id, rel in list(relationships.items()):
                time_since_interaction = time_step - rel.last_interaction_time
                if time_since_interaction > 100:  # Haven't interacted recently
                    rel.strength = max(0.0, rel.strength - decay_rate)
                    rel.trust = max(0.0, rel.trust - decay_rate * 0.5)
                    
                    # Remove very weak relationships
                    if rel.strength < 0.1:
                        del relationships[other_id]
                        if self.interaction_graph.has_edge(individual_id, other_id):
                            self.interaction_graph.remove_edge(individual_id, other_id)
    
    def detect_communities(self):
        """Detect communities using network analysis"""
        if len(self.interaction_graph.nodes()) > 3:
            try:
                self.communities = list(nx.community.greedy_modularity_communities(
                    self.interaction_graph.to_undirected()))
            except:
                self.communities = []
    
    def identify_leaders(self):
        """Identify influential individuals in the network"""
        self.leaders.clear()
        if len(self.interaction_graph.nodes()) > 0:
            # Use centrality measures to identify leaders
            centrality = nx.degree_centrality(self.interaction_graph)
            threshold = np.mean(list(centrality.values())) + np.std(list(centrality.values()))
            
            for node, score in centrality.items():
                if score > threshold:
                    self.leaders.add(node)

# =============================================================================
# ENVIRONMENTAL MEMORY & CO-EVOLUTION SYSTEM
# =============================================================================

class EnvironmentalPatch:
    """Environmental patch with memory and adaptation"""
    def __init__(self, x: float, y: float, patch_type: str = 'neutral'):
        self.x = x
        self.y = y
        self.patch_type = patch_type  # 'food', 'danger', 'shelter', 'neutral'
        self.resource_level = 1.0
        self.quality = 1.0
        self.visitation_history = deque(maxlen=200)
        self.species_preferences = defaultdict(float)
        self.modification_level = 0.0  # How much species have modified this patch
        
    def update_from_visitation(self, visitor_species: str, visitor_energy: float, 
                             time_step: int):
        """Update patch based on visitation"""
        self.visitation_history.append((visitor_species, visitor_energy, time_step))
        
        # Species modify environment
        if visitor_species in ['herbivore', 'scavenger']:
            self.resource_level = max(0.0, self.resource_level - 0.05)
        elif visitor_species == 'predator':
            # Predators might scare away prey, affecting resource regeneration
            self.quality = max(0.5, self.quality - 0.02)
        
        # Track species preferences
        self.species_preferences[visitor_species] += 0.1
        
        # Environmental modification
        self.modification_level = min(1.0, self.modification_level + 0.01)
    
    def regenerate(self, base_rate: float):
        """Natural resource regeneration"""
        if self.patch_type == 'food':
            # Regeneration affected by modification level
            regen_rate = base_rate * (1.0 - 0.5 * self.modification_level)
            self.resource_level = min(1.0, self.resource_level + regen_rate)
        
        # Quality slowly recovers
        self.quality = min(1.0, self.quality + 0.001)

class EnvironmentalMemory:
    """Environment with memory and co-evolutionary dynamics"""
    def __init__(self, width: int, height: int, patch_density: float = 0.001):
        self.width = width
        self.height = height
        self.patches = []
        self.spatial_grid = {}  # For spatial indexing
        self.grid_size = 50
        
        # Create initial patches
        num_patches = int(width * height * patch_density)
        for _ in range(num_patches):
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            patch_type = random.choice(['food', 'shelter', 'neutral', 'neutral'])
            patch = EnvironmentalPatch(x, y, patch_type)
            self.patches.append(patch)
            self._add_to_grid(patch)
    
    def _add_to_grid(self, patch: EnvironmentalPatch):
        """Add patch to spatial grid for efficient lookup"""
        grid_x = int(patch.x // self.grid_size)
        grid_y = int(patch.y // self.grid_size)
        
        if (grid_x, grid_y) not in self.spatial_grid:
            self.spatial_grid[(grid_x, grid_y)] = []
        
        self.spatial_grid[(grid_x, grid_y)].append(patch)
    
    def get_nearby_patches(self, x: float, y: float, radius: float) -> List[EnvironmentalPatch]:
        """Get patches within radius of position"""
        nearby = []
        grid_x = int(x // self.grid_size)
        grid_y = int(y // self.grid_size)
        
        # Check surrounding grid cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (grid_x + dx, grid_y + dy)
                if cell in self.spatial_grid:
                    for patch in self.spatial_grid[cell]:
                        dist = np.sqrt((x - patch.x)**2 + (y - patch.y)**2)
                        if dist <= radius:
                            nearby.append(patch)
        
        return nearby
    
    def update(self, individuals: List, time_step: int, regeneration_rate: float):
        """Update environmental state based on individual interactions"""
        # Record visitations
        for individual in individuals:
            nearby_patches = self.get_nearby_patches(individual.x, individual.y, 20)
            for patch in nearby_patches:
                patch.update_from_visitation(individual.species_name, 
                                           individual.energy, time_step)
        
        # Regenerate all patches
        for patch in self.patches:
            patch.regenerate(regeneration_rate)

# =============================================================================
# ENHANCED COGNITIVE ARCHITECTURE
# =============================================================================

class AttentionModule:
    """Attention and perception filtering system"""
    def __init__(self):
        self.attention_weights = {
            'social': 0.3,
            'environmental': 0.3,
            'danger': 0.4
        }
        self.current_focus = None
        self.attention_history = deque(maxlen=20)
    
    def update_attention(self, stimuli: Dict[str, float]):
        """Update attention based on stimuli salience"""
        max_stimulus = max(stimuli.items(), key=lambda x: x[1] * self.attention_weights.get(x[0], 0.1))
        self.current_focus = max_stimulus[0]
        self.attention_history.append((self.current_focus, max_stimulus[1]))
    
    def get_filtered_perception(self, raw_perception: Dict) -> Dict:
        """Filter perception based on current attention"""
        filtered = {}
        for key, value in raw_perception.items():
            if key == self.current_focus:
                filtered[key] = value * 1.5  # Amplify attended stimulus
            else:
                filtered[key] = value * 0.5  # Diminish unattended stimuli
        
        return filtered

class PlanningModule:
    """Forward planning and scenario simulation"""
    def __init__(self, planning_horizon: int = 20):
        self.planning_horizon = planning_horizon
        self.current_plan = []
        self.scenario_cache = {}
        self.success_history = deque(maxlen=50)
    
    def create_plan(self, current_state: Dict, goal: str, environment_model: Dict):
        """Create action plan using simplified forward simulation"""
        plan = []
        
        # Simple goal-directed planning
        if goal == 'find_food':
            if 'food_locations' in environment_model:
                closest_food = min(environment_model['food_locations'], 
                                 key=lambda loc: loc['distance'])
                plan = [('move_to', closest_food['position'])]
        
        elif goal == 'avoid_danger':
            if 'danger_locations' in environment_model:
                # Plan escape route
                plan = [('move_away', environment_model['danger_locations'][0]['position'])]
        
        elif goal == 'socialize':
            if 'social_opportunities' in environment_model:
                plan = [('approach', environment_model['social_opportunities'][0]['individual'])]
        
        self.current_plan = plan
        return plan
    
    def evaluate_plan_success(self, outcome: bool):
        """Learn from plan execution results"""
        self.success_history.append(outcome)

class MetacognitionModule:
    """Self-awareness and learning about learning"""
    def __init__(self):
        self.self_model = {
            'strengths': defaultdict(float),
            'weaknesses': defaultdict(float),
            'learning_rate': 0.1,
            'confidence': 0.5
        }
        self.strategy_success = defaultdict(list)
    
    def update_self_model(self, action: str, outcome: bool, context: str):
        """Update self-understanding based on experience"""
        if outcome:
            self.self_model['strengths'][context] += 0.1
            self.self_model['confidence'] = min(1.0, self.self_model['confidence'] + 0.02)
        else:
            self.self_model['weaknesses'][context] += 0.1
            self.self_model['confidence'] = max(0.0, self.self_model['confidence'] - 0.02)
        
        self.strategy_success[action].append(outcome)
    
    def adapt_learning_rate(self):
        """Adapt learning rate based on performance"""
        recent_success_rate = np.mean([1.0 if outcome else 0.0 
                                     for outcomes in self.strategy_success.values() 
                                     for outcome in list(outcomes)[-10:]])
        
        if recent_success_rate > 0.7:
            self.self_model['learning_rate'] *= 0.95  # Slow down when doing well
        elif recent_success_rate < 0.3:
            self.self_model['learning_rate'] *= 1.05  # Speed up when struggling

# =============================================================================
# ENHANCED INDIVIDUAL WITH COGNITIVE ARCHITECTURE
# =============================================================================

class EnhancedIndividual:
    """Individual with advanced cognitive architecture and social capabilities"""
    
    def __init__(self, x: float, y: float, species_name: str = 'herbivore', 
                 parent1=None, parent2=None, individual_id: int = None):
        # Basic properties
        self.id = individual_id or random.randint(100000, 999999)
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.species_name = species_name
        
        # Cognitive architecture
        self.attention = AttentionModule()
        self.planning = PlanningModule()
        self.metacognition = MetacognitionModule()
        
        # Communication system
        self.communication = CommunicationSystem()
        self.active_signals = []  # Currently broadcasting signals
        
        # Enhanced memory systems
        self.spatial_memory = deque(maxlen=Config.memory_length)
        self.social_memory = {}  # individual_id -> interaction memories
        self.environmental_memory = {}  # location -> environmental info
        
        # Social and cultural traits
        self.cultural_knowledge = defaultdict(float)  # Learned cultural information
        self.teaching_ability = random.uniform(0.0, 1.0)
        self.innovation_tendency = random.uniform(0.0, 1.0)
        
        # Enhanced inheritance
        if parent1 and parent2:
            self._inherit_from_parents(parent1, parent2)
        else:
            self._initialize_traits()
        
        # State variables
        self.energy = random.uniform(60, 100)
        self.age = 0
        self.generation = 1 if not parent1 else max(parent1.generation, parent2.generation) + 1
        self.stress_level = 0.0
        self.current_goal = 'explore'
        
        # Visual properties
        self.color = self._generate_color()
        self.trail = deque(maxlen=40)
        self.size = random.uniform(5, 12)
    
    def _inherit_from_parents(self, parent1, parent2):
        """Inherit traits from parents with cultural transmission"""
        # Genetic inheritance
        self.intelligence = self._inherit_trait(parent1.intelligence, parent2.intelligence, 0, 1)
        self.sociability = self._inherit_trait(parent1.sociability, parent2.sociability, 0, 1)
        self.aggression = self._inherit_trait(parent1.aggression, parent2.aggression, 0, 1)
        self.max_speed = self._inherit_trait(parent1.max_speed, parent2.max_speed, 1.0, 4.0)
        
        # Cultural inheritance
        combined_knowledge = {}
        for knowledge in [parent1.cultural_knowledge, parent2.cultural_knowledge]:
            for key, value in knowledge.items():
                combined_knowledge[key] = combined_knowledge.get(key, 0) + value * 0.5
        
        # Inherit cultural knowledge with some mutation
        for key, value in combined_knowledge.items():
            if random.random() < Config.cultural_inheritance_rate:
                mutation = random.uniform(-0.1, 0.1)
                self.cultural_knowledge[key] = max(0, value + mutation)
        
        # Inherit some communication patterns
        parent_signals = list(parent1.communication.signal_repertoire.keys())[:3]
        for signal_id in parent_signals:
            if random.random() < 0.7:  # 70% chance to inherit signal
                original_signal = parent1.communication.signal_repertoire[signal_id]
                new_signal = Signal(self.communication.next_signal_id)
                new_signal.meaning_vector = original_signal.meaning_vector.copy()
                # Add mutation
                new_signal.meaning_vector += np.random.normal(0, 0.05, 5)
                self.communication.signal_repertoire[self.communication.next_signal_id] = new_signal
                self.communication.next_signal_id += 1
    
    def _initialize_traits(self):
        """Initialize traits for first generation"""
        self.intelligence = random.uniform(0.2, 0.8)
        self.sociability = random.uniform(0.1, 0.9)
        self.aggression = random.uniform(0.0, 0.6)
        self.max_speed = random.uniform(1.5, 3.5)
    
    def _inherit_trait(self, parent1_val: float, parent2_val: float, 
                      min_val: float, max_val: float) -> float:
        """Inherit trait with mutation"""
        # Choose parent gene
        base_val = parent1_val if random.random() < 0.5 else parent2_val
        
        # Apply mutation
        if random.random() < Config.mutation_rate:
            mutation_strength = (max_val - min_val) * 0.1
            mutation = random.uniform(-mutation_strength, mutation_strength)
            base_val += mutation
        
        return np.clip(base_val, min_val, max_val)
    
    def perceive_environment(self, others: List, environment, social_network):
        """Advanced environmental perception with attention filtering"""
        raw_perception = {
            'nearby_individuals': [],
            'environmental_features': [],
            'social_information': [],
            'danger_signals': [],
            'opportunity_signals': []
        }
        
        # Perceive nearby individuals
        for other in others:
            if other.id != self.id:
                dx = other.x - self.x
                dy = other.y - self.y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < Config.communication_radius:
                    perception = {
                        'individual': other,
                        'distance': distance,
                        'relationship_strength': social_network.get_relationship_strength(self.id, other.id),
                        'species': other.species_name,
                        'energy_level': other.energy / 100.0,
                        'active_signals': other.active_signals.copy()
                    }
                    raw_perception['nearby_individuals'].append(perception)
        
        # Perceive environmental patches
        nearby_patches = environment.get_nearby_patches(self.x, self.y, 50)
        for patch in nearby_patches:
            perception = {
                'patch': patch,
                'distance': np.sqrt((self.x - patch.x)**2 + (self.y - patch.y)**2),
                'type': patch.patch_type,
                'quality': patch.quality,
                'resource_level': patch.resource_level
            }
            raw_perception['environmental_features'].append(perception)
        
        # Process signals from others
        for individual_perception in raw_perception['nearby_individuals']:
            for signal_id in individual_perception['active_signals']:
                meaning = self.communication.interpret_signal(signal_id, 'general')
                if meaning > 0.5:  # Significant signal
                    raw_perception['social_information'].append({
                        'sender': individual_perception['individual'],
                        'signal': signal_id,
                        'meaning': meaning,
                        'trust_level': individual_perception['relationship_strength']
                    })
        
        # Compute attention weights
        stimuli_salience = {
            'social': len(raw_perception['nearby_individuals']) * 0.1,
            'environmental': len(raw_perception['environmental_features']) * 0.1,
            'danger': len(raw_perception['danger_signals']) * 0.3
        }
        
        self.attention.update_attention(stimuli_salience)
        filtered_perception = self.attention.get_filtered_perception(raw_perception)
        
        return filtered_perception
    
    def make_decisions(self, perception: Dict):
        """High-level decision making with planning"""
        # Update current goal based on needs and perception
        if self.energy < 30:
            self.current_goal = 'find_food'
        elif self.stress_level > 0.7:
            self.current_goal = 'avoid_danger'
        elif len(perception.get('nearby_individuals', [])) > 0 and self.sociability > 0.6:
            self.current_goal = 'socialize'
        else:
            self.current_goal = 'explore'
        
        # Create environment model for planning
        environment_model = {
            'food_locations': [
                {'position': (p['patch'].x, p['patch'].y), 'distance': p['distance']}
                for p in perception.get('environmental_features', [])
                if p['type'] == 'food' and p['resource_level'] > 0.3
            ],
            'social_opportunities': [
                {'individual': p['individual'], 'distance': p['distance']}
                for p in perception.get('nearby_individuals', [])
                if p['relationship_strength'] > 0.3
            ]
        }
        
        # Create plan
        plan = self.planning.create_plan(
            {'position': (self.x, self.y), 'energy': self.energy},
            self.current_goal,
            environment_model
        )
        
        return plan
    
    def communicate(self, perception: Dict, social_network):
        """Generate communication signals based on context"""
        self.active_signals.clear()
        
        # Danger warning
        if self.stress_level > 0.6:
            danger_signal = self.communication.meaning_associations.get('danger', [])
            if danger_signal:
                self.active_signals.append(danger_signal[0])
        
        # Food sharing (if high energy and social)
        if self.energy > 80 and self.sociability > 0.7:
            food_signals = self.communication.meaning_associations.get('food', [])
            if food_signals:
                self.active_signals.append(food_signals[0])
        
        # Mating calls
        if self.energy > 70 and self.age > 100:
            mating_signals = self.communication.meaning_associations.get('mating', [])
            if mating_signals:
                self.active_signals.append(mating_signals[0])
        
        # Innovation: Create new signals occasionally
        if (random.random() < 0.001 * self.innovation_tendency and 
            len(self.communication.signal_repertoire) < 20):
            
            existing_signals = list(self.communication.signal_repertoire.keys())
            if len(existing_signals) >= 2:
                parent_signals = random.sample(existing_signals, 2)
                new_signal_id = self.communication.create_new_signal(parent_signals)
                self.active_signals.append(new_signal_id)
    
    def learn_from_interactions(self, interactions: List[Tuple], social_network):
        """Learn from social interactions and update knowledge"""
        for interaction_type, other_individual, success in interactions:
            # Update social memory
            if other_individual.id not in self.social_memory:
                self.social_memory[other_individual.id] = []
            
            self.social_memory[other_individual.id].append({
                'type': interaction_type,
                'success': success,
                'time': self.age,
                'context': self.current_goal
            })
            
            # Cultural learning
            if success and interaction_type == 'communication':
                # Learn from successful individuals
                if other_individual.energy > self.energy:
                    for knowledge, value in other_individual.cultural_knowledge.items():
                        learning_rate = self.intelligence * 0.1
                        self.cultural_knowledge[knowledge] += value * learning_rate
            
            # Update metacognition
            self.metacognition.update_self_model(interaction_type, success, self.current_goal)
            
            # Update communication system
            if interaction_type == 'communication':
                for signal_id in other_individual.active_signals:
                    self.communication.update_signal_success(signal_id, success)
    
    def update_physics(self, others: List, environment, social_network, config: Config):
        """Update position and state with enhanced cognitive processing"""
        # Perceive environment
        perception = self.perceive_environment(others, environment, social_network)
        
        # Make decisions
        plan = self.make_decisions(perception)
        
        # Communicate
        self.communicate(perception, social_network)
        
        # Calculate forces based on plan and perception
        forces = self._calculate_forces(perception, plan, config)
        
        # Update velocity and position
        self.vx += forces[0] * config.time_step
        self.vy += forces[1] * config.time_step
        
        # Limit speed
        speed = np.sqrt(self.vx**2 + self.vy**2)
        if speed > self.max_speed:
            self.vx = (self.vx / speed) * self.max_speed
            self.vy = (self.vy / speed) * self.max_speed
        
        # Update position with wraparound
        self.x = (self.x + self.vx) % config.width
        self.y = (self.y + self.vy) % config.height
        
        # Update memories
        self.spatial_memory.append((self.x, self.y, self.current_goal))
        
        # Update state
        self._update_state(config)
        
        # Update color
        self.color = self._generate_color()
        
        # Update trail
        self.trail.append((self.x, self.y))
    
    def _calculate_forces(self, perception: Dict, plan: List, config: Config) -> Tuple[float, float]:
        """Calculate movement forces based on perception and plan"""
        total_fx = total_fy = 0.0
        
        # Plan-based forces
        if plan:
            action, target = plan[0]
            if action == 'move_to':
                dx = target[0] - self.x
                dy = target[1] - self.y
                # Handle wraparound
                if abs(dx) > config.width / 2:
                    dx = dx - np.sign(dx) * config.width
                if abs(dy) > config.height / 2:
                    dy = dy - np.sign(dy) * config.height
                
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    total_fx += (dx / dist) * 2.0
                    total_fy += (dy / dist) * 2.0
            
            elif action == 'move_away':
                dx = self.x - target[0]
                dy = self.y - target[1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    total_fx += (dx / dist) * 3.0
                    total_fy += (dy / dist) * 3.0
        
        # Social forces based on relationships
        for individual_data in perception.get('nearby_individuals', []):
            other = individual_data['individual']
            relationship_strength = individual_data['relationship_strength']
            distance = individual_data['distance']
            
            if distance > 0:
                dx = other.x - self.x
                dy = other.y - self.y
                
                # Handle wraparound
                if abs(dx) > config.width / 2:
                    dx = dx - np.sign(dx) * config.width
                if abs(dy) > config.height / 2:
                    dy = dy - np.sign(dy) * config.height
                
                # Social attraction/repulsion based on relationship
                if relationship_strength > 0.5:  # Friends attract
                    force = relationship_strength * 0.5 / (distance + 1)
                    total_fx += (dx / distance) * force
                    total_fy += (dy / distance) * force
                elif relationship_strength < 0.2:  # Rivals repel
                    force = -0.5 / (distance + 1)
                    total_fx += (dx / distance) * force
                    total_fy += (dy / distance) * force
        
        # Add some random exploration
        if self.current_goal == 'explore':
            total_fx += random.uniform(-0.5, 0.5)
            total_fy += random.uniform(-0.5, 0.5)
        
        return (total_fx, total_fy)
    
    def _update_state(self, config: Config):
        """Update energy, age, stress, and other state variables"""
        # Energy consumption
        movement_cost = np.sqrt(self.vx**2 + self.vy**2) * 0.02
        cognitive_cost = (self.intelligence + len(self.active_signals) * 0.1) * 0.01
        social_cost = len(self.social_memory) * 0.001
        
        self.energy -= movement_cost + cognitive_cost + social_cost + 0.1
        self.energy = max(0, min(self.energy, 100))
        
        # Age
        self.age += 1
        
        # Stress adaptation
        if self.current_goal == 'avoid_danger':
            self.stress_level = min(1.0, self.stress_level + 0.1)
        else:
            self.stress_level = max(0.0, self.stress_level - 0.05)
        
        # Metacognitive adaptation
        self.metacognition.adapt_learning_rate()
    
    def _generate_color(self) -> Tuple[float, float, float]:
        """Generate color based on current state"""
        base_colors = {
            'predator': (0.8, 0.2, 0.2),
            'herbivore': (0.2, 0.8, 0.2),
            'scavenger': (0.8, 0.8, 0.2),
            'mystic': (0.6, 0.2, 0.8)
        }
        
        base_r, base_g, base_b = base_colors.get(self.species_name, (0.5, 0.5, 0.5))
        
        # Modify based on state
        energy_factor = self.energy / 100.0
        intelligence_factor = self.intelligence
        social_factor = len(self.social_memory) / 10.0
        
        r = np.clip(base_r + (self.stress_level - 0.5) * 0.3, 0, 1)
        g = np.clip(base_g * energy_factor + intelligence_factor * 0.2, 0, 1)
        b = np.clip(base_b + social_factor * 0.1, 0, 1)
        
        return (r, g, b)
    
    def can_breed(self, other, config: Config) -> bool:
        """Enhanced breeding conditions"""
        if (self.species_name != other.species_name or
            self.energy < config.breeding_energy_threshold or
            other.energy < config.breeding_energy_threshold):
            return False
        
        distance = np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        if distance > 30:
            return False
        
        # Social compatibility
        relationship_strength = 0.5  # Default for unknown individuals
        if other.id in self.social_memory:
            recent_interactions = self.social_memory[other.id][-5:]
            if recent_interactions:
                relationship_strength = np.mean([
                    1.0 if interaction['success'] else 0.0 
                    for interaction in recent_interactions
                ])
        
        return relationship_strength > 0.4 and random.random() < 0.3
    
    def breed(self, other, config: Config):
        """Create offspring with enhanced inheritance"""
        offspring_x = (self.x + other.x) / 2
        offspring_y = (self.y + other.y) / 2
        
        offspring = EnhancedIndividual(offspring_x, offspring_y, self.species_name, self, other)
        
        # Breeding costs
        self.energy -= 30
        other.energy -= 30
        
        return offspring
    
    def is_alive(self) -> bool:
        """Check if individual is still alive"""
        max_age = {'predator': 1000, 'herbivore': 800, 'scavenger': 600, 'mystic': 1200}
        return self.energy > 0 and self.age < max_age.get(self.species_name, 800)

# =============================================================================
# MAIN SIMULATION ENGINE
# =============================================================================

class EmergentIntelligenceSimulation:
    """Main simulation with all advanced systems integrated"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.time_step = 0
        self.individuals = []
        self.social_network = SocialNetwork()
        self.environment = EnvironmentalMemory(self.config.width, self.config.height)
        
        # Statistics and analysis
        self.generation_stats = []
        self.emergence_events = []
        self.cultural_evolution_data = []
        
        # Initialize population
        self._initialize_population()
        
        # Visualization setup
        self.fig = plt.figure(figsize=(18, 12))
        self._setup_visualization()
    
    def _initialize_population(self):
        """Initialize diverse population"""
        species_distribution = {'herbivore': 0.4, 'predator': 0.15, 'scavenger': 0.35, 'mystic': 0.1}
        
        for i in range(self.config.initial_population):
            species = np.random.choice(list(species_distribution.keys()), 
                                     p=list(species_distribution.values()))
            x = random.uniform(0, self.config.width)
            y = random.uniform(0, self.config.height)
            individual = EnhancedIndividual(x, y, species, individual_id=i)
            self.individuals.append(individual)
            self.social_network.add_individual(individual.id)
    
    def _setup_visualization(self):
        """Setup comprehensive visualization system"""
        # Main simulation view
        self.ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
        
        # Social network view
        self.ax_social = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
        
        # Population statistics
        self.ax_stats = plt.subplot2grid((4, 4), (2, 0))
        
        # Communication evolution
        self.ax_comm = plt.subplot2grid((4, 4), (2, 1))
        
        # Cultural knowledge
        self.ax_culture = plt.subplot2grid((4, 4), (2, 2))
        
        # Environmental state
        self.ax_env = plt.subplot2grid((4, 4), (2, 3))
        
        # Intelligence evolution
        self.ax_intel = plt.subplot2grid((4, 4), (3, 0))
        
        # Social complexity
        self.ax_social_metrics = plt.subplot2grid((4, 4), (3, 1))
        
        # Phase space
        self.ax_phase = plt.subplot2grid((4, 4), (3, 2))
        
        # Emergence events
        self.ax_emergence = plt.subplot2grid((4, 4), (3, 3))
    
    def update(self, frame):
        """Main simulation update with comprehensive analysis"""
        self.time_step += 1
        
        # Update environment
        self.environment.update(self.individuals, self.time_step, 
                              self.config.resource_regeneration_rate)
        
        # Update all individuals
        interactions = []
        for individual in self.individuals[:]:
            individual.update_physics(self.individuals, self.environment, 
                                    self.social_network, self.config)
            
            # Record interactions for social network updates
            nearby = [other for other in self.individuals 
                     if other.id != individual.id and 
                     np.sqrt((individual.x - other.x)**2 + (individual.y - other.y)**2) < 50]
            
            for other in nearby:
                # Communication interaction
                if individual.active_signals and other.active_signals:
                    success = random.random() < 0.7  # Communication success probability
                    interactions.append((individual.id, other.id, 'communication', success))
                
                # Cooperation attempts
                if (individual.sociability > 0.6 and other.sociability > 0.6 and 
                    random.random() < 0.1):
                    success = random.random() < (individual.sociability + other.sociability) / 2
                    interactions.append((individual.id, other.id, 'cooperation', success))
        
        # Update social network based on interactions
        for id1, id2, interaction_type, success in interactions:
            self.social_network.update_relationship(id1, id2, interaction_type, success, self.time_step)
        
        # Update individuals with interaction learning
        for individual in self.individuals:
            individual_interactions = [(itype, self._get_individual_by_id(id2), success) 
                                     for id1, id2, itype, success in interactions if id1 == individual.id]
            individual.learn_from_interactions(individual_interactions, self.social_network)
        
        # Social network analysis
        self.social_network.decay_relationships(self.time_step, self.config.relationship_decay)
        self.social_network.detect_communities()
        self.social_network.identify_leaders()
        
        # Breeding and evolution
        self._handle_breeding()
        
        # Remove dead individuals
        dead_individuals = [ind for ind in self.individuals if not ind.is_alive()]
        for dead_ind in dead_individuals:
            self.individuals.remove(dead_ind)
            # Remove from social network
            if dead_ind.id in self.social_network.relationships:
                del self.social_network.relationships[dead_ind.id]
        
        # Population management
        if len(self.individuals) > self.config.max_population:
            # Keep fittest individuals
            self.individuals.sort(key=lambda x: x.energy + x.intelligence * 20 + 
                                               len(x.social_memory) * 5, reverse=True)
            self.individuals = self.individuals[:self.config.max_population]
        
        # Detect emergent phenomena
        self._detect_emergence()
        
        # Update statistics
        self._update_statistics()
        
        # Render visualization
        self._render_all()
        
        return []
    
    def _get_individual_by_id(self, individual_id: int):
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
            
            potential_mates = [other for other in self.individuals[i+1:] 
                             if individual.can_breed(other, self.config)]
            
            if potential_mates and random.random() < 0.003:  # Low breeding rate
                # Choose mate based on relationship strength and compatibility
                mate_scores = []
                for mate in potential_mates:
                    relationship_strength = self.social_network.get_relationship_strength(
                        individual.id, mate.id)
                    compatibility = (individual.intelligence + mate.intelligence + 
                                   individual.sociability + mate.sociability) / 4
                    score = relationship_strength + compatibility
                    mate_scores.append(score)
                
                if mate_scores:
                    # Probabilistic selection based on scores
                    total_score = sum(mate_scores)
                    if total_score > 0:
                        probabilities = [score / total_score for score in mate_scores]
                        chosen_mate = np.random.choice(potential_mates, p=probabilities)
                        
                        offspring = individual.breed(chosen_mate, self.config)
                        new_offspring.append(offspring)
                        self.social_network.add_individual(offspring.id)
        
        self.individuals.extend(new_offspring)
    
    def _detect_emergence(self):
        """Detect emergent phenomena and phase transitions"""
        if len(self.individuals) < 10:
            return
        
        # Detect communication breakthroughs
        total_signals = sum(len(ind.communication.signal_repertoire) for ind in self.individuals)
        avg_signals = total_signals / len(self.individuals)
        
        if avg_signals > 10 and len(self.emergence_events) == 0:
            self.emergence_events.append(('communication_breakthrough', self.time_step))
        
        # Detect social complexity emergence
        if len(self.social_network.communities) > len(self.individuals) / 10:
            if not any(event[0] == 'social_complexity' for event in self.emergence_events[-5:]):
                self.emergence_events.append(('social_complexity', self.time_step))
        
        # Detect intelligence boom
        avg_intelligence = np.mean([ind.intelligence for ind in self.individuals])
        if len(self.generation_stats) > 50:
            prev_avg_intelligence = self.generation_stats[-50]['avg_intelligence']
            if avg_intelligence > prev_avg_intelligence + 0.2:
                self.emergence_events.append(('intelligence_boom', self.time_step))
        
        # Detect cultural accumulation
        total_cultural_knowledge = sum(sum(ind.cultural_knowledge.values()) 
                                     for ind in self.individuals)
        if total_cultural_knowledge > len(self.individuals) * 5:
            if not any(event[0] == 'cultural_accumulation' for event in self.emergence_events[-10:]):
                self.emergence_events.append(('cultural_accumulation', self.time_step))
    
    def _update_statistics(self):
        """Update comprehensive statistics"""
        if not self.individuals:
            return
        
        # Basic statistics
        species_counts = defaultdict(int)
        total_energy = 0
        intelligence_levels = []
        sociability_levels = []
        cultural_knowledge_levels = []
        
        for ind in self.individuals:
            species_counts[ind.species_name] += 1
            total_energy += ind.energy
            intelligence_levels.append(ind.intelligence)
            sociability_levels.append(ind.sociability)
            cultural_knowledge_levels.append(sum(ind.cultural_knowledge.values()))
        
        # Social network metrics
        num_relationships = sum(len(rels) for rels in self.social_network.relationships.values())
        num_communities = len(self.social_network.communities)
        num_leaders = len(self.social_network.leaders)
        
        # Communication metrics
        total_signals = sum(len(ind.communication.signal_repertoire) for ind in self.individuals)
        active_signals = sum(len(ind.active_signals) for ind in self.individuals)
        
        stats = {
            'time_step': self.time_step,
            'population': len(self.individuals),
            'species_counts': dict(species_counts),
            'avg_energy': total_energy / len(self.individuals),
            'avg_intelligence': np.mean(intelligence_levels),
            'avg_sociability': np.mean(sociability_levels),
            'avg_cultural_knowledge': np.mean(cultural_knowledge_levels),
            'num_relationships': num_relationships,
            'num_communities': num_communities,
            'num_leaders': num_leaders,
            'total_signals': total_signals,
            'active_signals': active_signals,
            'max_generation': max(ind.generation for ind in self.individuals)
        }
        
        self.generation_stats.append(stats)
    
    def _render_all(self):
        """Render all visualization components"""
        # Clear all axes
        for ax in [self.ax_main, self.ax_social, self.ax_stats, self.ax_comm, 
                   self.ax_culture, self.ax_env, self.ax_intel, self.ax_social_metrics,
                   self.ax_phase, self.ax_emergence]:
            ax.clear()
        
        self._render_main_simulation()
        self._render_social_network()
        self._render_statistics()
        self._render_communication_evolution()
        self._render_cultural_knowledge()
        self._render_environment()
        self._render_intelligence_evolution()
        self._render_social_complexity()
        self._render_phase_space()
        self._render_emergence_events()
    
    def _render_main_simulation(self):
        """Render main simulation view with enhanced visualization"""
        self.ax_main.set_xlim(0, self.config.width)
        self.ax_main.set_ylim(0, self.config.height)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_facecolor('black')
        self.ax_main.set_title('Emergent Intelligence Ecosystem', color='white', fontsize=16)
        
        # Draw environmental patches
        for patch in self.environment.patches:
            color = {'food': 'green', 'shelter': 'brown', 'neutral': 'gray'}[patch.patch_type]
            alpha = patch.resource_level * 0.3
            self.ax_main.scatter(patch.x, patch.y, c=color, s=30, alpha=alpha, marker='s')
        
        # Draw individuals
        for ind in self.individuals:
            # Size based on energy and intelligence
            size = (ind.size + ind.intelligence * 10) * (0.5 + 0.5 * ind.energy / 100)
            
            # Color based on species and state
            color = ind.color
            
            # Edge color based on social connectivity
            social_connections = len(self.social_network.relationships.get(ind.id, {}))
            edge_intensity = min(1.0, social_connections / 5.0)
            edge_color = (edge_intensity, edge_intensity, 1.0)
            
            self.ax_main.scatter(ind.x, ind.y, s=size*15, c=[color], 
                               alpha=0.8, edgecolors=edge_color, linewidth=2)
            
            # Draw communication signals
            if ind.active_signals:
                circle = plt.Circle((ind.x, ind.y), 20, fill=False, 
                                  color='yellow', alpha=0.3, linewidth=1)
                self.ax_main.add_patch(circle)
        
        # Draw social connections for leaders
        for leader_id in self.social_network.leaders:
            leader = self._get_individual_by_id(leader_id)
            if leader:
                relationships = self.social_network.relationships.get(leader_id, {})
                for other_id, rel in relationships.items():
                    if rel.strength > 0.5:
                        other = self._get_individual_by_id(other_id)
                        if other:
                            self.ax_main.plot([leader.x, other.x], [leader.y, other.y], 
                                            'cyan', alpha=0.3, linewidth=1)
        
        # Information text
        info_text = f"Population: {len(self.individuals)}\n"
        info_text += f"Generation: {max(ind.generation for ind in self.individuals) if self.individuals else 0}\n"
        info_text += f"Communities: {len(self.social_network.communities)}\n"
        info_text += f"Leaders: {len(self.social_network.leaders)}\n"
        info_text += f"Time: {self.time_step}"
        
        self.ax_main.text(10, self.config.height - 10, info_text, 
                         color='white', fontsize=11, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    def _render_social_network(self):
        """Render social network visualization"""
        self.ax_social.set_facecolor('black')
        self.ax_social.set_title('Social Network Structure', color='white', fontsize=12)
        
        if len(self.individuals) > 0 and len(self.social_network.relationships) > 0:
            # Create networkx graph for visualization
            G = nx.Graph()
            
            for ind in self.individuals:
                G.add_node(ind.id, species=ind.species_name, intelligence=ind.intelligence)
            
            for ind_id, relationships in self.social_network.relationships.items():
                for other_id, rel in relationships.items():
                    if rel.strength > 0.2:  # Only show significant relationships
                        G.add_edge(ind_id, other_id, weight=rel.strength)
            
            if len(G.nodes()) > 0:
                # Layout
                if len(G.nodes()) < 100:
                    pos = nx.spring_layout(G, k=3, iterations=50)
                else:
                    pos = nx.random_layout(G)
                
                # Draw nodes colored by species
                species_colors = {'predator': 'red', 'herbivore': 'green', 
                                'scavenger': 'yellow', 'mystic': 'purple'}
                
                for species, color in species_colors.items():
                    species_nodes = [n for n, d in G.nodes(data=True) if d.get('species') == species]
                    if species_nodes:
                        nx.draw_networkx_nodes(G, pos, nodelist=species_nodes, 
                                             node_color=color, node_size=50, 
                                             alpha=0.7, ax=self.ax_social)
                
                # Draw edges
                edges = G.edges(data=True)
                if edges:
                    edge_weights = [d['weight'] for _, _, d in edges]
                    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_weights, 
                                         edge_color='white', ax=self.ax_social)
                
                # Highlight leaders
                leader_nodes = [n for n in G.nodes() if n in self.social_network.leaders]
                if leader_nodes:
                    nx.draw_networkx_nodes(G, pos, nodelist=leader_nodes, 
                                         node_color='gold', node_size=100, 
                                         alpha=0.8, ax=self.ax_social)
        
        self.ax_social.set_aspect('equal')
    
    def _render_statistics(self):
        """Render population statistics"""
        if len(self.generation_stats) < 2:
            return
        
        self.ax_stats.set_facecolor('black')
        self.ax_stats.set_title('Population Dynamics', color='white', fontsize=10)
        
        recent_stats = self.generation_stats[-200:]
        time_steps = [s['time_step'] for s in recent_stats]
        populations = [s['population'] for s in recent_stats]
        
        self.ax_stats.plot(time_steps, populations, 'white', linewidth=2)
        self.ax_stats.set_ylabel('Population', color='white')
        self.ax_stats.grid(True, alpha=0.3)
        
        # Mark emergence events
        for event_type, event_time in self.emergence_events[-10:]:
            if event_time in time_steps:
                self.ax_stats.axvline(event_time, color='red', alpha=0.7, linestyle='--')
    
    def _render_communication_evolution(self):
        """Render communication system evolution"""
        if len(self.generation_stats) < 2:
            return
        
        self.ax_comm.set_facecolor('black')
        self.ax_comm.set_title('Communication Evolution', color='white', fontsize=10)
        
        recent_stats = self.generation_stats[-100:]
        time_steps = [s['time_step'] for s in recent_stats]
        total_signals = [s['total_signals'] for s in recent_stats]
        active_signals = [s['active_signals'] for s in recent_stats]
        
        self.ax_comm.plot(time_steps, total_signals, 'cyan', label='Total Signals', linewidth=2)
        self.ax_comm.plot(time_steps, active_signals, 'yellow', label='Active Signals', linewidth=2)
        
        self.ax_comm.set_ylabel('Signals', color='white')
        self.ax_comm.legend(fontsize=8)
        self.ax_comm.grid(True, alpha=0.3)
    
    def _render_cultural_knowledge(self):
        """Render cultural knowledge accumulation"""
        if len(self.generation_stats) < 2:
            return
        
        self.ax_culture.set_facecolor('black')
        self.ax_culture.set_title('Cultural Knowledge', color='white', fontsize=10)
        
        recent_stats = self.generation_stats[-100:]
        time_steps = [s['time_step'] for s in recent_stats]
        cultural_knowledge = [s['avg_cultural_knowledge'] for s in recent_stats]
        
        self.ax_culture.plot(time_steps, cultural_knowledge, 'orange', linewidth=2)
        self.ax_culture.set_ylabel('Avg Knowledge', color='white')
        self.ax_culture.grid(True, alpha=0.3)
    
    def _render_environment(self):
        """Render environmental state"""
        self.ax_env.set_facecolor('black')
        self.ax_env.set_title('Environmental Resources', color='white', fontsize=10)
        
        # Show resource distribution
        food_patches = [p for p in self.environment.patches if p.patch_type == 'food']
        if food_patches:
            resource_levels = [p.resource_level for p in food_patches]
            self.ax_env.hist(resource_levels, bins=20, color='green', alpha=0.7)
            self.ax_env.set_xlabel('Resource Level', color='white')
            self.ax_env.set_ylabel('Count', color='white')
    
    def _render_intelligence_evolution(self):
        """Render intelligence evolution over time"""
        if len(self.generation_stats) < 2:
            return
        
        self.ax_intel.set_facecolor('black')
        self.ax_intel.set_title('Intelligence Evolution', color='white', fontsize=10)
        
        recent_stats = self.generation_stats[-100:]
        time_steps = [s['time_step'] for s in recent_stats]
        intelligence = [s['avg_intelligence'] for s in recent_stats]
        
        self.ax_intel.plot(time_steps, intelligence, 'purple', linewidth=2)
        self.ax_intel.set_ylabel('Avg Intelligence', color='white')
        self.ax_intel.grid(True, alpha=0.3)
    
    def _render_social_complexity(self):
        """Render social complexity metrics"""
        if len(self.generation_stats) < 2:
            return
        
        self.ax_social_metrics.set_facecolor('black')
        self.ax_social_metrics.set_title('Social Complexity', color='white', fontsize=10)
        
        recent_stats = self.generation_stats[-100:]
        time_steps = [s['time_step'] for s in recent_stats]
        communities = [s['num_communities'] for s in recent_stats]
        leaders = [s['num_leaders'] for s in recent_stats]
        
        self.ax_social_metrics.plot(time_steps, communities, 'cyan', label='Communities', linewidth=2)
        self.ax_social_metrics.plot(time_steps, leaders, 'gold', label='Leaders', linewidth=2)
        
        self.ax_social_metrics.set_ylabel('Count', color='white')
        self.ax_social_metrics.legend(fontsize=8)
        self.ax_social_metrics.grid(True, alpha=0.3)
    
    def _render_phase_space(self):
        """Render phase space of key variables"""
        if not self.individuals:
            return
        
        self.ax_phase.set_facecolor('black')
        self.ax_phase.set_title('Intelligence vs Sociability', color='white', fontsize=10)
        
        intelligence = [ind.intelligence for ind in self.individuals]
        sociability = [ind.sociability for ind in self.individuals]
        species = [ind.species_name for ind in self.individuals]
        
        species_colors = {'predator': 'red', 'herbivore': 'green', 
                         'scavenger': 'yellow', 'mystic': 'purple'}
        
        for species_name, color in species_colors.items():
            species_intel = [intelligence[i] for i, s in enumerate(species) if s == species_name]
            species_social = [sociability[i] for i, s in enumerate(species) if s == species_name]
            
            if species_intel and species_social:
                self.ax_phase.scatter(species_intel, species_social, c=color, 
                                    label=species_name, alpha=0.7, s=30)
        
        self.ax_phase.set_xlabel('Intelligence', color='white')
        self.ax_phase.set_ylabel('Sociability', color='white')
        self.ax_phase.legend(fontsize=8)
        self.ax_phase.grid(True, alpha=0.3)
    
    def _render_emergence_events(self):
        """Render emergence event timeline"""
        self.ax_emergence.set_facecolor('black')
        self.ax_emergence.set_title('Emergence Events', color='white', fontsize=10)
        
        if self.emergence_events:
            event_types = [event[0] for event in self.emergence_events]
            event_times = [event[1] for event in self.emergence_events]
            
            # Create color map for event types
            unique_events = list(set(event_types))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_events)))
            event_colors = {event: colors[i] for i, event in enumerate(unique_events)}
            
            for i, (event_type, event_time) in enumerate(self.emergence_events):
                color = event_colors[event_type]
                self.ax_emergence.scatter(event_time, i, c=[color], s=100, alpha=0.8)
            
            # Legend
            for event_type, color in event_colors.items():
                self.ax_emergence.scatter([], [], c=[color], label=event_type.replace('_', ' ').title())
            
            self.ax_emergence.legend(fontsize=8, loc='upper left')
            self.ax_emergence.set_xlabel('Time Step', color='white')
            self.ax_emergence.set_ylabel('Event #', color='white')
    
    def run_simulation(self, frames: int = 5000, interval: int = 50):
        """Run the complete simulation"""
        anim = animation.FuncAnimation(self.fig, self.update, frames=frames,
                                     interval=interval, blit=False, repeat=False)
        plt.tight_layout()
        return anim

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_emergent_intelligence_simulation():
    """Initialize and run the emergent intelligence ecosystem"""
    print("🧠 EMERGENT INTELLIGENCE ECOSYSTEM")
    print("=" * 80)
    print("🌟 REVOLUTIONARY FEATURES:")
    print("• Hierarchical social networks with persistent relationships")
    print("• Communication and proto-language evolution")
    print("• Environmental memory and co-evolutionary dynamics")
    print("• Advanced cognitive architecture (attention, planning, metacognition)")
    print("• Cultural transmission and knowledge accumulation")
    print("• Multi-level selection (individual, group, cultural)")
    print("• Real-time emergence detection and phase transition analysis")
    print("• Comprehensive multi-panel visualization system")
    print("\n🔬 EMERGENT PHENOMENA TO OBSERVE:")
    print("• Language evolution and compositional communication")
    print("• Social hierarchy and leadership emergence")
    print("• Cultural knowledge accumulation and innovation")
    print("• Intelligence arms races and cognitive evolution")
    print("• Community formation and group identity")
    print("• Environmental modification and niche construction")
    print("• Collective decision-making and swarm intelligence")
    print("• Phase transitions in social organization")
    print("\n📊 VISUALIZATION PANELS:")
    print("• Main: Full ecosystem with environmental patches and social connections")
    print("• Social Network: Real-time social structure and community detection")
    print("• Statistics: Population dynamics with emergence event markers")
    print("• Communication: Evolution of signal repertoires and usage")
    print("• Culture: Accumulation of cultural knowledge over generations")
    print("• Environment: Resource distribution and environmental feedback")
    print("• Intelligence: Cognitive evolution and learning adaptation")
    print("• Social Complexity: Communities, leaders, and network metrics")
    print("• Phase Space: Intelligence vs sociability trait relationships")
    print("• Emergence: Timeline of major evolutionary breakthroughs")
    print("\n🚀 Starting simulation...")
    print("💡 Watch for spontaneous language emergence around step 1000-2000!")
    print("🏘️  Social communities typically form by step 1500!")
    print("🧬 Cultural knowledge accumulation accelerates after step 2000!")
    
    config = Config()
    sim = EmergentIntelligenceSimulation(config)
    return sim.run_simulation(frames=8000, interval=30)

# Run the simulation
if __name__ == "__main__":
    anim = run_emergent_intelligence_simulation()
    plt.show()