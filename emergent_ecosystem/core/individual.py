"""
Enhanced individual agent with cognitive architecture.

This module implements the main individual agent class that integrates
all cognitive modules, social systems, and environmental interactions.
"""

import time
import numpy as np
import random
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional, Any

from ..config import Config
from ..cognition.attention import AttentionModule
from ..cognition.planning import PlanningModule
from ..cognition.metacognition import MetacognitionModule
from ..social.communication import CommunicationSystem


class EnhancedIndividual:
    """Individual with advanced cognitive architecture and social capabilities"""
    
    def __init__(self, x: float, y: float, species_name: str = 'herbivore', 
                 parent1=None, parent2=None, individual_id: int = None, config: Config = None):
        # Configuration
        self.config = config or Config()
        
        # Basic properties
        self.id = individual_id or random.randint(100000, 999999)
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.species_name = species_name
        self.perception_time = 0
        self.perception_count = 0
        
        # Cognitive architecture
        self.attention = AttentionModule(self.config.attention_span)
        self.planning = PlanningModule(self.config.planning_horizon)
        self.metacognition = MetacognitionModule(self.config.learning_rate)
        
        # Communication system
        self.communication = CommunicationSystem(self.config.communication_mutation_rate)
        self.active_signals = []  # Currently broadcasting signals
        
        # Enhanced memory systems
        self.spatial_memory = deque(maxlen=self.config.memory_length)
        self.social_memory = {}  # individual_id -> interaction memories
        self.environmental_memory = {}  # location -> environmental info
        
        # Social and cultural traits
        self.cultural_knowledge = defaultdict(float)  # Learned cultural information
        self.teaching_ability = random.uniform(self.config.social.teaching_ability_min, 
                                             self.config.social.teaching_ability_max)
        self.innovation_tendency = random.uniform(self.config.social.innovation_tendency_min, 
                                                self.config.social.innovation_tendency_max)
        
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
        self.goal_history = deque(maxlen=20)
        
        # Visual properties
        self.color = self._generate_color()
        self.trail = deque(maxlen=self.config.visualization.trail_length)
        self.size = random.uniform(5, 12)
        
        # Performance tracking
        self.lifetime_rewards = 0.0
        self.successful_actions = 0
        self.total_actions = 0
        self.interaction_count = defaultdict(int)
        
    def _inherit_from_parents(self, parent1, parent2):
        """Inherit traits from parents with cultural transmission"""
        # Genetic inheritance
        self.intelligence = self._inherit_trait(parent1.intelligence, parent2.intelligence, 
                                              self.config.cognitive.intelligence_min, 
                                              self.config.cognitive.intelligence_max)
        self.sociability = self._inherit_trait(parent1.sociability, parent2.sociability, 0, 1)
        self.aggression = self._inherit_trait(parent1.aggression, parent2.aggression, 0, 1)
        self.max_speed = self._inherit_trait(parent1.max_speed, parent2.max_speed, 
                                           self.config.cognitive.max_speed_min, 
                                           self.config.cognitive.max_speed_max)
        
        # Inherit cognitive parameters
        self.learning_rate = self._inherit_trait(parent1.metacognition.self_model['learning_rate'], 
                                                parent2.metacognition.self_model['learning_rate'], 
                                                0.05, 0.3)
        
        # Cultural inheritance
        combined_knowledge = {}
        for knowledge in [parent1.cultural_knowledge, parent2.cultural_knowledge]:
            for key, value in knowledge.items():
                combined_knowledge[key] = combined_knowledge.get(key, 0) + value * 0.5
        
        # Inherit cultural knowledge with some mutation
        for key, value in combined_knowledge.items():
            if random.random() < self.config.cultural_inheritance_rate:
                mutation = random.uniform(-0.1, 0.1)
                self.cultural_knowledge[key] = max(0, value + mutation)
        
        # Inherit some communication patterns
        parent_signals = list(parent1.communication.signal_repertoire.keys())[:3]
        for signal_id in parent_signals:
            if random.random() < 0.7:  # 70% chance to inherit signal
                original_signal = parent1.communication.signal_repertoire[signal_id]
                new_signal_id = self.communication.create_new_signal(
                    parent_signals=[signal_id], 
                    context='inherited',
                    time_step=0
                )
    
    def _initialize_traits(self):
        """Initialize traits for first generation"""
        self.intelligence = random.uniform(self.config.cognitive.intelligence_min, 
                                         self.config.cognitive.intelligence_max)
        self.sociability = random.uniform(0.1, 0.9)
        self.aggression = random.uniform(0.0, 0.6)
        self.max_speed = random.uniform(self.config.cognitive.max_speed_min, 
                                      self.config.cognitive.max_speed_max)
        self.learning_rate = self.config.learning_rate
    
    def _inherit_trait(self, parent1_val: float, parent2_val: float, 
                      min_val: float, max_val: float) -> float:
        """Inherit trait with mutation"""
        # Choose parent gene
        base_val = parent1_val if random.random() < 0.5 else parent2_val
        
        # Apply mutation
        if random.random() < self.config.mutation_rate:
            mutation_strength = (max_val - min_val) * 0.1
            mutation = random.uniform(-mutation_strength, mutation_strength)
            base_val += mutation
        
        return np.clip(base_val, min_val, max_val)
    
    def perceive_environment(self, others: List['EnhancedIndividual'], environment, social_network, 
                       spatial_grid=None, use_spatial_optimization=True):
        """Advanced environmental perception with attention filtering and spatial optimization"""
        import time
        start_time = time.time()
        raw_perception = {
            'nearby_individuals': [],
            'environmental_features': [],
            'social_information': [],
            'danger_signals': [],
            'opportunity_signals': []
        }
        
        # Use spatial optimization if available
        if use_spatial_optimization and spatial_grid is not None:
            # Create a dictionary for fast lookup
            individuals_dict = {ind.id: ind for ind in others}
            # Get only nearby individuals using spatial grid
            nearby_others = spatial_grid.get_nearby_individuals(
                self.x, self.y, self.config.communication_radius, individuals_dict
            )
        else:
            # Fallback to checking all individuals
            nearby_others = []
            for other in others:
                if other.id != self.id:
                    dx = other.x - self.x
                    dy = other.y - self.y
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance < self.config.communication_radius:
                        nearby_others.append(other)
        
        # Perceive nearby individuals (now only processes nearby ones)
        for other in nearby_others:
            if other.id != self.id:
                dx = other.x - self.x
                dy = other.y - self.y
                distance = np.sqrt(dx**2 + dy**2)
                
                perception = {
                    'individual': other,
                    'distance': distance,
                    'relationship_strength': social_network.get_relationship_strength(self.id, other.id),
                    'species': other.species_name,
                    'energy_level': other.energy / 100.0,
                    'active_signals': other.active_signals.copy(),
                    'relative_position': (dx, dy)
                }
                raw_perception['nearby_individuals'].append(perception)
        
        # Rest of the method remains the same...
        # Perceive environmental patches
        if hasattr(environment, 'get_nearby_patches'):
            nearby_patches = environment.get_nearby_patches(self.x, self.y, 50)
            for patch in nearby_patches:
                perception = {
                    'patch': patch,
                    'distance': np.sqrt((self.x - patch.x)**2 + (self.y - patch.y)**2),
                    'type': patch.patch_type,
                    'quality': patch.quality,
                    'resource_level': patch.resource_level,
                    'signals': patch.get_environmental_signal()
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
            'danger': len(raw_perception['danger_signals']) * 0.3,
            'communication': len(raw_perception['social_information']) * 0.2
        }
        
        self.attention.update_attention(stimuli_salience, self.current_goal)
        filtered_perception = self.attention.get_filtered_perception(raw_perception)
        self.perception_time += time.time() - start_time
        self.perception_count += 1
        return filtered_perception
    
    def make_decisions(self, perception: Dict[str, Any]):
        """High-level decision making with planning"""
        # Update current goal based on needs and perception
        self._update_current_goal(perception)
        
        # Get confidence for current goal
        goal_confidence = self.metacognition.assess_confidence(self.current_goal, 'general')
        
        # Create environment model for planning
        environment_model = self._build_environment_model(perception)
        
        # Create plan
        current_state = {
            'position': (self.x, self.y),
            'energy': self.energy,
            'stress': self.stress_level,
            'urgency': 1.0 - (self.energy / 100.0)  # Urgency increases as energy decreases
        }
        
        plan = self.planning.create_plan(current_state, self.current_goal, environment_model)
        
        return plan, goal_confidence
    
    def _update_current_goal(self, perception: Dict[str, Any]):
        """Update current goal based on needs and perception"""
        old_goal = self.current_goal
        
        # Priority-based goal selection
        if self.energy < 30:
            self.current_goal = 'find_food'
        elif self.stress_level > 0.7:
            self.current_goal = 'avoid_danger'
        elif (len(perception.get('nearby_individuals', [])) > 0 and 
              self.sociability > 0.6 and self.energy > 50):
            self.current_goal = 'socialize'
        elif self.energy > 70 and self.innovation_tendency > 0.5:
            self.current_goal = 'explore'
        else:
            self.current_goal = 'explore'
        
        # Record goal changes
        if old_goal != self.current_goal:
            self.goal_history.append({
                'old_goal': old_goal,
                'new_goal': self.current_goal,
                'age': self.age,
                'trigger': self._identify_goal_trigger()
            })
    
    def _identify_goal_trigger(self) -> str:
        """Identify what triggered the goal change"""
        if self.energy < 30:
            return 'low_energy'
        elif self.stress_level > 0.7:
            return 'high_stress'
        elif self.energy > 70:
            return 'high_energy'
        else:
            return 'general'
    
    def _build_environment_model(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Build environment model for planning"""
        model = {
            'food_locations': [],
            'social_opportunities': [],
            'danger_locations': [],
            'safe_locations': [],
            'obstacles': []
        }
        
        # Process environmental features
        for feature in perception.get('environmental_features', []):
            patch = feature['patch']
            if patch.patch_type == 'food' and patch.resource_level > 0.3:
                model['food_locations'].append({
                    'position': (patch.x, patch.y),
                    'distance': feature['distance'],
                    'quality': patch.quality * patch.resource_level
                })
            elif patch.patch_type == 'shelter':
                model['safe_locations'].append({
                    'position': (patch.x, patch.y),
                    'distance': feature['distance'],
                    'quality': patch.quality
                })
        
        # Process social opportunities
        for individual_data in perception.get('nearby_individuals', []):
            if individual_data['relationship_strength'] > 0.3:
                model['social_opportunities'].append({
                    'individual': individual_data['individual'],
                    'distance': individual_data['distance'],
                    'relationship_strength': individual_data['relationship_strength']
                })
        
        return model
    
    def communicate(self, perception: Dict[str, Any], social_network):
        """Generate communication signals based on context"""
        self.active_signals.clear()
        
        # Danger warning
        if self.stress_level > 0.6:
            danger_signals = self.communication.get_signals_for_context('danger')
            if danger_signals:
                self.active_signals.append(danger_signals[0])
        
        # Food sharing (if high energy and social)
        if self.energy > 80 and self.sociability > 0.7:
            food_signals = self.communication.get_signals_for_context('food')
            if food_signals:
                self.active_signals.append(food_signals[0])
        
        # Mating calls
        if self.energy > 70 and self.age > 100:
            mating_signals = self.communication.get_signals_for_context('mating')
            if mating_signals:
                self.active_signals.append(mating_signals[0])
        
        # Innovation: Create new signals occasionally
        if (random.random() < 0.001 * self.innovation_tendency and 
            len(self.communication.signal_repertoire) < 20):
            
            existing_signals = list(self.communication.signal_repertoire.keys())
            if len(existing_signals) >= 2:
                parent_signals = random.sample(existing_signals, 2)
                new_signal_id = self.communication.create_new_signal(
                    parent_signals, self.current_goal, self.age)
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
            difficulty = self._assess_interaction_difficulty(interaction_type, other_individual)
            reward = 1.0 if success else 0.0
            self.metacognition.update_self_model(interaction_type, success, self.current_goal, 
                                               difficulty, reward)
            
            # Update communication system
            if interaction_type == 'communication':
                for signal_id in other_individual.active_signals:
                    self.communication.update_signal_success(signal_id, success, self.current_goal)
            
            # Theory of mind update
            self.metacognition.model_other_mind(
                other_individual.id, interaction_type, self.current_goal, success,
                {'cooperation': other_individual.sociability}
            )
            
            # Track interaction counts
            self.interaction_count[interaction_type] += 1
    
    def _assess_interaction_difficulty(self, interaction_type: str, other_individual) -> float:
        """Assess the difficulty of an interaction"""
        base_difficulty = {
            'communication': 0.4,
            'cooperation': 0.5,
            'competition': 0.7,
            'help': 0.3
        }.get(interaction_type, 0.5)
        
        # Adjust based on other individual's traits
        if hasattr(other_individual, 'intelligence'):
            intelligence_factor = other_individual.intelligence / (self.intelligence + 0.1)
            base_difficulty *= intelligence_factor
        
        return np.clip(base_difficulty, 0.1, 1.0)
    
    def update_physics(self, others: List['EnhancedIndividual'], environment, social_network):
        """Update position and state with enhanced cognitive processing"""
        # Perceive environment
        perception = self.perceive_environment(others, environment, social_network, spatial_grid)
        
        # Make decisions
        plan, confidence = self.make_decisions(perception)
        
        # Communicate
        self.communicate(perception, social_network)
        
        # Execute plan or calculate forces
        if self.planning.has_active_plan():
            # Execute current plan
            action = self.planning.execute_plan_step()
            forces = self._execute_planned_action(action, perception)
            
            # Evaluate action success
            action_success = self._evaluate_action_success(action, perception)
            self.planning.evaluate_plan_success(action_success)
            
        else:
            # No active plan - use reactive behavior
            forces = self._calculate_reactive_forces(perception)
        
        # Update velocity and position
        self.vx += forces[0] * self.config.time_step
        self.vy += forces[1] * self.config.time_step
        
        # Limit speed
        speed = np.sqrt(self.vx**2 + self.vy**2)
        if speed > self.max_speed:
            self.vx = (self.vx / speed) * self.max_speed
            self.vy = (self.vy / speed) * self.max_speed
        
        # Update position with wraparound
        self.x = (self.x + self.vx) % self.config.width
        self.y = (self.y + self.vy) % self.config.height
        
        # Update memories
        self.spatial_memory.append((self.x, self.y, self.current_goal))
        
        # Update state
        self._update_state()
        
        # Update color and trail
        self.color = self._generate_color()
        self.trail.append((self.x, self.y))
        
        # Adapt cognitive systems
        self._adapt_cognitive_systems()
    
    def _execute_planned_action(self, action: Optional[Tuple[str, Any]], 
                               perception: Dict[str, Any]) -> Tuple[float, float]:
        """Execute a planned action and return movement forces"""
        if not action:
            return (0.0, 0.0)
        
        action_type, action_target = action
        forces = (0.0, 0.0)
        
        if action_type == 'move_to' and action_target:
            target_pos = action_target if isinstance(action_target, tuple) else (0, 0)
            dx = target_pos[0] - self.x
            dy = target_pos[1] - self.y
            
            # Handle wraparound
            if abs(dx) > self.config.width / 2:
                dx = dx - np.sign(dx) * self.config.width
            if abs(dy) > self.config.height / 2:
                dy = dy - np.sign(dy) * self.config.height
            
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                forces = ((dx / dist) * 2.0, (dy / dist) * 2.0)
        
        elif action_type == 'move_away' and action_target:
            avoid_pos = action_target if isinstance(action_target, tuple) else (0, 0)
            dx = self.x - avoid_pos[0]
            dy = self.y - avoid_pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                forces = ((dx / dist) * 3.0, (dy / dist) * 3.0)
        
        elif action_type == 'approach' and hasattr(action_target, 'x'):
            dx = action_target.x - self.x
            dy = action_target.y - self.y
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                forces = ((dx / dist) * 1.5, (dy / dist) * 1.5)
        
        return forces
    
    def _calculate_reactive_forces(self, perception: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate movement forces based on reactive behavior"""
        total_fx = total_fy = 0.0
        
        # Social forces based on relationships
        for individual_data in perception.get('nearby_individuals', []):
            other = individual_data['individual']
            relationship_strength = individual_data['relationship_strength']
            distance = individual_data['distance']
            
            if distance > 0:
                dx, dy = individual_data['relative_position']
                
                # Social attraction/repulsion based on relationship
                if relationship_strength > 0.5:  # Friends attract
                    force = relationship_strength * 0.5 / (distance + 1)
                    total_fx += (dx / distance) * force
                    total_fy += (dy / distance) * force
                elif relationship_strength < 0.2:  # Rivals repel
                    force = -0.5 / (distance + 1)
                    total_fx += (dx / distance) * force
                    total_fy += (dy / distance) * force
        
        # Environmental attraction
        if self.current_goal == 'find_food':
            for feature in perception.get('environmental_features', []):
                if feature['patch'].patch_type == 'food':
                    dx = feature['patch'].x - self.x
                    dy = feature['patch'].y - self.y
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > 0:
                        force = feature['patch'].resource_level / (dist + 1)
                        total_fx += (dx / dist) * force
                        total_fy += (dy / dist) * force
        
        # Add some random exploration
        if self.current_goal == 'explore':
            total_fx += random.uniform(-0.5, 0.5)
            total_fy += random.uniform(-0.5, 0.5)
        
        return (total_fx, total_fy)
    
    def _evaluate_action_success(self, action: Optional[Tuple[str, Any]], 
                               perception: Dict[str, Any]) -> bool:
        """Evaluate whether an action was successful"""
        if not action:
            return False
        
        action_type, _ = action
        
        # Simple success criteria
        if action_type == 'move_to':
            return random.random() < 0.8  # Most movements are successful
        elif action_type == 'approach':
            return random.random() < 0.7  # Social approaches sometimes fail
        elif action_type == 'communicate':
            return len(perception.get('social_information', [])) > 0
        else:
            return random.random() < 0.6  # Default success rate
    
    def _update_state(self):
        """Update energy, age, stress, and other state variables"""
        # Energy consumption
        movement_cost = np.sqrt(self.vx**2 + self.vy**2) * 0.02
        cognitive_cost = (self.intelligence + len(self.active_signals) * 0.1) * 0.01
        social_cost = len(self.social_memory) * 0.001
        
        total_cost = movement_cost + cognitive_cost + social_cost + 0.1
        self.energy -= total_cost
        self.energy = max(0, min(self.energy, 100))
        
        # Age
        self.age += 1
        
        # Stress adaptation
        if self.current_goal == 'avoid_danger':
            self.stress_level = min(1.0, self.stress_level + 0.1)
        else:
            self.stress_level = max(0.0, self.stress_level - 0.05)
        
        # Update performance tracking
        self.total_actions += 1
        self.lifetime_rewards += self.energy / 100.0  # Simple reward based on energy
    
    def _adapt_cognitive_systems(self):
        """Adapt cognitive systems based on experience"""
        # Adapt metacognition
        self.metacognition.adapt_learning_rate()
        
        # Adapt attention based on environmental complexity
        env_complexity = len(self.social_memory) / 10.0 + self.stress_level
        self.attention.adapt_to_environment(env_complexity)
        
        # Learn attention patterns from outcomes
        recent_success = self.energy > 50  # Simple success measure
        self.attention.learn_attention_patterns(recent_success, self.lifetime_rewards / max(1, self.total_actions))
    
    def _generate_color(self) -> Tuple[float, float, float]:
        """Generate color based on current state"""
        base_colors = self.config.visualization.species_colors
        base_r, base_g, base_b = base_colors.get(self.species_name, (0.5, 0.5, 0.5))
        
        # Modify based on state
        energy_factor = self.energy / 100.0
        intelligence_factor = self.intelligence
        social_factor = len(self.social_memory) / 10.0
        
        r = np.clip(base_r + (self.stress_level - 0.5) * 0.3, 0, 1)
        g = np.clip(base_g * energy_factor + intelligence_factor * 0.2, 0, 1)
        b = np.clip(base_b + social_factor * 0.1, 0, 1)
        
        return (r, g, b)
    
    def can_breed(self, other: 'EnhancedIndividual') -> bool:
        """Enhanced breeding conditions"""
        if (self.species_name != other.species_name or
            self.energy < self.config.breeding_energy_threshold or
            other.energy < self.config.breeding_energy_threshold):
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
        
        # Metacognitive assessment of mate quality
        mate_prediction = self.metacognition.predict_other_behavior(other.id, 'mating')
        mate_quality = mate_prediction.get('cooperation_likelihood', 0.5)
        
        compatibility = (relationship_strength + mate_quality) / 2
        return compatibility > 0.4 and random.random() < self.config.evolution.breeding_probability
    
    def breed(self, other: 'EnhancedIndividual') -> 'EnhancedIndividual':
        """Create offspring with enhanced inheritance"""
        offspring_x = (self.x + other.x) / 2
        offspring_y = (self.y + other.y) / 2
        
        offspring = EnhancedIndividual(offspring_x, offspring_y, self.species_name, 
                                     self, other, config=self.config)
        
        # Breeding costs
        self.energy -= self.config.evolution.breeding_cost
        other.energy -= self.config.evolution.breeding_cost
        
        return offspring
    
    def is_alive(self) -> bool:
        """Check if individual is still alive"""
        max_age = self.config.evolution.max_ages.get(self.species_name, 800)
        return self.energy > 0 and self.age < max_age
    
    def get_cognitive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cognitive performance metrics"""
        return {
            'attention_metrics': self.attention.get_attention_metrics(),
            'planning_metrics': self.planning.get_planning_metrics(),
            'metacognitive_state': self.metacognition.get_metacognitive_state(),
            'communication_complexity': self.communication.get_communication_complexity(),
            'performance_summary': {
                'success_rate': self.successful_actions / max(1, self.total_actions),
                'lifetime_rewards': self.lifetime_rewards,
                'social_connections': len(self.social_memory),
                'cultural_knowledge_total': sum(self.cultural_knowledge.values()),
                'innovation_count': len([h for h in self.communication.innovation_history 
                                       if h['type'] == 'innovation'])
            }
        }
