"""
Cultural transmission and knowledge evolution system.

This module implements cultural knowledge systems, teaching and learning mechanisms,
innovation processes, and cultural evolution dynamics.
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any


class CulturalKnowledge:
    """Individual piece of cultural knowledge"""
    
    def __init__(self, knowledge_id: str, value: float = 1.0, 
                 knowledge_type: str = 'general'):
        self.id = knowledge_id
        self.value = value  # Strength/importance of this knowledge
        self.knowledge_type = knowledge_type  # 'survival', 'social', 'technical', 'artistic'
        self.creation_time = 0
        self.transmission_count = 0
        self.success_rate = 0.5
        self.complexity = 1.0
        self.prerequisites = []  # Other knowledge required to learn this
        
    def mutate(self, mutation_strength: float = 0.1):
        """Mutate the knowledge during transmission"""
        self.value += random.uniform(-mutation_strength, mutation_strength)
        self.value = max(0.1, min(2.0, self.value))  # Keep within bounds
        
        # Complexity can also evolve
        self.complexity += random.uniform(-0.05, 0.05)
        self.complexity = max(0.5, min(3.0, self.complexity))


class CulturalSystem:
    """Cultural knowledge management and transmission system"""
    
    def __init__(self):
        self.knowledge_base = {}  # id -> CulturalKnowledge
        self.knowledge_categories = defaultdict(list)
        self.innovation_history = []
        self.transmission_network = defaultdict(list)  # Track who learned from whom
        
        # Cultural parameters
        self.innovation_rate = 0.01
        self.transmission_fidelity = 0.8
        self.knowledge_decay_rate = 0.001
        
    def add_knowledge(self, knowledge: CulturalKnowledge):
        """Add new knowledge to the cultural system"""
        self.knowledge_base[knowledge.id] = knowledge
        self.knowledge_categories[knowledge.knowledge_type].append(knowledge.id)
    
    def innovate_knowledge(self, creator_id: int, innovation_tendency: float, 
                          existing_knowledge: Dict[str, float], time_step: int) -> Optional[CulturalKnowledge]:
        """Create new cultural knowledge through innovation"""
        if random.random() > innovation_tendency * self.innovation_rate:
            return None
        
        # Determine type of innovation based on existing knowledge
        knowledge_types = ['survival', 'social', 'technical', 'artistic']
        
        # Bias toward types where individual has existing knowledge
        type_weights = []
        for ktype in knowledge_types:
            existing_in_type = sum(1 for kid in existing_knowledge.keys() 
                                 if kid in self.knowledge_categories[ktype])
            type_weights.append(1.0 + existing_in_type * 0.5)
        
        # Normalize weights
        total_weight = sum(type_weights)
        type_probabilities = [w / total_weight for w in type_weights]
        
        chosen_type = np.random.choice(knowledge_types, p=type_probabilities)
        
        # Create new knowledge
        knowledge_id = f"{chosen_type}_{creator_id}_{time_step}"
        base_value = 0.5 + random.uniform(0, 0.5)
        complexity = 1.0 + random.uniform(0, innovation_tendency)
        
        new_knowledge = CulturalKnowledge(knowledge_id, base_value, chosen_type)
        new_knowledge.complexity = complexity
        new_knowledge.creation_time = time_step
        
        # Set prerequisites based on existing knowledge
        if existing_knowledge:
            potential_prereqs = list(existing_knowledge.keys())
            num_prereqs = min(2, len(potential_prereqs))
            if num_prereqs > 0:
                new_knowledge.prerequisites = random.sample(potential_prereqs, 
                                                           random.randint(0, num_prereqs))
        
        self.add_knowledge(new_knowledge)
        
        # Record innovation
        self.innovation_history.append({
            'creator_id': creator_id,
            'knowledge_id': knowledge_id,
            'time_step': time_step,
            'type': chosen_type,
            'complexity': complexity
        })
        
        return new_knowledge
    
    def transmit_knowledge(self, teacher_id: int, learner_id: int, 
                          teacher_knowledge: Dict[str, float], 
                          learner_knowledge: Dict[str, float],
                          teacher_ability: float, learner_intelligence: float,
                          relationship_strength: float) -> Dict[str, float]:
        """Transmit knowledge between individuals"""
        transmitted_knowledge = {}
        
        # Determine what can be taught
        teachable_knowledge = {}
        for kid, value in teacher_knowledge.items():
            if kid in self.knowledge_base:
                knowledge = self.knowledge_base[kid]
                
                # Check if learner meets prerequisites
                if self._check_prerequisites(knowledge, learner_knowledge):
                    # Teaching probability based on various factors
                    teach_prob = (
                        teacher_ability * 0.4 +
                        relationship_strength * 0.3 +
                        (1.0 / knowledge.complexity) * 0.3
                    )
                    
                    if random.random() < teach_prob:
                        teachable_knowledge[kid] = value
        
        # Transmit selected knowledge
        for kid, teacher_value in teachable_knowledge.items():
            if kid not in learner_knowledge or learner_knowledge[kid] < teacher_value:
                knowledge = self.knowledge_base[kid]
                
                # Learning success probability
                learn_prob = (
                    learner_intelligence * 0.5 +
                    self.transmission_fidelity * 0.3 +
                    relationship_strength * 0.2
                )
                
                if random.random() < learn_prob:
                    # Successful transmission with possible mutation
                    transmitted_value = teacher_value
                    
                    # Apply transmission noise
                    noise = random.uniform(-0.1, 0.1) * (1.0 - self.transmission_fidelity)
                    transmitted_value += noise
                    transmitted_value = max(0.1, min(2.0, transmitted_value))
                    
                    transmitted_knowledge[kid] = transmitted_value
                    
                    # Update knowledge statistics
                    knowledge.transmission_count += 1
                    knowledge.success_rate = (knowledge.success_rate * 0.9 + 1.0 * 0.1)
                    
                    # Record transmission
                    self.transmission_network[learner_id].append({
                        'teacher_id': teacher_id,
                        'knowledge_id': kid,
                        'value': transmitted_value
                    })
                else:
                    # Failed transmission
                    knowledge.success_rate = (knowledge.success_rate * 0.9 + 0.0 * 0.1)
        
        return transmitted_knowledge
    
    def _check_prerequisites(self, knowledge: CulturalKnowledge, 
                           learner_knowledge: Dict[str, float]) -> bool:
        """Check if learner has prerequisites for this knowledge"""
        for prereq_id in knowledge.prerequisites:
            if prereq_id not in learner_knowledge or learner_knowledge[prereq_id] < 0.5:
                return False
        return True
    
    def decay_knowledge(self):
        """Apply decay to unused knowledge"""
        for knowledge in self.knowledge_base.values():
            if knowledge.transmission_count == 0:  # Unused knowledge
                knowledge.value *= (1.0 - self.knowledge_decay_rate)
                if knowledge.value < 0.1:
                    knowledge.value = 0.1  # Minimum value
    
    def get_knowledge_complexity(self) -> float:
        """Calculate overall complexity of cultural knowledge"""
        if not self.knowledge_base:
            return 0
        
        complexities = [k.complexity for k in self.knowledge_base.values()]
        return np.mean(complexities)
    
    def get_knowledge_diversity(self) -> float:
        """Calculate diversity of knowledge types"""
        type_counts = defaultdict(int)
        for knowledge in self.knowledge_base.values():
            type_counts[knowledge.knowledge_type] += 1
        
        if not type_counts:
            return 0
        
        # Shannon diversity
        total = sum(type_counts.values())
        proportions = [count / total for count in type_counts.values()]
        return -sum(p * np.log(p) for p in proportions if p > 0)


class CulturalEvolution:
    """Population-level cultural evolution dynamics"""
    
    def __init__(self):
        self.cultural_system = CulturalSystem()
        self.population_knowledge = defaultdict(dict)  # individual_id -> knowledge
        self.cultural_fitness = defaultdict(float)
        self.cultural_lineages = defaultdict(list)  # Track knowledge inheritance
        
        # Evolution parameters
        self.selection_pressure = 0.1
        self.cultural_drift_rate = 0.05
        
    def update_population_culture(self, individuals: List, social_network, time_step: int):
        """Update cultural evolution for entire population"""
        # Innovation phase
        for individual in individuals:
            if hasattr(individual, 'innovation_tendency'):
                new_knowledge = self.cultural_system.innovate_knowledge(
                    individual.id,
                    individual.innovation_tendency,
                    individual.cultural_knowledge,
                    time_step
                )
                
                if new_knowledge:
                    individual.cultural_knowledge[new_knowledge.id] = new_knowledge.value
        
        # Transmission phase
        self._cultural_transmission_phase(individuals, social_network)
        
        # Selection phase
        self._cultural_selection_phase(individuals)
        
        # Drift phase
        self._cultural_drift_phase(individuals)
        
        # Update population knowledge tracking
        for individual in individuals:
            self.population_knowledge[individual.id] = individual.cultural_knowledge.copy()
    
    def _cultural_transmission_phase(self, individuals: List, social_network):
        """Handle cultural transmission between individuals"""
        for individual in individuals:
            # Find potential teachers (social connections)
            teachers = []
            if individual.id in social_network.relationships:
                for other_id, relationship in social_network.relationships[individual.id].items():
                    if relationship.strength > 0.3:  # Strong enough relationship
                        teacher = next((ind for ind in individuals if ind.id == other_id), None)
                        if teacher:
                            teachers.append((teacher, relationship.strength))
            
            # Learn from teachers
            for teacher, relationship_strength in teachers:
                if random.random() < 0.1:  # 10% chance of learning interaction
                    transmitted = self.cultural_system.transmit_knowledge(
                        teacher.id,
                        individual.id,
                        teacher.cultural_knowledge,
                        individual.cultural_knowledge,
                        teacher.teaching_ability,
                        individual.intelligence,
                        relationship_strength
                    )
                    
                    # Update individual's knowledge
                    for kid, value in transmitted.items():
                        individual.cultural_knowledge[kid] = max(
                            individual.cultural_knowledge.get(kid, 0), value
                        )
    
    def _cultural_selection_phase(self, individuals: List):
        """Apply selection pressure on cultural knowledge"""
        # Calculate cultural fitness for each individual
        for individual in individuals:
            fitness = self._calculate_cultural_fitness(individual)
            self.cultural_fitness[individual.id] = fitness
            
            # Remove less fit cultural knowledge
            if len(individual.cultural_knowledge) > 10:  # Limit knowledge capacity
                # Sort by value and keep top knowledge
                sorted_knowledge = sorted(
                    individual.cultural_knowledge.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Keep top 80% of knowledge
                keep_count = int(len(sorted_knowledge) * 0.8)
                individual.cultural_knowledge = dict(sorted_knowledge[:keep_count])
    
    def _cultural_drift_phase(self, individuals: List):
        """Apply random cultural drift"""
        for individual in individuals:
            for kid in list(individual.cultural_knowledge.keys()):
                if random.random() < self.cultural_drift_rate:
                    # Random loss of knowledge
                    if random.random() < 0.1:  # 10% chance of complete loss
                        del individual.cultural_knowledge[kid]
                    else:
                        # Gradual decay
                        individual.cultural_knowledge[kid] *= 0.95
                        if individual.cultural_knowledge[kid] < 0.1:
                            del individual.cultural_knowledge[kid]
    
    def _calculate_cultural_fitness(self, individual) -> float:
        """Calculate fitness benefit of cultural knowledge"""
        if not individual.cultural_knowledge:
            return 0
        
        fitness = 0
        
        # Different knowledge types provide different benefits
        type_benefits = {
            'survival': 2.0,
            'social': 1.5,
            'technical': 1.8,
            'artistic': 1.0
        }
        
        for kid, value in individual.cultural_knowledge.items():
            if kid in self.cultural_system.knowledge_base:
                knowledge = self.cultural_system.knowledge_base[kid]
                benefit = type_benefits.get(knowledge.knowledge_type, 1.0)
                fitness += value * benefit / knowledge.complexity
        
        return fitness
    
    def get_cultural_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cultural evolution metrics"""
        total_knowledge = len(self.cultural_system.knowledge_base)
        total_innovations = len(self.cultural_system.innovation_history)
        
        # Knowledge distribution by type
        type_distribution = defaultdict(int)
        for knowledge in self.cultural_system.knowledge_base.values():
            type_distribution[knowledge.knowledge_type] += 1
        
        # Average cultural fitness
        avg_fitness = np.mean(list(self.cultural_fitness.values())) if self.cultural_fitness else 0
        
        # Knowledge complexity
        avg_complexity = self.cultural_system.get_knowledge_complexity()
        
        # Knowledge diversity
        knowledge_diversity = self.cultural_system.get_knowledge_diversity()
        
        return {
            'total_knowledge': total_knowledge,
            'total_innovations': total_innovations,
            'knowledge_by_type': dict(type_distribution),
            'avg_cultural_fitness': avg_fitness,
            'avg_knowledge_complexity': avg_complexity,
            'knowledge_diversity': knowledge_diversity,
            'active_individuals': len(self.population_knowledge)
        }
    
    def detect_cultural_breakthroughs(self) -> List[Dict[str, Any]]:
        """Detect significant cultural breakthroughs"""
        breakthroughs = []
        
        # High complexity innovations
        for innovation in self.cultural_system.innovation_history[-10:]:
            if innovation['complexity'] > 2.0:
                breakthroughs.append({
                    'type': 'high_complexity_innovation',
                    'innovation': innovation,
                    'significance': innovation['complexity']
                })
        
        # Widely transmitted knowledge
        for knowledge in self.cultural_system.knowledge_base.values():
            if knowledge.transmission_count > 10:
                breakthroughs.append({
                    'type': 'viral_knowledge',
                    'knowledge_id': knowledge.id,
                    'transmission_count': knowledge.transmission_count,
                    'significance': knowledge.transmission_count / 10.0
                })
        
        return sorted(breakthroughs, key=lambda x: x['significance'], reverse=True)
    
    def get_knowledge_genealogy(self, knowledge_id: str) -> Dict[str, Any]:
        """Trace the genealogy of a piece of knowledge"""
        if knowledge_id not in self.cultural_system.knowledge_base:
            return {}
        
        knowledge = self.cultural_system.knowledge_base[knowledge_id]
        
        # Find who has this knowledge
        holders = []
        for ind_id, knowledge_dict in self.population_knowledge.items():
            if knowledge_id in knowledge_dict:
                holders.append(ind_id)
        
        # Find transmission history
        transmission_history = []
        for learner_id, transmissions in self.cultural_system.transmission_network.items():
            for transmission in transmissions:
                if transmission['knowledge_id'] == knowledge_id:
                    transmission_history.append(transmission)
        
        return {
            'knowledge_id': knowledge_id,
            'type': knowledge.knowledge_type,
            'complexity': knowledge.complexity,
            'creation_time': knowledge.creation_time,
            'current_holders': holders,
            'transmission_history': transmission_history,
            'total_transmissions': knowledge.transmission_count,
            'success_rate': knowledge.success_rate
        }
