"""
Communication and language evolution system.

This module implements evolving communication systems including signal repertoires,
meaning evolution, compositional communication, and proto-language emergence.
"""

import numpy as np
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class Signal:
    """Individual communication signal with meaning evolution"""
    
    def __init__(self, signal_id: int, intensity: float = 1.0):
        self.id = signal_id
        self.intensity = intensity  # Signal strength
        self.meaning_vector = np.random.random(5)  # Abstract meaning space
        self.usage_count = 0
        self.success_rate = 0.5
        self.creation_time = 0
        self.parent_signals = []  # For compositional signals
    
    def mutate(self, mutation_strength: float = 0.1):
        """Mutate the signal's meaning vector"""
        self.meaning_vector += np.random.normal(0, mutation_strength, 5)
        self.meaning_vector = np.clip(self.meaning_vector, 0, 1)
    
    def combine_with(self, other_signal: 'Signal') -> np.ndarray:
        """Combine meaning with another signal"""
        return (self.meaning_vector + other_signal.meaning_vector) / 2


class CommunicationSystem:
    """Evolving communication and proto-language system"""
    
    def __init__(self, mutation_rate: float = 0.05):
        self.signal_repertoire: Dict[int, Signal] = {}
        self.meaning_associations: Dict[str, List[int]] = defaultdict(list)
        self.syntax_rules: List[Tuple] = []  # Emerging grammar rules
        self.next_signal_id = 0
        self.mutation_rate = mutation_rate
        self.innovation_history = []
        
        # Initialize basic signals
        self._initialize_basic_signals()
    
    def _initialize_basic_signals(self):
        """Create basic survival signals"""
        basic_meanings = ['danger', 'food', 'mating', 'help', 'territory']
        for meaning in basic_meanings:
            signal = Signal(self.next_signal_id)
            signal.creation_time = 0
            self.signal_repertoire[self.next_signal_id] = signal
            self.meaning_associations[meaning].append(self.next_signal_id)
            self.next_signal_id += 1
    
    def create_new_signal(self, parent_signals: List[int] = None, 
                         context: str = None, time_step: int = 0) -> int:
        """Create new signal through combination or mutation"""
        signal = Signal(self.next_signal_id)
        signal.creation_time = time_step
        
        if parent_signals and len(parent_signals) >= 2:
            # Compositional signal creation
            parent1 = self.signal_repertoire[parent_signals[0]]
            parent2 = self.signal_repertoire[parent_signals[1]]
            signal.meaning_vector = parent1.combine_with(parent2)
            signal.parent_signals = parent_signals
            
            # Add some mutation
            signal.mutate(0.1)
            
            # Record innovation
            self.innovation_history.append({
                'type': 'compositional',
                'signal_id': self.next_signal_id,
                'parents': parent_signals,
                'time': time_step,
                'context': context
            })
        
        elif parent_signals and len(parent_signals) == 1:
            # Mutation-based signal creation
            parent = self.signal_repertoire[parent_signals[0]]
            signal.meaning_vector = parent.meaning_vector.copy()
            signal.parent_signals = parent_signals
            signal.mutate(0.2)
            
            self.innovation_history.append({
                'type': 'mutation',
                'signal_id': self.next_signal_id,
                'parent': parent_signals[0],
                'time': time_step,
                'context': context
            })
        
        else:
            # Completely novel signal
            self.innovation_history.append({
                'type': 'innovation',
                'signal_id': self.next_signal_id,
                'time': time_step,
                'context': context
            })
        
        self.signal_repertoire[self.next_signal_id] = signal
        
        # Associate with context if provided
        if context:
            self.meaning_associations[context].append(self.next_signal_id)
        
        self.next_signal_id += 1
        return signal.id
    
    def interpret_signal(self, signal_id: int, context: str = 'general') -> float:
        """Interpret signal meaning in given context"""
        if signal_id not in self.signal_repertoire:
            return 0.0
        
        signal = self.signal_repertoire[signal_id]
        
        # Context-dependent interpretation
        context_hash = hash(context) % 5
        interpretation = signal.meaning_vector[context_hash] * signal.intensity
        
        # Boost interpretation if signal has been successful in this context
        if signal.success_rate > 0.6:
            interpretation *= 1.2
        
        return np.clip(interpretation, 0, 1)
    
    def update_signal_success(self, signal_id: int, success: bool, context: str = None):
        """Update signal based on communication success"""
        if signal_id in self.signal_repertoire:
            signal = self.signal_repertoire[signal_id]
            signal.usage_count += 1
            
            # Update success rate with decay
            alpha = 0.1  # Learning rate
            signal.success_rate = (1 - alpha) * signal.success_rate + alpha * (1.0 if success else 0.0)
            
            # Context-specific learning could be added here
            if success and context:
                # Strengthen association with successful context
                context_hash = hash(context) % 5
                signal.meaning_vector[context_hash] = min(1.0, signal.meaning_vector[context_hash] + 0.05)
    
    def get_signals_for_context(self, context: str, threshold: float = 0.3) -> List[int]:
        """Get signals appropriate for a given context"""
        relevant_signals = []
        
        # Check explicitly associated signals
        if context in self.meaning_associations:
            relevant_signals.extend(self.meaning_associations[context])
        
        # Check signals with high interpretation in this context
        for signal_id, signal in self.signal_repertoire.items():
            interpretation = self.interpret_signal(signal_id, context)
            if interpretation > threshold:
                relevant_signals.append(signal_id)
        
        return list(set(relevant_signals))  # Remove duplicates
    
    def prune_signals(self, min_usage: int = 5, min_success_rate: float = 0.2):
        """Remove unsuccessful or unused signals"""
        signals_to_remove = []
        
        for signal_id, signal in self.signal_repertoire.items():
            if (signal.usage_count < min_usage and signal.success_rate < min_success_rate):
                signals_to_remove.append(signal_id)
        
        # Don't remove basic signals
        basic_signal_count = 5  # Number of initial basic signals
        signals_to_remove = [sid for sid in signals_to_remove if sid >= basic_signal_count]
        
        for signal_id in signals_to_remove:
            del self.signal_repertoire[signal_id]
            
            # Remove from meaning associations
            for meaning, signal_list in self.meaning_associations.items():
                if signal_id in signal_list:
                    signal_list.remove(signal_id)
    
    def evolve_syntax(self):
        """Develop simple syntax rules based on signal combinations"""
        if len(self.signal_repertoire) < 3:
            return
        
        # Look for patterns in compositional signals
        compositional_signals = [
            signal for signal in self.signal_repertoire.values() 
            if len(signal.parent_signals) == 2
        ]
        
        if len(compositional_signals) >= 2:
            # Find common patterns
            pattern_counts = defaultdict(int)
            
            for signal in compositional_signals:
                # Create pattern based on parent signal contexts
                parent_contexts = []
                for parent_id in signal.parent_signals:
                    parent_contexts.append(self._get_primary_context(parent_id))
                
                pattern = tuple(sorted(parent_contexts))
                pattern_counts[pattern] += 1
            
            # Add successful patterns as syntax rules
            for pattern, count in pattern_counts.items():
                if count >= 2 and pattern not in self.syntax_rules:
                    self.syntax_rules.append(pattern)
    
    def _get_primary_context(self, signal_id: int) -> str:
        """Get the primary context associated with a signal"""
        best_context = 'general'
        best_score = 0
        
        for context, signal_list in self.meaning_associations.items():
            if signal_id in signal_list:
                score = self.interpret_signal(signal_id, context)
                if score > best_score:
                    best_score = score
                    best_context = context
        
        return best_context
    
    def get_communication_complexity(self) -> Dict[str, float]:
        """Measure the complexity of the communication system"""
        if not self.signal_repertoire:
            return {'vocabulary_size': 0, 'syntax_rules': 0, 'compositionality': 0, 'success_rate': 0}
        
        # Vocabulary size
        vocab_size = len(self.signal_repertoire)
        
        # Number of syntax rules
        syntax_count = len(self.syntax_rules)
        
        # Compositionality measure
        compositional_signals = sum(1 for signal in self.signal_repertoire.values() 
                                  if len(signal.parent_signals) >= 2)
        compositionality = compositional_signals / vocab_size if vocab_size > 0 else 0
        
        # Average success rate
        success_rates = [signal.success_rate for signal in self.signal_repertoire.values() 
                        if signal.usage_count > 0]
        avg_success = np.mean(success_rates) if success_rates else 0
        
        return {
            'vocabulary_size': vocab_size,
            'syntax_rules': syntax_count,
            'compositionality': compositionality,
            'success_rate': avg_success
        }
    
    def copy(self) -> 'CommunicationSystem':
        """Create a copy of this communication system for inheritance"""
        new_system = CommunicationSystem(self.mutation_rate)
        
        # Copy a subset of signals (cultural transmission is partial)
        signal_ids = list(self.signal_repertoire.keys())
        num_to_copy = min(len(signal_ids), 10)  # Limit inherited signals
        
        if signal_ids:
            inherited_signals = random.sample(signal_ids, num_to_copy)
            
            for old_id in inherited_signals:
                old_signal = self.signal_repertoire[old_id]
                new_signal = Signal(new_system.next_signal_id)
                new_signal.meaning_vector = old_signal.meaning_vector.copy()
                new_signal.intensity = old_signal.intensity
                
                # Add some mutation during inheritance
                if random.random() < self.mutation_rate:
                    new_signal.mutate(0.05)
                
                new_system.signal_repertoire[new_system.next_signal_id] = new_signal
                new_system.next_signal_id += 1
        
        return new_system


class LanguageEvolution:
    """Population-level language evolution dynamics"""
    
    def __init__(self):
        self.population_signals = defaultdict(int)  # signal_id -> population count
        self.signal_history = []  # Track signal emergence over time
        self.language_tree = {}  # Phylogenetic tree of signals
    
    def update_population_language(self, individuals: List, time_step: int):
        """Update population-level language statistics"""
        self.population_signals.clear()
        
        # Count signal usage across population
        for individual in individuals:
            if hasattr(individual, 'communication'):
                for signal_id in individual.communication.signal_repertoire:
                    self.population_signals[signal_id] += 1
        
        # Record language state
        self.signal_history.append({
            'time': time_step,
            'unique_signals': len(self.population_signals),
            'total_signals': sum(self.population_signals.values()),
            'most_common': max(self.population_signals.items(), key=lambda x: x[1]) if self.population_signals else None
        })
    
    def detect_language_emergence(self, threshold: int = 50) -> bool:
        """Detect if a stable language has emerged"""
        if len(self.signal_history) < 100:
            return False
        
        recent_history = self.signal_history[-50:]
        unique_signals = [entry['unique_signals'] for entry in recent_history]
        
        # Check for stability in vocabulary size
        return (np.std(unique_signals) < 5 and np.mean(unique_signals) > threshold)
    
    def get_language_diversity(self) -> float:
        """Calculate language diversity in the population"""
        if not self.population_signals:
            return 0.0
        
        total = sum(self.population_signals.values())
        frequencies = [count / total for count in self.population_signals.values()]
        
        # Calculate Shannon diversity
        diversity = -sum(f * np.log(f) for f in frequencies if f > 0)
        return diversity
