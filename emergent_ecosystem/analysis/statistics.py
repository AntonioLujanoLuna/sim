"""
Statistics tracking and analysis system.

This module implements comprehensive statistics collection, analysis,
and reporting for the emergent intelligence ecosystem.
"""

import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional


class StatisticsTracker:
    """Comprehensive statistics tracking and analysis"""
    
    def __init__(self, history_length: int = 1000):
        self.history_length = history_length
        self.generation_stats = deque(maxlen=history_length)
        self.cultural_evolution_data = deque(maxlen=history_length)
        self.communication_evolution_data = deque(maxlen=history_length)
        
        # Specialized tracking
        self.species_evolution = defaultdict(lambda: deque(maxlen=history_length))
        self.intelligence_evolution = deque(maxlen=history_length)
        self.social_evolution = deque(maxlen=history_length)
        self.environmental_evolution = deque(maxlen=history_length)
        
    def update(self, individuals: List, social_network, environment, time_step: int):
        """Update all statistics with current simulation state"""
        if not individuals:
            return
        
        # Calculate comprehensive statistics
        stats = self._calculate_comprehensive_stats(
            individuals, social_network, environment, time_step
        )
        
        # Store in history
        self.generation_stats.append(stats)
        
        # Update specialized tracking
        self._update_species_evolution(individuals, time_step)
        self._update_intelligence_evolution(individuals, time_step)
        self._update_social_evolution(social_network, time_step)
        self._update_environmental_evolution(environment, time_step)
        self._update_cultural_evolution(individuals, time_step)
        self._update_communication_evolution(individuals, time_step)
    
    def _calculate_comprehensive_stats(self, individuals: List, social_network, 
                                     environment, time_step: int) -> Dict[str, Any]:
        """Calculate comprehensive statistics for current state"""
        # Basic population statistics
        population_size = len(individuals)
        species_counts = defaultdict(int)
        
        # Individual-level metrics
        energy_levels = []
        intelligence_levels = []
        sociability_levels = []
        aggression_levels = []
        ages = []
        generations = []
        cultural_knowledge_levels = []
        
        for ind in individuals:
            species_counts[ind.species_name] += 1
            energy_levels.append(ind.energy)
            intelligence_levels.append(ind.intelligence)
            sociability_levels.append(ind.sociability)
            aggression_levels.append(ind.aggression)
            ages.append(ind.age)
            generations.append(ind.generation)
            cultural_knowledge_levels.append(sum(ind.cultural_knowledge.values()))
        
        # Social network metrics
        network_metrics = social_network.get_network_metrics()
        num_relationships = sum(len(rels) for rels in social_network.relationships.values())
        num_communities = len(social_network.communities)
        num_leaders = len(social_network.leaders)
        
        # Communication metrics
        total_signals = sum(len(ind.communication.signal_repertoire) for ind in individuals)
        active_signals = sum(len(ind.active_signals) for ind in individuals)
        unique_signals = len(set().union(*[ind.communication.signal_repertoire.keys() 
                                         for ind in individuals]))
        
        # Environmental metrics
        env_summary = environment.get_environmental_summary()
        
        # Calculate derived metrics
        avg_energy = np.mean(energy_levels) if energy_levels else 0
        avg_intelligence = np.mean(intelligence_levels) if intelligence_levels else 0
        avg_sociability = np.mean(sociability_levels) if sociability_levels else 0
        avg_aggression = np.mean(aggression_levels) if aggression_levels else 0
        avg_age = np.mean(ages) if ages else 0
        max_generation = max(generations) if generations else 0
        avg_cultural_knowledge = np.mean(cultural_knowledge_levels) if cultural_knowledge_levels else 0
        
        # Diversity metrics
        species_diversity = self._calculate_shannon_diversity(list(species_counts.values()))
        intelligence_diversity = np.std(intelligence_levels) if len(intelligence_levels) > 1 else 0
        cultural_diversity = np.std(cultural_knowledge_levels) if len(cultural_knowledge_levels) > 1 else 0
        
        # Fitness metrics
        fitness_scores = [
            ind.energy + ind.intelligence * 10 + len(ind.social_memory) * 2
            for ind in individuals
        ]
        avg_fitness = np.mean(fitness_scores) if fitness_scores else 0
        fitness_variance = np.var(fitness_scores) if len(fitness_scores) > 1 else 0
        
        return {
            'time_step': time_step,
            'population_size': population_size,
            'species_counts': dict(species_counts),
            
            # Individual metrics
            'avg_energy': avg_energy,
            'avg_intelligence': avg_intelligence,
            'avg_sociability': avg_sociability,
            'avg_aggression': avg_aggression,
            'avg_age': avg_age,
            'max_generation': max_generation,
            'avg_cultural_knowledge': avg_cultural_knowledge,
            
            # Diversity metrics
            'species_diversity': species_diversity,
            'intelligence_diversity': intelligence_diversity,
            'cultural_diversity': cultural_diversity,
            
            # Fitness metrics
            'avg_fitness': avg_fitness,
            'fitness_variance': fitness_variance,
            
            # Social metrics
            'num_relationships': num_relationships,
            'num_communities': num_communities,
            'num_leaders': num_leaders,
            'social_density': network_metrics.get('density', 0),
            'social_clustering': network_metrics.get('clustering', 0),
            
            # Communication metrics
            'total_signals': total_signals,
            'active_signals': active_signals,
            'unique_signals': unique_signals,
            'avg_signals_per_individual': total_signals / population_size if population_size > 0 else 0,
            
            # Environmental metrics
            'environmental_health': env_summary.get('global_health', 0),
            'environmental_complexity': env_summary.get('environmental_complexity', 0),
            'total_patches': env_summary.get('total_patches', 0),
            'avg_resource_level': env_summary.get('avg_resource_level', 0)
        }
    
    def _update_species_evolution(self, individuals: List, time_step: int):
        """Track evolution of each species separately"""
        species_data = defaultdict(lambda: {
            'count': 0,
            'avg_intelligence': 0,
            'avg_energy': 0,
            'avg_sociability': 0,
            'avg_cultural_knowledge': 0
        })
        
        for ind in individuals:
            species = ind.species_name
            species_data[species]['count'] += 1
            species_data[species]['avg_intelligence'] += ind.intelligence
            species_data[species]['avg_energy'] += ind.energy
            species_data[species]['avg_sociability'] += ind.sociability
            species_data[species]['avg_cultural_knowledge'] += sum(ind.cultural_knowledge.values())
        
        # Calculate averages
        for species, data in species_data.items():
            if data['count'] > 0:
                data['avg_intelligence'] /= data['count']
                data['avg_energy'] /= data['count']
                data['avg_sociability'] /= data['count']
                data['avg_cultural_knowledge'] /= data['count']
            
            data['time_step'] = time_step
            self.species_evolution[species].append(data.copy())
    
    def _update_intelligence_evolution(self, individuals: List, time_step: int):
        """Track intelligence evolution over time"""
        if not individuals:
            return
        
        intelligence_data = {
            'time_step': time_step,
            'avg_intelligence': np.mean([ind.intelligence for ind in individuals]),
            'max_intelligence': max(ind.intelligence for ind in individuals),
            'min_intelligence': min(ind.intelligence for ind in individuals),
            'intelligence_std': np.std([ind.intelligence for ind in individuals]),
            'top_10_percent_avg': np.mean(sorted([ind.intelligence for ind in individuals], reverse=True)[:max(1, len(individuals)//10)])
        }
        
        self.intelligence_evolution.append(intelligence_data)
    
    def _update_social_evolution(self, social_network, time_step: int):
        """Track social network evolution"""
        metrics = social_network.get_network_metrics()
        
        social_data = {
            'time_step': time_step,
            'network_density': metrics.get('density', 0),
            'network_clustering': metrics.get('clustering', 0),
            'num_communities': metrics.get('communities', 0),
            'num_leaders': metrics.get('leaders', 0),
            'total_nodes': metrics.get('nodes', 0),
            'total_edges': metrics.get('edges', 0),
            'social_cohesion': social_network.calculate_social_cohesion()
        }
        
        self.social_evolution.append(social_data)
    
    def _update_environmental_evolution(self, environment, time_step: int):
        """Track environmental changes over time"""
        env_summary = environment.get_environmental_summary()
        
        env_data = {
            'time_step': time_step,
            'global_health': env_summary.get('global_health', 0),
            'biodiversity_index': env_summary.get('biodiversity_index', 0),
            'environmental_complexity': env_summary.get('environmental_complexity', 0),
            'avg_resource_level': env_summary.get('avg_resource_level', 0),
            'avg_quality': env_summary.get('avg_quality', 0),
            'avg_stress': env_summary.get('avg_stress', 0),
            'coevolution_events': env_summary.get('coevolution_events', 0)
        }
        
        self.environmental_evolution.append(env_data)
    
    def _update_cultural_evolution(self, individuals: List, time_step: int):
        """Track cultural knowledge evolution"""
        if not individuals:
            return
        
        # Aggregate cultural knowledge across population
        all_cultural_knowledge = defaultdict(float)
        for ind in individuals:
            for knowledge, value in ind.cultural_knowledge.items():
                all_cultural_knowledge[knowledge] += value
        
        # Calculate cultural metrics
        total_cultural_knowledge = sum(all_cultural_knowledge.values())
        num_cultural_items = len(all_cultural_knowledge)
        cultural_diversity = self._calculate_shannon_diversity(list(all_cultural_knowledge.values()))
        
        cultural_data = {
            'time_step': time_step,
            'total_cultural_knowledge': total_cultural_knowledge,
            'num_cultural_items': num_cultural_items,
            'cultural_diversity': cultural_diversity,
            'avg_cultural_per_individual': total_cultural_knowledge / len(individuals),
            'cultural_innovation_rate': self._calculate_cultural_innovation_rate(individuals)
        }
        
        self.cultural_evolution_data.append(cultural_data)
    
    def _update_communication_evolution(self, individuals: List, time_step: int):
        """Track communication system evolution"""
        if not individuals:
            return
        
        # Aggregate communication data
        all_signals = set()
        total_signal_usage = 0
        compositional_signals = 0
        
        for ind in individuals:
            all_signals.update(ind.communication.signal_repertoire.keys())
            total_signal_usage += sum(signal.usage_count 
                                    for signal in ind.communication.signal_repertoire.values())
            compositional_signals += sum(1 for signal in ind.communication.signal_repertoire.values()
                                       if hasattr(signal, 'parent_signals') and signal.parent_signals)
        
        # Calculate communication complexity
        avg_signals_per_individual = len(all_signals) / len(individuals) if individuals else 0
        compositionality_ratio = compositional_signals / len(all_signals) if all_signals else 0
        
        comm_data = {
            'time_step': time_step,
            'total_unique_signals': len(all_signals),
            'avg_signals_per_individual': avg_signals_per_individual,
            'total_signal_usage': total_signal_usage,
            'compositional_signals': compositional_signals,
            'compositionality_ratio': compositionality_ratio,
            'communication_complexity': self._calculate_communication_complexity(individuals)
        }
        
        self.communication_evolution_data.append(comm_data)
    
    def _calculate_shannon_diversity(self, values: List[float]) -> float:
        """Calculate Shannon diversity index"""
        if not values or sum(values) == 0:
            return 0
        
        total = sum(values)
        proportions = [v / total for v in values if v > 0]
        return -sum(p * np.log(p) for p in proportions)
    
    def _calculate_cultural_innovation_rate(self, individuals: List) -> float:
        """Calculate rate of cultural innovation"""
        innovation_scores = []
        for ind in individuals:
            if hasattr(ind, 'innovation_tendency'):
                innovation_scores.append(ind.innovation_tendency)
        
        return np.mean(innovation_scores) if innovation_scores else 0
    
    def _calculate_communication_complexity(self, individuals: List) -> float:
        """Calculate overall communication system complexity"""
        if not individuals:
            return 0
        
        complexity_factors = []
        
        for ind in individuals:
            comm_system = ind.communication
            
            # Vocabulary size factor
            vocab_size = len(comm_system.signal_repertoire)
            
            # Syntax rules factor
            syntax_complexity = len(comm_system.syntax_rules)
            
            # Signal success rate factor
            success_rates = [signal.success_rate for signal in comm_system.signal_repertoire.values()]
            avg_success = np.mean(success_rates) if success_rates else 0
            
            # Compositional complexity
            compositional_count = sum(1 for signal in comm_system.signal_repertoire.values()
                                    if hasattr(signal, 'parent_signals') and signal.parent_signals)
            compositionality = compositional_count / vocab_size if vocab_size > 0 else 0
            
            individual_complexity = vocab_size + syntax_complexity + avg_success + compositionality
            complexity_factors.append(individual_complexity)
        
        return np.mean(complexity_factors)
    
    def get_latest_stats(self) -> Dict[str, Any]:
        """Get the most recent statistics"""
        if not self.generation_stats:
            return {}
        return self.generation_stats[-1]
    
    def get_evolution_trends(self, metric: str, window_size: int = 50) -> Dict[str, float]:
        """Calculate evolution trends for a specific metric"""
        if len(self.generation_stats) < window_size:
            return {'trend': 0, 'acceleration': 0, 'volatility': 0}
        
        recent_data = list(self.generation_stats)[-window_size:]
        values = [entry.get(metric, 0) for entry in recent_data]
        
        if len(values) < 2:
            return {'trend': 0, 'acceleration': 0, 'volatility': 0}
        
        # Calculate trend (linear regression slope)
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]
        
        # Calculate acceleration (second derivative)
        if len(values) >= 3:
            acceleration = np.mean(np.diff(values, 2))
        else:
            acceleration = 0
        
        # Calculate volatility (standard deviation)
        volatility = np.std(values)
        
        return {
            'trend': trend,
            'acceleration': acceleration,
            'volatility': volatility,
            'current_value': values[-1],
            'change_from_start': values[-1] - values[0] if values else 0
        }
    
    def get_species_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare evolution across different species"""
        comparison = {}
        
        for species, evolution_data in self.species_evolution.items():
            if evolution_data:
                latest = evolution_data[-1]
                comparison[species] = {
                    'population': latest['count'],
                    'avg_intelligence': latest['avg_intelligence'],
                    'avg_energy': latest['avg_energy'],
                    'avg_sociability': latest['avg_sociability'],
                    'avg_cultural_knowledge': latest['avg_cultural_knowledge']
                }
        
        return comparison
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive statistics report"""
        if not self.generation_stats:
            return {'error': 'No data available'}
        
        latest_stats = self.generation_stats[-1]
        
        # Calculate trends for key metrics
        key_metrics = ['avg_intelligence', 'avg_energy', 'population_size', 
                      'avg_cultural_knowledge', 'social_density']
        trends = {}
        for metric in key_metrics:
            trends[metric] = self.get_evolution_trends(metric)
        
        return {
            'current_state': latest_stats,
            'evolution_trends': trends,
            'species_comparison': self.get_species_comparison(),
            'total_time_steps': len(self.generation_stats),
            'data_quality': {
                'completeness': len(self.generation_stats) / self.history_length,
                'latest_timestamp': latest_stats.get('time_step', 0)
            }
        } 