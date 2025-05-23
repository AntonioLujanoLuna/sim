"""
Evolutionary dynamics analysis and metrics.

This module implements analysis of evolutionary processes including selection pressures,
fitness landscapes, phylogenetic tracking, and co-evolutionary dynamics.
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import math
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram


@dataclass
class IndividualRecord:
    """Record of an individual for evolutionary tracking"""
    individual_id: int
    parent_ids: Tuple[Optional[int], Optional[int]]
    birth_time: int
    death_time: Optional[int] = None
    species: str = ""
    traits: Dict[str, float] = field(default_factory=dict)
    fitness_history: List[float] = field(default_factory=list)
    reproductive_success: int = 0
    generation: int = 0


@dataclass
class SpeciesRecord:
    """Record of a species for evolutionary tracking"""
    species_name: str
    emergence_time: int
    extinction_time: Optional[int] = None
    population_history: List[Tuple[int, int]] = field(default_factory=list)  # (time, population)
    trait_evolution: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    fitness_evolution: List[Tuple[int, float]] = field(default_factory=list)


class PhylogeneticTree:
    """Phylogenetic tree construction and analysis"""
    
    def __init__(self):
        self.nodes = {}  # individual_id -> {parent, children, traits, time}
        self.species_lineages = defaultdict(list)
        self.extinction_events = []
        
    def add_individual(self, individual_id: int, parent_ids: Tuple[Optional[int], Optional[int]],
                      traits: Dict[str, float], birth_time: int, species: str):
        """Add individual to phylogenetic tracking"""
        self.nodes[individual_id] = {
            'parents': parent_ids,
            'children': [],
            'traits': traits.copy(),
            'birth_time': birth_time,
            'death_time': None,
            'species': species,
            'fitness': 0.0
        }
        
        # Update parent-child relationships
        for parent_id in parent_ids:
            if parent_id is not None and parent_id in self.nodes:
                self.nodes[parent_id]['children'].append(individual_id)
        
        # Track species lineage
        self.species_lineages[species].append(individual_id)
    
    def mark_death(self, individual_id: int, death_time: int, final_fitness: float):
        """Mark individual as dead"""
        if individual_id in self.nodes:
            self.nodes[individual_id]['death_time'] = death_time
            self.nodes[individual_id]['fitness'] = final_fitness
    
    def calculate_genetic_distance(self, id1: int, id2: int, trait_weights: Dict[str, float] = None) -> float:
        """Calculate genetic distance between two individuals"""
        if id1 not in self.nodes or id2 not in self.nodes:
            return float('inf')
        
        traits1 = self.nodes[id1]['traits']
        traits2 = self.nodes[id2]['traits']
        
        if trait_weights is None:
            trait_weights = {trait: 1.0 for trait in traits1.keys()}
        
        distance = 0.0
        for trait in traits1.keys():
            if trait in traits2 and trait in trait_weights:
                distance += trait_weights[trait] * (traits1[trait] - traits2[trait])**2
        
        return math.sqrt(distance)
    
    def find_common_ancestor(self, id1: int, id2: int) -> Optional[int]:
        """Find most recent common ancestor of two individuals"""
        if id1 not in self.nodes or id2 not in self.nodes:
            return None
        
        # Get all ancestors of id1
        ancestors1 = set()
        queue = [id1]
        while queue:
            current = queue.pop(0)
            if current in self.nodes:
                ancestors1.add(current)
                parents = self.nodes[current]['parents']
                for parent in parents:
                    if parent is not None:
                        queue.append(parent)
        
        # Find first common ancestor for id2
        queue = [id2]
        visited = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            if current in ancestors1:
                return current
            
            if current in self.nodes:
                parents = self.nodes[current]['parents']
                for parent in parents:
                    if parent is not None:
                        queue.append(parent)
        
        return None
    
    def calculate_species_diversity(self, current_time: int) -> float:
        """Calculate phylogenetic diversity at current time"""
        living_individuals = [
            ind_id for ind_id, data in self.nodes.items()
            if data['death_time'] is None or data['death_time'] > current_time
        ]
        
        if len(living_individuals) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(living_individuals)):
            for j in range(i + 1, len(living_individuals)):
                dist = self.calculate_genetic_distance(living_individuals[i], living_individuals[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def get_lineage_statistics(self, species: str) -> Dict[str, Any]:
        """Get statistics for a species lineage"""
        if species not in self.species_lineages:
            return {}
        
        lineage = self.species_lineages[species]
        
        # Calculate lineage metrics
        birth_times = [self.nodes[ind_id]['birth_time'] for ind_id in lineage 
                      if ind_id in self.nodes]
        death_times = [self.nodes[ind_id]['death_time'] for ind_id in lineage 
                      if ind_id in self.nodes and self.nodes[ind_id]['death_time'] is not None]
        
        fitnesses = [self.nodes[ind_id]['fitness'] for ind_id in lineage 
                    if ind_id in self.nodes and self.nodes[ind_id]['fitness'] > 0]
        
        return {
            'population_size': len(lineage),
            'emergence_time': min(birth_times) if birth_times else 0,
            'extinction_time': max(death_times) if death_times and len(death_times) == len(lineage) else None,
            'avg_fitness': np.mean(fitnesses) if fitnesses else 0,
            'fitness_variance': np.var(fitnesses) if len(fitnesses) > 1 else 0,
            'lineage_length': max(birth_times) - min(birth_times) if len(birth_times) > 1 else 0
        }


class FitnessLandscape:
    """Fitness landscape analysis and visualization"""
    
    def __init__(self, trait_names: List[str]):
        self.trait_names = trait_names
        self.fitness_samples = []  # List of (traits, fitness) tuples
        self.landscape_resolution = 20
        self.adaptive_peaks = []
        
    def add_fitness_sample(self, traits: Dict[str, float], fitness: float):
        """Add a fitness sample to the landscape"""
        trait_vector = [traits.get(name, 0.0) for name in self.trait_names]
        self.fitness_samples.append((trait_vector, fitness))
        
        # Limit sample size for performance
        if len(self.fitness_samples) > 10000:
            self.fitness_samples = self.fitness_samples[-5000:]
    
    def estimate_fitness(self, traits: Dict[str, float], k: int = 10) -> float:
        """Estimate fitness using k-nearest neighbors"""
        if not self.fitness_samples:
            return 0.5
        
        trait_vector = [traits.get(name, 0.0) for name in self.trait_names]
        
        # Calculate distances to all samples
        distances = []
        for sample_traits, sample_fitness in self.fitness_samples:
            dist = np.sqrt(sum((t1 - t2)**2 for t1, t2 in zip(trait_vector, sample_traits)))
            distances.append((dist, sample_fitness))
        
        # Get k nearest neighbors
        distances.sort()
        nearest_k = distances[:min(k, len(distances))]
        
        # Weight by inverse distance
        if nearest_k[0][0] == 0:  # Exact match
            return nearest_k[0][1]
        
        weights = [1.0 / (dist + 1e-6) for dist, _ in nearest_k]
        weighted_fitness = sum(w * fitness for w, (_, fitness) in zip(weights, nearest_k))
        total_weight = sum(weights)
        
        return weighted_fitness / total_weight if total_weight > 0 else 0.5
    
    def find_adaptive_peaks(self, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find adaptive peaks in the fitness landscape"""
        if len(self.fitness_samples) < 10:
            return []
        
        peaks = []
        high_fitness_samples = [(traits, fitness) for traits, fitness in self.fitness_samples 
                               if fitness > threshold]
        
        # Cluster high-fitness samples
        if len(high_fitness_samples) < 2:
            return peaks
        
        trait_vectors = [traits for traits, _ in high_fitness_samples]
        fitnesses = [fitness for _, fitness in high_fitness_samples]
        
        # Simple peak detection using local maxima
        for i, (traits, fitness) in enumerate(high_fitness_samples):
            is_peak = True
            
            # Check if this is a local maximum
            for j, (other_traits, other_fitness) in enumerate(high_fitness_samples):
                if i != j:
                    distance = np.sqrt(sum((t1 - t2)**2 for t1, t2 in zip(traits, other_traits)))
                    if distance < 0.2 and other_fitness > fitness:  # Close neighbor with higher fitness
                        is_peak = False
                        break
            
            if is_peak:
                peak_traits = {name: traits[idx] for idx, name in enumerate(self.trait_names)}
                peaks.append({
                    'traits': peak_traits,
                    'fitness': fitness,
                    'prominence': self._calculate_peak_prominence(traits, fitnesses)
                })
        
        self.adaptive_peaks = peaks
        return peaks
    
    def _calculate_peak_prominence(self, peak_traits: List[float], all_fitnesses: List[float]) -> float:
        """Calculate prominence of a fitness peak"""
        peak_fitness = self.estimate_fitness(
            {name: peak_traits[idx] for idx, name in enumerate(self.trait_names)}
        )
        
        # Prominence as difference from surrounding fitness
        nearby_fitness = []
        for traits, fitness in self.fitness_samples:
            distance = np.sqrt(sum((t1 - t2)**2 for t1, t2 in zip(peak_traits, traits)))
            if 0.1 < distance < 0.3:  # Ring around peak
                nearby_fitness.append(fitness)
        
        if nearby_fitness:
            return peak_fitness - np.mean(nearby_fitness)
        else:
            return peak_fitness - np.mean(all_fitnesses)
    
    def calculate_landscape_ruggedness(self) -> float:
        """Calculate ruggedness of fitness landscape"""
        if len(self.fitness_samples) < 10:
            return 0.0
        
        # Sample fitness at random points and calculate variance
        sample_size = min(1000, len(self.fitness_samples))
        sample_indices = random.sample(range(len(self.fitness_samples)), sample_size)
        
        fitness_values = [self.fitness_samples[i][1] for i in sample_indices]
        return np.std(fitness_values)
    
    def get_landscape_gradient(self, traits: Dict[str, float], epsilon: float = 0.01) -> Dict[str, float]:
        """Calculate fitness gradient at given point"""
        base_fitness = self.estimate_fitness(traits)
        gradient = {}
        
        for trait_name in self.trait_names:
            # Perturb trait slightly
            perturbed_traits = traits.copy()
            perturbed_traits[trait_name] += epsilon
            
            # Calculate fitness difference
            perturbed_fitness = self.estimate_fitness(perturbed_traits)
            gradient[trait_name] = (perturbed_fitness - base_fitness) / epsilon
        
        return gradient


class EvolutionAnalyzer:
    """Main evolutionary dynamics analyzer"""
    
    def __init__(self):
        self.individual_records = {}
        self.species_records = {}
        self.phylogenetic_tree = PhylogeneticTree()
        self.fitness_landscapes = {}  # species -> FitnessLandscape
        
        # Evolutionary metrics tracking
        self.generation_metrics = []
        self.selection_pressures = defaultdict(list)
        self.mutation_rates = defaultdict(list)
        self.gene_flow_matrix = defaultdict(lambda: defaultdict(int))
        
        # Co-evolutionary tracking
        self.coevolution_matrix = defaultdict(lambda: defaultdict(list))
        self.species_interactions = defaultdict(lambda: defaultdict(int))
        
    def add_individual(self, individual_id: int, traits: Dict[str, float], 
                      species: str, generation: int, parent_ids: Tuple[Optional[int], Optional[int]] = (None, None),
                      birth_time: int = 0):
        """Add individual to evolutionary tracking"""
        record = IndividualRecord(
            individual_id=individual_id,
            parent_ids=parent_ids,
            birth_time=birth_time,
            species=species,
            traits=traits.copy(),
            generation=generation
        )
        
        self.individual_records[individual_id] = record
        
        # Add to phylogenetic tree
        self.phylogenetic_tree.add_individual(individual_id, parent_ids, traits, birth_time, species)
        
        # Initialize fitness landscape for species if needed
        if species not in self.fitness_landscapes:
            trait_names = list(traits.keys())
            self.fitness_landscapes[species] = FitnessLandscape(trait_names)
        
        # Add to species record
        if species not in self.species_records:
            self.species_records[species] = SpeciesRecord(species, birth_time)
    
    def update_fitness(self, individual_id: int, fitness: float):
        """Update individual fitness"""
        if individual_id in self.individual_records:
            record = self.individual_records[individual_id]
            record.fitness_history.append(fitness)
            
            # Add to fitness landscape
            species = record.species
            if species in self.fitness_landscapes:
                self.fitness_landscapes[species].add_fitness_sample(record.traits, fitness)
    
    def mark_death(self, individual_id: int, death_time: int):
        """Mark individual as dead"""
        if individual_id in self.individual_records:
            record = self.individual_records[individual_id]
            record.death_time = death_time
            
            # Calculate final fitness
            final_fitness = np.mean(record.fitness_history) if record.fitness_history else 0.0
            self.phylogenetic_tree.mark_death(individual_id, death_time, final_fitness)
    
    def record_reproduction(self, parent_id: int, offspring_id: int):
        """Record reproduction event"""
        if parent_id in self.individual_records:
            self.individual_records[parent_id].reproductive_success += 1
    
    def calculate_selection_pressure(self, species: str, trait: str, current_time: int) -> float:
        """Calculate selection pressure on a specific trait"""
        living_individuals = [
            record for record in self.individual_records.values()
            if record.species == species and 
            (record.death_time is None or record.death_time > current_time)
        ]
        
        if len(living_individuals) < 10:
            return 0.0
        
        # Calculate correlation between trait value and fitness
        trait_values = [record.traits.get(trait, 0) for record in living_individuals]
        fitness_values = [np.mean(record.fitness_history) if record.fitness_history else 0 
                         for record in living_individuals]
        
        if len(set(trait_values)) <= 1 or len(set(fitness_values)) <= 1:
            return 0.0
        
        correlation, p_value = stats.pearsonr(trait_values, fitness_values)
        return correlation if not np.isnan(correlation) else 0.0
    
    def calculate_heritability(self, species: str, trait: str) -> float:
        """Calculate heritability of a trait"""
        parent_child_pairs = []
        
        for record in self.individual_records.values():
            if record.species == species and record.parent_ids[0] is not None:
                parent_id = record.parent_ids[0]
                if parent_id in self.individual_records:
                    parent_trait = self.individual_records[parent_id].traits.get(trait, 0)
                    child_trait = record.traits.get(trait, 0)
                    parent_child_pairs.append((parent_trait, child_trait))
        
        if len(parent_child_pairs) < 10:
            return 0.0
        
        parent_values = [pair[0] for pair in parent_child_pairs]
        child_values = [pair[1] for pair in parent_child_pairs]
        
        if len(set(parent_values)) <= 1 or len(set(child_values)) <= 1:
            return 0.0
        
        correlation, _ = stats.pearsonr(parent_values, child_values)
        return correlation if not np.isnan(correlation) else 0.0
    
    def detect_evolutionary_events(self, current_time: int) -> List[Dict[str, Any]]:
        """Detect significant evolutionary events"""
        events = []
        
        for species in self.species_records.keys():
            # Check for adaptive radiations
            recent_births = [
                record for record in self.individual_records.values()
                if record.species == species and 
                current_time - record.birth_time < 100
            ]
            
            if len(recent_births) > 20:  # Rapid population expansion
                events.append({
                    'type': 'adaptive_radiation',
                    'species': species,
                    'time': current_time,
                    'magnitude': len(recent_births)
                })
            
            # Check for evolutionary stasis
            if species in self.fitness_landscapes:
                landscape = self.fitness_landscapes[species]
                if len(landscape.fitness_samples) > 100:
                    recent_fitness = [f for _, f in landscape.fitness_samples[-50:]]
                    older_fitness = [f for _, f in landscape.fitness_samples[-100:-50]]
                    
                    if len(recent_fitness) > 10 and len(older_fitness) > 10:
                        recent_mean = np.mean(recent_fitness)
                        older_mean = np.mean(older_fitness)
                        
                        if abs(recent_mean - older_mean) < 0.05:  # Very small change
                            events.append({
                                'type': 'evolutionary_stasis',
                                'species': species,
                                'time': current_time,
                                'fitness_stability': abs(recent_mean - older_mean)
                            })
            
            # Check for rapid evolution
            for trait in ['intelligence', 'sociability', 'aggression']:
                selection_pressure = self.calculate_selection_pressure(species, trait, current_time)
                if abs(selection_pressure) > 0.7:  # Strong selection
                    events.append({
                        'type': 'rapid_evolution',
                        'species': species,
                        'trait': trait,
                        'time': current_time,
                        'selection_pressure': selection_pressure
                    })
        
        return events
    
    def analyze_coevolution(self, species1: str, species2: str) -> Dict[str, float]:
        """Analyze co-evolutionary dynamics between two species"""
        # Get trait evolution for both species
        species1_records = [r for r in self.individual_records.values() if r.species == species1]
        species2_records = [r for r in self.individual_records.values() if r.species == species2]
        
        if len(species1_records) < 10 or len(species2_records) < 10:
            return {'coevolution_strength': 0.0}
        
        # Time-aligned trait evolution
        time_series1 = defaultdict(list)
        time_series2 = defaultdict(list)
        
        for record in species1_records:
            for trait, value in record.traits.items():
                time_series1[(record.birth_time // 100, trait)].append(value)
        
        for record in species2_records:
            for trait, value in record.traits.items():
                time_series2[(record.birth_time // 100, trait)].append(value)
        
        # Calculate cross-correlations
        correlations = []
        for trait in ['intelligence', 'aggression', 'sociability']:
            trait1_evolution = []
            trait2_evolution = []
            
            for time_bin in range(10):  # Look at last 10 time bins
                if (time_bin, trait) in time_series1 and (time_bin, trait) in time_series2:
                    trait1_evolution.append(np.mean(time_series1[(time_bin, trait)]))
                    trait2_evolution.append(np.mean(time_series2[(time_bin, trait)]))
            
            if len(trait1_evolution) > 3:
                corr, _ = stats.pearsonr(trait1_evolution, trait2_evolution)
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        coevolution_strength = np.mean(correlations) if correlations else 0.0
        
        return {
            'coevolution_strength': coevolution_strength,
            'trait_correlations': correlations
        }
    
    def get_evolutionary_summary(self, current_time: int) -> Dict[str, Any]:
        """Get comprehensive evolutionary summary"""
        summary = {
            'total_individuals': len(self.individual_records),
            'species_count': len(self.species_records),
            'phylogenetic_diversity': self.phylogenetic_tree.calculate_species_diversity(current_time),
            'species_statistics': {},
            'evolutionary_events': self.detect_evolutionary_events(current_time),
            'selection_pressures': {},
            'heritabilities': {},
            'fitness_landscapes': {}
        }
        
        # Species-specific statistics
        for species in self.species_records.keys():
            living_individuals = [
                r for r in self.individual_records.values()
                if r.species == species and (r.death_time is None or r.death_time > current_time)
            ]
            
            summary['species_statistics'][species] = {
                'population': len(living_individuals),
                'avg_generation': np.mean([r.generation for r in living_individuals]) if living_individuals else 0,
                'lineage_stats': self.phylogenetic_tree.get_lineage_statistics(species)
            }
            
            # Selection pressures
            summary['selection_pressures'][species] = {
                trait: self.calculate_selection_pressure(species, trait, current_time)
                for trait in ['intelligence', 'sociability', 'aggression']
            }
            
            # Heritabilities
            summary['heritabilities'][species] = {
                trait: self.calculate_heritability(species, trait)
                for trait in ['intelligence', 'sociability', 'aggression']
            }
            
            # Fitness landscape info
            if species in self.fitness_landscapes:
                landscape = self.fitness_landscapes[species]
                peaks = landscape.find_adaptive_peaks()
                summary['fitness_landscapes'][species] = {
                    'adaptive_peaks': len(peaks),
                    'ruggedness': landscape.calculate_landscape_ruggedness(),
                    'peak_details': peaks[:3]  # Top 3 peaks
                }
        
        return summary