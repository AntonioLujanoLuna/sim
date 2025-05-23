"""
Memory management system for the simulation.

This module provides utilities for managing memory usage, cleaning up old data,
and optimizing storage to prevent memory leaks and excessive growth.
"""

import gc
import psutil
import time
from collections import deque
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger('EmergentEcosystem.MemoryManager')


class MemoryManager:
    """Centralized memory management for the simulation"""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.memory_warnings = []
        self.cleanup_history = []
        self.last_cleanup_time = 0
        self.cleanup_interval = 50  # Cleanup every 50 simulation steps
        
        # Memory limits for different components
        self.limits = {
            'individual_memory': 200,  # Max memories per individual
            'social_memory': 100,      # Max social memories per individual
            'cultural_knowledge': 50,  # Max cultural knowledge items
            'interaction_history': 100, # Max interaction history
            'trail_length': 30,        # Max trail length
            'data_history': 500,       # Max simulation data history
            'network_history': 100,    # Max network state history
            'environmental_memory': 1000  # Max environmental memories
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage is approaching limits"""
        memory_usage = self.get_memory_usage()
        
        if memory_usage['rss_mb'] > self.max_memory_mb * 0.8:
            warning = {
                'timestamp': time.time(),
                'memory_mb': memory_usage['rss_mb'],
                'percent': memory_usage['percent'],
                'message': 'High memory usage detected'
            }
            self.memory_warnings.append(warning)
            
            # Keep only recent warnings
            if len(self.memory_warnings) > 10:
                self.memory_warnings.pop(0)
            
            logger.warning(f"Memory pressure detected: {memory_usage['rss_mb']:.1f}MB ({memory_usage['percent']:.1f}%)")
            return True
        
        return False
    
    def cleanup_individual_memory(self, individual) -> int:
        """Clean up memory for a single individual"""
        cleaned_items = 0
        
        try:
            # Limit spatial memory
            if hasattr(individual, 'spatial_memory') and len(individual.spatial_memory) > self.limits['individual_memory']:
                excess = len(individual.spatial_memory) - self.limits['individual_memory']
                for _ in range(excess):
                    individual.spatial_memory.popleft()
                cleaned_items += excess
            
            # Limit social memory
            if hasattr(individual, 'social_memory'):
                if len(individual.social_memory) > self.limits['social_memory']:
                    # Keep only the most recent and important social memories
                    sorted_memories = sorted(
                        individual.social_memory.items(),
                        key=lambda x: len(x[1]) if isinstance(x[1], list) else 1,
                        reverse=True
                    )
                    
                    # Keep top memories
                    individual.social_memory = dict(sorted_memories[:self.limits['social_memory']])
                    cleaned_items += len(sorted_memories) - self.limits['social_memory']
            
            # Limit cultural knowledge
            if hasattr(individual, 'cultural_knowledge') and len(individual.cultural_knowledge) > self.limits['cultural_knowledge']:
                # Keep most valuable cultural knowledge
                sorted_knowledge = sorted(
                    individual.cultural_knowledge.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                excess = len(individual.cultural_knowledge) - self.limits['cultural_knowledge']
                individual.cultural_knowledge = dict(sorted_knowledge[:self.limits['cultural_knowledge']])
                cleaned_items += excess
            
            # Limit interaction history
            if hasattr(individual, 'interaction_history') and len(individual.interaction_history) > self.limits['interaction_history']:
                excess = len(individual.interaction_history) - self.limits['interaction_history']
                for _ in range(excess):
                    individual.interaction_history.popleft()
                cleaned_items += excess
            
            # Limit trail length
            if hasattr(individual, 'trail') and len(individual.trail) > self.limits['trail_length']:
                excess = len(individual.trail) - self.limits['trail_length']
                for _ in range(excess):
                    individual.trail.popleft()
                cleaned_items += excess
            
            # Clean up old environmental memories
            if hasattr(individual, 'environmental_memory'):
                current_time = getattr(individual, 'age', 0)
                old_threshold = current_time - 100  # Remove memories older than 100 time steps
                
                old_locations = [
                    loc for loc, data in individual.environmental_memory.items()
                    if isinstance(data, dict) and data.get('timestamp', 0) < old_threshold
                ]
                
                for loc in old_locations:
                    del individual.environmental_memory[loc]
                    cleaned_items += 1
            
        except Exception as e:
            logger.error(f"Error cleaning individual memory: {e}")
        
        return cleaned_items
    
    def cleanup_simulation_memory(self, simulation) -> int:
        """Clean up memory for the entire simulation"""
        cleaned_items = 0
        
        try:
            # Clean individual memories
            for individual in simulation.individuals:
                cleaned_items += self.cleanup_individual_memory(individual)
            
            # Limit simulation data history
            if hasattr(simulation, 'data_history') and len(simulation.data_history) > self.limits['data_history']:
                excess = len(simulation.data_history) - self.limits['data_history']
                for _ in range(excess):
                    simulation.data_history.popleft()
                cleaned_items += excess
            
            # Limit information history
            if hasattr(simulation, 'information_history') and len(simulation.information_history) > self.limits['data_history']:
                excess = len(simulation.information_history) - self.limits['data_history']
                for _ in range(excess):
                    simulation.information_history.popleft()
                cleaned_items += excess
            
            # Clean up social network history
            if hasattr(simulation.social_network, 'network_history') and len(simulation.social_network.network_history) > self.limits['network_history']:
                excess = len(simulation.social_network.network_history) - self.limits['network_history']
                simulation.social_network.network_history = simulation.social_network.network_history[-self.limits['network_history']:]
                cleaned_items += excess
            
            # Clean up environmental memory
            if hasattr(simulation.environment, 'memory_data'):
                if len(simulation.environment.memory_data) > self.limits['environmental_memory']:
                    # Keep most recent environmental memories
                    sorted_memories = sorted(
                        simulation.environment.memory_data.items(),
                        key=lambda x: x[1].get('last_update', 0) if isinstance(x[1], dict) else 0,
                        reverse=True
                    )
                    
                    simulation.environment.memory_data = dict(sorted_memories[:self.limits['environmental_memory']])
                    cleaned_items += len(sorted_memories) - self.limits['environmental_memory']
            
            # Clean up weak social network connections
            if hasattr(simulation.social_network, 'relationships'):
                weak_connections = []
                for id1, relationships in simulation.social_network.relationships.items():
                    for id2, rel in list(relationships.items()):
                        if rel.strength < 0.1 and rel.trust < 0.1:
                            weak_connections.append((id1, id2))
                
                for id1, id2 in weak_connections:
                    if id1 in simulation.social_network.relationships and id2 in simulation.social_network.relationships[id1]:
                        del simulation.social_network.relationships[id1][id2]
                        cleaned_items += 1
                        
                        # Also remove from graph
                        if simulation.social_network.interaction_graph.has_edge(id1, id2):
                            simulation.social_network.interaction_graph.remove_edge(id1, id2)
            
        except Exception as e:
            logger.error(f"Error cleaning simulation memory: {e}")
        
        return cleaned_items
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics"""
        before_objects = len(gc.get_objects())
        
        # Run garbage collection
        collected = {
            'gen0': gc.collect(0),
            'gen1': gc.collect(1),
            'gen2': gc.collect(2)
        }
        
        after_objects = len(gc.get_objects())
        collected['objects_freed'] = before_objects - after_objects
        
        return collected
    
    def optimize_memory(self, simulation, force: bool = False) -> Dict[str, Any]:
        """Comprehensive memory optimization"""
        start_time = time.time()
        memory_before = self.get_memory_usage()
        
        # Check if cleanup is needed
        if not force and time.time() - self.last_cleanup_time < self.cleanup_interval:
            return {'skipped': True, 'reason': 'Too soon since last cleanup'}
        
        # Perform cleanup
        cleaned_items = self.cleanup_simulation_memory(simulation)
        
        # Force garbage collection if memory pressure is high
        gc_stats = None
        if self.check_memory_pressure() or force:
            gc_stats = self.force_garbage_collection()
        
        # Update cleanup history
        memory_after = self.get_memory_usage()
        cleanup_record = {
            'timestamp': time.time(),
            'memory_before_mb': memory_before['rss_mb'],
            'memory_after_mb': memory_after['rss_mb'],
            'memory_freed_mb': memory_before['rss_mb'] - memory_after['rss_mb'],
            'items_cleaned': cleaned_items,
            'gc_stats': gc_stats,
            'duration_seconds': time.time() - start_time
        }
        
        self.cleanup_history.append(cleanup_record)
        
        # Keep only recent cleanup history
        if len(self.cleanup_history) > 20:
            self.cleanup_history.pop(0)
        
        self.last_cleanup_time = time.time()
        
        logger.info(f"Memory cleanup completed: {cleaned_items} items cleaned, "
                   f"{cleanup_record['memory_freed_mb']:.1f}MB freed")
        
        return cleanup_record
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report"""
        current_usage = self.get_memory_usage()
        
        return {
            'current_usage': current_usage,
            'limits': self.limits,
            'warnings': self.memory_warnings[-5:],  # Last 5 warnings
            'cleanup_history': self.cleanup_history[-5:],  # Last 5 cleanups
            'memory_pressure': self.check_memory_pressure(),
            'recommendations': self._get_memory_recommendations(current_usage)
        }
    
    def _get_memory_recommendations(self, usage: Dict[str, float]) -> List[str]:
        """Get memory optimization recommendations"""
        recommendations = []
        
        if usage['rss_mb'] > self.max_memory_mb * 0.7:
            recommendations.append("Consider reducing population size or simulation complexity")
        
        if usage['percent'] > 80:
            recommendations.append("High memory usage detected - enable more frequent cleanup")
        
        if len(self.memory_warnings) > 5:
            recommendations.append("Frequent memory warnings - consider increasing memory limits")
        
        return recommendations
    
    def set_memory_limits(self, **limits):
        """Update memory limits for different components"""
        for key, value in limits.items():
            if key in self.limits:
                self.limits[key] = value
                logger.info(f"Updated memory limit for {key}: {value}")
            else:
                logger.warning(f"Unknown memory limit key: {key}")


# Global memory manager instance
memory_manager = MemoryManager() 