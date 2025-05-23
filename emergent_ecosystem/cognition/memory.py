"""
Memory architectures and systems.

This module implements multiple memory types including spatial, episodic, semantic,
and working memory with realistic forgetting and interference patterns.
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import math


@dataclass
class MemoryItem:
    """Individual memory item with decay and retrieval properties"""
    content: Any
    encoding_time: int
    retrieval_count: int = 0
    strength: float = 1.0
    emotional_weight: float = 0.0
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
    
    def decay(self, current_time: int, decay_rate: float = 0.001):
        """Apply time-based memory decay"""
        time_passed = current_time - self.encoding_time
        self.strength *= np.exp(-decay_rate * time_passed)
        
        # Emotional memories decay slower
        if self.emotional_weight > 0.5:
            self.strength *= 1.1  # Boost emotional memories
    
    def strengthen(self, amount: float = 0.1):
        """Strengthen memory through retrieval or rehearsal"""
        self.retrieval_count += 1
        self.strength = min(1.0, self.strength + amount)


class SpatialMemory:
    """Spatial memory for locations and navigation"""
    
    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self.locations = {}  # (x, y) -> MemoryItem
        self.landmarks = {}  # landmark_id -> (x, y, properties)
        self.paths = defaultdict(list)  # start_location -> [(end_location, path_quality)]
        self.current_time = 0
        
    def encode_location(self, x: float, y: float, properties: Dict[str, Any], 
                       emotional_weight: float = 0.0):
        """Encode a spatial location with its properties"""
        location_key = (round(x/10)*10, round(y/10)*10)  # Grid-based storage
        
        memory_item = MemoryItem(
            content={'properties': properties, 'exact_position': (x, y)},
            encoding_time=self.current_time,
            emotional_weight=emotional_weight,
            context={'location_type': properties.get('type', 'unknown')}
        )
        
        self.locations[location_key] = memory_item
        
        # Maintain capacity
        if len(self.locations) > self.capacity:
            self._forget_weakest_location()
    
    def recall_location(self, x: float, y: float, radius: float = 50) -> List[Tuple[Tuple[float, float], Dict]]:
        """Recall locations within radius of given position"""
        location_key = (round(x/10)*10, round(y/10)*10)
        recalled_locations = []
        
        for (lx, ly), memory_item in self.locations.items():
            distance = np.sqrt((lx - x)**2 + (ly - y)**2)
            if distance <= radius:
                # Strengthen memory through retrieval
                memory_item.strengthen(0.05)
                
                # Recall probability based on memory strength
                if random.random() < memory_item.strength:
                    exact_pos = memory_item.content['exact_position']
                    recalled_locations.append((exact_pos, memory_item.content['properties']))
        
        return recalled_locations
    
    def find_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Find remembered path between two locations"""
        start_key = (round(start[0]/10)*10, round(start[1]/10)*10)
        goal_key = (round(goal[0]/10)*10, round(goal[1]/10)*10)
        
        # Simple pathfinding using remembered locations
        if start_key in self.paths and goal_key in [p[0] for p in self.paths[start_key]]:
            # Direct path exists
            return [start, goal]
        
        # Multi-hop pathfinding through known locations
        visited = set()
        queue = [(start_key, [start])]
        
        while queue:
            current, path = queue.pop(0)
            if current == goal_key:
                return path + [goal]
            
            if current in visited:
                continue
            visited.add(current)
            
            # Add neighboring remembered locations
            for location_key in self.locations.keys():
                if location_key not in visited:
                    lx, ly = location_key
                    exact_pos = self.locations[location_key].content['exact_position']
                    queue.append((location_key, path + [exact_pos]))
        
        return [start, goal]  # Fallback direct path
    
    def add_landmark(self, landmark_id: str, x: float, y: float, properties: Dict[str, Any]):
        """Add a memorable landmark"""
        self.landmarks[landmark_id] = (x, y, properties)
        # Landmarks are encoded as high-strength memories
        self.encode_location(x, y, properties, emotional_weight=0.8)
    
    def update(self, current_time: int):
        """Update spatial memory with decay"""
        self.current_time = current_time
        
        # Apply decay to all location memories
        for memory_item in self.locations.values():
            memory_item.decay(current_time)
    
    def _forget_weakest_location(self):
        """Remove the weakest spatial memory"""
        if self.locations:
            weakest_key = min(self.locations.keys(), 
                            key=lambda k: self.locations[k].strength)
            del self.locations[weakest_key]


class EpisodicMemory:
    """Episodic memory for events and experiences"""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        self.episode_index = {}  # event_type -> [episode_indices]
        self.current_time = 0
        
    def encode_episode(self, event_type: str, participants: List[int], 
                      outcome: bool, context: Dict[str, Any], 
                      emotional_weight: float = 0.0):
        """Encode an episodic memory"""
        episode = MemoryItem(
            content={
                'event_type': event_type,
                'participants': participants,
                'outcome': outcome,
                'context': context
            },
            encoding_time=self.current_time,
            emotional_weight=emotional_weight
        )
        
        self.episodes.append(episode)
        
        # Index by event type
        if event_type not in self.episode_index:
            self.episode_index[event_type] = []
        self.episode_index[event_type].append(len(self.episodes) - 1)
    
    def recall_episodes(self, event_type: str = None, participant: int = None, 
                       recent_only: bool = False) -> List[MemoryItem]:
        """Recall episodes matching criteria"""
        recalled_episodes = []
        
        episodes_to_check = self.episodes
        if recent_only:
            episodes_to_check = list(self.episodes)[-20:]  # Last 20 episodes
        
        for episode in episodes_to_check:
            # Check if episode matches criteria
            matches = True
            
            if event_type and episode.content['event_type'] != event_type:
                matches = False
            
            if participant and participant not in episode.content['participants']:
                matches = False
            
            # Recall probability based on memory strength
            if matches and random.random() < episode.strength:
                episode.strengthen(0.05)
                recalled_episodes.append(episode)
        
        return recalled_episodes
    
    def get_success_rate(self, event_type: str, participant: int = None) -> float:
        """Calculate success rate for specific event type"""
        relevant_episodes = self.recall_episodes(event_type, participant)
        
        if not relevant_episodes:
            return 0.5  # Default neutral probability
        
        successes = sum(1 for ep in relevant_episodes if ep.content['outcome'])
        return successes / len(relevant_episodes)
    
    def update(self, current_time: int):
        """Update episodic memory with decay"""
        self.current_time = current_time
        
        for episode in self.episodes:
            episode.decay(current_time)


class SemanticMemory:
    """Semantic memory for factual and conceptual knowledge"""
    
    def __init__(self):
        self.concepts = {}  # concept_name -> MemoryItem
        self.associations = defaultdict(list)  # concept -> [related_concepts]
        self.current_time = 0
        
    def learn_concept(self, concept_name: str, properties: Dict[str, Any], 
                     confidence: float = 0.5):
        """Learn a new concept or update existing one"""
        if concept_name in self.concepts:
            # Update existing concept
            existing = self.concepts[concept_name]
            existing.strengthen(0.1)
            existing.content.update(properties)
        else:
            # Create new concept
            self.concepts[concept_name] = MemoryItem(
                content=properties,
                encoding_time=self.current_time,
                strength=confidence
            )
    
    def recall_concept(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """Recall information about a concept"""
        if concept_name in self.concepts:
            concept = self.concepts[concept_name]
            if random.random() < concept.strength:
                concept.strengthen(0.02)
                return concept.content
        return None
    
    def associate_concepts(self, concept1: str, concept2: str, strength: float = 0.5):
        """Create association between concepts"""
        if concept1 in self.concepts and concept2 in self.concepts:
            if concept2 not in self.associations[concept1]:
                self.associations[concept1].append((concept2, strength))
            if concept1 not in self.associations[concept2]:
                self.associations[concept2].append((concept1, strength))
    
    def get_related_concepts(self, concept_name: str, threshold: float = 0.3) -> List[str]:
        """Get concepts related to the given concept"""
        related = []
        if concept_name in self.associations:
            for related_concept, strength in self.associations[concept_name]:
                if strength > threshold and random.random() < strength:
                    related.append(related_concept)
        return related
    
    def update(self, current_time: int):
        """Update semantic memory with decay"""
        self.current_time = current_time
        
        for concept in self.concepts.values():
            concept.decay(current_time, decay_rate=0.0005)  # Slower decay for semantic memory


class WorkingMemory:
    """Working memory for temporary information processing"""
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity  # Miller's magic number 7±2
        self.items = deque(maxlen=capacity)
        self.attention_focus = None
        self.rehearsal_buffer = []
        
    def add_item(self, item: Any, priority: float = 0.5):
        """Add item to working memory"""
        memory_item = MemoryItem(
            content=item,
            encoding_time=0,
            strength=priority
        )
        
        # If at capacity, remove least important item
        if len(self.items) >= self.capacity:
            self._remove_weakest_item()
        
        self.items.append(memory_item)
    
    def get_items(self, threshold: float = 0.1) -> List[Any]:
        """Get all items above strength threshold"""
        return [item.content for item in self.items if item.strength > threshold]
    
    def focus_attention(self, item: Any):
        """Focus attention on specific item"""
        for memory_item in self.items:
            if memory_item.content == item:
                memory_item.strengthen(0.2)
                self.attention_focus = memory_item
                break
    
    def rehearse(self):
        """Rehearse items to maintain them in working memory"""
        for item in self.items:
            if random.random() < 0.7:  # 70% chance of successful rehearsal
                item.strengthen(0.1)
    
    def decay_items(self):
        """Apply rapid decay to working memory items"""
        for item in self.items:
            item.strength *= 0.9  # Rapid decay
        
        # Remove very weak items
        self.items = deque([item for item in self.items if item.strength > 0.05], 
                          maxlen=self.capacity)
    
    def _remove_weakest_item(self):
        """Remove the weakest item from working memory"""
        if self.items:
            weakest_idx = min(range(len(self.items)), 
                            key=lambda i: self.items[i].strength)
            del self.items[weakest_idx]


class IntegratedMemorySystem:
    """Integrated memory system combining all memory types"""
    
    def __init__(self, spatial_capacity: int = 200, episodic_capacity: int = 100, 
                 working_capacity: int = 7):
        self.spatial = SpatialMemory(spatial_capacity)
        self.episodic = EpisodicMemory(episodic_capacity)
        self.semantic = SemanticMemory()
        self.working = WorkingMemory(working_capacity)
        self.current_time = 0
        
    def update(self, current_time: int):
        """Update all memory systems"""
        self.current_time = current_time
        self.spatial.update(current_time)
        self.episodic.update(current_time)
        self.semantic.update(current_time)
        self.working.decay_items()
    
    def consolidate_memories(self):
        """Transfer information between memory systems"""
        # Transfer important working memory items to long-term memory
        important_items = [item for item in self.working.items if item.strength > 0.7]
        
        for item in important_items:
            content = item.content
            if isinstance(content, dict):
                if 'location' in content:
                    # Consolidate to spatial memory
                    loc = content['location']
                    props = content.get('properties', {})
                    self.spatial.encode_location(loc[0], loc[1], props, item.emotional_weight)
                
                elif 'event_type' in content:
                    # Consolidate to episodic memory
                    self.episodic.encode_episode(
                        content['event_type'],
                        content.get('participants', []),
                        content.get('outcome', False),
                        content.get('context', {}),
                        item.emotional_weight
                    )
                
                elif 'concept' in content:
                    # Consolidate to semantic memory
                    self.semantic.learn_concept(
                        content['concept'],
                        content.get('properties', {}),
                        item.strength
                    )
    
    def retrieve_relevant_memories(self, query: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Retrieve relevant memories across all systems"""
        results = {
            'spatial': [],
            'episodic': [],
            'semantic': [],
            'working': []
        }
        
        # Spatial retrieval
        if 'location' in query:
            x, y = query['location']
            radius = query.get('radius', 50)
            results['spatial'] = self.spatial.recall_location(x, y, radius)
        
        # Episodic retrieval
        if 'event_type' in query:
            results['episodic'] = self.episodic.recall_episodes(
                query['event_type'],
                query.get('participant'),
                query.get('recent_only', False)
            )
        
        # Semantic retrieval
        if 'concept' in query:
            concept_info = self.semantic.recall_concept(query['concept'])
            if concept_info:
                results['semantic'] = [concept_info]
                # Add related concepts
                related = self.semantic.get_related_concepts(query['concept'])
                for rel_concept in related:
                    rel_info = self.semantic.recall_concept(rel_concept)
                    if rel_info:
                        results['semantic'].append(rel_info)
        
        # Working memory retrieval
        results['working'] = self.working.get_items()
        
        return results
    
    def get_memory_metrics(self) -> Dict[str, Any]:
        """Get comprehensive memory system metrics"""
        return {
            'spatial_locations': len(self.spatial.locations),
            'spatial_landmarks': len(self.spatial.landmarks),
            'episodic_episodes': len(self.episodic.episodes),
            'semantic_concepts': len(self.semantic.concepts),
            'working_items': len(self.working.items),
            'total_memories': (len(self.spatial.locations) + 
                             len(self.episodic.episodes) + 
                             len(self.semantic.concepts) + 
                             len(self.working.items))
        }