"""
Social network dynamics and analysis.

This module implements dynamic social relationship networks, community detection,
leadership emergence, and social influence patterns.
"""

import numpy as np
import networkx as nx
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional


class SocialRelationship:
    """Individual social relationship with history and dynamics"""
    
    def __init__(self, individual_id: int):
        self.individual_id = individual_id
        self.strength = 0.5  # Relationship strength (0-1)
        self.trust = 0.5     # Trust level (0-1)
        self.interaction_history = deque(maxlen=50)
        self.shared_experiences = 0
        self.last_interaction_time = 0
        self.relationship_type = 'neutral'  # 'friend', 'rival', 'mate', 'kin'
        self.cooperation_count = 0
        self.conflict_count = 0
        
    def update_from_interaction(self, interaction_type: str, success: bool, 
                              time_step: int, context: str = None):
        """Update relationship based on an interaction"""
        self.last_interaction_time = time_step
        self.interaction_history.append({
            'type': interaction_type,
            'success': success,
            'time': time_step,
            'context': context
        })
        
        # Update trust and strength based on interaction outcome
        if success:
            if interaction_type == 'cooperation':
                self.trust = min(1.0, self.trust + 0.15)
                self.strength = min(1.0, self.strength + 0.1)
                self.cooperation_count += 1
            elif interaction_type == 'communication':
                self.trust = min(1.0, self.trust + 0.1)
                self.strength = min(1.0, self.strength + 0.05)
            elif interaction_type == 'help':
                self.trust = min(1.0, self.trust + 0.2)
                self.strength = min(1.0, self.strength + 0.15)
        else:
            if interaction_type == 'conflict':
                self.trust = max(0.0, self.trust - 0.1)
                self.strength = max(0.0, self.strength - 0.05)
                self.conflict_count += 1
            else:
                self.trust = max(0.0, self.trust - 0.05)
        
        # Update relationship type based on history
        self._update_relationship_type()
        
        # Track shared experiences
        self.shared_experiences += 1
    
    def _update_relationship_type(self):
        """Update relationship type based on interaction history"""
        if self.trust > 0.8 and self.strength > 0.8:
            if self.cooperation_count > self.conflict_count * 3:
                self.relationship_type = 'friend'
        elif self.trust < 0.3 and self.conflict_count > self.cooperation_count:
            self.relationship_type = 'rival'
        elif self.strength > 0.9 and self.cooperation_count > 10:
            self.relationship_type = 'mate'
        else:
            self.relationship_type = 'neutral'
    
    def decay(self, decay_rate: float = 0.01):
        """Natural decay of relationship strength over time"""
        self.strength = max(0.0, self.strength - decay_rate)
        self.trust = max(0.0, self.trust - decay_rate * 0.5)
    
    def get_relationship_value(self) -> float:
        """Get overall relationship value combining strength and trust"""
        return (self.strength + self.trust) / 2


class SocialNetwork:
    """Dynamic social network with emergent hierarchy and communities"""
    
    def __init__(self):
        self.relationships: Dict[int, Dict[int, SocialRelationship]] = defaultdict(dict)
        self.interaction_graph = nx.DiGraph()
        self.communities: List[Set[int]] = []
        self.leaders: Set[int] = set()
        self.influence_network: Dict[int, float] = defaultdict(float)
        self.network_history = []
        self.community_stability = defaultdict(int)
        
    def add_individual(self, individual_id: int):
        """Add individual to social network"""
        if individual_id not in self.relationships:
            self.relationships[individual_id] = {}
            self.interaction_graph.add_node(individual_id)
    
    def remove_individual(self, individual_id: int):
        """Remove individual from social network"""
        if individual_id in self.relationships:
            # Remove all relationships involving this individual
            del self.relationships[individual_id]
            
            # Remove from others' relationships
            for other_relationships in self.relationships.values():
                if individual_id in other_relationships:
                    del other_relationships[individual_id]
            
            # Remove from graph
            if self.interaction_graph.has_node(individual_id):
                self.interaction_graph.remove_node(individual_id)
            
            # Remove from leaders
            self.leaders.discard(individual_id)
    
    def update_relationship(self, id1: int, id2: int, interaction_type: str, 
                          success: bool, time_step: int, context: str = None):
        """Update relationship based on interaction"""
        # Ensure both individuals exist in network
        self.add_individual(id1)
        self.add_individual(id2)
        
        # Update relationship from id1's perspective
        if id2 not in self.relationships[id1]:
            self.relationships[id1][id2] = SocialRelationship(id2)
        
        rel = self.relationships[id1][id2]
        rel.update_from_interaction(interaction_type, success, time_step, context)
        
        # Update relationship from id2's perspective (symmetric for most interactions)
        if interaction_type in ['cooperation', 'communication', 'help']:
            if id1 not in self.relationships[id2]:
                self.relationships[id2][id1] = SocialRelationship(id1)
            
            rel2 = self.relationships[id2][id1]
            rel2.update_from_interaction(interaction_type, success, time_step, context)
        
        # Update interaction graph
        weight = rel.get_relationship_value()
        if self.interaction_graph.has_edge(id1, id2):
            self.interaction_graph[id1][id2]['weight'] = weight
            self.interaction_graph[id1][id2]['interactions'] += 1
        else:
            self.interaction_graph.add_edge(id1, id2, weight=weight, interactions=1)
        
        # For symmetric relationships, update reverse edge
        if interaction_type in ['cooperation', 'communication', 'help']:
            rel2 = self.relationships[id2][id1]
            weight2 = rel2.get_relationship_value()
            if self.interaction_graph.has_edge(id2, id1):
                self.interaction_graph[id2][id1]['weight'] = weight2
                self.interaction_graph[id2][id1]['interactions'] += 1
            else:
                self.interaction_graph.add_edge(id2, id1, weight=weight2, interactions=1)
    
    def get_relationship_strength(self, id1: int, id2: int) -> float:
        """Get relationship strength between two individuals"""
        if id1 in self.relationships and id2 in self.relationships[id1]:
            return self.relationships[id1][id2].strength
        return 0.0
    
    def get_relationship_trust(self, id1: int, id2: int) -> float:
        """Get trust level between two individuals"""
        if id1 in self.relationships and id2 in self.relationships[id1]:
            return self.relationships[id1][id2].trust
        return 0.0
    
    def get_friends(self, individual_id: int, min_strength: float = 0.6) -> List[int]:
        """Get list of friends for an individual"""
        friends = []
        if individual_id in self.relationships:
            for other_id, rel in self.relationships[individual_id].items():
                if rel.relationship_type == 'friend' or rel.strength >= min_strength:
                    friends.append(other_id)
        return friends
    
    def get_rivals(self, individual_id: int) -> List[int]:
        """Get list of rivals for an individual"""
        rivals = []
        if individual_id in self.relationships:
            for other_id, rel in self.relationships[individual_id].items():
                if rel.relationship_type == 'rival':
                    rivals.append(other_id)
        return rivals
    
    def decay_relationships(self, time_step: int, decay_rate: float = 0.02):
        """Decay unused relationships over time"""
        relationships_to_remove = []
        
        for individual_id, relationships in self.relationships.items():
            for other_id, rel in list(relationships.items()):
                time_since_interaction = time_step - rel.last_interaction_time
                
                if time_since_interaction > 100:  # Haven't interacted recently
                    rel.decay(decay_rate)
                    
                    # Remove very weak relationships
                    if rel.strength < 0.1 and rel.trust < 0.1:
                        relationships_to_remove.append((individual_id, other_id))
        
        # Remove weak relationships
        for id1, id2 in relationships_to_remove:
            if id1 in self.relationships and id2 in self.relationships[id1]:
                del self.relationships[id1][id2]
            
            if self.interaction_graph.has_edge(id1, id2):
                self.interaction_graph.remove_edge(id1, id2)
    
    def detect_communities(self, min_community_size: int = 3):
        """Detect communities using network analysis"""
        if len(self.interaction_graph.nodes()) < min_community_size:
            self.communities = []
            return
        
        try:
            # Convert to undirected graph for community detection
            undirected_graph = self.interaction_graph.to_undirected()
            
            # Remove weak edges
            edges_to_remove = []
            for u, v, data in undirected_graph.edges(data=True):
                if data.get('weight', 0) < 0.3:
                    edges_to_remove.append((u, v))
            
            undirected_graph.remove_edges_from(edges_to_remove)
            
            # Detect communities
            communities = list(nx.community.greedy_modularity_communities(undirected_graph))
            
            # Filter communities by minimum size
            self.communities = [community for community in communities 
                              if len(community) >= min_community_size]
            
            # Track community stability
            for community in self.communities:
                community_id = tuple(sorted(community))
                self.community_stability[community_id] += 1
                
        except Exception:
            self.communities = []
    
    def identify_leaders(self, min_influence_threshold: float = 0.6):
        """Identify influential individuals in the network"""
        self.leaders.clear()
        self.influence_network.clear()
        
        if len(self.interaction_graph.nodes()) == 0:
            return
        
        try:
            # Calculate various centrality measures
            degree_centrality = nx.degree_centrality(self.interaction_graph)
            betweenness_centrality = nx.betweenness_centrality(self.interaction_graph)
            
            # Calculate influence scores
            for node in self.interaction_graph.nodes():
                # Combine different centrality measures
                degree_score = degree_centrality.get(node, 0)
                betweenness_score = betweenness_centrality.get(node, 0)
                
                # Weight by relationship quality
                relationship_quality = 0
                if node in self.relationships:
                    relationships = self.relationships[node].values()
                    if relationships:
                        relationship_quality = np.mean([rel.get_relationship_value() 
                                                      for rel in relationships])
                
                # Calculate composite influence score
                influence_score = (degree_score * 0.4 + 
                                 betweenness_score * 0.4 + 
                                 relationship_quality * 0.2)
                
                self.influence_network[node] = influence_score
                
                # Identify leaders
                if influence_score > min_influence_threshold:
                    self.leaders.add(node)
                    
        except Exception:
            pass
    
    def get_network_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive network metrics"""
        if not self.interaction_graph.nodes():
            return {'density': 0, 'clustering': 0, 'diameter': 0, 'components': 0}
        
        try:
            # Basic metrics
            density = nx.density(self.interaction_graph)
            
            # Clustering coefficient
            undirected = self.interaction_graph.to_undirected()
            clustering = nx.average_clustering(undirected)
            
            # Diameter (for largest connected component)
            if nx.is_connected(undirected):
                diameter = nx.diameter(undirected)
            else:
                largest_cc = max(nx.connected_components(undirected), key=len)
                diameter = nx.diameter(undirected.subgraph(largest_cc))
            
            # Number of connected components
            components = nx.number_connected_components(undirected)
            
            return {
                'density': density,
                'clustering': clustering,
                'diameter': diameter,
                'components': components,
                'nodes': len(self.interaction_graph.nodes()),
                'edges': len(self.interaction_graph.edges()),
                'communities': len(self.communities),
                'leaders': len(self.leaders)
            }
            
        except Exception:
            return {'density': 0, 'clustering': 0, 'diameter': 0, 'components': 0}
    
    def get_individual_network_position(self, individual_id: int) -> Dict[str, float]:
        """Get network position metrics for an individual"""
        if individual_id not in self.interaction_graph.nodes():
            return {'degree': 0, 'betweenness': 0, 'closeness': 0, 'influence': 0}
        
        try:
            degree = self.interaction_graph.degree(individual_id)
            
            betweenness = nx.betweenness_centrality(self.interaction_graph).get(individual_id, 0)
            closeness = nx.closeness_centrality(self.interaction_graph).get(individual_id, 0)
            influence = self.influence_network.get(individual_id, 0)
            
            return {
                'degree': degree,
                'betweenness': betweenness,
                'closeness': closeness,
                'influence': influence
            }
            
        except Exception:
            return {'degree': 0, 'betweenness': 0, 'closeness': 0, 'influence': 0}
    
    def find_community_for_individual(self, individual_id: int) -> Optional[Set[int]]:
        """Find which community an individual belongs to"""
        for community in self.communities:
            if individual_id in community:
                return community
        return None
    
    def get_community_leaders(self, community: Set[int]) -> List[int]:
        """Get leaders within a specific community"""
        return [leader for leader in self.leaders if leader in community]
    
    def calculate_social_cohesion(self) -> float:
        """Calculate overall social cohesion of the network"""
        if not self.relationships:
            return 0.0
        
        total_relationships = 0
        positive_relationships = 0
        
        for individual_relationships in self.relationships.values():
            for rel in individual_relationships.values():
                total_relationships += 1
                if rel.get_relationship_value() > 0.5:
                    positive_relationships += 1
        
        return positive_relationships / total_relationships if total_relationships > 0 else 0.0
    
    def record_network_state(self, time_step: int):
        """Record current network state for historical analysis"""
        metrics = self.get_network_metrics()
        metrics['time_step'] = time_step
        metrics['social_cohesion'] = self.calculate_social_cohesion()
        self.network_history.append(metrics)
    
    def detect_network_phase_transitions(self, window_size: int = 50) -> List[Dict]:
        """Detect sudden changes in network structure"""
        if len(self.network_history) < window_size * 2:
            return []
        
        transitions = []
        recent_history = self.network_history[-window_size*2:]
        
        # Split into two windows
        early_window = recent_history[:window_size]
        late_window = recent_history[window_size:]
        
        # Compare metrics between windows
        for metric in ['density', 'clustering', 'communities', 'social_cohesion']:
            early_values = [entry[metric] for entry in early_window]
            late_values = [entry[metric] for entry in late_window]
            
            early_mean = np.mean(early_values)
            late_mean = np.mean(late_values)
            
            # Detect significant changes
            if early_mean > 0:
                change_ratio = abs(late_mean - early_mean) / early_mean
                if change_ratio > 0.3:  # 30% change threshold
                    transitions.append({
                        'metric': metric,
                        'change_ratio': change_ratio,
                        'direction': 'increase' if late_mean > early_mean else 'decrease',
                        'time_step': recent_history[-1]['time_step']
                    })
        
        return transitions
