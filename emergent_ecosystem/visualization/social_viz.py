"""
Social network and relationship visualization.

This module provides specialized visualization tools for social networks,
communication patterns, community structures, and relationship dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
import random
import colorsys
from collections import defaultdict
import math


class SocialNetworkVisualizer:
    """Specialized visualizer for social networks"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.node_positions = {}
        self.layout_cache = {}
        self.color_schemes = {
            'species': {
                'predator': '#FF4444',
                'herbivore': '#44FF44', 
                'scavenger': '#FFFF44',
                'mystic': '#AA44FF'
            },
            'relationship': {
                'friend': '#00FF00',
                'rival': '#FF0000',
                'mate': '#FF69B4',
                'neutral': '#CCCCCC',
                'leader': '#FFD700'
            }
        }
    
    def create_network_graph(self, individuals: List, social_network) -> nx.Graph:
        """Create NetworkX graph from social network data"""
        G = nx.Graph()
        
        # Add nodes
        for individual in individuals:
            G.add_node(individual.id, 
                      species=individual.species_name,
                      intelligence=individual.intelligence,
                      sociability=individual.sociability,
                      energy=individual.energy,
                      position=(individual.x, individual.y))
        
        # Add edges from relationships
        for individual_id, relationships in social_network.relationships.items():
            for other_id, relationship in relationships.items():
                if relationship.strength > 0.1:  # Only significant relationships
                    G.add_edge(individual_id, other_id,
                              weight=relationship.strength,
                              trust=relationship.trust,
                              relationship_type=relationship.relationship_type,
                              interactions=len(relationship.interaction_history))
        
        return G
    
    def calculate_layout(self, G: nx.Graph, layout_type: str = 'spring') -> Dict[int, Tuple[float, float]]:
        """Calculate node positions using various layout algorithms"""
        cache_key = f"{layout_type}_{len(G.nodes())}"
        
        if cache_key in self.layout_cache and len(self.layout_cache[cache_key]) == len(G.nodes()):
            # Use cached layout if available and size matches
            cached_pos = self.layout_cache[cache_key]
            if all(node in cached_pos for node in G.nodes()):
                return cached_pos
        
        if layout_type == 'spring':
            pos = nx.spring_layout(G, k=3, iterations=50, scale=min(self.width, self.height) * 0.4)
        elif layout_type == 'circular':
            pos = nx.circular_layout(G, scale=min(self.width, self.height) * 0.4)
        elif layout_type == 'hierarchical':
            pos = self._hierarchical_layout(G)
        elif layout_type == 'community':
            pos = self._community_layout(G)
        elif layout_type == 'force_atlas':
            pos = self._force_atlas_layout(G)
        else:
            pos = nx.random_layout(G)
        
        # Convert to screen coordinates
        screen_pos = {}
        for node, (x, y) in pos.items():
            screen_x = self.width/2 + x
            screen_y = self.height/2 + y
            screen_pos[node] = (screen_x, screen_y)
        
        self.layout_cache[cache_key] = screen_pos
        return screen_pos
    
    def _hierarchical_layout(self, G: nx.Graph) -> Dict[int, Tuple[float, float]]:
        """Create hierarchical layout based on leadership and influence"""
        pos = {}
        
        # Get leadership information
        leaders = set()
        for node in G.nodes():
            if G.degree(node) > np.mean([G.degree(n) for n in G.nodes()]) + np.std([G.degree(n) for n in G.nodes()]):
                leaders.add(node)
        
        # Assign levels
        levels = defaultdict(list)
        levels[0] = list(leaders) if leaders else [random.choice(list(G.nodes()))]
        
        remaining_nodes = set(G.nodes()) - set(levels[0])
        level = 1
        
        while remaining_nodes and level < 5:
            current_level = []
            for node in list(remaining_nodes):
                # Check if connected to previous level
                if any(G.has_edge(node, prev_node) for prev_node in levels[level-1]):
                    current_level.append(node)
            
            if current_level:
                levels[level] = current_level
                remaining_nodes -= set(current_level)
                level += 1
            else:
                levels[level] = list(remaining_nodes)
                break
        
        # Position nodes in levels
        y_spacing = self.height / (len(levels) + 1)
        
        for level_idx, nodes in levels.items():
            y = (level_idx + 1) * y_spacing - self.height/2
            x_spacing = self.width / (len(nodes) + 1) if nodes else self.width/2
            
            for i, node in enumerate(nodes):
                x = (i + 1) * x_spacing - self.width/2
                pos[node] = (x, y)
        
        return pos
    
    def _community_layout(self, G: nx.Graph) -> Dict[int, Tuple[float, float]]:
        """Layout nodes based on community structure"""
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
        except:
            communities = [set(G.nodes())]
        
        pos = {}
        
        # Arrange communities in circle
        community_angles = np.linspace(0, 2*np.pi, len(communities), endpoint=False)
        community_radius = min(self.width, self.height) * 0.3
        
        for i, community in enumerate(communities):
            # Community center
            center_x = community_radius * np.cos(community_angles[i])
            center_y = community_radius * np.sin(community_angles[i])
            
            # Layout within community
            if len(community) == 1:
                pos[list(community)[0]] = (center_x, center_y)
            else:
                # Small spring layout within community
                subgraph = G.subgraph(community)
                sub_pos = nx.spring_layout(subgraph, scale=50)
                
                for node, (x, y) in sub_pos.items():
                    pos[node] = (center_x + x, center_y + y)
        
        return pos
    
    def _force_atlas_layout(self, G: nx.Graph, iterations: int = 50) -> Dict[int, Tuple[float, float]]:
        """Simplified Force Atlas layout algorithm"""
        if not G.nodes():
            return {}
        
        # Initialize random positions
        pos = {node: (random.uniform(-100, 100), random.uniform(-100, 100)) for node in G.nodes()}
        
        # Force Atlas parameters
        repulsion_strength = 1000
        attraction_strength = 1
        gravity = 0.1
        
        for iteration in range(iterations):
            forces = {node: [0, 0] for node in G.nodes()}
            
            # Repulsion forces (all pairs)
            for node1 in G.nodes():
                for node2 in G.nodes():
                    if node1 != node2:
                        x1, y1 = pos[node1]
                        x2, y2 = pos[node2]
                        
                        dx = x1 - x2
                        dy = y1 - y2
                        distance = math.sqrt(dx*dx + dy*dy) + 1e-6
                        
                        # Repulsion force
                        force = repulsion_strength / (distance * distance)
                        forces[node1][0] += force * dx / distance
                        forces[node1][1] += force * dy / distance
            
            # Attraction forces (connected nodes)
            for edge in G.edges():
                node1, node2 = edge
                x1, y1 = pos[node1]
                x2, y2 = pos[node2]
                
                dx = x2 - x1
                dy = y2 - y1
                distance = math.sqrt(dx*dx + dy*dy) + 1e-6
                
                # Attraction force
                weight = G[node1][node2].get('weight', 1.0)
                force = attraction_strength * weight * distance
                
                forces[node1][0] += force * dx / distance
                forces[node1][1] += force * dy / distance
                forces[node2][0] -= force * dx / distance
                forces[node2][1] -= force * dy / distance
            
            # Gravity (toward center)
            for node in G.nodes():
                x, y = pos[node]
                distance = math.sqrt(x*x + y*y) + 1e-6
                gravity_force = gravity * distance
                
                forces[node][0] -= gravity_force * x / distance
                forces[node][1] -= gravity_force * y / distance
            
            # Update positions
            for node in G.nodes():
                x, y = pos[node]
                fx, fy = forces[node]
                
                # Damping
                damping = 0.1
                pos[node] = (x + fx * damping, y + fy * damping)
        
        return pos
    
    def render_network(self, ax: plt.Axes, G: nx.Graph, 
                      layout_type: str = 'spring',
                      node_color_by: str = 'species',
                      edge_color_by: str = 'strength',
                      show_labels: bool = False,
                      highlight_communities: bool = False) -> None:
        """Render the social network"""
        if not G.nodes():
            ax.text(0.5, 0.5, 'No Social Network Data', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Calculate layout
        pos = self.calculate_layout(G, layout_type)
        
        # Draw communities if requested
        if highlight_communities:
            self._draw_communities(ax, G, pos)
        
        # Draw edges
        self._draw_edges(ax, G, pos, edge_color_by)
        
        # Draw nodes
        self._draw_nodes(ax, G, pos, node_color_by)
        
        # Draw labels if requested
        if show_labels and len(G.nodes()) < 50:  # Only for small networks
            self._draw_labels(ax, G, pos)
        
        # Set axis properties
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_communities(self, ax: plt.Axes, G: nx.Graph, pos: Dict[int, Tuple[float, float]]):
        """Draw community boundaries"""
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
        except:
            return
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        
        for i, community in enumerate(communities):
            if len(community) > 1:
                # Get bounding box of community
                xs = [pos[node][0] for node in community]
                ys = [pos[node][1] for node in community]
                
                # Calculate convex hull or simple bounding circle
                center_x = np.mean(xs)
                center_y = np.mean(ys)
                radius = max(np.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in zip(xs, ys)) + 20
                
                circle = plt.Circle((center_x, center_y), radius, 
                                  fill=False, color=colors[i], 
                                  linewidth=2, alpha=0.7, linestyle='--')
                ax.add_patch(circle)
    
    def _draw_edges(self, ax: plt.Axes, G: nx.Graph, pos: Dict[int, Tuple[float, float]], 
                   edge_color_by: str):
        """Draw network edges"""
        if not G.edges():
            return
        
        # Prepare edge data
        edge_positions = []
        edge_colors = []
        edge_widths = []
        
        for edge in G.edges(data=True):
            node1, node2, data = edge
            if node1 in pos and node2 in pos:
                x1, y1 = pos[node1]
                x2, y2 = pos[node2]
                edge_positions.append([(x1, y1), (x2, y2)])
                
                # Edge color
                if edge_color_by == 'strength':
                    strength = data.get('weight', 0.5)
                    edge_colors.append(plt.cm.viridis(strength))
                elif edge_color_by == 'trust':
                    trust = data.get('trust', 0.5)
                    edge_colors.append(plt.cm.RdYlGn(trust))
                elif edge_color_by == 'type':
                    rel_type = data.get('relationship_type', 'neutral')
                    color = self.color_schemes['relationship'].get(rel_type, '#CCCCCC')
                    edge_colors.append(color)
                else:
                    edge_colors.append('#CCCCCC')
                
                # Edge width
                weight = data.get('weight', 0.5)
                edge_widths.append(max(0.5, weight * 3))
        
        # Draw edges as line collection
        if edge_positions:
            lc = LineCollection(edge_positions, colors=edge_colors, 
                              linewidths=edge_widths, alpha=0.6)
            ax.add_collection(lc)
    
    def _draw_nodes(self, ax: plt.Axes, G: nx.Graph, pos: Dict[int, Tuple[float, float]], 
                   node_color_by: str):
        """Draw network nodes"""
        for node, data in G.nodes(data=True):
            if node not in pos:
                continue
            
            x, y = pos[node]
            
            # Node size based on degree or other metric
            degree = G.degree(node)
            max_degree = max(G.degree(n) for n in G.nodes()) if G.nodes() else 1
            size = 50 + (degree / max_degree) * 200
            
            # Node color
            if node_color_by == 'species':
                species = data.get('species', 'unknown')
                color = self.color_schemes['species'].get(species, '#888888')
            elif node_color_by == 'intelligence':
                intelligence = data.get('intelligence', 0.5)
                color = plt.cm.plasma(intelligence)
            elif node_color_by == 'energy':
                energy = data.get('energy', 50) / 100.0
                color = plt.cm.RdYlGn(energy)
            elif node_color_by == 'centrality':
                centrality = degree / max_degree
                color = plt.cm.cool(centrality)
            else:
                color = '#4444AA'
            
            # Draw node
            circle = plt.Circle((x, y), np.sqrt(size), color=color, alpha=0.8)
            ax.add_patch(circle)
            
            # Add border for leaders or special nodes
            if degree > np.mean([G.degree(n) for n in G.nodes()]) + np.std([G.degree(n) for n in G.nodes()]):
                border_circle = plt.Circle((x, y), np.sqrt(size) + 2, 
                                         fill=False, color='gold', linewidth=3)
                ax.add_patch(border_circle)
    
    def _draw_labels(self, ax: plt.Axes, G: nx.Graph, pos: Dict[int, Tuple[float, float]]):
        """Draw node labels"""
        for node in G.nodes():
            if node in pos:
                x, y = pos[node]
                ax.text(x, y, str(node), ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='white')
    
    def create_relationship_matrix(self, social_network, individuals: List) -> np.ndarray:
        """Create relationship strength matrix"""
        n = len(individuals)
        matrix = np.zeros((n, n))
        
        id_to_index = {ind.id: i for i, ind in enumerate(individuals)}
        
        for individual_id, relationships in social_network.relationships.items():
            if individual_id in id_to_index:
                i = id_to_index[individual_id]
                for other_id, relationship in relationships.items():
                    if other_id in id_to_index:
                        j = id_to_index[other_id]
                        matrix[i, j] = relationship.strength
        
        return matrix
    
    def render_relationship_matrix(self, ax: plt.Axes, matrix: np.ndarray, 
                                  individuals: List):
        """Render relationship matrix as heatmap"""
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Relationship Strength')
        
        # Set labels
        if len(individuals) < 20:  # Only for small networks
            labels = [f"{ind.species_name[:3]}-{ind.id}" for ind in individuals]
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticklabels(labels)
        
        ax.set_title('Relationship Strength Matrix')
    
    def create_communication_flow_diagram(self, ax: plt.Axes, individuals: List):
        """Create communication flow visualization"""
        # Filter individuals with active signals
        communicating = [ind for ind in individuals if ind.active_signals]
        
        if not communicating:
            ax.text(0.5, 0.5, 'No Active Communication', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Arrange in circle
        n = len(communicating)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = min(self.width, self.height) * 0.3
        
        positions = {}
        for i, individual in enumerate(communicating):
            x = self.width/2 + radius * np.cos(angles[i])
            y = self.height/2 + radius * np.sin(angles[i])
            positions[individual.id] = (x, y)
        
        # Draw communication nodes
        for individual in communicating:
            x, y = positions[individual.id]
            
            # Node size based on number of signals
            size = 20 + len(individual.active_signals) * 10
            
            # Color by species
            color = self.color_schemes['species'].get(individual.species_name, '#888888')
            
            circle = plt.Circle((x, y), size, color=color, alpha=0.8)
            ax.add_patch(circle)
            
            # Draw signal indicators
            for i, signal_id in enumerate(individual.active_signals):
                angle = 2 * np.pi * i / len(individual.active_signals)
                signal_x = x + (size + 15) * np.cos(angle)
                signal_y = y + (size + 15) * np.sin(angle)
                
                signal_circle = plt.Circle((signal_x, signal_y), 5, 
                                         color='yellow', alpha=0.9)
                ax.add_patch(signal_circle)
        
        # Draw communication links
        for i, sender in enumerate(communicating):
            for j, receiver in enumerate(communicating):
                if i != j:
                    # Check if they can communicate (within range)
                    distance = np.sqrt((sender.x - receiver.x)**2 + (sender.y - receiver.y)**2)
                    if distance < 120:  # Communication radius
                        x1, y1 = positions[sender.id]
                        x2, y2 = positions[receiver.id]
                        
                        # Draw communication beam
                        ax.plot([x1, x2], [y1, y2], 'yellow', alpha=0.3, linewidth=2)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Communication Flow')


class CommunityAnalysisViz:
    """Specialized visualization for community analysis"""
    
    def __init__(self):
        pass
    
    def plot_community_evolution(self, ax: plt.Axes, community_history: List[List[set]]):
        """Plot how communities evolve over time"""
        if not community_history:
            return
        
        # Track community IDs over time
        community_tracker = {}
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        time_steps = []
        community_sizes = defaultdict(list)
        
        for t, communities in enumerate(community_history):
            time_steps.append(t)
            
            # Match communities across time steps
            for i, community in enumerate(communities):
                community_id = f"C{i}"
                size = len(community)
                community_sizes[community_id].append(size)
        
        # Plot community size evolution
        for i, (comm_id, sizes) in enumerate(community_sizes.items()):
            if len(sizes) > 1:  # Only plot persistent communities
                color = colors[i % len(colors)]
                ax.plot(time_steps[:len(sizes)], sizes, 
                       color=color, label=comm_id, linewidth=2)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Community Size')
        ax.set_title('Community Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_modularity_over_time(self, ax: plt.Axes, G_history: List[nx.Graph]):
        """Plot network modularity over time"""
        time_steps = []
        modularities = []
        
        for t, G in enumerate(G_history):
            if G.edges():
                try:
                    communities = nx.community.greedy_modularity_communities(G)
                    modularity = nx.community.modularity(G, communities)
                    modularities.append(modularity)
                    time_steps.append(t)
                except:
                    pass
        
        if modularities:
            ax.plot(time_steps, modularities, 'b-', linewidth=2)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Modularity')
            ax.set_title('Network Modularity Over Time')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)


def create_social_visualization(simulation, config) -> SocialNetworkVisualizer:
    """Create and return a social network visualizer"""
    return SocialNetworkVisualizer(config.width, config.height)