"""
Main visualization system for the Emergent Intelligence Ecosystem.

This module provides comprehensive multi-panel visualization including the main simulation view,
social networks, statistics, and real-time analytics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from matplotlib.collections import LineCollection
from typing import List, Dict, Any, Optional

from ..core.simulation import EmergentIntelligenceSimulation
from ..config import Config


class MainVisualization:
    """Comprehensive multi-panel visualization system"""
    
    def __init__(self, simulation: EmergentIntelligenceSimulation, config: Config = None):
        self.simulation = simulation
        self.config = config or Config()
        
        # Setup figure and subplots
        self.fig = plt.figure(figsize=self.config.visualization.figure_size)
        self.fig.patch.set_facecolor(self.config.visualization.background_color)
        
        self._setup_subplots()
        
        # Animation state
        self.animation = None
        
    def _setup_subplots(self):
        """Setup all visualization subplots"""
        # Main simulation view (large, top-left)
        self.ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
        
        # Social network view (large, top-right)
        self.ax_social = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
        
        # Statistics panels (bottom row)
        self.ax_stats = plt.subplot2grid((4, 4), (2, 0))
        self.ax_comm = plt.subplot2grid((4, 4), (2, 1))
        self.ax_culture = plt.subplot2grid((4, 4), (2, 2))
        self.ax_env = plt.subplot2grid((4, 4), (2, 3))
        
        # Analysis panels (bottom row)
        self.ax_intel = plt.subplot2grid((4, 4), (3, 0))
        self.ax_social_metrics = plt.subplot2grid((4, 4), (3, 1))
        self.ax_phase = plt.subplot2grid((4, 4), (3, 2))
        self.ax_emergence = plt.subplot2grid((4, 4), (3, 3))
        
        # Set background colors
        for ax in [self.ax_main, self.ax_social, self.ax_stats, self.ax_comm, 
                   self.ax_culture, self.ax_env, self.ax_intel, self.ax_social_metrics,
                   self.ax_phase, self.ax_emergence]:
            ax.set_facecolor(self.config.visualization.background_color)
    
    def update_visualization(self, frame: int = None) -> List:
        """Update all visualization components"""
        # Update simulation
        self.simulation.update(frame)
        
        # Clear all axes
        self._clear_all_axes()
        
        # Render all components
        self._render_main_simulation()
        self._render_social_network()
        self._render_population_statistics()
        self._render_communication_evolution()
        self._render_cultural_knowledge()
        self._render_environment_state()
        self._render_intelligence_evolution()
        self._render_social_complexity()
        self._render_phase_space()
        self._render_emergence_events()
        
        return []
    
    def _clear_all_axes(self):
        """Clear all axes for redrawing"""
        for ax in [self.ax_main, self.ax_social, self.ax_stats, self.ax_comm, 
                   self.ax_culture, self.ax_env, self.ax_intel, self.ax_social_metrics,
                   self.ax_phase, self.ax_emergence]:
            ax.clear()
            ax.set_facecolor(self.config.visualization.background_color)
    
    def _render_main_simulation(self):
        """Render main simulation view with individuals and environment"""
        self.ax_main.set_xlim(0, self.config.width)
        self.ax_main.set_ylim(0, self.config.height)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('Emergent Intelligence Ecosystem', 
                              color=self.config.visualization.text_color, fontsize=16)
        
        # Draw environmental patches
        for patch in self.simulation.environment.patches:
            patch_colors = {'food': 'green', 'shelter': 'brown', 'neutral': 'gray', 'danger': 'red'}
            color = patch_colors.get(patch.patch_type, 'gray')
            alpha = patch.resource_level * 0.3 + 0.1
            self.ax_main.scatter(patch.x, patch.y, c=color, s=30, alpha=alpha, marker='s')
        
        # Draw individuals
        for ind in self.simulation.individuals:
            # Size based on energy and intelligence
            size = (ind.size + ind.intelligence * 10) * (0.5 + 0.5 * ind.energy / 100)
            
            # Color based on species and state
            color = ind.color
            
            # Edge color based on social connectivity
            social_connections = len(self.simulation.social_network.relationships.get(ind.id, {}))
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
        for leader_id in self.simulation.social_network.leaders:
            leader = next((ind for ind in self.simulation.individuals if ind.id == leader_id), None)
            if leader:
                relationships = self.simulation.social_network.relationships.get(leader_id, {})
                for other_id, rel in relationships.items():
                    if rel.strength > 0.5:
                        other = next((ind for ind in self.simulation.individuals if ind.id == other_id), None)
                        if other:
                            self.ax_main.plot([leader.x, other.x], [leader.y, other.y], 
                                            'cyan', alpha=0.3, linewidth=1)
        
        # Information text
        info_text = f"Population: {len(self.simulation.individuals)}\n"
        if self.simulation.individuals:
            info_text += f"Generation: {max(ind.generation for ind in self.simulation.individuals)}\n"
        info_text += f"Communities: {len(self.simulation.social_network.communities)}\n"
        info_text += f"Leaders: {len(self.simulation.social_network.leaders)}\n"
        info_text += f"Time: {self.simulation.time_step}"
        
        self.ax_main.text(10, self.config.height - 10, info_text, 
                         color=self.config.visualization.text_color, fontsize=11, 
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    def _render_social_network(self):
        """Render social network visualization"""
        self.ax_social.set_title('Social Network Structure', 
                                color=self.config.visualization.text_color, fontsize=12)
        
        if len(self.simulation.individuals) > 0 and len(self.simulation.social_network.relationships) > 0:
            # Create networkx graph for visualization
            G = nx.Graph()
            
            for ind in self.simulation.individuals:
                G.add_node(ind.id, species=ind.species_name, intelligence=ind.intelligence)
            
            for ind_id, relationships in self.simulation.social_network.relationships.items():
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
                for species, color in self.config.visualization.species_colors.items():
                    species_nodes = [n for n, d in G.nodes(data=True) if d.get('species') == species]
                    if species_nodes:
                        nx.draw_networkx_nodes(G, pos, nodelist=species_nodes, 
                                             node_color=[color], node_size=50, 
                                             alpha=0.7, ax=self.ax_social)
                
                # Draw edges
                edges = G.edges(data=True)
                if edges:
                    edge_weights = [d['weight'] for _, _, d in edges]
                    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_weights, 
                                         edge_color='white', ax=self.ax_social)
                
                # Highlight leaders
                leader_nodes = [n for n in G.nodes() if n in self.simulation.social_network.leaders]
                if leader_nodes:
                    nx.draw_networkx_nodes(G, pos, nodelist=leader_nodes, 
                                         node_color='gold', node_size=100, 
                                         alpha=0.8, ax=self.ax_social)
        
        self.ax_social.set_aspect('equal')
    
    def _render_population_statistics(self):
        """Render population dynamics statistics"""
        stats_history = list(self.simulation.statistics_tracker.generation_stats)
        
        if len(stats_history) < 2:
            self.ax_stats.text(0.5, 0.5, 'Insufficient Data', 
                              transform=self.ax_stats.transAxes, ha='center',
                              color=self.config.visualization.text_color)
            return
        
        self.ax_stats.set_title('Population Dynamics', 
                               color=self.config.visualization.text_color, fontsize=10)
        
        recent_stats = stats_history[-200:]
        time_steps = [s['time_step'] for s in recent_stats]
        populations = [s['population_size'] for s in recent_stats]
        
        self.ax_stats.plot(time_steps, populations, 'white', linewidth=2)
        self.ax_stats.set_ylabel('Population', color=self.config.visualization.text_color)
        self.ax_stats.grid(True, alpha=0.3)
        
        # Mark emergence events
        emergence_events = self.simulation.emergence_detector.get_recent_events(10)
        for event in emergence_events:
            event_time = event['time_step']
            if event_time in time_steps:
                self.ax_stats.axvline(event_time, color='red', alpha=0.7, linestyle='--')
    
    def _render_communication_evolution(self):
        """Render communication system evolution"""
        comm_history = list(self.simulation.statistics_tracker.communication_evolution_data)
        
        if len(comm_history) < 2:
            self.ax_comm.text(0.5, 0.5, 'No Communication Data', 
                             transform=self.ax_comm.transAxes, ha='center',
                             color=self.config.visualization.text_color)
            return
        
        self.ax_comm.set_title('Communication Evolution', 
                              color=self.config.visualization.text_color, fontsize=10)
        
        recent_comm = comm_history[-100:]
        time_steps = [c['time_step'] for c in recent_comm]
        total_signals = [c['total_unique_signals'] for c in recent_comm]
        complexity = [c['communication_complexity'] for c in recent_comm]
        
        self.ax_comm.plot(time_steps, total_signals, 'cyan', label='Total Signals', linewidth=2)
        self.ax_comm.plot(time_steps, complexity, 'yellow', label='Complexity', linewidth=2)
        
        self.ax_comm.set_ylabel('Signals/Complexity', color=self.config.visualization.text_color)
        self.ax_comm.legend(fontsize=8)
        self.ax_comm.grid(True, alpha=0.3)
    
    def _render_cultural_knowledge(self):
        """Render cultural knowledge accumulation"""
        cultural_history = list(self.simulation.statistics_tracker.cultural_evolution_data)
        
        if len(cultural_history) < 2:
            self.ax_culture.text(0.5, 0.5, 'No Cultural Data', 
                                transform=self.ax_culture.transAxes, ha='center',
                                color=self.config.visualization.text_color)
            return
        
        self.ax_culture.set_title('Cultural Knowledge', 
                                 color=self.config.visualization.text_color, fontsize=10)
        
        recent_culture = cultural_history[-100:]
        time_steps = [c['time_step'] for c in recent_culture]
        total_knowledge = [c['total_cultural_knowledge'] for c in recent_culture]
        diversity = [c['cultural_diversity'] for c in recent_culture]
        
        self.ax_culture.plot(time_steps, total_knowledge, 'orange', label='Total Knowledge', linewidth=2)
        self.ax_culture.plot(time_steps, diversity, 'purple', label='Diversity', linewidth=2)
        
        self.ax_culture.set_ylabel('Knowledge', color=self.config.visualization.text_color)
        self.ax_culture.legend(fontsize=8)
        self.ax_culture.grid(True, alpha=0.3)
    
    def _render_environment_state(self):
        """Render environmental state"""
        env_history = list(self.simulation.statistics_tracker.environmental_evolution)
        
        if len(env_history) < 2:
            self.ax_env.text(0.5, 0.5, 'No Environmental Data', 
                            transform=self.ax_env.transAxes, ha='center',
                            color=self.config.visualization.text_color)
            return
        
        self.ax_env.set_title('Environmental Health', 
                             color=self.config.visualization.text_color, fontsize=10)
        
        recent_env = env_history[-100:]
        time_steps = [e['time_step'] for e in recent_env]
        health = [e['global_health'] for e in recent_env]
        complexity = [e['environmental_complexity'] for e in recent_env]
        
        self.ax_env.plot(time_steps, health, 'green', label='Health', linewidth=2)
        self.ax_env.plot(time_steps, complexity, 'brown', label='Complexity', linewidth=2)
        
        self.ax_env.set_ylabel('Environmental Metrics', color=self.config.visualization.text_color)
        self.ax_env.legend(fontsize=8)
        self.ax_env.grid(True, alpha=0.3)
    
    def _render_intelligence_evolution(self):
        """Render intelligence evolution over time"""
        intel_history = list(self.simulation.statistics_tracker.intelligence_evolution)
        
        if len(intel_history) < 2:
            self.ax_intel.text(0.5, 0.5, 'No Intelligence Data', 
                              transform=self.ax_intel.transAxes, ha='center',
                              color=self.config.visualization.text_color)
            return
        
        self.ax_intel.set_title('Intelligence Evolution', 
                               color=self.config.visualization.text_color, fontsize=10)
        
        recent_intel = intel_history[-100:]
        time_steps = [i['time_step'] for i in recent_intel]
        avg_intelligence = [i['avg_intelligence'] for i in recent_intel]
        max_intelligence = [i['max_intelligence'] for i in recent_intel]
        
        self.ax_intel.plot(time_steps, avg_intelligence, 'purple', label='Average', linewidth=2)
        self.ax_intel.plot(time_steps, max_intelligence, 'magenta', label='Maximum', linewidth=2)
        
        self.ax_intel.set_ylabel('Intelligence', color=self.config.visualization.text_color)
        self.ax_intel.legend(fontsize=8)
        self.ax_intel.grid(True, alpha=0.3)
    
    def _render_social_complexity(self):
        """Render social complexity metrics"""
        social_history = list(self.simulation.statistics_tracker.social_evolution)
        
        if len(social_history) < 2:
            self.ax_social_metrics.text(0.5, 0.5, 'No Social Data', 
                                       transform=self.ax_social_metrics.transAxes, ha='center',
                                       color=self.config.visualization.text_color)
            return
        
        self.ax_social_metrics.set_title('Social Complexity', 
                                        color=self.config.visualization.text_color, fontsize=10)
        
        recent_social = social_history[-100:]
        time_steps = [s['time_step'] for s in recent_social]
        communities = [s['num_communities'] for s in recent_social]
        leaders = [s['num_leaders'] for s in recent_social]
        
        self.ax_social_metrics.plot(time_steps, communities, 'cyan', label='Communities', linewidth=2)
        self.ax_social_metrics.plot(time_steps, leaders, 'gold', label='Leaders', linewidth=2)
        
        self.ax_social_metrics.set_ylabel('Count', color=self.config.visualization.text_color)
        self.ax_social_metrics.legend(fontsize=8)
        self.ax_social_metrics.grid(True, alpha=0.3)
    
    def _render_phase_space(self):
        """Render phase space of key variables"""
        if not self.simulation.individuals:
            self.ax_phase.text(0.5, 0.5, 'No Individuals', 
                              transform=self.ax_phase.transAxes, ha='center',
                              color=self.config.visualization.text_color)
            return
        
        self.ax_phase.set_title('Intelligence vs Sociability', 
                               color=self.config.visualization.text_color, fontsize=10)
        
        intelligence = [ind.intelligence for ind in self.simulation.individuals]
        sociability = [ind.sociability for ind in self.simulation.individuals]
        species = [ind.species_name for ind in self.simulation.individuals]
        
        for species_name, color in self.config.visualization.species_colors.items():
            species_intel = [intelligence[i] for i, s in enumerate(species) if s == species_name]
            species_social = [sociability[i] for i, s in enumerate(species) if s == species_name]
            
            if species_intel and species_social:
                self.ax_phase.scatter(species_intel, species_social, c=[color], 
                                    label=species_name, alpha=0.7, s=30)
        
        self.ax_phase.set_xlabel('Intelligence', color=self.config.visualization.text_color)
        self.ax_phase.set_ylabel('Sociability', color=self.config.visualization.text_color)
        self.ax_phase.legend(fontsize=8)
        self.ax_phase.grid(True, alpha=0.3)
    
    def _render_emergence_events(self):
        """Render emergence event timeline"""
        emergence_events = self.simulation.emergence_detector.get_recent_events(20)
        
        if not emergence_events:
            self.ax_emergence.text(0.5, 0.5, 'No Emergence Events', 
                                  transform=self.ax_emergence.transAxes, ha='center',
                                  color=self.config.visualization.text_color)
            return
        
        self.ax_emergence.set_title('Emergence Events', 
                                   color=self.config.visualization.text_color, fontsize=10)
        
        event_types = [event['type'] for event in emergence_events]
        event_times = [event['time_step'] for event in emergence_events]
        
        # Create color map for event types
        unique_events = list(set(event_types))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_events)))
        event_colors = {event: colors[i] for i, event in enumerate(unique_events)}
        
        for i, (event_type, event_time) in enumerate(zip(event_types, event_times)):
            color = event_colors[event_type]
            self.ax_emergence.scatter(event_time, i, c=[color], s=100, alpha=0.8)
        
        # Legend
        for event_type, color in event_colors.items():
            self.ax_emergence.scatter([], [], c=[color], 
                                    label=event_type.replace('_', ' ').title())
        
        self.ax_emergence.legend(fontsize=6, loc='upper left')
        self.ax_emergence.set_xlabel('Time Step', color=self.config.visualization.text_color)
        self.ax_emergence.set_ylabel('Event #', color=self.config.visualization.text_color)
    
    def start_animation(self, frames: int = 5000, interval: int = 50):
        """Start the animation"""
        self.animation = animation.FuncAnimation(
            self.fig, self.update_visualization, frames=frames,
            interval=interval, blit=False, repeat=False
        )
        plt.tight_layout()
        return self.animation
    
    def save_frame(self, filename: str):
        """Save current frame as image"""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight', 
                        facecolor=self.config.visualization.background_color)
    
    def show(self):
        """Show the visualization"""
        plt.show()


def create_visualization(simulation: EmergentIntelligenceSimulation, 
                        config: Config = None) -> MainVisualization:
    """Create and return a main visualization instance"""
    return MainVisualization(simulation, config)
