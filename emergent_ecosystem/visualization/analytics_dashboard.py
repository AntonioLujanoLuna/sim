"""
Multi-panel analytics and metrics dashboard.

This module provides comprehensive real-time analytics dashboard with
interactive data exploration, export capabilities, and advanced metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, CheckButtons
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, deque
import json
import os
from datetime import datetime
import pickle


class MetricsPanel:
    """Individual analytics panel for specific metrics"""
    
    def __init__(self, panel_id: str, title: str, panel_type: str = 'line'):
        self.panel_id = panel_id
        self.title = title
        self.panel_type = panel_type  # 'line', 'bar', 'scatter', 'heatmap', 'network'
        self.data_history = deque(maxlen=1000)
        self.metrics = {}
        self.update_functions = []
        self.is_active = True
        
    def add_data_point(self, timestamp: int, data: Dict[str, Any]):
        """Add data point to panel history"""
        data_point = {'timestamp': timestamp, **data}
        self.data_history.append(data_point)
        
    def set_update_function(self, func: Callable):
        """Set function to update this panel"""
        self.update_functions.append(func)
    
    def render(self, ax: plt.Axes, **kwargs):
        """Render the panel content"""
        if not self.data_history:
            ax.text(0.5, 0.5, f'No data for {self.title}', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        ax.clear()
        ax.set_title(self.title, fontsize=10, color='white')
        ax.set_facecolor('black')
        
        if self.panel_type == 'line':
            self._render_line_plot(ax, **kwargs)
        elif self.panel_type == 'bar':
            self._render_bar_plot(ax, **kwargs)
        elif self.panel_type == 'scatter':
            self._render_scatter_plot(ax, **kwargs)
        elif self.panel_type == 'heatmap':
            self._render_heatmap(ax, **kwargs)
        elif self.panel_type == 'network':
            self._render_network(ax, **kwargs)
        elif self.panel_type == 'histogram':
            self._render_histogram(ax, **kwargs)
        
        # Style the axes
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, alpha=0.3)
        
        for spine in ax.spines.values():
            spine.set_color('white')
    
    def _render_line_plot(self, ax: plt.Axes, **kwargs):
        """Render line plot"""
        recent_data = list(self.data_history)[-100:]  # Last 100 points
        
        if not recent_data:
            return
        
        timestamps = [d['timestamp'] for d in recent_data]
        
        # Get all numeric keys except timestamp
        numeric_keys = []
        for key in recent_data[0].keys():
            if key != 'timestamp':
                try:
                    float(recent_data[0][key])
                    numeric_keys.append(key)
                except (ValueError, TypeError):
                    pass
        
        # Plot up to 5 metrics to avoid clutter
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(numeric_keys), 5)))
        
        for i, key in enumerate(numeric_keys[:5]):
            values = []
            for d in recent_data:
                try:
                    values.append(float(d.get(key, 0)))
                except (ValueError, TypeError):
                    values.append(0)
            
            if values:
                ax.plot(timestamps, values, color=colors[i], 
                       label=key.replace('_', ' ').title(), linewidth=2)
        
        if numeric_keys:
            ax.legend(fontsize=8, loc='upper left')
            ax.set_xlabel('Time Step', color='white')
    
    def _render_bar_plot(self, ax: plt.Axes, **kwargs):
        """Render bar plot"""
        if not self.data_history:
            return
        
        latest_data = self.data_history[-1]
        
        # Get numeric data for bar plot
        categories = []
        values = []
        
        for key, value in latest_data.items():
            if key != 'timestamp':
                try:
                    float_val = float(value)
                    categories.append(key.replace('_', ' ').title())
                    values.append(float_val)
                except (ValueError, TypeError):
                    if isinstance(value, dict):
                        # Handle dictionary values (e.g., species counts)
                        for sub_key, sub_value in value.items():
                            try:
                                categories.append(f"{key}: {sub_key}")
                                values.append(float(sub_value))
                            except (ValueError, TypeError):
                                pass
        
        if categories and values:
            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
            bars = ax.bar(range(len(categories)), values, color=colors)
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8, color='white')
    
    def _render_scatter_plot(self, ax: plt.Axes, **kwargs):
        """Render scatter plot"""
        recent_data = list(self.data_history)[-100:]
        
        if len(recent_data) < 2:
            return
        
        # Try to find two numeric variables for x and y
        numeric_keys = []
        for key in recent_data[0].keys():
            if key != 'timestamp':
                try:
                    float(recent_data[0][key])
                    numeric_keys.append(key)
                except (ValueError, TypeError):
                    pass
        
        if len(numeric_keys) >= 2:
            x_key, y_key = numeric_keys[:2]
            
            x_values = []
            y_values = []
            colors = []
            
            for d in recent_data:
                try:
                    x_val = float(d.get(x_key, 0))
                    y_val = float(d.get(y_key, 0))
                    x_values.append(x_val)
                    y_values.append(y_val)
                    colors.append(d['timestamp'])
                except (ValueError, TypeError):
                    pass
            
            if x_values and y_values:
                scatter = ax.scatter(x_values, y_values, c=colors, 
                                   cmap='viridis', alpha=0.7, s=30)
                ax.set_xlabel(x_key.replace('_', ' ').title(), color='white')
                ax.set_ylabel(y_key.replace('_', ' ').title(), color='white')
                
                if len(colors) > 1:
                    plt.colorbar(scatter, ax=ax, label='Time')
    
    def _render_heatmap(self, ax: plt.Axes, **kwargs):
        """Render heatmap"""
        if not self.data_history:
            return
        
        # Look for matrix-like data in the latest entry
        latest_data = self.data_history[-1]
        
        matrix_data = None
        matrix_key = None
        
        for key, value in latest_data.items():
            if isinstance(value, (list, np.ndarray)):
                try:
                    matrix_data = np.array(value)
                    if len(matrix_data.shape) == 2:
                        matrix_key = key
                        break
                except:
                    pass
        
        if matrix_data is not None:
            im = ax.imshow(matrix_data, cmap='viridis', aspect='auto')
            ax.set_title(f"{matrix_key.replace('_', ' ').title()}")
            plt.colorbar(im, ax=ax)
        else:
            # Create heatmap from time series data
            recent_data = list(self.data_history)[-20:]
            if len(recent_data) > 5:
                numeric_keys = []
                for key in recent_data[0].keys():
                    if key != 'timestamp':
                        try:
                            float(recent_data[0][key])
                            numeric_keys.append(key)
                        except:
                            pass
                
                if numeric_keys:
                    matrix = []
                    for d in recent_data:
                        row = []
                        for key in numeric_keys:
                            try:
                                row.append(float(d.get(key, 0)))
                            except:
                                row.append(0)
                        matrix.append(row)
                    
                    if matrix:
                        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
                        ax.set_ylabel('Time')
                        ax.set_xlabel('Metrics')
                        ax.set_xticks(range(len(numeric_keys)))
                        ax.set_xticklabels([k.replace('_', ' ') for k in numeric_keys], 
                                         rotation=45, ha='right')
                        plt.colorbar(im, ax=ax)
    
    def _render_network(self, ax: plt.Axes, **kwargs):
        """Render network visualization"""
        if not self.data_history:
            return
        
        # Look for network data in latest entry
        latest_data = self.data_history[-1]
        
        # Try to extract network data
        G = nx.Graph()
        
        # Look for nodes and edges data
        if 'nodes' in latest_data and 'edges' in latest_data:
            nodes = latest_data['nodes']
            edges = latest_data['edges']
            
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
        
        elif 'adjacency_matrix' in latest_data:
            matrix = np.array(latest_data['adjacency_matrix'])
            G = nx.from_numpy_array(matrix)
        
        else:
            # Create dummy network for demonstration
            for i in range(10):
                G.add_node(i)
            for i in range(10):
                for j in range(i+1, 10):
                    if np.random.random() < 0.3:
                        G.add_edge(i, j)
        
        if G.nodes():
            pos = nx.spring_layout(G, k=1, iterations=50)
            nx.draw(G, pos, ax=ax, node_color='lightblue', 
                   node_size=100, with_labels=True, font_size=8,
                   edge_color='gray', alpha=0.7)
    
    def _render_histogram(self, ax: plt.Axes, **kwargs):
        """Render histogram"""
        recent_data = list(self.data_history)[-100:]
        
        if not recent_data:
            return
        
        # Find a numeric variable to histogram
        for key in recent_data[0].keys():
            if key != 'timestamp':
                values = []
                for d in recent_data:
                    try:
                        values.append(float(d.get(key, 0)))
                    except:
                        pass
                
                if len(values) > 5:
                    ax.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_xlabel(key.replace('_', ' ').title(), color='white')
                    ax.set_ylabel('Frequency', color='white')
                    break
    
    def get_latest_summary(self) -> Dict[str, Any]:
        """Get summary of latest data"""
        if not self.data_history:
            return {}
        
        latest = self.data_history[-1]
        summary = {
            'panel_id': self.panel_id,
            'title': self.title,
            'latest_timestamp': latest.get('timestamp', 0),
            'data_points': len(self.data_history),
            'latest_values': {}
        }
        
        for key, value in latest.items():
            if key != 'timestamp':
                try:
                    if isinstance(value, (int, float)):
                        summary['latest_values'][key] = value
                    elif isinstance(value, dict):
                        summary['latest_values'][key] = len(value)
                    else:
                        summary['latest_values'][key] = str(value)[:50]
                except:
                    pass
        
        return summary


class AnalyticsDashboard:
    """Main analytics dashboard coordinator"""
    
    def __init__(self, simulation, config, figsize=(20, 14)):
        self.simulation = simulation
        self.config = config
        self.panels = {}
        self.fig = None
        self.gs = None
        self.figsize = figsize
        
        # Dashboard state
        self.is_paused = False
        self.auto_export = False
        self.export_interval = 100
        self.last_export_step = 0
        
        # Data storage
        self.data_history = deque(maxlen=10000)
        self.export_directory = "simulation_exports"
        
        # Interactive widgets
        self.widgets = {}
        
        # Initialize panels
        self._initialize_panels()
        self._setup_dashboard()
    
    def _initialize_panels(self):
        """Initialize all analytics panels"""
        # Core metrics panels
        self.panels['population'] = MetricsPanel('population', 'Population Dynamics', 'line')
        self.panels['intelligence'] = MetricsPanel('intelligence', 'Intelligence Evolution', 'line')
        self.panels['social_network'] = MetricsPanel('social_network', 'Social Network', 'network')
        self.panels['communication'] = MetricsPanel('communication', 'Communication', 'bar')
        self.panels['species_distribution'] = MetricsPanel('species_distribution', 'Species Distribution', 'bar')
        self.panels['fitness_landscape'] = MetricsPanel('fitness_landscape', 'Fitness Distribution', 'histogram')
        self.panels['cultural_knowledge'] = MetricsPanel('cultural_knowledge', 'Cultural Evolution', 'line')
        self.panels['environmental_health'] = MetricsPanel('environmental_health', 'Environment Health', 'line')
        self.panels['emergence_events'] = MetricsPanel('emergence_events', 'Emergence Timeline', 'scatter')
        self.panels['phase_space'] = MetricsPanel('phase_space', 'Phase Space', 'scatter')
        self.panels['information_flow'] = MetricsPanel('information_flow', 'Information Theory', 'heatmap')
        self.panels['evolution_metrics'] = MetricsPanel('evolution_metrics', 'Evolution Metrics', 'line')
    
    def _setup_dashboard(self):
        """Setup the dashboard layout"""
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.patch.set_facecolor('black')
        
        # Create grid layout
        self.gs = gridspec.GridSpec(4, 4, figure=self.fig, 
                                   left=0.05, bottom=0.15, right=0.98, top=0.95,
                                   wspace=0.3, hspace=0.4)
        
        # Assign panels to grid positions
        panel_positions = {
            'population': (0, 0),
            'intelligence': (0, 1),
            'social_network': (0, 2),
            'communication': (0, 3),
            'species_distribution': (1, 0),
            'fitness_landscape': (1, 1),
            'cultural_knowledge': (1, 2),
            'environmental_health': (1, 3),
            'emergence_events': (2, 0),
            'phase_space': (2, 1),
            'information_flow': (2, 2),
            'evolution_metrics': (2, 3)
        }
        
        # Create axes for each panel
        self.axes = {}
        for panel_id, (row, col) in panel_positions.items():
            if panel_id in self.panels:
                self.axes[panel_id] = self.fig.add_subplot(self.gs[row, col])
        
        # Add control panel at bottom
        self._setup_control_panel()
    
    def _setup_control_panel(self):
        """Setup interactive control panel"""
        # Control panel area
        control_ax = self.fig.add_subplot(self.gs[3, :])
        control_ax.set_xlim(0, 10)
        control_ax.set_ylim(0, 1)
        control_ax.axis('off')
        
        # Pause/Resume button
        pause_ax = plt.axes([0.1, 0.02, 0.1, 0.05])
        self.widgets['pause_button'] = Button(pause_ax, 'Pause')
        self.widgets['pause_button'].on_clicked(self._toggle_pause)
        
        # Export button
        export_ax = plt.axes([0.25, 0.02, 0.1, 0.05])
        self.widgets['export_button'] = Button(export_ax, 'Export')
        self.widgets['export_button'].on_clicked(self._export_data)
        
        # Auto-export checkbox
        autoexport_ax = plt.axes([0.4, 0.02, 0.15, 0.05])
        self.widgets['autoexport_check'] = CheckButtons(autoexport_ax, ['Auto Export'], [self.auto_export])
        self.widgets['autoexport_check'].on_clicked(self._toggle_auto_export)
        
        # Export interval slider
        interval_ax = plt.axes([0.6, 0.02, 0.2, 0.05])
        self.widgets['interval_slider'] = Slider(interval_ax, 'Export Interval', 
                                                50, 500, valinit=self.export_interval, valfmt='%d')
        self.widgets['interval_slider'].on_changed(self._update_export_interval)
        
        # Reset button
        reset_ax = plt.axes([0.85, 0.02, 0.1, 0.05])
        self.widgets['reset_button'] = Button(reset_ax, 'Reset')
        self.widgets['reset_button'].on_clicked(self._reset_dashboard)
    
    def update_dashboard(self, timestamp: int):
        """Update all dashboard panels"""
        if self.is_paused:
            return
        
        # Collect comprehensive data from simulation
        dashboard_data = self._collect_simulation_data(timestamp)
        
        # Store data
        self.data_history.append(dashboard_data)
        
        # Update each panel
        for panel_id, panel in self.panels.items():
            if panel.is_active and panel_id in self.axes:
                # Extract relevant data for this panel
                panel_data = self._extract_panel_data(panel_id, dashboard_data)
                panel.add_data_point(timestamp, panel_data)
                
                # Render panel
                try:
                    panel.render(self.axes[panel_id])
                except Exception as e:
                    print(f"Error rendering panel {panel_id}: {e}")
        
        # Auto-export if enabled
        if (self.auto_export and 
            timestamp - self.last_export_step >= self.export_interval):
            self._export_data(None)
            self.last_export_step = timestamp
        
        # Update the display
        self.fig.canvas.draw_idle()
    
    def _collect_simulation_data(self, timestamp: int) -> Dict[str, Any]:
        """Collect comprehensive data from simulation"""
        data = {'timestamp': timestamp}
        
        # Population metrics
        if self.simulation.individuals:
            data['population_size'] = len(self.simulation.individuals)
            data['avg_intelligence'] = np.mean([ind.intelligence for ind in self.simulation.individuals])
            data['avg_energy'] = np.mean([ind.energy for ind in self.simulation.individuals])
            data['avg_sociability'] = np.mean([ind.sociability for ind in self.simulation.individuals])
            data['max_generation'] = max(ind.generation for ind in self.simulation.individuals)
            
            # Species distribution
            species_counts = defaultdict(int)
            for ind in self.simulation.individuals:
                species_counts[ind.species_name] += 1
            data['species_counts'] = dict(species_counts)
            
            # Fitness metrics
            fitness_scores = [ind.energy + ind.intelligence * 20 for ind in self.simulation.individuals]
            data['avg_fitness'] = np.mean(fitness_scores)
            data['fitness_variance'] = np.var(fitness_scores)
            data['fitness_distribution'] = fitness_scores
        
        # Social network metrics
        if hasattr(self.simulation, 'social_network'):
            network_metrics = self.simulation.social_network.get_network_metrics()
            data.update(network_metrics)
            data['num_communities'] = len(self.simulation.social_network.communities)
            data['num_leaders'] = len(self.simulation.social_network.leaders)
        
        # Communication metrics
        if self.simulation.individuals:
            total_signals = sum(len(ind.communication.signal_repertoire) for ind in self.simulation.individuals)
            active_signals = sum(len(ind.active_signals) for ind in self.simulation.individuals)
            data['total_signals'] = total_signals
            data['active_signals'] = active_signals
            data['avg_signals_per_individual'] = total_signals / len(self.simulation.individuals)
        
        # Cultural metrics
        if self.simulation.individuals:
            total_cultural_knowledge = sum(sum(ind.cultural_knowledge.values()) for ind in self.simulation.individuals)
            data['total_cultural_knowledge'] = total_cultural_knowledge
            data['avg_cultural_knowledge'] = total_cultural_knowledge / len(self.simulation.individuals)
        
        # Environmental metrics
        if hasattr(self.simulation, 'environment'):
            env_summary = self.simulation.environment.get_environmental_summary()
            data.update(env_summary)
        
        # Emergence events
        if hasattr(self.simulation, 'emergence_detector'):
            recent_events = self.simulation.emergence_detector.get_recent_events(5)
            data['recent_emergence_events'] = len(recent_events)
            data['emergence_event_types'] = [event['type'] for event in recent_events]
        
        # Statistics from tracker
        if hasattr(self.simulation, 'statistics_tracker'):
            latest_stats = self.simulation.statistics_tracker.get_latest_stats()
            data.update(latest_stats)
        
        return data
    
    def _extract_panel_data(self, panel_id: str, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant data for specific panel"""
        if panel_id == 'population':
            return {
                'population_size': dashboard_data.get('population_size', 0),
                'max_generation': dashboard_data.get('max_generation', 0),
                'avg_fitness': dashboard_data.get('avg_fitness', 0)
            }
        
        elif panel_id == 'intelligence':
            return {
                'avg_intelligence': dashboard_data.get('avg_intelligence', 0),
                'intelligence_variance': dashboard_data.get('intelligence_variance', 0)
            }
        
        elif panel_id == 'social_network':
            return {
                'nodes': list(range(dashboard_data.get('population_size', 0))),
                'edges': [],  # Would need actual edge data
                'communities': dashboard_data.get('num_communities', 0),
                'leaders': dashboard_data.get('num_leaders', 0)
            }
        
        elif panel_id == 'communication':
            return {
                'total_signals': dashboard_data.get('total_signals', 0),
                'active_signals': dashboard_data.get('active_signals', 0),
                'avg_signals': dashboard_data.get('avg_signals_per_individual', 0)
            }
        
        elif panel_id == 'species_distribution':
            return dashboard_data.get('species_counts', {})
        
        elif panel_id == 'fitness_landscape':
            return {
                'fitness_distribution': dashboard_data.get('fitness_distribution', [])
            }
        
        elif panel_id == 'cultural_knowledge':
            return {
                'total_cultural_knowledge': dashboard_data.get('total_cultural_knowledge', 0),
                'avg_cultural_knowledge': dashboard_data.get('avg_cultural_knowledge', 0)
            }
        
        elif panel_id == 'environmental_health':
            return {
                'global_health': dashboard_data.get('global_health', 0),
                'avg_resource_level': dashboard_data.get('avg_resource_level', 0),
                'environmental_complexity': dashboard_data.get('environmental_complexity', 0)
            }
        
        elif panel_id == 'emergence_events':
            return {
                'recent_events': dashboard_data.get('recent_emergence_events', 0),
                'event_types': len(set(dashboard_data.get('emergence_event_types', [])))
            }
        
        elif panel_id == 'phase_space':
            return {
                'avg_intelligence': dashboard_data.get('avg_intelligence', 0),
                'avg_sociability': dashboard_data.get('avg_sociability', 0)
            }
        
        elif panel_id == 'information_flow':
            # Create dummy information flow matrix
            size = min(10, dashboard_data.get('population_size', 1))
            return {
                'adjacency_matrix': np.random.random((size, size)).tolist()
            }
        
        elif panel_id == 'evolution_metrics':
            return {
                'avg_fitness': dashboard_data.get('avg_fitness', 0),
                'fitness_variance': dashboard_data.get('fitness_variance', 0),
                'species_diversity': len(dashboard_data.get('species_counts', {}))
            }
        
        return {}
    
    def _toggle_pause(self, event):
        """Toggle dashboard pause state"""
        self.is_paused = not self.is_paused
        self.widgets['pause_button'].label.set_text('Resume' if self.is_paused else 'Pause')
    
    def _export_data(self, event):
        """Export current dashboard data"""
        if not os.path.exists(self.export_directory):
            os.makedirs(self.export_directory)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export raw data
        data_filename = os.path.join(self.export_directory, f"simulation_data_{timestamp}.json")
        with open(data_filename, 'w') as f:
            # Convert deque to list for JSON serialization
            export_data = {
                'simulation_data': list(self.data_history),
                'panel_summaries': {pid: panel.get_latest_summary() 
                                  for pid, panel in self.panels.items()},
                'export_timestamp': timestamp,
                'simulation_step': self.data_history[-1]['timestamp'] if self.data_history else 0
            }
            json.dump(export_data, f, indent=2, default=str)
        
        # Export dashboard image
        image_filename = os.path.join(self.export_directory, f"dashboard_{timestamp}.png")
        self.fig.savefig(image_filename, dpi=150, bbox_inches='tight', 
                        facecolor='black', edgecolor='none')
        
        # Export pickle for complete state
        pickle_filename = os.path.join(self.export_directory, f"dashboard_state_{timestamp}.pkl")
        with open(pickle_filename, 'wb') as f:
            pickle.dump({
                'data_history': self.data_history,
                'panels': {pid: {'title': p.title, 'type': p.panel_type, 
                                'data': list(p.data_history)} 
                          for pid, p in self.panels.items()}
            }, f)
        
        print(f"Dashboard exported to {self.export_directory}")
    
    def _toggle_auto_export(self, label):
        """Toggle auto-export feature"""
        self.auto_export = not self.auto_export
    
    def _update_export_interval(self, val):
        """Update export interval"""
        self.export_interval = int(val)
    
    def _reset_dashboard(self, event):
        """Reset dashboard data"""
        for panel in self.panels.values():
            panel.data_history.clear()
        self.data_history.clear()
        self.last_export_step = 0
        print("Dashboard reset")
    
    def add_custom_panel(self, panel_id: str, title: str, panel_type: str, 
                        position: Tuple[int, int]):
        """Add custom panel to dashboard"""
        panel = MetricsPanel(panel_id, title, panel_type)
        self.panels[panel_id] = panel
        
        # Add axis at specified position
        row, col = position
        if row < 4 and col < 4:  # Within grid bounds
            self.axes[panel_id] = self.fig.add_subplot(self.gs[row, col])
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary"""
        if not self.data_history:
            return {}
        
        latest_data = self.data_history[-1]
        
        summary = {
            'current_step': latest_data.get('timestamp', 0),
            'total_data_points': len(self.data_history),
            'active_panels': len([p for p in self.panels.values() if p.is_active]),
            'latest_metrics': {}
        }
        
        # Extract key metrics
        key_metrics = [
            'population_size', 'avg_intelligence', 'avg_energy', 
            'num_communities', 'total_signals', 'total_cultural_knowledge'
        ]
        
        for metric in key_metrics:
            if metric in latest_data:
                summary['latest_metrics'][metric] = latest_data[metric]
        
        return summary
    
    def show(self):
        """Show the dashboard"""
        plt.show()


def create_analytics_dashboard(simulation, config) -> AnalyticsDashboard:
    """Create and return analytics dashboard"""
    return AnalyticsDashboard(simulation, config)