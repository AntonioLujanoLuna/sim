"""
Emergence detection and phase transition analysis.

This module implements algorithms to detect emergent phenomena, phase transitions,
and critical points in the complex adaptive system.
"""

import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional


class EmergenceDetector:
    """Detects emergent phenomena and phase transitions"""
    
    def __init__(self, detection_window: int = 100):
        self.detection_window = detection_window
        self.emergence_events = []
        self.phase_transitions = []
        
        # Historical data for analysis
        self.population_history = deque(maxlen=detection_window * 2)
        self.communication_history = deque(maxlen=detection_window * 2)
        self.social_complexity_history = deque(maxlen=detection_window * 2)
        self.intelligence_history = deque(maxlen=detection_window * 2)
        self.cultural_history = deque(maxlen=detection_window * 2)
        
        # Detection thresholds
        self.communication_threshold = 10  # Average signals per individual
        self.social_complexity_threshold = 0.3  # Network density threshold
        self.intelligence_growth_threshold = 0.2  # Intelligence increase rate
        self.cultural_accumulation_threshold = 5  # Cultural knowledge per individual
        
        # Phase transition detection
        self.variance_threshold = 0.5  # Threshold for detecting phase transitions
        self.trend_threshold = 0.3  # Threshold for detecting trends
        
    def update(self, individuals: List, social_network, environment, time_step: int):
        """Update emergence detection with current state"""
        # Calculate current metrics
        current_metrics = self._calculate_metrics(individuals, social_network, environment)
        
        # Store historical data
        self._store_historical_data(current_metrics, time_step)
        
        # Detect emergence events
        self._detect_communication_emergence(current_metrics, time_step)
        self._detect_social_complexity_emergence(current_metrics, time_step)
        self._detect_intelligence_boom(current_metrics, time_step)
        self._detect_cultural_accumulation(current_metrics, time_step)
        self._detect_collective_behavior(current_metrics, time_step)
        
        # Detect phase transitions
        self._detect_phase_transitions(time_step)
    
    def _calculate_metrics(self, individuals: List, social_network, environment) -> Dict[str, float]:
        """Calculate current system metrics"""
        if not individuals:
            return {
                'population_size': 0,
                'avg_intelligence': 0,
                'communication_complexity': 0,
                'social_density': 0,
                'cultural_knowledge': 0,
                'species_diversity': 0,
                'network_clustering': 0,
                'environmental_complexity': 0
            }
        
        # Population metrics
        population_size = len(individuals)
        avg_intelligence = np.mean([ind.intelligence for ind in individuals])
        
        # Communication metrics
        total_signals = sum(len(ind.communication.signal_repertoire) for ind in individuals)
        active_signals = sum(len(ind.active_signals) for ind in individuals)
        communication_complexity = total_signals / population_size if population_size > 0 else 0
        
        # Social network metrics
        network_metrics = social_network.get_network_metrics()
        social_density = network_metrics.get('density', 0)
        network_clustering = network_metrics.get('clustering', 0)
        
        # Cultural metrics
        total_cultural_knowledge = sum(sum(ind.cultural_knowledge.values()) for ind in individuals)
        cultural_knowledge = total_cultural_knowledge / population_size if population_size > 0 else 0
        
        # Species diversity
        species_counts = defaultdict(int)
        for ind in individuals:
            species_counts[ind.species_name] += 1
        
        species_diversity = len(species_counts) / 4.0  # Normalized by max species (4)
        
        # Environmental complexity
        env_summary = environment.get_environmental_summary()
        environmental_complexity = env_summary.get('environmental_complexity', 0)
        
        return {
            'population_size': population_size,
            'avg_intelligence': avg_intelligence,
            'communication_complexity': communication_complexity,
            'social_density': social_density,
            'cultural_knowledge': cultural_knowledge,
            'species_diversity': species_diversity,
            'network_clustering': network_clustering,
            'environmental_complexity': environmental_complexity,
            'active_signals': active_signals
        }
    
    def _store_historical_data(self, metrics: Dict[str, float], time_step: int):
        """Store metrics in historical data structures"""
        self.population_history.append((time_step, metrics['population_size']))
        self.communication_history.append((time_step, metrics['communication_complexity']))
        self.social_complexity_history.append((time_step, metrics['social_density']))
        self.intelligence_history.append((time_step, metrics['avg_intelligence']))
        self.cultural_history.append((time_step, metrics['cultural_knowledge']))
    
    def _detect_communication_emergence(self, metrics: Dict[str, float], time_step: int):
        """Detect emergence of complex communication systems"""
        communication_complexity = metrics['communication_complexity']
        
        # Check if communication complexity exceeds threshold
        if communication_complexity > self.communication_threshold:
            # Check if this is a new emergence (not detected recently)
            recent_events = [e for e in self.emergence_events[-10:] 
                           if e['type'] == 'communication_emergence']
            
            if not recent_events:
                self.emergence_events.append({
                    'type': 'communication_emergence',
                    'time_step': time_step,
                    'complexity': communication_complexity,
                    'description': f'Complex communication system emerged with {communication_complexity:.2f} signals per individual'
                })
    
    def _detect_social_complexity_emergence(self, metrics: Dict[str, float], time_step: int):
        """Detect emergence of complex social structures"""
        social_density = metrics['social_density']
        network_clustering = metrics['network_clustering']
        
        # Combined social complexity metric
        social_complexity = (social_density + network_clustering) / 2
        
        if social_complexity > self.social_complexity_threshold:
            recent_events = [e for e in self.emergence_events[-10:] 
                           if e['type'] == 'social_complexity_emergence']
            
            if not recent_events:
                self.emergence_events.append({
                    'type': 'social_complexity_emergence',
                    'time_step': time_step,
                    'complexity': social_complexity,
                    'description': f'Complex social structure emerged with density {social_density:.3f} and clustering {network_clustering:.3f}'
                })
    
    def _detect_intelligence_boom(self, metrics: Dict[str, float], time_step: int):
        """Detect rapid intelligence evolution"""
        if len(self.intelligence_history) < 50:
            return
        
        # Calculate intelligence growth rate over recent history
        recent_intelligence = [entry[1] for entry in list(self.intelligence_history)[-50:]]
        early_intelligence = [entry[1] for entry in list(self.intelligence_history)[-100:-50]]
        
        if early_intelligence and recent_intelligence:
            early_avg = np.mean(early_intelligence)
            recent_avg = np.mean(recent_intelligence)
            
            if early_avg > 0:
                growth_rate = (recent_avg - early_avg) / early_avg
                
                if growth_rate > self.intelligence_growth_threshold:
                    recent_events = [e for e in self.emergence_events[-10:] 
                                   if e['type'] == 'intelligence_boom']
                    
                    if not recent_events:
                        self.emergence_events.append({
                            'type': 'intelligence_boom',
                            'time_step': time_step,
                            'growth_rate': growth_rate,
                            'description': f'Intelligence boom detected with {growth_rate*100:.1f}% growth rate'
                        })
    
    def _detect_cultural_accumulation(self, metrics: Dict[str, float], time_step: int):
        """Detect cultural knowledge accumulation"""
        cultural_knowledge = metrics['cultural_knowledge']
        
        if cultural_knowledge > self.cultural_accumulation_threshold:
            recent_events = [e for e in self.emergence_events[-20:] 
                           if e['type'] == 'cultural_accumulation']
            
            if not recent_events:
                self.emergence_events.append({
                    'type': 'cultural_accumulation',
                    'time_step': time_step,
                    'knowledge_level': cultural_knowledge,
                    'description': f'Significant cultural accumulation reached {cultural_knowledge:.2f} knowledge per individual'
                })
    
    def _detect_collective_behavior(self, metrics: Dict[str, float], time_step: int):
        """Detect emergence of collective behaviors"""
        # Look for synchronized patterns in communication
        active_signals = metrics['active_signals']
        population_size = metrics['population_size']
        
        if population_size > 0:
            signal_synchronization = active_signals / population_size
            
            # High synchronization indicates collective behavior
            if signal_synchronization > 0.8:  # 80% of population signaling
                recent_events = [e for e in self.emergence_events[-10:] 
                               if e['type'] == 'collective_behavior']
                
                if not recent_events:
                    self.emergence_events.append({
                        'type': 'collective_behavior',
                        'time_step': time_step,
                        'synchronization': signal_synchronization,
                        'description': f'Collective behavior detected with {signal_synchronization*100:.1f}% synchronization'
                    })
    
    def _detect_phase_transitions(self, time_step: int):
        """Detect phase transitions in system dynamics"""
        if len(self.population_history) < self.detection_window:
            return
        
        # Analyze variance in key metrics to detect phase transitions
        metrics_to_analyze = [
            ('population', self.population_history),
            ('communication', self.communication_history),
            ('social_complexity', self.social_complexity_history),
            ('intelligence', self.intelligence_history),
            ('cultural', self.cultural_history)
        ]
        
        for metric_name, history in metrics_to_analyze:
            if len(history) >= self.detection_window:
                self._analyze_phase_transition(metric_name, history, time_step)
    
    def _analyze_phase_transition(self, metric_name: str, history: deque, time_step: int):
        """Analyze a specific metric for phase transitions"""
        # Split history into two windows
        window_size = self.detection_window // 2
        recent_data = list(history)[-window_size:]
        older_data = list(history)[-self.detection_window:-window_size]
        
        if len(recent_data) < window_size or len(older_data) < window_size:
            return
        
        # Extract values
        recent_values = [entry[1] for entry in recent_data]
        older_values = [entry[1] for entry in older_data]
        
        # Calculate variance change
        recent_variance = np.var(recent_values)
        older_variance = np.var(older_values)
        
        # Calculate mean change
        recent_mean = np.mean(recent_values)
        older_mean = np.mean(older_values)
        
        # Detect phase transition based on variance and mean changes
        if older_variance > 0:
            variance_change = abs(recent_variance - older_variance) / older_variance
            
            if variance_change > self.variance_threshold:
                # Significant variance change detected
                if older_mean > 0:
                    mean_change = (recent_mean - older_mean) / older_mean
                else:
                    mean_change = 0
                
                transition_type = self._classify_transition(variance_change, mean_change)
                
                # Check if this transition was already detected recently
                recent_transitions = [t for t in self.phase_transitions[-5:] 
                                    if t['metric'] == metric_name and 
                                    abs(t['time_step'] - time_step) < 50]
                
                if not recent_transitions:
                    self.phase_transitions.append({
                        'type': transition_type,
                        'metric': metric_name,
                        'time_step': time_step,
                        'variance_change': variance_change,
                        'mean_change': mean_change,
                        'description': f'{transition_type} in {metric_name} detected'
                    })
    
    def _classify_transition(self, variance_change: float, mean_change: float) -> str:
        """Classify the type of phase transition"""
        if variance_change > 1.0:  # Very high variance change
            if abs(mean_change) > 0.5:
                return 'critical_transition'
            else:
                return 'fluctuation_increase'
        elif variance_change > 0.5:
            if mean_change > 0.3:
                return 'growth_phase'
            elif mean_change < -0.3:
                return 'decline_phase'
            else:
                return 'stability_change'
        else:
            return 'minor_transition'
    
    def get_recent_events(self, num_events: int = 10) -> List[Dict[str, Any]]:
        """Get recent emergence events"""
        return self.emergence_events[-num_events:]
    
    def get_recent_transitions(self, num_transitions: int = 5) -> List[Dict[str, Any]]:
        """Get recent phase transitions"""
        return self.phase_transitions[-num_transitions:]
    
    def get_emergence_summary(self) -> Dict[str, Any]:
        """Get summary of all detected emergence phenomena"""
        event_types = defaultdict(int)
        for event in self.emergence_events:
            event_types[event['type']] += 1
        
        transition_types = defaultdict(int)
        for transition in self.phase_transitions:
            transition_types[transition['type']] += 1
        
        return {
            'total_emergence_events': len(self.emergence_events),
            'total_phase_transitions': len(self.phase_transitions),
            'event_types': dict(event_types),
            'transition_types': dict(transition_types),
            'recent_events': self.get_recent_events(5),
            'recent_transitions': self.get_recent_transitions(3)
        }
    
    def detect_critical_points(self) -> List[Dict[str, Any]]:
        """Detect critical points where multiple emergence events coincide"""
        critical_points = []
        
        # Group events by time windows
        time_windows = defaultdict(list)
        window_size = 50  # Time steps
        
        for event in self.emergence_events:
            window = event['time_step'] // window_size
            time_windows[window].append(event)
        
        # Find windows with multiple events (critical points)
        for window, events in time_windows.items():
            if len(events) >= 2:  # Multiple emergence events in same window
                critical_points.append({
                    'time_window': (window * window_size, (window + 1) * window_size),
                    'event_count': len(events),
                    'events': events,
                    'criticality': len(events) / 5.0  # Normalized criticality score
                })
        
        return sorted(critical_points, key=lambda x: x['criticality'], reverse=True)
