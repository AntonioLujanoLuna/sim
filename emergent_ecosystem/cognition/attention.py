"""
Attention and perception filtering systems.

This module implements selective attention mechanisms, perceptual filtering,
and attention learning for cognitive agents.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any


class AttentionModule:
    """Attention and perception filtering system"""
    
    def __init__(self, attention_span: int = 5):
        # Attention weights for different stimulus types
        self.attention_weights = {
            'social': 0.3,
            'environmental': 0.3,
            'danger': 0.4,
            'food': 0.35,
            'communication': 0.25,
            'shelter': 0.2
        }
        
        self.current_focus = None
        self.attention_history = deque(maxlen=attention_span * 4)
        self.attention_span = attention_span
        
        # Attention learning parameters
        self.attention_learning_rate = 0.05
        self.attention_adaptation_rate = 0.01
        
        # Perceptual filters
        self.noise_threshold = 0.1
        self.salience_threshold = 0.2
        
        # Attention state
        self.attention_energy = 1.0
        self.cognitive_load = 0.0
        self.distraction_level = 0.0
        
    def update_attention(self, stimuli: Dict[str, float], context: str = None):
        """Update attention based on stimuli salience"""
        if not stimuli:
            self.current_focus = None
            return
        
        # Apply cognitive load effects
        effective_weights = self._apply_cognitive_load_effects()
        
        # Calculate weighted salience for each stimulus
        weighted_stimuli = {}
        for stimulus_type, intensity in stimuli.items():
            base_weight = effective_weights.get(stimulus_type, 0.1)
            context_modifier = self._get_context_modifier(stimulus_type, context)
            weighted_stimuli[stimulus_type] = intensity * base_weight * context_modifier
        
        # Add noise and apply threshold
        filtered_stimuli = {}
        for stimulus_type, weighted_intensity in weighted_stimuli.items():
            noise = np.random.normal(0, self.noise_threshold)
            final_intensity = weighted_intensity + noise
            
            if final_intensity > self.salience_threshold:
                filtered_stimuli[stimulus_type] = final_intensity
        
        # Select focus based on highest weighted salience
        if filtered_stimuli:
            max_stimulus = max(filtered_stimuli.items(), key=lambda x: x[1])
            self.current_focus = max_stimulus[0]
            focus_strength = max_stimulus[1]
        else:
            self.current_focus = None
            focus_strength = 0.0
        
        # Record attention history
        self.attention_history.append({
            'focus': self.current_focus,
            'strength': focus_strength,
            'all_stimuli': stimuli.copy(),
            'context': context,
            'cognitive_load': self.cognitive_load
        })
        
        # Update cognitive load and attention energy
        self._update_attention_state(len(stimuli), focus_strength)
    
    def _apply_cognitive_load_effects(self) -> Dict[str, float]:
        """Apply cognitive load effects to attention weights"""
        effective_weights = self.attention_weights.copy()
        
        # High cognitive load reduces ability to attend to complex stimuli
        if self.cognitive_load > 0.7:
            load_factor = 1.0 - (self.cognitive_load - 0.7) * 0.5
            for key in effective_weights:
                if key in ['social', 'communication']:
                    effective_weights[key] *= load_factor
        
        # Low attention energy reduces overall sensitivity
        if self.attention_energy < 0.5:
            energy_factor = 0.5 + self.attention_energy * 0.5
            for key in effective_weights:
                effective_weights[key] *= energy_factor
        
        return effective_weights
    
    def _get_context_modifier(self, stimulus_type: str, context: str) -> float:
        """Get context-dependent modifier for stimulus attention"""
        if not context:
            return 1.0
        
        context_modifiers = {
            'find_food': {
                'food': 1.5,
                'environmental': 1.2,
                'social': 0.8
            },
            'avoid_danger': {
                'danger': 2.0,
                'social': 0.6,
                'environmental': 1.1
            },
            'socialize': {
                'social': 1.5,
                'communication': 1.8,
                'environmental': 0.7
            },
            'explore': {
                'environmental': 1.3,
                'social': 1.0,
                'danger': 1.2
            }
        }
        
        return context_modifiers.get(context, {}).get(stimulus_type, 1.0)
    
    def _update_attention_state(self, num_stimuli: int, focus_strength: float):
        """Update attention energy and cognitive load"""
        # Cognitive load increases with number of stimuli
        stimulus_load = min(1.0, num_stimuli / 10.0)
        self.cognitive_load = 0.8 * self.cognitive_load + 0.2 * stimulus_load
        
        # Attention energy decreases with use, recovers over time
        energy_consumption = 0.02 * focus_strength if focus_strength > 0 else -0.01
        self.attention_energy = np.clip(self.attention_energy - energy_consumption, 0.0, 1.0)
        
        # Distraction increases with cognitive load
        self.distraction_level = self.cognitive_load * (1.0 - self.attention_energy)
    
    def get_filtered_perception(self, raw_perception: Dict[str, Any]) -> Dict[str, Any]:
        """Filter perception based on current attention"""
        if not self.current_focus:
            # No focus - apply general attenuation
            filtered = {}
            for key, value in raw_perception.items():
                if isinstance(value, (int, float)):
                    filtered[key] = value * 0.5
                elif isinstance(value, list):
                    # Limit list length when unfocused
                    filtered[key] = value[:3] if len(value) > 3 else value
                else:
                    filtered[key] = value
            return filtered
        
        filtered = {}
        for key, value in raw_perception.items():
            if key == self.current_focus:
                # Amplify attended stimulus
                if isinstance(value, (int, float)):
                    amplification = 1.5 - self.distraction_level * 0.3
                    filtered[key] = value * amplification
                elif isinstance(value, list):
                    # More detailed processing for attended lists
                    filtered[key] = value
                else:
                    filtered[key] = value
            else:
                # Diminish unattended stimuli
                if isinstance(value, (int, float)):
                    attenuation = 0.5 + self.distraction_level * 0.3
                    filtered[key] = value * attenuation
                elif isinstance(value, list):
                    # Reduce detail for unattended lists
                    reduction_factor = max(1, len(value) // 2)
                    filtered[key] = value[::reduction_factor]
                else:
                    filtered[key] = value
        
        return filtered
    
    def learn_attention_patterns(self, outcome: bool, reward: float = 0.0):
        """Learn from attention outcomes to adapt weights"""
        if not self.attention_history:
            return
        
        recent_attention = self.attention_history[-1]
        focused_stimulus = recent_attention['focus']
        
        if focused_stimulus:
            # Update attention weight based on outcome
            learning_signal = self.attention_learning_rate
            if outcome:
                learning_signal *= (1.0 + reward)
            else:
                learning_signal *= -0.5
            
            # Adapt attention weight
            old_weight = self.attention_weights.get(focused_stimulus, 0.1)
            new_weight = old_weight + learning_signal
            self.attention_weights[focused_stimulus] = np.clip(new_weight, 0.05, 1.0)
            
            # Normalize weights to prevent runaway inflation
            total_weight = sum(self.attention_weights.values())
            if total_weight > 5.0:  # Arbitrary normalization threshold
                normalization_factor = 5.0 / total_weight
                for key in self.attention_weights:
                    self.attention_weights[key] *= normalization_factor
    
    def adapt_to_environment(self, environmental_complexity: float):
        """Adapt attention parameters based on environmental complexity"""
        # Adjust thresholds based on complexity
        if environmental_complexity > 0.7:
            # High complexity - be more selective
            self.salience_threshold = min(0.5, self.salience_threshold + self.attention_adaptation_rate)
            self.noise_threshold = max(0.05, self.noise_threshold - self.attention_adaptation_rate)
        else:
            # Low complexity - be more open to stimuli
            self.salience_threshold = max(0.1, self.salience_threshold - self.attention_adaptation_rate)
            self.noise_threshold = min(0.2, self.noise_threshold + self.attention_adaptation_rate)
    
    def get_attention_metrics(self) -> Dict[str, float]:
        """Get comprehensive attention metrics"""
        # Calculate attention stability
        if len(self.attention_history) > 1:
            focus_changes = sum(1 for i in range(1, len(self.attention_history))
                              if self.attention_history[i]['focus'] != self.attention_history[i-1]['focus'])
            attention_stability = 1.0 - (focus_changes / (len(self.attention_history) - 1))
        else:
            attention_stability = 1.0
        
        # Calculate average focus strength
        focus_strengths = [entry['strength'] for entry in self.attention_history if entry['strength'] > 0]
        avg_focus_strength = np.mean(focus_strengths) if focus_strengths else 0.0
        
        # Calculate attention efficiency (focus strength vs cognitive load)
        attention_efficiency = avg_focus_strength / (1.0 + self.cognitive_load)
        
        return {
            'current_focus': self.current_focus or 'none',
            'attention_energy': self.attention_energy,
            'cognitive_load': self.cognitive_load,
            'distraction_level': self.distraction_level,
            'attention_stability': attention_stability,
            'avg_focus_strength': avg_focus_strength,
            'attention_efficiency': attention_efficiency,
            'salience_threshold': self.salience_threshold
        }
    
    def reset_attention(self):
        """Reset attention state (e.g., after sleep or major event)"""
        self.attention_energy = 1.0
        self.cognitive_load = 0.0
        self.distraction_level = 0.0
        self.current_focus = None
    
    def get_attention_breakdown(self) -> Dict[str, float]:
        """Get breakdown of attention weights by stimulus type"""
        return self.attention_weights.copy()
    
    def set_attention_bias(self, stimulus_type: str, bias_strength: float):
        """Temporarily bias attention toward a specific stimulus type"""
        if stimulus_type in self.attention_weights:
            self.attention_weights[stimulus_type] *= (1.0 + bias_strength)
    
    def get_perceptual_priority_list(self, stimuli: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get prioritized list of stimuli based on attention weights"""
        priority_list = []
        
        for stimulus_type, intensity in stimuli.items():
            weight = self.attention_weights.get(stimulus_type, 0.1)
            priority_score = intensity * weight
            priority_list.append((stimulus_type, priority_score))
        
        # Sort by priority score (highest first)
        priority_list.sort(key=lambda x: x[1], reverse=True)
        return priority_list


class PerceptualFilter:
    """Advanced perceptual filtering with adaptation"""
    
    def __init__(self):
        self.filter_parameters = {
            'spatial_resolution': 1.0,
            'temporal_resolution': 1.0,
            'contrast_sensitivity': 1.0,
            'motion_sensitivity': 1.0,
            'social_sensitivity': 1.0
        }
        
        self.adaptation_history = deque(maxlen=100)
        self.filter_learning_rate = 0.02
    
    def apply_spatial_filter(self, spatial_data: List[Dict], focus_location: Tuple[float, float] = None) -> List[Dict]:
        """Apply spatial attention filtering"""
        if not focus_location:
            return spatial_data
        
        filtered_data = []
        focus_x, focus_y = focus_location
        
        for item in spatial_data:
            if 'x' in item and 'y' in item:
                distance = np.sqrt((item['x'] - focus_x)**2 + (item['y'] - focus_y)**2)
                attention_falloff = np.exp(-distance / (50 * self.filter_parameters['spatial_resolution']))
                
                # Create filtered version of item
                filtered_item = item.copy()
                
                # Reduce precision for distant items
                if attention_falloff < 0.5:
                    for key, value in filtered_item.items():
                        if isinstance(value, float) and key not in ['x', 'y']:
                            filtered_item[key] = round(value, 1)
                
                filtered_item['attention_weight'] = attention_falloff
                filtered_data.append(filtered_item)
        
        return filtered_data
    
    def apply_temporal_filter(self, temporal_data: List[Dict], current_time: int) -> List[Dict]:
        """Apply temporal attention filtering"""
        filtered_data = []
        temporal_window = 20 * self.filter_parameters['temporal_resolution']
        
        for item in temporal_data:
            if 'timestamp' in item:
                time_diff = abs(current_time - item['timestamp'])
                relevance = max(0.1, 1.0 - time_diff / temporal_window)
                
                filtered_item = item.copy()
                filtered_item['temporal_relevance'] = relevance
                filtered_data.append(filtered_item)
        
        return filtered_data
    
    def adapt_filters(self, feedback: Dict[str, float]):
        """Adapt filter parameters based on feedback"""
        for parameter, adjustment in feedback.items():
            if parameter in self.filter_parameters:
                old_value = self.filter_parameters[parameter]
                new_value = old_value + adjustment * self.filter_learning_rate
                self.filter_parameters[parameter] = np.clip(new_value, 0.1, 2.0)
        
        # Record adaptation
        self.adaptation_history.append({
            'parameters': self.filter_parameters.copy(),
            'feedback': feedback.copy()
        })
    
    def get_filter_state(self) -> Dict[str, float]:
        """Get current state of perceptual filters"""
        return self.filter_parameters.copy()
