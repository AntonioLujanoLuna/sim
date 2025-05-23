"""
Metacognition and self-awareness systems.

This module implements self-monitoring, self-awareness, metacognitive knowledge,
learning-to-learn mechanisms, and theory of mind capabilities.
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any


class MetacognitionModule:
    """Self-awareness and meta-learning system"""
    
    def __init__(self, learning_rate: float = 0.1):
        # Self-model components
        self.self_model = {
            'strengths': defaultdict(float),
            'weaknesses': defaultdict(float),
            'learning_rate': learning_rate,
            'confidence': 0.5,
            'self_efficacy': 0.5,
            'metacognitive_awareness': 0.3
        }
        
        # Strategy and performance tracking
        self.strategy_success = defaultdict(list)
        self.performance_history = deque(maxlen=100)
        self.learning_curves = defaultdict(list)
        
        # Confidence and uncertainty tracking
        self.confidence_history = deque(maxlen=50)
        self.uncertainty_estimates = {}
        self.calibration_data = []
        
        # Theory of mind
        self.other_models = {}  # Models of other agents' minds
        self.social_understanding = defaultdict(float)
        
        # Meta-learning parameters
        self.adaptation_rate = 0.05
        self.confidence_learning_rate = 0.08
        self.metacognitive_threshold = 0.6
        
    def update_self_model(self, action: str, outcome: bool, context: str, 
                         difficulty: float = 0.5, reward: float = 0.0):
        """Update self-understanding based on experience"""
        # Track performance in different contexts
        performance_key = f"{action}_{context}"
        
        if outcome:
            self.self_model['strengths'][context] += 0.1
            if self.self_model['weaknesses'][context] > 0:
                self.self_model['weaknesses'][context] = max(0, self.self_model['weaknesses'][context] - 0.05)
            
            # Confidence boost for successful actions
            confidence_gain = 0.02 * (1 + reward)
            self.self_model['confidence'] = min(1.0, self.self_model['confidence'] + confidence_gain)
            
        else:
            self.self_model['weaknesses'][context] += 0.1
            if self.self_model['strengths'][context] > 0:
                self.self_model['strengths'][context] = max(0, self.self_model['strengths'][context] - 0.05)
            
            # Confidence reduction for failures
            confidence_loss = 0.02 * difficulty
            self.self_model['confidence'] = max(0.0, self.self_model['confidence'] - confidence_loss)
        
        # Update strategy success tracking
        self.strategy_success[action].append({
            'outcome': outcome,
            'context': context,
            'difficulty': difficulty,
            'reward': reward,
            'confidence_before': self.confidence_history[-1] if self.confidence_history else 0.5
        })
        
        # Limit strategy history
        if len(self.strategy_success[action]) > 20:
            self.strategy_success[action] = self.strategy_success[action][-20:]
        
        # Record overall performance
        self.performance_history.append({
            'action': action,
            'outcome': outcome,
            'context': context,
            'difficulty': difficulty,
            'reward': reward
        })
        
        # Update metacognitive awareness
        self._update_metacognitive_awareness()
    
    def _update_metacognitive_awareness(self):
        """Update awareness of own cognitive processes"""
        if len(self.performance_history) < 10:
            return
        
        # Analyze recent performance patterns
        recent_performance = list(self.performance_history)[-20:]
        
        # Calculate performance variability
        outcomes = [p['outcome'] for p in recent_performance]
        performance_variability = np.std(outcomes) if len(outcomes) > 1 else 0
        
        # Calculate context sensitivity
        context_performance = defaultdict(list)
        for p in recent_performance:
            context_performance[p['context']].append(p['outcome'])
        
        context_differences = []
        if len(context_performance) > 1:
            context_means = [np.mean(outcomes) for outcomes in context_performance.values() if outcomes]
            if len(context_means) > 1:
                context_differences = [abs(m1 - m2) for i, m1 in enumerate(context_means) 
                                     for m2 in context_means[i+1:]]
        
        context_sensitivity = np.mean(context_differences) if context_differences else 0
        
        # Update metacognitive awareness based on self-knowledge
        awareness_factors = [
            performance_variability,  # Awareness of own variability
            context_sensitivity,      # Awareness of context effects
            len(self.strategy_success) / 10.0,  # Strategy repertoire awareness
        ]
        
        new_awareness = min(1.0, np.mean(awareness_factors))
        self.self_model['metacognitive_awareness'] = (
            0.9 * self.self_model['metacognitive_awareness'] + 
            0.1 * new_awareness
        )
    
    def adapt_learning_rate(self):
        """Adapt learning rate based on performance patterns"""
        if len(self.performance_history) < 10:
            return
        
        # Analyze recent success rate
        recent_outcomes = [p['outcome'] for p in list(self.performance_history)[-10:]]
        recent_success_rate = np.mean(recent_outcomes)
        
        # Analyze learning trend
        if len(self.performance_history) >= 20:
            early_outcomes = [p['outcome'] for p in list(self.performance_history)[-20:-10]]
            late_outcomes = [p['outcome'] for p in list(self.performance_history)[-10:]]
            
            early_success = np.mean(early_outcomes)
            late_success = np.mean(late_outcomes)
            learning_trend = late_success - early_success
        else:
            learning_trend = 0
        
        # Adapt learning rate
        old_lr = self.self_model['learning_rate']
        
        if recent_success_rate > 0.7 and learning_trend > 0:
            # Doing well and improving - slow down learning
            new_lr = old_lr * 0.95
        elif recent_success_rate < 0.3 or learning_trend < -0.2:
            # Doing poorly or getting worse - speed up learning
            new_lr = old_lr * 1.05
        else:
            # Stable performance - slight adjustment toward optimal rate
            optimal_lr = 0.1
            new_lr = old_lr * 0.98 + optimal_lr * 0.02
        
        self.self_model['learning_rate'] = np.clip(new_lr, 0.01, 0.5)
    
    def assess_confidence(self, action: str, context: str) -> float:
        """Assess confidence for taking an action in a context"""
        # Base confidence from self-model
        base_confidence = self.self_model['confidence']
        
        # Adjust based on strengths/weaknesses in this context
        strength_bonus = self.self_model['strengths'].get(context, 0) * 0.3
        weakness_penalty = self.self_model['weaknesses'].get(context, 0) * 0.2
        
        # Adjust based on action-specific experience
        action_experience = self.strategy_success.get(action, [])
        if action_experience:
            action_success_rate = np.mean([exp['outcome'] for exp in action_experience])
            action_confidence = action_success_rate * 0.4
        else:
            action_confidence = 0.1  # Low confidence for untried actions
        
        # Combine factors
        total_confidence = (base_confidence + strength_bonus - weakness_penalty + action_confidence) / 2
        
        # Apply metacognitive awareness - higher awareness allows more accurate confidence
        if self.self_model['metacognitive_awareness'] > self.metacognitive_threshold:
            # More accurate confidence assessment
            confidence = np.clip(total_confidence, 0.0, 1.0)
        else:
            # Less accurate - add noise and bias toward overconfidence
            noise = np.random.normal(0, 0.1)
            bias = 0.1  # Slight overconfidence bias
            confidence = np.clip(total_confidence + noise + bias, 0.0, 1.0)
        
        # Record confidence for calibration
        self.confidence_history.append(confidence)
        
        return confidence
    
    def update_confidence_calibration(self, predicted_confidence: float, actual_outcome: bool):
        """Update confidence calibration based on prediction accuracy"""
        # Record calibration data
        self.calibration_data.append({
            'predicted_confidence': predicted_confidence,
            'actual_outcome': actual_outcome
        })
        
        # Limit calibration history
        if len(self.calibration_data) > 100:
            self.calibration_data = self.calibration_data[-100:]
        
        # Update confidence learning if enough data
        if len(self.calibration_data) >= 10:
            # Calculate calibration error
            recent_data = self.calibration_data[-10:]
            predicted_probs = [d['predicted_confidence'] for d in recent_data]
            actual_outcomes = [1.0 if d['actual_outcome'] else 0.0 for d in recent_data]
            
            # Simple calibration error
            calibration_error = np.mean(np.abs(np.array(predicted_probs) - np.array(actual_outcomes)))
            
            # Adjust confidence parameters based on calibration
            if calibration_error > 0.3:
                # Poor calibration - increase metacognitive awareness
                awareness_boost = 0.02
                self.self_model['metacognitive_awareness'] = min(1.0, 
                    self.self_model['metacognitive_awareness'] + awareness_boost)
    
    def model_other_mind(self, other_id: int, observed_action: str, context: str, 
                        outcome: bool, other_traits: Dict[str, float] = None):
        """Model another agent's mental state and capabilities"""
        if other_id not in self.other_models:
            self.other_models[other_id] = {
                'estimated_strengths': defaultdict(float),
                'estimated_weaknesses': defaultdict(float),
                'estimated_confidence': 0.5,
                'interaction_history': deque(maxlen=30),
                'cooperation_tendency': 0.5,
                'predictability': 0.5,
                'trustworthiness': 0.5
            }
        
        other_model = self.other_models[other_id]
        
        # Update model based on observed outcome
        if outcome:
            other_model['estimated_strengths'][context] += 0.15
            other_model['estimated_confidence'] = min(1.0, other_model['estimated_confidence'] + 0.05)
        else:
            other_model['estimated_weaknesses'][context] += 0.15
            other_model['estimated_confidence'] = max(0.0, other_model['estimated_confidence'] - 0.05)
        
        # Record interaction
        other_model['interaction_history'].append({
            'action': observed_action,
            'context': context,
            'outcome': outcome,
            'my_prediction': None  # Could add prediction tracking
        })
        
        # Update social understanding metrics
        if other_traits:
            # Learn about individual differences
            if 'cooperation' in other_traits:
                other_model['cooperation_tendency'] = (
                    0.8 * other_model['cooperation_tendency'] + 
                    0.2 * other_traits['cooperation']
                )
        
        # Update predictability based on action consistency
        if len(other_model['interaction_history']) > 5:
            recent_actions = [h['action'] for h in list(other_model['interaction_history'])[-5:]]
            action_variety = len(set(recent_actions))
            # Lower variety suggests higher predictability
            other_model['predictability'] = max(0.1, 1.0 - (action_variety / 5.0))
    
    def predict_other_behavior(self, other_id: int, context: str) -> Dict[str, float]:
        """Predict another agent's likely behavior"""
        if other_id not in self.other_models:
            return {'confidence': 0.1, 'predicted_success': 0.5, 'cooperation_likelihood': 0.5}
        
        other_model = self.other_models[other_id]
        
        # Predict success based on estimated strengths/weaknesses
        context_strength = other_model['estimated_strengths'].get(context, 0)
        context_weakness = other_model['estimated_weaknesses'].get(context, 0)
        predicted_success = (context_strength - context_weakness + 0.5) / 2
        
        # Prediction confidence based on interaction history
        interaction_count = len(other_model['interaction_history'])
        prediction_confidence = min(0.9, interaction_count / 20.0)
        
        # Cooperation likelihood
        cooperation_likelihood = other_model['cooperation_tendency']
        
        return {
            'confidence': prediction_confidence,
            'predicted_success': np.clip(predicted_success, 0.0, 1.0),
            'cooperation_likelihood': cooperation_likelihood,
            'predictability': other_model['predictability'],
            'trustworthiness': other_model['trustworthiness']
        }
    
    def reflect_on_performance(self) -> Dict[str, Any]:
        """Conduct metacognitive reflection on recent performance"""
        if len(self.performance_history) < 5:
            return {'reflection_quality': 'insufficient_data'}
        
        recent_performance = list(self.performance_history)[-20:]
        
        # Analyze performance patterns
        context_performance = defaultdict(list)
        action_performance = defaultdict(list)
        
        for perf in recent_performance:
            context_performance[perf['context']].append(perf['outcome'])
            action_performance[perf['action']].append(perf['outcome'])
        
        # Identify strengths and weaknesses
        identified_strengths = []
        identified_weaknesses = []
        
        for context, outcomes in context_performance.items():
            if len(outcomes) >= 3:
                success_rate = np.mean(outcomes)
                if success_rate > 0.7:
                    identified_strengths.append(context)
                elif success_rate < 0.3:
                    identified_weaknesses.append(context)
        
        # Identify effective strategies
        effective_strategies = []
        ineffective_strategies = []
        
        for action, outcomes in action_performance.items():
            if len(outcomes) >= 3:
                success_rate = np.mean(outcomes)
                if success_rate > 0.7:
                    effective_strategies.append(action)
                elif success_rate < 0.3:
                    ineffective_strategies.append(action)
        
        # Learning insights
        insights = []
        
        if identified_strengths:
            insights.append(f"Strong performance in: {', '.join(identified_strengths)}")
        
        if identified_weaknesses:
            insights.append(f"Need improvement in: {', '.join(identified_weaknesses)}")
        
        if effective_strategies:
            insights.append(f"Effective strategies: {', '.join(effective_strategies)}")
        
        if ineffective_strategies:
            insights.append(f"Ineffective strategies: {', '.join(ineffective_strategies)}")
        
        # Overall assessment
        overall_success = np.mean([p['outcome'] for p in recent_performance])
        confidence_accuracy = self._assess_confidence_accuracy()
        
        reflection = {
            'overall_performance': overall_success,
            'confidence_accuracy': confidence_accuracy,
            'identified_strengths': identified_strengths,
            'identified_weaknesses': identified_weaknesses,
            'effective_strategies': effective_strategies,
            'ineffective_strategies': ineffective_strategies,
            'insights': insights,
            'metacognitive_awareness': self.self_model['metacognitive_awareness'],
            'reflection_quality': 'good' if len(insights) > 2 else 'basic'
        }
        
        return reflection
    
    def _assess_confidence_accuracy(self) -> float:
        """Assess how well calibrated confidence predictions are"""
        if len(self.calibration_data) < 10:
            return 0.5
        
        recent_calibration = self.calibration_data[-20:]
        predicted_probs = [d['predicted_confidence'] for d in recent_calibration]
        actual_outcomes = [1.0 if d['actual_outcome'] else 0.0 for d in recent_calibration]
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(np.array(predicted_probs) - np.array(actual_outcomes)))
        
        # Convert to accuracy score (lower error = higher accuracy)
        accuracy = max(0.0, 1.0 - mae * 2)
        return accuracy
    
    def get_metacognitive_state(self) -> Dict[str, Any]:
        """Get comprehensive metacognitive state"""
        # Calculate dynamic metrics
        recent_performance = np.mean([p['outcome'] for p in list(self.performance_history)[-10:]]) if self.performance_history else 0.5
        
        confidence_stability = 1.0 - np.std(list(self.confidence_history)[-10:]) if len(self.confidence_history) >= 10 else 0.5
        
        strategy_diversity = len(self.strategy_success)
        
        social_understanding = np.mean(list(self.social_understanding.values())) if self.social_understanding else 0.0
        
        return {
            'self_model': dict(self.self_model),
            'recent_performance': recent_performance,
            'confidence_stability': confidence_stability,
            'strategy_diversity': strategy_diversity,
            'social_understanding': social_understanding,
            'known_others': len(self.other_models),
            'confidence_accuracy': self._assess_confidence_accuracy(),
            'total_experiences': len(self.performance_history)
        }
    
    def reset_metacognition(self):
        """Reset metacognitive state (e.g., for new learning phase)"""
        # Keep core self-model but reset recent tracking
        self.performance_history.clear()
        self.confidence_history.clear()
        self.calibration_data.clear()
        
        # Reset learning rate to default
        self.self_model['learning_rate'] = 0.1
        self.self_model['confidence'] = 0.5
