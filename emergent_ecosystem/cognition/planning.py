"""
Forward planning and scenario simulation.

This module implements planning algorithms, mental simulation of future scenarios,
goal formation and decomposition, and plan execution monitoring.
"""

import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Union


class PlanningModule:
    """Forward planning and scenario simulation system"""
    
    def __init__(self, planning_horizon: int = 20, max_plan_length: int = 10):
        self.planning_horizon = planning_horizon
        self.max_plan_length = max_plan_length
        self.current_plan = []
        self.plan_execution_index = 0
        
        # Planning state
        self.scenario_cache = {}
        self.success_history = deque(maxlen=50)
        self.planning_confidence = 0.5
        
        # Goal system
        self.current_goals = []
        self.goal_hierarchy = {}
        self.goal_priorities = {}
        
        # Learning parameters
        self.plan_learning_rate = 0.1
        self.exploration_rate = 0.2
        self.plan_adaptation_rate = 0.05
        
        # Planning strategies
        self.planning_strategies = ['greedy', 'lookahead', 'monte_carlo', 'heuristic']
        self.current_strategy = 'lookahead'
        self.strategy_performance = {strategy: 0.5 for strategy in self.planning_strategies}
    
    def create_plan(self, current_state: Dict[str, Any], goal: str, 
                   environment_model: Dict[str, Any], time_horizon: int = None) -> List[Tuple[str, Any]]:
        """Create action plan using forward simulation"""
        if time_horizon is None:
            time_horizon = self.planning_horizon
        
        # Clear current plan
        self.current_plan = []
        self.plan_execution_index = 0
        
        # Select planning strategy based on situation
        strategy = self._select_planning_strategy(current_state, goal, environment_model)
        
        if strategy == 'greedy':
            plan = self._greedy_planning(current_state, goal, environment_model)
        elif strategy == 'lookahead':
            plan = self._lookahead_planning(current_state, goal, environment_model, time_horizon)
        elif strategy == 'monte_carlo':
            plan = self._monte_carlo_planning(current_state, goal, environment_model, time_horizon)
        else:  # heuristic
            plan = self._heuristic_planning(current_state, goal, environment_model)
        
        self.current_plan = plan
        return plan
    
    def _select_planning_strategy(self, current_state: Dict[str, Any], goal: str, 
                                environment_model: Dict[str, Any]) -> str:
        """Select the best planning strategy for the current situation"""
        # Assess situation complexity
        complexity_factors = {
            'environmental_complexity': len(environment_model.get('obstacles', [])) / 10.0,
            'goal_difficulty': self._assess_goal_difficulty(goal, current_state),
            'uncertainty_level': 1.0 - self.planning_confidence,
            'time_pressure': current_state.get('urgency', 0.0)
        }
        
        overall_complexity = np.mean(list(complexity_factors.values()))
        
        # Select strategy based on complexity and performance history
        if overall_complexity < 0.3:
            # Simple situation - use greedy
            return 'greedy'
        elif overall_complexity < 0.6:
            # Moderate complexity - use lookahead
            return 'lookahead'
        elif self.strategy_performance['monte_carlo'] > 0.6:
            # Complex situation with good Monte Carlo performance
            return 'monte_carlo'
        else:
            # Fallback to heuristic
            return 'heuristic'
    
    def _greedy_planning(self, current_state: Dict[str, Any], goal: str, 
                        environment_model: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Simple greedy planning - choose best immediate action"""
        plan = []
        
        if goal == 'find_food':
            food_locations = environment_model.get('food_locations', [])
            if food_locations:
                closest_food = min(food_locations, key=lambda loc: loc.get('distance', float('inf')))
                plan = [('move_to', closest_food['position'])]
        
        elif goal == 'avoid_danger':
            danger_locations = environment_model.get('danger_locations', [])
            if danger_locations:
                # Move away from closest danger
                closest_danger = min(danger_locations, key=lambda loc: loc.get('distance', float('inf')))
                plan = [('move_away', closest_danger['position'])]
        
        elif goal == 'socialize':
            social_opportunities = environment_model.get('social_opportunities', [])
            if social_opportunities:
                best_opportunity = max(social_opportunities, 
                                     key=lambda opp: opp.get('relationship_strength', 0))
                plan = [('approach', best_opportunity['individual'])]
        
        elif goal == 'explore':
            # Random exploration
            current_pos = current_state.get('position', (0, 0))
            explore_target = (
                current_pos[0] + random.uniform(-100, 100),
                current_pos[1] + random.uniform(-100, 100)
            )
            plan = [('move_to', explore_target)]
        
        return plan
    
    def _lookahead_planning(self, current_state: Dict[str, Any], goal: str, 
                           environment_model: Dict[str, Any], horizon: int) -> List[Tuple[str, Any]]:
        """Lookahead planning with state prediction"""
        plan = []
        simulated_state = current_state.copy()
        
        for step in range(min(horizon, self.max_plan_length)):
            # Generate possible actions
            possible_actions = self._generate_possible_actions(simulated_state, goal, environment_model)
            
            if not possible_actions:
                break
            
            # Evaluate each action by simulating forward
            best_action = None
            best_value = -float('inf')
            
            for action in possible_actions:
                # Simulate action outcome
                future_state = self._simulate_action(simulated_state, action, environment_model)
                action_value = self._evaluate_state(future_state, goal, step + 1, horizon)
                
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            
            if best_action:
                plan.append(best_action)
                simulated_state = self._simulate_action(simulated_state, best_action, environment_model)
            
            # Early termination if goal likely achieved
            if self._is_goal_achieved(simulated_state, goal):
                break
        
        return plan
    
    def _monte_carlo_planning(self, current_state: Dict[str, Any], goal: str, 
                             environment_model: Dict[str, Any], horizon: int) -> List[Tuple[str, Any]]:
        """Monte Carlo tree search planning"""
        num_simulations = 20
        action_values = {}
        
        # Generate possible first actions
        possible_actions = self._generate_possible_actions(current_state, goal, environment_model)
        
        if not possible_actions:
            return []
        
        # Run multiple simulations for each action
        for action in possible_actions:
            total_value = 0
            
            for _ in range(num_simulations):
                # Simulate entire trajectory
                trajectory_value = self._simulate_trajectory(current_state, action, goal, 
                                                           environment_model, horizon)
                total_value += trajectory_value
            
            action_values[action] = total_value / num_simulations
        
        # Select best action and build plan
        if action_values:
            best_action = max(action_values.items(), key=lambda x: x[1])[0]
            
            # Build multi-step plan using the best first action
            plan = [best_action]
            
            # Add follow-up actions using simpler planning
            simulated_state = self._simulate_action(current_state, best_action, environment_model)
            follow_up_plan = self._greedy_planning(simulated_state, goal, environment_model)
            plan.extend(follow_up_plan[:3])  # Limit follow-up length
            
            return plan
        
        return []
    
    def _heuristic_planning(self, current_state: Dict[str, Any], goal: str, 
                           environment_model: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Heuristic-based planning using domain knowledge"""
        plan = []
        current_pos = current_state.get('position', (0, 0))
        energy = current_state.get('energy', 50)
        
        if goal == 'find_food':
            # Multi-step food finding strategy
            food_locations = environment_model.get('food_locations', [])
            if food_locations:
                # Sort by value/distance ratio
                food_priorities = []
                for food in food_locations:
                    distance = food.get('distance', 1.0)
                    quality = food.get('quality', 1.0)
                    priority = quality / (distance + 1.0)
                    food_priorities.append((priority, food))
                
                food_priorities.sort(reverse=True)
                
                # Plan route to best food sources
                for priority, food in food_priorities[:3]:
                    plan.append(('move_to', food['position']))
                    if len(plan) >= self.max_plan_length:
                        break
        
        elif goal == 'avoid_danger':
            # Complex avoidance strategy
            danger_locations = environment_model.get('danger_locations', [])
            safe_locations = environment_model.get('safe_locations', [])
            
            if danger_locations:
                # Find safe refuge
                if safe_locations:
                    closest_safe = min(safe_locations, key=lambda loc: loc.get('distance', float('inf')))
                    plan = [('move_to', closest_safe['position'])]
                else:
                    # Create multiple evasion moves
                    for danger in danger_locations[:2]:
                        plan.append(('move_away', danger['position']))
        
        elif goal == 'socialize':
            # Social approach strategy
            social_opportunities = environment_model.get('social_opportunities', [])
            if social_opportunities:
                # Plan social interaction sequence
                for opportunity in social_opportunities[:2]:
                    plan.append(('approach', opportunity['individual']))
                    plan.append(('communicate', None))
        
        # Add energy management if low energy
        if energy < 30 and goal != 'find_food':
            food_locations = environment_model.get('food_locations', [])
            if food_locations:
                closest_food = min(food_locations, key=lambda loc: loc.get('distance', float('inf')))
                plan.insert(0, ('move_to', closest_food['position']))
        
        return plan[:self.max_plan_length]
    
    def _generate_possible_actions(self, state: Dict[str, Any], goal: str, 
                                  environment_model: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Generate possible actions for current state"""
        actions = []
        
        # Movement actions
        current_pos = state.get('position', (0, 0))
        
        # Add location-based actions
        for location_type in ['food_locations', 'safe_locations', 'social_opportunities']:
            locations = environment_model.get(location_type, [])
            for location in locations[:3]:  # Limit to closest/best options
                if location_type == 'social_opportunities':
                    actions.append(('approach', location['individual']))
                else:
                    actions.append(('move_to', location['position']))
        
        # Add danger avoidance actions
        danger_locations = environment_model.get('danger_locations', [])
        for danger in danger_locations:
            actions.append(('move_away', danger['position']))
        
        # Add general actions
        actions.extend([
            ('explore', None),
            ('rest', None),
            ('communicate', None)
        ])
        
        return actions
    
    def _simulate_action(self, state: Dict[str, Any], action: Tuple[str, Any], 
                        environment_model: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the result of taking an action"""
        new_state = state.copy()
        action_type, action_target = action
        
        current_pos = state.get('position', (0, 0))
        energy = state.get('energy', 50)
        
        if action_type == 'move_to' and action_target:
            # Simulate movement
            target_pos = action_target if isinstance(action_target, tuple) else (0, 0)
            distance = np.sqrt((target_pos[0] - current_pos[0])**2 + (target_pos[1] - current_pos[1])**2)
            
            # Update position (simplified)
            new_state['position'] = target_pos
            new_state['energy'] = max(0, energy - distance * 0.01)
            
        elif action_type == 'move_away' and action_target:
            # Simulate avoidance movement
            avoid_pos = action_target if isinstance(action_target, tuple) else (0, 0)
            # Move in opposite direction
            dx = current_pos[0] - avoid_pos[0]
            dy = current_pos[1] - avoid_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                # Normalize and move away
                new_x = current_pos[0] + (dx / distance) * 50
                new_y = current_pos[1] + (dy / distance) * 50
                new_state['position'] = (new_x, new_y)
            
            new_state['energy'] = max(0, energy - 2)
            
        elif action_type == 'rest':
            # Simulate resting
            new_state['energy'] = min(100, energy + 5)
            
        elif action_type == 'explore':
            # Simulate exploration
            new_state['energy'] = max(0, energy - 1)
            new_state['exploration_bonus'] = state.get('exploration_bonus', 0) + 1
        
        return new_state
    
    def _simulate_trajectory(self, initial_state: Dict[str, Any], first_action: Tuple[str, Any], 
                            goal: str, environment_model: Dict[str, Any], horizon: int) -> float:
        """Simulate a complete trajectory and return value"""
        state = initial_state.copy()
        total_value = 0
        
        # Apply first action
        state = self._simulate_action(state, first_action, environment_model)
        
        # Random rollout for remaining steps
        for step in range(1, horizon):
            if self._is_goal_achieved(state, goal):
                # Bonus for early goal achievement
                total_value += (horizon - step) * 2
                break
            
            # Choose random action for rollout
            possible_actions = self._generate_possible_actions(state, goal, environment_model)
            if possible_actions:
                action = random.choice(possible_actions)
                state = self._simulate_action(state, action, environment_model)
                total_value += self._evaluate_state(state, goal, step, horizon)
        
        return total_value
    
    def _evaluate_state(self, state: Dict[str, Any], goal: str, step: int, horizon: int) -> float:
        """Evaluate how good a state is for achieving the goal"""
        value = 0
        energy = state.get('energy', 50)
        position = state.get('position', (0, 0))
        
        # Energy bonus/penalty
        if energy > 70:
            value += 2
        elif energy < 20:
            value -= 5
        
        # Goal-specific evaluation
        if goal == 'find_food':
            # Favor states that lead toward food
            value += energy * 0.1  # Value maintaining energy
            
        elif goal == 'avoid_danger':
            # Heavily favor safe states
            value += 10 if energy > 50 else 5
            
        elif goal == 'socialize':
            # Value social opportunities
            social_bonus = state.get('social_bonus', 0)
            value += social_bonus * 3
            
        elif goal == 'explore':
            # Value exploration progress
            exploration_bonus = state.get('exploration_bonus', 0)
            value += exploration_bonus * 2
        
        # Temporal discounting
        discount_factor = 0.9 ** step
        value *= discount_factor
        
        return value
    
    def _is_goal_achieved(self, state: Dict[str, Any], goal: str) -> bool:
        """Check if goal is likely achieved in this state"""
        energy = state.get('energy', 50)
        
        if goal == 'find_food':
            return energy > 80  # High energy suggests food found
        elif goal == 'avoid_danger':
            return energy > 60  # Maintained energy suggests safety
        elif goal == 'socialize':
            return state.get('social_bonus', 0) > 0
        elif goal == 'explore':
            return state.get('exploration_bonus', 0) > 3
        
        return False
    
    def _assess_goal_difficulty(self, goal: str, current_state: Dict[str, Any]) -> float:
        """Assess how difficult a goal is to achieve"""
        energy = current_state.get('energy', 50)
        
        base_difficulty = {
            'find_food': 0.3,
            'avoid_danger': 0.7,
            'socialize': 0.4,
            'explore': 0.2
        }.get(goal, 0.5)
        
        # Adjust based on current state
        if energy < 30:
            base_difficulty += 0.3  # Everything harder when low energy
        
        return np.clip(base_difficulty, 0.0, 1.0)
    
    def execute_plan_step(self) -> Optional[Tuple[str, Any]]:
        """Execute the next step in the current plan"""
        if self.plan_execution_index < len(self.current_plan):
            action = self.current_plan[self.plan_execution_index]
            self.plan_execution_index += 1
            return action
        return None
    
    def evaluate_plan_success(self, outcome: bool, reward: float = 0.0):
        """Learn from plan execution results"""
        self.success_history.append({
            'outcome': outcome,
            'reward': reward,
            'strategy': self.current_strategy,
            'plan_length': len(self.current_plan)
        })
        
        # Update planning confidence
        recent_successes = [entry['outcome'] for entry in list(self.success_history)[-10:]]
        if recent_successes:
            success_rate = sum(recent_successes) / len(recent_successes)
            self.planning_confidence = 0.8 * self.planning_confidence + 0.2 * success_rate
        
        # Update strategy performance
        if self.current_strategy in self.strategy_performance:
            old_performance = self.strategy_performance[self.current_strategy]
            performance_update = self.plan_learning_rate * (1.0 if outcome else 0.0)
            new_performance = old_performance + performance_update
            self.strategy_performance[self.current_strategy] = np.clip(new_performance, 0.0, 1.0)
    
    def get_planning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive planning metrics"""
        recent_outcomes = [entry['outcome'] for entry in list(self.success_history)[-20:]]
        success_rate = np.mean(recent_outcomes) if recent_outcomes else 0.5
        
        avg_plan_length = np.mean([entry['plan_length'] for entry in self.success_history]) if self.success_history else 0
        
        return {
            'planning_confidence': self.planning_confidence,
            'success_rate': success_rate,
            'current_strategy': self.current_strategy,
            'avg_plan_length': avg_plan_length,
            'strategy_performance': self.strategy_performance.copy(),
            'plan_progress': self.plan_execution_index / max(1, len(self.current_plan))
        }
    
    def reset_plan(self):
        """Reset current plan execution"""
        self.current_plan = []
        self.plan_execution_index = 0
    
    def has_active_plan(self) -> bool:
        """Check if there's an active plan being executed"""
        return self.plan_execution_index < len(self.current_plan)
    
    def get_remaining_plan(self) -> List[Tuple[str, Any]]:
        """Get remaining steps in current plan"""
        return self.current_plan[self.plan_execution_index:]
