"""
Comprehensive error handling and logging system for the simulation.

This module provides specific exception types, logging utilities, and
error recovery mechanisms to improve robustness and debugging.
"""

import logging
import traceback
import time
from typing import Any, Dict, Optional, Callable
from functools import wraps
import networkx as nx


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('EmergentEcosystem')


class SimulationError(Exception):
    """Base exception for simulation-related errors"""
    pass


class ConfigurationError(SimulationError):
    """Raised when there are configuration issues"""
    pass


class PerceptionError(SimulationError):
    """Raised when individual perception fails"""
    pass


class SocialNetworkError(SimulationError):
    """Raised when social network operations fail"""
    pass


class EnvironmentError(SimulationError):
    """Raised when environmental operations fail"""
    pass


class CognitionError(SimulationError):
    """Raised when cognitive operations fail"""
    pass


class MemoryError(SimulationError):
    """Raised when memory operations fail"""
    pass


def safe_execute(func: Callable, *args, fallback_value: Any = None, 
                error_message: str = None, **kwargs) -> Any:
    """Safely execute a function with error handling and logging"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_msg = error_message or f"Error in {func.__name__}: {str(e)}"
        logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
        return fallback_value


def retry_on_failure(max_retries: int = 3, delay: float = 0.1):
    """Decorator to retry function execution on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator


def log_performance(func):
    """Decorator to log function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            if execution_time > 0.1:  # Log slow operations
                logger.info(f"{func.__name__} took {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
            raise
    return wrapper


class ErrorHandler:
    """Centralized error handling for simulation components"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.max_history = 1000
    
    def handle_social_network_error(self, operation: str, error: Exception, 
                                  fallback_action: Callable = None) -> Any:
        """Handle social network specific errors"""
        self._log_error('social_network', operation, error)
        
        if isinstance(error, nx.NetworkXError):
            logger.warning(f"NetworkX error in {operation}: {str(error)}")
            if fallback_action:
                return fallback_action()
        elif isinstance(error, (KeyError, AttributeError)):
            logger.warning(f"Data structure error in {operation}: {str(error)}")
            if fallback_action:
                return fallback_action()
        else:
            logger.error(f"Unexpected error in social network {operation}: {str(error)}")
            raise SocialNetworkError(f"Social network operation '{operation}' failed: {str(error)}")
    
    def handle_perception_error(self, individual_id: int, error: Exception) -> Dict:
        """Handle individual perception errors"""
        self._log_error('perception', f'individual_{individual_id}', error)
        
        # Return minimal safe perception
        return {
            'nearby_individuals': [],
            'environmental_features': [],
            'social_information': [],
            'danger_signals': [],
            'opportunity_signals': []
        }
    
    def handle_cognition_error(self, individual_id: int, operation: str, 
                             error: Exception, fallback_value: Any = None) -> Any:
        """Handle cognitive operation errors"""
        self._log_error('cognition', f'{operation}_individual_{individual_id}', error)
        
        if isinstance(error, (IndexError, KeyError)):
            logger.warning(f"Data access error in {operation} for individual {individual_id}")
        elif isinstance(error, ValueError):
            logger.warning(f"Value error in {operation} for individual {individual_id}")
        else:
            logger.error(f"Unexpected cognition error in {operation} for individual {individual_id}")
        
        return fallback_value
    
    def handle_environment_error(self, operation: str, error: Exception, 
                                fallback_value: Any = None) -> Any:
        """Handle environmental operation errors"""
        self._log_error('environment', operation, error)
        
        if isinstance(error, (IndexError, KeyError)):
            logger.warning(f"Environment data access error in {operation}: {str(error)}")
        else:
            logger.error(f"Environment error in {operation}: {str(error)}")
        
        return fallback_value
    
    def handle_memory_error(self, individual_id: int, memory_type: str, 
                          error: Exception) -> bool:
        """Handle memory operation errors"""
        self._log_error('memory', f'{memory_type}_individual_{individual_id}', error)
        
        if isinstance(error, MemoryError):
            logger.warning(f"Memory limit reached for {memory_type} in individual {individual_id}")
            return False
        elif isinstance(error, (KeyError, IndexError)):
            logger.warning(f"Memory access error for {memory_type} in individual {individual_id}")
            return False
        else:
            logger.error(f"Unexpected memory error for {memory_type} in individual {individual_id}")
            return False
    
    def _log_error(self, component: str, operation: str, error: Exception):
        """Log error with categorization"""
        error_key = f"{component}_{operation}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        error_record = {
            'timestamp': time.time(),
            'component': component,
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'count': self.error_counts[error_key]
        }
        
        self.error_history.append(error_record)
        
        # Limit history size
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Log frequent errors differently
        if self.error_counts[error_key] > 10:
            logger.warning(f"Frequent error in {component}.{operation}: {str(error)} (count: {self.error_counts[error_key]})")
        else:
            logger.debug(f"Error in {component}.{operation}: {str(error)}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered"""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def reset_error_tracking(self):
        """Reset error tracking data"""
        self.error_counts.clear()
        self.error_history.clear()


# Global error handler instance
error_handler = ErrorHandler()


def safe_community_detection(graph: nx.Graph) -> list:
    """Safely perform community detection with proper error handling"""
    try:
        if len(graph.nodes()) == 0:
            return []
        
        # Try greedy modularity first
        try:
            communities = list(nx.community.greedy_modularity_communities(graph))
            return communities
        except (nx.NetworkXError, AttributeError) as e:
            logger.warning(f"Greedy modularity failed: {e}, trying label propagation")
            
            # Fallback to label propagation
            try:
                communities = list(nx.community.label_propagation_communities(graph))
                return communities
            except (nx.NetworkXError, AttributeError) as e:
                logger.warning(f"Label propagation failed: {e}, using connected components")
                
                # Final fallback to connected components
                communities = list(nx.connected_components(graph))
                return communities
                
    except Exception as e:
        error_handler.handle_social_network_error('community_detection', e)
        return []


def safe_centrality_calculation(graph: nx.Graph, centrality_type: str = 'betweenness') -> Dict:
    """Safely calculate network centrality measures"""
    try:
        if len(graph.nodes()) == 0:
            return {}
        
        if centrality_type == 'betweenness':
            return nx.betweenness_centrality(graph)
        elif centrality_type == 'closeness':
            return nx.closeness_centrality(graph)
        elif centrality_type == 'degree':
            return nx.degree_centrality(graph)
        elif centrality_type == 'eigenvector':
            try:
                return nx.eigenvector_centrality(graph, max_iter=1000)
            except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
                logger.warning("Eigenvector centrality failed, using degree centrality")
                return nx.degree_centrality(graph)
        else:
            logger.warning(f"Unknown centrality type: {centrality_type}, using degree")
            return nx.degree_centrality(graph)
            
    except Exception as e:
        error_handler.handle_social_network_error(f'{centrality_type}_centrality', e)
        return {}


def validate_configuration(config) -> bool:
    """Validate configuration parameters"""
    try:
        # Check required attributes
        required_attrs = ['width', 'height', 'max_population', 'initial_population']
        for attr in required_attrs:
            if not hasattr(config, attr):
                raise ConfigurationError(f"Missing required configuration: {attr}")
            
            value = getattr(config, attr)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ConfigurationError(f"Invalid value for {attr}: {value}")
        
        # Check logical constraints
        if config.initial_population > config.max_population:
            raise ConfigurationError("Initial population cannot exceed max population")
        
        if config.width <= 0 or config.height <= 0:
            raise ConfigurationError("Width and height must be positive")
        
        logger.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        return False 