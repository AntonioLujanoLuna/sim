"""
Information-theoretic measures and analysis.

This module implements information-theoretic metrics including entropy, mutual information,
transfer entropy, and complexity measures for analyzing emergence and organization.
"""

import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any, Union
import math
from scipy import stats
from scipy.spatial.distance import pdist
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class EntropyCalculator:
    """Calculate various entropy measures"""
    
    @staticmethod
    def shannon_entropy(data: List[Any], base: float = 2) -> float:
        """Calculate Shannon entropy of a dataset"""
        if not data:
            return 0.0
        
        # Count frequencies
        counter = Counter(data)
        total = len(data)
        
        # Calculate entropy
        entropy = 0.0
        for count in counter.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log(probability, base)
        
        return entropy
    
    @staticmethod
    def conditional_entropy(x_data: List[Any], y_data: List[Any], base: float = 2) -> float:
        """Calculate conditional entropy H(X|Y)"""
        if len(x_data) != len(y_data) or not x_data:
            return 0.0
        
        # Group X values by Y values
        y_groups = defaultdict(list)
        for x, y in zip(x_data, y_data):
            y_groups[y].append(x)
        
        # Calculate weighted conditional entropy
        total_count = len(x_data)
        conditional_entropy = 0.0
        
        for y_value, x_values in y_groups.items():
            y_probability = len(x_values) / total_count
            x_entropy = EntropyCalculator.shannon_entropy(x_values, base)
            conditional_entropy += y_probability * x_entropy
        
        return conditional_entropy
    
    @staticmethod
    def joint_entropy(x_data: List[Any], y_data: List[Any], base: float = 2) -> float:
        """Calculate joint entropy H(X,Y)"""
        if len(x_data) != len(y_data) or not x_data:
            return 0.0
        
        # Create joint distribution
        joint_data = list(zip(x_data, y_data))
        return EntropyCalculator.shannon_entropy(joint_data, base)
    
    @staticmethod
    def mutual_information(x_data: List[Any], y_data: List[Any], base: float = 2) -> float:
        """Calculate mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)"""
        if len(x_data) != len(y_data) or not x_data:
            return 0.0
        
        h_x = EntropyCalculator.shannon_entropy(x_data, base)
        h_y = EntropyCalculator.shannon_entropy(y_data, base)
        h_xy = EntropyCalculator.joint_entropy(x_data, y_data, base)
        
        return h_x + h_y - h_xy
    
    @staticmethod
    def relative_entropy(p_data: List[Any], q_data: List[Any], base: float = 2) -> float:
        """Calculate Kullback-Leibler divergence D(P||Q)"""
        if not p_data or not q_data:
            return float('inf')
        
        # Get probability distributions
        p_counter = Counter(p_data)
        q_counter = Counter(q_data)
        
        # Normalize
        p_total = len(p_data)
        q_total = len(q_data)
        
        # Calculate KL divergence
        kl_div = 0.0
        for item in p_counter:
            p_prob = p_counter[item] / p_total
            q_prob = q_counter.get(item, 0) / q_total
            
            if q_prob == 0:
                return float('inf')  # Undefined when Q(x) = 0 but P(x) > 0
            
            kl_div += p_prob * math.log(p_prob / q_prob, base)
        
        return kl_div
    
    @staticmethod
    def differential_entropy(data: List[float], bins: int = 10, base: float = 2) -> float:
        """Calculate differential entropy for continuous data"""
        if not data:
            return 0.0
        
        # Discretize continuous data
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Calculate differential entropy
        entropy = 0.0
        for density in hist:
            if density > 0:
                entropy -= density * bin_width * math.log(density * bin_width, base)
        
        return entropy


class TransferEntropyCalculator:
    """Calculate transfer entropy for causal analysis"""
    
    def __init__(self, lag: int = 1, bins: int = 10):
        self.lag = lag
        self.bins = bins
    
    def discretize_data(self, data: List[float]) -> List[int]:
        """Discretize continuous data for transfer entropy calculation"""
        if not data:
            return []
        
        # Use quantile-based binning for better distribution
        quantiles = np.linspace(0, 1, self.bins + 1)
        thresholds = np.quantile(data, quantiles)
        
        discretized = []
        for value in data:
            bin_idx = np.searchsorted(thresholds[1:-1], value)
            discretized.append(bin_idx)
        
        return discretized
    
    def transfer_entropy(self, source: List[float], target: List[float]) -> float:
        """Calculate transfer entropy from source to target"""
        if len(source) != len(target) or len(source) <= self.lag:
            return 0.0
        
        # Discretize data
        source_disc = self.discretize_data(source)
        target_disc = self.discretize_data(target)
        
        # Create time series with lag
        target_present = target_disc[self.lag:]
        target_past = target_disc[:-self.lag]
        source_past = source_disc[:-self.lag]
        
        # Calculate transfer entropy: TE = H(X_t | X_{t-1}) - H(X_t | X_{t-1}, Y_{t-1})
        h_target_given_target_past = EntropyCalculator.conditional_entropy(
            target_present, target_past
        )
        
        # Joint conditioning on both target and source past
        joint_past = list(zip(target_past, source_past))
        h_target_given_both_past = EntropyCalculator.conditional_entropy(
            target_present, joint_past
        )
        
        return h_target_given_target_past - h_target_given_both_past
    
    def effective_transfer_entropy(self, source: List[float], target: List[float], 
                                  num_surrogates: int = 100) -> Tuple[float, float]:
        """Calculate transfer entropy with statistical significance testing"""
        # Calculate actual transfer entropy
        actual_te = self.transfer_entropy(source, target)
        
        # Generate surrogate data by shuffling source
        surrogate_tes = []
        source_array = np.array(source)
        
        for _ in range(num_surrogates):
            shuffled_source = np.random.permutation(source_array).tolist()
            surrogate_te = self.transfer_entropy(shuffled_source, target)
            surrogate_tes.append(surrogate_te)
        
        # Calculate statistical significance
        surrogate_mean = np.mean(surrogate_tes)
        surrogate_std = np.std(surrogate_tes)
        
        if surrogate_std > 0:
            z_score = (actual_te - surrogate_mean) / surrogate_std
            effective_te = max(0, actual_te - surrogate_mean)
        else:
            z_score = 0
            effective_te = actual_te
        
        return effective_te, z_score


class ComplexityMeasures:
    """Calculate various complexity measures"""
    
    @staticmethod
    def logical_depth(data: List[Any], compression_algorithm: str = 'lz77') -> int:
        """Estimate logical depth using compression"""
        if not data:
            return 0
        
        # Convert data to string for compression
        data_str = ''.join(str(item) for item in data)
        
        if compression_algorithm == 'lz77':
            return ComplexityMeasures._lz77_complexity(data_str)
        else:
            # Use length as simple proxy
            return len(data_str)
    
    @staticmethod
    def _lz77_complexity(string: str) -> int:
        """Simple LZ77-style compression complexity"""
        if not string:
            return 0
        
        complexity = 0
        i = 0
        
        while i < len(string):
            # Find longest match in previous characters
            max_length = 0
            match_pos = -1
            
            for j in range(max(0, i - 255), i):  # Limited lookback window
                length = 0
                while (i + length < len(string) and 
                       j + length < i and 
                       string[i + length] == string[j + length]):
                    length += 1
                
                if length > max_length:
                    max_length = length
                    match_pos = j
            
            if max_length > 2:  # Worthwhile compression
                i += max_length
            else:
                i += 1
            
            complexity += 1
        
        return complexity
    
    @staticmethod
    def effective_complexity(data: List[Any], randomness_threshold: float = 0.5) -> float:
        """Calculate effective complexity (information that's neither random nor regular)"""
        if not data:
            return 0.0
        
        # Total information content
        total_entropy = EntropyCalculator.shannon_entropy(data)
        
        # Estimate randomness (high entropy regions)
        # Split data into chunks and measure local entropy
        chunk_size = max(10, len(data) // 20)
        local_entropies = []
        
        for i in range(0, len(data) - chunk_size + 1, chunk_size):
            chunk = data[i:i + chunk_size]
            local_entropy = EntropyCalculator.shannon_entropy(chunk)
            local_entropies.append(local_entropy)
        
        # High local entropy suggests randomness
        if local_entropies:
            avg_local_entropy = np.mean(local_entropies)
            randomness_estimate = min(total_entropy, avg_local_entropy)
        else:
            randomness_estimate = 0
        
        # Effective complexity is total information minus randomness
        effective_complex = total_entropy - randomness_estimate
        return max(0, effective_complex)
    
    @staticmethod
    def emergence_index(micro_states: List[Any], macro_states: List[Any]) -> float:
        """Calculate emergence index comparing micro and macro level information"""
        if len(micro_states) != len(macro_states) or not micro_states:
            return 0.0
        
        # Information at micro level
        micro_entropy = EntropyCalculator.shannon_entropy(micro_states)
        
        # Information at macro level
        macro_entropy = EntropyCalculator.shannon_entropy(macro_states)
        
        # Mutual information between levels
        mutual_info = EntropyCalculator.mutual_information(micro_states, macro_states)
        
        # Emergence as macro information not predictable from micro
        if micro_entropy > 0:
            emergence = macro_entropy - mutual_info
            normalized_emergence = emergence / macro_entropy if macro_entropy > 0 else 0
            return max(0, normalized_emergence)
        
        return 0.0


class InformationAnalyzer:
    """Main information-theoretic analyzer for complex systems"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.time_series_data = defaultdict(list)
        self.entropy_calculator = EntropyCalculator()
        self.transfer_entropy_calc = TransferEntropyCalculator()
        self.analysis_history = []
        
    def add_data_point(self, variable_name: str, value: Any, timestamp: int):
        """Add data point for analysis"""
        self.time_series_data[variable_name].append((timestamp, value))
        
        # Maintain window size
        if len(self.time_series_data[variable_name]) > self.window_size * 2:
            self.time_series_data[variable_name] = self.time_series_data[variable_name][-self.window_size:]
    
    def add_system_state(self, state: Dict[str, Any], timestamp: int):
        """Add complete system state for analysis"""
        for variable, value in state.items():
            self.add_data_point(variable, value, timestamp)
    
    def calculate_system_entropy(self, variables: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate entropy for system variables"""
        if variables is None:
            variables = list(self.time_series_data.keys())
        
        entropies = {}
        for var in variables:
            if var in self.time_series_data and self.time_series_data[var]:
                values = [value for _, value in self.time_series_data[var][-self.window_size:]]
                entropies[var] = self.entropy_calculator.shannon_entropy(values)
        
        return entropies
    
    def calculate_pairwise_mutual_information(self, variables: Optional[List[str]] = None) -> Dict[Tuple[str, str], float]:
        """Calculate mutual information between all pairs of variables"""
        if variables is None:
            variables = list(self.time_series_data.keys())
        
        mutual_info = {}
        for var1, var2 in combinations(variables, 2):
            if var1 in self.time_series_data and var2 in self.time_series_data:
                # Get recent values
                values1 = [value for _, value in self.time_series_data[var1][-self.window_size:]]
                values2 = [value for _, value in self.time_series_data[var2][-self.window_size:]]
                
                # Ensure same length
                min_length = min(len(values1), len(values2))
                if min_length > 10:
                    values1 = values1[-min_length:]
                    values2 = values2[-min_length:]
                    
                    mi = self.entropy_calculator.mutual_information(values1, values2)
                    mutual_info[(var1, var2)] = mi
        
        return mutual_info
    
    def calculate_causal_network(self, variables: Optional[List[str]] = None) -> Dict[Tuple[str, str], float]:
        """Calculate transfer entropy network showing causal relationships"""
        if variables is None:
            variables = list(self.time_series_data.keys())
        
        causal_network = {}
        
        for source_var in variables:
            for target_var in variables:
                if source_var != target_var:
                    if (source_var in self.time_series_data and 
                        target_var in self.time_series_data):
                        
                        # Get time series values
                        source_values = [value for _, value in self.time_series_data[source_var][-self.window_size:]]
                        target_values = [value for _, value in self.time_series_data[target_var][-self.window_size:]]
                        
                        # Ensure same length and sufficient data
                        min_length = min(len(source_values), len(target_values))
                        if min_length > 20:
                            source_values = source_values[-min_length:]
                            target_values = target_values[-min_length:]
                            
                            # Calculate transfer entropy
                            if all(isinstance(v, (int, float)) for v in source_values + target_values):
                                te = self.transfer_entropy_calc.transfer_entropy(source_values, target_values)
                                causal_network[(source_var, target_var)] = te
        
        return causal_network
    
    def detect_emergence_events(self, micro_variables: List[str], 
                               macro_variables: List[str]) -> List[Dict[str, Any]]:
        """Detect emergence events by analyzing micro-macro relationships"""
        emergence_events = []
        
        if not self.time_series_data:
            return emergence_events
        
        # Get recent time window
        recent_window = 50
        
        for macro_var in macro_variables:
            if macro_var not in self.time_series_data:
                continue
            
            macro_values = [value for _, value in self.time_series_data[macro_var][-recent_window:]]
            
            # Aggregate micro state
            micro_state = []
            for timestamp in range(len(macro_values)):
                micro_snapshot = []
                for micro_var in micro_variables:
                    if (micro_var in self.time_series_data and 
                        timestamp < len(self.time_series_data[micro_var])):
                        _, value = self.time_series_data[micro_var][-recent_window + timestamp]
                        micro_snapshot.append(value)
                
                # Convert to tuple for hashability
                micro_state.append(tuple(micro_snapshot) if micro_snapshot else ())
            
            if len(micro_state) == len(macro_values) and len(micro_state) > 10:
                # Calculate emergence index
                emergence_idx = ComplexityMeasures.emergence_index(micro_state, macro_values)
                
                if emergence_idx > 0.3:  # Threshold for significant emergence
                    emergence_events.append({
                        'type': 'emergence_detected',
                        'macro_variable': macro_var,
                        'emergence_index': emergence_idx,
                        'micro_entropy': self.entropy_calculator.shannon_entropy(micro_state),
                        'macro_entropy': self.entropy_calculator.shannon_entropy(macro_values)
                    })
        
        return emergence_events
    
    def calculate_information_integration(self, variables: List[str]) -> float:
        """Calculate information integration (Φ) as a measure of consciousness/integration"""
        if len(variables) < 2:
            return 0.0
        
        # Get system state
        system_states = []
        min_length = float('inf')
        
        for var in variables:
            if var in self.time_series_data:
                min_length = min(min_length, len(self.time_series_data[var]))
        
        if min_length < 10:
            return 0.0
        
        # Create joint system states
        for i in range(int(min_length)):
            state = []
            for var in variables:
                if i < len(self.time_series_data[var]):
                    _, value = self.time_series_data[var][-int(min_length) + i]
                    state.append(value)
            system_states.append(tuple(state))
        
        # Calculate whole system entropy
        whole_entropy = self.entropy_calculator.shannon_entropy(system_states)
        
        # Calculate sum of parts entropy
        parts_entropy = 0.0
        for var in variables:
            if var in self.time_series_data:
                values = [value for _, value in self.time_series_data[var][-int(min_length):]]
                parts_entropy += self.entropy_calculator.shannon_entropy(values)
        
        # Information integration as difference
        integration = parts_entropy - whole_entropy
        return max(0, integration)
    
    def get_information_flow_summary(self) -> Dict[str, Any]:
        """Get comprehensive information flow analysis"""
        variables = list(self.time_series_data.keys())
        
        summary = {
            'system_entropy': self.calculate_system_entropy(),
            'mutual_information_matrix': self.calculate_pairwise_mutual_information(),
            'causal_network': self.calculate_causal_network(),
            'information_integration': self.calculate_information_integration(variables),
            'emergence_events': self.detect_emergence_events(
                [v for v in variables if 'micro' in v.lower()],
                [v for v in variables if 'macro' in v.lower()]
            )
        }
        
        # Calculate network properties
        causal_network = summary['causal_network']
        if causal_network:
            # Find strongest causal relationships
            strongest_links = sorted(causal_network.items(), key=lambda x: x[1], reverse=True)[:5]
            summary['strongest_causal_links'] = strongest_links
            
            # Calculate average causal strength
            summary['avg_causal_strength'] = np.mean(list(causal_network.values()))
        
        return summary
    
    def analyze_criticality(self, variable: str, window_size: Optional[int] = None) -> Dict[str, float]:
        """Analyze if system is near critical point using information measures"""
        if variable not in self.time_series_data:
            return {}
        
        if window_size is None:
            window_size = self.window_size
        
        values = [value for _, value in self.time_series_data[variable][-window_size:]]
        
        if len(values) < 50:
            return {}
        
        # Split into chunks to analyze temporal correlations
        chunk_size = 10
        chunks = [values[i:i+chunk_size] for i in range(0, len(values), chunk_size)]
        chunk_entropies = [self.entropy_calculator.shannon_entropy(chunk) for chunk in chunks if len(chunk) == chunk_size]
        
        # Criticality indicators
        analysis = {}
        
        # Entropy fluctuations
        if len(chunk_entropies) > 1:
            analysis['entropy_fluctuation'] = np.std(chunk_entropies)
        
        # Susceptibility (variance)
        analysis['susceptibility'] = np.var(values)
        
        # Correlation length (autocorrelation decay)
        autocorrs = []
        for lag in range(1, min(20, len(values)//4)):
            if len(values) > lag:
                corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(abs(corr))
        
        if autocorrs:
            # Correlation length as decay rate
            analysis['correlation_length'] = np.sum(autocorrs)
        
        # Peak in susceptibility suggests criticality
        analysis['criticality_score'] = analysis.get('susceptibility', 0) * analysis.get('entropy_fluctuation', 0)
        
        return analysis