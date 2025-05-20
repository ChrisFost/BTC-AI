import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import torch
import logging
import os
from datetime import datetime
from scipy import stats
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('reasoning_analyzer')

def log(message):
    """Utility function for logging"""
    logger.info(message)

class ReasoningAnalyzer:
    """
    Analyzes reasoning chains to provide reflection, learning, and improvement.
    
    Features:
    - Post-episode reflection on reasoning quality vs outcomes
    - Contrastive learning between successful and unsuccessful reasoning
    - Performance attribution across reasoning components
    - Temporal consistency analysis
    - Meta-parameter adjustment based on reasoning effectiveness
    """
    def __init__(self, max_history=1000, reflection_interval=100, 
                 success_threshold=0.75, enable_meta_adjustment=True):
        """
        Initialize the ReasoningAnalyzer.
        
        Args:
            max_history (int): Maximum number of reasoning chains to store
            reflection_interval (int): Steps between reflection cycles
            success_threshold (float): Profit threshold to classify trades as successful
            enable_meta_adjustment (bool): Enable automatic adjustment of reasoning weights
        """
        # Storage for reasoning chains and outcomes
        self.reasoning_history = deque(maxlen=max_history)
        self.outcome_history = deque(maxlen=max_history)
        self.profit_history = deque(maxlen=max_history)
        self.trades_history = deque(maxlen=max_history)
        
        # Parameters
        self.reflection_interval = reflection_interval
        self.success_threshold = success_threshold
        self.enable_meta_adjustment = enable_meta_adjustment
        self.step_counter = 0
        
        # Component effectiveness tracking
        self.component_attribution = {
            'market_regime': [],
            'patterns': [],
            'support_resistance': [],
            'volatility': [],
            'liquidity': [],
            'entry_exit': []
        }
        
        # Temporal consistency tracking
        self.regime_shifts = []
        self.pattern_shifts = []
        self.signal_shifts = []
        
        # Meta-parameter storage (optimized reasoning weights)
        self.meta_weights = {
            'market_regime': 0.2,
            'patterns': 0.15,
            'support_resistance': 0.15,
            'volatility': 0.15,
            'liquidity': 0.15,
            'entry_exit': 0.2
        }
        
        # Best performers by trading type
        self.bucket_best_weights = {
            'scalping': None,
            'short': None,
            'medium': None,
            'long': None
        }
        
        # Learning rate for meta-parameter updates
        self.learning_rate = 0.05
        
        log(f"Initialized ReasoningAnalyzer with reflection_interval={reflection_interval}")
    
    def add_reasoning_step(self, reasoning_chain, action_taken, profit=None, trade_info=None):
        """
        Add a reasoning step to the analyzer's history.
        
        Args:
            reasoning_chain (dict): The chain of draft reasoning components
            action_taken (dict): The action taken based on this reasoning
            profit (float, optional): The profit from this action if available
            trade_info (dict, optional): Additional trade information
        """
        # Store reasoning chain
        self.reasoning_history.append(reasoning_chain)
        
        # Store action
        self.outcome_history.append(action_taken)
        
        # Store profit if available
        if profit is not None:
            self.profit_history.append(profit)
        
        # Store trade info if available
        if trade_info is not None:
            self.trades_history.append(trade_info)
        
        # Increment step counter
        self.step_counter += 1
        
        # Check if it's time for reflection
        if self.step_counter % self.reflection_interval == 0:
            self.reflect()
            
        # Update temporal consistency metrics
        if len(self.reasoning_history) >= 2:
            self._update_temporal_metrics(reasoning_chain, self.reasoning_history[-2])
    
    def _update_temporal_metrics(self, current_reasoning, previous_reasoning):
        """
        Update metrics that track how consistent reasoning is over time.
        
        Args:
            current_reasoning (dict): Current reasoning chain
            previous_reasoning (dict): Previous reasoning chain
        """
        # Skip if either reasoning chain is missing key components
        if ('market_regime' not in current_reasoning or 
            'market_regime' not in previous_reasoning):
            return
            
        # Calculate regime shift (how much did the regime assessment change)
        curr_regime = current_reasoning['market_regime'].cpu().numpy()[0]
        prev_regime = previous_reasoning['market_regime'].cpu().numpy()[0]
        
        # Calculate cosine similarity (1 = identical, -1 = opposite)
        regime_similarity = np.dot(curr_regime, prev_regime) / (
            np.linalg.norm(curr_regime) * np.linalg.norm(prev_regime)
        )
        regime_shift = 1 - regime_similarity
        self.regime_shifts.append(regime_shift)
        
        # Similar calculations for other components if available
        if 'patterns' in current_reasoning and 'patterns' in previous_reasoning:
            curr_patterns = current_reasoning['patterns'].cpu().numpy()[0]
            prev_patterns = previous_reasoning['patterns'].cpu().numpy()[0]
            patterns_similarity = np.dot(curr_patterns, prev_patterns) / (
                np.linalg.norm(curr_patterns) * np.linalg.norm(prev_patterns)
            )
            pattern_shift = 1 - patterns_similarity
            self.pattern_shifts.append(pattern_shift)
        
        if 'entry_exit' in current_reasoning and 'entry_exit' in previous_reasoning:
            curr_signals = current_reasoning['entry_exit'].cpu().numpy()[0]
            prev_signals = previous_reasoning['entry_exit'].cpu().numpy()[0]
            signals_similarity = np.dot(curr_signals, prev_signals) / (
                np.linalg.norm(curr_signals) * np.linalg.norm(prev_signals)
            )
            signal_shift = 1 - signals_similarity
            self.signal_shifts.append(signal_shift)
    
    def reflect(self):
        """
        Perform reflection on the recent reasoning history and outcomes.
        Identifies patterns between reasoning and performance.
        """
        if len(self.reasoning_history) < 10 or len(self.profit_history) < 10:
            log("Not enough history for meaningful reflection yet")
            return
            
        log(f"Performing reflection analysis on {len(self.reasoning_history)} reasoning steps")
        
        # Separate successful and unsuccessful reasoning chains
        successful_indices = []
        unsuccessful_indices = []
        
        for i, profit in enumerate(self.profit_history):
            if profit >= self.success_threshold:
                successful_indices.append(i)
            else:
                unsuccessful_indices.append(i)
        
        # Perform contrastive analysis
        if successful_indices and unsuccessful_indices:
            self._contrastive_analysis(successful_indices, unsuccessful_indices)
        
        # Perform performance attribution
        self._performance_attribution()
        
        # Analyze temporal consistency
        self._analyze_temporal_consistency()
        
        # Update meta-parameters if enabled
        if self.enable_meta_adjustment:
            self._adjust_meta_parameters()
            
        log("Reflection completed")
    
    def _contrastive_analysis(self, successful_indices, unsuccessful_indices):
        """
        Compare reasoning chains between successful and unsuccessful outcomes.
        
        Args:
            successful_indices (list): Indices of successful reasoning chains
            unsuccessful_indices (list): Indices of unsuccessful reasoning chains
        """
        # Skip if we don't have at least some of each
        if len(successful_indices) < 5 or len(unsuccessful_indices) < 5:
            return
            
        log(f"Performing contrastive analysis between {len(successful_indices)} successful and {len(unsuccessful_indices)} unsuccessful chains")
        
        # Extract key components from successful and unsuccessful reasoning
        successful_regimes = self._extract_component_values('market_regime', successful_indices)
        unsuccessful_regimes = self._extract_component_values('market_regime', unsuccessful_indices)
        
        successful_patterns = self._extract_component_values('patterns', successful_indices)
        unsuccessful_patterns = self._extract_component_values('patterns', unsuccessful_indices)
        
        successful_entry_exit = self._extract_component_values('entry_exit', successful_indices)
        unsuccessful_entry_exit = self._extract_component_values('entry_exit', unsuccessful_indices)
        
        # Compare market regimes
        if successful_regimes and unsuccessful_regimes:
            # Calculate average regime distribution for each group
            avg_successful_regime = np.mean(successful_regimes, axis=0)
            avg_unsuccessful_regime = np.mean(unsuccessful_regimes, axis=0)
            
            # Find the regime index with the biggest difference
            regime_diff = avg_successful_regime - avg_unsuccessful_regime
            most_diff_idx = np.argmax(np.abs(regime_diff))
            regimes = ['trending', 'ranging', 'volatile', 'mixed']
            
            if regime_diff[most_diff_idx] > 0.1:  # Significant difference
                log(f"Successful trades show higher '{regimes[most_diff_idx]}' regime component " 
                    f"(+{regime_diff[most_diff_idx]:.2f})")
        
        # Compare entry/exit signals
        if successful_entry_exit and unsuccessful_entry_exit:
            avg_successful_ee = np.mean(successful_entry_exit, axis=0)
            avg_unsuccessful_ee = np.mean(unsuccessful_entry_exit, axis=0)
            
            entry_diff = avg_successful_ee[0] - avg_unsuccessful_ee[0]
            exit_diff = avg_successful_ee[1] - avg_unsuccessful_ee[1]
            
            if abs(entry_diff) > 0.1:
                log(f"Successful trades show {'higher' if entry_diff > 0 else 'lower'} entry signals "
                    f"({entry_diff:.2f} difference)")
                
            if abs(exit_diff) > 0.1:
                log(f"Successful trades show {'higher' if exit_diff > 0 else 'lower'} exit signals "
                    f"({exit_diff:.2f} difference)")
    
    def _extract_component_values(self, component_name, indices):
        """
        Extract values for a specific reasoning component from history.
        
        Args:
            component_name (str): Name of the reasoning component
            indices (list): Indices to extract from
            
        Returns:
            list: Extracted component values
        """
        values = []
        
        for idx in indices:
            if idx >= len(self.reasoning_history):
                continue
                
            reasoning = self.reasoning_history[idx]
            if component_name in reasoning:
                values.append(reasoning[component_name].cpu().numpy()[0])
                
        return values
    
    def _performance_attribution(self):
        """
        Analyze which reasoning components contributed most to successful outcomes.
        Updates component_attribution with correlation scores.
        """
        if len(self.reasoning_history) < 20 or len(self.profit_history) < 20:
            return
            
        # For each component, calculate correlation with profit
        for component in self.component_attribution.keys():
            values = []
            
            # Extract component values (taking the first/primary value from each component)
            for reasoning in self.reasoning_history:
                if component in reasoning:
                    if component == 'market_regime':
                        # For market regime, use the maximum value (strongest regime)
                        val = float(torch.max(reasoning[component][0]).cpu().numpy())
                    elif component == 'entry_exit':
                        # For entry/exit, use the maximum signal
                        val = float(torch.max(reasoning[component][0]).cpu().numpy())
                    else:
                        # For other components, use the first value
                        val = float(reasoning[component][0][0].cpu().numpy())
                    values.append(val)
                else:
                    values.append(0.0)
            
            # Ensure we have enough values to calculate correlation
            if len(values) < len(self.profit_history):
                values.extend([0.0] * (len(self.profit_history) - len(values)))
            elif len(values) > len(self.profit_history):
                values = values[:len(self.profit_history)]
                
            # Calculate correlation with profit
            if len(values) > 1 and len(set(values)) > 1:  # Need variation in values
                correlation, p_value = stats.pearsonr(values, list(self.profit_history))
                
                # Store the correlation
                self.component_attribution[component].append((correlation, p_value))
                
                # Log significant correlations
                if p_value < 0.05 and abs(correlation) > 0.2:
                    log(f"Component '{component}' shows {abs(correlation):.2f} "
                        f"{'positive' if correlation > 0 else 'negative'} correlation with profit "
                        f"(p={p_value:.3f})")
    
    def _analyze_temporal_consistency(self):
        """
        Analyze how consistent the reasoning is over time.
        High volatility in reasoning might indicate model confusion.
        """
        # Skip if we don't have enough history
        if (len(self.regime_shifts) < 10 or 
            len(self.pattern_shifts) < 10 or 
            len(self.signal_shifts) < 10):
            return
            
        # Calculate rolling average of shifts to detect sustained inconsistency
        window = min(10, len(self.regime_shifts))
        recent_regime_shifts = np.mean(self.regime_shifts[-window:])
        recent_pattern_shifts = np.mean(self.pattern_shifts[-window:])
        recent_signal_shifts = np.mean(self.signal_shifts[-window:])
        
        # High shift values indicate inconsistent reasoning
        if recent_regime_shifts > 0.5:
            log(f"Warning: High regime inconsistency detected ({recent_regime_shifts:.2f})")
            
        if recent_pattern_shifts > 0.5:
            log(f"Warning: High pattern recognition inconsistency detected ({recent_pattern_shifts:.2f})")
            
        if recent_signal_shifts > 0.5:
            log(f"Warning: High entry/exit signal inconsistency detected ({recent_signal_shifts:.2f})")
            
        # Check if all components show high inconsistency - potential confusion
        if (recent_regime_shifts > 0.4 and 
            recent_pattern_shifts > 0.4 and 
            recent_signal_shifts > 0.4):
            log("Critical: High inconsistency across all reasoning components indicates possible model confusion")
    
    def _adjust_meta_parameters(self):
        """
        Adjust reasoning component weights based on their contribution to performance.
        Higher correlation with profit = higher weight.
        """
        if not self.enable_meta_adjustment:
            return
            
        # Check if we have correlation data for all components
        if not all(len(corr) > 0 for corr in self.component_attribution.values()):
            return
            
        # Get latest correlation for each component
        correlations = {}
        for component, corr_data in self.component_attribution.items():
            if corr_data:
                # Use absolute correlation (positive or negative relationship is valuable)
                correlations[component] = abs(corr_data[-1][0])
        
        # Skip if any correlation is NaN
        if any(np.isnan(c) for c in correlations.values()):
            return
        
        # Normalize to get new weights (ensure they sum to 1.0)
        total_corr = sum(correlations.values())
        if total_corr > 0:
            new_weights = {c: v/total_corr for c, v in correlations.items()}
            
            # Apply smoothing with learning rate to avoid drastic changes
            for component in self.meta_weights:
                if component in new_weights:
                    self.meta_weights[component] = (
                        (1 - self.learning_rate) * self.meta_weights[component] + 
                        self.learning_rate * new_weights[component]
                    )
            
            # Log significant weight adjustments
            log(f"Updated meta-weights: {', '.join(f'{c}: {w:.2f}' for c, w in self.meta_weights.items())}")
    
    def get_optimized_weights(self, bucket_type=None):
        """
        Get optimized reasoning weights, optionally for a specific trading bucket.
        
        Args:
            bucket_type (str, optional): Trading bucket type ('scalping', 'short', 'medium', 'long')
            
        Returns:
            dict: Optimized component weights
        """
        # If we have bucket-specific weights and a bucket type is specified, use those
        if bucket_type and bucket_type in self.bucket_best_weights and self.bucket_best_weights[bucket_type]:
            return self.bucket_best_weights[bucket_type]
        
        # Otherwise, return the general meta weights
        return self.meta_weights
    
    def generate_reflection_report(self, bucket_type=None):
        """
        Generate a comprehensive reflection report on reasoning performance.
        
        Args:
            bucket_type (str, optional): Trading bucket type to focus on
            
        Returns:
            dict: Report with various reflection metrics
        """
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'history_length': len(self.reasoning_history),
            'meta_weights': self.get_optimized_weights(bucket_type),
            'component_attribution': {},
            'temporal_consistency': {
                'regime_stability': 1.0 - np.mean(self.regime_shifts[-20:]) if self.regime_shifts else None,
                'pattern_stability': 1.0 - np.mean(self.pattern_shifts[-20:]) if self.pattern_shifts else None,
                'signal_stability': 1.0 - np.mean(self.signal_shifts[-20:]) if self.signal_shifts else None
            }
        }
        
        # Add component attribution
        for component, corr_data in self.component_attribution.items():
            if corr_data:
                report['component_attribution'][component] = {
                    'correlation': corr_data[-1][0],
                    'p_value': corr_data[-1][1],
                    'significance': 'high' if abs(corr_data[-1][0]) > 0.5 and corr_data[-1][1] < 0.01 else
                                   'medium' if abs(corr_data[-1][0]) > 0.3 and corr_data[-1][1] < 0.05 else 'low'
                }
        
        # Add success rate
        if self.profit_history:
            successful_trades = sum(1 for p in self.profit_history if p >= self.success_threshold)
            report['success_rate'] = successful_trades / len(self.profit_history)
        
        return report
    
    def visualize_reasoning_effectiveness(self, save_path=None, bucket_type=None):
        """
        Visualize the effectiveness of different reasoning components over time.
        
        Args:
            save_path (str, optional): Path to save the visualization
            bucket_type (str, optional): Trading bucket type to focus on
            
        Returns:
            matplotlib.figure.Figure: Figure with visualizations
        """
        if not self.component_attribution or len(self.profit_history) < 10:
            log("Not enough data for visualization")
            return None
            
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # Plot 1: Component correlation with profit
        ax1 = fig.add_subplot(gs[0, :])
        components = list(self.component_attribution.keys())
        
        # Get latest correlation for each component
        latest_correlations = []
        for component in components:
            if self.component_attribution[component]:
                latest_correlations.append(self.component_attribution[component][-1][0])
            else:
                latest_correlations.append(0)
        
        # Create bar chart
        bars = ax1.bar(components, latest_correlations)
        
        # Color bars based on correlation (positive=green, negative=red)
        for bar, corr in zip(bars, latest_correlations):
            if corr >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        ax1.set_title('Component Correlation with Profit')
        ax1.set_ylabel('Correlation Coefficient')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Temporal consistency
        ax2 = fig.add_subplot(gs[1, 0])
        x = range(len(self.regime_shifts))
        
        if self.regime_shifts:
            ax2.plot(x, self.regime_shifts, label='Regime Shifts', color='blue')
        if self.pattern_shifts:
            ax2.plot(x, self.pattern_shifts, label='Pattern Shifts', color='orange')
        if self.signal_shifts:
            ax2.plot(x, self.signal_shifts, label='Signal Shifts', color='green')
            
        ax2.set_title('Reasoning Consistency Over Time')
        ax2.set_ylabel('Shift Magnitude (lower is better)')
        ax2.set_xlabel('Step')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Meta-weight evolution
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Plot optimized weights as radar chart
        categories = list(self.meta_weights.keys())
        values = list(self.meta_weights.values())
        
        # Close the loop for the radar chart
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        # Convert to radians and plot
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax3.plot(angles, values, linewidth=2, linestyle='solid')
        ax3.fill(angles, values, alpha=0.1)
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories[:-1])
        ax3.set_title('Optimized Reasoning Weights')
        
        # Plot 4: Profit correlation over time
        ax4 = fig.add_subplot(gs[2, :])
        
        # For each component, plot correlation evolution
        for component in components:
            corr_values = [c[0] for c in self.component_attribution[component]]
            if corr_values:
                x = range(len(corr_values))
                ax4.plot(x, corr_values, label=component)
                
        ax4.set_title('Component Performance Attribution Over Time')
        ax4.set_ylabel('Correlation with Profit')
        ax4.set_xlabel('Reflection Cycle')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)
            log(f"Visualization saved to {save_path}")
            
        return fig
    
    def save_reflection_data(self, save_dir='reasoning_reflections'):
        """
        Save reflection data to disk for later analysis.
        
        Args:
            save_dir (str): Directory to save data
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save reflection report
        report = self.generate_reflection_report()
        report_path = os.path.join(save_dir, f"reflection_report_{timestamp}.json")
        
        # Convert numpy values to Python native types for JSON serialization
        for component, data in report.get('component_attribution', {}).items():
            for key, value in data.items():
                if isinstance(value, (np.float32, np.float64)):
                    data[key] = float(value)
        
        for key, value in report.get('temporal_consistency', {}).items():
            if isinstance(value, (np.float32, np.float64)):
                report['temporal_consistency'][key] = float(value) if value is not None else None
        
        # Save as JSON
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        log(f"Reflection report saved to {report_path}")
        
        # Save visualization
        vis_path = os.path.join(save_dir, f"reasoning_effectiveness_{timestamp}.png")
        self.visualize_reasoning_effectiveness(save_path=vis_path)
        
        return report_path
    
    def contrastive_learning_update(self, agent):
        """
        Update agent reasoning weights based on contrastive learning.
        
        Args:
            agent: The agent whose weights should be updated
            
        Returns:
            bool: Whether weights were updated
        """
        if not hasattr(agent, 'reasoning_weights') or not self.enable_meta_adjustment:
            return False
            
        # Get optimized weights for the agent's bucket type
        bucket_type = agent.trading_style if hasattr(agent, 'trading_style') else None
        optimized_weights = self.get_optimized_weights(bucket_type)
        
        # Update agent's reasoning weights
        agent.reasoning_weights = optimized_weights
        
        log(f"Updated agent reasoning weights based on contrastive learning")
        return True 