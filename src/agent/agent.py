#!/usr/bin/env python
"""Reinforcement Learning Agents for Trading

This module implements reinforcement learning agents for trading,
with a focus on the PPO (Proximal Policy Optimization) algorithm.

Classes:
- PPOAgent: Proximal Policy Optimization agent for trading
- ESPopulation: Population of agents for evolutionary strategies
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
import os
import contextlib  # Add contextlib for nullcontext
import time  # Add time module for lesson memory
import gc  # Add gc module for memory optimization

# Add new imports for adaptive learning features
from torch.nn.utils import prune
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import importlib

# Import modules dynamically
try:
    # Import tensor utils
    tensor_utils_module = importlib.import_module("src.utils.tensor_utils")
    detect_market_regime = tensor_utils_module.detect_market_regime
    compute_fractal_dimension_tensor = tensor_utils_module.compute_fractal_dimension_tensor
    
    # Import models
    models_module = importlib.import_module("src.models.models")
    create_model = models_module.create_model
    
    # Import utils
    utils_module = importlib.import_module("src.utils.utils")
    log = utils_module.log
    optimize_memory = utils_module.optimize_memory
    
    # Import reasoning analyzer
    try:
        reasoning_module = importlib.import_module("src.utils.reasoning")
        ReasoningAnalyzer = reasoning_module.ReasoningAnalyzer
    except ImportError:
        # If reasoning module is not available, define a stub
        class ReasoningAnalyzer:
            def __init__(self, *args, **kwargs):
                pass
            def analyze(self, *args, **kwargs):
                return {"reasoning": "Reasoning analysis not available"}
except Exception as e:
    print(f"Error importing modules: {e}")
    # Define fallback components if imports fail
    def log(message, level="info"):
        print(f"[{level.upper()}] {message}")
    
    def optimize_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    class ReasoningAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
        def analyze(self, *args, **kwargs):
            return {"reasoning": "Reasoning analysis not available"}

# Configure logging
import logging

# Import log manager dynamically
try:
    log_manager_module = importlib.import_module("src.utils.log_manager")
    LogManager = log_manager_module.LogManager
    log_exception = log_manager_module.log_exception
    
    # Set up agent logger
    logger = LogManager.get_logger('agent')
except ImportError:
    # Define fallback logging if log_manager import fails
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('agent')
    
    def log_exception(e, context=""):
        logger.error(f"Exception in {context}: {str(e)}")
    
    class LogManager:
        @staticmethod
        def get_logger(name):
            return logging.getLogger(name)

# Export the agent classes for other modules to import
__all__ = ['PPOAgent', 'ESPopulation']

class PrioritizedReplayBuffer:
    """
    Replay buffer with prioritization based on surprise/importance of experiences.
    
    This allows the agent to focus more on unusual, surprising, or important experiences,
    creating a more naturalistic learning process similar to how humans remember unusual events.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize a prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform sampling, 1 = fully prioritized)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Amount to increase beta each time buffer is sampled
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done, error=None):
        """
        Add an experience to the buffer with priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            error: TD error or surprise measure (if None, use max priority)
        """
        # Compute max priority for new experience
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # Store experience
        self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # Set priority based on error or maximum value
        if error is not None:
            priority = (abs(error) + 1e-5) ** self.alpha
        else:
            priority = self.max_priority
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences with prioritization.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        # Only sample from filled part of buffer
        buffer_len = min(len(self.buffer), self.capacity)
        if buffer_len == 0:
            return None
        
        # Cannot sample more than buffer size
        batch_size = min(batch_size, buffer_len)
        
        # Get sampling probabilities from priorities
        priorities = self.priorities[:buffer_len]
        probs = priorities / np.sum(priorities)
        
        # Sample according to priorities
        indices = np.random.choice(buffer_len, batch_size, replace=False, p=probs)
        
        # Calculate importance sampling weights
        # These correct for bias introduced by prioritized sampling
        weights = (buffer_len * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Increment beta for annealing (gradually increasing correction)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Extract experiences
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, weights)
    
    def update_priorities(self, indices, errors):
        """
        Update priorities based on new error estimates.
        
        Args:
            indices: Indices of experiences to update
            errors: New TD errors or surprise measures
        """
        for idx, error in zip(indices, errors):
            if idx < len(self.priorities):
                priority = (abs(error) + 1e-5) ** self.alpha
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """Return the number of experiences in the buffer."""
        return len(self.buffer)

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent with enhanced learning capabilities.
    
    This agent implementation incorporates probabilistic price predictions and
    multiple enhancements for more naturalistic learning.
    """
    def __init__(self, input_dim, hidden_size, lr, horizons=None, use_mixed_precision=False,
                 model_type='actor_critic', device="cpu", config=None):
        """
        Initialize PPOAgent.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_size (int): Size of hidden layers
            lr (float): Learning rate
            horizons (list, optional): Prediction horizons. Defaults to None.
            use_mixed_precision (bool, optional): Whether to use mixed precision. Defaults to False.
            model_type (str, optional): Type of model to use. Defaults to 'actor_critic'.
            device (str, optional): Device to use. Defaults to "cpu".
            config (dict, optional): Additional configuration parameters. Defaults to None.
        """
        self.config = config or {}
        self.device = device
        
        # Initialize model
        self.model = create_model(input_dim, hidden_size, horizons=horizons,
                                  model_type=model_type, device=device, config=self.config)
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Initialize learning rate scheduler if enabled
        if self.config.get("USE_LR_SCHEDULING", False):
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=self.config.get("LR_FACTOR", 0.5),
                patience=self.config.get("LR_PATIENCE", 10),
                verbose=True
            )
        
        # Initialize mixed precision if requested
        self.use_mixed_precision = use_mixed_precision
        
        # Create old model for PPO
        self.old_model = create_model(input_dim, hidden_size, horizons=horizons,
                                      model_type=model_type, device=device, config=self.config)
        self.old_model.to(device)
        self.old_model.load_state_dict(self.model.state_dict())
        
        # Initialize exploration parameters
        self.exploration_rate = self.config.get("INITIAL_EXPLORATION", 0.5)
        self.min_exploration = self.config.get("MIN_EXPLORATION", 0.05)
        self.exploration_decay = self.config.get("EXPLORATION_DECAY", 0.995)
        self.exploration_decay_method = self.config.get("EXPLORATION_DECAY_METHOD", "adaptive")
        
        # Track reward history for adaptive exploration
        self.recent_rewards = deque(maxlen=self.config.get("REWARD_HISTORY_SIZE", 100))
        
        # Track market regimes
        self.market_regimes = {
            "trending": 0.25,
            "ranging": 0.25,
            "volatile": 0.25,
            "mixed": 0.25
        }
        
        # Track prediction confidence
        self.confidence_history = deque(maxlen=50)
        
        # Initialize loss stats
        self.actor_loss_stats = deque(maxlen=10)
        self.critic_loss_stats = deque(maxlen=10)
        self.prediction_loss_stats = {}
        
        # Initialize adaptive learning mode
        self.use_adaptive_exploration = self.config.get("USE_ADAPTIVE_EXPLORATION", True)
        self.use_adaptive_features = self.config.get("USE_ADAPTIVE_FEATURES", True)
        
        # Initialize surprise-based replay
        self.use_surprise_based_replay = self.config.get("USE_SURPRISE_BASED_REPLAY", True)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.get("REPLAY_BUFFER_SIZE", 10000),
            alpha=self.config.get("REPLAY_ALPHA", 0.6),
            beta=self.config.get("REPLAY_BETA", 0.4),
            beta_increment=self.config.get("REPLAY_BETA_INCREMENT", 0.001)
        )
        self.surprise_threshold = self.config.get("SURPRISE_THRESHOLD", 0.7)
        
        # Initialize post-trade analysis
        self.use_post_trade_analysis = self.config.get("USE_POST_TRADE_ANALYSIS", True)
        self.lesson_memory = LessonMemory(capacity=self.config.get("LESSON_MEMORY_SIZE", 100))
        
        # Initialize meta-learning
        self.use_meta_learning = self.config.get("USE_META_LEARNING", True)
        self.hyperparameter_optimizer = HyperparamOptimizer(
            config=self.config,
            tunable_params=self.config.get("TUNABLE_HYPERPARAMS", None)
        )
        
        # Initialize contextual memory
        self.use_contextual_memory = self.config.get("USE_CONTEXTUAL_MEMORY", True)
        if self.use_contextual_memory:
            self.contextual_memory = ContextualMemory(
                capacity=self.config.get("CONTEXTUAL_MEMORY_SIZE", 200),
                similarity_threshold=self.config.get("MEMORY_SIMILARITY_THRESHOLD", 0.7),
                feature_dim=input_dim
            )
        
        # Track recent trades for post-analysis
        self.recent_trades = deque(maxlen=100)
        
        # Initialize feature importance map
        self.feature_importance = np.ones(input_dim)
        
        # Initialize additional hyperparameters
        self.clip_ratio = self.config.get("PPO_CLIP_RATIO", 0.2)
        self.ppo_epochs = self.config.get("PPO_EPOCHS", 4)
        self.value_loss_coef = self.config.get("VALUE_LOSS_COEF", 0.5)
        self.entropy_coef = self.config.get("ENTROPY_COEF", 0.01)
        self.max_grad_norm = self.config.get("MAX_GRAD_NORM", 0.5)
        self.gamma = self.config.get("GAMMA", 0.99) # Add gamma attribute
        self.batch_size = self.config.get("BATCH_SIZE", 128) # Add batch_size attribute
        
        # Initialize market regime detection
        self.use_regime_detection = self.config.get("USE_REGIME_DETECTION", True)
        self.fractal_window_size = self.config.get("FRACTAL_WINDOW_SIZE", 30)
        self.regime_buffer = {
            "trending": [],
            "ranging": [],
            "volatile": [],
            "mixed": []
        }
        self.regime_models = {}
        self.general_model = self.model  # Store reference to general model
        self.current_model_regime = 'general'
        
        # Initialize dynamic pruning
        self.use_dynamic_pruning = self.config.get("USE_DYNAMIC_PRUNING", False)
        self.pruning_interval = self.config.get("PRUNING_INTERVAL", 20)
        self.prune_count = 0
        
        # Initialize learning rate scheduling
        self.use_lr_scheduling = self.config.get("USE_LR_SCHEDULING", False)
        
        # Initialize reasoning style
        self.reasoning_style = self.config.get("REASONING_STYLE", "balanced")
        self.initialize_style_specific_reasoning()
        
        # Logger for tracking performance metrics
        self.logger = logging.getLogger(f'agent-{id(self)}')
        self.logger.setLevel(logging.INFO)
        
        # Ensure we have a handler
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)
        
        # Log initialization
        self.logger.info(f"Initialized {model_type} agent with {input_dim} inputs, "\
                         f"{hidden_size} hidden units on {device}")
    
    def initialize_style_specific_reasoning(self):
        """
        Initialize trading style-specific reasoning parameters.
        Different trading buckets have different reasoning focuses.
        """
        if self.reasoning_style == "scalping":
            # Scalping focuses on short-term patterns, liquidity, and entry timing
            self.reasoning_weights = {
                'market_regime': 0.1,      # Less important for scalping
                'patterns': 0.15,          # Medium importance for pattern recognition
                'support_resistance': 0.2,  # High importance for levels
                'volatility': 0.15,        # Medium importance
                'liquidity': 0.25,         # Very high importance for scalping
                'entry_exit': 0.15,        # Medium importance
            }
            log("Initialized scalping-specific reasoning weights")
        
        elif self.reasoning_style == "short":
            # Short-term focuses on patterns and support/resistance
            self.reasoning_weights = {
                'market_regime': 0.15,     # Medium importance
                'patterns': 0.25,          # High importance
                'support_resistance': 0.25, # High importance
                'volatility': 0.1,         # Lower importance
                'liquidity': 0.1,          # Lower importance
                'entry_exit': 0.15,        # Medium importance
            }
            log("Initialized short-term specific reasoning weights")
        
        elif self.reasoning_style == "medium":
            # Medium-term focuses on regime and trend detection
            self.reasoning_weights = {
                'market_regime': 0.25,     # High importance
                'patterns': 0.15,          # Medium importance
                'support_resistance': 0.15, # Medium importance
                'volatility': 0.15,        # Medium importance
                'liquidity': 0.05,         # Low importance
                'entry_exit': 0.25,        # High importance for timing
            }
            log("Initialized medium-term specific reasoning weights")
        
        elif self.reasoning_style == "long":
            # Long-term focuses primarily on regime and fundamentals
            self.reasoning_weights = {
                'market_regime': 0.35,     # Very high importance
                'patterns': 0.05,          # Low importance
                'support_resistance': 0.1,  # Low importance
                'volatility': 0.25,        # High importance for risk management
                'liquidity': 0.05,         # Low importance
                'entry_exit': 0.2,         # Medium importance
            }
            log("Initialized long-term specific reasoning weights")
        else:
            # Balanced default configuration
            self.reasoning_weights = {
                'market_regime': 0.15,
                'patterns': 0.15,
                'support_resistance': 0.15,
                'volatility': 0.15,
                'liquidity': 0.15,
                'entry_exit': 0.2,
            }
            log("Initialized default balanced reasoning weights")

    def update_old_model(self):
        """Copy current model parameters to old model"""
        self.old_model.load_state_dict(self.model.state_dict())

    def _get_exploration_factor(self, state=None):
        """
        Calculate dynamic exploration factor based on recent performance and market conditions.
        
        This creates more naturalistic learning where the agent explores more in uncertain markets
        and exploits more in well-understood market conditions.
        
        Args:
            state: Current state tensor for market regime assessment
            
        Returns:
            float: Adjusted exploration factor (0.0-1.0)
        """
        base_factor = self.exploration_rate
        
        if self.exploration_decay_method == "time":
            # Simply decay over time
            return base_factor
        
        # Get market regime if state is provided
        if state is not None and hasattr(self.model, 'market_regime_head'):
            with torch.no_grad():
                state_batched = state.unsqueeze(0) if state.dim() == 2 else state
                # Get market regime prediction
                regime_outputs = self.model.market_regime_head(state_batched[:, -1])  # Use last timestep
                
                # Update regime tracking (exponential moving average)
                alpha = 0.1  # Smoothing factor
                for i, regime in enumerate(["trending", "ranging", "volatile", "mixed"]):
                    self.market_regimes[regime] = alpha * regime_outputs[0, i].item() + (1 - alpha) * self.market_regimes[regime]
                
                # Adjust exploration based on market regime
                # Explore more in ranging, less in trending
                trending_factor = 0.8 * self.market_regimes["trending"]  # 0.8x exploration in trending 
                ranging_factor = 1.2 * self.market_regimes["ranging"]    # 1.2x exploration in ranging
                volatile_factor = 1.0 * self.market_regimes["volatile"]  # Normal exploration in volatile
                mixed_factor = 1.0 * self.market_regimes["mixed"]        # Normal exploration in mixed
                
                # Weight total adjustment by strongest regime
                regime_adjustment = trending_factor + ranging_factor + volatile_factor + mixed_factor
        else:
            # Default adjustment if no regime information
            regime_adjustment = 1.0
        
        # Performance-based adjustment
        if self.recent_rewards and self.exploration_decay_method in ["performance", "adaptive"]:
            # If rewards are consistently improving, reduce exploration
            # If rewards are declining or inconsistent, increase exploration
            
            # Calculate trend in recent rewards
            if len(self.recent_rewards) >= 5:
                recent_trend = sum(x - y for x, y in zip(list(self.recent_rewards)[1:], list(self.recent_rewards)[:-1]))
                normalized_trend = max(-1.0, min(1.0, recent_trend / (max(self.recent_rewards) - min(self.recent_rewards) + 1e-8)))
                
                # Adjust exploration based on trend
                if normalized_trend > 0.2:  # Consistent improvement
                    performance_factor = 0.85  # Reduce exploration to exploit good strategy
                elif normalized_trend < -0.2:  # Decline in performance
                    performance_factor = 1.2   # Increase exploration to find better strategy
                else:
                    performance_factor = 1.0   # No change
            else:
                performance_factor = 1.0
        else:
            performance_factor = 1.0

        # Confidence-based adjustment (if we have confidence data)
        if self.confidence_history and len(self.confidence_history) > 10:
            # If agent is very confident, reduce exploration
            # If agent is uncertain, increase exploration
            avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
            confidence_factor = 1.5 - avg_confidence  # Lower confidence = more exploration
        else:
            confidence_factor = 1.0
        
        # Combine adjustments
        adjusted_exploration = base_factor * regime_adjustment * performance_factor * confidence_factor
        
        # Clip to ensure we're within bounds
        adjusted_exploration = max(self.min_exploration, min(1.0, adjusted_exploration))
        
        return adjusted_exploration
    
    def _get_future_outcome_prediction(self, state):
        """
        Internal method to get the predicted outcome for a model-determined horizon.
        Returns the initial prediction dict, the horizon H, and the first model call's outputs.
        """
        log = get_logger(__name__) # Ensure logger is available
        
        # --- First Model Call (Get Horizon Distribution + Base Outputs) ---
        with torch.no_grad():
            # NOTE: Ensure self.model is the correct model instance
            # Pass requested_horizon=None initially
            model_outputs_horizon = self.model(state, requested_horizon=None)

        # --- Extract Horizon Distribution ---
        horizon_mean_tensor = model_outputs_horizon.get("horizon_mean")
        
        # --- Determine Target Horizon H ---
        if horizon_mean_tensor is not None:
            # Use mean across batch if needed, ensure integer >= 1
            horizon_h = max(1, round(horizon_mean_tensor.mean().item()))
        else:
            # Fallback if model doesn't provide it
            horizon_h = self.config.get("DEFAULT_PREDICTION_HORIZON", 20)
            log(f"[WARNING] Horizon mean not found in model output, using default H={horizon_h}.", "warning")

        # --- Second Model Call (Get Outcome Prediction for Horizon H) ---
        with torch.no_grad():
            # NOTE: Ensure self.model is the correct model instance
            outcome_outputs = self.model(state, requested_horizon=horizon_h)

        # Extract outcome prediction for horizon H
        outcome_mean = outcome_outputs.get("outcome_mean")
        outcome_std = outcome_outputs.get("outcome_std")
        outcome_confidence = outcome_outputs.get("outcome_confidence")

        initial_outcome_pred = {
            "mean": outcome_mean.item() if outcome_mean is not None else None,
            "std": outcome_std.item() if outcome_std is not None else None,
            "confidence": outcome_confidence.item() if outcome_confidence is not None else None
        }
        
        # Return prediction, horizon, and outputs from first call (needed for action selection)
        return initial_outcome_pred, horizon_h, model_outputs_horizon 

    def select_action(self, state, hidden=None, contextual_guidance=None, is_new_episode=False, current_step=None):
        log = get_logger(__name__) # Ensure logger is available
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if hidden is not None:
            hidden = tuple(h.to(self.device) for h in hidden)
        
        # --- Get Prediction and First Model Outputs via Internal Method ---
        initial_outcome_pred, horizon_h, model_outputs = self._get_future_outcome_prediction(state)

        # --- Extract Base Outputs from First Model Call Outputs ---
        # These were returned by _get_future_outcome_prediction
        actions_mean = model_outputs.get("actions_mean")
        actions_logstd = model_outputs.get("actions_logstd")
        value = model_outputs.get("value")

        # Determine exploration factor (Placeholder - adjust as needed)
        # Example: Lower exploration if confidence in prediction is high?
        exploration_factor = 1.0 
        # if initial_outcome_pred and initial_outcome_pred.get('confidence'):
        #     exploration_factor = max(0.1, 1.0 - initial_outcome_pred['confidence'] * 0.5) 

        # --- Apply Contextual Guidance (If provided) ---
        if contextual_guidance is not None and actions_mean is not None:
            # Simple weight, could be more sophisticated
            contextual_weight = 1.0 - exploration_factor 
            actions_mean = (1.0 - contextual_weight) * actions_mean + contextual_weight * contextual_guidance

        # --- Action Sampling (Using base outputs from first model call) ---
        if actions_mean is not None and actions_logstd is not None:
            actions_std = torch.exp(actions_logstd)
            # Apply exploration factor to std dev for sampling
            action_dist = Normal(actions_mean, actions_std * exploration_factor) 
            action = action_dist.sample()
            # Sum log probs if action space is multi-dimensional, otherwise don't sum
            if action.ndim > 1 and action.shape[-1] > 1:
                 log_prob = action_dist.log_prob(action).sum(axis=-1)
            else:
                 log_prob = action_dist.log_prob(action)
        else:
            log("[ERROR] Missing actions_mean or actions_logstd in model output.", "error")
            # Fallback if model outputs are missing
            action = torch.zeros(self.action_dim).to(self.device) # Use self.action_dim if available
            log_prob = torch.tensor(-float('inf')).expand(action.shape[0] if action.ndim > 0 else 1).to(self.device)
            value = torch.zeros_like(log_prob) if value is None else value # Handle None value

        # --- Store Plan State (If initiating a new plan) ---
        # Ensure action is on CPU and numpy for this logic
        action_np_full = action.detach().cpu().numpy()
        # Handle potential scalar action if action dim is 1
        action_value = action_np_full[0][0] if action_np_full.ndim > 1 and action_np_full.shape[1] > 0 else action_np_full[0]

        # Reset plan at the start of a new episode
        if is_new_episode:
             self.active_plan = False
             self.plan_entry_time = None
             self.plan_target_time = None
             self.plan_mid_check_time = None
             self.plan_initial_horizon_h = None
             self.plan_initial_outcome_pred = None # Reset prediction storage

        if current_step is not None:
            # Assuming action_value != 0 signifies wanting a position (long/short)
            # Plan initiated based on the SAMPLED action, using prediction info obtained earlier
            if not self.active_plan and action_value != 0: # Entering a new position
                self.active_plan = True
                self.plan_entry_time = current_step
                self.plan_initial_horizon_h = horizon_h # Use H from _get_future_outcome_prediction
                self.plan_target_time = current_step + horizon_h
                # Ensure mid-check is at least 1 step after entry
                self.plan_mid_check_time = current_step + max(1, horizon_h // 2)
                self.plan_initial_outcome_pred = initial_outcome_pred # Use pred from _get_future_outcome_prediction
                log(f"[PLAN INIT] Step {current_step}: Action {action_value:.2f}. Plan H={horizon_h}, Target={self.plan_target_time}, Mid={self.plan_mid_check_time}", "debug")
            elif self.active_plan and action_value == 0: # Closing position, ending plan
                log(f"[PLAN END] Step {current_step}: Closing action. Plan ended.", "debug")
                self.active_plan = False
                self.plan_entry_time = None
                self.plan_target_time = None
                self.plan_mid_check_time = None
                self.plan_initial_horizon_h = None
                self.plan_initial_outcome_pred = None # Clear prediction storage
            # Else: Holding position (active_plan=True, action!=0) or Flat (active_plan=False, action=0) - no state change needed here
        # --- End Plan State Management ---

        # --- Prepare Return Values ---
        # Ensure tensors are detached, moved to CPU, and converted to numpy for return
        final_action_np = action_np_full[0] if action_np_full.ndim > 0 else action_np_full # Handle scalar case
        # Ensure log_prob and value are scalars if batch size is 1 and originally scalar/single value
        final_log_prob_np = log_prob.detach().cpu().numpy()[0] if log_prob.ndim > 0 else log_prob.detach().cpu().numpy()
        final_value_np = value.detach().cpu().numpy()[0] if value is not None and value.ndim > 0 else (value.detach().cpu().numpy() if value is not None else 0.0) # Handle None value

        # Include current plan info in return dict (using stored state)
        action_info = {
             'active_plan': self.active_plan,
             'plan_target_time': self.plan_target_time,
             'plan_mid_check_time': self.plan_mid_check_time,
             'plan_horizon': self.plan_initial_horizon_h, # Use the stored horizon from plan init
             'initial_outcome_pred': self.plan_initial_outcome_pred # Use the stored prediction from plan init
        }

        # Record sampled action for memory (if needed) - Placed before final return
        if self.use_contextual_memory:
            try:
                 # Store the sampled action
                 self._current_action = final_action_np 
            except Exception as e:
                 log(f"[ERROR] Could not store action for contextual memory: {e}", "error")
                 self._current_action = None

            # Store the state that led to this action
            try:
                # state is the input tensor, detach and move to cpu/numpy
                self._current_state = state.detach().cpu().numpy()[0] # Assuming batch dim was added
            except Exception as e:
                log(f"[ERROR] Could not store state for contextual memory: {e}", "error")
                self._current_state = None

        # Return the sampled action, its log prob, state value, and current plan info
        return final_action_np, final_log_prob_np, final_value_np, action_info
    
    def predict_batch(self, states, batch_size=32):
        """
        Make batch predictions for multiple states.
        
        Args:
            states (list): List of state observations.
            batch_size (int, optional): Batch size for prediction. Defaults to 32.
            
        Returns:
            list: List of prediction results.
        """
        if not states:
            return []
        
        results = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(states), batch_size):
            batch = states[i:i+batch_size]
            
            # Convert states to tensors
            batch_tensors = []
            for s in batch:
                if isinstance(s, torch.Tensor):
                    batch_tensors.append(s.to(self.device))
                else:
                    batch_tensors.append(torch.from_numpy(s).float().to(self.device))
            
            batch_tensor = torch.stack(batch_tensors)
            
            with torch.no_grad():
                # Get model outputs
                outputs = self.old_model(batch_tensor, None)
            
            # Extract outputs
            mean = outputs['actions_mean']
            log_std = outputs['actions_logstd']
            val = outputs['value']
            pred_means = outputs['predictions']
            pred_stds = outputs['prediction_stds']
            confs = outputs['confidence']
            mid_means = outputs['mid_point_means']
            mid_stds = outputs['mid_point_stds']
            trend = outputs['trend_strength']
            
            # Sample actions
            dist = Normal(mean, torch.exp(log_std))
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=1)
            
            # Store results
            for j in range(len(batch)):
                # Convert dictionary outputs to numpy for this batch item
                pred_means_j = {h: pred_means[h][j].cpu().numpy() for h in pred_means}
                pred_stds_j = {h: pred_stds[h][j].cpu().numpy() for h in pred_stds}
                confs_j = {h: confs[h][j].cpu().numpy() for h in confs}
                mid_means_j = {h: mid_means[h][j].cpu().numpy() for h in mid_means}
                mid_stds_j = {h: mid_stds[h][j].cpu().numpy() for h in mid_stds}
                
                results.append((
                    actions[j].cpu().numpy(),
                    log_probs[j].item(),
                    val[j].item(),
                    pred_means_j,
                    pred_stds_j,
                    confs_j,
                    mid_means_j,
                    mid_stds_j,
                    trend[j].cpu().item()
                ))
        
        return results
    
    def update_feature_importance(self, states, actions, rewards):
        """
        Update feature importance based on mutual information with rewards.
        
        Args:
            states (torch.Tensor): Batch of state observations
            actions (torch.Tensor): Batch of actions taken
            rewards (torch.Tensor): Batch of rewards received
        """
        if not self.use_adaptive_features or len(states) < 100:
            return
        
        try:
            # Convert to numpy for sklearn
            states_np = states.cpu().numpy()
            rewards_np = rewards.cpu().numpy()
            
            # Calculate mutual information between features and rewards
            mi_scores = mutual_info_regression(states_np, rewards_np, random_state=42)
            
            # Normalize and convert to tensor
            if np.sum(mi_scores) > 0:
                mi_scores = mi_scores / np.sum(mi_scores) * len(mi_scores)
            new_importance = torch.tensor(mi_scores, device=self.device).float()
            
            # Update feature importance with exponential moving average
            alpha = 0.3  # Weight for new importance
            self.feature_importance = (1 - alpha) * self.feature_importance + alpha * new_importance
            
            # Store history for tracking
            self.feature_importance_history.append(self.feature_importance.cpu().numpy())
            
            # Apply feature importance in forward pass (handled in model)
            self.model.update_feature_weights(self.feature_importance)
        except Exception as e:
            log(f"Error in feature importance update: {str(e)}")
    
    def detect_regime(self, price_history, volume_history=None):
        """
        Detect current market regime to adapt trading strategy.
        
        Args:
            price_history (torch.Tensor): Recent price history
            volume_history (torch.Tensor, optional): Recent volume history
            
        Returns:
            str: Detected regime ('trending', 'ranging', 'volatile', 'mixed')
        """
        if not self.use_regime_detection or price_history is None or len(price_history) < self.fractal_window_size:
            return "mixed"  # Default regime
        
        try:
            # Use tensor utility function for regime detection
            regime_info = detect_market_regime(
                price_history[-self.fractal_window_size:],
                volume_history[-self.fractal_window_size:] if volume_history is not None else None,
                window_size=self.fractal_window_size,
                device=self.device
            )
            
            # Get regime label
            self.current_regime = regime_info['regime_label']
            
            # Log regime change if changed
            if hasattr(self, 'prev_regime') and self.prev_regime != self.current_regime:
                log(f"Regime changed from {self.prev_regime} to {self.current_regime}")
                log(f"Regime metrics: trend_strength={regime_info['trend_strength']:.3f}, " +
                    f"range_strength={regime_info['range_strength']:.3f}, " +
                    f"volatility={regime_info['volatility']:.6f}")
            
            self.prev_regime = self.current_regime
            return self.current_regime
        except Exception as e:
            log(f"Error in regime detection: {str(e)}")
            return "mixed"  # Default regime
    
    def update(self, states, actions, log_probs, returns, advantages,
               prediction_targets=None, prediction_data=None, calibrate_uncertainty=False):
        """
        Update policy with PPO algorithm.
        
        Args:
            states (torch.Tensor): States from trajectories.
            actions (torch.Tensor): Actions from trajectories.
            log_probs (torch.Tensor): Action log probabilities from trajectories.
            returns (torch.Tensor): Returns from trajectories.
            advantages (torch.Tensor): Advantages from trajectories.
            prediction_targets (dict, optional): Actual target values for predictions.
            prediction_data (dict, optional): Prediction data (mean, std, conf) from trajectories.
            calibrate_uncertainty (bool, optional): Whether to calibrate uncertainty. Defaults to False.
            
        Returns:
            dict: Loss metrics.
        """
        # Prepare for training
        batch_size = self.batch_size
        num_samples = states.size(0)
        
        # Set model to training mode
        self.model.train()
        
        # Track losses
        total_losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []
        prediction_losses = []
        
        # Main training loop (PPO epochs)
        for _ in range(self.ppo_epochs):
            # Generate random indices
            indices = torch.randperm(num_samples).to(self.device)
            
            # Process in minibatches
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                minibatch_indices = indices[start_idx:end_idx]
                
                # Get minibatch data
                mb_states = states[minibatch_indices]
                mb_actions = actions[minibatch_indices]
                mb_old_log_probs = log_probs[minibatch_indices]
                mb_returns = returns[minibatch_indices]
                mb_advantages = advantages[minibatch_indices]
                
                # Forward pass
                with self.amp.autocast() if self.use_mixed_precision else contextlib.nullcontext():
                    # Get model outputs
                    outputs = self.model(mb_states)
                    action_mean = outputs['actions_mean']
                    action_log_std = outputs['actions_logstd']
                    values = outputs['value'].squeeze()
                    
                    # Get prediction outputs
                    pred_means = outputs['predictions']
                    pred_stds = outputs['prediction_stds']
                    
                    # Calculate action distribution
                    dist = Normal(action_mean, torch.exp(action_log_std))
                    
                    # Calculate entropy (for exploration)
                    entropy = dist.entropy().mean()
                    
                    # Calculate new log probabilities
                    new_log_probs = dist.log_prob(mb_actions).sum(1)
                    
                    # Calculate ratios for PPO
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    
                    # Calculate surrogate loss
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantages
                    
                    # Calculate policy loss (negative to maximize)
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Calculate value loss
                    value_loss = F.mse_loss(values, mb_returns)
                    
                    # Calculate prediction loss (if targets provided)
                    prediction_loss = torch.tensor(0.0, device=self.device)
                    
                    if prediction_targets is not None:
                        # Calculate negative log likelihood loss for each horizon
                        for horizon_name, target in prediction_targets.items():
                            if horizon_name in pred_means and horizon_name in pred_stds:
                                # Get predictions for this horizon
                                horizon_mean = pred_means[horizon_name].squeeze()
                                horizon_std = pred_stds[horizon_name].squeeze()
                                
                                # Get targets for this batch
                                if isinstance(target, torch.Tensor):
                                    horizon_target = target[minibatch_indices].to(self.device)
                                else:
                                    horizon_target = torch.tensor(target, device=self.device)
                                
                                # Create normal distribution
                                pred_dist = Normal(horizon_mean, horizon_std)
                                
                                # Calculate negative log likelihood
                                nll = -pred_dist.log_prob(horizon_target)
                                
                                # Add to prediction loss
                                prediction_loss = prediction_loss + nll.mean()
                    
                    # Calibrate uncertainty if requested
                    if calibrate_uncertainty and hasattr(self.model, 'update_calibration_params'):
                        for horizon_name, target in prediction_targets.items():
                            if horizon_name in pred_means and horizon_name in pred_stds:
                                # Extract horizon number from name (e.g., 'h12' -> 12)
                                horizon = int(horizon_name[1:])
                                
                                # Check coverage of prediction intervals
                                mean = pred_means[horizon_name].detach().cpu().numpy()
                                std = pred_stds[horizon_name].detach().cpu().numpy()
                                target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
                                
                                # Count how many targets fall within 1-sigma interval
                                in_1sigma = np.sum(np.abs(mean - target_np) < std)
                                coverage_1sigma = in_1sigma / len(target_np)
                                
                                # Calculate calibration scale and bias
                                # Ideal coverage is 68% for 1-sigma
                                target_coverage = 0.68
                                if coverage_1sigma > 0:
                                    # If coverage too high, increase std (scale > 1)
                                    # If coverage too low, decrease std (scale < 1)
                                    scale = target_coverage / coverage_1sigma
                                    bias = 0.0  # Start with no bias
                                
                                # Update model calibration parameters
                                self.model.update_calibration_params(horizon, scale, bias)
                    
                    # Combine losses
                    loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                    
                    # Add prediction loss if available
                    if prediction_loss.item() != 0:
                        prediction_weight = self.config.get("PREDICTION_LOSS_WEIGHT", 0.5)
                        loss = loss + prediction_weight * prediction_loss
                    
                    # Backpropagate
                    if self.use_mixed_precision:
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.optimizer.step()
                    
                    # Record losses
                    total_losses.append(loss.item())
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy.item())
                    if prediction_loss.item() != 0:
                        prediction_losses.append(prediction_loss.item())
                
                # Update old model
                self.update_old_model()
            
            # Compute average losses
            avg_total_loss = sum(total_losses) / len(total_losses) if total_losses else 0
            avg_policy_loss = sum(policy_losses) / len(policy_losses) if policy_losses else 0
            avg_value_loss = sum(value_losses) / len(value_losses) if value_losses else 0
            avg_entropy_loss = sum(entropy_losses) / len(entropy_losses) if entropy_losses else 0
            avg_prediction_loss = sum(prediction_losses) / len(prediction_losses) if prediction_losses else 0
            
            # Update error threshold for adaptive training
            self.recent_errors.append(avg_total_loss)
            self.error_threshold = np.mean(self.recent_errors) * 1.5 if self.recent_errors else 1.0
            
            # Return metrics
            return {
                'total_loss': avg_total_loss,
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'entropy_loss': avg_entropy_loss,
                'prediction_loss': avg_prediction_loss
            }
    
    def update_with_rollouts(self, rollouts):
        """
        Update agent using PPO algorithm with adaptive features.
        
        Args:
            rollouts (dict): Collected rollout data with states, actions, rewards, etc.
            
        Returns:
            dict: Training metrics
        """
        # Detect current market regime if enabled
        if self.use_regime_detection and 'price_history' in rollouts:
            regime = self.detect_regime(
                rollouts['price_history'],
                rollouts.get('volume_history', None)
            )
            
            # Store data in regime-specific buffer
            if regime in self.regime_buffer:
                for i in range(len(rollouts['states'])):
                    self.regime_buffer[regime].append({
                        'state': rollouts['states'][i],
                        'action': rollouts['actions'][i],
                        'reward': rollouts['rewards'][i],
                        'next_state': rollouts['next_states'][i],
                        'done': rollouts['dones'][i]
                    })
            
            # Check if we should use a regime-specific model
            if regime in self.regime_models and len(self.regime_buffer[regime]) > 100:
                # Use regime-specific model for update
                if self.current_model_regime != regime:
                    log(f"Switching to {regime} regime model")
                    self.model = self.regime_models[regime]
                    self.current_model_regime = regime
            else:
                # Use general model
                if self.current_model_regime != 'general':
                    log(f"Switching to general model from {self.current_model_regime}")
                    self.model = self.general_model
                    self.current_model_regime = 'general'
            
            # Extract rollout data
            states = rollouts['states']
            actions = rollouts['actions']
            rewards = rollouts['rewards']
            old_logps = rollouts.get('old_logps')
            returns = rollouts.get('returns')
            advantages = rollouts.get('advantages')
            mid_pred_targets = rollouts.get('mid_pred_targets')
            pred_targets = rollouts.get('pred_targets')
            
            # Update feature importance
            if self.use_adaptive_features and rewards is not None:
                self.update_feature_importance(states, actions, rewards)
            
            # Call original update method
            metrics = {}
            if old_logps is not None and returns is not None and advantages is not None:
                metrics = self.update(states, actions, old_logps, returns, advantages, mid_pred_targets, pred_targets)
            
            # Dynamic pruning if enabled
            if self.use_dynamic_pruning and self.prune_count % self.pruning_interval == 0:
                self.prune_model()
                self.prune_count += 1
            
            # Learning rate scheduling if enabled
            if self.use_lr_scheduling and hasattr(self, 'scheduler') and 'mean_reward' in rollouts:
                self.scheduler.step(rollouts['mean_reward'])
                metrics['lr'] = self.optimizer.param_groups[0]['lr']
            
            return metrics
    
    def prune_model(self):
        """
        Apply dynamic pruning to reduce model complexity and improve generalization.
        """
        if not self.use_dynamic_pruning:
            return
        
        try:
            # Only prune linear layers to maintain architecture
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Prune smallest weights by magnitude
                    prune.l1_unstructured(module, name='weight', amount=self.pruning_rate)
            
            # Make pruning permanent (remove masks)
            prune.remove(module, 'weight')
            
            log(f"Applied pruning with rate {self.pruning_rate}")
        except Exception as e:
            log(f"Error in model pruning: {str(e)}")
    
    def create_regime_models(self):
        """
        Create regime-specific models for different market conditions.
        """
        if not self.use_regime_detection:
            return
        
        try:
            # Store general model
            self.general_model = self.model
            self.current_model_regime = 'general'
            
            # Initialize regime-specific models if we have enough data
            for regime, buffer in self.regime_buffer.items():
                if len(buffer) > 200:  # Minimum data required
                    log(f"Creating model for {regime} regime with {len(buffer)} samples")
                    
                    # Create new model with same architecture
                    regime_model = create_model(
                        self.input_dim,
                        self.hidden_size,
                        model_type=self.model_type,
                        horizons=self.horizons,
                        device=self.device
                    )
                    
                    # Copy weights from general model as starting point
                    regime_model.load_state_dict(self.general_model.state_dict())
                    
                    # Create optimizer for this model
                    regime_optimizer = optim.Adam(regime_model.parameters(), lr=self.lr)
                    
                    # Store model and optimizer
                    self.regime_models[regime] = regime_model
                    self.regime_optimizers = {regime: regime_optimizer}
                    
                    log(f"Created {len(self.regime_models)} regime-specific models")
        except Exception as e:
            log(f"Error creating regime models: {str(e)}")
    
    def save(self, path):
        """
        Save agent with all regime-specific models.
        
        Args:
            path (str): Base path to save models
        """
        # Save general model
        general_path = f"{path}_general.pt"
        torch.save({
            'model': self.general_model.state_dict() if hasattr(self, 'general_model') else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'feature_importance': self.feature_importance,
            'feature_history': self.feature_importance_history,
            'config': {
                'input_dim': self.input_dim,
                'hidden_size': self.hidden_size,
                'horizons': self.horizons,
                'model_type': self.model_type
            }
        }, general_path)
        
        # Save regime-specific models if available
        if hasattr(self, 'regime_models'):
            for regime, model in self.regime_models.items():
                regime_path = f"{path}_{regime}.pt"
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': self.regime_optimizers[regime].state_dict() if regime in self.regime_optimizers else None,
                    'config': {
                        'input_dim': self.input_dim,
                        'hidden_size': self.hidden_size,
                        'horizons': self.horizons,
                        'model_type': self.model_type,
                        'regime': regime
                    }
                }, regime_path)
    
    def update_learning_rate(self, reward):
        """
        Update learning rate based on performance.
        
        Args:
            reward (float): Reward metric for scheduler.
        """
        self.scheduler.step(reward)
        current_lr = self.optimizer.param_groups[0]['lr']
        log(f"Learning rate updated: {current_lr:.6f}")
    
    def apply_reward_adjustment(self, adjustment):
        """
        Apply reward adjustment from enhanced predictive evaluation.
        
        This method allows the enhanced predictive evaluation system to provide
        direct feedback to the agent based on prediction quality and horizon appropriateness.
        
        Args:
            adjustment (float): Reward adjustment value (positive for good predictions, negative for poor ones)
        """
        try:
            # Store the adjustment for potential future use
            if not hasattr(self, 'reward_adjustments'):
                self.reward_adjustments = []
            
            self.reward_adjustments.append({
                'adjustment': adjustment,
                'timestamp': time.time()
            })
            
            # Keep only recent adjustments (last 100)
            if len(self.reward_adjustments) > 100:
                self.reward_adjustments = self.reward_adjustments[-100:]
            
            # Apply adjustment to learning rate if significant
            if abs(adjustment) > 0.1:
                # Positive adjustment -> increase learning rate slightly
                # Negative adjustment -> decrease learning rate slightly
                adjustment_factor = 1.0 + (adjustment * 0.01)  # Small adjustment
                adjustment_factor = max(0.5, min(2.0, adjustment_factor))  # Clamp to reasonable range
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= adjustment_factor
                
                log(f"Applied reward adjustment: {adjustment:+.3f}, LR factor: {adjustment_factor:.3f}")
            
            # Optionally adjust prediction confidence or other model parameters
            if hasattr(self.model, 'apply_feedback'):
                self.model.apply_feedback(adjustment)
            
        except Exception as e:
            log(f"Error applying reward adjustment: {str(e)}")
    
    def get_reward_adjustment_history(self):
        """
        Get the history of reward adjustments from enhanced evaluation.
        
        Returns:
            list: List of reward adjustment records
        """
        if hasattr(self, 'reward_adjustments'):
            return self.reward_adjustments.copy()
        return []
    
    def save(self, path):
        """
        Save agent model and optimizer state.
        
        Args:
            path (str): Path to save the model.
            
        Returns:
            bool: True if save was successful, False otherwise.
        """
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'input_dim': self.input_dim,
                'hidden_size': self.hidden_size,
                'config': self.config
            }, path)
            log(f"Agent saved to {path}")
            return True
        except Exception as e:
            log(f"Failed to save agent: {e}")
            return False
    
    @classmethod
    def load(cls, path, device="cpu"):
        """
        Load agent from saved file.
        
        Args:
            path (str): Path to load the model from.
            device (str, optional): Device to load the model to. Defaults to "cpu".
            
        Returns:
            PPOAgent: Loaded agent or None if loading failed.
        """
        try:
            checkpoint = torch.load(path, map_location=device)
            input_dim = checkpoint.get('input_dim', 64)
            hidden_size = checkpoint.get('hidden_size', 128)
            config = checkpoint.get('config', {})
            
            # Create new agent
            agent = cls(input_dim, hidden_size, 1e-4, device=device, config=config)
            
            # Load model and optimizer state
            agent.model.load_state_dict(checkpoint['model_state_dict'])
            agent.update_old_model()
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler if available
            if 'scheduler_state_dict' in checkpoint:
                agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            log(f"Agent loaded from {path}")
            return agent
        except Exception as e:
            log(f"Failed to load agent from {path}: {e}")
            return None
    
    @classmethod
    def create_population(cls, num_agents, input_dim, hidden_size, lr, horizons=None,
                          use_mixed_precision=False, model_type='actor_critic', device="cpu", config=None):
        """
        Create a population of agents for evolutionary strategies.
        
        Args:
            num_agents (int): Number of agents to create.
            input_dim (int): Dimension of input features.
            hidden_size (int): Size of hidden layers.
            lr (float): Learning rate.
            horizons (list, optional): List of prediction horizons. Defaults to None.
            use_mixed_precision (bool, optional): Whether to use mixed precision training. Defaults to False.
            model_type (str, optional): Type of model to use. Defaults to 'actor_critic'.
            device (str, optional): Device to run on. Defaults to "cpu".
            config (dict, optional): Additional configuration parameters. Defaults to None.
            
        Returns:
            list: List of PPOAgent instances.
        """
        log(f"Creating population of {num_agents} agents")
        population = []
        
        for i in range(num_agents):
            agent = cls(input_dim, hidden_size, lr, horizons, use_mixed_precision, model_type, device, config)
            population.append(agent)
        
        # Free memory after each agent creation
        optimize_memory()
        
        return population
    
    def generate_prediction_samples(self, pred_means, pred_stds, num_samples=10):
        """
        Generate samples from the probabilistic predictions.
        
        Args:
            pred_means (numpy.ndarray): Mean predictions for each horizon.
            pred_stds (numpy.ndarray): Standard deviation predictions for each horizon.
            num_samples (int, optional): Number of samples to generate. Defaults to 10.
            
        Returns:
            numpy.ndarray: Samples from the prediction distributions with shape (num_samples, num_horizons).
        """
        if not isinstance(pred_means, torch.Tensor):
            pred_means = torch.tensor(pred_means, dtype=torch.float32)
        if not isinstance(pred_stds, torch.Tensor):
            pred_stds = torch.tensor(pred_stds, dtype=torch.float32)
        
        # Ensure proper device placement
        pred_means = pred_means.to(self.device)
        pred_stds = pred_stds.to(self.device)
        
        # Create distribution
        dist = Normal(pred_means, pred_stds)
        
        # Generate samples
        samples = dist.sample((num_samples,))
        
        return samples.cpu().numpy()
    
    def calculate_prediction_intervals(self, pred_means, pred_stds, confidence=0.95):
        """
        Calculate prediction intervals for each horizon.
        
        Args:
            pred_means (numpy.ndarray): Mean predictions for each horizon.
            pred_stds (numpy.ndarray): Standard deviation predictions for each horizon.
            confidence (float, optional): Confidence level (0-1). Defaults to 0.95 (95% CI).
            
        Returns:
            tuple: (lower_bounds, upper_bounds) for each horizon.
        """
        # Convert to numpy if tensors
        if isinstance(pred_means, torch.Tensor):
            pred_means = pred_means.cpu().numpy()
        if isinstance(pred_stds, torch.Tensor):
            pred_stds = pred_stds.cpu().numpy()
        
        # Z-score for the given confidence level (e.g., 1.96 for 95% CI)
        from scipy.stats import norm
        z_score = norm.ppf((1 + confidence) / 2)
        
        # Calculate bounds
        lower_bounds = pred_means - z_score * pred_stds
        upper_bounds = pred_means + z_score * pred_stds
        
        return lower_bounds, upper_bounds
    
    def visualize_predictions(self, pred_means, pred_stds, horizons=None, confidence_levels=[0.68, 0.95], file_path=None):
        """
        Visualize probabilistic predictions with confidence intervals.
        
        Args:
            pred_means (numpy.ndarray): Mean predictions for each horizon.
            pred_stds (numpy.ndarray): Standard deviation predictions for each horizon.
            horizons (list, optional): List of horizon steps. Defaults to model's horizons.
            confidence_levels (list, optional): Confidence levels to show. Defaults to [0.68, 0.95].
            file_path (str, optional): Path to save the figure. If None, displays the figure.
            
        Returns:
            matplotlib.figure.Figure: Figure object.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Check if matplotlib is interactive
            was_interactive = plt.isinteractive()
            if was_interactive:
                plt.ioff()  # Turn off interactive mode temporarily
            
            # Convert to numpy if tensors
            if isinstance(pred_means, torch.Tensor):
                pred_means = pred_means.cpu().numpy()
            if isinstance(pred_stds, torch.Tensor):
                pred_stds = pred_stds.cpu().numpy()
            
            # Use model's horizons if not provided
            if horizons is None:
                horizons = self.horizons or [12, 36, 72, 144]  # Default horizons
            
            if len(horizons) != len(pred_means):
                horizons = range(1, len(pred_means) + 1)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot mean prediction line
            ax.plot(horizons, pred_means, 'b-', label='Mean Prediction')
            
            # Color map for different confidence levels
            cmap = plt.cm.Blues
            n_levels = len(confidence_levels)
            
            # Plot confidence intervals from highest to lowest (for better visualization)
            confidence_levels = sorted(confidence_levels, reverse=True)
            for i, conf_level in enumerate(confidence_levels):
                lower, upper = self.calculate_prediction_intervals(pred_means, pred_stds, conf_level)
                color = cmap(0.5 + (i+1)/(n_levels+1))
                ax.fill_between(
                    horizons, lower, upper,
                    color=color, alpha=0.3,
                    label=f"{int(conf_level * 100)}% Confidence Interval"
                )
            
            # Generate random samples
            if len(pred_means) <= 10:  # Only show samples for small number of horizons
                samples = self.generate_prediction_samples(pred_means, pred_stds, num_samples=5)
                # Plot each sample path with light, transparent lines
                for i, sample in enumerate(samples):
                    ax.plot(horizons, sample, 'b-', alpha=0.15, linewidth=0.8)
            
            # Add labels and title
            ax.set_xlabel('Time Horizon')
            ax.set_ylabel('Predicted Value')
            ax.set_title('Probabilistic Price Predictions with Uncertainty')
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set y-axis limits with some padding
            all_values = np.concatenate([
                pred_means + 3*pred_stds,
                pred_means - 3*pred_stds
            ])
            y_range = max(all_values) - min(all_values)
            ax.set_ylim(
                min(all_values) - 0.1 * y_range,
                max(all_values) + 0.1 * y_range
            )
            
            plt.tight_layout()
            
            # Save or display
            if file_path:
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                log(f"Saved prediction visualization to {file_path}")
            else:
                if was_interactive:
                    plt.ion()  # Turn interactive mode back on if it was on
                plt.show()
            
            return fig
        except ImportError as e:
            log(f"Error visualizing predictions: {e}. Install matplotlib to use this feature.")
            return None
        except Exception as e:
            log(f"Error visualizing predictions: {e}")
            return None
    
    def calculate_surprise(self, state, action, reward, next_state, predictions=None):
        """
        Calculate how surprising an experience was based on prediction errors and reward.
        
        This provides a more naturalistic learning approach, focusing on unexpected outcomes.
        
        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            predictions: Agent's predictions if available
            
        Returns:
            float: Surprise score (0-1, higher = more surprising)
        """
        surprise_components = []
        
        # 1. Reward surprise - was reward much higher/lower than typical?
        if len(self.recent_rewards) > 0:
            mean_reward = np.mean(self.recent_rewards)
            std_reward = np.std(self.recent_rewards) + 1e-8
            reward_z_score = abs(reward - mean_reward) / std_reward
            reward_surprise = min(1.0, reward_z_score / 3.0)  # Cap at 1.0 (3 std devs)
            surprise_components.append(reward_surprise)
        
        # 2. Prediction error - was outcome different than predicted?
        if predictions is not None and isinstance(predictions, dict):
            prediction_errors = []
            for horizon, data in predictions.items():
                if 'mean' in data and next_state is not None:
                    # Simple check if we have price prediction and next price
                    pred_price = data['mean']
                    if hasattr(next_state, 'shape') and next_state.shape[0] > 0:
                        actual_price = next_state[-1, 0]  # Assuming price is first column
                        if hasattr(actual_price, 'item'):
                            actual_price = actual_price.item()
                        
                        # Normalized prediction error
                        if abs(pred_price) > 1e-8:
                            pred_error = abs(actual_price - pred_price) / abs(pred_price)
                            prediction_errors.append(min(1.0, pred_error))
            if prediction_errors:
                avg_pred_error = sum(prediction_errors) / len(prediction_errors)
                surprise_components.append(avg_pred_error)
        
        # 3. State novelty - was the next state unusual?
        if hasattr(self, 'novelty_buffer') and len(self.novelty_buffer) > 0:
            if isinstance(next_state, np.ndarray):
                flat_state = next_state.reshape(-1)
            elif isinstance(next_state, torch.Tensor):
                flat_state = next_state.reshape(-1).cpu().numpy()
            else:
                flat_state = None
            
            if flat_state is not None:
                # Calculate novelty as minimum distance to recent states
                distances = []
                for i in range(min(10, len(self.novelty_buffer))):  # Sample at most 10 states for efficiency
                    idx = random.randint(0, len(self.novelty_buffer) - 1)
                    stored_state = self.novelty_buffer[idx]
                    dist = np.linalg.norm(flat_state - stored_state)
                    distances.append(dist)
                
                state_novelty = min(1.0, np.min(distances) * 0.1)  # Normalize and cap
                surprise_components.append(state_novelty)
        
        # Combine surprise components
        if surprise_components:
            # Weighted average - give more weight to reward surprise
            weights = [0.5, 0.3, 0.2][:len(surprise_components)]
            weights = [w / sum(weights) for w in weights]  # Normalize
            total_surprise = sum(s * w for s, w in zip(surprise_components, weights))
            
            # Update surprise history
            self.surprise_history.append(total_surprise)
            
            return total_surprise
        else:
            return 0.0
    
    def update_with_prioritized_replay(self, batch_size=64):
        """
        Update agent using prioritized experience replay focusing on surprising events.
        
        This creates a more human-like learning process that focuses on unusual or important
        experiences rather than treating all experiences equally.
        
        Args:
            batch_size: Number of experiences to sample for update
            
        Returns:
            dict: Dictionary of loss metrics
        """
        if len(self.replay_buffer) < batch_size:
            return None  # Not enough data
        
        # Sample prioritized batch from replay buffer
        batch_data = self.replay_buffer.sample(batch_size)
        if batch_data is None:
            return None
        
        states, actions, rewards, next_states, dones, indices, weights = batch_data
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Get value estimates
        with torch.no_grad():
            next_outputs = self.old_model(next_states_tensor)
            next_values = next_outputs['value']
        
        # Calculate returns and advantages
        returns = rewards_tensor + (1.0 - dones_tensor) * self.gamma * next_values # Use self.gamma
        outputs = self.model(states_tensor)
        values = outputs['value']
        advantages = returns - values
        
        # Get log probabilities of actions
        means = outputs['actions_mean']
        stds = torch.exp(outputs['actions_logstd'])
        dist = Normal(means, stds)
        log_probs = dist.log_prob(actions_tensor).sum(1, keepdim=True)
        
        # Get old log probabilities
        with torch.no_grad():
            old_outputs = self.old_model(states_tensor)
            old_means = old_outputs['actions_mean']
            old_stds = torch.exp(old_outputs['actions_logstd'])
            old_dist = Normal(old_means, old_stds)
            old_log_probs = old_dist.log_prob(actions_tensor).sum(1, keepdim=True)
        
        # Calculate policy loss with importance weights
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages * weights_tensor
        surr2 = torch.clamp(ratio, 1.0 - self.config.get("EPS_CLIP", 0.2), 1.0 + self.config.get("EPS_CLIP", 0.2)) * advantages * weights_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate value loss with importance weights
        value_loss = F.mse_loss(values, returns, reduction='none') * weights_tensor
        value_loss = value_loss.mean()
        
        # Calculate entropy bonus
        entropy = dist.entropy().mean()
        entropy_loss = -self.config.get("ENTROPY_COEF", 0.01) * entropy
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss
        
        # Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Calculate new TD errors for priority updates
        with torch.no_grad():
            new_outputs = self.model(states_tensor)
            new_values = new_outputs['value']
            td_errors = (returns - new_values).squeeze().cpu().numpy()
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item(),
        }
    
    def process_step_outcome(self, action_taken, reward, next_state=None, info=None):
        """Process the outcome of an action, now with contextual memory updates."""
        # Update reward history for exploration adjustment
        self.recent_rewards.append(reward)
        
        # Store any prediction confidence info
        if info and 'prediction_confidence' in info:
            self.confidence_history.append(info['prediction_confidence'])
        
        # Calculate surprise level
        predictions = info.get('predictions', None) if info else None
        state = info.get('state', None) if info else None
        
        if self.use_surprise_based_replay and state is not None and next_state is not None:
            # Calculate how surprising this experience was
            surprise = self.calculate_surprise(state, action_taken, reward, next_state, predictions)
            
            # Store in replay buffer with priority based on surprise
            self.replay_buffer.push(state, action_taken, reward, next_state,
                                    info.get('done', False) if info else False,
                                    error=surprise)
            
            # Log highly surprising events
            if surprise > self.surprise_threshold:
                log(f"Surprising event detected (level: {surprise:.2f}). Adding to priority replay.")
                
                # Maybe trigger a replay update on highly surprising events
                if random.random() < 0.3:  # 30% chance to learn immediately from surprising events
                    self.update_with_prioritized_replay(batch_size=min(64, len(self.replay_buffer)))
        
        # Add to contextual memory if enabled
        if self.use_contextual_memory and hasattr(self, '_current_state'):
            # Extract market regime if available
            metadata = {}
            if hasattr(self, 'market_regimes') and self.market_regimes:
                dominant_regime = max(self.market_regimes.items(), key=lambda x: x[1])
                metadata['market_regime'] = dominant_regime[0]
                
                # Add prediction data if available
                if predictions is not None:
                    metadata['predictions'] = predictions
                
                # Determine importance based on reward magnitude or surprise
                importance = 0.5  # Default mid-level importance
                if hasattr(self, 'surprise_threshold') and 'surprise' in locals():
                    # More important if surprising
                    importance = min(1.0, surprise / self.surprise_threshold)
                elif reward is not None:
                    # More important if high absolute reward
                    importance = min(1.0, 0.5 + abs(reward) / 10.0)
                
                # Store in contextual memory
                outcome = "positive" if reward > 0 else "negative" if reward < 0 else "neutral"
                memory_id = self.contextual_memory.add_memory(
                    state=self._current_state,
                    outcome=outcome,
                    action_taken=action_taken,
                    reward=reward,
                    metadata=metadata,
                    importance=importance
                )
                
                # Periodically prune memory
                if random.random() < 0.05:  # 5% chance each step
                    self.contextual_memory.prune_memories()
                
                # Clear current state/action
                delattr(self, '_current_state')
                if hasattr(self, '_current_action'):
                    delattr(self, '_current_action')
        
        # ... rest of existing code ...
    
    def analyze_trade(self, trade_info):
        """
        Analyze a completed trade and extract lessons.
        
        This enables more naturalistic learning where the agent improves by reflecting
        on completed trades rather than just optimizing a reward function.
        
        Args:
            trade_info: Dictionary with trade information
            
        Returns:
            str: Lesson extracted from trade
        """
        if not self.use_post_trade_analysis or not hasattr(self, 'lesson_memory'):
            return None
        
        # Extract key trade data
        entry_price = trade_info.get('entry_price', 0)
        exit_price = trade_info.get('exit_price', 0)
        profit = trade_info.get('profit', 0)
        profit_pct = trade_info.get('profit_pct', 0)
        hold_time = trade_info.get('hold_time', 0)
        direction = trade_info.get('direction', 'unknown')  # 'long' or 'short'
        position_size = trade_info.get('position_size', 0)
        market_context = trade_info.get('market_context', {})
        
        # Determine trade outcome category
        if profit > 0:
            if profit_pct > 10:
                outcome = "big_win"
            else:
                outcome = "win"
        elif profit < 0:
            if profit_pct < -10:
                outcome = "big_loss"
            else:
                outcome = "loss"
        else:
            outcome = "breakeven"
        
        # Build context features for pattern recognition
        context_features = {}
        
        # Add market regime if available
        if hasattr(self, 'market_regimes') and self.market_regimes:
            # Get dominant regime
            dominant_regime = max(self.market_regimes.items(), key=lambda x: x[1])
            context_features['regime'] = dominant_regime[0]
            context_features['regime_strength'] = dominant_regime[1]
        
        # Add hold time characteristics
        if hold_time > 0:
            if hold_time < 12:  # Less than 1 hour at 5-min bars
                context_features['hold_time_category'] = 'very_short'
            elif hold_time < 72:  # Less than 6 hours
                context_features['hold_time_category'] = 'short'
            elif hold_time < 288:  # Less than 1 day
                context_features['hold_time_category'] = 'medium'
            else:
                context_features['hold_time_category'] = 'long'
        
        # Add position sizing info
        context_features['position_size_category'] = 'small' if position_size < 0.3 else 'medium' if position_size < 0.7 else 'large'
        
        # Add any market-specific context
        if market_context:
            for key, value in market_context.items():
                context_features[f'market_{key}'] = value
        
        # Store trade in recent_trades
        self.recent_trades.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': profit,
            'profit_pct': profit_pct,
            'hold_time': hold_time,
            'direction': direction,
            'position_size': position_size,
            'context': context_features,
            'outcome': outcome
        })
        
        # Generate lesson based on outcome
        lesson_text = ""
        lesson_tags = [outcome, direction, context_features.get('regime', 'unknown')]
        importance = 0.5  # Default importance
        if outcome == "big_win":
            lesson_text = f"Large profit ({profit_pct:.1f}%) achieved in {context_features.get('regime', 'unknown')} market with {direction} position."
            if 'hold_time_category' in context_features:
                lesson_text += f" {context_features['hold_time_category'].title()} hold time was effective."
            lesson_tags.append('success_pattern')
            importance = 0.8  # High importance
        elif outcome == "win":
            lesson_text = f"Profit ({profit_pct:.1f}%) in {context_features.get('regime', 'unknown')} market with {direction} position."
            if len(self.recent_trades) > 1:
                prev_trade = self.recent_trades[-2]
                if prev_trade['outcome'] in ['win', 'big_win'] and prev_trade['direction'] == direction:
                    lesson_text += f" Consecutive wins with {direction} positions in this regime."
                    lesson_tags.append('streak')
            importance = 0.6
        elif outcome == "big_loss":
            lesson_text = f"Large loss ({profit_pct:.1f}%) in {context_features.get('regime', 'unknown')} market with {direction} position."
            lesson_text += f" {context_features.get('position_size_category', 'unknown').title()} position size amplified the loss."
            lesson_tags.append('risk_pattern')
            importance = 0.9  # Very high importance for avoiding big losses
        elif outcome == "loss":
            lesson_text = f"Loss ({profit_pct:.1f}%) in {context_features.get('regime', 'unknown')} market with {direction} position."
            
            # Check if this is part of a pattern of losses
            loss_count = sum(1 for t in self.recent_trades if t['outcome'] in ['loss', 'big_loss'])
            if loss_count > 2:
                lesson_text += f" Part of a streak of {loss_count} recent losses."
                lesson_tags.append('losing_streak')
            else:
                importance = 0.6
        
        # Add lesson to memory
        if self.lesson_memory and lesson_text:
            self.lesson_memory.add_lesson(
                context=context_features,
                outcome=outcome,
                lesson_text=lesson_text,
                tags=lesson_tags,
                importance=importance
            )
        
        return lesson_text
    
    def get_relevant_lessons(self, market_state, tags=None):
        """
        Retrieve lessons relevant to current market conditions.
        
        Args:
            market_state: Current market state or features
            tags: Optional list of tags to filter lessons
            
        Returns:
            list: List of relevant lessons
        """
        if not hasattr(self, 'lesson_memory') or self.lesson_memory is None:
            return []
        
        # Extract context features from market state
        context_features = {}
        if hasattr(self, 'market_regimes') and self.market_regimes:
            # Get dominant regime
            dominant_regime = max(self.market_regimes.items(), key=lambda x: x[1])
            context_features['regime'] = dominant_regime[0]
        
        # Add additional tags based on market state
        if tags is None:
            tags = []
        
        # Add regime tag
        if 'regime' in context_features:
            tags.append(context_features['regime'])
        
        # Query lessons from memory
        return self.lesson_memory.query_lessons(context=context_features, tags=tags)
    
    def select_action(self, state, hidden):
        """Select action using policy network, now influenced by past lessons."""
        # ... existing code ...
        
        # Get relevant lessons for current situation
        if self.use_post_trade_analysis and hasattr(self, 'lesson_memory'):
            # Extract market context for lesson retrieval
            with torch.no_grad():
                if hasattr(self.model, 'market_regime_head'):
                    state_batched = state.unsqueeze(0) if state.dim() == 2 else state
                    # Get market regime prediction
                    regime_outputs = self.model.market_regime_head(state_batched[:, -1])
                    
                    # Determine dominant regime for contextual lesson retrieval
                    regime_probs = regime_outputs[0].cpu().numpy()
                    regime_idx = np.argmax(regime_probs)
                    regime_types = ["trending", "ranging", "volatile", "mixed"]
                    dominant_regime = regime_types[regime_idx]
                    
                    # Get relevant lessons
                    relevant_lessons = self.get_relevant_lessons(state, tags=[dominant_regime])
                    
                    # Apply insights from lessons to adjust action if needed
                    if relevant_lessons:
                        # Log that we're using lessons
                        log(f"Using {len(relevant_lessons)} relevant lessons for decision making")
                        
                        # Extract common themes from lessons
                        loss_lessons = [l for l in relevant_lessons if 'loss' in l['outcome']]
                        win_lessons = [l for l in relevant_lessons if 'win' in l['outcome']]
                        
                        # Adjust action based on lessons (subtly influence decisions)
                        if len(loss_lessons) > len(win_lessons) and random.random() < 0.5:
                            # More losses than wins in similar situations - be more cautious
                            # Reduce position size slightly
                            position_size_factor *= 0.8
                            log("Applying caution from past lessons: reducing position size")
                        
                        elif len(win_lessons) > len(loss_lessons) and random.random() < 0.5:
                            # More wins than losses in similar situations - be more aggressive
                            # Slightly increase position size but cap for safety
                            position_size_factor = min(1.2 * position_size_factor, 1.0)
                            log("Applying confidence from past lessons: increasing position size")
        
        # ... rest of select_action code ...
    
    # Add a method to PPOAgent for hyperparameter adaptation
    def adapt_hyperparameters(self, episode, recent_rewards):
        """
        Adapt hyperparameters based on recent performance.
        
        This enables more naturalistic learning where the agent discovers
        not just what to do but how to learn most effectively.
        
        Args:
            episode: Current episode number
            recent_rewards: List of recent episode rewards
            
        Returns:
            dict: Dictionary of updated parameters
        """
        if not self.use_meta_learning or not hasattr(self, 'param_optimizer'):
            return {}
        
        # Get parameter suggestions from optimizer
        suggested_params = self.param_optimizer.suggest_parameters(episode, recent_rewards)
        
        if not suggested_params:
            return {}
        
        # Apply suggested parameters
        updated_params = {}
        
        # Learning rate adjustment
        if "LEARNING_RATE" in suggested_params:
            new_lr = suggested_params["LEARNING_RATE"]
            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            updated_params["LEARNING_RATE"] = new_lr
            log(f"Meta-learning: Adjusted learning rate to {new_lr:.6f}")
        
        # Other hyperparameter adjustments
        for param, value in suggested_params.items():
            if param != "LEARNING_RATE":  # Already handled above
                # Identify the corresponding agent attribute for the parameter
                # Examples (adjust based on actual attributes used):
                if param == "ENTROPY_COEF":
                    self.entropy_coef = value
                elif param == "EPS_CLIP":
                    self.clip_ratio = value # Assuming eps_clip maps to clip_ratio
                elif param == "GAMMA":
                    self.gamma = value # Ensure self.gamma exists/is used
                elif param == "BATCH_SIZE":
                    self.batch_size = value # Ensure self.batch_size exists/is used
                elif param == "INITIAL_EXPLORATION_RATE": # Check if this is actually tuned
                     self.exploration_rate = value # Update the base rate
                # Add other parameter mappings as needed...
                else:
                     # Optionally log unhandled param or try direct attribute setting
                     # if hasattr(self, param.lower()):
                     #    setattr(self, param.lower(), value)
                     # else:
                     log(f"Meta-learning: No specific attribute update for {param}", level="warning")

                # Remove runtime config update:
                # self.config[param] = value 
                updated_params[param] = value
                log(f"Meta-learning: Adjusted {param} to {value}")
        
        return updated_params
    
    # Add code to record hyperparameter performance in training loop
    def record_hyperparameter_performance(self, episode_reward, episode):
        """
        Record performance of current hyperparameters.
        
        Args:
            episode_reward: Average reward for recent episodes
            episode: Current episode number
        """
        if not self.use_meta_learning or not hasattr(self, 'param_optimizer'):
            return
        
        # Get current hyperparameters
        current_params = {
            "LEARNING_RATE": self.optimizer.param_groups[0]['lr'],
            "ENTROPY_COEF": self.config.get("ENTROPY_COEF", 0.01),
            "EPS_CLIP": self.config.get("EPS_CLIP", 0.2),
            "GAMMA": self.config.get("GAMMA", 0.99),
            "BATCH_SIZE": self.config.get("BATCH_SIZE", 64),
            "GAE_LAMBDA": self.config.get("GAE_LAMBDA", 0.95)
        }
        
        # Record performance with these parameters
        self.param_optimizer.record_performance(current_params, episode_reward, episode)

class ESPopulation:
    """
    Population of agents for evolutionary strategies.
    Manages a population of agents with evolutionary selection and mutation.
    """
    def __init__(self, num_agents, input_dim, hidden_size, lr, horizons=None,
                 use_mixed_precision=False, model_type='actor_critic', device="cpu", config=None):
        """
        Initialize population.
        
        Args:
            num_agents (int): Number of agents in population.
            input_dim (int): Dimension of input features.
            hidden_size (int): Size of hidden layers.
            lr (float): Learning rate.
            horizons (list, optional): List of prediction horizons. Defaults to None.
            use_mixed_precision (bool, optional): Whether to use mixed precision training. Defaults to False.
            model_type (str, optional): Type of model to use. Defaults to 'actor_critic'.
            device (str, optional): Device to run on. Defaults to "cpu".
            config (dict, optional): Additional configuration parameters. Defaults to None.
        """
        self.config = config or {}
        self.device = device
        self.mutation_rate = self.config.get("ES_MUTATION_RATE", 0.1)
        self.mutation_strength = self.config.get("ES_MUTATION_STRENGTH", 0.1)
        self.elite_fraction = self.config.get("ES_ELITE_FRACTION", 0.2)
        
        # Create population
        self.agents = PPOAgent.create_population(
            num_agents, input_dim, hidden_size, lr, horizons,
            use_mixed_precision, model_type, device, config
        )
        
        # Track fitness
        self.fitness = [float('-inf')] * num_agents
        self.best_agent_idx = 0
        self.best_fitness = float('-inf')
        
        log(f"Initialized population with {num_agents} agents")
    
    def evolve(self, fitness_scores):
        """
        Evolve population based on fitness scores.
        
        Args:
            fitness_scores (list): List of fitness scores for each agent.
            
        Returns:
            int: Index of best agent after evolution.
        """
        if len(fitness_scores) != len(self.agents):
            log(f"Warning: fitness scores length ({len(fitness_scores)}) doesn't match population size ({len(self.agents)})")
            return self.best_agent_idx
        
        # Update fitness
        self.fitness = fitness_scores
        
        # Find best agent
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_agent_idx = best_idx
            self.best_fitness = fitness_scores[best_idx]
            log(f"New best agent found: {best_idx} with fitness {self.best_fitness:.4f}")
        
        # Sort by fitness
        indices = np.argsort(fitness_scores)[::-1]
        
        # Determine number of elite agents
        num_elite = max(1, int(len(self.agents) * self.elite_fraction))
        elite_indices = indices[:num_elite]
        non_elite_indices = indices[num_elite:]
        
        log(f"Evolution: {num_elite} elite agents, top fitness: {fitness_scores[indices[0]]:.4f}")
        
        # Replace bottom agents with mutated versions of top performers
        for idx, target_idx in enumerate(non_elite_indices):
            # Select a random elite agent to clone
            elite_idx = random.choice(elite_indices)
            
            # Clone elite agent to target
            self.agents[target_idx].model.load_state_dict(self.agents[elite_idx].model.state_dict())
            
            # Apply mutation
            self._mutate_agent(self.agents[target_idx])
            
            # Update old model with new weights
            self.agents[target_idx].update_old_model()
        
        return self.best_agent_idx
    
    def _mutate_agent(self, agent):
        """
        Apply mutation to agent parameters.
        
        Args:
            agent (PPOAgent): Agent to mutate.
        """
        # Apply random mutations to model parameters
        for param in agent.model.parameters():
            if random.random() < self.mutation_rate:
                # Add random noise scaled by mutation strength
                noise = torch.randn_like(param) * self.mutation_strength
                param.data += noise
        
        # For probabilistic predictions, we might want different mutation strategies
        # for mean and std parameters to ensure stable behavior
        
        # Find mean prediction parameters
        mean_params = []
        std_params = []
        
        for name, param in agent.model.named_parameters():
            if 'pred_mean' in name:
                mean_params.append(param)
            elif 'pred_std' in name:
                std_params.append(param)
        
        # Apply moderate mutations to mean parameters
        for param in mean_params:
            if random.random() < self.mutation_rate:
                noise = torch.randn_like(param) * self.mutation_strength
                param.data += noise
        
        # Apply controlled mutations to std parameters to maintain positive values
        # and avoid unstable predictions
        
        for param in std_params:
            if random.random() < self.mutation_rate:
                # Smaller noise for std parameters
                noise = torch.randn_like(param) * (self.mutation_strength * 0.5)
                # Ensure std parameters remain positive
                param.data = torch.clamp(param.data + noise, min=1e-6)

class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.
    Currently not used by PPO (which is on-policy), but available for
    future extensions to algorithms like SAC, DQN, etc.
    """
    def __init__(self, capacity):
        """
        Initialize replay buffer.
        
        Args:
            capacity (int): Maximum capacity of buffer.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward (float): Reward received.
            next_state: Next state.
            done (bool): Whether episode is done.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample batch from buffer.
        
        Args:
            batch_size (int): Size of batch to sample.
            
        Returns:
            tuple: Batch of experiences.
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)

class LessonMemory:
    """
    Memory system for storing and retrieving lessons learned from trades.
    This creates a more naturalistic learning process by allowing the agent to
    remember important patterns and outcomes from previous trades.
    """
    def __init__(self, capacity=100):
        """
        Initialize the lesson memory system.
        
        Args:
            capacity: Maximum number of lessons to store
        """
        self.capacity = capacity
        self.lessons = deque(maxlen=capacity)
        self.lesson_tags = {}  # Tag-based indexing for efficient retrieval
        self.importance_scores = {}  # Track importance of each lesson
    
    def add_lesson(self, context, outcome, lesson_text, tags=None, importance=0.5):
        """
        Add a new lesson to memory.
        
        Args:
            context: The market context where this lesson applies (dict or embedding)
            outcome: The outcome that led to this lesson (e.g., "profit", "loss")
            lesson_text: The textual description of the lesson
            tags: List of tags for categorizing the lesson
            importance: Importance score (0-1) for this lesson
        """
        # Generate a unique ID for this lesson
        lesson_id = len(self.lessons)
        
        # Create lesson object
        lesson = {
            'id': lesson_id,
            'context': context,
            'outcome': outcome,
            'text': lesson_text,
            'tags': tags or [],
            'importance': importance,
            'usage_count': 0,
            'created_at': time.time()
        }
        
        # Add to lesson collection
        self.lessons.append(lesson)
        self.importance_scores[lesson_id] = importance
        
        # Update tag indices
        if tags:
            for tag in tags:
                if tag not in self.lesson_tags:
                    self.lesson_tags[tag] = []
                self.lesson_tags[tag].append(lesson_id)
    
    def query_lessons(self, context=None, tags=None, top_k=3):
        """
        Query lessons based on context similarity and/or tags.
        
        Args:
            context: Current market context to match against
            tags: List of tags to filter by
            top_k: Number of lessons to return
            
        Returns:
            list: Top k matching lessons
        """
        # Start with all lessons
        candidate_ids = set(range(len(self.lessons)))
        
        # Filter by tags if provided
        if tags:
            tag_matches = set()
            for tag in tags:
                if tag in self.lesson_tags:
                    tag_matches.update(self.lesson_tags[tag])
            
            if tag_matches:
                candidate_ids = candidate_ids.intersection(tag_matches)
        
        # If no candidates after filtering, return empty list
        if not candidate_ids:
            return []
        
        # Score candidates by relevance and importance
        scored_candidates = []
        
        for lesson_id in candidate_ids:
            if lesson_id >= len(self.lessons):
                continue
            
            lesson = self.lessons[lesson_id]
            
            # Base score on importance
            score = lesson['importance']
            
            # If context provided, adjust score based on context similarity
            if context is not None and 'context' in lesson and lesson['context'] is not None:
                try:
                    # Simple similarity calculation
                    if isinstance(context, dict) and isinstance(lesson['context'], dict):
                        # Compare dictionaries - check overlap of keys/values
                        sim_keys = set(context.keys()).intersection(set(lesson['context'].keys()))
                        similarity = len(sim_keys) / max(len(context), len(lesson['context']))
                        score *= (0.5 + 0.5 * similarity)  # Scale factor (0.5-1.0)
                    elif hasattr(context, 'shape') and hasattr(lesson['context'], 'shape'):
                        # Compare arrays/tensors - use cosine similarity
                        context_flat = context.reshape(-1)
                        lesson_flat = lesson['context'].reshape(-1)
                        dot_product = np.dot(context_flat, lesson_flat)
                        norm_product = np.linalg.norm(context_flat) * np.linalg.norm(lesson_flat)
                        if norm_product > 0:
                            similarity = dot_product / norm_product
                            score *= (0.5 + 0.5 * similarity)  # Scale factor (0.5-1.0)
                except Exception as e:
                    # Fallback if similarity calculation fails
                    print(f"Error calculating similarity: {e}")
            
            # Adjust score for recency (favor more recent lessons)
            age = time.time() - lesson['created_at']
            recency_factor = 1.0 / (1.0 + age / (3600 * 24 * 7))  # Decay over weeks
            score *= (0.5 + 0.5 * recency_factor)  # Scale factor (0.5-1.0)
            
            scored_candidates.append((lesson_id, score))
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k lessons
        top_lessons = []
        for i, (lesson_id, _) in enumerate(scored_candidates[:top_k]):
            lesson = self.lessons[lesson_id]
            lesson['usage_count'] += 1  # Increment usage counter
            top_lessons.append(lesson)
        
        return top_lessons
    
    def update_importance(self, lesson_id, new_importance):
        """
        Update the importance score of a lesson.
        
        Args:
            lesson_id: ID of the lesson to update
            new_importance: New importance score (0-1)
        """
        if lesson_id < len(self.lessons):
            lesson = self.lessons[lesson_id]
            lesson['importance'] = new_importance
            self.importance_scores[lesson_id] = new_importance
    
    def forget_least_important(self, count=10):
        """
        Forget the least important lessons to make room for new ones.
        
        Args:
            count: Number of lessons to forget
        """
        if len(self.lessons) <= count:
            return  # Don't forget all lessons
        
        # Get IDs of least important lessons
        lesson_ids = list(range(len(self.lessons)))
        lesson_ids.sort(key=lambda i: self.importance_scores.get(i, 0))
        
        # Remove the least important lessons
        to_forget = lesson_ids[:count]
        for lesson_id in to_forget:
            if lesson_id < len(self.lessons):
                lesson = self.lessons[lesson_id]
                # Remove from tag indices
                for tag in lesson.get('tags', []):
                    if tag in self.lesson_tags and lesson_id in self.lesson_tags[tag]:
                        self.lesson_tags[tag].remove(lesson_id)
                
                # Reset the lesson (will be overwritten in the deque)
                self.lessons[lesson_id] = None
                
                # Remove from importance scores
                if lesson_id in self.importance_scores:
                    del self.importance_scores[lesson_id]

# Add HyperparamOptimizer class to manage hyperparameter tuning
class HyperparamOptimizer:
    """
    Simple meta-learning system that allows the agent to optimize its own hyperparameters.
    This creates a more naturalistic learning process where the agent learns not just
    a policy but also how to learn effectively.
    """
    def __init__(self, config, tunable_params=None):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            config: Base configuration dictionary
            tunable_params: Dict of parameter names and their tuning ranges (min, max)
        """
        self.config = config
        
        # Default tunable parameters if not specified
        self.tunable_params = tunable_params or {
            "LEARNING_RATE": (1e-5, 1e-2),
            "ENTROPY_COEF": (0.001, 0.1),
            "EPS_CLIP": (0.1, 0.3),
            "GAMMA": (0.9, 0.999),
            "BATCH_SIZE": (32, 256, "discrete"),
            "INITIAL_EXPLORATION_RATE": (0.5, 1.0)
        }
        
        # Track performance history for each parameter setting
        self.param_history = []
        
        # Track best parameter settings found
        self.best_params = {}
        self.best_reward = float('-inf')
        
        # Track parameter iterations
        self.param_iterations = {param: 0 for param in self.tunable_params}
        
        # Default adaptation parameters
        self.adaptation_frequency = config.get("ADAPT_PARAM_FREQ", 10)  # Episodes between adaptation
        self.max_step_size = 0.2  # Maximum parameter adjustment size
        self.success_threshold = 0.05  # Reward improvement needed for "success"
        self.exploration_decay = 0.95  # Exploration decay for parameter search
        self.current_exploration = 1.0  # Start with full exploration
    
    def suggest_parameters(self, episode, recent_rewards=None):
        """
        Suggest parameter values to try based on history and exploration.
        
        Args:
            episode: Current episode number
            recent_rewards: List of recent rewards for trend analysis
            
        Returns:
            dict: Dictionary of suggested parameter values
        """
        # Only adapt parameters on schedule
        if episode % self.adaptation_frequency != 0 or episode == 0:
            return {}
        
        # Decay exploration rate
        self.current_exploration *= self.exploration_decay
        
        # Determine which parameter to update
        # Cycle through parameters or prioritize underperforming ones
        if recent_rewards and len(recent_rewards) >= 5:
            # If rewards are improving, make smaller changes
            reward_trend = sum(recent_rewards[-5:]) / 5 - sum(recent_rewards[-10:-5]) / 5
            if reward_trend > 0:
                # Reducing exploration rate
                explore_rate = self.current_exploration * 0.5
            else:
                # Increasing exploration rate
                explore_rate = min(1.0, self.current_exploration * 2.0)
        else:
            explore_rate = self.current_exploration
        
        # Select parameters to adjust
        params_to_adjust = []
        for param, count in sorted(self.param_iterations.items(), key=lambda x: x[1]):
            # Prioritize less frequently adjusted parameters
            if random.random() < 0.7:  # 70% chance to pick least adjusted
                params_to_adjust.append(param)
                break
        
        # If nothing selected, pick randomly
        if not params_to_adjust:
            params_to_adjust = [random.choice(list(self.tunable_params.keys()))]
        
        # Generate new parameter values
        new_params = {}
        for param in params_to_adjust:
            # Get tuning range
            param_range = self.tunable_params[param]
            min_val, max_val = param_range[0], param_range[1]
            is_discrete = len(param_range) > 2 and param_range[2] == "discrete"
            
            # Get current value
            current_val = self.config.get(param, (min_val + max_val) / 2)
            
            # Generate new value with exploration
            if random.random() < explore_rate:
                # Exploratory change - random within range
                if is_discrete:
                    new_val = 2 ** random.randint(int(np.log2(min_val)), int(np.log2(max_val)))
                else:
                    if param.startswith("LEARNING_RATE"):
                        # Log scale for learning rate
                        log_min, log_max = np.log10(min_val), np.log10(max_val)
                        new_val = 10 ** random.uniform(log_min, log_max)
                    else:
                        new_val = random.uniform(min_val, max_val)
            else:
                # Exploitative change - small adjustment to current value
                step_size = self.max_step_size * (max_val - min_val)
                delta = random.uniform(-step_size, step_size)
                new_val = current_val + delta
            
            # Ensure within bounds
            new_val = max(min_val, min(max_val, new_val))
            
            # Round if discrete
            if is_discrete:
                new_val = 2 ** round(np.log2(new_val))
            
            # Add to suggested parameters
            new_params[param] = new_val
            
            # Update iteration count
            self.param_iterations[param] += 1
        
        return new_params
    
    def record_performance(self, params, reward, episode):
        """
        Record performance for a set of parameters.
        
        Args:
            params: Parameter dictionary used
            reward: Average reward achieved
            episode: Episode number
        """
        # Record in history
        entry = {
            "params": params.copy(),
            "reward": reward,
            "episode": episode,
            "timestamp": time.time()
        }
        self.param_history.append(entry)
        
        # Update best parameters if improved
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = params.copy()
    
    def get_best_parameters(self):
        """
        Get the best parameter settings found so far.
        
        Returns:
            dict: Best parameter settings
        """
        return self.best_params

class ContextualMemory:
    """
    Memory system for storing and recalling memorable market situations.
    This enables the agent to recognize and recall similar market conditions
    from past experiences, mimicking human episodic and associative memory.
    """
    def __init__(self, capacity=200, similarity_threshold=0.7, feature_dim=None):
        """
        Initialize contextual memory.
        
        Args:
            capacity (int): Maximum number of memories to store
            similarity_threshold (float): Threshold for considering situations similar (0-1)
            feature_dim (int, optional): Dimension of feature vectors for memory
        """
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.feature_dim = feature_dim
        self.memories = deque(maxlen=capacity)
        self.memory_importance = {}  # Track importance of each memory
        self.last_accessed = {}  # Track when each memory was last accessed
        self.market_regime_memories = {}  # Index by market regime for faster retrieval
        self.embeddings = None  # Will store numpy array of embeddings when needed
        self.current_id = 0  # Unique ID counter for memories
        self.staleness_penalty = 0.05  # Penalty for not being accessed recently
        
        # Scaler for normalizing market state features
        self.scaler = StandardScaler()
        self.scaler_fitted = False
    
    def add_memory(self, state, outcome, action_taken=None, reward=None, metadata=None, importance=0.5):
        """
        Add a new market situation to memory.
        
        Args:
            state: Market state or embedding
            outcome: Outcome that resulted from this situation
            action_taken: Action taken in this situation (optional)
            reward: Reward received (optional)
            metadata: Additional information about this market state
            importance: Initial importance score (0-1) for this memory
            
        Returns:
            int: ID of the added memory
        """
        # Generate unique ID
        memory_id = self.current_id
        self.current_id += 1
        
        # Ensure state is a numpy array
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        # Preprocess state for storage if it's a raw feature vector
        if len(state.shape) == 1 or (len(state.shape) == 2 and state.shape[0] == 1):
            # Flatten if needed
            state_vector = state.flatten()
            
            # Update scaler if needed
            if not self.scaler_fitted and len(state_vector) > 1:
                # Reshape for scaler
                state_reshaped = state_vector.reshape(1, -1)
                self.scaler.partial_fit(state_reshaped)
                self.scaler_fitted = True
        else:
            # If it's already processed (e.g., an embedding), store as is
            state_vector = state
        
        # Extract market regime if present in metadata
        market_regime = None
        if metadata and 'market_regime' in metadata:
            market_regime = metadata['market_regime']
        
        # Create memory object
        memory = {
            'id': memory_id,
            'state': state_vector,
            'outcome': outcome,
            'action_taken': action_taken,
            'reward': reward,
            'metadata': metadata or {},
            'market_regime': market_regime,
            'timestamp': time.time(),
            'access_count': 0,
            'success_count': 0 if reward is None else (1 if reward > 0 else 0)
        }
        
        # Add to memory collection
        self.memories.append(memory)
        self.memory_importance[memory_id] = importance
        self.last_accessed[memory_id] = time.time()
        
        # Index by market regime for faster retrieval
        if market_regime:
            if market_regime not in self.market_regime_memories:
                self.market_regime_memories[market_regime] = []
            self.market_regime_memories[market_regime].append(memory_id)
        
        # Invalidate embedding cache since we added a new memory
        self.embeddings = None
        
        return memory_id
    
    def update_memory_importance(self, memory_id, new_importance=None, success=None):
        """
        Update the importance of a memory.
        
        Args:
            memory_id: ID of the memory to update
            new_importance: New importance score (if None, calculate based on success)
            success: Whether applying this memory led to a successful outcome
        
        Returns:
            bool: Whether the update was successful
        """
        if memory_id not in self.memory_importance:
            return False
        
        # If explicit importance provided, use it
        if new_importance is not None:
            self.memory_importance[memory_id] = max(0.1, min(1.0, new_importance))
        
        # If success information provided, adjust importance
        elif success is not None:
            current_importance = self.memory_importance[memory_id]
            
            # Find the memory
            memory = None
            for m in self.memories:
                if m['id'] == memory_id:
                    memory = m
                    break
            
            if memory:
                # Update success count
                if success:
                    memory['success_count'] = memory.get('success_count', 0) + 1
                
                # Calculate success rate
                success_rate = memory['success_count'] / max(1, memory['access_count'])
                
                # Adjust importance based on success rate
                if success:
                    # Increase importance when successful
                    new_importance = current_importance * 1.1
                else:
                    # Decrease importance when unsuccessful
                    new_importance = current_importance * 0.9
                
                # Ensure within bounds
                self.memory_importance[memory_id] = max(0.1, min(1.0, new_importance))
        
        return True
    
    def recall_similar_situations(self, state, top_k=3, market_regime=None):
        """
        Find situations in memory similar to the current state.
        
        Args:
            state: Current market state or embedding
            top_k: Number of similar situations to return
            market_regime: Current market regime (optional, for faster filtering)
            
        Returns:
            list: Top k similar situations with similarity scores
        """
        # Ensure state is a numpy array
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        # Flatten and normalize if needed
        if len(state.shape) == 1 or (len(state.shape) == 2 and state.shape[0] == 1):
            query_vector = state.flatten()
            if self.scaler_fitted:
                query_vector = self.scaler.transform(query_vector.reshape(1, -1)).flatten()
        else:
            query_vector = state
        
        # First try to filter by market regime for efficiency
        candidate_memories = list(self.memories)
        if market_regime and market_regime in self.market_regime_memories:
            # Get memory IDs for this regime
            regime_memory_ids = self.market_regime_memories[market_regime]
            # Filter memories by these IDs
            candidate_memories = [m for m in self.memories if m['id'] in regime_memory_ids]
        
        # Calculate similarity for each memory
        similarities = []
        
        for memory in candidate_memories:
            # Skip if not compatible
            if isinstance(memory['state'], np.ndarray) and isinstance(query_vector, np.ndarray):
                # Only compare if dimensions match
                if memory['state'].shape != query_vector.shape and memory['state'].size != query_vector.size:
                    continue
            
            memory_state = memory['state']
            
            # Reshape if needed
            if memory_state.size == query_vector.size and memory_state.shape != query_vector.shape:
                memory_state = memory_state.reshape(query_vector.shape)
            
            try:
                # Calculate cosine similarity
                if memory_state.size > 1 and query_vector.size > 1:
                    similarity = np.dot(memory_state, query_vector) / (
                        np.linalg.norm(memory_state) * np.linalg.norm(query_vector) + 1e-8)
                else:
                    # For single-dimension states, use simple comparison
                    similarity = 1.0 - min(1.0, abs(memory_state.item() - query_vector.item()) /
                                          (abs(memory_state.item()) + 1e-8))
                
                # Apply recency and importance adjustments
                time_factor = np.exp(-0.01 * (time.time() - self.last_accessed[memory['id']]))
                importance = self.memory_importance[memory['id']]
                
                # Final score is a combination of similarity, recency, and importance
                final_score = 0.6 * similarity + 0.2 * time_factor + 0.2 * importance
                
                similarities.append((memory, final_score))
            except Exception as e:
                # Skip this memory if comparison fails
                continue
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update access counts and timestamps for returned memories
        for memory, _ in similarities[:top_k]:
            memory['access_count'] = memory.get('access_count', 0) + 1
            self.last_accessed[memory['id']] = time.time()
        
        return similarities[:top_k]
    
    def get_memory_by_id(self, memory_id):
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: ID of memory to retrieve
            
        Returns:
            dict: Memory object or None if not found
        """
        for memory in self.memories:
            if memory['id'] == memory_id:
                # Update access information
                memory['access_count'] = memory.get('access_count', 0) + 1
                self.last_accessed[memory_id] = time.time()
                return memory
        return None
    
    def prune_memories(self):
        """
        Remove less important or stale memories to stay within capacity.
        """
        if len(self.memories) < self.capacity:
            return
        
        # Calculate scores based on importance, recency, and access frequency
        scores = []
        current_time = time.time()
        
        for memory in self.memories:
            memory_id = memory['id']
            importance = self.memory_importance.get(memory_id, 0.1)
            last_access = self.last_accessed.get(memory_id, 0)
            recency = np.exp(-0.001 * (current_time - last_access))  # Exponential decay
            access_count = memory.get('access_count', 0)
            access_score = min(1.0, access_count / 10.0)  # Normalize access count
            
            # Combined score
            score = 0.5 * importance + 0.3 * recency + 0.2 * access_score
            scores.append((memory_id, score))
        
        # Sort by score (ascending, so lowest scores first)
        scores.sort(key=lambda x: x[1])
        
        # Number to remove to get back to 90% capacity (leaving room for new memories)
        target_size = int(self.capacity * 0.9)
        num_to_remove = max(0, len(self.memories) - target_size)
        
        if num_to_remove > 0:
            # Get IDs to remove
            ids_to_remove = [id for id, _ in scores[:num_to_remove]]
            
            # Remove from main collection
            self.memories = deque([m for m in self.memories if m['id'] not in ids_to_remove], maxlen=self.capacity)
            
            # Clean up other data structures
            for id in ids_to_remove:
                if id in self.memory_importance:
                    del self.memory_importance[id]
                if id in self.last_accessed:
                    del self.last_accessed[id]
            
            # Remove from regime index
            for regime, ids in self.market_regime_memories.items():
                if id in ids:
                    self.market_regime_memories[regime].remove(id)
            
            # Invalidate embeddings cache
            self.embeddings = None

if __name__ == "__main__":
    # Simple test for agent
    import torch
    import numpy as np
    # Use the already imported models module instead of a direct import
    ActorCritic = models.ActorCritic
    
    # Test parameters
    input_dim = 64
    hidden_size = 128
    batch_size = 5
    sequence_length = 288
    
    # Create agent
    agent = PPOAgent(input_dim, hidden_size, lr=3e-4)
    
    # Create dummy state
    state = np.random.randn(sequence_length, input_dim).astype(np.float32)
    
    # Test action selection
    action, log_prob, value, pred_means, pred_stds, confs, mid_means, mid_stds, trend, novelty, hidden = agent.select_action(state, None)
    print(f"Action: {action}")
    print(f"Log prob: {log_prob}")
    print(f"Value: {value}")
    print(f"Prediction means shape: {pred_means.shape}")
    print(f"Prediction std shape: {pred_stds.shape}")
    print(f"Confidence shape: {confs.shape}")
    print(f"Mid-points means shape: {mid_means.shape}")
    print(f"Mid-points std shape: {mid_stds.shape}")
    print(f"Trend strength: {trend}")
    print(f"Novelty: {novelty}")
    
    # Test batch prediction
    states = [np.random.randn(sequence_length, input_dim).astype(np.float32) for _ in range(10)]
    batch_results = agent.predict_batch(states, batch_size=5)
    print(f"Batch prediction results: {len(batch_results)}")
    
    # Test probabilistic predictions
    print("\nDemonstrating probabilistic prediction capabilities:")
    
    # Get prediction intervals
    horizons = [12, 36, 72, 144]  # Default horizons
    lower_bounds, upper_bounds = agent.calculate_prediction_intervals(pred_means, pred_stds, confidence=0.95)
    print(f"95% confidence intervals:")
    for i, h in enumerate(horizons):
        print(f"  Horizon {h}: {pred_means[i]:.4f} [{lower_bounds[i]:.4f}, {upper_bounds[i]:.4f}]")
    
    # Generate samples
    samples = agent.generate_prediction_samples(pred_means, pred_stds, num_samples=3)
    print(f"\nSample predictions from distribution:")
    for i, sample in enumerate(samples):
        print(f"  Sample {i+1}: {sample}")
    
    # Visualize predictions (if matplotlib is available)
    try:
        print("\nGenerating probabilistic prediction visualization...")
        fig = agent.visualize_predictions(pred_means, pred_stds, horizons=[12, 36, 72, 144])
        if fig:
            print("Visualization successful!")
    except Exception as e:
        print(f"Visualization error: {e}")
    
    # Test update
    dummy_states = np.random.randn(batch_size, sequence_length, input_dim).astype(np.float32)
    dummy_actions = np.random.randn(batch_size, 2).astype(np.float32)
    dummy_old_logps = np.random.randn(batch_size).astype(np.float32)
    dummy_returns = np.random.randn(batch_size).astype(np.float32)
    dummy_advantages = np.random.randn(batch_size).astype(np.float32)
    loss_info = agent.update(dummy_states, dummy_actions, dummy_old_logps, dummy_returns, dummy_advantages)
    print(f"\nUpdate loss info: {loss_info}")
    
    # Test population creation
    population = PPOAgent.create_population(3, input_dim, hidden_size, lr=3e-4)
    print(f"Population created with {len(population)} agents")
    
    # Test ES population
    es_pop = ESPopulation(3, input_dim, hidden_size, lr=3e-4)
    dummy_fitness = [1.0, 2.0, 0.5]
    best_idx = es_pop.evolve(dummy_fitness)
    print(f"Evolution complete, best agent: {best_idx}")