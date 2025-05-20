#!/usr/bin/env python
"""
Neural Network Models for Trading Agent

This module defines the neural network architectures used by the trading agent,
including the ActorCritic network and other model components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import importlib

# Use dynamic imports for utils
utils_module = importlib.import_module("src.utils.utils")
log = utils_module.log

# Import the new predictor module
try:
    dynamic_predictor_module = importlib.import_module("src.models.dynamic_horizon_predictor")
    DynamicHorizonPredictor = dynamic_predictor_module.DynamicHorizonPredictor
except ImportError as e:
    log(f"[ERROR] Could not import DynamicHorizonPredictor: {e}", "error")
    # Define a dummy class if import fails to avoid crashing later
    class DynamicHorizonPredictor(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            log("[ERROR] Using dummy DynamicHorizonPredictor due to import failure.", "error")
        def forward(self, features, requested_horizon=None):
            # Return None or zero tensors with expected keys
            return {
                "horizon_mean": torch.zeros_like(features[:, :1]),
                "horizon_std": torch.ones_like(features[:, :1]),
                "outcome_mean": None,
                "outcome_std": None,
                "outcome_confidence": None
            }

# Add new imports for explainability
import numpy as np
import scipy.stats as stats

# Known issues:
# FIXED: Gradient flow to calibration parameters now works correctly after ensuring
#       parameters are properly connected in the computational graph.

class ActorCritic(nn.Module):
    """
    Actor-Critic network with multiple probabilistic prediction heads.
    
    This enhanced model uses a shared feature extractor and includes:
    - Actor (policy) head for trading actions
    - Critic (value) head for state value estimation
    - Multiple probabilistic prediction heads for different time horizons
    - Risk assessment head
    - Trend strength estimator
    """
    
    def __init__(self, input_dim, hidden_size, model_type="transformer", 
                 max_seq_len=60, device="cpu", config=None):
        """
        Initialize actor-critic model. Now predicts horizon distribution instead of fixed horizons.
        
        Args:
            input_dim (int): Dimensionality of input features
            hidden_size (int): Size of hidden layers
            model_type (str): Model architecture type ('transformer', 'lstm', 'gru', or 'ensemble')
            max_seq_len (int): Maximum sequence length for attention mechanisms
            device (str): Device to run model on ('cpu' or 'cuda')
            config (dict): Additional configuration parameters
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.model_type = model_type
        self.config = config or {}
        self.config['device'] = device # Ensure device is in config for predictor
        
        # Configurable dimension for horizon embedding
        self.horizon_embedding_dim = self.config.get("HORIZON_EMBEDDING_DIM", 16)
        
        # Add feature importance weights for explainability
        self.feature_weights = nn.Parameter(torch.ones(input_dim, device=device), requires_grad=False)
        
        # Add explainability flags
        self.explain_mode = False
        self.activation_maps = {}
        self.gradients = {}
        
        # Log model configuration
        log(f"Creating ActorCritic with input_dim={input_dim}, hidden_size={hidden_size}. Using Dynamic Horizon Predictor.")
        
        # Move model to the specified device
        self.to(device)
        
        # Transformer for sequence processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=8, 
            dim_feedforward=hidden_size, 
            batch_first=True,
            dropout=0.1
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # ======= Chain of Draft Components =======
        # 1. Market Regime Analysis
        self.market_regime_head = nn.Sequential(
            nn.Linear(input_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4),  # Four market regimes: trending, ranging, volatile, mixed
            nn.Softmax(dim=-1)
        )
        
        # 2. Technical Pattern Recognition
        self.pattern_recognition_head = nn.Sequential(
            nn.Linear(input_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 8)  # 8 common patterns (continuation, reversal, etc.)
        )
        
        # 3. Support/Resistance Detection
        self.support_resistance_head = nn.Sequential(
            nn.Linear(input_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3)  # 3 outputs: [support strength, resistance strength, current position]
        )
        
        # 4. Volatility Assessment (influenced by regime)
        regime_and_input_dim = input_dim + 4  # Add regime features
        self.volatility_head = nn.Sequential(
            nn.Linear(regime_and_input_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 0-1 value representing volatility
        )
        
        # 5. Liquidity Assessment
        self.liquidity_head = nn.Sequential(
            nn.Linear(input_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 0-1 value representing market liquidity
        )
        
        # 6. Entry/Exit Point Detection (influenced by patterns and S/R)
        pattern_sr_dim = 8 + 3  # Pattern + support/resistance features
        self.entry_exit_head = nn.Sequential(
            nn.Linear(input_dim + pattern_sr_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),  # [entry score, exit score]
            nn.Sigmoid()
        )
        
        # 7. Trading Factor Integration Layer
        trading_factors_dim = 4 + 8 + 3 + 1 + 1 + 2  # All previous components
        self.trading_factor_integration = nn.Sequential(
            nn.Linear(input_dim + trading_factors_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )
        
        # Policy head (direction and fraction) - now influenced by trading factors
        self.fc_policy_mean = nn.Linear(hidden_size // 2, 2)
        self.log_std = nn.Parameter(torch.zeros(2))
        
        # Value head (critic) - now influenced by trading factors
        self.fc_value = nn.Linear(hidden_size // 2, 1)
        
        # Shared encoder for all prediction heads
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size // 2),
            nn.ReLU()
        )
        
        # Create a module dictionary for each type of prediction head
        # This allows us to access them by name
        
        # Trend strength estimation (-1 to 1 scale)
        self.trend_strength = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1), 
            nn.Tanh()
        )
        
        # Add uncertainty-aware risk assessment
        self.risk_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # --- Instantiate the Dynamic Horizon Predictor --- 
        self.dynamic_predictor = DynamicHorizonPredictor(
            feature_size=intermediate_feature_size, 
            config=self.config # Pass config to predictor
        )
        # --- End Instantiation ---
    
    def forward(self, x, mask=None, explain=False, requested_horizon=None):
        """
        Forward pass through the network with chain of draft reasoning.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim]
            mask (torch.Tensor, optional): Attention mask. Defaults to None.
            explain (bool, optional): Whether to collect activation maps for explainability. Defaults to False.
            requested_horizon (int, optional): Target prediction horizon. Defaults to None.
            
        Returns:
            dict: Model outputs with keys for different prediction heads and reasoning chain
        """
        # Store explain mode
        self.explain_mode = explain
        self.activation_maps = {} if explain else None
        
        # Apply feature weights if available (element-wise multiplication)
        if hasattr(self, 'feature_weights') and self.feature_weights is not None:
            x = x * self.feature_weights.view(1, 1, -1)
        
        # Process sequence with transformer
        transformer_out = self.transformer(x, mask)
        if explain:
            self.activation_maps['transformer'] = transformer_out.detach().clone()
        
        # Get last time step for predictions
        last_hidden = transformer_out[:, -1]
        
        # ======= Chain of Draft Reasoning =======
        # 1. Market Regime Analysis
        market_regime = self.market_regime_head(last_hidden)
        
        # 2. Technical Pattern Recognition 
        patterns = self.pattern_recognition_head(last_hidden)
        
        # 3. Support/Resistance Detection
        support_resistance = self.support_resistance_head(last_hidden)
        
        # 4. Volatility Assessment (influenced by market regime)
        volatility_input = torch.cat([last_hidden, market_regime], dim=1)
        volatility = self.volatility_head(volatility_input)
        
        # 5. Liquidity Assessment
        liquidity = self.liquidity_head(last_hidden)
        
        # 6. Entry/Exit Point Detection
        pattern_sr_features = torch.cat([patterns, support_resistance], dim=1)
        entry_exit_input = torch.cat([last_hidden, pattern_sr_features], dim=1)
        entry_exit = self.entry_exit_head(entry_exit_input)
        
        # 7. Integrate all trading factors
        trading_factors = torch.cat([
            market_regime, 
            patterns, 
            support_resistance, 
            volatility, 
            liquidity, 
            entry_exit
        ], dim=1)
        
        # Final trading factor integration
        integrated_features = torch.cat([last_hidden, trading_factors], dim=1)
        decision_features = self.trading_factor_integration(integrated_features)
        
        # Store reasoning chain
        reasoning_chain = {
            'market_regime': market_regime,
            'patterns': patterns,
            'support_resistance': support_resistance,
            'volatility': volatility,
            'liquidity': liquidity,
            'entry_exit': entry_exit,
            'integrated_factors': decision_features
        }
        
        # Pass through policy network to get action distribution parameters
        action_mean = self.fc_policy_mean(decision_features)
        action_logstd = self.log_std.expand_as(action_mean)
        action_logstd = torch.clamp(action_logstd, -20, 2)  # Prevent numerical issues
        action_std = torch.exp(action_logstd)
        
        # Pass through critic network to get value
        value = self.fc_value(decision_features)
        
        # Get common embedding for predictions
        pred_hidden = self.shared_encoder(last_hidden)
        if explain:
            self.activation_maps['pred_hidden'] = pred_hidden.detach().clone()
        
        # Calculate risk assessment
        risk_assessment = self.risk_head(last_hidden)
        
        # Calculate horizon distribution
        horizon_mean = self.horizon_mean_head(decision_features)
        horizon_log_std = self.horizon_log_std_head(decision_features)
        horizon_std = torch.exp(horizon_log_std) + 1e-6  # Ensure positive std
        
        # Calculate conditional outcome prediction
        outcome_mean = None
        outcome_std = None
        outcome_confidence = None

        if requested_horizon is not None:
            # Ensure requested_horizon is a tensor on the correct device
            if not isinstance(requested_horizon, torch.Tensor):
                 # Assuming requested_horizon is a scalar or numpy array compatible value for the batch
                 # Create a tensor matching the batch size
                 batch_size = decision_features.shape[0]
                 horizon_tensor = torch.full((batch_size, 1), float(requested_horizon), 
                                             device=self.device, dtype=torch.float32)
            elif requested_horizon.ndim == 0: # If it's a scalar tensor
                 batch_size = decision_features.shape[0]
                 horizon_tensor = requested_horizon.expand(batch_size, 1).to(self.device)
            elif requested_horizon.ndim == 1: # If it's a batch of scalars
                 horizon_tensor = requested_horizon.unsqueeze(-1).to(self.device)
            else: # Assume it's already [batch_size, 1]
                 horizon_tensor = requested_horizon.to(self.device)

            # Encode the requested horizon
            encoded_horizon = self.horizon_encoder(horizon_tensor)
            
            # Combine features with encoded horizon
            combined_features = torch.cat([decision_features, encoded_horizon], dim=-1)
            
            # Pass through outcome prediction heads
            outcome_mean = self.outcome_mean_head(combined_features)
            outcome_log_std = self.outcome_log_std_head(combined_features)
            outcome_std = torch.exp(outcome_log_std) + 1e-6 # Ensure positive
            outcome_confidence = self.outcome_confidence_head(combined_features)

        # Trend strength
        trend = self.trend_strength(last_hidden)
        
        # Return all outputs as dictionary for maximum flexibility
        results = {
            'actions_mean': action_mean,
            'actions_logstd': action_logstd,
            'actions_std': action_std,
            'value': value,
            'horizon_mean': horizon_mean,
            'horizon_std': horizon_std,
            'trend_strength': trend,
            'transformer_output': transformer_out,
            'last_hidden': last_hidden,
            'risk_assessment': risk_assessment,
            'reasoning_chain': reasoning_chain,  # Chain of draft reasoning
            'outcome_mean': outcome_mean,
            'outcome_std': outcome_std,
            'outcome_confidence': outcome_confidence
        }
        
        if explain:
            results['activations'] = self.activation_maps
            results['gradients'] = self.gradients
        
        return results
    
    def _register_hooks(self):
        """Register hooks for computing gradients during backward pass"""
        def save_gradient(name):
            def hook(grad):
                self.gradients[name] = grad.detach().clone()
            return hook
        
        # Register hooks for key layers
        self.fc_policy_mean.weight.register_hook(save_gradient('fc_policy_mean'))
        self.fc_value.weight.register_hook(save_gradient('fc_value'))
        for horizon in self.horizons:
            horizon_name = f"h{horizon}"
            self.pred_mean_heads[f"pred_mean_{horizon_name}"].weight.register_hook(
                save_gradient(f'pred_mean_{horizon_name}')
            )
        
        # Access the weights of the first layer in the Sequential module
        if hasattr(self, 'trend_strength') and isinstance(self.trend_strength, nn.Sequential):
            if hasattr(self.trend_strength[0], 'weight'):
                self.trend_strength[0].weight.register_hook(save_gradient('trend_strength'))
    
    def explain_prediction(self, x, target_idx=0, target_horizon=None):
        """
        Generate feature attributions for a prediction using Grad-CAM.
        
        Args:
            x (torch.Tensor): Input tensor
            target_idx (int, optional): Target output index. Defaults to 0.
            target_horizon (int, optional): Target prediction horizon. Defaults to None.
            
        Returns:
            torch.Tensor: Feature attributions (importance scores)
        """
        # Enable gradient calculation
        self.explain_mode = True
        self.gradients = {}
        self.activation_maps = {}
        
        # Forward pass with explain mode
        x.requires_grad_(True)
        outputs = self(x, explain=True)
        
        # Determine target for gradient calculation
        if target_horizon is not None:
            # Target price prediction for specific horizon
            horizon_name = f"h{target_horizon}"
            target = outputs['predictions'][horizon_name][0, target_idx]
            target_name = f'horizon_{horizon_name}'
        else:
            # Default to action prediction (policy)
            target = outputs['actions_mean'][0, target_idx]
            target_name = 'action_mean'
        
        # Zero gradients
        self.zero_grad()
        
        # Backward pass to compute gradients
        target.backward(retain_graph=True)
        
        # Get transformer output and gradients
        transformer_output = self.activation_maps['transformer']
        
        # Calculate feature importance using GradCAM approach
        feature_importances = torch.zeros(x.shape[-1], device=self.device)
        
        # Compute weighted activation map
        for name, gradient in self.gradients.items():
            if name in self.activation_maps:
                activation = self.activation_maps[name]
                weights = gradient.mean(dim=0)
                weighted_activation = weights.unsqueeze(0) * activation
                feature_importances += weighted_activation.sum(dim=1)[0]
        
        # Normalize feature importances
        feature_importances = F.relu(feature_importances)  # Keep only positive influences
        if feature_importances.sum() > 0:
            feature_importances = feature_importances / feature_importances.sum()
        
        # Clear explain mode
        self.explain_mode = False
        
        # Return feature attributions directly as a tensor
        return feature_importances
    
    def get_decision_explanation(self, x, feature_names=None):
        """
        Generate human-readable explanation for a prediction.
        
        Args:
            x (torch.Tensor): Input tensor
            feature_names (list, optional): Names of input features. Defaults to None.
            
        Returns:
            dict: Explanation dictionary with text explanation and supporting data
        """
        # Default feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(x.shape[-1])]
        
        # Get attributions for action
        action_attr = self.explain_prediction(x, target_idx=0)
        
        # Get attributions for value
        # We trick the model by using a different target
        self.zero_grad()
        outputs = self(x, explain=True)
        value_target = outputs['value'][0]
        self.zero_grad()
        value_target.backward(retain_graph=True)
        
        # Calculate value attribution
        transformer_output = self.activation_maps['transformer']
        value_gradient = self.gradients.get('fc_value', None)
        value_importances = torch.zeros(x.shape[-1], device=self.device)
        
        if value_gradient is not None:
            weights = value_gradient.mean(dim=0)
            weighted_activation = weights.unsqueeze(0) * transformer_output
            value_importances = weighted_activation.sum(dim=1)[0]
            
            # Normalize
            value_importances = F.relu(value_importances)
            if value_importances.sum() > 0:
                value_importances = value_importances / value_importances.sum()
        
        # Get top features for action and value
        action_features = []
        value_features = []
        
        action_imp = action_attr.cpu().detach().numpy()
        value_imp = value_importances.cpu().detach().numpy()
        
        # Get top 5 features for each
        action_top5 = np.argsort(action_imp)[-5:][::-1]
        value_top5 = np.argsort(value_imp)[-5:][::-1]
        
        for idx in action_top5:
            if action_imp[idx] > 0.05:  # Only include significant features
                action_features.append({
                    'name': feature_names[idx],
                    'importance': float(action_imp[idx]),
                    'value': float(x[0, -1, idx].item())
                })
        
        for idx in value_top5:
            if value_imp[idx] > 0.05:  # Only include significant features
                value_features.append({
                    'name': feature_names[idx],
                    'importance': float(value_imp[idx]),
                    'value': float(x[0, -1, idx].item())
                })
        
        # Generate simple explanation text
        action_value = outputs['actions_mean'][0, 0].item()
        state_value = outputs['value'][0].item()
        
        action_direction = "BUY" if action_value > 0 else "SELL"
        confidence = "high" if abs(action_value) > 0.5 else "moderate" if abs(action_value) > 0.2 else "low"
        
        # Create explanation text
        explanation = f"Model suggests {action_direction} with {confidence} confidence. "
        explanation += f"Current state value estimate: {state_value:.2f}. "
        
        # Add feature contributions to explanation
        if action_features:
            explanation += "Action influenced by: "
            for i, feat in enumerate(action_features[:3]):  # Top 3 only for readability
                if i > 0:
                    explanation += ", "
                explanation += f"{feat['name']} ({feat['importance']:.2f})"
        
        # Reset explain mode
        self.explain_mode = False
        
        return {
            'explanation': explanation,
            'action_value': action_value,
            'state_value': state_value,
            'action_features': action_features,
            'value_features': value_features,
            'action_direction': action_direction,
            'confidence': confidence
        }

    def sample_predictions(self, means, stds, num_samples=10):
        """
        Generate samples from probabilistic predictions.
        
        Args:
            means (dict): Dictionary of prediction means by horizon
            stds (dict): Dictionary of prediction standard deviations by horizon
            num_samples (int): Number of samples to generate
            
        Returns:
            dict: Dictionary of prediction samples by horizon
        """
        samples = {}
        
        for horizon_name in means.keys():
            if horizon_name in stds:
                mean = means[horizon_name]
                std = stds[horizon_name]
                
                # Create normal distribution
                dist = Normal(mean, std)
                
                # Sample from distribution
                horizon_samples = dist.sample((num_samples,))
                
                samples[horizon_name] = horizon_samples
        
        return samples
    
    def calculate_prediction_intervals(self, means, stds, confidence_levels=[0.68, 0.95, 0.99]):
        """
        Calculate prediction intervals for different confidence levels.
        
        Args:
            means (dict): Dictionary of prediction means by horizon
            stds (dict): Dictionary of prediction standard deviations by horizon
            confidence_levels (list): List of confidence levels (0-1)
            
        Returns:
            dict: Dictionary of prediction intervals by horizon and confidence level
        """
        intervals = {}
        
        for horizon_name in means.keys():
            if horizon_name in stds:
                mean = means[horizon_name]
                std = stds[horizon_name]
                
                horizon_intervals = {}
                
                for conf_level in confidence_levels:
                    # Calculate z-score for confidence level
                    z_score = stats.norm.ppf((1 + conf_level) / 2)
                    
                    # Calculate interval bounds
                    lower_bound = mean - z_score * std
                    upper_bound = mean + z_score * std
                    
                    horizon_intervals[conf_level] = {
                        'lower': lower_bound,
                        'upper': upper_bound
                    }
                
                intervals[horizon_name] = horizon_intervals
        
        return intervals

    def get_reasoning_explanation(self, reasoning_chain, feature_names=None):
        """
        Generate human-readable explanation for the chain of draft reasoning.
        
        Args:
            reasoning_chain (dict): Reasoning chain from forward pass
            feature_names (list, optional): Names of input features. Defaults to None.
            
        Returns:
            dict: Explanation of reasoning chain steps
        """
        explanation = {}
        
        # Market regime explanation
        if 'market_regime' in reasoning_chain:
            regime_probs = reasoning_chain['market_regime'].detach().cpu().numpy()[0]
            regimes = ['trending', 'ranging', 'volatile', 'mixed']
            primary_regime = regimes[np.argmax(regime_probs)]
            regime_strength = np.max(regime_probs).item()
            
            explanation['market_regime'] = {
                'primary_regime': primary_regime,
                'strength': regime_strength,
                'probabilities': {r: float(p) for r, p in zip(regimes, regime_probs)}
            }
        
        # Technical patterns explanation
        if 'patterns' in reasoning_chain:
            pattern_scores = reasoning_chain['patterns'].detach().cpu().numpy()[0]
            pattern_types = ['trend_continuation', 'reversal', 'breakout', 'breakdown', 
                            'consolidation', 'support_test', 'resistance_test', 'volatility_expansion']
            
            # Get top 3 patterns
            top_indices = np.argsort(pattern_scores)[-3:][::-1]
            top_patterns = [(pattern_types[i], float(pattern_scores[i])) for i in top_indices]
            
            explanation['technical_patterns'] = {
                'top_patterns': top_patterns,
                'all_patterns': {p: float(s) for p, s in zip(pattern_types, pattern_scores)}
            }
        
        # Support/Resistance explanation
        if 'support_resistance' in reasoning_chain:
            sr_values = reasoning_chain['support_resistance'].detach().cpu().numpy()[0]
            
            explanation['support_resistance'] = {
                'support_strength': float(sr_values[0]),
                'resistance_strength': float(sr_values[1]),
                'price_position': float(sr_values[2])  # -1 to 1, negative = closer to support, positive = closer to resistance
            }
            
            # Interpret position
            if sr_values[2] < -0.3:
                explanation['support_resistance']['interpretation'] = "Price near support level"
            elif sr_values[2] > 0.3:
                explanation['support_resistance']['interpretation'] = "Price near resistance level"
            else:
                explanation['support_resistance']['interpretation'] = "Price in middle zone"
        
        # Entry/Exit point explanation
        if 'entry_exit' in reasoning_chain:
            entry_exit = reasoning_chain['entry_exit'].detach().cpu().numpy()[0]
            
            explanation['entry_exit'] = {
                'entry_signal': float(entry_exit[0]),
                'exit_signal': float(entry_exit[1])
            }
            
            # Interpret signals
            if entry_exit[0] > 0.7:
                explanation['entry_exit']['interpretation'] = "Strong entry signal"
            elif entry_exit[0] > 0.5:
                explanation['entry_exit']['interpretation'] = "Moderate entry signal"
            elif entry_exit[1] > 0.7:
                explanation['entry_exit']['interpretation'] = "Strong exit signal"
            elif entry_exit[1] > 0.5:
                explanation['entry_exit']['interpretation'] = "Moderate exit signal"
            else:
                explanation['entry_exit']['interpretation'] = "No clear entry or exit signal"
        
        # Risk factors explanation
        risk_factors = {}
        if 'volatility' in reasoning_chain:
            risk_factors['volatility'] = float(reasoning_chain['volatility'].detach().cpu().numpy()[0][0])
        
        if 'liquidity' in reasoning_chain:
            risk_factors['liquidity'] = float(reasoning_chain['liquidity'].detach().cpu().numpy()[0][0])
            
        if risk_factors:
            explanation['risk_factors'] = risk_factors
        
        return explanation

    def save_model(self, path):
        """
        Save model to file with proper device handling.
        
        Args:
            path (str): Path to save model
        """
        # Save model to CPU for better compatibility
        cpu_state_dict = {}
        for key, param in self.state_dict().items():
            cpu_state_dict[key] = param.cpu()
        
        torch.save({
            'state_dict': cpu_state_dict,
            'input_dim': self.input_dim,
            'hidden_size': self.hidden_size,
            'device': 'cpu'  # Always save as CPU
        }, path)
        
    @classmethod
    def load_model(cls, path, device=None):
        """
        Load model from file with proper device handling.
        
        Args:
            path (str): Path to load model from
            device (str, optional): Device to load model to. Defaults to None.
            
        Returns:
            ActorCritic: Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location='cpu')
        
        # Get model configuration
        input_dim = checkpoint.get('input_dim')
        hidden_size = checkpoint.get('hidden_size')
        target_device = device or checkpoint.get('device', 'cpu')
        
        # Create new model
        model = cls(
            input_dim=input_dim,
            hidden_size=hidden_size,
            device='cpu'  # Initialize on CPU first
        )
        
        # Load state dict
        state_dict = checkpoint.get('state_dict')
        model.load_state_dict(state_dict)
        
        # Move to target device
        model.to(target_device)
        
        return model

class CNNFeatureExtractor(nn.Module):
    """
    CNN-based feature extractor for processing raw price data.
    
    Can be used as a preprocessing step before feeding data to the main policy network.
    """
    def __init__(self, input_channels, output_features):
        """
        Initialize CNN Feature Extractor.
        
        Args:
            input_channels (int): Number of input channels (features per time step).
            output_features (int): Number of output features.
        """
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and normalization
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.norm1 = nn.BatchNorm1d(32)
        self.norm2 = nn.BatchNorm1d(64)
        self.norm3 = nn.BatchNorm1d(128)
        
        # Final fully connected layer
        self.fc = nn.Linear(128, output_features)
        
    def forward(self, x):
        """
        Forward pass through the CNN feature extractor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, sequence_length, channels).
            
        Returns:
            torch.Tensor: Extracted features of shape (batch, sequence_length // 4, output_features).
        """
        # Transpose to put channels dim in the correct position for Conv1d
        # From (batch, seq_len, channels) to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.norm3(self.conv3(x)))
        
        # Transpose back to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Apply final FC layer to each time step
        x = self.fc(x)
        
        return x


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for integrating multiple data sources.
    
    Useful for combining price data with order book data or other features.
    """
    def __init__(self, dim, num_heads=8):
        """
        Initialize cross-attention module.
        
        Args:
            dim (int): Feature dimension.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # Query, key, value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x, context):
        """
        Forward pass for cross-attention.
        
        Args:
            x (torch.Tensor): Primary input of shape (batch, seq_len, dim).
            context (torch.Tensor): Context input of shape (batch, context_len, dim).
            
        Returns:
            torch.Tensor: Attention output of shape (batch, seq_len, dim).
        """
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, device="cpu"):
        """
        Transformer Block for time series processing.
        
        Args:
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multiheadattention models.
            dim_feedforward (int, optional): The dimension of the feedforward network model. Defaults to 2048.
            dropout (float, optional): The dropout value. Defaults to 0.1.
            device (str, optional): Device to run model on. Defaults to "cpu".
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Move to device
        self.to(device)

def create_model(input_dim, hidden_size, model_type='actor_critic', device="cpu", config=None):
    """
    Factory function to create different models.
    Removed horizons argument.
    """
    if model_type == 'actor_critic_transformer': # Example specific name
         model = ActorCritic(input_dim=input_dim,
                             hidden_size=hidden_size,
                             model_type='transformer', # Pass type if needed internally
                             device=device,
                             config=config) # Pass config, no horizon_config
    elif model_type == 'actor_critic': # Handle generic case if needed
         # Assuming transformer is the default complex ActorCritic now
         model = ActorCritic(input_dim=input_dim,
                             hidden_size=hidden_size,
                             model_type='transformer', 
                             device=device,
                             config=config)
    # --- Commented out old model types that need updating --- 
    # elif model_type.lower() == 'lstm':
    #     # TODO: Update LSTMPolicyNetwork to use dynamic horizon predictor or remove
    #     log("[WARNING] LSTMPolicyNetwork not updated for dynamic horizons. Using ActorCritic instead.", "warning")
    #     model = ActorCritic(input_dim=input_dim, hidden_size=hidden_size, model_type='transformer', device=device, config=config)
    #     # model = LSTMPolicyNetwork(input_dim, hidden_size, device=device, config=config) # Original call needs update
    # elif model_type.lower() == 'hybrid':
    #     # TODO: Update HybridPolicyNetwork to use dynamic horizon predictor or remove
    #     log("[WARNING] HybridPolicyNetwork not updated for dynamic horizons. Using ActorCritic instead.", "warning")
    #     model = ActorCritic(input_dim=input_dim, hidden_size=hidden_size, model_type='transformer', device=device, config=config)
    #     # model = HybridPolicyNetwork(input_dim, hidden_size, device=device, config=config) # Original call needs update
    else:
        log(f"Unknown model type: {model_type}. Defaulting to ActorCritic.", "warning")
        model = ActorCritic(input_dim=input_dim, hidden_size=hidden_size, model_type='transformer', device=device, config=config)
    
    # Move model to device before returning
    model.to(device)
    return model


if __name__ == "__main__":
    # Test the model
    try:
        # Get the count_trainable_parameters function from utils
        count_trainable_parameters = utils_module.count_trainable_parameters
        
        # Create a model with standard settings
        model = ActorCritic(
            input_dim=50, 
            hidden_size=128,
            model_type="transformer",
            max_seq_len=288,
            device="cpu"
        )
        
        # Display the model architecture
        print(f"Model Architecture:\n{model}")
        
        # Count and display trainable parameters
        total_params = count_trainable_parameters(model)
        print(f"Total trainable parameters: {total_params:,}")
        
        # Test forward pass with random input
        batch_size = 4
        seq_len = 10
        input_features = torch.randn(batch_size, seq_len, 50)
        hidden = (
            torch.zeros(1, batch_size, 128),
            torch.zeros(1, batch_size, 128)
        )
        
        action_probs, state_values, pred_dists, hidden_out = model(input_features, hidden)
        
        print(f"Action probabilities shape: {action_probs.shape}")
        print(f"State values shape: {state_values.shape}")
        print(f"Prediction distributions shape: {len(pred_dists)} horizons")
        print(f"LSTM hidden state shape: {hidden_out[0].shape}")
        
        print("Model test successful!")
    except Exception as e:
        print(f"Error testing model: {e}")
