from abc import ABC, abstractmethod
import os
import torch

class Agent(ABC):
    """
    Base class for all agents.
    """
    
    def __init__(self, state_dim, action_dim, device=None, config=None):
        """
        Initialize the agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            device (str, optional): Device to use (cuda or cpu)
            config (dict, optional): Configuration for the agent
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Initialize memory optimization flags
        self.use_mixed_precision = self.config.get("USE_MIXED_PRECISION", False)
        self.use_gradient_checkpointing = self.config.get("USE_GRADIENT_CHECKPOINTING", False)
        self.gradient_accumulation_steps = self.config.get("GRADIENT_ACCUMULATION_STEPS", 1)
        self._grad_accumulation_count = 0
        
        # Initialize mixed precision scaler if needed
        if self.use_mixed_precision and torch.cuda.is_available():
            from torch.cuda import amp
            self.scaler = amp.GradScaler()
    
    @abstractmethod
    def get_action(self, state):
        """
        Get an action from the agent based on the current state.
        
        Args:
            state (numpy.ndarray): Current state
            
        Returns:
            numpy.ndarray: Action to take
        """
        pass
    
    @abstractmethod
    def train(self, state, action, reward, next_state, done):
        """
        Train the agent on a single transition.
        
        Args:
            state (numpy.ndarray): Current state
            action (numpy.ndarray): Action taken
            reward (float): Reward received
            next_state (numpy.ndarray): Next state
            done (bool): Whether the episode is done
        """
        pass
    
    def get_optimizer(self):
        """
        Get the optimizer used by the agent.
        
        Returns:
            torch.optim.Optimizer: Optimizer
        """
        return None
    
    def get_metrics(self):
        """
        Get metrics for the agent's performance.
        
        Returns:
            dict: Dictionary of metrics
        """
        return {}
    
    def optimize_step(self, loss, optimizer):
        """
        Perform an optimization step with support for mixed precision and gradient accumulation.
        
        Args:
            loss (torch.Tensor): Loss to optimize
            optimizer (torch.optim.Optimizer): Optimizer to use
            
        Returns:
            float: Loss value
        """
        # Skip if no optimizer
        if optimizer is None:
            return loss.item() if hasattr(loss, "item") else loss
        
        # Scale loss if using gradient accumulation
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        
        # Mixed precision training
        if self.use_mixed_precision and hasattr(self, "scaler"):
            # Scale gradients and perform backward pass
            self.scaler.scale(loss).backward()
            
            # Only step and update if this is an update step
            if self.gradient_accumulation_steps == 1 or self._grad_accumulation_count == 0:
                # Unscale before clip to get correct gradient values
                self.scaler.unscale_(optimizer)
                
                # Clip gradients if configured
                if "CLIP_GRAD" in self.config and self.config["CLIP_GRAD"] > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.config["CLIP_GRAD"])
                
                # Step optimizer and update scaler
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
        else:
            # Regular backward pass
            loss.backward()
            
            # Only step if this is an update step
            if self.gradient_accumulation_steps == 1 or self._grad_accumulation_count == 0:
                # Clip gradients if configured
                if "CLIP_GRAD" in self.config and self.config["CLIP_GRAD"] > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.config["CLIP_GRAD"])
                
                # Step optimizer
                optimizer.step()
                optimizer.zero_grad()
        
        # Update gradient accumulation counter
        if self.gradient_accumulation_steps > 1:
            self._grad_accumulation_count = (self._grad_accumulation_count + 1) % self.gradient_accumulation_steps
        
        return loss.item() if hasattr(loss, "item") else loss
    
    def parameters(self):
        """
        Get the parameters of the agent.
        
        Returns:
            list: List of parameters
        """
        # Default implementation for agents with no parameters
        return []
    
    def state_dict(self):
        """
        Get the state dict of the agent.
        
        Returns:
            dict: State dict
        """
        # Default implementation for agents with no state
        return {}
    
    def load_state_dict(self, state_dict):
        """
        Load the state dict of the agent.
        
        Args:
            state_dict (dict): State dict
        """
        # Default implementation for agents with no state
        pass
    
    def save(self, path):
        """
        Save the agent to a file.
        
        Args:
            path (str): Path to save to
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """
        Load the agent from a file.
        
        Args:
            path (str): Path to load from
        """
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict) 