#!/usr/bin/env python
"""
Environment Interfaces

This module defines the standard interfaces and data structures for the trading environment components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Type
from enum import Enum
from datetime import datetime, timedelta


class Position:
    """
    Trading position representation.
    """
    def __init__(self, size_btc, entry_price, entry_step):
        """
        Initialize a new position.
        
        Args:
            size_btc (float): Position size in BTC.
            entry_price (float): Entry price in USD.
            entry_step (int): Entry time step.
        """
        self.size_btc = size_btc
        self.entry_price = entry_price
        self.entry_step = entry_step
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            "size_btc": self.size_btc,
            "entry_price": self.entry_price,
            "entry_step": self.entry_step
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary representation"""
        return cls(
            data["size_btc"],
            data["entry_price"],
            data["entry_step"]
        )


class Order:
    """
    Trading order representation.
    """
    def __init__(self, size_btc, direction, entry_step, timeout, reference_price=None, reference_step=None):
        """
        Initialize a new order.
        
        Args:
            size_btc (float): Order size in BTC.
            direction (float): Order direction (1.0 for buy, -1.0 for sell).
            entry_step (int): Time step when order was created.
            timeout (int): Order timeout in time steps.
            reference_price (float, optional): Reference price for sell orders. Defaults to None.
            reference_step (int, optional): Reference time step for sell orders. Defaults to None.
        """
        self.size_btc = size_btc
        self.direction = direction
        self.entry_step = entry_step
        self.timeout = timeout
        self.reference_price = reference_price
        self.reference_step = reference_step
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            "size_btc": self.size_btc,
            "direction": self.direction,
            "entry_step": self.entry_step,
            "timeout": self.timeout,
            "reference_price": self.reference_price,
            "reference_step": self.reference_step
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary representation"""
        return cls(
            data["size_btc"],
            data["direction"],
            data["entry_step"],
            data["timeout"],
            data.get("reference_price"),
            data.get("reference_step")
        )


class Trade:
    """
    Completed trade representation.
    """
    def __init__(self, profit, percentage_gain, hold_time, entry_step):
        """
        Initialize a new trade record.
        
        Args:
            profit (float): Profit/loss amount in USD.
            percentage_gain (float): Percentage gain/loss.
            hold_time (int): Hold time in time steps.
            entry_step (int): Entry time step.
        """
        self.profit = profit
        self.percentage_gain = percentage_gain
        self.hold_time = hold_time
        self.entry_step = entry_step
    
    def to_tuple(self):
        """Convert to tuple representation"""
        return (self.profit, self.percentage_gain, self.hold_time, self.entry_step)
    
    @classmethod
    def from_tuple(cls, data):
        """Create from tuple representation"""
        return cls(data[0], data[1], data[2], data[3] if len(data) > 3 else 0)


class TradingEnvInterface(ABC):
    """
    Abstract interface for trading environments.
    
    Defines the standard methods that all trading environments must implement.
    """
    
    @abstractmethod
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            object: Initial observation.
        """
        pass
    
    @abstractmethod
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take.
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        pass
    
    @abstractmethod
    def _get_observation(self):
        """
        Get current observation.
        
        Returns:
            object: Current observation.
        """
        pass
    
    @abstractmethod
    def _calculate_portfolio_risk(self):
        """
        Calculate portfolio risk metrics.
        
        Returns:
            dict: Risk metrics.
        """
        pass


class EnvRegistry:
    """
    Registry for environment implementations.
    
    Allows for dynamic registration and retrieval of environment implementations.
    Helps prevent circular imports by providing a central registry.
    """
    _registry = {}
    
    @classmethod
    def register(cls, name, env_class):
        """
        Register an environment class.
        
        Args:
            name (str): Environment name.
            env_class: Environment class.
        """
        cls._registry[name.lower()] = env_class
    
    @classmethod
    def get(cls, name):
        """
        Get an environment class by name.
        
        Args:
            name (str): Environment name.
            
        Returns:
            Environment class or None if not found.
        """
        return cls._registry.get(name.lower())


def validate_action(action):
    """
    Validate agent action is in correct format for environment.
    
    Args:
        action: Action to validate.
        
    Returns:
        bool: True if action is valid, False otherwise.
    """
    if not isinstance(action, (list, tuple, np.ndarray)) or len(action) != 2:
        return False
    return True


def format_agent_action(action):
    """
    Convert agent action to standard environment format.
    
    Args:
        action: Agent action.
        
    Returns:
        list: Standardized action [direction, fraction].
    """
    direction, fraction = float(action[0]), float(action[1])
    # Clamp values to valid ranges
    direction = max(-1.0, min(1.0, direction))
    fraction = max(0.0, min(1.0, fraction))
    return [direction, fraction]


class WithdrawalType(Enum):
    """Enum for different types of withdrawal requests"""
    TIMED = 0       # Withdrawal with a specific deadline
    EMERGENCY = 1   # Immediate withdrawal needed
    STANDARD = 2    # Standard withdrawal with no specific timeframe

class WithdrawalStatus(Enum):
    """Enum for tracking the status of withdrawal requests"""
    PENDING = 0     # Withdrawal is being processed
    PARTIAL = 1     # Withdrawal has been partially fulfilled
    COMPLETE = 2    # Withdrawal has been fully fulfilled
    CANCELED = 3    # Withdrawal was canceled

class Withdrawal:
    """
    Class representing a withdrawal request in the trading system.
    Tracks amount, type, deadline, and fulfillment status.
    """
    def __init__(
        self,
        withdrawal_id: str,
        amount_usd: float,
        withdrawal_type: WithdrawalType,
        deadline: Optional[datetime] = None,
        created_at: Optional[datetime] = None
    ):
        """
        Initialize a withdrawal request.

        Args:
            withdrawal_id (str): Unique identifier for the withdrawal
            amount_usd (float): Amount requested in USD (not USDT)
            withdrawal_type (WithdrawalType): Type of withdrawal request
            deadline (datetime, optional): Target date for timed withdrawals
            created_at (datetime, optional): When the withdrawal was created
        """
        self.withdrawal_id = withdrawal_id
        self.amount_usd = amount_usd
        self.withdrawal_type = withdrawal_type
        self.deadline = deadline
        self.created_at = created_at or datetime.now()
        self.status = WithdrawalStatus.PENDING
        self.fulfilled_amount = 0.0  # Amount fulfilled so far
        self.reserved_amount = 0.0   # Amount currently reserved from profits
        self.last_updated = self.created_at
        
    def update_reserved(self, amount: float) -> None:
        """
        Update the amount reserved for this withdrawal.
        
        Args:
            amount (float): Additional amount to reserve
        """
        self.reserved_amount += amount
        self.last_updated = datetime.now()
        
    def fulfill(self, amount: float) -> float:
        """
        Mark a portion of the withdrawal as fulfilled.
        
        Args:
            amount (float): Amount to fulfill
            
        Returns:
            float: Remaining amount needed
        """
        fulfill_amount = min(amount, self.amount_usd - self.fulfilled_amount)
        self.fulfilled_amount += fulfill_amount
        
        # Update status
        if self.fulfilled_amount >= self.amount_usd:
            self.status = WithdrawalStatus.COMPLETE
        elif self.fulfilled_amount > 0:
            self.status = WithdrawalStatus.PARTIAL
            
        self.last_updated = datetime.now()
        return self.amount_usd - self.fulfilled_amount
    
    def cancel(self) -> float:
        """
        Cancel this withdrawal request.
        
        Returns:
            float: Amount that was reserved but not fulfilled
        """
        self.status = WithdrawalStatus.CANCELED
        self.last_updated = datetime.now()
        return self.reserved_amount - self.fulfilled_amount
    
    def get_remaining_amount(self) -> float:
        """
        Get the remaining amount needed to fulfill this withdrawal.
        
        Returns:
            float: Remaining amount
        """
        return self.amount_usd - self.fulfilled_amount
    
    def get_days_until_deadline(self) -> Optional[float]:
        """
        Calculate days remaining until the deadline.
        
        Returns:
            float or None: Days until deadline, or None if no deadline
        """
        if not self.deadline:
            return None
        
        delta = self.deadline - datetime.now()
        return max(0, delta.total_seconds() / (24 * 3600))
    
    def get_urgency(self) -> float:
        """
        Calculate urgency factor (0.0-1.0) based on deadline or type.
        Higher values indicate more urgency.
        
        Returns:
            float: Urgency factor
        """
        if self.withdrawal_type == WithdrawalType.EMERGENCY:
            return 1.0
            
        days_left = self.get_days_until_deadline()
        if days_left is None:
            return 0.1  # Standard withdrawals have low urgency
            
        # Increase urgency as deadline approaches
        # 1.0 = very urgent (today), 0.2 = not urgent (30+ days)
        if days_left <= 1:
            return 0.9
        elif days_left <= 7:
            return 0.7
        elif days_left <= 14:
            return 0.5
        elif days_left <= 30:
            return 0.3
        else:
            return 0.2
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            dict: Dictionary representation
        """
        return {
            "withdrawal_id": self.withdrawal_id,
            "amount_usd": self.amount_usd,
            "withdrawal_type": self.withdrawal_type.value,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "fulfilled_amount": self.fulfilled_amount,
            "reserved_amount": self.reserved_amount,
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Withdrawal':
        """
        Create a Withdrawal instance from a dictionary.
        
        Args:
            data (dict): Dictionary representation
            
        Returns:
            Withdrawal: New instance
        """
        w = cls(
            withdrawal_id=data["withdrawal_id"],
            amount_usd=data["amount_usd"],
            withdrawal_type=WithdrawalType(data["withdrawal_type"]),
            deadline=datetime.fromisoformat(data["deadline"]) if data["deadline"] else None,
            created_at=datetime.fromisoformat(data["created_at"])
        )
        w.status = WithdrawalStatus(data["status"])
        w.fulfilled_amount = data["fulfilled_amount"]
        w.reserved_amount = data["reserved_amount"]
        w.last_updated = datetime.fromisoformat(data["last_updated"])
        return w


if __name__ == "__main__":
    # Demonstration of interfaces usage
    import numpy as np
    
    # Create position
    position = Position(0.5, 50000.0, 100)
    position_dict = position.to_dict()
    restored_position = Position.from_dict(position_dict)
    print(f"Position: {restored_position.size_btc} BTC at ${restored_position.entry_price}")
    
    # Create order
    order = Order(0.2, 1.0, 100, 10)
    order_dict = order.to_dict()
    restored_order = Order.from_dict(order_dict)
    print(f"Order: {restored_order.size_btc} BTC, direction {restored_order.direction}")
    
    # Create trade
    trade = Trade(500.0, 2.5, 50, 75)
    trade_tuple = trade.to_tuple()
    restored_trade = Trade.from_tuple(trade_tuple)
    print(f"Trade: ${restored_trade.profit} profit, {restored_trade.percentage_gain}% gain")
    
    # Test action validation
    valid_action = [0.5, 0.2]
    invalid_action = [0.5]
    print(f"Valid action: {validate_action(valid_action)}")
    print(f"Invalid action: {validate_action(invalid_action)}")
    
    # Test action formatting
    out_of_range_action = [1.5, 1.2]
    formatted_action = format_agent_action(out_of_range_action)
    print(f"Formatted action: {formatted_action}")
    
    # Register a dummy environment
    EnvRegistry.register("dummy", lambda df, config, device: None)
    print(f"Registered environments: {EnvRegistry.list()}")
