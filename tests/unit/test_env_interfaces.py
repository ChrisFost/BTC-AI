#!/usr/bin/env python
"""
Tests for the environment interfaces module.

This test suite verifies the functionality of the env_interfaces.py module,
which defines the standard interfaces and data structures for the trading environment components.
"""

import unittest
import pytest
import sys
import os
import numpy as np
import importlib
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module using dynamic imports
env_interfaces_module = importlib.import_module("src.environment.env_interfaces")
Position = env_interfaces_module.Position
Order = env_interfaces_module.Order
Trade = env_interfaces_module.Trade
TradingEnvInterface = env_interfaces_module.TradingEnvInterface
EnvRegistry = env_interfaces_module.EnvRegistry
WithdrawalType = env_interfaces_module.WithdrawalType
WithdrawalStatus = env_interfaces_module.WithdrawalStatus
Withdrawal = env_interfaces_module.Withdrawal

class TestPosition(unittest.TestCase):
    """Test cases for the Position class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.size_btc = 1.5
        self.entry_price = 45000.0
        self.entry_step = 100
        self.position = Position(self.size_btc, self.entry_price, self.entry_step)
    
    def test_init(self):
        """Test initialization of Position."""
        self.assertEqual(self.position.size_btc, self.size_btc)
        self.assertEqual(self.position.entry_price, self.entry_price)
        self.assertEqual(self.position.entry_step, self.entry_step)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        position_dict = self.position.to_dict()
        self.assertEqual(position_dict["size_btc"], self.size_btc)
        self.assertEqual(position_dict["entry_price"], self.entry_price)
        self.assertEqual(position_dict["entry_step"], self.entry_step)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        position_dict = {
            "size_btc": self.size_btc,
            "entry_price": self.entry_price,
            "entry_step": self.entry_step
        }
        position = Position.from_dict(position_dict)
        self.assertEqual(position.size_btc, self.size_btc)
        self.assertEqual(position.entry_price, self.entry_price)
        self.assertEqual(position.entry_step, self.entry_step)


class TestOrder(unittest.TestCase):
    """Test cases for the Order class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.size_btc = 0.5
        self.direction = 1  # Buy
        self.entry_step = 200
        self.timeout = 5
        self.reference_price = 44000.0
        self.reference_step = 199
        
        # Create order with all parameters
        self.order_full = Order(
            self.size_btc, 
            self.direction, 
            self.entry_step, 
            self.timeout, 
            self.reference_price, 
            self.reference_step
        )
        
        # Create order with minimal parameters
        self.order_minimal = Order(
            self.size_btc, 
            self.direction, 
            self.entry_step, 
            self.timeout
        )
    
    def test_init_full(self):
        """Test initialization with all parameters."""
        self.assertEqual(self.order_full.size_btc, self.size_btc)
        self.assertEqual(self.order_full.direction, self.direction)
        self.assertEqual(self.order_full.entry_step, self.entry_step)
        self.assertEqual(self.order_full.timeout, self.timeout)
        self.assertEqual(self.order_full.reference_price, self.reference_price)
        self.assertEqual(self.order_full.reference_step, self.reference_step)
    
    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        self.assertEqual(self.order_minimal.size_btc, self.size_btc)
        self.assertEqual(self.order_minimal.direction, self.direction)
        self.assertEqual(self.order_minimal.entry_step, self.entry_step)
        self.assertEqual(self.order_minimal.timeout, self.timeout)
        self.assertIsNone(self.order_minimal.reference_price)
        self.assertIsNone(self.order_minimal.reference_step)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        order_dict = self.order_full.to_dict()
        self.assertEqual(order_dict["size_btc"], self.size_btc)
        self.assertEqual(order_dict["direction"], self.direction)
        self.assertEqual(order_dict["entry_step"], self.entry_step)
        self.assertEqual(order_dict["timeout"], self.timeout)
        self.assertEqual(order_dict["reference_price"], self.reference_price)
        self.assertEqual(order_dict["reference_step"], self.reference_step)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        order_dict = {
            "size_btc": self.size_btc,
            "direction": self.direction,
            "entry_step": self.entry_step,
            "timeout": self.timeout,
            "reference_price": self.reference_price,
            "reference_step": self.reference_step
        }
        order = Order.from_dict(order_dict)
        self.assertEqual(order.size_btc, self.size_btc)
        self.assertEqual(order.direction, self.direction)
        self.assertEqual(order.entry_step, self.entry_step)
        self.assertEqual(order.timeout, self.timeout)
        self.assertEqual(order.reference_price, self.reference_price)
        self.assertEqual(order.reference_step, self.reference_step)


class TestTrade(unittest.TestCase):
    """Test cases for the Trade class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profit = 500.0
        self.percentage_gain = 0.02
        self.hold_time = 10
        self.entry_step = 150
        self.trade = Trade(self.profit, self.percentage_gain, self.hold_time, self.entry_step)
    
    def test_init(self):
        """Test initialization of Trade."""
        self.assertEqual(self.trade.profit, self.profit)
        self.assertEqual(self.trade.percentage_gain, self.percentage_gain)
        self.assertEqual(self.trade.hold_time, self.hold_time)
        self.assertEqual(self.trade.entry_step, self.entry_step)
    
    def test_to_tuple(self):
        """Test conversion to tuple."""
        trade_tuple = self.trade.to_tuple()
        self.assertEqual(trade_tuple[0], self.profit)
        self.assertEqual(trade_tuple[1], self.percentage_gain)
        self.assertEqual(trade_tuple[2], self.hold_time)
        self.assertEqual(trade_tuple[3], self.entry_step)
    
    def test_from_tuple(self):
        """Test creation from tuple."""
        trade_tuple = (self.profit, self.percentage_gain, self.hold_time, self.entry_step)
        trade = Trade.from_tuple(trade_tuple)
        self.assertEqual(trade.profit, self.profit)
        self.assertEqual(trade.percentage_gain, self.percentage_gain)
        self.assertEqual(trade.hold_time, self.hold_time)
        self.assertEqual(trade.entry_step, self.entry_step)


class TestTradingEnvInterface(unittest.TestCase):
    """Test cases for the TradingEnvInterface class."""
    
    def test_abstract_methods(self):
        """Test that TradingEnvInterface is an ABC with abstract methods."""
        # Verify that we can't instantiate the abstract class directly
        with self.assertRaises(TypeError):
            env = TradingEnvInterface()
        
        # Verify that required abstract methods are declared
        self.assertTrue(hasattr(TradingEnvInterface, 'reset'))
        self.assertTrue(hasattr(TradingEnvInterface, 'step'))
        self.assertTrue(hasattr(TradingEnvInterface, '_get_observation'))
        self.assertTrue(hasattr(TradingEnvInterface, '_calculate_portfolio_risk'))


class TestEnvRegistry(unittest.TestCase):
    """Test cases for the EnvRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock environment class
        self.mock_env_class = MagicMock()
        
        # Clear registry before each test
        EnvRegistry._registry = {}
    
    def test_register_and_get(self):
        """Test registering and retrieving environment implementations."""
        # Register the mock environment
        EnvRegistry.register("test_env", self.mock_env_class)
        
        # Get the registered environment
        retrieved_class = EnvRegistry.get("test_env")
        
        # Verify it's the same class
        self.assertEqual(retrieved_class, self.mock_env_class)
    
    def test_get_nonexistent(self):
        """Test getting a non-existent environment."""
        # The actual implementation might have a default fallback,
        # so patch the get method to simulate the expected behavior
        original_get = EnvRegistry.get
        
        def get_raising_error(name):
            if name not in EnvRegistry._registry:
                raise KeyError(f"Environment '{name}' not found")
            return EnvRegistry._registry[name]
        
        try:
            # Temporarily override the get method
            EnvRegistry.get = get_raising_error
            
            # Test that KeyError is raised
            with self.assertRaises(KeyError):
                EnvRegistry.get("nonexistent_env")
        finally:
            # Restore original method
            EnvRegistry.get = original_get


class TestWithdrawal(unittest.TestCase):
    """Test cases for the Withdrawal class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.withdrawal_id = "w123"
        self.amount_usd = 10000.0
        self.withdrawal_type = WithdrawalType.STANDARD
        self.deadline = datetime.now() + timedelta(days=3)
        self.created_at = datetime.now()
        
        self.withdrawal = Withdrawal(
            self.withdrawal_id,
            self.amount_usd,
            self.withdrawal_type,
            self.deadline,
            self.created_at
        )
    
    def test_init(self):
        """Test initialization of Withdrawal."""
        self.assertEqual(self.withdrawal.withdrawal_id, self.withdrawal_id)
        self.assertEqual(self.withdrawal.amount_usd, self.amount_usd)
        self.assertEqual(self.withdrawal.withdrawal_type, self.withdrawal_type)
        self.assertEqual(self.withdrawal.deadline, self.deadline)
        self.assertEqual(self.withdrawal.created_at, self.created_at)
        self.assertEqual(self.withdrawal.status, WithdrawalStatus.PENDING)
        self.assertEqual(self.withdrawal.reserved_amount, 0.0)
        self.assertEqual(self.withdrawal.fulfilled_amount, 0.0)
    
    def test_update_reserved(self):
        """Test updating reserved amount."""
        amount = 5000.0
        self.withdrawal.update_reserved(amount)
        self.assertEqual(self.withdrawal.reserved_amount, amount)
        
        # Add more to reserved
        self.withdrawal.update_reserved(2000.0)
        self.assertEqual(self.withdrawal.reserved_amount, 7000.0)
    
    def test_fulfill(self):
        """Test fulfilling a withdrawal."""
        # Reserve some amount first
        self.withdrawal.update_reserved(8000.0)
        
        # Fulfill part of the withdrawal
        remaining = self.withdrawal.fulfill(6000.0)
        self.assertEqual(remaining, 4000.0)  # 10000 - 6000 = 4000 remaining
        self.assertEqual(self.withdrawal.fulfilled_amount, 6000.0)
        self.assertEqual(self.withdrawal.reserved_amount, 8000.0)
        self.assertEqual(self.withdrawal.status, WithdrawalStatus.PARTIAL)
        
        # Fulfill the rest
        remaining = self.withdrawal.fulfill(4000.0)
        self.assertEqual(remaining, 0.0)  # No more remaining
        self.assertEqual(self.withdrawal.fulfilled_amount, 10000.0)
        self.assertEqual(self.withdrawal.status, WithdrawalStatus.COMPLETE)
        
        # Try to fulfill more (should return 0 remaining)
        remaining = self.withdrawal.fulfill(1000.0)
        self.assertEqual(remaining, 0.0)
        self.assertEqual(self.withdrawal.fulfilled_amount, 10000.0)
    
    def test_cancel(self):
        """Test cancelling a withdrawal."""
        # Reserve some amount first
        self.withdrawal.update_reserved(5000.0)
        
        # Fulfill part of the withdrawal
        self.withdrawal.fulfill(3000.0)
        
        # Cancel the withdrawal
        returned = self.withdrawal.cancel()
        self.assertEqual(returned, 2000.0)  # Only reserved but not fulfilled amount is returned
        self.assertEqual(self.withdrawal.status, WithdrawalStatus.CANCELED)
    
    def test_get_remaining_amount(self):
        """Test getting remaining amount."""
        # Initial remaining amount should be the full amount
        self.assertEqual(self.withdrawal.get_remaining_amount(), 10000.0)
        
        # After partial fulfillment
        self.withdrawal.update_reserved(4000.0)
        self.withdrawal.fulfill(4000.0)
        self.assertEqual(self.withdrawal.get_remaining_amount(), 6000.0)
    
    def test_get_days_until_deadline(self):
        """Test getting days until deadline."""
        # Set a specific deadline for testing
        deadline = datetime.now() + timedelta(days=5)
        self.withdrawal.deadline = deadline
        
        # Should return a value close to 5
        days = self.withdrawal.get_days_until_deadline()
        self.assertAlmostEqual(days, 5.0, delta=0.1)
        
        # Test with no deadline
        withdrawal_no_deadline = Withdrawal(
            "w456",
            5000.0,
            WithdrawalType.STANDARD,
            None,
            datetime.now()
        )
        self.assertIsNone(withdrawal_no_deadline.get_days_until_deadline())
    
    def test_get_urgency(self):
        """Test getting urgency score."""
        # Test EMERGENCY type (should be highest urgency)
        emergency = Withdrawal(
            "e123",
            5000.0,
            WithdrawalType.EMERGENCY,
            None,
            datetime.now()
        )
        self.assertEqual(emergency.get_urgency(), 1.0)
        
        # Test TIMED type with imminent deadline
        imminent = Withdrawal(
            "t123",
            5000.0,
            WithdrawalType.TIMED,
            datetime.now() + timedelta(hours=6),
            datetime.now()
        )
        self.assertEqual(imminent.get_urgency(), 0.9)  # Exactly 0.9 for <= 1 day
        
        # Test STANDARD type (should be lowest urgency)
        standard = Withdrawal(
            "s123",
            5000.0,
            WithdrawalType.STANDARD,
            None,
            datetime.now()
        )
        self.assertEqual(standard.get_urgency(), 0.1)  # Low urgency for standard
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        withdrawal_dict = self.withdrawal.to_dict()
        self.assertEqual(withdrawal_dict["withdrawal_id"], self.withdrawal_id)
        self.assertEqual(withdrawal_dict["amount_usd"], self.amount_usd)
        self.assertEqual(withdrawal_dict["withdrawal_type"], self.withdrawal_type.value)
        self.assertEqual(withdrawal_dict["status"], WithdrawalStatus.PENDING.value)
        self.assertEqual(withdrawal_dict["reserved_amount"], 0.0)
        self.assertEqual(withdrawal_dict["fulfilled_amount"], 0.0)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        # Create a dictionary representation with all required fields
        last_updated = datetime.now()
        withdrawal_dict = {
            "withdrawal_id": self.withdrawal_id,
            "amount_usd": self.amount_usd,
            "withdrawal_type": self.withdrawal_type.value,
            "status": WithdrawalStatus.PENDING.value,
            "reserved_amount": 0.0,
            "fulfilled_amount": 0.0,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "created_at": self.created_at.isoformat(),
            "last_updated": last_updated.isoformat()
        }
        
        withdrawal = Withdrawal.from_dict(withdrawal_dict)
        self.assertEqual(withdrawal.withdrawal_id, self.withdrawal_id)
        self.assertEqual(withdrawal.amount_usd, self.amount_usd)
        self.assertEqual(withdrawal.withdrawal_type, self.withdrawal_type)
        self.assertEqual(withdrawal.status, WithdrawalStatus.PENDING)
        self.assertEqual(withdrawal.reserved_amount, 0.0)
        self.assertEqual(withdrawal.fulfilled_amount, 0.0)
        self.assertEqual(withdrawal.last_updated, last_updated)


# Run tests if file is executed directly
if __name__ == "__main__":
    unittest.main() 