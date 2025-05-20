#!/usr/bin/env python
"""
Validation Utilities

This module provides functions for validating inputs, parameters, and data
structures used throughout the BTC-AI application.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate that a DataFrame has the required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        
    Returns:
        True if valid, False otherwise
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if required_columns is not None:
        for col in required_columns:
            if col not in df.columns:
                return False
    
    return True

def validate_model_config(config: Dict[str, Any]) -> bool:
    """
    Validate model configuration parameters.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    if config is None or not isinstance(config, dict):
        return False
    
    # Check for required keys
    required_keys = ['learning_rate', 'batch_size', 'epochs']
    for key in required_keys:
        if key not in config:
            return False
    
    # Check value types and ranges
    if not isinstance(config['learning_rate'], (int, float)) or config['learning_rate'] <= 0:
        return False
    
    if not isinstance(config['batch_size'], int) or config['batch_size'] <= 0:
        return False
    
    if not isinstance(config['epochs'], int) or config['epochs'] <= 0:
        return False
    
    return True

def validate_backtest_params(params: Dict[str, Any]) -> bool:
    """
    Validate backtesting parameters.
    
    Args:
        params: Dictionary of backtesting parameters
        
    Returns:
        True if valid, False otherwise
    """
    if params is None or not isinstance(params, dict):
        return False
    
    # Validate required parameters
    if 'start_date' not in params or 'end_date' not in params:
        return False
    
    # Validate optional parameters
    if 'slippage' in params and (not isinstance(params['slippage'], (int, float)) or params['slippage'] < 0):
        return False
    
    if 'commission' in params and (not isinstance(params['commission'], (int, float)) or params['commission'] < 0):
        return False
    
    return True

def validate_backtesting_results(results: Dict[str, Any]) -> bool:
    """
    Validate backtesting results structure.
    
    Args:
        results: Dictionary of backtesting results
        
    Returns:
        True if valid, False otherwise
    """
    if results is None or not isinstance(results, dict):
        return False
    
    # Check for required keys
    required_keys = ['metrics', 'equity_curve']
    for key in required_keys:
        if key not in results:
            return False
    
    # Check metrics dictionary
    if not isinstance(results['metrics'], dict):
        return False
    
    # Check equity curve
    if not isinstance(results['equity_curve'], (list, np.ndarray)) or len(results['equity_curve']) == 0:
        return False
    
    return True 