#!/usr/bin/env python
"""
Enhanced Data Frame Builder
---------------------------
Builds a DataFrame for trading across all buckets with:
  - Smarter caching for expensive operations
  - Better error handling and progress tracking
  - Dynamic PCA components based on explained variance
  - Memory optimizations for large datasets
  - Path consistency with agent script
  - Column verification with agent expectations
"""

import os
import json
import time
import pickle
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import talib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
from tqdm import tqdm  # For progress bars

# Import error handling framework
try:
    from src.ui.error_handler import handle_error, ErrorSeverity
    error_handler_available = True
except ImportError:
    error_handler_available = False
    # Create stub function for error handling if not available
    def handle_error(error, context="", window=None, retry_func=None, additional_context=None):
        print(f"Error in {context}: {str(error)}")
        return {"message": str(error), "handled": False}

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Dynamically determine base directory
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/utils
src_dir = os.path.dirname(current_dir)  # src
base_dir = os.path.dirname(src_dir)  # project root

# Load configuration to align with agent script
CONFIG_PATHS = [
    os.path.join(base_dir, "Scripts", "final_config.json"),
    os.path.join(base_dir, "config", "config.json"),
    os.path.join(base_dir, "config.json")
]

config = {}
for config_path in CONFIG_PATHS:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            print(f"Loaded configuration from {config_path}")
            break

# Directory structure - use relative paths based on project root
CACHE_DIR = os.path.join(base_dir, "Cache")
DATA_DIR = os.path.join(base_dir, "data")  # Main data directory
# Check for alternate data locations if default isn't found
ALT_DATA_PATHS = [
    os.path.join(base_dir, "..", "AI Version 3", "2020-2024_BTCUSDT_DATA"),
    os.path.join(base_dir, "2020-2024_BTCUSDT_DATA")
]

# Find the first valid data directory
if not os.path.exists(DATA_DIR):
    for alt_path in ALT_DATA_PATHS:
        if os.path.exists(alt_path):
            DATA_DIR = alt_path
            print(f"Using alternate data directory: {DATA_DIR}")
            break
    else:
        print(f"Warning: Could not find data directory. Please update DATA_DIR in {__file__}")

# Create output directories
OUTPUTS_DIR = os.path.join(base_dir, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Use a more appropriate location for the output dataframe
OUTPUT_PATH = os.path.join(OUTPUTS_DIR, "final_dataframe.csv")
# Also maintain backward compatibility
LEGACY_OUTPUT_PATH = os.path.join(base_dir, "Scripts", "final_dataframe.csv")

# Create necessary directories
os.makedirs(CACHE_DIR, exist_ok=True)

# Expected columns by agent (used for verification)
EXPECTED_COLUMNS = [
    'close', 'high', 'low', 'volume', 'SMA9', 'SMA21', 'SMA50', 'SMA100', 'SMA200', 'SMA400', 'ParabolicSAR',
    'RSI14', 'RSI28', 'Stoch_K', 'Stoch_D', 'CCI', 'MFI', 'ROC', 'BB_upper20', 'BB_mid20', 'BB_lower20',
    'BB_upper50', 'BB_mid50', 'BB_lower50', 'BB_upper100', 'BB_mid100', 'BB_lower100', 'ATR', 'Keltner_lower',
    'Donch_lower', 'Donch_upper', 'OBV', 'ChaikinMF', 'ForceIndex', 'VolumeOsc', 'MACD_hist_diff',
    'RSI_overbought', 'price_accel', 'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8',
    'pca_9', 'pca_10', 'ae_1', 'ae_2', 'ae_3', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    'day_of_month_sin', 'day_of_month_cos', 'day_of_year_sin', 'day_of_year_cos'
]

def log(message):
    """Print timestamped message to console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def cached_operation(func, cache_file, force_recompute=False):
    """Decorator-like function for caching expensive operations"""
    start_time = time.time()
    cache_path = os.path.join(CACHE_DIR, cache_file)
    
    # Create cache directory if it doesn't exist
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
            log(f"Created cache directory: {CACHE_DIR}")
    except Exception as e:
        log(f"Error creating cache directory: {e}")
        if error_handler_available:
            handle_error(
                e,
                "cached_operation.create_dir",
                additional_context={"directory": CACHE_DIR}
            )
    
    if not force_recompute and os.path.exists(cache_path):
        log(f"Loading cached {cache_file}...")
        try:
            with open(cache_path, 'rb') as f:
                result = pickle.load(f)
            log(f"Loaded cached result in {time.time() - start_time:.2f}s")
            return result
        except Exception as e:
            log(f"Error loading cache: {e}. Recomputing...")
            if error_handler_available:
                handle_error(
                    e,
                    "cached_operation.load_cache",
                    additional_context={"cache_file": cache_file}
                )
            # Continue with recomputation
    
    # Before computation, create a backup of existing cache file
    try:
        if os.path.exists(cache_path):
            backup_path = f"{cache_path}.backup"
            with open(cache_path, 'rb') as src, open(backup_path, 'wb') as dst:
                dst.write(src.read())
            log(f"Created backup of cache file: {backup_path}")
    except Exception as e:
        log(f"Error creating cache backup: {e}")
        if error_handler_available:
            handle_error(
                e,
                "cached_operation.create_backup",
                additional_context={"cache_file": cache_file}
            )
        # Continue without backup
    
    # Compute the result
    try:
        log(f"Computing {cache_file}...")
        result = func()
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            log(f"Cached result to {cache_file} in {time.time() - start_time:.2f}s")
        except Exception as e:
            log(f"Error saving cache: {e}")
            if error_handler_available:
                handle_error(
                    e,
                    "cached_operation.save_cache",
                    additional_context={"cache_file": cache_file}
                )
            # Continue without caching
        
        return result
    except Exception as e:
        log(f"Error during computation of {cache_file}: {e}")
        if error_handler_available:
            handle_error(
                e,
                "cached_operation.computation",
                additional_context={"operation": cache_file}
            )
        
        # Try to restore from backup
        backup_path = f"{cache_path}.backup"
        if os.path.exists(backup_path):
            try:
                log(f"Attempting to restore from backup: {backup_path}")
                with open(backup_path, 'rb') as f:
                    result = pickle.load(f)
                log(f"Successfully restored from backup")
                return result
            except Exception as backup_error:
                log(f"Error restoring from backup: {backup_error}")
                if error_handler_available:
                    handle_error(
                        backup_error,
                        "cached_operation.restore_backup",
                        additional_context={"backup_file": backup_path}
                    )
                # No recovery possible, re-raise original error
                raise e
        else:
            # No backup available, re-raise the original error
            raise e

### Load 5-Minute Data
def load_5m_data(force_recompute=False):
    def _load():
        try:
            # Dynamic file discovery
            if not os.path.exists(DATA_DIR):
                error_msg = f"Data directory not found: {DATA_DIR}"
                log(error_msg)
                if error_handler_available:
                    handle_error(
                        FileNotFoundError(error_msg),
                        "load_5m_data",
                        additional_context={"file_path": DATA_DIR}
                    )
                return pd.DataFrame()

            # Look for all CSV files in the data directory
            log(f"Scanning for data files in {DATA_DIR}...")
            all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) 
                        if f.endswith('.csv') and 'BTCUSDT-5m' in f]
            
            # If no files found with dynamic pattern, try specific year files
            if not all_files:
                specific_years = [f"BTCUSDT-5m-{year}.csv" for year in range(2020, 2025)]
                all_files = [os.path.join(DATA_DIR, f) for f in specific_years 
                            if os.path.isfile(os.path.join(DATA_DIR, f))]
                log(f"Looking for specific year files: found {len(all_files)} files")

            if not all_files:
                error_msg = "No matching data files found in data directory!"
                log(error_msg)
                if error_handler_available:
                    handle_error(
                        FileNotFoundError(error_msg),
                        "load_5m_data.validation",
                        additional_context={"data_dir": DATA_DIR}
                    )
                return pd.DataFrame()
                
            log(f"Found {len(all_files)} data files")
            df_list = []
            
            for file_path in all_files:
                try:
                    df = pd.read_csv(file_path)
                    # Verify it has the expected columns
                    if 'timestamp' in df.columns and 'open' in df.columns and 'close' in df.columns:
                        df_list.append(df)
                        log(f"Loaded {os.path.basename(file_path)} with {len(df)} rows")
                    else:
                        error_msg = f"File {os.path.basename(file_path)} is missing required columns"
                        log(f"Skipping {file_path}: {error_msg}")
                        if error_handler_available:
                            handle_error(
                                ValueError(error_msg),
                                "load_5m_data.file_validation",
                                additional_context={"file_path": file_path, "missing_columns": "timestamp, open, or close"}
                            )
                except Exception as e:
                    log(f"Error loading {file_path}: {e}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "load_5m_data.file_loading",
                            additional_context={"file_path": file_path}
                        )
            
            if not df_list:
                error_msg = "No valid data files loaded!"
                log(error_msg)
                if error_handler_available:
                    handle_error(
                        ValueError(error_msg),
                        "load_5m_data.validation",
                        additional_context={"data_dir": DATA_DIR}
                    )
                return pd.DataFrame()
            
            # Concatenate and process
            df_all = pd.concat(df_list, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
            log(f"Combined dataframe: {df_all.shape} rows")
            
            # Convert timestamp
            try:
                df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
            except Exception as e:
                error_msg = "Failed to convert timestamp column to datetime"
                log(f"{error_msg}: {e}")
                if error_handler_available:
                    handle_error(
                        e,
                        "load_5m_data.timestamp_conversion",
                        additional_context={"error_details": str(e)}
                    )
                # Try to continue with original format
            
            # Add time-related features
            log("Adding time-related features...")
            try:
                df_all['hour'] = df_all['timestamp'].dt.hour
                df_all['day_of_week'] = df_all['timestamp'].dt.dayofweek
                df_all['day_of_month'] = df_all['timestamp'].dt.day
                df_all['day_of_year'] = df_all['timestamp'].dt.dayofyear

                # Sinusoidal encoding for cyclical patterns
                df_all['hour_sin'] = np.sin(2 * np.pi * df_all['hour'] / 24)
                df_all['hour_cos'] = np.cos(2 * np.pi * df_all['hour'] / 24)
                df_all['day_of_week_sin'] = np.sin(2 * np.pi * df_all['day_of_week'] / 7)
                df_all['day_of_week_cos'] = np.cos(2 * np.pi * df_all['day_of_week'] / 7)
                df_all['day_of_month_sin'] = np.sin(2 * np.pi * df_all['day_of_month'] / 31)
                df_all['day_of_month_cos'] = np.cos(2 * np.pi * df_all['day_of_month'] / 31)
                df_all['day_of_year_sin'] = np.sin(2 * np.pi * df_all['day_of_year'] / 366)
                df_all['day_of_year_cos'] = np.cos(2 * np.pi * df_all['day_of_year'] / 366)
            except Exception as e:
                error_msg = "Failed to add time-related features"
                log(f"{error_msg}: {e}")
                if error_handler_available:
                    handle_error(
                        e,
                        "load_5m_data.time_features",
                        additional_context={"error_details": str(e)}
                    )
                # Continue with what we have
            
            return df_all
        except Exception as e:
            log(f"Unexpected error in load_5m_data: {e}")
            if error_handler_available:
                handle_error(
                    e,
                    "load_5m_data.unexpected",
                    additional_context={"error_details": str(e)}
                )
            return pd.DataFrame()
    
    return cached_operation(_load, "raw_data.pkl", force_recompute)

### Compute Technical Indicators
def compute_indicators(df, force_recompute=False):
    def _compute():
        try:
            log("Computing technical indicators...")
            
            # Validate input dataframe
            if df is None or df.empty:
                error_msg = "Cannot compute indicators on empty dataframe"
                log(error_msg)
                if error_handler_available:
                    handle_error(
                        ValueError(error_msg),
                        "compute_indicators.validation",
                        additional_context={"error_msg": error_msg}
                    )
                return pd.DataFrame()
            
            # Check required columns
            required_cols = ['close', 'high', 'low', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                error_msg = f"Missing required columns for indicators: {', '.join(missing_cols)}"
                log(error_msg)
                if error_handler_available:
                    handle_error(
                        ValueError(error_msg),
                        "compute_indicators.missing_columns",
                        additional_context={"missing_columns": ', '.join(missing_cols)}
                    )
                return pd.DataFrame()
            
            result = df.copy()
            
            indicators = [
                # Trend indicators
                lambda df: {'SMA9': talib.SMA(df['close'], 9)},
                lambda df: {'SMA21': talib.SMA(df['close'], 21)},
                lambda df: {'SMA50': talib.SMA(df['close'], 50)},
                lambda df: {'SMA100': talib.SMA(df['close'], 100)},
                lambda df: {'SMA200': talib.SMA(df['close'], 200)},
                lambda df: {'SMA400': talib.SMA(df['close'], 400)},
                lambda df: {'MACD_line': talib.MACD(df['close'], 12, 26, 9)[0],
                        'MACD_signal': talib.MACD(df['close'], 12, 26, 9)[1],
                        'MACD_hist': talib.MACD(df['close'], 12, 26, 9)[2]},
                lambda df: {'MACD_line_long': talib.MACD(df['close'], 24, 52, 18)[0],
                        'MACD_signal_long': talib.MACD(df['close'], 24, 52, 18)[1],
                        'MACD_hist_long': talib.MACD(df['close'], 24, 52, 18)[2]},
                lambda df: {'ParabolicSAR': talib.SAR(df['high'], df['low'])},
                lambda df: {'Ichimoku_baseline': talib.EMA(df['close'], 26)},
                lambda df: {'ADX': talib.ADX(df['high'], df['low'], df['close'], 14)},
                
                # Momentum indicators
                lambda df: {'RSI14': talib.RSI(df['close'], 14)},
                lambda df: {'RSI28': talib.RSI(df['close'], 28)},
                lambda df: {'Stoch_K': talib.STOCH(df['high'], df['low'], df['close'], 14, 3, 0, 3, 0)[0],
                        'Stoch_D': talib.STOCH(df['high'], df['low'], df['close'], 14, 3, 0, 3, 0)[1]},
                lambda df: {'CCI': talib.CCI(df['high'], df['low'], df['close'], 20)},
                lambda df: {'MFI': talib.MFI(df['high'], df['low'], df['close'], df['volume'], 14)},
                lambda df: {'ROC': talib.ROC(df['close'], 12)},
                
                # Volatility indicators
                lambda df: {'BB_upper20': talib.BBANDS(df['close'], 20)[0],
                        'BB_mid20': talib.BBANDS(df['close'], 20)[1],
                        'BB_lower20': talib.BBANDS(df['close'], 20)[2]},
                lambda df: {'BB_upper50': talib.BBANDS(df['close'], 50)[0],
                        'BB_mid50': talib.BBANDS(df['close'], 50)[1],
                        'BB_lower50': talib.BBANDS(df['close'], 50)[2]},
                lambda df: {'BB_upper100': talib.BBANDS(df['close'], 100)[0],
                        'BB_mid100': talib.BBANDS(df['close'], 100)[1],
                        'BB_lower100': talib.BBANDS(df['close'], 100)[2]},
                lambda df: {'ATR': talib.ATR(df['high'], df['low'], df['close'], 14)},
            ]
            
            # Apply indicators with progress bar
            failed_indicators = []
            for indicator_func in tqdm(indicators, desc="Computing indicators"):
                try:
                    indicator_values = indicator_func(result)
                    for name, values in indicator_values.items():
                        result[name] = values
                except Exception as e:
                    indicator_name = str(indicator_func).split(":")[0] if ":" in str(indicator_func) else "unknown"
                    error_msg = f"Error computing indicator {indicator_name}: {str(e)}"
                    log(error_msg)
                    failed_indicators.append(indicator_name)
                    if error_handler_available:
                        handle_error(
                            e,
                            "compute_indicators.indicator_calculation",
                            additional_context={
                                "indicator": indicator_name,
                                "error_msg": str(e)
                            }
                        )
            
            # Log summary of failed indicators
            if failed_indicators:
                log(f"Failed to compute {len(failed_indicators)} indicators: {', '.join(failed_indicators)}")
            
            # Additional indicators that depend on prior calculations
            try:
                log("Computing derived indicators...")
                
                # Check if ATR was successfully calculated
                if 'ATR' not in result.columns:
                    log("Warning: ATR not available for Keltner channel calculation")
                    result['ATR'] = 0  # Default value to continue
                
                # Calculate derived indicators with appropriate error handling
                try:
                    result['Keltner_lower'] = talib.EMA(result['close'], 20) - 2 * result['ATR']
                except Exception as e:
                    log(f"Error computing Keltner_lower: {e}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "compute_indicators.keltner_calculation",
                            additional_context={"error_msg": str(e)}
                        )
                    # Provide fallback value
                    result['Keltner_lower'] = result['close'] * 0.98
                
                try:
                    result['Donch_lower'] = result['low'].rolling(window=20).min()
                    result['Donch_upper'] = result['high'].rolling(window=20).max()
                except Exception as e:
                    log(f"Error computing Donchian channels: {e}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "compute_indicators.donchian_calculation",
                            additional_context={"error_msg": str(e)}
                        )
                    # Provide fallback values
                    result['Donch_lower'] = result['low']
                    result['Donch_upper'] = result['high']
                
                # Volume indicators
                try:
                    result['OBV'] = talib.OBV(result['close'], result['volume'])
                except Exception as e:
                    log(f"Error computing OBV: {e}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "compute_indicators.obv_calculation",
                            additional_context={"error_msg": str(e)}
                        )
                
                try:
                    result['ChaikinMF'] = talib.CMO(result['volume'], 20)
                except Exception as e:
                    log(f"Error computing ChaikinMF: {e}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "compute_indicators.chaikin_calculation",
                            additional_context={"error_msg": str(e)}
                        )
                
                try:
                    result['ForceIndex'] = result['volume'] * result['close'].diff()
                except Exception as e:
                    log(f"Error computing ForceIndex: {e}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "compute_indicators.force_index_calculation",
                            additional_context={"error_msg": str(e)}
                        )
                
                try:
                    result['VolumeOsc'] = talib.EMA(result['volume'], 5) - talib.EMA(result['volume'], 20)
                except Exception as e:
                    log(f"Error computing VolumeOsc: {e}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "compute_indicators.volume_osc_calculation",
                            additional_context={"error_msg": str(e)}
                        )
                
            except Exception as e:
                log(f"Error in derived indicators calculation: {e}")
                if error_handler_available:
                    handle_error(
                        e,
                        "compute_indicators.derived_calculation",
                        additional_context={"error_msg": str(e)}
                    )
            
            # Fill NaN values to prevent issues downstream
            null_counts = result.isnull().sum()
            columns_with_nulls = null_counts[null_counts > 0].index.tolist()
            
            if columns_with_nulls:
                log(f"Warning: Found NaN values in {len(columns_with_nulls)} columns, filling with appropriate methods")
                
                # Forward fill first, then backward fill any remaining NaNs
                result = result.fillna(method='ffill').fillna(method='bfill')
                
                # If still have NaNs, fill with zeros
                remaining_nulls = result.isnull().sum().sum()
                if remaining_nulls > 0:
                    log(f"Warning: {remaining_nulls} NaN values remain after forward/backward fill, replacing with zeros")
                    result = result.fillna(0)
            
            return result
        
        except MemoryError as e:
            error_msg = "Not enough memory to compute indicators"
            log(error_msg)
            if error_handler_available:
                handle_error(
                    e,
                    "compute_indicators.memory_error",
                    additional_context={"error_msg": error_msg}
                )
            # Return original dataframe without indicators
            return df.copy()
            
        except Exception as e:
            error_msg = f"Unexpected error computing indicators: {str(e)}"
            log(error_msg)
            if error_handler_available:
                handle_error(
                    e,
                    "compute_indicators.unexpected_error",
                    additional_context={"error_msg": error_msg}
                )
            # Return original dataframe without indicators
            return df.copy()
    
    return cached_operation(_compute, "indicators.pkl", force_recompute)

### Additional Signals
def compute_additional_signals(df, force_recompute=False):
    def _compute():
        try:
            log("Computing additional signals...")
            
            # Validate input dataframe
            if df is None or df.empty:
                error_msg = "Cannot compute signals on empty dataframe"
                log(error_msg)
                if error_handler_available:
                    handle_error(
                        ValueError(error_msg),
                        "compute_additional_signals.validation",
                        additional_context={"error_msg": error_msg}
                    )
                return pd.DataFrame(index=range(0))
            
            # Create dataframe for signals
            signals = pd.DataFrame(index=df.index)
            
            # Check and compute MACD histogram difference
            try:
                if 'MACD_hist' in df.columns:
                    signals['MACD_hist_diff'] = df['MACD_hist'].diff()
                else:
                    error_msg = "MACD_hist column not available for diff calculation"
                    log(f"Warning: {error_msg}")
                    if error_handler_available:
                        handle_error(
                            ValueError(error_msg),
                            "compute_additional_signals.macd_hist",
                            additional_context={"missing_column": "MACD_hist"}
                        )
                    signals['MACD_hist_diff'] = 0.0
            except Exception as e:
                log(f"Error computing MACD_hist_diff: {e}")
                if error_handler_available:
                    handle_error(
                        e,
                        "compute_additional_signals.macd_diff_calculation",
                        additional_context={"error_details": str(e)}
                    )
                signals['MACD_hist_diff'] = 0.0
            
            # Check and compute RSI overbought
            try:
                if 'RSI14' in df.columns:
                    signals['RSI_overbought'] = (df['RSI14'] > 70).astype(int)
                else:
                    error_msg = "RSI14 column not available for overbought calculation"
                    log(f"Warning: {error_msg}")
                    if error_handler_available:
                        handle_error(
                            ValueError(error_msg),
                            "compute_additional_signals.rsi_overbought",
                            additional_context={"missing_column": "RSI14"}
                        )
                    signals['RSI_overbought'] = 0
            except Exception as e:
                log(f"Error computing RSI_overbought: {e}")
                if error_handler_available:
                    handle_error(
                        e,
                        "compute_additional_signals.rsi_overbought_calculation",
                        additional_context={"error_details": str(e)}
                    )
                signals['RSI_overbought'] = 0
            
            # Check and compute price acceleration
            try:
                if 'close' in df.columns:
                    signals['price_accel'] = df['close'].diff().diff()
                else:
                    error_msg = "close column not available for price acceleration calculation"
                    log(f"Warning: {error_msg}")
                    if error_handler_available:
                        handle_error(
                            ValueError(error_msg),
                            "compute_additional_signals.price_accel",
                            additional_context={"missing_column": "close"}
                        )
                    signals['price_accel'] = 0.0
            except Exception as e:
                log(f"Error computing price_accel: {e}")
                if error_handler_available:
                    handle_error(
                        e,
                        "compute_additional_signals.price_accel_calculation",
                        additional_context={"error_details": str(e)}
                    )
                signals['price_accel'] = 0.0
                
            # Fill NaN values to prevent issues downstream
            null_counts = signals.isnull().sum()
            columns_with_nulls = null_counts[null_counts > 0].index.tolist()
            
            if columns_with_nulls:
                log(f"Warning: Found NaN values in {len(columns_with_nulls)} signal columns, filling with zeros")
                signals = signals.fillna(0)
                
            return signals
            
        except Exception as e:
            error_msg = f"Unexpected error in compute_additional_signals: {str(e)}"
            log(error_msg)
            if error_handler_available:
                handle_error(
                    e,
                    "compute_additional_signals.unexpected",
                    additional_context={"error_details": str(e)}
                )
            # Return empty DataFrame with expected columns
            empty_signals = pd.DataFrame(index=df.index if df is not None else range(0))
            for col in ['MACD_hist_diff', 'RSI_overbought', 'price_accel']:
                empty_signals[col] = 0.0
            return empty_signals
    
    return cached_operation(_compute, "signals.pkl", force_recompute)

### Dynamic PCA with Explained Variance Threshold
def dynamic_pca(data, variance_threshold=0.95, min_components=2, max_components=10):
    """Dynamically select number of PCA components based on explained variance"""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    
    # Determine optimal number of components
    pca = PCA()
    pca.fit(scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find number of components that explain at least variance_threshold
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    n_components = max(min(n_components, max_components), min_components)
    
    # Re-fit with optimal components
    pca = PCA(n_components=n_components)
    return pd.DataFrame(pca.fit_transform(scaled), index=data.index), n_components

### Group-Wise PCA
def groupwise_pca(df, force_recompute=False):
    def _compute():
        try:
            log("Performing group-wise PCA...")
            
            # Validate input dataframe
            if df is None or df.empty:
                error_msg = "Cannot perform PCA on empty dataframe"
                log(error_msg)
                if error_handler_available:
                    handle_error(
                        ValueError(error_msg),
                        "groupwise_pca.validation",
                        additional_context={"error_msg": error_msg}
                    )
                return pd.DataFrame(index=range(0))
            
            # Check for any NaN values that would break PCA
            nan_count = df.isnull().sum().sum()
            if nan_count > 0:
                error_msg = f"Input dataframe contains {nan_count} NaN values"
                log(f"Warning: {error_msg}, filling NaNs before PCA")
                if error_handler_available:
                    handle_error(
                        ValueError(error_msg),
                        "groupwise_pca.nan_values",
                        additional_context={"nan_count": nan_count}
                    )
                # Fill NaN values to prevent errors
                df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            trend_cols = ['SMA9', 'SMA21', 'SMA50', 'SMA100', 'SMA200', 'SMA400', 'MACD_line', 'MACD_line_long', 'ParabolicSAR', 'Ichimoku_baseline', 'ADX']
            momentum_cols = ['RSI14', 'RSI28', 'Stoch_K', 'Stoch_D', 'CCI', 'MFI', 'ROC']
            volatility_cols = ['BB_upper20', 'BB_mid20', 'BB_lower20', 'BB_upper50', 'BB_mid50', 'BB_lower50', 'BB_upper100', 'BB_mid100', 'BB_lower100', 'ATR', 'Keltner_lower', 'Donch_lower', 'Donch_upper']
            volume_cols = ['OBV', 'ChaikinMF', 'ForceIndex', 'VolumeOsc']

            # Group definitions with their columns
            groups = {
                "trend": trend_cols,
                "momentum": momentum_cols,
                "volatility": volatility_cols,
                "volume": volume_cols
            }

            results = []
            component_counts = {}
            
            # Track groups with errors
            groups_with_errors = []
            
            # Process each group
            for group_name, columns in groups.items():
                try:
                    # Skip if missing columns
                    if not all(col in df.columns for col in columns):
                        missing = [col for col in columns if col not in df.columns]
                        error_msg = f"Missing columns for {group_name} PCA: {', '.join(missing)}"
                        log(f"Warning: {error_msg}")
                        if error_handler_available:
                            handle_error(
                                ValueError(error_msg),
                                "groupwise_pca.missing_columns",
                                additional_context={
                                    "group": group_name,
                                    "missing_columns": ', '.join(missing)
                                }
                            )
                        groups_with_errors.append(group_name)
                        continue
                        
                    # Ensure enough data points for PCA
                    if len(df) < len(columns):
                        error_msg = f"Not enough data points for {group_name} PCA"
                        log(f"Warning: {error_msg}")
                        if error_handler_available:
                            handle_error(
                                ValueError(error_msg),
                                "groupwise_pca.insufficient_data",
                                additional_context={
                                    "group": group_name,
                                    "rows": len(df),
                                    "columns": len(columns)
                                }
                            )
                        groups_with_errors.append(group_name)
                        continue
                    
                    # Get data with only rows that have all values
                    group_data = df[columns].dropna()
                    
                    # Check if we have enough data after dropping NaNs
                    if len(group_data) < 10:  # Minimum threshold
                        error_msg = f"Insufficient data after removing NaNs for {group_name} PCA"
                        log(f"Warning: {error_msg}")
                        if error_handler_available:
                            handle_error(
                                ValueError(error_msg),
                                "groupwise_pca.insufficient_clean_data",
                                additional_context={
                                    "group": group_name,
                                    "remaining_rows": len(group_data)
                                }
                            )
                        groups_with_errors.append(group_name)
                        continue
                    
                    # Apply dynamic PCA
                    try:
                        group_pca, n_components = dynamic_pca(
                            group_data, 
                            variance_threshold=0.95,
                            min_components=2,
                            max_components=5
                        )
                        
                        component_counts[group_name] = n_components
                        log(f"PCA for {group_name}: {n_components} components explain >95% variance")
                        
                        # Rename columns for this group
                        group_pca.columns = [f"pca_{group_name}_{i+1}" for i in range(n_components)]
                        results.append(group_pca)
                    except Exception as e:
                        error_msg = f"Error in PCA calculation for {group_name}: {str(e)}"
                        log(f"Error: {error_msg}")
                        if error_handler_available:
                            handle_error(
                                e,
                                "groupwise_pca.pca_calculation",
                                additional_context={
                                    "group": group_name,
                                    "error_details": str(e)
                                }
                            )
                        groups_with_errors.append(group_name)
                    
                except Exception as e:
                    log(f"Error in PCA for {group_name}: {e}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "groupwise_pca.group_processing",
                            additional_context={
                                "group": group_name,
                                "error_details": str(e)
                            }
                        )
                    groups_with_errors.append(group_name)
            
            # Log summary of errors
            if groups_with_errors:
                log(f"PCA failed for {len(groups_with_errors)} groups: {', '.join(groups_with_errors)}")
            
            # Combine all PCA results
            if not results:
                error_msg = "No PCA results generated for any group"
                log(f"Warning: {error_msg}")
                if error_handler_available:
                    handle_error(
                        ValueError(error_msg),
                        "groupwise_pca.no_results",
                        additional_context={"error_msg": error_msg}
                    )
                # Create dummy PCA features
                dummy_pca = pd.DataFrame(
                    np.zeros((len(df), 10)),
                    columns=[f"pca_{i+1}" for i in range(10)]
                )
                return dummy_pca
                
            # Combine and rename for compatibility with the model
            all_pca = pd.concat(results, axis=1)
            all_pca.columns = [f"pca_{i+1}" for i in range(all_pca.shape[1])]
            
            # Ensure we have at least the expected number of PCA components (10)
            current_columns = all_pca.shape[1]
            if current_columns < 10:
                log(f"Warning: Only generated {current_columns} PCA components, padding to 10")
                for i in range(current_columns + 1, 11):
                    all_pca[f"pca_{i}"] = 0.0
            
            return all_pca
            
        except Exception as e:
            error_msg = f"Unexpected error in groupwise_pca: {str(e)}"
            log(f"Error: {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "groupwise_pca.unexpected",
                    additional_context={"error_details": str(e)}
                )
            # Return dummy PCA dataframe
            if df is not None and not df.empty:
                index_len = len(df)
            else:
                index_len = 0
                
            dummy_pca = pd.DataFrame(
                np.zeros((index_len, 10)),
                columns=[f"pca_{i+1}" for i in range(10)]
            )
            return dummy_pca
    
    return cached_operation(_compute, "pca_features.pkl", force_recompute)

### Autoencoder
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2), 
            nn.ReLU(), 
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2), 
            nn.ReLU(), 
            nn.Linear(latent_dim * 2, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def build_autoencoder_features(data, latent_dim=3, epochs=50, batch_size=32, force_recompute=False):
    def _build():
        log(f"Training autoencoder with latent dim={latent_dim}, epochs={epochs}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Using device: {device}")
        
        # Ensure data is a numpy array
        data_array = data.values if hasattr(data, 'values') else data
        
        model = SimpleAutoencoder(data_array.shape[1], latent_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Create DataLoader with validation split
        data_tensor = torch.tensor(data_array, dtype=torch.float32)
        dataset = TensorDataset(data_tensor)
        
        # Split into train/val
        val_size = min(1000, int(0.1 * len(dataset)))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        # Training loop with progress bar and early stopping
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                x = batch[0].to(device)
                optimizer.zero_grad()
                recon, _ = model(x)
                loss = criterion(recon, x)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({'train_loss': loss.item()})
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    recon, _ = model(x)
                    loss = criterion(recon, x)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            train_loss /= len(train_loader)
            
            log(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss - 0.0001:  # Improvement threshold
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Generate latent features
        model.eval()
        with torch.no_grad():
            latent = model(torch.tensor(data_array, dtype=torch.float32).to(device))[1].cpu().numpy()
        
        log(f"Generated autoencoder features with shape {latent.shape}")
        df_latent = pd.DataFrame(
            latent, 
            index=np.arange(data_array.shape[0]), 
            columns=[f"ae_{i+1}" for i in range(latent.shape[1])]
        )
        
        return df_latent
    
    return cached_operation(_build, "autoencoder_features.pkl", force_recompute)

### Verify columns match agent expectations
def verify_columns(df, required_columns=EXPECTED_COLUMNS):
    """Check if DataFrame has all required columns and add missing ones with zeros"""
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        log(f"Warning: Adding missing columns required by agent: {missing}")
        for col in missing:
            df[col] = 0.0
    
    # Reorder columns to match expected order
    available_cols = [col for col in required_columns if col in df.columns]
    extra_cols = [col for col in df.columns if col not in required_columns]
    
    return df[available_cols + extra_cols]

### Main Function
def build_dataframe(force_recompute=False):
    try:
        log("Building trading DataFrame...")
        start_time = time.time()
        
        # Check if output exists and we don't need to recompute
        if not force_recompute and os.path.exists(OUTPUT_PATH):
            log(f"Loading existing DataFrame from {OUTPUT_PATH}")
            try:
                return pd.read_csv(OUTPUT_PATH)
            except Exception as e:
                log(f"Error loading existing dataframe: {e}. Will recompute.")
                if error_handler_available:
                    handle_error(
                        e,
                        "build_dataframe.load_existing",
                        additional_context={"file_path": OUTPUT_PATH}
                    )
                # Continue with recomputation
        
        # Step 1: Load raw data
        log("Step 1/5: Loading raw data...")
        df_raw = load_5m_data(force_recompute)
        if df_raw.empty:
            error_msg = "Empty DataFrame - no data loaded."
            log(f"Error: {error_msg}")
            if error_handler_available:
                handle_error(
                    ValueError(error_msg),
                    "build_dataframe.empty_data",
                    additional_context={"error_msg": error_msg}
                )
            return None
        
        # Create checkpoint for raw data
        try:
            checkpoint_path = os.path.join(CACHE_DIR, "df_raw_checkpoint.pkl")
            df_raw.to_pickle(checkpoint_path)
            log(f"Created checkpoint after data loading: {checkpoint_path}")
        except Exception as e:
            log(f"Warning: Failed to create raw data checkpoint: {e}")
            # Continue without checkpoint
        
        # Step 2: Compute technical indicators
        log("Step 2/5: Computing technical indicators...")
        try:
            df_ind = compute_indicators(df_raw, force_recompute)
            
            # Basic validation of indicators dataframe
            if df_ind.empty:
                error_msg = "Failed to compute indicators - empty dataframe returned"
                log(f"Error: {error_msg}")
                if error_handler_available:
                    handle_error(
                        ValueError(error_msg),
                        "build_dataframe.indicators_empty",
                        additional_context={"error_msg": error_msg}
                    )
                # Try to continue with raw data
                df_ind = df_raw.copy()
            
            # Drop NaNs and reset index
            df_ind = df_ind.dropna().reset_index(drop=True)
            log(f"DataFrame after indicators: {df_ind.shape}")
            
            # Create checkpoint after indicators
            try:
                checkpoint_path = os.path.join(CACHE_DIR, "df_ind_checkpoint.pkl")
                df_ind.to_pickle(checkpoint_path)
                log(f"Created checkpoint after indicators: {checkpoint_path}")
            except Exception as e:
                log(f"Warning: Failed to create indicators checkpoint: {e}")
                # Continue without checkpoint
                
        except Exception as e:
            error_msg = f"Failed to compute indicators: {str(e)}"
            log(f"Error: {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "build_dataframe.compute_indicators",
                    additional_context={"error_msg": error_msg}
                )
            # Continue with raw data
            df_ind = df_raw.copy()
            log("Falling back to raw data without indicators")
        
        # Step 3: Additional signals
        log("Step 3/5: Computing additional signals...")
        try:
            # Verify required columns exist
            required_for_signals = ['MACD_hist', 'RSI14', 'close']
            missing_cols = [col for col in required_for_signals if col not in df_ind.columns]
            
            if missing_cols:
                error_msg = f"Missing columns required for signals: {', '.join(missing_cols)}"
                log(f"Warning: {error_msg}")
                if error_handler_available:
                    handle_error(
                        ValueError(error_msg),
                        "build_dataframe.signals_missing_columns",
                        additional_context={"missing_columns": ', '.join(missing_cols)}
                    )
                
                # Add missing columns with zeros to allow signals calculation
                for col in missing_cols:
                    df_ind[col] = 0.0
                log("Added missing columns with zeros to continue processing")
                
            # Compute signals
            signals = compute_additional_signals(df_ind, force_recompute)
            log(f"Generated {len(signals.columns)} signal features")
            
        except Exception as e:
            error_msg = f"Failed to compute additional signals: {str(e)}"
            log(f"Error: {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "build_dataframe.compute_signals",
                    additional_context={"error_msg": error_msg}
                )
            # Create empty signals dataframe to continue
            signals = pd.DataFrame(index=df_ind.index)
            for col in ['MACD_hist_diff', 'RSI_overbought', 'price_accel']:
                signals[col] = 0.0
            log("Created fallback signal features with zeros")
        
        # Step 4: Group-wise PCA
        log("Step 4/5: Performing group-wise PCA...")
        try:
            pca_features = groupwise_pca(df_ind, force_recompute)
            log(f"Generated {len(pca_features.columns)} PCA features")
        except Exception as e:
            error_msg = f"Failed to compute PCA features: {str(e)}"
            log(f"Error: {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "build_dataframe.compute_pca",
                    additional_context={"error_msg": error_msg}
                )
            # Create dummy PCA features to continue
            pca_features = pd.DataFrame(
                np.zeros((len(df_ind), 10)),
                columns=[f'pca_{i+1}' for i in range(10)]
            )
            log("Created fallback PCA features with zeros")
        
        # Step 5: Autoencoder features
        log("Step 5/5: Training autoencoder...")
        try:
            auto_cols = ['SMA9', 'SMA21', 'SMA50', 'SMA100', 'SMA200', 'SMA400', 'RSI14', 'RSI28', 'MACD_line', 'MACD_line_long']
            # Check if all columns exist
            missing_cols = [col for col in auto_cols if col not in df_ind.columns]
            if missing_cols:
                error_msg = f"Missing columns for autoencoder: {', '.join(missing_cols)}"
                log(f"Warning: {error_msg}")
                if error_handler_available:
                    handle_error(
                        ValueError(error_msg),
                        "build_dataframe.autoencoder_missing_columns",
                        additional_context={"missing_columns": ', '.join(missing_cols)}
                    )
                auto_cols = [col for col in auto_cols if col in df_ind.columns]
            
            if auto_cols:
                try:
                    ae_features = build_autoencoder_features(df_ind[auto_cols].values, force_recompute=force_recompute)
                    log(f"Generated {len(ae_features.columns)} autoencoder features")
                except Exception as e:
                    error_msg = f"Autoencoder training failed: {str(e)}"
                    log(f"Error: {error_msg}")
                    if error_handler_available:
                        handle_error(
                            e,
                            "build_dataframe.autoencoder_training",
                            additional_context={"error_msg": error_msg}
                        )
                    # Create dummy autoencoder features
                    ae_features = pd.DataFrame(
                        np.zeros((len(df_ind), 3)),
                        columns=['ae_1', 'ae_2', 'ae_3']
                    )
                    log("Created fallback autoencoder features with zeros")
            else:
                error_msg = "No columns available for autoencoder"
                log(f"Error: {error_msg}")
                if error_handler_available:
                    handle_error(
                        ValueError(error_msg),
                        "build_dataframe.autoencoder_no_columns",
                        additional_context={"error_msg": error_msg}
                    )
                # Create dummy features
                ae_features = pd.DataFrame(
                    np.zeros((len(df_ind), 3)),
                    columns=['ae_1', 'ae_2', 'ae_3']
                )
                log("Created dummy autoencoder features with zeros")
        except Exception as e:
            error_msg = f"Unexpected error in autoencoder processing: {str(e)}"
            log(f"Error: {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "build_dataframe.autoencoder_unexpected",
                    additional_context={"error_msg": error_msg}
                )
            # Create dummy features
            ae_features = pd.DataFrame(
                np.zeros((len(df_ind), 3)),
                columns=['ae_1', 'ae_2', 'ae_3']
            )
            log("Created fallback autoencoder features with zeros")
        
        # Combine all features, including time-related cyclical encodings
        log("Combining all features...")
        try:
            # Check if time features are available
            time_cols = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
                        'day_of_month_sin', 'day_of_month_cos', 'day_of_year_sin', 'day_of_year_cos']
            
            if all(col in df_raw.columns for col in time_cols) and len(df_raw) >= len(df_ind):
                time_features = df_raw.loc[df_ind.index, time_cols]
            else:
                error_msg = "Time features not available in raw data"
                log(f"Warning: {error_msg}")
                if error_handler_available:
                    handle_error(
                        ValueError(error_msg),
                        "build_dataframe.missing_time_features",
                        additional_context={"error_msg": error_msg}
                    )
                # Create dummy time features
                time_features = pd.DataFrame(
                    np.zeros((len(df_ind), len(time_cols))),
                    columns=time_cols
                )
                log("Created fallback time features with zeros")
            
            # Concatenate all features
            final_df = pd.concat([
                df_ind, 
                pca_features, 
                ae_features, 
                signals, 
                time_features
            ], axis=1)
        except Exception as e:
            error_msg = f"Failed to combine features: {str(e)}"
            log(f"Error: {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "build_dataframe.combine_features",
                    additional_context={"error_msg": error_msg}
                )
            # Use indicators dataframe as fallback
            final_df = df_ind.copy()
            log("Using indicators dataframe as fallback")
        
        # Verify and fix columns required by agent
        try:
            final_df = verify_columns(final_df)
        except Exception as e:
            error_msg = f"Failed to verify columns: {str(e)}"
            log(f"Error: {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "build_dataframe.verify_columns",
                    additional_context={"error_msg": error_msg}
                )
            # Continue with what we have
        
        # Save output to primary location
        try:
            log(f"Saving final DataFrame with shape {final_df.shape} to {OUTPUT_PATH}")
            final_df.to_csv(OUTPUT_PATH, index=False)
            
            # Also save to legacy location for backward compatibility
            legacy_dir = os.path.dirname(LEGACY_OUTPUT_PATH)
            if not os.path.exists(legacy_dir):
                os.makedirs(legacy_dir, exist_ok=True)
            log(f"Also saving to legacy location: {LEGACY_OUTPUT_PATH}")
            final_df.to_csv(LEGACY_OUTPUT_PATH, index=False)
        except Exception as e:
            error_msg = f"Failed to save dataframe: {str(e)}"
            log(f"Error: {error_msg}")
            if error_handler_available:
                handle_error(
                    e,
                    "build_dataframe.save_output",
                    additional_context={"file_path": OUTPUT_PATH}
                )
            # Continue without saving
        
        elapsed = time.time() - start_time
        log(f"DataFrame building completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        
        return final_df
        
    except MemoryError as e:
        error_msg = "Not enough memory to build dataframe"
        log(f"Critical Error: {error_msg}")
        if error_handler_available:
            handle_error(
                e,
                "build_dataframe.memory_error",
                additional_context={"error_msg": error_msg},
            )
        return None
        
    except Exception as e:
        error_msg = f"Unexpected error in build_dataframe: {str(e)}"
        log(f"Critical Error: {error_msg}")
        if error_handler_available:
            handle_error(
                e,
                "build_dataframe.unexpected_error",
                additional_context={"error_msg": error_msg}
            )
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Build trading DataFrame')
    parser.add_argument('--force', action='store_true', help='Force recomputation of all data')
    args = parser.parse_args()
    
    df = build_dataframe(force_recompute=args.force)
    if df is not None:
        print(f"Successfully built DataFrame with {len(df)} rows and {len(df.columns)} columns")
    else:
        print("Failed to build DataFrame")
        sys.exit(1)
