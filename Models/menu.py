#!/usr/bin/env python
"""
Menu system for probabilistic trading models.
"""

import PySimpleGUI as sg
import json
import os
import subprocess
import sys
import threading
import time
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Any

# Configure theme
sg.theme('DarkBlue')

# Get the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
# Add project root to system path to ensure imports work
sys.path.insert(0, project_root)

# File paths based on project root
CONFIG_FILE = os.path.join(current_dir, "config.json")
DEFAULT_DATA_DIR = os.path.join(project_root, "data")

# Default configuration
default_config = {
    # Data Configuration
    "DATA_PATH": os.path.join(DEFAULT_DATA_DIR, "btc_5min.csv"),
    "PRICE_COLUMN": "close",
    "TIMESTAMP_COLUMN": "timestamp",
    
    # Model Architecture
    "MODEL_TYPE": "cnn_lstm",  # Options: "lstm", "cnn_lstm"
    "HIDDEN_SIZE": 128,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.2,
    "KERNEL_SIZE": 3,
    "NUM_FILTERS": 64,
    
    # Training Parameters
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.001,
    "NUM_EPOCHS": 100,
    "SEQ_LEN": 100,
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,
    "SAVE_FREQ": 5,
    
    # Prediction Configuration
    "HORIZON_MINUTES": [5, 15, 30, 60, 240],  # 5min, 15min, 30min, 1hr, 4hr
    "TARGET_TYPE": "returns",  # "price", "returns", "log_returns"
    "CONFIDENCE_THRESHOLDS": [0.3, 0.5, 0.7, 0.9],
    
    # Backtesting Parameters
    "INITIAL_CAPITAL": 100000.0,
    "POSITION_SIZING": "confidence",  # "fixed", "confidence", "kelly"
    "MAX_POSITION_SIZE": 0.2,
    "TRADING_FEE_PCT": 0.001,
    "SLIPPAGE_PCT": 0.001,
    "STOP_LOSS_PCT": 0.05,
    "TAKE_PROFIT_PCT": 0.1,
    
    # Visualization
    "PLOT_STYLE": "seaborn-v0_8-whitegrid",
    "SAVE_PLOTS": True,
    "CONFIDENCE_LEVELS": [0.5, 0.8, 0.9, 0.95]
}

# Load existing config or use default
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
    # Fill in any missing parameters with defaults
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
else:
    config = default_config.copy()
    # Save default config
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, indent=4, sort_keys=True, fp=f)

# Helper functions
def find_checkpoints(prefix="model_"):
    """Find all model checkpoints in the 'checkpoints' directory"""
    checkpoints_dir = os.path.join(current_dir, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)
        return []
    
    pattern = os.path.join(checkpoints_dir, f"{prefix}*.pt")
    checkpoints = glob.glob(pattern)
    return sorted(checkpoints, key=os.path.getmtime, reverse=True)

def check_training_status():
    """Check if training is currently running"""
    status_file = os.path.join(current_dir, "training_status.json")
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status = json.load(f)
            if status.get("status") == "running":
                return True
    return False

def start_live_log_window(log_file):
    """Open a window displaying live log updates"""
    window = sg.Window('Training Log', 
                      [[sg.Multiline(size=(90, 30), key='-LOG-', autoscroll=True, reroute_stdout=False, disabled=True)]], 
                      resizable=True, finalize=True)
    
    last_pos = 0
    try:
        while True:
            event, values = window.read(timeout=1000)
            if event == sg.WIN_CLOSED:
                break
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    f.seek(last_pos)
                    new_text = f.read()
                    if new_text:
                        window['-LOG-'].update(new_text, append=True)
                        last_pos = f.tell()
    finally:
        window.close()

def load_performance_metrics(model_path):
    """Load performance metrics for a model"""
    metrics_path = model_path.replace('.pt', '_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

# Main application layout
def create_main_layout():
    model_list = find_checkpoints()
    model_names = [os.path.basename(m) for m in model_list] if model_list else ['No models found']
    
    data_tab = [
        [sg.Text('Data Configuration', font=('Helvetica', 12))],
        [sg.Text('Data File:', size=(15, 1)), 
         sg.Input(config['DATA_PATH'], key='DATA_PATH', size=(50, 1)), 
         sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))],
        [sg.Text('Price Column:', size=(15, 1)), 
         sg.Input(config['PRICE_COLUMN'], key='PRICE_COLUMN', size=(15, 1))],
        [sg.Text('Timestamp Column:', size=(15, 1)), 
         sg.Input(config['TIMESTAMP_COLUMN'], key='TIMESTAMP_COLUMN', size=(15, 1))],
        [sg.Text('Sequence Length:', size=(15, 1)), 
         sg.Spin([i for i in range(10, 501)], initial_value=config['SEQ_LEN'], key='SEQ_LEN', size=(10, 1))],
        [sg.Text('Test Size:', size=(15, 1)), 
         sg.Slider(range=(0.1, 0.5), default_value=config['TEST_SIZE'], resolution=0.05, orientation='h', key='TEST_SIZE', size=(20, 10))],
        [sg.Text('Random State:', size=(15, 1)), 
         sg.Spin([i for i in range(0, 100)], initial_value=config['RANDOM_STATE'], key='RANDOM_STATE', size=(10, 1))],
    ]
    
    model_tab = [
        [sg.Text('Model Architecture', font=('Helvetica', 12))],
        [sg.Text('Model Type:', size=(15, 1)), 
         sg.Combo(['lstm', 'cnn_lstm'], default_value=config['MODEL_TYPE'], key='MODEL_TYPE', size=(10, 1))],
        [sg.Text('Hidden Size:', size=(15, 1)), 
         sg.Spin([2**i for i in range(4, 11)], initial_value=config['HIDDEN_SIZE'], key='HIDDEN_SIZE', size=(10, 1))],
        [sg.Text('Num Layers:', size=(15, 1)), 
         sg.Spin([i for i in range(1, 6)], initial_value=config['NUM_LAYERS'], key='NUM_LAYERS', size=(10, 1))],
        [sg.Text('Dropout:', size=(15, 1)), 
         sg.Slider(range=(0, 0.5), default_value=config['DROPOUT'], resolution=0.05, orientation='h', key='DROPOUT', size=(20, 10))],
        [sg.Text('CNN Parameters:', font=('Helvetica', 10))],
        [sg.Text('Kernel Size:', size=(15, 1)), 
         sg.Spin([i for i in range(2, 8)], initial_value=config['KERNEL_SIZE'], key='KERNEL_SIZE', size=(10, 1))],
        [sg.Text('Num Filters:', size=(15, 1)), 
         sg.Spin([2**i for i in range(4, 9)], initial_value=config['NUM_FILTERS'], key='NUM_FILTERS', size=(10, 1))],
    ]
    
    training_tab = [
        [sg.Text('Training Parameters', font=('Helvetica', 12))],
        [sg.Text('Batch Size:', size=(15, 1)), 
         sg.Spin([2**i for i in range(3, 9)], initial_value=config['BATCH_SIZE'], key='BATCH_SIZE', size=(10, 1))],
        [sg.Text('Learning Rate:', size=(15, 1)), 
         sg.Combo(['0.0001', '0.0003', '0.001', '0.003', '0.01'], default_value=str(config['LEARNING_RATE']), key='LEARNING_RATE', size=(10, 1))],
        [sg.Text('Num Epochs:', size=(15, 1)), 
         sg.Spin([i for i in range(10, 1001, 10)], initial_value=config['NUM_EPOCHS'], key='NUM_EPOCHS', size=(10, 1))],
        [sg.Text('Save Frequency:', size=(15, 1)), 
         sg.Spin([i for i in range(1, 21)], initial_value=config['SAVE_FREQ'], key='SAVE_FREQ', size=(10, 1))],
        [sg.Text('')],
        [sg.Text('Prediction Horizons (minutes):', size=(25, 1))],
        [sg.Input(', '.join(map(str, config['HORIZON_MINUTES'])), key='HORIZON_MINUTES', size=(30, 1))],
        [sg.Text('Target Type:', size=(15, 1)), 
         sg.Combo(['price', 'returns', 'log_returns'], default_value=config['TARGET_TYPE'], key='TARGET_TYPE', size=(15, 1))],
        [sg.Button('Start Training', key='START_TRAINING', size=(15, 1)), 
         sg.Button('View Training Log', key='VIEW_LOG', size=(15, 1)), 
         sg.Button('Stop Training', key='STOP_TRAINING', size=(15, 1))]
    ]
    
    backtest_tab = [
        [sg.Text('Backtesting Configuration', font=('Helvetica', 12))],
        [sg.Text('Model Selection:', size=(15, 1))],
        [sg.Listbox(model_names, size=(60, 5), key='MODEL_SELECTION')],
        [sg.Text('Initial Capital:', size=(15, 1)), 
         sg.Input(config['INITIAL_CAPITAL'], key='INITIAL_CAPITAL', size=(15, 1))],
        [sg.Text('Position Sizing:', size=(15, 1)), 
         sg.Combo(['fixed', 'confidence', 'kelly'], default_value=config['POSITION_SIZING'], key='POSITION_SIZING', size=(15, 1))],
        [sg.Text('Max Position Size:', size=(15, 1)), 
         sg.Slider(range=(0.05, 0.5), default_value=config['MAX_POSITION_SIZE'], resolution=0.05, orientation='h', key='MAX_POSITION_SIZE', size=(20, 10))],
        [sg.Text('Trading Fee (%):', size=(15, 1)), 
         sg.Input(config['TRADING_FEE_PCT'], key='TRADING_FEE_PCT', size=(10, 1))],
        [sg.Text('Slippage (%):', size=(15, 1)), 
         sg.Input(config['SLIPPAGE_PCT'], key='SLIPPAGE_PCT', size=(10, 1))],
        [sg.Text('Stop Loss (%):', size=(15, 1)), 
         sg.Input(config['STOP_LOSS_PCT'], key='STOP_LOSS_PCT', size=(10, 1))],
        [sg.Text('Take Profit (%):', size=(15, 1)), 
         sg.Input(config['TAKE_PROFIT_PCT'], key='TAKE_PROFIT_PCT', size=(10, 1))],
        [sg.Button('Run Backtest', key='RUN_BACKTEST', size=(15, 1)),
         sg.Button('View Results', key='VIEW_BACKTEST', size=(15, 1))]
    ]
    
    visualize_tab = [
        [sg.Text('Visualization Options', font=('Helvetica', 12))],
        [sg.Text('Model Selection:', size=(15, 1))],
        [sg.Listbox(model_names, size=(60, 5), key='VIZ_MODEL_SELECTION')],
        [sg.Text('Plot Style:', size=(15, 1)), 
         sg.Combo(['seaborn-v0_8-whitegrid', 'seaborn-v0_8-darkgrid', 'ggplot', 'dark_background', 'bmh', 'fivethirtyeight'], 
                  default_value=config['PLOT_STYLE'], key='PLOT_STYLE', size=(20, 1))],
        [sg.Text('Confidence Levels:')],
        [sg.Checkbox('50%', default=0.5 in config['CONFIDENCE_LEVELS'], key='CONF_50')],
        [sg.Checkbox('80%', default=0.8 in config['CONFIDENCE_LEVELS'], key='CONF_80')],
        [sg.Checkbox('90%', default=0.9 in config['CONFIDENCE_LEVELS'], key='CONF_90')],
        [sg.Checkbox('95%', default=0.95 in config['CONFIDENCE_LEVELS'], key='CONF_95')],
        [sg.Text('Visualization Type:')],
        [sg.Button('Price Predictions', key='VIZ_PRICE', size=(20, 1))],
        [sg.Button('Confidence Metrics', key='VIZ_CONF', size=(20, 1))],
        [sg.Button('Calibration Curve', key='VIZ_CALIB', size=(20, 1))],
        [sg.Button('Prediction Samples', key='VIZ_SAMPLES', size=(20, 1))]
    ]
    
    about_tab = [
        [sg.Text('Probabilistic Trading Model UI', font=('Helvetica', 16))],
        [sg.Text('This interface provides tools for training, testing, and deploying probabilistic trading models.')],
        [sg.Text('')],
        [sg.Text('Key Features:', font=('Helvetica', 12))],
        [sg.Text('• Probabilistic prediction with uncertainty estimation')],
        [sg.Text('• Multiple prediction horizons')],
        [sg.Text('• Confidence-based position sizing')],
        [sg.Text('• Comprehensive backtesting')],
        [sg.Text('• Interactive visualization tools')],
        [sg.Text('')],
        [sg.Text('Version: 1.0.0')]
    ]
    
    layout = [
        [sg.TabGroup([[
            sg.Tab('Data', data_tab),
            sg.Tab('Model', model_tab),
            sg.Tab('Training', training_tab),
            sg.Tab('Backtest', backtest_tab),
            sg.Tab('Visualize', visualize_tab),
            sg.Tab('About', about_tab)
        ]])],
        [sg.Button('Save Config', key='SAVE_CONFIG'), sg.Button('Exit')]
    ]
    
    return layout

def main():
    """Main application entry point"""
    layout = create_main_layout()
    window = sg.Window('Probabilistic Trading Model', layout, resizable=True, finalize=True)
    
    training_thread = None
    backtest_thread = None
    visualization_thread = None
    
    while True:
        event, values = window.read(timeout=1000)  # 1-second timeout for updates
        
        # Check if training is running
        is_training = check_training_status()
        window['START_TRAINING'].update(disabled=is_training)
        window['STOP_TRAINING'].update(disabled=not is_training)
        
        # Update model list periodically
        if event == sg.TIMEOUT_EVENT:
            model_list = find_checkpoints()
            if model_list:
                model_names = [os.path.basename(m) for m in model_list]
                window['MODEL_SELECTION'].update(values=model_names)
                window['VIZ_MODEL_SELECTION'].update(values=model_names)
        
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        
        if event == 'SAVE_CONFIG':
            # Update config from UI values
            for key in default_config.keys():
                if key in values:
                    if key == 'HORIZON_MINUTES':
                        try:
                            config[key] = [int(x.strip()) for x in values[key].split(',')]
                        except ValueError:
                            sg.popup_error('Invalid format for horizon minutes. Please use comma-separated integers.')
                            continue
                    elif key == 'LEARNING_RATE':
                        config[key] = float(values[key])
                    else:
                        config[key] = values[key]
            
            # Update confidence levels
            confidence_levels = []
            if values['CONF_50']: confidence_levels.append(0.5)
            if values['CONF_80']: confidence_levels.append(0.8)
            if values['CONF_90']: confidence_levels.append(0.9)
            if values['CONF_95']: confidence_levels.append(0.95)
            if confidence_levels:
                config['CONFIDENCE_LEVELS'] = confidence_levels
            
            # Save to file
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, indent=4, sort_keys=True, fp=f)
            
            sg.popup('Configuration saved successfully.')
        
        elif event == 'START_TRAINING':
            # Update config before training
            window.perform_long_operation(lambda: window.write_event_value('SAVE_CONFIG', None), 'CONFIG_SAVED')
            
            # Construct command
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "main.py"),
                "--data_path", values['DATA_PATH'],
                "--price_column", values['PRICE_COLUMN'],
                "--seq_len", str(values['SEQ_LEN']),
                "--test_size", str(values['TEST_SIZE']),
                "--model_type", values['MODEL_TYPE'],
                "--hidden_size", str(values['HIDDEN_SIZE']),
                "--num_layers", str(values['NUM_LAYERS']),
                "--dropout", str(values['DROPOUT']),
                "--batch_size", str(values['BATCH_SIZE']),
                "--learning_rate", str(values['LEARNING_RATE']),
                "--num_epochs", str(values['NUM_EPOCHS']),
                "--save_freq", str(values['SAVE_FREQ']),
                "--device", "cuda" if torch.cuda.is_available() else "cpu",
                "--random_state", str(values['RANDOM_STATE'])
            ]
            
            if values['MODEL_TYPE'] == 'cnn_lstm':
                cmd.extend([
                    "--kernel_size", str(values['KERNEL_SIZE']),
                    "--num_filters", str(values['NUM_FILTERS'])
                ])
            
            # Run training in a separate thread
            def run_training():
                status_file = os.path.join(SCRIPT_DIR, "training_status.json")
                with open(status_file, "w") as f:
                    json.dump({"status": "running", "start_time": datetime.now().isoformat()}, f)
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                with open(os.path.join(SCRIPT_DIR, "training.log"), "w") as log_file:
                    while process.poll() is None:
                        output = process.stdout.readline()
                        if output:
                            log_file.write(output)
                            log_file.flush()
                
                with open(status_file, "w") as f:
                    json.dump({"status": "completed", "end_time": datetime.now().isoformat()}, f)
            
            if training_thread is None or not training_thread.is_alive():
                training_thread = threading.Thread(target=run_training, daemon=True)
                training_thread.start()
                sg.popup('Training started. You can view progress in the log window.')
        
        elif event == 'VIEW_LOG':
            log_file = os.path.join(SCRIPT_DIR, "training.log")
            threading.Thread(target=start_live_log_window, args=(log_file,), daemon=True).start()
        
        elif event == 'STOP_TRAINING':
            status_file = os.path.join(SCRIPT_DIR, "training_status.json")
            with open(status_file, "w") as f:
                json.dump({"status": "stopping", "stop_time": datetime.now().isoformat()}, f)
            sg.popup('Training will stop after the current epoch completes.')
        
        elif event == 'RUN_BACKTEST':
            if not values['MODEL_SELECTION']:
                sg.popup_error('Please select a model for backtesting.')
                continue
            
            model_path = os.path.join(SCRIPT_DIR, "checkpoints", values['MODEL_SELECTION'][0])
            if not os.path.exists(model_path):
                sg.popup_error('Selected model file not found.')
                continue
            
            from backtest import Backtester
            from trading_model import TradingModel
            import prepare_data
            
            # Run backtest in a separate thread
            def run_backtest():
                # Load data
                data = pd.read_csv(values['DATA_PATH'])
                
                # Initialize model
                model = TradingModel.load_from_checkpoint(model_path)
                
                # Initialize backtester
                backtester = Backtester(
                    initial_capital=float(values['INITIAL_CAPITAL']),
                    position_sizing=values['POSITION_SIZING'],
                    max_position_size=float(values['MAX_POSITION_SIZE']),
                    stop_loss_pct=float(values['STOP_LOSS_PCT']),
                    take_profit_pct=float(values['TAKE_PROFIT_PCT']),
                    trading_fee_pct=float(values['TRADING_FEE_PCT']),
                    slippage_pct=float(values['SLIPPAGE_PCT'])
                )
                
                # Run backtest
                results = backtester.run_backtest(
                    data=data,
                    model=model,
                    price_column=values['PRICE_COLUMN'],
                    timestamp_column=values['TIMESTAMP_COLUMN']
                )
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = os.path.join(SCRIPT_DIR, f"backtest_results_{timestamp}.json")
                with open(results_file, "w") as f:
                    # Convert numpy values to Python types
                    serializable_results = {}
                    for k, v in results.items():
                        if isinstance(v, (np.ndarray, np.number)):
                            serializable_results[k] = v.item() if isinstance(v, np.number) else v.tolist()
                        else:
                            serializable_results[k] = v
                    
                    json.dump(serializable_results, f, indent=4)
                
                sg.popup(f'Backtest completed. Results saved to {results_file}')
            
            if backtest_thread is None or not backtest_thread.is_alive():
                try:
                    backtest_thread = threading.Thread(target=run_backtest, daemon=True)
                    backtest_thread.start()
                    sg.popup('Backtesting started.')
                except Exception as e:
                    sg.popup_error(f'Error starting backtest: {str(e)}')
        
        elif event.startswith('VIZ_'):
            if not values['VIZ_MODEL_SELECTION']:
                sg.popup_error('Please select a model for visualization.')
                continue
            
            model_path = os.path.join(SCRIPT_DIR, "checkpoints", values['VIZ_MODEL_SELECTION'][0])
            if not os.path.exists(model_path):
                sg.popup_error('Selected model file not found.')
                continue
            
            # Run visualization in a separate thread
            def run_visualization():
                try:
                    from visualize import PredictionVisualizer
                    from trading_model import TradingModel
                    import prepare_data
                    
                    # Load data
                    data = pd.read_csv(values['DATA_PATH'])
                    
                    # Initialize model
                    model = TradingModel.load_from_checkpoint(model_path)
                    
                    # Initialize visualizer
                    confidence_levels = []
                    if values['CONF_50']: confidence_levels.append(0.5)
                    if values['CONF_80']: confidence_levels.append(0.8)
                    if values['CONF_90']: confidence_levels.append(0.9)
                    if values['CONF_95']: confidence_levels.append(0.95)
                    
                    visualizer = PredictionVisualizer(
                        plot_style=values['PLOT_STYLE'],
                        confidence_levels=confidence_levels if confidence_levels else [0.5, 0.8, 0.9, 0.95]
                    )
                    
                    # Prepare test data
                    preparator = prepare_data.DataPreparator(
                        seq_len=values['SEQ_LEN'],
                        test_size=values['TEST_SIZE']
                    )
                    
                    # Process data and make predictions
                    test_data = preparator.prepare_test_data(data, price_column=values['PRICE_COLUMN'])
                    predictions = model.predict_batch(test_data)
                    
                    # Get actual prices
                    actual_prices = data[values['PRICE_COLUMN']].values[-len(predictions['means']):]
                    
                    # Show visualization based on selected type
                    if event == 'VIZ_PRICE':
                        visualizer.plot_price_predictions(
                            prices=actual_prices,
                            pred_means=predictions['means'],
                            pred_stds=predictions['stds'],
                            show_plot=True
                        )
                    elif event == 'VIZ_CONF':
                        visualizer.plot_confidence_metrics(
                            confidence_values={'default': predictions['confidence']},
                            show_plot=True
                        )
                    elif event == 'VIZ_CALIB':
                        visualizer.plot_calibration_curve(
                            targets=actual_prices,
                            pred_means=predictions['means'],
                            pred_stds=predictions['stds'],
                            show_plot=True
                        )
                    elif event == 'VIZ_SAMPLES':
                        visualizer.plot_prediction_samples(
                            prices=actual_prices,
                            samples=predictions['samples'],
                            show_plot=True
                        )
                        
                except Exception as e:
                    sg.popup_error(f'Visualization error: {str(e)}')
            
            if visualization_thread is None or not visualization_thread.is_alive():
                visualization_thread = threading.Thread(target=run_visualization, daemon=True)
                visualization_thread.start()
    
    window.close()

if __name__ == "__main__":
    main() 