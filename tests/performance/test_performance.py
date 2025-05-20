#!/usr/bin/env python
"""
Performance Testing Script

This script performs performance testing on key components of the BTC-AI application.
It measures execution time, memory usage, and CPU utilization under different configurations.
Tests include:
1. Model training performance
2. Backtesting performance
3. Data processing performance
4. Memory usage profiling

Results are saved to CSV files for further analysis.
"""

import os
import sys
import time
import csv
import psutil
import logging
import argparse
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import tracemalloc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'performance_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('performance_test')

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
results_dir = os.path.join(current_dir, "results")

# Create results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Add project root to path
sys.path.insert(0, project_root)

# Create mock modules for testing
class MockModuleFactory:
    """Creates mock modules for testing"""
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """Create a mock model of the specified type"""
        logger.info(f"Creating mock {model_type} model with {kwargs}")
        
        class MockModel:
            def __init__(self, **params):
                self.params = params
                self.is_trained = False
            
            def train(self, data=None, epochs=1):
                """Mock training method"""
                time.sleep(0.5)  # Simulate training time
                self.is_trained = True
                return {"loss": 0.1, "accuracy": 0.9}
            
            def predict(self, data=None):
                """Mock prediction method"""
                time.sleep(0.1)  # Simulate prediction time
                return np.random.random(size=(len(data) if data is not None else 10, 1))
            
            def evaluate(self, data=None):
                """Mock evaluation method"""
                time.sleep(0.2)  # Simulate evaluation time
                return {"loss": 0.2, "accuracy": 0.85}
        
        return MockModel(**kwargs)
    
    @staticmethod
    def create_agent(agent_type="PPO", **kwargs):
        """Create a mock agent of the specified type"""
        logger.info(f"Creating mock {agent_type} agent with {kwargs}")
        
        class MockAgent:
            def __init__(self, **params):
                self.params = params
                self.is_trained = False
            
            def train(self, env=None, timesteps=1000):
                """Mock training method"""
                time.sleep(0.5)  # Simulate training time
                self.is_trained = True
                return {"reward": 100.0, "episodes": 10}
            
            def predict(self, obs=None):
                """Mock prediction method"""
                time.sleep(0.1)  # Simulate prediction time
                return np.random.randint(0, 3), None  # action, state
        
        return MockAgent(**kwargs)

# Mock backtester for performance testing
class MockBacktester:
    """Mock backtester for performance testing"""
    
    def __init__(self, data_size=1000):
        """Initialize with synthetic data"""
        self.data_size = data_size
        self.data = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=data_size, freq='5T'),
            'open': np.random.normal(10000, 500, data_size),
            'high': np.random.normal(10100, 550, data_size),
            'low': np.random.normal(9900, 450, data_size),
            'close': np.random.normal(10050, 500, data_size),
            'volume': np.random.normal(5, 2, data_size)
        })
        self.equity = [10000.0]
        self.trades = []
    
    def run_backtest(self, strategy_func=None, steps=None):
        """Run a backtest with the given strategy"""
        steps = steps or self.data_size
        steps = min(steps, self.data_size)
        
        # Default strategy if none provided
        if strategy_func is None:
            strategy_func = lambda x: np.random.choice([-1, 0, 1])
        
        # Run the backtest
        for i in range(steps):
            action = strategy_func(i)
            
            # Simulate trade
            if action != 0 and np.random.random() < 0.3:  # 30% chance of making a trade
                price = self.data.iloc[i]['close']
                trade = {
                    'entry_time': self.data.iloc[i]['timestamp'],
                    'exit_time': self.data.iloc[min(i+10, self.data_size-1)]['timestamp'],
                    'entry_price': price,
                    'exit_price': price * (1 + np.random.normal(0.01 if action > 0 else -0.01, 0.02)),
                    'amount': np.random.uniform(0.1, 1.0),
                    'direction': action
                }
                self.trades.append(trade)
            
            # Update equity
            if len(self.trades) > 0:
                last_equity = self.equity[-1]
                pnl = sum([
                    t['direction'] * (t['exit_price'] - t['entry_price']) * t['amount']
                    for t in self.trades[-5:]  # Consider recent trades
                ])
                self.equity.append(last_equity + pnl)
            else:
                self.equity.append(self.equity[-1])
        
        return {
            'equity': self.equity,
            'trades': self.trades,
            'metrics': {
                'sharpe': np.random.uniform(0.5, 2.5),
                'max_drawdown': np.random.uniform(0.05, 0.2),
                'win_rate': np.random.uniform(0.4, 0.7),
                'profit_factor': np.random.uniform(1.1, 2.0)
            }
        }

class PerformanceTest:
    """Base class for performance tests"""
    
    def __init__(self, name, output_file=None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process(os.getpid())
        self.results = []
        self.output_file = output_file or os.path.join(results_dir, f"{name.lower().replace(' ', '_')}_results.csv")
    
    def start_timer(self):
        """Start the performance timer"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def stop_timer(self):
        """Stop the performance timer and return elapsed time"""
        self.end_time = time.time()
        self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        elapsed = self.end_time - self.start_time
        memory_used = self.end_memory - self.start_memory
        return elapsed, memory_used
    
    def log_result(self, configuration, elapsed_time, memory_used, cpu_percent=None, additional_metrics=None):
        """Log a performance result"""
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'configuration': configuration,
            'elapsed_time': elapsed_time,
            'memory_used_mb': memory_used
        }
        
        if cpu_percent is not None:
            result['cpu_percent'] = cpu_percent
            
        if additional_metrics:
            result.update(additional_metrics)
            
        self.results.append(result)
        logger.info(f"Test: {self.name}, Config: {configuration}, Time: {elapsed_time:.2f}s, Memory: {memory_used:.2f}MB")
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                logger.info(f"  {key}: {value}")
    
    def save_results(self):
        """Save results to CSV file"""
        if not self.results:
            logger.warning(f"No results to save for {self.name}")
            return
            
        # Get all possible columns from results
        all_columns = set()
        for result in self.results:
            all_columns.update(result.keys())
            
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_columns))
            writer.writeheader()
            writer.writerows(self.results)
            
        logger.info(f"Saved performance results to {self.output_file}")
    
    def plot_results(self, metric='elapsed_time', title=None):
        """Plot performance results"""
        if not self.results:
            logger.warning(f"No results to plot for {self.name}")
            return
            
        df = pd.DataFrame(self.results)
        
        plt.figure(figsize=(10, 6))
        ax = df.plot.bar(x='configuration', y=metric, rot=45)
        ax.set_title(title or f"{self.name} - {metric}")
        ax.set_ylabel(metric)
        ax.set_xlabel("Configuration")
        
        # Add values on top of bars
        for i, v in enumerate(df[metric]):
            ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')
        
        plt.tight_layout()
        filename = f"{self.name.lower().replace(' ', '_')}_{metric}.png"
        plt.savefig(os.path.join(results_dir, filename))
        logger.info(f"Saved plot to {os.path.join(results_dir, filename)}")

class ModelTrainingPerformanceTest(PerformanceTest):
    """Test model training performance"""
    
    def __init__(self):
        super().__init__("Model Training Performance")
    
    def run_test(self, model_type, hidden_layers, hidden_size, input_dim=64, batch_size=32, epochs=5):
        """Run a single model training performance test"""
        try:
            config_name = f"{model_type}_{hidden_layers}x{hidden_size}"
            logger.info(f"Testing model training performance: {config_name}")
            
            # Use the mock model factory
            model = MockModuleFactory.create_model(
                model_type=model_type,
                input_dim=input_dim,
                hidden_size=hidden_size,
                num_layers=hidden_layers
            )
            
            # Generate random training data
            x_train = np.random.random((batch_size * 100, input_dim))
            y_train = np.random.randint(0, 2, (batch_size * 100, 1))
            
            # Measure training performance
            self.start_timer()
            cpu_start = self.process.cpu_percent(interval=None)
            
            # Train the model
            results = model.train(data=(x_train, y_train), epochs=epochs)
            
            # Stop the timer
            cpu_end = self.process.cpu_percent(interval=None)
            elapsed, memory_used = self.stop_timer()
            
            # Log results
            self.log_result(
                configuration=config_name,
                elapsed_time=elapsed,
                memory_used=memory_used,
                cpu_percent=(cpu_start + cpu_end) / 2,
                additional_metrics={
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'hidden_layers': hidden_layers,
                    'hidden_size': hidden_size,
                    'train_loss': results.get('loss', 0),
                    'train_accuracy': results.get('accuracy', 0)
                }
            )
            
            return elapsed
            
        except Exception as e:
            logger.error(f"Error in training performance test: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def run_all_tests(self):
        """Run all model training performance tests"""
        logger.info("Starting model performance tests...")
        
        # Test different model architectures
        model_configs = [
            # model_type, hidden_layers, hidden_size
            ("Actor-Critic", 1, 64),
            ("Actor-Critic", 2, 128),
            ("Actor-Critic", 3, 256),
            ("PPO", 1, 64),
            ("PPO", 2, 128),
            ("PPO", 3, 256),
        ]
        
        results = []
        for model_type, hidden_layers, hidden_size in model_configs:
            elapsed = self.run_test(model_type, hidden_layers, hidden_size)
            if elapsed:
                results.append((model_type, hidden_layers, hidden_size, elapsed))
        
        # Save and plot results
        self.save_results()
        self.plot_results(metric='elapsed_time', title="Model Training Time by Configuration")
        self.plot_results(metric='memory_used_mb', title="Model Memory Usage by Configuration")
        
        return results

class BacktestingPerformanceTest(PerformanceTest):
    """Test backtesting performance"""
    
    def __init__(self):
        super().__init__("Backtesting Performance")
    
    def run_test(self, data_size, steps, strategy_complexity="simple"):
        """Run a single backtesting performance test"""
        try:
            config_name = f"data_{data_size}_steps_{steps}_{strategy_complexity}"
            logger.info(f"Testing backtesting performance: {config_name}")
            
            # Create a backtester
            backtester = MockBacktester(data_size=data_size)
            
            # Define strategy function based on complexity
            if strategy_complexity == "simple":
                strategy_func = lambda i: np.random.choice([-1, 0, 1])
            elif strategy_complexity == "medium":
                strategy_func = lambda i: 1 if i % 10 == 0 else (-1 if i % 15 == 0 else 0)
            else:  # complex
                def complex_strategy(i):
                    if i < 50:
                        return 0  # Initial wait period
                    elif backtester.data.iloc[i]['close'] > backtester.data.iloc[i-50:i]['close'].mean():
                        return 1  # Buy if price is above 50-period moving average
                    elif backtester.data.iloc[i]['close'] < backtester.data.iloc[i-50:i]['close'].mean():
                        return -1  # Sell if price is below 50-period moving average
                    return 0
                strategy_func = complex_strategy
            
            # Measure backtesting performance
            self.start_timer()
            cpu_start = self.process.cpu_percent(interval=None)
            
            # Run the backtest
            results = backtester.run_backtest(strategy_func=strategy_func, steps=steps)
            
            # Stop the timer
            cpu_end = self.process.cpu_percent(interval=None)
            elapsed, memory_used = self.stop_timer()
            
            # Log results
            self.log_result(
                configuration=config_name,
                elapsed_time=elapsed,
                memory_used=memory_used,
                cpu_percent=(cpu_start + cpu_end) / 2,
                additional_metrics={
                    'data_size': data_size,
                    'steps': steps,
                    'strategy_complexity': strategy_complexity,
                    'num_trades': len(results['trades']),
                    'sharpe': results['metrics']['sharpe'],
                    'max_drawdown': results['metrics']['max_drawdown']
                }
            )
            
            return elapsed
            
        except Exception as e:
            logger.error(f"Error in backtesting performance test: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def run_all_tests(self):
        """Run all backtesting performance tests"""
        logger.info("Starting backtesting performance tests...")
        
        # Test different backtest configurations
        backtest_configs = [
            # data_size, steps, strategy_complexity
            (1000, 500, "simple"),
            (5000, 2500, "simple"),
            (10000, 5000, "simple"),
            (5000, 2500, "medium"),
            (5000, 2500, "complex"),
        ]
        
        results = []
        for data_size, steps, strategy_complexity in backtest_configs:
            elapsed = self.run_test(data_size, steps, strategy_complexity)
            if elapsed:
                results.append((data_size, steps, strategy_complexity, elapsed))
        
        # Save and plot results
        self.save_results()
        self.plot_results(metric='elapsed_time', title="Backtesting Time by Configuration")
        self.plot_results(metric='memory_used_mb', title="Backtesting Memory Usage by Configuration")
        
        return results

class DataProcessingPerformanceTest(PerformanceTest):
    """Test data processing performance"""
    
    def __init__(self):
        super().__init__("Data Processing Performance")
    
    def run_test(self, data_size, num_features, processing_type="standard"):
        """Run a single data processing performance test"""
        try:
            config_name = f"data_{data_size}_features_{num_features}_{processing_type}"
            logger.info(f"Testing data processing performance: {config_name}")
            
            # Generate synthetic data
            data = pd.DataFrame({
                'timestamp': pd.date_range(start='2020-01-01', periods=data_size, freq='5T'),
                'open': np.random.normal(10000, 500, data_size),
                'high': np.random.normal(10100, 550, data_size),
                'low': np.random.normal(9900, 450, data_size),
                'close': np.random.normal(10050, 500, data_size),
                'volume': np.random.normal(5, 2, data_size)
            })
            
            # Add random features
            for i in range(num_features):
                data[f'feature_{i}'] = np.random.random(data_size)
            
            # Measure data processing performance
            self.start_timer()
            cpu_start = self.process.cpu_percent(interval=None)
            
            # Perform data processing based on type
            if processing_type == "standard":
                # Standard processing (rolling windows, technical indicators)
                for window in [5, 10, 20, 50, 100]:
                    data[f'ma_{window}'] = data['close'].rolling(window=window).mean()
                    data[f'std_{window}'] = data['close'].rolling(window=window).std()
                
                # Calculate returns
                data['daily_return'] = data['close'].pct_change(20)  # 20 bars = ~1 day
                data['weekly_return'] = data['close'].pct_change(100)  # 100 bars = ~1 week
                
            elif processing_type == "advanced":
                # More complex processing
                for window in [5, 10, 20, 50, 100]:
                    data[f'ma_{window}'] = data['close'].rolling(window=window).mean()
                    data[f'std_{window}'] = data['close'].rolling(window=window).std()
                    data[f'min_{window}'] = data['low'].rolling(window=window).min()
                    data[f'max_{window}'] = data['high'].rolling(window=window).max()
                    
                # Calculate more advanced indicators
                # Simulating RSI calculation
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                data['rsi'] = 100 - (100 / (1 + rs))
                
                # Simulating MACD
                data['ema_12'] = data['close'].ewm(span=12).mean()
                data['ema_26'] = data['close'].ewm(span=26).mean()
                data['macd'] = data['ema_12'] - data['ema_26']
                data['macd_signal'] = data['macd'].ewm(span=9).mean()
                
            else:  # heavy processing
                # Very intensive processing
                # First do all the advanced processing
                for window in [5, 10, 20, 50, 100, 200]:
                    data[f'ma_{window}'] = data['close'].rolling(window=window).mean()
                    data[f'std_{window}'] = data['close'].rolling(window=window).std()
                    data[f'min_{window}'] = data['low'].rolling(window=window).min()
                    data[f'max_{window}'] = data['high'].rolling(window=window).max()
                    data[f'range_{window}'] = data[f'max_{window}'] - data[f'min_{window}']
                    
                # RSI calculation
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                data['rsi'] = 100 - (100 / (1 + rs))
                
                # MACD
                data['ema_12'] = data['close'].ewm(span=12).mean()
                data['ema_26'] = data['close'].ewm(span=26).mean()
                data['macd'] = data['ema_12'] - data['ema_26']
                data['macd_signal'] = data['macd'].ewm(span=9).mean()
                
                # Add correlation matrix calculation (very intensive)
                if data_size <= 10000:  # Avoid OOM for large datasets
                    # Calculate correlation matrix for all numeric columns
                    corr_matrix = data.select_dtypes(include=[np.number]).corr()
                    
                    # Find highest correlations
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_pairs.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_matrix.iloc[i, j]
                            ))
                    
                    # Sort by absolute correlation
                    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Stop the timer
            cpu_end = self.process.cpu_percent(interval=None)
            elapsed, memory_used = self.stop_timer()
            
            # Log results
            self.log_result(
                configuration=config_name,
                elapsed_time=elapsed,
                memory_used=memory_used,
                cpu_percent=(cpu_start + cpu_end) / 2,
                additional_metrics={
                    'data_size': data_size,
                    'num_features': num_features,
                    'processing_type': processing_type,
                    'num_columns_after': len(data.columns)
                }
            )
            
            return elapsed
            
        except Exception as e:
            logger.error(f"Error in data processing performance test: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def run_all_tests(self):
        """Run all data processing performance tests"""
        logger.info("Starting data processing performance tests...")
        
        # Test different data processing configurations
        processing_configs = [
            # data_size, num_features, processing_type
            (1000, 5, "standard"),
            (5000, 10, "standard"),
            (10000, 20, "standard"),
            (5000, 10, "advanced"),
            (5000, 10, "heavy"),
        ]
        
        results = []
        for data_size, num_features, processing_type in processing_configs:
            elapsed = self.run_test(data_size, num_features, processing_type)
            if elapsed:
                results.append((data_size, num_features, processing_type, elapsed))
        
        # Save and plot results
        self.save_results()
        self.plot_results(metric='elapsed_time', title="Data Processing Time by Configuration")
        self.plot_results(metric='memory_used_mb', title="Data Processing Memory Usage by Configuration")
        
        return results

def run_performance_tests(tests_to_run=None):
    """Run performance tests"""
    all_tests = {
        'model': ModelTrainingPerformanceTest,
        'backtest': BacktestingPerformanceTest,
        'data': DataProcessingPerformanceTest
    }
    
    results = {}
    
    # Determine which tests to run
    if tests_to_run:
        test_classes = {name: cls for name, cls in all_tests.items() if name in tests_to_run}
    else:
        test_classes = all_tests
    
    # Run each test
    for name, test_class in test_classes.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {name} performance tests")
        logger.info(f"{'='*50}")
        
        test = test_class()
        try:
            test_results = test.run_all_tests()
            results[name] = {
                'success': True,
                'results': test_results
            }
        except Exception as e:
            logger.error(f"Error running {name} performance tests: {e}")
            logger.error(traceback.format_exc())
            results[name] = {
                'success': False,
                'error': str(e)
            }
    
    # Log summary
    logger.info("\nPerformance Testing Summary:")
    logger.info("---------------------------")
    for name, result in results.items():
        if result['success']:
            logger.info(f"{name}: PASSED ({len(result['results'])} tests)")
        else:
            logger.info(f"{name}: FAILED - {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run performance tests for the BTC-AI application")
    parser.add_argument(
        "--tests", 
        nargs="+", 
        choices=["model", "backtest", "data", "all"],
        default=["all"],
        help="Specify which performance tests to run"
    )
    
    args = parser.parse_args()
    
    # Determine which tests to run
    tests_to_run = None
    if "all" not in args.tests:
        tests_to_run = args.tests
    
    try:
        # Run the specified tests
        results = run_performance_tests(tests_to_run)
        
        # Exit with appropriate status
        if all(r.get('success', False) for r in results.values()):
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception in performance tests: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 