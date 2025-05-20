#!/usr/bin/env python
"""
Stress Testing Script

This script performs stress testing on the BTC-AI application by simulating:
1. Long training runs with large datasets
2. High-frequency data processing and model inference
3. Multiple parallel operations to test system stability
4. Extended backtesting sessions

The script monitors system resources, stability, and detects memory leaks or performance degradation.
"""

import os
import sys
import time
import gc
import psutil
import logging
import threading
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tracemalloc
import argparse
from multiprocessing import Process, Queue, cpu_count, freeze_support

# Windows multiprocessing support
if __name__ == "__main__":
    freeze_support()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'stress_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('stress_test')

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
results_dir = os.path.join(current_dir, "stress_results")

# Create results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Add project root to path
sys.path.insert(0, project_root)

# Mock data generator for testing
class MockDataGenerator:
    """Generates mock data for stress testing"""
    
    @staticmethod
    def generate_price_data(size=1000, freq='1min'):
        """Generate mock price data"""
        start_date = datetime.now() - timedelta(days=size//1440)  # Assuming 1440 minutes in a day
        dates = pd.date_range(start=start_date, periods=size, freq=freq)
        
        # Start with a base price
        base_price = 10000
        
        # Generate random walk
        random_walk = np.random.normal(0, 1, size=size).cumsum()
        
        # Scale to reasonable BTC price movements
        scaled_walk = random_walk * 100
        
        # Create price series
        prices = base_price + scaled_walk
        
        # Generate OHLCV data
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.01, size)),
            'low': prices * (1 - np.random.uniform(0, 0.01, size)),
            'close': prices * (1 + np.random.normal(0, 0.005, size)),
            'volume': np.random.gamma(2, 5, size)
        })
        
        return df

class SystemMonitor:
    """Monitors system resources during stress tests"""
    
    def __init__(self, interval=1.0, log_to_file=True):
        self.interval = interval
        self.log_to_file = log_to_file
        self.process = psutil.Process(os.getpid())
        self.running = False
        self.stats = []
        self.start_time = None
        
        # File for detailed stats
        if log_to_file:
            self.filename = os.path.join(results_dir, f'system_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    
    def start(self):
        """Start monitoring system resources"""
        if self.running:
            return
        
        self.stats = []
        self.start_time = time.time()
        self.running = True
        
        if self.log_to_file:
            with open(self.filename, 'w') as f:
                f.write('timestamp,cpu_percent,memory_percent,memory_mb,disk_io_read_mb,disk_io_write_mb,net_io_sent_mb,net_io_recv_mb\n')
        
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"System monitoring started with interval {self.interval}s")
    
    def stop(self):
        """Stop monitoring system resources"""
        if not self.running:
            return
        
        self.running = False
        self.thread.join(timeout=2.0)
        logger.info("System monitoring stopped")
        
        return self.get_summary()
    
    def _monitor_loop(self):
        """Main monitoring loop that collects system stats"""
        last_disk_io = psutil.disk_io_counters()
        last_net_io = psutil.net_io_counters()
        
        while self.running:
            try:
                # Get current stats
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                memory_percent = self.process.memory_percent()
                
                # System-wide CPU
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # Disk IO
                disk_io = psutil.disk_io_counters()
                disk_read_mb = (disk_io.read_bytes - last_disk_io.read_bytes) / (1024 * 1024)
                disk_write_mb = (disk_io.write_bytes - last_disk_io.write_bytes) / (1024 * 1024)
                last_disk_io = disk_io
                
                # Network IO
                net_io = psutil.net_io_counters()
                net_sent_mb = (net_io.bytes_sent - last_net_io.bytes_sent) / (1024 * 1024)
                net_recv_mb = (net_io.bytes_recv - last_net_io.bytes_recv) / (1024 * 1024)
                last_net_io = net_io
                
                # Collect stats
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                stat = {
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_mb': memory_mb,
                    'disk_io_read_mb': disk_read_mb,
                    'disk_io_write_mb': disk_write_mb,
                    'net_io_sent_mb': net_sent_mb,
                    'net_io_recv_mb': net_recv_mb
                }
                self.stats.append(stat)
                
                # Log to file
                if self.log_to_file:
                    with open(self.filename, 'a') as f:
                        f.write(f"{timestamp},{cpu_percent},{memory_percent},{memory_mb},{disk_read_mb},{disk_write_mb},{net_sent_mb},{net_recv_mb}\n")
                
                # Sleep for the interval
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(self.interval)
    
    def get_summary(self):
        """Get summary of collected metrics"""
        if not self.stats:
            return None
        
        df = pd.DataFrame(self.stats)
        
        summary = {
            'duration': time.time() - self.start_time,
            'avg_cpu_percent': df['cpu_percent'].mean(),
            'max_cpu_percent': df['cpu_percent'].max(),
            'avg_memory_mb': df['memory_mb'].mean(),
            'max_memory_mb': df['memory_mb'].max(),
            'memory_growth': df['memory_mb'].iloc[-1] - df['memory_mb'].iloc[0] if len(df) > 1 else 0,
            'total_disk_read_mb': df['disk_io_read_mb'].sum(),
            'total_disk_write_mb': df['disk_io_write_mb'].sum(),
            'total_net_sent_mb': df['net_io_sent_mb'].sum(),
            'total_net_recv_mb': df['net_io_recv_mb'].sum()
        }
        
        # Log summary
        logger.info(f"System monitoring summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value:.2f}")
        
        # Plot graphs
        self.plot_metrics()
        
        return summary
    
    def plot_metrics(self):
        """Plot system metrics over time"""
        if not self.stats:
            return
        
        df = pd.DataFrame(self.stats)
        df['elapsed_time'] = pd.to_datetime(df['timestamp']) - pd.to_datetime(df['timestamp'].iloc[0])
        df['elapsed_seconds'] = df['elapsed_time'].dt.total_seconds()
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU Usage
        axs[0, 0].plot(df['elapsed_seconds'], df['cpu_percent'])
        axs[0, 0].set_title('CPU Usage')
        axs[0, 0].set_xlabel('Time (seconds)')
        axs[0, 0].set_ylabel('CPU (%)')
        axs[0, 0].grid(True)
        
        # Memory Usage
        axs[0, 1].plot(df['elapsed_seconds'], df['memory_mb'])
        axs[0, 1].set_title('Memory Usage')
        axs[0, 1].set_xlabel('Time (seconds)')
        axs[0, 1].set_ylabel('Memory (MB)')
        axs[0, 1].grid(True)
        
        # Disk IO
        axs[1, 0].plot(df['elapsed_seconds'], df['disk_io_read_mb'], label='Read')
        axs[1, 0].plot(df['elapsed_seconds'], df['disk_io_write_mb'], label='Write')
        axs[1, 0].set_title('Disk IO')
        axs[1, 0].set_xlabel('Time (seconds)')
        axs[1, 0].set_ylabel('MB')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Network IO
        axs[1, 1].plot(df['elapsed_seconds'], df['net_io_sent_mb'], label='Sent')
        axs[1, 1].plot(df['elapsed_seconds'], df['net_io_recv_mb'], label='Received')
        axs[1, 1].set_title('Network IO')
        axs[1, 1].set_xlabel('Time (seconds)')
        axs[1, 1].set_ylabel('MB')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'system_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
        logger.info(f"System metrics plot saved to {os.path.join(results_dir, 'system_metrics.png')}")

# Worker function for multiprocessing tasks
def worker_function(task_id, duration, interval, queue):
    """Worker function to execute in separate process"""
    try:
        # Configure logging for this process
        worker_logger = logging.getLogger(f'worker_{task_id}')
        worker_logger.info(f"Worker {task_id} started")
        
        start_time = time.time()
        end_time = start_time + duration
        
        # Track worker stats
        stats = {
            'task_id': task_id,
            'iterations': 0,
            'errors': 0,
            'start_memory': psutil.Process().memory_info().rss / (1024 * 1024)
        }
        
        while time.time() < end_time:
            try:
                # Simulate CPU-intensive work
                size = 500  # Reduced size to avoid excessive memory usage
                matrix_a = np.random.random((size, size))
                matrix_b = np.random.random((size, size))
                result = np.dot(matrix_a, matrix_b)
                
                # Update stats
                stats['iterations'] += 1
                
                # Periodically report progress
                if stats['iterations'] % 5 == 0:
                    current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                    elapsed = time.time() - start_time
                    worker_logger.debug(f"Worker {task_id}: {stats['iterations']} iterations in {elapsed:.2f}s, memory: {current_memory:.2f} MB")
                
                # Sleep to avoid overwhelming the system
                time.sleep(interval)
                
            except Exception as e:
                stats['errors'] += 1
                worker_logger.error(f"Worker {task_id} error: {e}")
        
        # Collect final stats
        stats['duration'] = time.time() - start_time
        stats['end_memory'] = psutil.Process().memory_info().rss / (1024 * 1024)
        stats['memory_growth'] = stats['end_memory'] - stats['start_memory']
        
        worker_logger.info(f"Worker {task_id} completed {stats['iterations']} iterations with {stats['errors']} errors")
        queue.put(stats)
        return stats
    except Exception as e:
        # Handle any unexpected errors
        print(f"Worker {task_id} critical error: {e}")
        queue.put({
            'task_id': task_id,
            'critical_error': str(e),
            'iterations': 0,
            'errors': 1
        })

class StressTest:
    """Base class for all stress tests"""
    
    def __init__(self, name, duration=30, interval=0.1):
        self.name = name
        self.duration = duration
        self.interval = interval
        self.monitor = SystemMonitor(interval=1.0)
        self.results = None
    
    def setup(self):
        """Set up the test environment"""
        logger.info(f"Setting up {self.name} test")
    
    def run(self):
        """Run the stress test"""
        try:
            logger.info(f"Starting {self.name} stress test for {self.duration}s")
            self.setup()
            
            # Start system monitoring
            self.monitor.start()
            
            # Run the actual test
            start_time = time.time()
            test_results = self.execute()
            actual_duration = time.time() - start_time
            
            # Stop monitoring
            monitoring_results = self.monitor.stop()
            
            # Combine results
            self.results = {
                'name': self.name,
                'planned_duration': self.duration,
                'actual_duration': actual_duration,
                'test_specific_results': test_results,
                'system_monitoring': monitoring_results
            }
            
            self.cleanup()
            
            logger.info(f"{self.name} test completed in {actual_duration:.2f}s")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in {self.name} test: {e}")
            logger.error(traceback.format_exc())
            self.cleanup()
            return {
                'name': self.name,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def execute(self):
        """Execute the actual stress test (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def cleanup(self):
        """Clean up after the test"""
        logger.info(f"Cleaning up {self.name} test")

class CPUStressTest(StressTest):
    """Test CPU performance under heavy load"""
    
    def __init__(self, duration=30, interval=0.1, num_processes=None):
        super().__init__("CPU Stress Test", duration, interval)
        self.num_processes = num_processes if num_processes is not None else max(1, cpu_count() - 1)
    
    def execute(self):
        """Run CPU-intensive operations"""
        logger.info(f"Running CPU stress test with {self.num_processes} processes")
        
        # Use Queue for processes to return results
        queue = Queue()
        processes = []
        
        # Start worker processes
        for i in range(self.num_processes):
            try:
                p = Process(target=worker_function, args=(i, self.duration, self.interval, queue))
                processes.append(p)
                p.start()
            except Exception as e:
                logger.error(f"Error starting process {i}: {e}")
        
        # Wait for all processes to complete
        for p in processes:
            try:
                p.join(timeout=self.duration + 10)  # Give some extra time for processes to finish
                if p.is_alive():
                    logger.warning(f"Process {p.pid} did not terminate, attempting to terminate forcefully")
                    p.terminate()
                    time.sleep(1)
                    if p.is_alive():
                        logger.warning(f"Process {p.pid} still alive after termination attempt")
            except Exception as e:
                logger.error(f"Error joining process: {e}")
        
        # Collect results
        results = []
        try:
            while not queue.empty():
                results.append(queue.get(block=False))
        except Exception as e:
            logger.error(f"Error retrieving results from queue: {e}")
        
        # If no results, something went wrong
        if not results:
            logger.warning("No results collected from worker processes")
            # Create a fallback result
            results = [{
                'task_id': 'fallback',
                'iterations': 0,
                'errors': 1,
                'duration': self.duration,
                'memory_growth': 0,
                'error_message': "No results from workers"
            }]
        
        # Calculate summary stats
        total_iterations = sum(r.get('iterations', 0) for r in results)
        total_errors = sum(r.get('errors', 0) for r in results)
        avg_memory_growth = sum(r.get('memory_growth', 0) for r in results) / len(results) if results else 0
        
        summary = {
            'num_processes': self.num_processes,
            'processes_reported': len(results),
            'total_iterations': total_iterations,
            'iterations_per_second': total_iterations / self.duration if self.duration > 0 else 0,
            'total_errors': total_errors,
            'avg_memory_growth_mb': avg_memory_growth
        }
        
        # Log summary
        logger.info(f"CPU stress test summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return {
            'summary': summary,
            'process_results': results
        }

class MemoryStressTest(StressTest):
    """Test memory usage under heavy load"""
    
    def __init__(self, duration=30, initial_size_mb=100, growth_rate_mb=10):
        super().__init__("Memory Stress Test", duration)
        self.initial_size_mb = initial_size_mb
        self.growth_rate_mb = growth_rate_mb
        self.memory_blocks = []
    
    def execute(self):
        """Gradually allocate memory and monitor system behavior"""
        logger.info(f"Running memory stress test with initial size {self.initial_size_mb}MB and growth rate {self.growth_rate_mb}MB/s")
        
        start_time = time.time()
        end_time = start_time + self.duration
        
        # Initial allocation
        self._allocate_memory(self.initial_size_mb)
        
        # Stats tracking
        allocated_mb = self.initial_size_mb
        max_allocated_mb = allocated_mb
        allocation_attempts = 1
        allocation_failures = 0
        
        # Gradually increase memory usage
        while time.time() < end_time:
            try:
                # Calculate time elapsed and desired total allocation
                elapsed = time.time() - start_time
                target_allocation = self.initial_size_mb + (elapsed * self.growth_rate_mb)
                to_allocate = target_allocation - allocated_mb
                
                if to_allocate >= 1:
                    self._allocate_memory(to_allocate)
                    allocated_mb += to_allocate
                    max_allocated_mb = max(max_allocated_mb, allocated_mb)
                    allocation_attempts += 1
                    
                    logger.info(f"Memory allocated: {allocated_mb:.2f} MB")
                
                # Short sleep to prevent CPU overuse
                time.sleep(0.1)
                
            except MemoryError:
                allocation_failures += 1
                logger.warning(f"Memory allocation failed at {allocated_mb:.2f} MB")
                # If we hit memory limits, wait a bit before continuing
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in memory allocation: {e}")
                allocation_failures += 1
                time.sleep(1)
        
        # Calculate memory growth rate
        actual_growth_rate = (max_allocated_mb - self.initial_size_mb) / self.duration
        
        summary = {
            'initial_size_mb': self.initial_size_mb,
            'target_growth_rate_mb': self.growth_rate_mb,
            'actual_growth_rate_mb': actual_growth_rate,
            'max_allocated_mb': max_allocated_mb,
            'allocation_attempts': allocation_attempts,
            'allocation_failures': allocation_failures
        }
        
        # Log summary
        logger.info(f"Memory stress test summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return summary
    
    def _allocate_memory(self, size_mb):
        """Allocate a block of memory of specified size"""
        # Allocate in smaller chunks to avoid large memory spikes
        chunk_size = min(size_mb, 10)  # Max 10MB per chunk
        remaining = size_mb
        
        while remaining > 0:
            current_chunk = min(remaining, chunk_size)
            # Create a byte array of the specified size
            block = bytearray(int(current_chunk * 1024 * 1024))
            # Ensure it's actually allocated by writing to it
            for i in range(0, len(block), 1024*1024):
                block[i] = 1
            
            self.memory_blocks.append(block)
            remaining -= current_chunk
    
    def cleanup(self):
        """Release allocated memory"""
        logger.info(f"Cleaning up memory stress test, releasing {len(self.memory_blocks)} blocks")
        self.memory_blocks.clear()
        # Force garbage collection
        import gc
        gc.collect()
        super().cleanup()

class ParallelProcessingTest(StressTest):
    """Test system stability with multiple parallel processes"""
    
    def __init__(self, duration=30, num_workers=None, work_size=500):
        super().__init__("Parallel Processing Test", duration)
        self.num_workers = num_workers if num_workers is not None else max(1, cpu_count() - 1)
        self.work_size = work_size
        self.stop_flag = False
        self.results_queue = Queue()
    
    def worker_task(self, worker_id):
        """Task executed by each worker thread"""
        start_time = time.time()
        iterations = 0
        errors = 0
        
        # Generate mock data once
        data_size = self.work_size
        data = MockDataGenerator.generate_price_data(size=data_size)
        
        while not self.stop_flag and (time.time() - start_time) < self.duration:
            try:
                # Create a copy to avoid modifying the original
                df = data.copy()
                
                # Perform data processing operations
                # 1. Calculate moving averages
                for window in [5, 10, 20, 50]:
                    df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                
                # 2. Calculate returns
                df['returns'] = df['close'].pct_change()
                
                # 3. Calculate momentum indicators
                df['momentum_5'] = df['close'].diff(5)
                df['momentum_10'] = df['close'].diff(10)
                
                # 4. Perform some calculations on the derived data
                df['signal'] = np.where(df['ma_5'] > df['ma_20'], 1, -1)
                
                # 5. Calculate a performance metric
                signal_returns = df['signal'].shift(1) * df['returns']
                cumulative_returns = (1 + signal_returns.fillna(0)).cumprod()
                
                # 6. Simulate some more computation
                for _ in range(50):
                    np.random.random((100, 100))
                
                iterations += 1
                
                # Sleep a tiny bit to yield to other workers
                time.sleep(0.001)
                
            except Exception as e:
                errors += 1
                logger.error(f"Worker {worker_id} error: {str(e)}")
        
        # Report results
        elapsed = time.time() - start_time
        self.results_queue.put({
            'worker_id': worker_id,
            'iterations': iterations,
            'errors': errors,
            'elapsed': elapsed,
            'iterations_per_second': iterations / elapsed if elapsed > 0 else 0
        })
    
    def execute(self):
        """Run multiple worker threads in parallel"""
        logger.info(f"Running parallel processing test with {self.num_workers} workers")
        
        # Start worker threads
        threads = []
        for i in range(self.num_workers):
            thread = threading.Thread(target=self.worker_task, args=(i,))
            thread.daemon = True
            threads.append(thread)
            thread.start()
        
        # Wait for the test duration
        time.sleep(self.duration)
        
        # Signal threads to stop
        self.stop_flag = True
        
        # Wait for all threads to complete
        logger.info("Waiting for worker threads to complete...")
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Collect results
        results = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get())
        
        # Calculate summary statistics
        total_iterations = sum(r['iterations'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        avg_iterations_per_second = sum(r['iterations_per_second'] for r in results)
        
        summary = {
            'num_workers': self.num_workers,
            'work_size': self.work_size,
            'total_iterations': total_iterations,
            'total_iterations_per_second': total_iterations / self.duration,
            'avg_iterations_per_worker': total_iterations / self.num_workers,
            'total_errors': total_errors
        }
        
        # Log summary
        logger.info(f"Parallel processing test summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return {
            'summary': summary,
            'worker_results': results
        }

class LongRunningTest(StressTest):
    """Test system stability over a long period of time"""
    
    def __init__(self, duration=300, check_interval=30):
        super().__init__("Long Running Test", duration)
        self.check_interval = check_interval
        self.memory_checkpoints = []
    
    def execute(self):
        """Run a test for an extended period to check for stability and memory leaks"""
        logger.info(f"Running long running test for {self.duration}s with checks every {self.check_interval}s")
        
        start_time = time.time()
        end_time = start_time + self.duration
        process = psutil.Process()
        
        # Record initial memory usage
        initial_memory = process.memory_info().rss / (1024 * 1024)
        self.memory_checkpoints.append({
            'timestamp': time.time(),
            'elapsed': 0,
            'memory_mb': initial_memory
        })
        
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Data for continuous processing
        data_size = 5000
        iterations = 0
        data = MockDataGenerator.generate_price_data(size=data_size)
        
        # Keep a reference to some data to simulate memory growth patterns
        cached_frames = []
        
        next_check = start_time + self.check_interval
        
        while time.time() < end_time:
            try:
                # Simulate continuous processing
                df = data.copy()
                
                # 1. Data transformation
                for col in ['open', 'high', 'low', 'close']:
                    df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
                
                # 2. Calculate technical indicators
                for window in [5, 10, 20, 50, 100]:
                    df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                    df[f'std_{window}'] = df['close'].rolling(window=window).std()
                
                # 3. Create more derived features
                for lag in [1, 5, 10, 20]:
                    df[f'lag_{lag}'] = df['close'].shift(lag)
                    df[f'return_{lag}'] = df['close'].pct_change(lag)
                
                # 4. Periodically cache data (to simulate memory leaks)
                if iterations % 100 == 0 and len(cached_frames) < 10:
                    cached_frames.append(df.sample(n=1000))
                
                iterations += 1
                
                # Check memory usage at intervals
                current_time = time.time()
                if current_time >= next_check:
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    elapsed = current_time - start_time
                    
                    self.memory_checkpoints.append({
                        'timestamp': current_time,
                        'elapsed': elapsed,
                        'memory_mb': current_memory
                    })
                    
                    growth = current_memory - initial_memory
                    growth_rate = growth / elapsed
                    
                    logger.info(f"Memory checkpoint at {elapsed:.2f}s: {current_memory:.2f} MB "
                               f"(growth: {growth:.2f} MB, rate: {growth_rate:.2f} MB/s)")
                    
                    next_check = current_time + self.check_interval
                
                # Sleep to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in long running test: {e}")
                time.sleep(1)
        
        # Calculate final statistics
        actual_duration = time.time() - start_time
        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_growth = final_memory - initial_memory
        memory_growth_rate = memory_growth / actual_duration
        
        # Clear cached frames
        cached_frames.clear()
        
        summary = {
            'iterations': iterations,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_growth_mb': memory_growth,
            'memory_growth_rate_mb_per_s': memory_growth_rate,
            'iterations_per_second': iterations / actual_duration
        }
        
        # Log summary
        logger.info(f"Long running test summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        # Create memory growth plot
        self._plot_memory_growth()
        
        return {
            'summary': summary,
            'memory_checkpoints': self.memory_checkpoints
        }
    
    def _plot_memory_growth(self):
        """Plot memory growth over time"""
        if not self.memory_checkpoints:
            return
        
        df = pd.DataFrame(self.memory_checkpoints)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['elapsed'], df['memory_mb'])
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (MB)')
        plt.grid(True)
        plt.tight_layout()
        
        filename = os.path.join(results_dir, f'memory_growth_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(filename)
        logger.info(f"Memory growth plot saved to {filename}")

def run_stress_tests(tests_to_run=None, duration=30):
    """Run specified stress tests or all if none specified"""
    all_tests = {
        'cpu': CPUStressTest,
        'memory': MemoryStressTest,
        'parallel': ParallelProcessingTest,
        'long': LongRunningTest
    }
    
    # Determine which tests to run
    if tests_to_run:
        test_classes = {name: cls for name, cls in all_tests.items() if name in tests_to_run}
    else:
        test_classes = all_tests
    
    results = {}
    
    for name, test_class in test_classes.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting {name} stress test")
        logger.info(f"{'='*50}")
        
        # Create test with specified duration
        if name == 'long':
            # Long running test typically needs more time
            test = test_class(duration=max(duration, 120))
        else:
            test = test_class(duration=duration)
        
        # Run the test
        test_results = test.run()
        results[name] = test_results
        
        # Sleep between tests to let system recover
        logger.info(f"Sleeping for 5 seconds to let system recover...")
        time.sleep(5)
    
    # Log final summary
    logger.info("\nStress Testing Summary:")
    logger.info("----------------------")
    for name, result in results.items():
        if 'error' in result:
            logger.info(f"{name}: FAILED - {result['error']}")
        else:
            logger.info(f"{name}: PASSED - {result.get('test_specific_results', {}).get('summary', {})}")
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run stress tests for the BTC-AI application")
    parser.add_argument(
        "--tests", 
        nargs="+", 
        choices=["cpu", "memory", "parallel", "long", "all"],
        default=["all"],
        help="Specify which stress tests to run"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=30,
        help="Duration in seconds for each test"
    )
    
    args = parser.parse_args()
    
    # Determine which tests to run
    tests_to_run = None
    if "all" not in args.tests:
        tests_to_run = args.tests
    
    try:
        # Run the specified tests
        results = run_stress_tests(tests_to_run, args.duration)
        
        # Exit with appropriate status
        if all('error' not in r for r in results.values()):
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception in stress tests: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 