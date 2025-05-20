#!/usr/bin/env python
"""
Performance Optimization Module

This module extends the existing performance optimization capabilities with
profiling, visualization rendering optimization, and advanced caching strategies.
"""

import os
import sys
import time
import json
import logging
import functools
import gc
import threading
import numpy as np
import psutil
import torch
import importlib
from typing import Dict, List, Callable, Any, Optional, Union

# Configure logging
logger = logging.getLogger('performance_optimizer')
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Try to import visualization dependencies for rendering optimization
try:
    import matplotlib
    # Use non-interactive backend for headless operation
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib not available, visualization optimizations disabled")

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Dynamically import utility functions
try:
    utils_module = importlib.import_module("src.utils.utils")
    optimize_memory = utils_module.optimize_memory
    measure_gpu_usage = utils_module.measure_gpu_usage
    get_optimal_gpu_targets = utils_module.get_optimal_gpu_targets
    has_utils = True
except ImportError as e:
    logger.warning(f"Could not import utility functions: {e}")
    # Define fallback functions if imports fail
    def optimize_memory():
        """Fallback memory optimization function."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"success": True, "message": "Basic memory optimization (fallback)"}
    
    def measure_gpu_usage():
        """Fallback GPU usage measurement function."""
        if torch.cuda.is_available():
            return {f"gpu_{i}": {"used_memory": torch.cuda.memory_allocated(i) / 1024**3} 
                    for i in range(torch.cuda.device_count())}
        return {}
    
    def get_optimal_gpu_targets():
        """Fallback function for GPU targets."""
        return {"batch_size": 32, "window_size": 50}

# Performance profiling and optimization class
class PerformanceOptimizer:
    """
    Provides profiling, optimization, and resource management capabilities.
    
    This class extends the existing optimization capabilities with:
    1. Function-level profiling to identify bottlenecks
    2. Visualization rendering optimization
    3. Smart caching strategies for data and models
    4. Dynamic resource allocation based on system load
    """
    
    def __init__(self, config=None):
        """
        Initialize the performance optimizer.
        
        Args:
            config (dict, optional): Configuration options
        """
        self.config = config or {}
        
        # Initialize profiling data
        self.profiling_data = {}
        
        # Cache for models and data
        self.model_cache = {}
        self.data_cache = {}
        
        # Set up performance monitoring
        self.system_stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'disk_io': []
        }
        
        # Performance thresholds
        self.memory_threshold, self.high_memory_threshold = get_optimal_gpu_targets()
        self.cpu_threshold = self.config.get('CPU_THRESHOLD', 0.85)
        
        # Optimization flags
        self.enable_profiling = self.config.get('ENABLE_PROFILING', True)
        self.enable_caching = self.config.get('ENABLE_CACHING', True)
        self.enable_viz_optimization = self.config.get('ENABLE_VIZ_OPTIMIZATION', True)
        
        # Start monitoring thread if requested
        if self.config.get('BACKGROUND_MONITORING', False):
            self._start_monitoring()
            
        logger.info(f"Performance optimizer initialized with thresholds: GPU={self.memory_threshold:.2f}, CPU={self.cpu_threshold:.2f}")
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_system_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Background monitoring started")
    
    def _stop_monitoring(self):
        """Stop background monitoring thread."""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        logger.info("Background monitoring stopped")
    
    def _monitor_system_resources(self, interval=5.0):
        """Monitor system resources in a background thread."""
        while self.monitoring_active:
            # Collect system stats
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent / 100.0
                gpu_percent = measure_gpu_usage() if torch.cuda.is_available() else 0.0
                
                # Get disk I/O stats
                disk_io = psutil.disk_io_counters()
                disk_stats = disk_io.read_bytes + disk_io.write_bytes if disk_io else 0
                
                # Store stats with timestamp
                timestamp = time.time()
                stats = {
                    'timestamp': timestamp,
                    'cpu': cpu_percent / 100.0,
                    'memory': memory_percent,
                    'gpu': gpu_percent,
                    'disk_io': disk_stats
                }
                
                # Add to history (limit length to avoid memory bloat)
                max_history = self.config.get('MAX_STATS_HISTORY', 100)
                for key in self.system_stats:
                    if key in stats:
                        self.system_stats[key].append((timestamp, stats[key]))
                        # Trim if too long
                        if len(self.system_stats[key]) > max_history:
                            self.system_stats[key] = self.system_stats[key][-max_history:]
                
                # Check for resource issues
                self._check_resource_issues(stats)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
            
            # Sleep between monitoring cycles
            time.sleep(interval)
    
    def _check_resource_issues(self, stats):
        """Check for resource-related issues and take corrective action."""
        # High GPU memory usage - trigger cleanup
        if stats.get('gpu', 0) > self.high_memory_threshold:
            logger.warning(f"High GPU memory usage detected: {stats['gpu']:.1%}")
            self.optimize_memory_usage(aggressive=True)
        
        # High CPU usage - may need to reduce worker threads
        if stats.get('cpu', 0) > self.cpu_threshold:
            logger.warning(f"High CPU usage detected: {stats['cpu']:.1%}")
    
    def profile(self, func=None, *, name=None):
        """
        Decorator for profiling function execution time.
        
        Args:
            func: Function to profile
            name: Optional custom name for the function
            
        Returns:
            Wrapped function with profiling
        """
        def decorator(f):
            func_name = name or f.__qualname__
            
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                if not self.enable_profiling:
                    return f(*args, **kwargs)
                
                start_time = time.time()
                result = f(*args, **kwargs)
                elapsed_time = time.time() - start_time
                
                # Record profiling data
                if func_name not in self.profiling_data:
                    self.profiling_data[func_name] = {
                        'calls': 0,
                        'total_time': 0,
                        'min_time': float('inf'),
                        'max_time': 0,
                        'avg_time': 0
                    }
                
                # Update stats
                self.profiling_data[func_name]['calls'] += 1
                self.profiling_data[func_name]['total_time'] += elapsed_time
                self.profiling_data[func_name]['min_time'] = min(self.profiling_data[func_name]['min_time'], elapsed_time)
                self.profiling_data[func_name]['max_time'] = max(self.profiling_data[func_name]['max_time'], elapsed_time)
                self.profiling_data[func_name]['avg_time'] = (
                    self.profiling_data[func_name]['total_time'] / 
                    self.profiling_data[func_name]['calls']
                )
                
                # Log slow functions
                threshold = self.config.get('SLOW_FUNCTION_THRESHOLD', 1.0)  # seconds
                if elapsed_time > threshold:
                    logger.warning(f"Slow function detected: {func_name} took {elapsed_time:.2f}s")
                
                return result
            
            return wrapper
        
        if func is None:
            return decorator
        return decorator(func)
    
    def get_profiling_report(self):
        """
        Generate a profiling report.
        
        Returns:
            dict: Dictionary with profiling statistics
        """
        # Sort functions by total time
        sorted_funcs = sorted(
            self.profiling_data.items(), 
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        
        # Format report
        report = {
            'timestamp': time.time(),
            'functions': {},
            'summary': {
                'total_profiled_time': sum(f[1]['total_time'] for f in sorted_funcs),
                'total_profiled_calls': sum(f[1]['calls'] for f in sorted_funcs),
                'slowest_function': sorted_funcs[0][0] if sorted_funcs else None,
                'fastest_function': sorted(
                    [(name, data) for name, data in self.profiling_data.items() if data['calls'] > 0],
                    key=lambda x: x[1]['avg_time']
                )[0][0] if self.profiling_data else None
            }
        }
        
        # Add function details
        for func_name, stats in sorted_funcs:
            report['functions'][func_name] = {
                'calls': stats['calls'],
                'total_time': stats['total_time'],
                'avg_time': stats['avg_time'],
                'min_time': stats['min_time'],
                'max_time': stats['max_time'],
                'percent_of_total': (
                    stats['total_time'] / report['summary']['total_profiled_time'] * 100
                    if report['summary']['total_profiled_time'] > 0 else 0
                )
            }
        
        return report
    
    def save_profiling_report(self, output_path=None):
        """
        Save profiling report to a file.
        
        Args:
            output_path (str, optional): Path to save the report
            
        Returns:
            str: Path to saved report
        """
        report = self.get_profiling_report()
        
        # Generate default path if not provided
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(current_dir, "profiling")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"profiling_report_{timestamp}.json")
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved profiling report to {output_path}")
        return output_path
    
    def clear_profiling_data(self):
        """Clear profiling data."""
        self.profiling_data.clear()
        logger.info("Profiling data cleared")
    
    def optimize_memory_usage(self, aggressive=False):
        """
        Optimize memory usage by cleaning up unused resources.
        
        Args:
            aggressive (bool): Whether to use aggressive optimization
            
        Returns:
            float: Current memory usage after optimization
        """
        # Call existing memory optimization
        optimize_memory()
        
        # Additional targeted optimization
        if aggressive:
            # Clear model cache if it exists
            if self.model_cache:
                logger.info("Clearing model cache")
                self.model_cache.clear()
            
            # Clear matplotlib figure cache if available
            if HAS_MATPLOTLIB:
                logger.info("Clearing matplotlib figures")
                plt.close('all')
        
        # Return current memory usage
        return measure_gpu_usage()
    
    def optimize_visualization(self, fig=None, dpi=100, max_size=(1200, 800)):
        """
        Optimize a matplotlib figure for memory usage.
        
        Args:
            fig: Matplotlib figure to optimize
            dpi: DPI for rendering
            max_size: Maximum size (width, height) in pixels
            
        Returns:
            The optimized figure
        """
        if not HAS_MATPLOTLIB or not self.enable_viz_optimization:
            return fig
        
        if fig is None:
            fig = plt.gcf()
        
        try:
            # Get current size
            figsize = fig.get_size_inches()
            
            # Calculate current pixels
            current_width = figsize[0] * dpi
            current_height = figsize[1] * dpi
            
            # Resize if needed
            if current_width > max_size[0] or current_height > max_size[1]:
                # Calculate new size to maintain aspect ratio
                aspect_ratio = figsize[0] / figsize[1]
                
                if current_width / max_size[0] > current_height / max_size[1]:
                    # Width-limited
                    new_width = max_size[0] / dpi
                    new_height = new_width / aspect_ratio
                else:
                    # Height-limited
                    new_height = max_size[1] / dpi
                    new_width = new_height * aspect_ratio
                
                # Set new size
                fig.set_size_inches(new_width, new_height)
                logger.info(f"Resized figure from {figsize} to {(new_width, new_height)}")
            
            # Reduce complexity for large plots
            if len(fig.axes) > 0:
                for ax in fig.axes:
                    # Simplify line plots if too many points
                    for line in ax.get_lines():
                        xdata = line.get_xdata()
                        if len(xdata) > 1000:
                            # Subsample data for display
                            subsample = max(1, len(xdata) // 1000)
                            line.set_data(xdata[::subsample], line.get_ydata()[::subsample])
                            logger.info(f"Reduced line data points from {len(xdata)} to {len(xdata[::subsample])}")
            
            # Use a simpler renderer
            fig.canvas.draw_idle()
            
            return fig
        
        except Exception as e:
            logger.error(f"Error optimizing visualization: {e}")
            return fig
    
    def cache_data(self, key, data):
        """
        Cache data with the given key.
        
        Args:
            key: Cache key
            data: Data to cache
            
        Returns:
            bool: Whether caching was successful
        """
        if not self.enable_caching:
            return False
            
        try:
            # Check available memory before caching
            current_memory = measure_gpu_usage()
            if current_memory > self.memory_threshold:
                logger.warning(f"Memory usage ({current_memory:.1%}) above threshold, not caching data")
                return False
            
            self.data_cache[key] = {
                'data': data,
                'timestamp': time.time(),
                'access_count': 0
            }
            return True
        except Exception as e:
            logger.error(f"Error caching data: {e}")
            return False
    
    def get_cached_data(self, key):
        """
        Retrieve cached data.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found
        """
        if not self.enable_caching or key not in self.data_cache:
            return None
        
        cache_entry = self.data_cache[key]
        cache_entry['access_count'] += 1
        cache_entry['last_access'] = time.time()
        
        return cache_entry['data']
    
    def clear_cache(self, older_than=None):
        """
        Clear cache entries.
        
        Args:
            older_than (float, optional): Clear entries older than this many seconds
            
        Returns:
            int: Number of entries cleared
        """
        if not self.enable_caching:
            return 0
            
        if older_than is None:
            # Clear all
            count = len(self.data_cache)
            self.data_cache.clear()
            return count
        
        # Clear entries older than the threshold
        current_time = time.time()
        keys_to_remove = [
            k for k, v in self.data_cache.items()
            if current_time - v['timestamp'] > older_than
        ]
        
        for key in keys_to_remove:
            del self.data_cache[key]
        
        return len(keys_to_remove)
    
    def optimize_tensor_operations(self, function):
        """
        Decorator for optimizing tensor operations in a function.
        
        Args:
            function: Function to optimize
            
        Returns:
            Wrapped function with tensor optimizations
        """
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # Set tensor allocation optimization
            if torch.cuda.is_available():
                # Using PyTorch 2.0+ memory efficient features if available
                try:
                    # Get current memory usage
                    before_memory = measure_gpu_usage()
                    
                    # Try to use memory efficient operations
                    torch.backends.cuda.matmul.allow_tf32 = True
                    
                    # Execute function
                    result = function(*args, **kwargs)
                    
                    # Get memory usage after
                    after_memory = measure_gpu_usage()
                    
                    # Log memory difference
                    if after_memory - before_memory > 0.1:  # More than 10% increase
                        logger.warning(
                            f"Function {function.__name__} increased GPU memory usage "
                            f"by {(after_memory - before_memory) * 100:.1f}%"
                        )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in tensor optimization for {function.__name__}: {e}")
                    # Fall back to normal execution
                    return function(*args, **kwargs)
            else:
                # No GPU available, just execute normally
                return function(*args, **kwargs)
                
        return wrapper
    
    def optimize_dataframe_operations(self, function):
        """
        Decorator for optimizing pandas DataFrame operations.
        
        Args:
            function: Function to optimize
            
        Returns:
            Wrapped function with DataFrame optimizations
        """
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            try:
                import pandas as pd
                
                # Get memory usage before
                process = psutil.Process(os.getpid())
                before_memory = process.memory_info().rss / 1024**2  # MB
                
                # Execute function
                result = function(*args, **kwargs)
                
                # Check if result is a DataFrame
                if isinstance(result, pd.DataFrame):
                    # Check size
                    df_size_mb = result.memory_usage(deep=True).sum() / 1024**2
                    
                    # For large DataFrames, suggest optimization
                    if df_size_mb > 100:  # More than 100MB
                        logger.warning(
                            f"Large DataFrame returned by {function.__name__} "
                            f"({df_size_mb:.1f} MB). Consider reducing columns or using "
                            f"lower precision dtypes."
                        )
                
                # Get memory after
                after_memory = process.memory_info().rss / 1024**2
                
                # Log significant memory increases
                if after_memory - before_memory > 500:  # More than 500MB increase
                    logger.warning(
                        f"Function {function.__name__} increased memory usage "
                        f"by {after_memory - before_memory:.1f} MB"
                    )
                
                return result
                
            except ImportError:
                # pandas not available
                return function(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in DataFrame optimization for {function.__name__}: {e}")
                return function(*args, **kwargs)
                
        return wrapper
    
    def get_optimization_suggestions(self):
        """
        Generate optimization suggestions based on profiling data.
        
        Returns:
            list: List of optimization suggestions
        """
        suggestions = []
        
        # Analyze profiling data for bottlenecks
        if self.profiling_data:
            # Identify slow functions (>10% of total time)
            report = self.get_profiling_report()
            total_time = report['summary']['total_profiled_time']
            
            for func_name, stats in report['functions'].items():
                if stats['percent_of_total'] > 10:
                    suggestions.append({
                        'type': 'bottleneck',
                        'function': func_name,
                        'percent_of_total': stats['percent_of_total'],
                        'suggestion': f"Optimize {func_name} which takes {stats['percent_of_total']:.1f}% of profiled time"
                    })
        
        # Check memory usage patterns
        if 'gpu_usage' in self.system_stats and self.system_stats['gpu_usage']:
            gpu_usages = [u[1] for u in self.system_stats['gpu_usage']]
            avg_gpu = sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0
            max_gpu = max(gpu_usages) if gpu_usages else 0
            
            if max_gpu > 0.9:  # >90% max usage
                suggestions.append({
                    'type': 'memory',
                    'value': max_gpu,
                    'suggestion': "GPU memory usage peaked above 90%. Consider smaller batch sizes or model simplification."
                })
            
            if avg_gpu > 0.8:  # >80% average usage
                suggestions.append({
                    'type': 'memory',
                    'value': avg_gpu,
                    'suggestion': f"Average GPU usage is high ({avg_gpu:.1%}). Consider memory optimizations."
                })
        
        # Check CPU usage patterns
        if 'cpu_usage' in self.system_stats and self.system_stats['cpu_usage']:
            cpu_usages = [u[1] for u in self.system_stats['cpu_usage']]
            avg_cpu = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
            
            if avg_cpu > 0.9:  # >90% average CPU
                suggestions.append({
                    'type': 'cpu',
                    'value': avg_cpu,
                    'suggestion': f"CPU usage is very high ({avg_cpu:.1%}). Consider reducing parallel operations."
                })
        
        return suggestions
    
    def visualize_system_resources(self, output_path=None):
        """
        Create visualization of system resource usage.
        
        Args:
            output_path (str, optional): Path to save the visualization
            
        Returns:
            str or None: Path to saved visualization if successful
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not available, cannot create visualization")
            return None
            
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot resource usage over time
            plt.subplot(2, 1, 1)
            
            # Get data for plotting
            timestamps = {}
            values = {}
            
            for resource in ['cpu_usage', 'memory_usage', 'gpu_usage']:
                if resource in self.system_stats and self.system_stats[resource]:
                    timestamps[resource] = [t[0] for t in self.system_stats[resource]]
                    values[resource] = [v[1] for v in self.system_stats[resource]]
            
            # Convert timestamps to relative time (minutes)
            start_time = min([min(ts) for ts in timestamps.values()]) if timestamps else time.time()
            
            for resource in timestamps:
                rel_times = [(t - start_time) / 60 for t in timestamps[resource]]  # Convert to minutes
                
                if resource == 'cpu_usage':
                    label = 'CPU Usage'
                    color = 'blue'
                elif resource == 'memory_usage':
                    label = 'System Memory'
                    color = 'green'
                elif resource == 'gpu_usage':
                    label = 'GPU Memory'
                    color = 'red'
                else:
                    label = resource
                    color = 'gray'
                
                plt.plot(rel_times, values[resource], label=label, color=color, alpha=0.7)
            
            plt.xlabel('Time (minutes)')
            plt.ylabel('Usage (0-1)')
            plt.title('System Resource Usage')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot profiling data
            plt.subplot(2, 1, 2)
            
            if self.profiling_data:
                # Get top 10 functions by time
                report = self.get_profiling_report()
                top_funcs = sorted(
                    report['functions'].items(),
                    key=lambda x: x[1]['total_time'],
                    reverse=True
                )[:10]
                
                func_names = [f.__name__ if callable(f) else str(f) for f, _ in top_funcs]
                func_times = [f[1]['total_time'] for f in top_funcs]
                
                # Create bar chart
                bars = plt.barh(func_names, func_times, alpha=0.7)
                
                # Add time labels
                for i, bar in enumerate(bars):
                    plt.text(
                        bar.get_width() + 0.1,
                        bar.get_y() + bar.get_height()/2,
                        f"{func_times[i]:.2f}s",
                        va='center'
                    )
                
                plt.xlabel('Time (seconds)')
                plt.title('Top 10 Functions by Execution Time')
                plt.grid(True, axis='x', alpha=0.3)
            else:
                plt.text(0.5, 0.5, "No profiling data available", ha='center', va='center')
                plt.axis('off')
            
            # Optimize visualization for memory
            plt.tight_layout()
            fig = plt.gcf()
            self.optimize_visualization(fig)
            
            # Save figure if path provided
            if output_path:
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                plt.close()
                return output_path
            else:
                # Generate default path
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(current_dir, "profiling")
                os.makedirs(output_dir, exist_ok=True)
                default_path = os.path.join(output_dir, f"resources_{timestamp}.png")
                plt.savefig(default_path, dpi=100, bbox_inches='tight')
                plt.close()
                return default_path
                
        except Exception as e:
            logger.error(f"Error creating resource visualization: {e}")
            plt.close()
            return None
    
    def cleanup(self):
        """Clean up resources and stop monitoring."""
        if hasattr(self, 'monitoring_active') and self.monitoring_active:
            self._stop_monitoring()
        
        # Clear caches
        self.data_cache.clear()
        self.model_cache.clear()
        
        # Optimize memory
        self.optimize_memory_usage(aggressive=True)
        
        logger.info("Performance optimizer cleaned up")


# Decorator for easy profiling
def profile_function(name=None):
    """
    Decorator for profiling function execution time.
    
    Args:
        name (str, optional): Optional custom name for the function
        
    Returns:
        Decorated function
    """
    # Create an optimizer instance if needed
    optimizer = None
    
    def decorator(func):
        nonlocal optimizer
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal optimizer
            
            # Create optimizer on first use
            if optimizer is None:
                optimizer = PerformanceOptimizer()
            
            # Apply profiling
            profiled_func = optimizer.profile(name=name)(func)
            return profiled_func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Create global optimizer instance
global_optimizer = None

def get_optimizer(config=None):
    """
    Get or create a global performance optimizer instance.
    
    Args:
        config (dict, optional): Configuration options
        
    Returns:
        PerformanceOptimizer: Performance optimizer instance
    """
    global global_optimizer
    
    if global_optimizer is None:
        global_optimizer = PerformanceOptimizer(config)
    
    return global_optimizer


if __name__ == "__main__":
    # Example usage
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Performance optimization")
    parser.add_argument('--profile', action='store_true', help='Run profiling tests')
    parser.add_argument('--report', action='store_true', help='Generate profiling report')
    parser.add_argument('--visualize', action='store_true', help='Generate resource visualization')
    
    args = parser.parse_args()
    
    optimizer = PerformanceOptimizer()
    
    # Start monitoring thread
    optimizer.config['BACKGROUND_MONITORING'] = True
    optimizer._start_monitoring()
    
    # Example profiled functions
    @optimizer.profile
    def slow_function():
        time.sleep(1.0)
        return "Done"
    
    @optimizer.profile
    def fast_function():
        time.sleep(0.05)
        return "Done"
    
    if args.profile:
        print("Running profiling tests...")
        
        # Run functions multiple times
        for _ in range(3):
            slow_function()
            
        for _ in range(20):
            fast_function()
        
        # Simulate memory allocations if GPU is available
        if torch.cuda.is_available():
            @optimizer.profile(name="tensor_allocation")
            def allocate_tensors():
                tensors = []
                for _ in range(5):
                    t = torch.rand(1000, 1000, device="cuda")
                    tensors.append(t)
                time.sleep(0.5)
                return tensors
            
            tensors = allocate_tensors()
            time.sleep(1.0)
            del tensors
            optimizer.optimize_memory_usage()
    
    # Keep monitoring for a bit
    print("Monitoring system...")
    time.sleep(5)
    
    if args.report:
        print("Generating profiling report...")
        report_path = optimizer.save_profiling_report()
        print(f"Report saved to: {report_path}")
    
    if args.visualize:
        print("Generating resource visualization...")
        viz_path = optimizer.visualize_system_resources()
        if viz_path:
            print(f"Visualization saved to: {viz_path}")
    
    # Clean up
    optimizer.cleanup()
    print("Done") 