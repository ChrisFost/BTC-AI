"""
Performance Monitor Module

This module provides system performance monitoring capabilities for the
BTC-AI Training Interface.
"""

import os
import sys
import threading
import time
import datetime
import logging
import PySimpleGUI as sg
from typing import Dict, Any, Optional, Tuple
import pynvml

# Set up logger
try:
    from src.utils.log_manager import get_logger
    logger = get_logger('performance_monitor')
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('performance_monitor')

# Initialize GPU monitoring variables at the module level
gputil_available = False
pynvml_available = False

# Try to import psutil for system monitoring
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
    logger.warning("psutil not available. Performance monitoring will be limited.")

# Try to import GPU monitoring libraries and set flags
try:
    if 'linux' in sys.platform:
        try:
            import GPUtil
            gputil_available = True # Set flag if import succeeds
            logger.debug("GPUtil library found for Linux GPU monitoring.")
        except ImportError:
            logger.warning("GPUtil not available for Linux GPU monitoring")
            gputil_available = False # Ensure flag is False if import fails
    elif sys.platform == 'win32':
        try:
            # Keep pynvml import here
            pynvml.nvmlInit()
            pynvml_available = True # Set flag if init succeeds
            logger.debug("pynvml library initialized for Windows GPU monitoring.")
        except Exception as e:
            logger.warning(f"NVIDIA Management Library (pynvml) not available or failed to initialize: {e}")
            pynvml_available = False # Ensure flag is False if init fails
except Exception as e:
    logger.error(f"Error during GPU monitoring library initialization: {e}")
    gputil_available = False # Ensure flags are False on any other exception
    pynvml_available = False


class PerformanceMonitor:
    """
    Monitors system performance metrics such as CPU usage, memory, disk I/O,
    and GPU utilization if available.
    """
    
    def __init__(self, app_state=None):
        """
        Initialize the performance monitor.
        
        Args:
            app_state: Application state reference
        """
        self.app_state = app_state
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.metrics = {
            'cpu': 0.0,
            'memory': 0.0,
            'memory_used': 0,
            'memory_total': 0,
            'disk_io': 0.0,
            'gpu_util': 0.0,
            'gpu_memory': 0.0,
            'training_elapsed': 0,
            'last_update': 0
        }
        
        # Use module-level flags directly
        self.can_monitor_system = psutil_available
        self.can_monitor_gpu = gputil_available or pynvml_available
        
        logger.info(f"Performance monitor initialized (System: {self.can_monitor_system}, GPU: {self.can_monitor_gpu})")
    
    def start_monitoring(self, window, interval=2.0):
        """
        Start the performance monitoring thread.
        
        Args:
            window: The main application window for updating UI
            interval: Update interval in seconds
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread already running")
            return False
        
        if not self.can_monitor_system:
            logger.warning("System monitoring not available")
            return False
        
        # Reset stop event
        self.stop_event.clear()
        
        # Define monitoring function
        def monitor_loop():
            last_io_counters = None
            last_io_time = time.time()
            
            while not self.stop_event.is_set():
                try:
                    # Update system metrics
                    self._update_system_metrics()
                    
                    # Update disk I/O
                    current_time = time.time()
                    if last_io_counters:
                        # Calculate disk I/O rate
                        io_counters = psutil.disk_io_counters()
                        read_bytes = io_counters.read_bytes - last_io_counters.read_bytes
                        write_bytes = io_counters.write_bytes - last_io_counters.write_bytes
                        elapsed = current_time - last_io_time
                        
                        if elapsed > 0:
                            # Calculate total I/O in MB/s
                            io_rate = (read_bytes + write_bytes) / (1024 * 1024 * elapsed)
                            self.metrics['disk_io'] = io_rate
                        
                        last_io_counters = io_counters
                        last_io_time = current_time
                    else:
                        # Initialize disk I/O counters
                        last_io_counters = psutil.disk_io_counters()
                        last_io_time = current_time
                    
                    # Update GPU metrics if available
                    if self.can_monitor_gpu:
                        self._update_gpu_metrics()
                    
                    # Update training elapsed time if training
                    self._update_training_time()
                    
                    # Send metrics to UI
                    if window:
                        window.write_event_value("-PERFORMANCE-UPDATE-", self.metrics.copy())
                    
                    # Update timestamp
                    self.metrics['last_update'] = time.time()
                    
                except Exception as e:
                    logger.error(f"Error in performance monitoring: {e}")
                
                # Sleep for the specified interval
                time.sleep(interval)
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Performance monitoring started with {interval}s interval")
        return True
    
    def stop_monitoring(self):
        """
        Stop the performance monitoring thread.
        
        Returns:
            bool: True if monitoring was stopped, False if not running
        """
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            # logger.warning("No active monitoring thread to stop") # Commented out - can be noisy
            return False
        
        # Set stop event and wait for thread to terminate
        self.stop_event.set()
        self.monitoring_thread.join(timeout=2.0)
        
        # Cleanup pynvml if it was initialized
        if pynvml_available and sys.platform == 'win32':
            try:
                pynvml.nvmlShutdown()
                logger.debug("Shutdown pynvml")
            except pynvml.NVMLError as nvml_error:
                # Don't log error if already shut down or never initialized properly
                if "NVML_ERROR_UNINITIALIZED" not in str(nvml_error) and "NVML_ERROR_LIBRARY_NOT_FOUND" not in str(nvml_error):
                    logger.error(f"Error shutting down pynvml: {nvml_error}")
            except Exception as e:
                 logger.error(f"Unexpected error shutting down pynvml: {e}")
                
        logger.info("Performance monitoring stopped")
        return True
    
    def _update_system_metrics(self):
        """Update system metrics (CPU, memory)."""
        if not psutil_available:
            return
        
        try:
            # Get CPU usage (averaged across all cores)
            self.metrics['cpu'] = psutil.cpu_percent(interval=None)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            self.metrics['memory'] = memory.percent
            self.metrics['memory_used'] = memory.used // (1024 * 1024)  # MB
            self.metrics['memory_total'] = memory.total // (1024 * 1024)  # MB
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def _update_gpu_metrics(self):
        """Update GPU metrics if available."""
        # Check module-level flags
        if not (gputil_available or pynvml_available):
            return # No GPU monitoring available
            
        try:
            if gputil_available and 'linux' in sys.platform:
                # Using GPUtil (Linux primarily)
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.metrics['gpu_util'] = gpu.load * 100.0
                    self.metrics['gpu_memory'] = gpu.memoryUtil * 100.0
            
            elif pynvml_available and sys.platform == 'win32':
                # Using PYNVML (Windows primarily)
                try:
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.metrics['gpu_util'] = util.gpu
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        if memory.total > 0:
                            self.metrics['gpu_memory'] = (memory.used / memory.total) * 100.0
                except pynvml.NVMLError as nvml_error:
                    # Handle specific NVMLError, e.g., if driver/GPU is not running
                    logger.warning(f"Could not get NVIDIA GPU metrics: {nvml_error}")
                    # Reset GPU metrics if they can't be read
                    self.metrics['gpu_util'] = 0.0
                    self.metrics['gpu_memory'] = 0.0
                except Exception as e:
                    logger.error(f"Unexpected error getting NVIDIA GPU metrics: {e}")
                    self.metrics['gpu_util'] = 0.0
                    self.metrics['gpu_memory'] = 0.0
        except Exception as e:
            logger.error(f"General error updating GPU metrics: {e}")
            # Reset metrics on general failure
            self.metrics['gpu_util'] = 0.0
            self.metrics['gpu_memory'] = 0.0
    
    def _update_training_time(self):
        """Update training time if training is in progress."""
        # This can be expanded to check if training is actually running
        # For now we'll just use a placeholder
        training_start_time = getattr(self.app_state, 'training_start_time', None)
        if training_start_time:
            self.metrics['training_elapsed'] = int(time.time() - training_start_time)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            dict: Current performance metrics
        """
        return self.metrics.copy()
    
    def update_system_status(self, window, is_training: bool, 
                             training_start_time: Optional[float] = None,
                             last_status_update: Optional[float] = None) -> float:
        """
        Update system status in the UI.
        
        Args:
            window: PySimpleGUI window
            is_training: Whether training is in progress
            training_start_time: When training started (timestamp)
            last_status_update: Timestamp of last status update
            
        Returns:
            float: Current timestamp
        """
        current_time = time.time()
        
        # Check if we should update yet (throttle updates)
        if last_status_update and current_time - last_status_update < 1.0:
            return last_status_update
        
        try:
            if window and "-STATUS-INFO-" in window.AllKeysDict:
                metrics = self.get_current_metrics()
                
                # Format training time if training
                training_time_str = ""
                if is_training and training_start_time:
                    elapsed = int(current_time - training_start_time)
                    hours, remainder = divmod(elapsed, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    training_time_str = f" | Training: {hours:02}:{minutes:02}:{seconds:02}"
                
                # Format system metrics
                cpu_str = f"CPU: {metrics['cpu']:.1f}%" if 'cpu' in metrics else ""
                mem_str = f"Mem: {metrics['memory']:.1f}%" if 'memory' in metrics else ""
                gpu_str = f"GPU: {metrics['gpu_util']:.1f}%" if 'gpu_util' in metrics and metrics['gpu_util'] > 0 else ""
                
                # Combine status elements
                status_str = " | ".join(filter(None, [cpu_str, mem_str, gpu_str, training_time_str]))
                
                # Update status display
                window["-STATUS-INFO-"].update(status_str)
            
        except Exception as e:
            logger.error(f"Error updating system status: {e}")
        
        return current_time


# Singleton pattern
_performance_monitor_instance = None

def get_performance_monitor(app_state=None):
    """
    Get or create the performance monitor singleton.
    
    Args:
        app_state: Application state reference
        
    Returns:
        PerformanceMonitor: Performance monitor instance
    """
    global _performance_monitor_instance
    
    if _performance_monitor_instance is None:
        _performance_monitor_instance = PerformanceMonitor(app_state)
    
    return _performance_monitor_instance 