#!/usr/bin/env python
"""
Build script for the BTC-AI application.

This script:
1. Sets up environment variables to prevent UI components from launching
2. First tries building with a minimal spec to isolate issues
3. Then builds with the full spec if minimal build succeeds
4. Monitors system resources during the build process
5. Packages the application with necessary files
"""

import os
import sys
import shutil
import subprocess
import platform
import time
import threading
import psutil
from datetime import datetime

def log(message, level="INFO"):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def ensure_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        log(f"Created directory: {path}")

def kill_process_tree(process):
    """Kill a process and all its children recursively."""
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except:
                pass
        parent.kill()
    except:
        pass

def monitor_resources(stop_event):
    """Monitor system resources during the build."""
    log("Starting resource monitoring...")
    last_log = time.time()
    while not stop_event.is_set():
        if time.time() - last_log >= 20:  # Log every 20 seconds
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=1)
            mem_percent = process.memory_percent()
            
            log(f"Memory usage: {memory_info.rss / (1024*1024):.1f} MB ({mem_percent:.1f}%)")
            log(f"CPU usage: {cpu_percent:.1f}%")
            
            # Check for any Python processes that might be consuming lots of resources
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
                if 'python' in proc.info['name'].lower() and proc.info['pid'] != os.getpid():
                    if proc.info['cpu_percent'] > 50 or proc.info['memory_percent'] > 10:
                        cmd = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else 'Unknown'
                        log(f"High resource Python process: PID {proc.info['pid']}, "
                            f"CPU: {proc.info['cpu_percent']:.1f}%, MEM: {proc.info['memory_percent']:.1f}%"
                            f" - {cmd[:100]}", "WARNING")
            
            last_log = time.time()
        time.sleep(1)

def run_with_timeout(cmd, timeout=900, description="command"):  # 15 minutes timeout
    """Run a command with timeout protection."""
    start_time = time.time()
    
    # Start process
    log(f"Executing {description}: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    # Set up output collectors
    stdout_chunks = []
    stderr_chunks = []
    
    # Set up monitoring thread to show progress
    stop_monitoring = threading.Event()
    
    def monitor_process():
        last_log = time.time()
        while not stop_monitoring.is_set():
            current_time = time.time()
            elapsed = current_time - start_time
            if current_time - last_log >= 30:  # Log every 30 seconds
                log(f"{description} still running... (elapsed: {elapsed:.1f}s)")
                last_log = current_time
            time.sleep(1)
    
    monitor_thread = threading.Thread(target=monitor_process)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Read output in real-time
    def read_output(pipe, chunks):
        for line in pipe:
            chunks.append(line)
            print(line.rstrip())
            
    stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_chunks))
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_chunks))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    try:
        # Wait for process to complete or timeout
        process.wait(timeout=timeout)
        stop_monitoring.set()
        
        # Make sure we've read all output
        stdout_thread.join(2)
        stderr_thread.join(2)
        
        stdout = ''.join(stdout_chunks)
        stderr = ''.join(stderr_chunks)
        
        elapsed = time.time() - start_time
        log(f"{description} completed in {elapsed:.1f} seconds with exit code {process.returncode}")
        
        return process.returncode, stdout, stderr
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        log(f"{description} timed out after {elapsed:.1f} seconds", "ERROR")
        kill_process_tree(process)
        stop_monitoring.set()
        
        stdout = ''.join(stdout_chunks)
        stderr = ''.join(stderr_chunks)
        
        return 1, stdout, stderr + f"\nProcess timed out after {timeout} seconds"

def clean_temp_files():
    """Clean up temporary files that might be causing issues."""
    log("Cleaning temporary build files...")
    temp_files = ['minimal_launcher.py', 'btc_ai_launcher.py']
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                log(f"Removed temporary file: {file}")
            except Exception as e:
                log(f"Could not remove {file}: {e}", "WARNING")
    
    # Clean PyInstaller cache
    cache_dir = os.path.join(os.path.expanduser('~'), '.pyinstaller')
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            log(f"Cleaned PyInstaller cache: {cache_dir}")
        except Exception as e:
            log(f"Could not clean PyInstaller cache: {e}", "WARNING")

def run_build(spec_file, description, timeout=900):
    """Run PyInstaller with a specific spec file."""
    # PyInstaller command with verbose output
    pyinstaller_cmd = [
        'pyinstaller',
        '--clean',
        '--noconfirm',
        '--log-level=DEBUG',  # More verbose output
        spec_file
    ]
    
    # Run PyInstaller with timeout
    log(f"Starting PyInstaller build with {spec_file}...")
    returncode, stdout, stderr = run_with_timeout(
        pyinstaller_cmd, 
        timeout=timeout,
        description=description
    )
    
    if returncode != 0:
        log(f"PyInstaller failed with exit code {returncode}", "ERROR")
        
        # Check for common errors
        if "RecursionError" in stderr:
            log("Build failed due to recursion error - check for circular imports", "ERROR")
        elif "SyntaxError" in stderr:
            log("Build failed due to syntax error in one of the modules", "ERROR")
        elif "FileNotFoundError" in stderr:
            log("Build failed due to missing file - check the file paths in the spec", "ERROR")
        elif "ImportError" in stderr:
            log("Build failed due to import error - a module could not be found", "ERROR")
        elif "MemoryError" in stderr:
            log("Build failed due to memory error - try reducing the number of modules", "ERROR")
        
        # Look for any Python tracebacks in the output
        import re
        tracebacks = re.findall(r'Traceback \(most recent call last\):(.*?)(?=\n\n|\Z)', 
                               stderr, re.DOTALL)
        if tracebacks:
            log("Found Python tracebacks in the output:", "ERROR")
            for i, tb in enumerate(tracebacks[:3]):  # Show up to 3 tracebacks
                log(f"Traceback {i+1}: {tb[:500]}...", "ERROR")
        
        return False
    
    log(f"PyInstaller completed successfully with {spec_file}")
    return True

def main():
    """Main build process."""
    overall_start_time = time.time()
    
    # Get script directory
    script_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(script_dir)
    log(f"Working directory: {script_dir}")
    
    # Start resource monitoring
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_monitor,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # Set environment variables to prevent UI from launching
        os.environ['BTC_AI_HEADLESS'] = '1'
        os.environ['PYINSTALLER_BUILD'] = '1'
        log("Set environment variables to prevent UI from launching")
        
        # Clean up any temporary files from previous runs
        clean_temp_files()
        
        # Ensure output directories exist
        log("Creating necessary directories...")
        for directory in ['build', 'dist', 'Models', 'Logs', 'Cache', 'configs', 'data']:
            ensure_directory(os.path.join(script_dir, directory))
        
        # Check if Python modules can be imported
        log("Verifying critical imports...")
        try:
            step_start = time.time()
            import pandas
            import numpy
            log(f"Critical imports verified in {time.time() - step_start:.1f}s")
        except ImportError as e:
            log(f"Import verification failed: {e}", "ERROR")
            return 1
        
        # First try with minimal spec
        log("PHASE 1: Building with minimal spec first to isolate potential issues...")
        minimal_success = run_build(
            'minimal_spec.spec', 
            'Minimal build', 
            timeout=300  # 5 minutes should be enough for minimal build
        )
        
        if not minimal_success:
            log("Minimal build failed, stopping build process", "ERROR")
            return 1
        
        # If minimal spec succeeds, try the full spec
        log("PHASE 2: Building with full spec...")
        full_success = run_build(
            'simple_spec.spec', 
            'Full build', 
            timeout=1800  # 30 minutes for full build
        )
        
        if not full_success:
            log("Full build failed, but minimal build succeeded - likely an issue with imported modules", "ERROR")
            return 1
        
        # Copy additional files to dist directory
        dist_dir = os.path.join(script_dir, 'dist', 'BTC-AI')
        if not os.path.exists(dist_dir):
            log(f"Error: Expected distribution directory not found: {dist_dir}", "ERROR")
            return 1
        
        log(f"Preparing final distribution in {dist_dir}")
        
        # Create empty directories for runtime data if they don't exist
        for dir_name in ['Models', 'Logs', 'Cache', 'configs', 'data']:
            ensure_directory(os.path.join(dist_dir, dir_name))
        
        # Create a simple README in the dist directory
        log("Creating README file...")
        with open(os.path.join(dist_dir, 'README.txt'), 'w') as f:
            f.write(f"""BTC-AI Application
===============

Built on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Platform: {platform.platform()}
Python: {platform.python_version()}

To run the application, double-click on the BTC-AI executable.

Important folders:
- Models: Saved model files
- Logs: Application logs
- Cache: Temporary data cache
- configs: Configuration files
""")
        
        # Clean up temporary files
        clean_temp_files()
        
        total_time = time.time() - overall_start_time
        log(f"Build completed successfully in {total_time:.1f} seconds. Output in: {dist_dir}")
        return 0
        
    except Exception as e:
        import traceback
        log(f"Unhandled exception: {e}", "ERROR")
        traceback.print_exc()
        return 1
    finally:
        # Stop resource monitoring
        stop_monitor.set()
        monitor_thread.join(2)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        log(f"Unhandled exception: {e}", "ERROR")
        traceback.print_exc()
        sys.exit(1) 