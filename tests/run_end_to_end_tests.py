#!/usr/bin/env python
"""
End-to-End Test Runner

This script runs all end-to-end tests for the BTC AI Trading system:
1. Menu Integration Test - Tests the menu interface and its interaction with training
2. Progressive Pipeline Test - Tests the progressive training pipeline

Running this script provides a comprehensive validation of the entire system.
"""

import os
import sys
import time
import logging
import subprocess
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'e2e_test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('e2e_test_runner')

# Get the current directory and project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# Add project root to path to ensure imports work correctly
sys.path.insert(0, project_root)

def run_menu_integration_test(verbose=False):
    """Run the menu integration test."""
    logger.info("Starting Menu Integration Test...")
    
    # Update to use the test in the integration folder
    test_script = os.path.join(current_dir, "integration", "test_menu_fixes.py")
    
    if not os.path.exists(test_script):
        logger.error(f"Test script not found: {test_script}")
        return False
    
    cmd = [sys.executable, test_script]
    if verbose:
        cmd.append("-v")
    
    try:
        # Run the test and capture output
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream the output
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("Menu Integration Test passed!")
            return True
        else:
            logger.error(f"Menu Integration Test failed with return code {return_code}")
            return False
    
    except Exception as e:
        logger.error(f"Error running Menu Integration Test: {e}")
        return False

def run_progressive_pipeline_test(episodes=3, view_time=5, verbose=False):
    """Run the progressive pipeline test."""
    logger.info("Starting Progressive Pipeline Test...")
    
    # Update to use the test in the end_to_end folder
    test_script = os.path.join(current_dir, "end_to_end", "test_progressive_pipeline.py")
    
    if not os.path.exists(test_script):
        logger.error(f"Test script not found: {test_script}")
        return False
    
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update Models path to be under project root
    models_dir = os.path.join(project_root, "Models", f"e2e_test_{timestamp}")
    
    cmd = [
        sys.executable, 
        test_script,
        "--episodes", str(episodes),
        "--view-time", str(view_time),
        "--models-dir", models_dir
    ]
    
    if verbose:
        cmd.append("--debug")
    
    try:
        # Run the test and capture output
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream the output
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("Progressive Pipeline Test passed!")
            return True
        else:
            logger.error(f"Progressive Pipeline Test failed with return code {return_code}")
            return False
    
    except Exception as e:
        logger.error(f"Error running Progressive Pipeline Test: {e}")
        return False

def run_backtesting_integration_test(verbose=False):
    """Run the backtesting integration test."""
    logger.info("Starting Backtesting Integration Test...")
    
    test_script = os.path.join(current_dir, "integration", "test_backtesting_integration.py")
    
    if not os.path.exists(test_script):
        logger.error(f"Test script not found: {test_script}")
        return False
    
    cmd = [sys.executable, test_script]
    if verbose:
        cmd.append("-v")
    
    try:
        # Run the test and capture output
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream the output
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("Backtesting Integration Test passed!")
            return True
        else:
            logger.error(f"Backtesting Integration Test failed with return code {return_code}")
            return False
    
    except Exception as e:
        logger.error(f"Error running Backtesting Integration Test: {e}")
        return False

def run_cross_bucket_transfer_test(verbose=False):
    """Run the cross bucket knowledge transfer test."""
    logger.info("Starting Cross Bucket Transfer Test...")
    
    test_script = os.path.join(current_dir, "integration", "test_cross_bucket_transfer.py")
    
    if not os.path.exists(test_script):
        logger.error(f"Test script not found: {test_script}")
        return False
    
    cmd = [sys.executable, test_script]
    if verbose:
        cmd.append("-v")
    
    try:
        # Run the test and capture output
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream the output
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("Cross Bucket Transfer Test passed!")
            return True
        else:
            logger.error(f"Cross Bucket Transfer Test failed with return code {return_code}")
            return False
    
    except Exception as e:
        logger.error(f"Error running Cross Bucket Transfer Test: {e}")
        return False

def create_test_report(test_results):
    """Create a test report summary."""
    report_file = os.path.join(current_dir, f"e2e_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_file, 'w') as f:
        f.write("==================================\n")
        f.write("BTC AI End-to-End Test Report\n")
        f.write("==================================\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Test Results Summary:\n")
        f.write("---------------------\n")
        
        all_passed = True
        for test_name, result in test_results.items():
            f.write(f"{test_name}: {'PASSED' if result else 'FAILED'}\n")
            if not result:
                all_passed = False
                
        f.write(f"Overall Result: {'PASSED' if all_passed else 'FAILED'}\n\n")
        
        f.write("Detailed Results:\n")
        f.write("---------------------\n")
        f.write("See log files for detailed test output.\n\n")
        
        f.write("Next Steps:\n")
        f.write("---------------------\n")
        if all_passed:
            f.write("All tests passed. The system is ready for use.\n")
        else:
            f.write("The following issues need to be addressed:\n")
            for test_name, result in test_results.items():
                if not result:
                    f.write(f"- Fix issues with the {test_name}.\n")
    
    logger.info(f"Test report generated: {report_file}")
    return report_file

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run end-to-end tests for the BTC AI Trading system")
    parser.add_argument("--menu-only", action="store_true", help="Run only the menu integration test")
    parser.add_argument("--pipeline-only", action="store_true", help="Run only the progressive pipeline test")
    parser.add_argument("--backtesting-only", action="store_true", help="Run only the backtesting integration test")
    parser.add_argument("--cross-bucket-only", action="store_true", help="Run only the cross bucket transfer test")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes per bucket for pipeline test")
    parser.add_argument("--view-time", type=int, default=5, help="Dashboard view time in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    logger.info("Starting End-to-End Test Suite")
    start_time = time.time()
    
    # Track test results
    test_results = {
        "Menu Integration Test": True,
        "Progressive Pipeline Test": True,
        "Backtesting Integration Test": True,
        "Cross Bucket Transfer Test": True
    }
    
    # Run tests based on flags
    if args.menu_only:
        test_results["Menu Integration Test"] = run_menu_integration_test(args.verbose)
        # Skip other tests
        test_results.pop("Progressive Pipeline Test")
        test_results.pop("Backtesting Integration Test")
        test_results.pop("Cross Bucket Transfer Test")
    elif args.pipeline_only:
        test_results["Progressive Pipeline Test"] = run_progressive_pipeline_test(args.episodes, args.view_time, args.verbose)
        # Skip other tests
        test_results.pop("Menu Integration Test")
        test_results.pop("Backtesting Integration Test")
        test_results.pop("Cross Bucket Transfer Test")
    elif args.backtesting_only:
        test_results["Backtesting Integration Test"] = run_backtesting_integration_test(args.verbose)
        # Skip other tests
        test_results.pop("Menu Integration Test")
        test_results.pop("Progressive Pipeline Test")
        test_results.pop("Cross Bucket Transfer Test")
    elif args.cross_bucket_only:
        test_results["Cross Bucket Transfer Test"] = run_cross_bucket_transfer_test(args.verbose)
        # Skip other tests
        test_results.pop("Menu Integration Test")
        test_results.pop("Progressive Pipeline Test")
        test_results.pop("Backtesting Integration Test")
    else:
        # Run all tests
        test_results["Menu Integration Test"] = run_menu_integration_test(args.verbose)
        test_results["Progressive Pipeline Test"] = run_progressive_pipeline_test(args.episodes, args.view_time, args.verbose)
        test_results["Backtesting Integration Test"] = run_backtesting_integration_test(args.verbose)
        test_results["Cross Bucket Transfer Test"] = run_cross_bucket_transfer_test(args.verbose)
    
    # Generate test report
    report_file = create_test_report(test_results)
    
    # Calculate execution time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    logger.info(f"Tests completed in {int(minutes)} minutes {int(seconds)} seconds")
    logger.info(f"Test report available at: {report_file}")
    
    # Return exit code based on test results
    if all(test_results.values()):
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("One or more tests failed. See report for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 