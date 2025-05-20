#!/usr/bin/env python
"""
Test Runner for BTC-AI

This script runs all tests or specific test suites based on command line arguments.
Usage:
    python run_tests.py [--unit] [--integration] [--e2e] [--all]
    
If no arguments are provided, all tests will be run.
"""

import os
import sys
import unittest
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_run.log')
    ]
)
logger = logging.getLogger('test_runner')

def discover_and_run_tests(test_dir: str, pattern: str = 'test_*.py') -> unittest.TestResult:
    """
    Discover and run tests in the specified directory.
    
    Args:
        test_dir (str): Directory containing test files
        pattern (str): Pattern to match test files
        
    Returns:
        unittest.TestResult: Results of the test run
    """
    logger.info(f"Running tests in {test_dir}")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

def main():
    """Main function to run tests based on command line arguments."""
    parser = argparse.ArgumentParser(description='Run BTC-AI tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--e2e', action='store_true', help='Run end-to-end tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    # If no arguments provided, run all tests
    run_all = args.all or not (args.unit or args.integration or args.e2e)
    
    # Get the base test directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    results = []
    
    if args.unit or run_all:
        unit_result = discover_and_run_tests(os.path.join(base_dir, 'unit'))
        results.append(('Unit Tests', unit_result))
        
    if args.integration or run_all:
        integration_result = discover_and_run_tests(os.path.join(base_dir, 'integration'))
        results.append(('Integration Tests', integration_result))
        
    if args.e2e or run_all:
        e2e_result = discover_and_run_tests(os.path.join(base_dir, 'e2e'))
        results.append(('E2E Tests', e2e_result))
    
    # Print summary
    print("\nTest Summary:")
    print("=" * 50)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_type, result in results:
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        
        total_tests += tests_run
        total_failures += failures
        total_errors += errors
        
        print(f"\n{test_type}:")
        print(f"  Tests Run: {tests_run}")
        print(f"  Failures: {failures}")
        print(f"  Errors: {errors}")
        
        if failures > 0 or errors > 0:
            print("\nFailures and Errors:")
            for failure in result.failures:
                print(f"\nFAILURE: {failure[0]}")
                print(failure[1])
            for error in result.errors:
                print(f"\nERROR: {error[0]}")
                print(error[1])
    
    print("\nOverall Summary:")
    print(f"Total Tests Run: {total_tests}")
    print(f"Total Failures: {total_failures}")
    print(f"Total Errors: {total_errors}")
    
    # Return non-zero exit code if any tests failed
    if total_failures > 0 or total_errors > 0:
        sys.exit(1)

if __name__ == '__main__':
    main() 