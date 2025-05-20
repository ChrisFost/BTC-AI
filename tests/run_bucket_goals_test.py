#!/usr/bin/env python
"""
Script to run the Bucket Goals UI Integration tests.

This script runs both the unit tests for bucket goals UI interaction
and the integration tests for bucket goals functionality.
"""

import sys
import os
import unittest
import logging
from datetime import datetime

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Set up logging
log_dir = os.path.join(project_root, "tests", "logs")
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"bucket_goals_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("bucket_goals_test")

def run_tests():
    """Run all bucket goals tests."""
    logger.info("Starting Bucket Goals UI Integration Tests")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Print the current path and project root for debugging
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Project root: {project_root}")
    
    # First, check if we can import the necessary modules
    try:
        # Attempt to import TradeConfig first to isolate any issues
        from src.utils.trade_config import TradeConfig, trade_config
        logger.info("Successfully imported TradeConfig and trade_config instance")
    except ImportError as e:
        logger.error(f"Error importing TradeConfig: {e}")
        logger.warning("Running tests may fail due to TradeConfig import issues")
    
    # Add tests from test_bucket_goals_ui.py
    try:
        from tests.unit.test_bucket_goals_ui import TestBucketGoalsUIIntegration
        unit_tests = unittest.defaultTestLoader.loadTestsFromTestCase(TestBucketGoalsUIIntegration)
        test_suite.addTest(unit_tests)
        logger.info(f"Added {unit_tests.countTestCases()} unit tests")
    except ImportError as e:
        logger.error(f"Error importing unit tests: {e}")
    
    # Add tests from test_bucket_goals_integration.py
    try:
        from tests.integration.test_bucket_goals_integration import TestBucketGoalsIntegration
        integration_tests = unittest.defaultTestLoader.loadTestsFromTestCase(TestBucketGoalsIntegration)
        test_suite.addTest(integration_tests)
        logger.info(f"Added {integration_tests.countTestCases()} integration tests")
    except ImportError as e:
        logger.error(f"Error importing integration tests: {e}")
    
    # If no tests were added, return False
    if test_suite.countTestCases() == 0:
        logger.error("No tests were added to the test suite")
        return False
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Log results
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    
    # Print errors and failures for debugging
    if result.errors:
        logger.error("Errors:")
        for test, error in result.errors:
            logger.error(f"{test}: {error}")
    
    if result.failures:
        logger.error("Failures:")
        for test, failure in result.failures:
            logger.error(f"{test}: {failure}")
    
    # Return True if all tests passed
    return len(result.errors) == 0 and len(result.failures) == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 