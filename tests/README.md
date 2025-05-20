# BTC-AI Tests

This directory contains tests for the BTC-AI application. The tests are organized into three categories:

## Test Categories

1. **Unit Tests** (`/unit`):
   - Tests for individual components and functions
   - Fast execution, no external dependencies
   - Located in `tests/unit/`

2. **Integration Tests** (`/integration`):
   - Tests for component interactions
   - May require some external dependencies
   - Located in `tests/integration/`

3. **End-to-End Tests** (`/e2e`):
   - Full system tests
   - Requires complete environment setup
   - Located in `tests/e2e/`

## Running Tests

Use the test runner script to execute tests:

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --unit        # Run only unit tests
python run_tests.py --integration # Run only integration tests
python run_tests.py --e2e        # Run only end-to-end tests

# Run multiple categories
python run_tests.py --unit --integration
```

## Test Logs

Test execution logs are stored in `test_run.log`. This file contains detailed information about test runs, including timestamps and error messages.

## Creating New Tests

1. Create a new test file in the appropriate directory
2. Name the file with a `test_` prefix (e.g., `test_setup_wizard.py`)
3. Inherit from `unittest.TestCase`
4. Use descriptive test method names starting with `test_`
5. Include docstrings explaining the test's purpose

Example:

```python
import unittest

class TestMyFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass
        
    def tearDown(self):
        """Clean up after each test method."""
        pass
        
    def test_my_function(self):
        """Test that my_function behaves correctly."""
        result = my_function()
        self.assertTrue(result)
```

## Best Practices

1. Keep tests independent and isolated
2. Clean up resources in `tearDown`
3. Use meaningful assertions
4. Mock external dependencies
5. Follow the AAA pattern:
   - Arrange (set up test data)
   - Act (execute the code being tested)
   - Assert (verify the results)

## Continuous Integration

The test runner returns:
- Exit code 0: All tests passed
- Exit code 1: One or more tests failed

This enables integration with CI/CD pipelines. 