#!/usr/bin/env python
"""
Test script to verify the launcher approach works with our UI module.

This script tests:
1. Importing the UI module without launching the UI
2. Verifying headless mode works correctly 
3. Checking that we can access functions without infinite loops
"""

import os
import sys
import time
import importlib

def test_import_without_ui():
    """Test that we can import the UI module without launching the UI."""
    print("\n=== Testing import without UI launching ===")
    
    # Set headless mode environment variable
    os.environ['BTC_AI_HEADLESS'] = '1'
    
    try:
        print("Importing src.ui.main module...")
        main_module = importlib.import_module("src.ui.main")
        print("✓ Successfully imported src.ui.main without launching UI")
        
        # Verify we can access functions
        print("Accessing functions from the module...")
        if hasattr(main_module, 'main'):
            print("✓ Found main() function")
        else:
            print("✗ Could not find main() function")
            return False
        
        # Print additional diagnostics
        print(f"Module: {main_module}")
        print(f"Functions: {[f for f in dir(main_module) if callable(getattr(main_module, f)) and not f.startswith('__')]}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to import src.ui.main: {e}")
        return False

def test_launcher_script():
    """Test that we can use the launcher script approach."""
    print("\n=== Testing launcher script ===")
    
    # Set headless mode environment variable
    os.environ['BTC_AI_HEADLESS'] = '1'
    
    try:
        # Create the testing launcher code
        launcher_code = """
import os
os.environ['BTC_AI_HEADLESS'] = '1'
import importlib
main_module = importlib.import_module("src.ui.main")
if __name__ == "__main__":
    print("Launcher running in test mode")
    if hasattr(main_module, 'main'):
        main_module.main()
    else:
        print("No main() function found")
"""
        # Save the launcher code to a temporary file
        with open("test_launcher_temp.py", "w") as f:
            f.write(launcher_code)
        
        # Run it in a subprocess
        import subprocess
        print("Running the launcher script...")
        result = subprocess.run([sys.executable, "test_launcher_temp.py"], 
                                capture_output=True, text=True, timeout=10)
        
        # Output results
        print(f"Return code: {result.returncode}")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        # Clean up the temp file
        import os
        if os.path.exists("test_launcher_temp.py"):
            os.remove("test_launcher_temp.py")
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error testing launcher: {e}")
        return False

def main():
    """Run all tests for the launcher approach."""
    print("=== BTC-AI Launcher Tests ===")
    
    # Run tests
    import_success = test_import_without_ui()
    launcher_success = test_launcher_script()
    
    # Report results
    print("\n=== Test Results ===")
    print(f"Import without UI: {'PASS' if import_success else 'FAIL'}")
    print(f"Launcher script:   {'PASS' if launcher_success else 'FAIL'}")
    
    # Return appropriate exit code
    return 0 if import_success and launcher_success else 1

if __name__ == "__main__":
    sys.exit(main()) 