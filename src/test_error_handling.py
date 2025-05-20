#!/usr/bin/env python
"""
Test Error Handling and Persistent Logging

This script tests the error handling and persistent logging functionality.
"""

import os
import sys
import logging
import PySimpleGUI as sg

# Get the base directory (project root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add project root to path to ensure imports work
sys.path.insert(0, project_root)

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_error_handling")

def main():
    """Main test function"""
    print("BTC-AI Error Handling and Persistent Logging Test")
    print("=" * 50)
    
    # Test persistent logger
    test_persistent_logger()
    
    # Test error handler
    test_error_handler()
    
    # Test integrated functionality with GUI
    test_gui_error_handling()

def test_persistent_logger():
    """Test the persistent logger functionality"""
    print("\nTesting Persistent Logger...")
    
    try:
        from src.utils.persistent_logger import (
            log_persistent_error, 
            get_log_directory_path,
            create_log_locations_file
        )
        
        # Log a test error
        log_persistent_error(
            Exception("Test error for persistent logger"), 
            "test_persistent_logger", 
            "low",
            {"test_data": "This is test data"}
        )
        
        # Get log directory path
        log_dir = get_log_directory_path()
        print(f"Log directory: {log_dir}")
        
        # Create log locations file
        locations_file = create_log_locations_file()
        print(f"Log locations file created: {locations_file}")
        
        print("Persistent logger test completed successfully!")
    except ImportError:
        print("Persistent logger module not available")
    except Exception as e:
        print(f"Error testing persistent logger: {e}")

def test_error_handler():
    """Test the error handler functionality"""
    print("\nTesting Error Handler...")
    
    try:
        from src.ui.error_handler import (
            handle_error, 
            ErrorSeverity,
            get_last_error,
            get_error_history,
            clear_error_history
        )
        
        # Clear error history
        clear_error_history()
        
        # Test with different error types
        try:
            # Test file not found error
            with open("non_existent_file.txt", "r") as f:
                pass
        except Exception as e:
            handle_error(e, "test_file_not_found", None, None, {"file_path": "non_existent_file.txt"})
        
        try:
            # Test value error
            value = int("not_a_number")
        except Exception as e:
            handle_error(e, "test_value_error")
        
        try:
            # Test key error
            test_dict = {}
            value = test_dict["missing_key"]
        except Exception as e:
            handle_error(e, "test_key_error")
        
        # Get last error
        last_error = get_last_error()
        print(f"Last error: {last_error['message']}")
        
        # Get error history
        history = get_error_history()
        print(f"Error history count: {len(history)}")
        
        print("Error handler test completed successfully!")
    except ImportError:
        print("Error handler module not available")
    except Exception as e:
        print(f"Error testing error handler: {e}")

def test_gui_error_handling():
    """Test GUI error handling"""
    print("\nTesting GUI Error Handling...")
    
    try:
        from src.ui.error_handler import handle_error
        
        # Create a simple GUI window for testing
        layout = [
            [sg.Text("Error Handling Test")],
            [sg.Button("Test File Error"), sg.Button("Test Value Error"), sg.Button("Test Custom Error")],
            [sg.Button("Close")]
        ]
        
        window = sg.Window("Error Handling Test", layout, finalize=True)
        
        # Event loop
        while True:
            event, values = window.read()
            
            if event in (None, "Close"):
                break
            elif event == "Test File Error":
                try:
                    with open("non_existent_file.txt", "r") as f:
                        pass
                except Exception as e:
                    handle_error(e, "test_file_error", window, None, {"file_path": "non_existent_file.txt"})
            elif event == "Test Value Error":
                try:
                    value = int("not_a_number")
                except Exception as e:
                    handle_error(e, "test_value_error", window)
            elif event == "Test Custom Error":
                class CustomError(Exception):
                    pass
                
                try:
                    raise CustomError("This is a custom error for testing")
                except Exception as e:
                    handle_error(e, "test_custom_error", window)
        
        window.close()
        print("GUI error handling test completed!")
    except ImportError:
        print("Error handler module or PySimpleGUI not available")
    except Exception as e:
        print(f"Error testing GUI error handling: {e}")
        if 'window' in locals() and window:
            window.close()

if __name__ == "__main__":
    main() 