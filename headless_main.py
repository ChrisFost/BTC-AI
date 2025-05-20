#!/usr/bin/env python3
"""
Headless version of main.py for testing

This script mocks the PySimpleGUI module to prevent UI windows from opening
while still allowing the main application code to execute for testing.
"""

import os
import sys
import unittest.mock
import importlib
import traceback

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Create a more comprehensive mock for matplotlib
class MockMatplotlib:
    def __init__(self):
        self.pyplot = unittest.mock.MagicMock()
        self.figure = unittest.mock.MagicMock()
        self.axes = unittest.mock.MagicMock()
        self.colors = unittest.mock.MagicMock()
        self.cm = unittest.mock.MagicMock()
        self.gridspec = unittest.mock.MagicMock()
        
    def __getattr__(self, name):
        # Create a mock for any attribute that doesn't exist
        mock = unittest.mock.MagicMock()
        setattr(self, name, mock)
        return mock

# Create matplotlib mock package
matplotlib_mock = MockMatplotlib()
sys.modules['matplotlib'] = matplotlib_mock
sys.modules['matplotlib.pyplot'] = matplotlib_mock.pyplot
sys.modules['matplotlib.figure'] = matplotlib_mock.figure
sys.modules['matplotlib.colors'] = matplotlib_mock.colors
sys.modules['matplotlib.cm'] = matplotlib_mock.cm
sys.modules['matplotlib.gridspec'] = matplotlib_mock.gridspec

# Mock the backends module
backends_mock = unittest.mock.MagicMock()
backends_mock.backend_tkagg = unittest.mock.MagicMock()
backends_mock.backend_tkagg.FigureCanvasTkAgg = unittest.mock.MagicMock()
sys.modules['matplotlib.backends'] = backends_mock
sys.modules['matplotlib.backends.backend_tkagg'] = backends_mock.backend_tkagg

# Mock PySimpleGUI before importing any UI-based modules
sys.modules['PySimpleGUI'] = unittest.mock.MagicMock()
import PySimpleGUI as sg

# Create a mock Window class
sg.Window = unittest.mock.MagicMock()
mock_window = unittest.mock.MagicMock()
sg.Window.return_value = mock_window

# Make window.read() return window closed after a few calls to simulate a short run
read_call_count = 0
def mock_read(*args, **kwargs):
    global read_call_count
    read_call_count += 1
    if read_call_count <= 3:
        return ("DUMMY_EVENT", {})
    else:
        return (sg.WIN_CLOSED, None)
        
mock_window.read.side_effect = mock_read

# Create a proper AllKeysDict property for the window
mock_window.AllKeysDict = {}

# Mock other commonly used sg functions
sg.popup = unittest.mock.MagicMock()
sg.popup_error = unittest.mock.MagicMock()
sg.popup_yes_no = unittest.mock.MagicMock(return_value="Yes")
sg.popup_ok = unittest.mock.MagicMock()
sg.WIN_CLOSED = "WIN_CLOSED"
sg.TIMEOUT_KEY = "__TIMEOUT__"
sg.Push = unittest.mock.MagicMock()
sg.Column = unittest.mock.MagicMock()
sg.Text = unittest.mock.MagicMock()
sg.Button = unittest.mock.MagicMock()
sg.Input = unittest.mock.MagicMock()
sg.Checkbox = unittest.mock.MagicMock()
sg.Radio = unittest.mock.MagicMock()
sg.Slider = unittest.mock.MagicMock()
sg.TabGroup = unittest.mock.MagicMock()
sg.Tab = unittest.mock.MagicMock()
sg.theme = unittest.mock.MagicMock()

try:
    print("Importing main module with mocked UI...")
    main_module = importlib.import_module("src.ui.main")
    
    # Override main function to add our own exit handling
    original_main = main_module.main
    
    def mock_main():
        try:
            original_main()
        except Exception as e:
            print(f"Error in main function: {e}")
            traceback.print_exc()
        print("Headless main execution completed")
    
    main_module.main = mock_main
    
    # Run the main function
    print("Running main in headless mode...")
    main_module.main()
    
except ImportError as e:
    print(f"Failed to import main module: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error during execution: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Headless execution completed successfully") 