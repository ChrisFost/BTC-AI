#!/usr/bin/env python3
"""
Headless test runner for BTC-AI tests

This script mocks UI components before running tests to prevent 
UI windows from opening during test execution.
"""

import os
import sys
import unittest.mock
import importlib
import argparse

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
        self.dates = unittest.mock.MagicMock()
        self.cbook = unittest.mock.MagicMock()
        self.ticker = unittest.mock.MagicMock()
        self.patches = unittest.mock.MagicMock()
        self.backend_bases = unittest.mock.MagicMock()
        self.lines = unittest.mock.MagicMock()
        
        # Add normalize_kwargs to cbook for seaborn
        self.cbook.normalize_kwargs = unittest.mock.MagicMock()
        
        # Add common dates functions
        self.dates.DateFormatter = unittest.mock.MagicMock()
        self.dates.AutoDateLocator = unittest.mock.MagicMock()
        self.dates.AutoDateFormatter = unittest.mock.MagicMock()
        self.dates.date2num = unittest.mock.MagicMock()
        self.dates.num2date = unittest.mock.MagicMock()
        
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
sys.modules['matplotlib.dates'] = matplotlib_mock.dates
sys.modules['matplotlib.cbook'] = matplotlib_mock.cbook
sys.modules['matplotlib.ticker'] = matplotlib_mock.ticker
sys.modules['matplotlib.patches'] = matplotlib_mock.patches
sys.modules['matplotlib.backend_bases'] = matplotlib_mock.backend_bases
sys.modules['matplotlib.lines'] = matplotlib_mock.lines

# Mock the backends module
backends_mock = unittest.mock.MagicMock()
backends_mock.backend_tkagg = unittest.mock.MagicMock()
backends_mock.backend_tkagg.FigureCanvasTkAgg = unittest.mock.MagicMock()
backends_mock.backend_agg = unittest.mock.MagicMock()
sys.modules['matplotlib.backends'] = backends_mock
sys.modules['matplotlib.backends.backend_tkagg'] = backends_mock.backend_tkagg
sys.modules['matplotlib.backends.backend_agg'] = backends_mock.backend_agg

# Mock seaborn
seaborn_mock = unittest.mock.MagicMock()
sys.modules['seaborn'] = seaborn_mock

# Mock PySimpleGUI before importing any UI-based modules
psg_mock = unittest.mock.MagicMock()
sys.modules['PySimpleGUI'] = psg_mock
import PySimpleGUI as sg

# Create a mock Window class with all needed attributes and methods
sg.Window = unittest.mock.MagicMock()
mock_window = unittest.mock.MagicMock()
sg.Window.return_value = mock_window
mock_window.AllKeysDict = {}
mock_window.read.return_value = ("WIN_CLOSED", None)

# Define essential constants
sg.WIN_CLOSED = "WIN_CLOSED"
sg.TIMEOUT_KEY = "__TIMEOUT__"

# Mock all commonly used PySimpleGUI components
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
sg.popup = unittest.mock.MagicMock()
sg.popup_error = unittest.mock.MagicMock()
sg.popup_yes_no = unittest.mock.MagicMock(return_value="Yes")
sg.popup_ok = unittest.mock.MagicMock()
sg.HorizontalSeparator = unittest.mock.MagicMock()
sg.Graph = unittest.mock.MagicMock()
sg.Frame = unittest.mock.MagicMock()
sg.Canvas = unittest.mock.MagicMock()

# Also mock tkinter since it's used by matplotlib and PySimpleGUI
tk_mock = unittest.mock.MagicMock()
sys.modules['tkinter'] = tk_mock
sys.modules['tkinter.filedialog'] = unittest.mock.MagicMock()
sys.modules['tkinter.messagebox'] = unittest.mock.MagicMock()

# Now import and run the test script
try:
    print("Setting up headless test environment...")
    
    # Parse command line arguments to match the original test script
    parser = argparse.ArgumentParser(description='Run BTC-AI tests in headless mode')
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--e2e', action='store_true', help='Run end-to-end tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    # Build the command line arguments for the test runner
    test_args = []
    if args.unit:
        test_args.append("--unit")
    if args.integration:
        test_args.append("--integration")
    if args.e2e:
        test_args.append("--e2e")
    if args.all:
        test_args.append("--all")
    
    # Clear any existing arguments and replace with our filtered ones
    sys.argv = [sys.argv[0]] + test_args
    
    # Import the test runner
    print("Importing test runner module...")
    test_runner = importlib.import_module("tests.run_tests")
    
    # Run tests
    print("Running tests in headless mode...")
    test_runner.main()
    
except ImportError as e:
    print(f"Failed to import test module: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error during test execution: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Headless test execution completed successfully") 