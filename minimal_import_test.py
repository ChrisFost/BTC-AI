#!/usr/bin/env python3
"""
Minimal script to test imports - with UI mocking
"""

import os
import sys
import unittest.mock

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Mock PySimpleGUI before importing any UI-based modules
sys.modules['PySimpleGUI'] = unittest.mock.MagicMock()
import PySimpleGUI as sg
# Create a mock Window class
sg.Window = unittest.mock.MagicMock()
# Ensure the Window mock returns another mock for method chaining
sg.Window.return_value = unittest.mock.MagicMock()
# Simulate window.read() returning window closed
sg.Window.return_value.read.return_value = (sg.WIN_CLOSED, None)

try:
    import src.utils.utils
    print("Successfully imported src.utils.utils")
except ImportError as e:
    print(f"Failed to import src.utils.utils: {e}")

try:
    from src.environment import env_utils
    print("Successfully imported src.environment.env_utils")
except ImportError as e:
    print(f"Failed to import src.environment.env_utils: {e}")

try:
    from src.models import models
    print("Successfully imported src.models.models")
except ImportError as e:
    print(f"Failed to import src.models.models: {e}")

try:
    from src.ui import main
    print("Successfully imported src.ui.main")
except ImportError as e:
    print(f"Failed to import src.ui.main: {e}")

try:
    from src.ui import setup_wizard
    print("Successfully imported src.ui.setup_wizard")
except ImportError as e:
    print(f"Failed to import src.ui.setup_wizard: {e}")

print("All imports completed. UI components have been mocked to prevent windows from opening.") 