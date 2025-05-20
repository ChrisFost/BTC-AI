"""
Menu System for the RL Trader Parameter Tuner

This module contains the menu definitions and handlers for the application.
"""

import PySimpleGUI as sg
import os
import sys

# Make sure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_menu_definition():
    """
    Returns the menu definition for the main window.
    
    The menu includes File, Training, Tools, and Help sections with
    various options for managing the application.
    """
    menu_def = [
        ['&File', ['Save Settings', 'Load Preset', '---', 'Exit']],
        ['&Training', ['Start Training', 'Stop Training', 'Pause Training', 'Resume Training', 
                      'Restart Training', '---', 'Create Checkpoint', 'Load Checkpoint']],
        ['&Tools', ['Run Comparison', 'View Performance', 'Hardware Monitor', '---', 
                   'Export Results', 'Import Configuration', '---', 'Clear Cache']],
        ['&Help', ['Documentation', 'Quick Start Guide', 'Parameter Reference', '---', 
                  'Check for Updates', 'About']]
    ]
    return menu_def

def handle_menu_event(event, window, values, app_state=None):
    """
    Handle menu events triggered by user interactions.
    
    Args:
        event (str): The event name from PySimpleGUI
        window (PySimpleGUI.Window): The main application window
        values (dict): The current values from the window
        app_state (AppState, optional): The application state object
        
    Returns:
        bool: True if the application should continue, False if it should exit
    """
    # File menu handlers
    if event == 'Save Settings':
        if app_state:
            app_state.save_all_state(values)
        return True
    
    elif event == 'Load Preset':
        # Will be implemented by the main application
        return True
    
    elif event == 'Exit':
        # Ask for confirmation before exiting
        if sg.popup_yes_no('Are you sure you want to exit?', 
                          title='Exit Confirmation') == 'Yes':
            return False  # Signal to exit the application
        return True
    
    # Training menu handlers
    elif event == 'Start Training':
        # Will be implemented by the main application
        return True
    
    elif event == 'Stop Training':
        # Will be implemented by the main application
        return True
    
    elif event == 'Pause Training':
        # Will be implemented by the main application
        return True
    
    elif event == 'Resume Training':
        # Will be implemented by the main application
        return True
    
    elif event == 'Restart Training':
        # Will be implemented by the main application
        return True
    
    elif event == 'Create Checkpoint':
        if app_state:
            app_state.create_checkpoint(notify=True)
        return True
    
    elif event == 'Load Checkpoint':
        # Will be implemented by the main application
        return True
    
    # Tools menu handlers  
    elif event == 'Run Comparison':
        # Will be implemented by the main application
        return True
    
    elif event == 'View Performance':
        # Will be implemented by the main application
        return True
    
    elif event == 'Hardware Monitor':
        # Will be implemented by the main application
        return True
    
    elif event == 'Export Results':
        # Will be implemented by the main application
        return True
    
    elif event == 'Import Configuration':
        # Will be implemented by the main application
        return True
    
    elif event == 'Clear Cache':
        # Will be implemented by the main application
        return True
    
    # Help menu handlers
    elif event == 'Documentation':
        # Will be implemented by the main application
        return True
    
    elif event == 'Quick Start Guide':
        # Will be implemented by the main application
        return True
    
    elif event == 'Parameter Reference':
        # Will be implemented by the main application
        return True
    
    elif event == 'Check for Updates':
        # Will be implemented by the main application
        return True
    
    elif event == 'About':
        show_about_dialog()
        return True
    
    # If we get here, the event wasn't handled
    return True

def show_about_dialog():
    """
    Display the About dialog with application information.
    """
    layout = [
        [sg.Text("RL Trader Parameter Tuner", font=("Helvetica", 16, "bold"))],
        [sg.Text("Version 1.0.0", font=("Helvetica", 10))],
        [sg.Text("Developed by the RL Trader Team")],
        [sg.Text("© 2023-2025 All Rights Reserved")],
        [sg.Button("OK", key="OK")]
    ]
    
    window = sg.Window("About", layout, modal=True, finalize=True)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "OK":
            break
    
    window.close() 