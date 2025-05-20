"""
Layout System for the RL Trader Parameter Tuner

This module contains functions to create and manage layouts for the application.
"""

import PySimpleGUI as sg
import os
import sys

# Make sure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_status_bar():
    """
    Create the status bar layout for the application.
    
    Returns:
        list: Status bar layout definition for PySimpleGUI
    """
    # Status bar
    status_bar = [
        [sg.Text("Status: Idle", key="-STATUS-", size=(20, 1)),
         sg.Text("", key="-STATUS-INFO-", expand_x=True, justification='left')]
    ]
    
    return status_bar

def create_training_control():
    """
    Create the training control panel layout.
    
    Returns:
        list: Training control layout definition for PySimpleGUI
    """
    # Training control area
    training_control = [
        [sg.Text("Training Control", font=('Helvetica', 10, 'bold'))],
        [sg.Button("Start Training", key="START_TRAINING", button_color=("white", "darkblue")), 
         sg.Button("Stop Training", key="STOP_TRAINING"), 
         sg.Button("Pause Training", key="PAUSE_TRAINING"), 
         sg.Button("Resume Training", key="RESUME_TRAINING"), 
         sg.Button("Restart Training", key="RESTART_TRAINING"), 
         sg.Button("Run Comparison", key="RUN_COMPARISON")],
        [sg.Button("Save Settings", key="SAVE_SETTINGS", button_color=("white", "darkblue")), 
         sg.Button("Load Preset", key="LOAD_PRESET"), 
         sg.Button("Help", key="HELP"), 
         sg.Button("Exit", key="EXIT")]
    ]
    
    return training_control

def create_layout(tabs, notes_content=""):
    """
    Create the main layout for the application.
    
    Args:
        tabs (dict): Dictionary of tabs to include in the layout
        notes_content (str, optional): Initial content for notes area
        
    Returns:
        list: Layout definition for PySimpleGUI
    """
    # Import menus here to avoid circular imports
    from src.ui.menus import get_menu_definition
    
    # Create tab group for main content
    tab_group_layout = []
    for tab_key, tab_content in tabs.items():
        tab_group_layout.append(sg.Tab(tab_key, tab_content))

    # Left column with tabs
    left_column = [
        [sg.TabGroup([tab_group_layout], key='-TAB-GROUP-', 
                    enable_events=True, expand_x=True, expand_y=True)]
    ]
    
    # Right column with log and notes
    right_column = [
        [sg.Frame("Live Log", [
            [sg.Multiline(size=(70, 35), key="-LOG-", autoscroll=True, disabled=True,
                         text_color='black', font=('Courier New', 10))]
        ], size=(550, 450))],
        [sg.Button("Clear Log", key="CLEAR_LOG"), 
         sg.Button("Pop Out Log", key="POP_OUT_LOG", button_color=("white", "darkblue")),
         sg.Button("Save Log", key="SAVE_LOG")],
        [sg.Frame("Notes", [
            [sg.Multiline(notes_content, key="-NOTES-", size=(70, 15), font=('Arial', 10))]
        ])],
        [sg.Button("Save Notes", key="SAVE_NOTES"), 
         sg.Button("Pop Out Notes", key="POP_OUT_NOTES")]
    ]
    
    # Get status bar layout
    status_bar = create_status_bar()
    
    # Get training control layout
    training_control = create_training_control()
    
    # Main layout
    menu_def = get_menu_definition()
    layout = [
        [sg.Menu(menu_def, key='-MENU-')],  # Add menu bar
        [sg.Column(left_column, vertical_alignment='top'), 
         sg.VSeparator(),
         sg.Column(right_column, vertical_alignment='top')],
        [sg.HorizontalSeparator()],
        training_control,
        [sg.HorizontalSeparator()],
        status_bar
    ]
    
    return layout

def create_combined_tabs():
    """
    Create a dictionary of combined tabs based on the current UI needs.
    This function demonstrates how to combine multiple tabs into a more
    efficient layout.
    
    Returns:
        dict: Dictionary of tab names and their layouts
    """
    from src.ui.tabs import (
        create_dashboard_tab, create_main_tab, create_advanced_tab,
        create_reward_tab, create_probabilistic_tab, create_natural_learning_tab,
        create_monitoring_tab, create_withdrawal_tab, create_checkpoint_management_tab,
        create_presets_tab, create_recovery_dashboard_tab, create_help_tab
    )
    
    # Create standard tabs
    dashboard_tab = create_dashboard_tab()
    
    # Combine Strategy, Model, and Reward into a single tab
    model_reward_layout = [
        [sg.TabGroup([
            [sg.Tab('Strategy', create_main_tab())],
            [sg.Tab('Model', create_advanced_tab())],
            [sg.Tab('Reward', create_reward_tab())]
        ], key='-MODEL-REWARD-TABS-', expand_x=True, expand_y=True)]
    ]
    
    # Create the rest of the tabs
    probabilistic_tab = create_probabilistic_tab()
    natural_learning_tab = create_natural_learning_tab()
    monitoring_tab = create_monitoring_tab()
    withdrawal_tab = create_withdrawal_tab()
    checkpoints_tab = create_checkpoint_management_tab()
    presets_tab = create_presets_tab()
    recovery_tab = create_recovery_dashboard_tab()
    help_tab = create_help_tab()
    
    # Create a dictionary of tabs
    tabs = {
        'Dashboard': dashboard_tab,
        'Trading Parameters': model_reward_layout,
        'Probabilistic': probabilistic_tab,
        'Natural Learning': natural_learning_tab,
        'Monitoring': monitoring_tab,
        'Withdrawal Management': withdrawal_tab,
        'Checkpoints': checkpoints_tab,
        'Parameter Presets': presets_tab,
        'Recovery Dashboard': recovery_tab,
        'Help': help_tab
    }
    
    return tabs

def create_log_window(log_text=""):
    """
    Create a pop-out log window.
    
    Args:
        log_text (str, optional): Initial log content
        
    Returns:
        PySimpleGUI.Window: Log window
    """
    layout = [
        [sg.Text("Bitcoin Trading AI - Live Log Monitor", font=("Helvetica", 12, "bold"))],
        [sg.Multiline(log_text, size=(100, 30), key="-POPUP-LOG-", autoscroll=True, disabled=True, 
                    text_color='black', font=('Courier New', 10))],
        [sg.Button("Clear", key="CLEAR_POPUP_LOG"), 
         sg.Button("Save", key="SAVE_POPUP_LOG"),
         sg.Button("Close", key="CLOSE_POPUP_LOG")]
    ]
    
    return sg.Window("Live Log", layout, resizable=True, finalize=True)

def create_notes_window(notes_text=""):
    """
    Create a pop-out notes window.
    
    Args:
        notes_text (str, optional): Initial notes content
        
    Returns:
        PySimpleGUI.Window: Notes window
    """
    layout = [
        [sg.Text("Training Notes", font=("Helvetica", 12, "bold"))],
        [sg.Multiline(notes_text, key="-NOTES-PIP-", size=(100, 25), autoscroll=True, 
                     font=('Arial', 10))],
        [sg.Button("Save", key="SAVE_POPUP_NOTES"),
         sg.Button("Close", key="CLOSE_POPUP_NOTES")]
    ]
    
    return sg.Window("Notes Editor", layout, resizable=True, finalize=True) 