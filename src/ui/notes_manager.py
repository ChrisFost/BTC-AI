"""
Notes Manager for the RL Trader Parameter Tuner

This module contains functions for loading, saving, and displaying notes
in the application.
"""

import os
import sys
import logging
from datetime import datetime
import PySimpleGUI as sg

# Make sure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import error handling
try:
    from src.ui.error_handler import handle_error, ErrorSeverity
    from src.utils.persistent_logger import log_persistent_error
    error_handling_available = True
except ImportError:
    error_handling_available = False
    # Define stub functions if error handling is not available
    def handle_error(error, context="", window=None, retry_func=None, additional_context=None):
        if isinstance(error, Exception):
            logging.error(f"Error in {context}: {str(error)}")
        return {"handled": False}
    
    class ErrorSeverity:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        FATAL = "fatal"
    
    def log_persistent_error(error, context="", severity="medium", additional_info=None):
        pass

# Setup logger
logger = logging.getLogger(__name__)

# Constants
NOTES_FILE = "training_notes.txt"
DEFAULT_NOTES = "Use this area to keep notes about your training sessions.\n\nTips:\n- Document successful configurations\n- Track experiment outcomes\n- Note interesting observations"

# Global variable for notes content
notes_content = ""

def load_notes():
    """
    Load notes from file or use default if file doesn't exist.
    
    Returns:
        str: The loaded notes content
    """
    global notes_content
    
    # Load notes
    notes_content = ""
    try:
        if os.path.exists(NOTES_FILE):
            with open(NOTES_FILE, "r", encoding="utf-8") as f:
                notes_content = f.read()
        else:
            notes_content = DEFAULT_NOTES
            # Create the notes file with default content
            with open(NOTES_FILE, "w", encoding="utf-8") as f:
                f.write(DEFAULT_NOTES)
            logger.info(f"Created new notes file at {NOTES_FILE}")
    except Exception as e:
        if error_handling_available:
            handle_error(
                e,
                context="Loading training notes",
                additional_context={"file_path": NOTES_FILE, "action": "read"}
            )
            log_persistent_error(
                e,
                context="Loading training notes",
                severity=ErrorSeverity.LOW,
                additional_info={"file_path": NOTES_FILE}
            )
        else:
            logger.error(f"Error loading notes: {e}")
        
        # Use default notes content in case of error
        notes_content = DEFAULT_NOTES
    
    return notes_content

def save_notes(content):
    """
    Save notes content to file.
    
    Args:
        content (str): The notes content to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    global notes_content
    
    try:
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        notes_content = content
        logger.info(f"Notes saved to {NOTES_FILE}")
        return True
    except Exception as e:
        if error_handling_available:
            handle_error(
                e, 
                context="Saving notes",
                additional_context={"file_path": NOTES_FILE, "action": "write"}
            )
            log_persistent_error(
                e,
                context="Saving notes",
                severity=ErrorSeverity.LOW,
                additional_info={"file_path": NOTES_FILE}
            )
        else:
            logger.error(f"Error saving notes: {e}")
        return False

def create_notes_window(initial_content):
    """
    Create a pop-out window for notes.
    
    Args:
        initial_content (str): The initial notes content
        
    Returns:
        sg.Window: The notes window
    """
    layout = [
        [sg.Multiline(initial_content, key="-POPUP-NOTES-", size=(80, 20), font=('Arial', 10))],
        [sg.Button("Save", key="SAVE_POPUP_NOTES"), sg.Button("Close", key="CLOSE_POPUP_NOTES")]
    ]
    
    return sg.Window("Training Notes", layout, resizable=True, finalize=True)

def handle_notes_event(window, event, values, notes_window=None):
    """
    Handle notes-related events from the main application.
    
    Args:
        window (sg.Window): The main application window
        event (str): The event name
        values (dict): The values dictionary from the window
        notes_window (sg.Window, optional): The notes popup window if it exists
        
    Returns:
        tuple: (handled, updated_notes_window) where:
            - handled (bool): True if the event was handled, False otherwise
            - updated_notes_window (sg.Window or None): The updated notes window
    """
    global notes_content
    
    # Handle notes events
    if event == "SAVE_NOTES":
        # Save notes from main window
        if values and "-NOTES-" in values:
            if save_notes(values["-NOTES-"]):
                sg.popup_quick_message("Notes saved successfully", auto_close_duration=2)
                logger.info("Notes saved successfully")
                return True, notes_window
            else:
                sg.popup_error("Error saving notes")
                logger.error("Error saving notes")
                return True, notes_window
    
    elif event == "POP_OUT_NOTES":
        # Create pop-out notes window
        if notes_window is None:
            # Get current notes content from main window
            current_notes = values["-NOTES-"]
            notes_window = create_notes_window(current_notes)
            return True, notes_window
        else:
            notes_window.bring_to_front()
            return True, notes_window
    
    elif event == "SAVE_POPUP_NOTES":
        # Save notes from popup window
        notes_content = values["-POPUP-NOTES-"]
        if save_notes(notes_content):
            sg.popup_quick_message("Notes saved successfully", auto_close_duration=2)
            # Update main window notes
            window["-NOTES-"].update(notes_content)
            logger.info("Notes saved successfully from popup window")
            return True, notes_window
        else:
            sg.popup_error("Error saving notes")
            logger.error("Error saving notes from popup window")
            return True, notes_window
    
    elif event == "CLOSE_POPUP_NOTES":
        # Save notes from popup window before closing
        notes_content = values["-POPUP-NOTES-"]
        save_notes(notes_content)
        # Update main window notes
        window["-NOTES-"].update(notes_content)
        notes_window.close()
        return True, None
    
    # Event not handled
    return False, notes_window 