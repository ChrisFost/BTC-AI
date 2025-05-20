"""
Log Window Manager module for handling log windows and live logging functionality.

This module provides functions for creating popup log windows, handling log-related 
events, and starting live log monitoring. It's designed to work with the existing 
log_manager.py utility which handles the core logging infrastructure.
"""

import os
import sys
import time
import logging
import threading
import PySimpleGUI as sg
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Import layouts to use the create_log_window function
from src.ui.layouts import create_log_window

def start_live_log(window, bucket, models_dir):
    """
    Start reading the log file and updating the log window
    
    Args:
        window (sg.Window): The main application window to send events to
        bucket (str): The bucket (strategy type) name
        models_dir (str): Directory where model logs are stored
    """
    log_path = os.path.join(models_dir, bucket, "training_log.txt")
    
    # Create empty log file if it doesn't exist
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write(f"Log initialized at {datetime.now()}\n")
    
    def log_reader():
        """Read new log lines and send them to the window"""
        with open(log_path, "r") as f:
            # Go to the end of file
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    window.write_event_value('-LOG_UPDATE-', line)
                time.sleep(0.1)
    
    # Start log reader thread
    threading.Thread(target=log_reader, daemon=True).start()

def handle_log_events(window, event, values, popup_log_window=None):
    """
    Handle log-related events from the main application.
    
    Args:
        window (sg.Window): The main application window
        event (str): The event name
        values (dict): The values dictionary from the window
        popup_log_window (sg.Window, optional): The popup log window if it exists
        
    Returns:
        tuple: (handled, updated_popup_log_window) where:
            - handled (bool): True if the event was handled, False otherwise
            - updated_popup_log_window (sg.Window or None): The updated popup log window
    """
    # Handle log-related events
    if event == "CLEAR_LOG":
        # Clear log in main window
        window["-LOG-"].update("")
        return True, popup_log_window
    
    elif event == "SAVE_LOG":
        # Save log from main window
        log_text = window["-LOG-"].get()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"log_{timestamp}.txt"
        
        try:
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            log_path = os.path.join(logs_dir, save_filename)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(log_text)
            
            sg.popup_quick_message(f"Log saved to {log_path}", auto_close_duration=2)
            logger.info(f"Log saved to {log_path}")
            return True, popup_log_window
        except Exception as e:
            sg.popup_error(f"Error saving log: {str(e)}")
            logger.error(f"Error saving log: {str(e)}")
            return True, popup_log_window
    
    elif event == "POP_OUT_LOG":
        # Create pop-out log window
        if popup_log_window is None:
            # Get current log content from main window
            current_log = window["-LOG-"].get()
            popup_log_window = create_log_window(current_log)
            return True, popup_log_window
        else:
            popup_log_window.bring_to_front()
            return True, popup_log_window
    
    elif event == "CLEAR_POPUP_LOG":
        # Clear popup log
        if popup_log_window:
            popup_log_window["-POPUP-LOG-"].update("")
            return True, popup_log_window
    
    elif event == "SAVE_POPUP_LOG":
        # Save popup log
        if popup_log_window:
            log_text = popup_log_window["-POPUP-LOG-"].get()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"log_{timestamp}.txt"
            
            try:
                # Create logs directory if it doesn't exist
                logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
                os.makedirs(logs_dir, exist_ok=True)
                
                log_path = os.path.join(logs_dir, save_filename)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(log_text)
                
                sg.popup_quick_message(f"Log saved to {log_path}", auto_close_duration=2)
                logger.info(f"Log saved to {log_path}")
                return True, popup_log_window
            except Exception as e:
                sg.popup_error(f"Error saving log: {str(e)}")
                logger.error(f"Error saving log: {str(e)}")
                return True, popup_log_window
    
    elif event == "CLOSE_POPUP_LOG":
        # Close popup log window
        if popup_log_window:
            popup_log_window.close()
            return True, None
    
    # Handle log update event
    elif event == "-LOG_UPDATE-":
        # Update main window log
        line = values[event]
        window["-LOG-"].print(line.strip())
        
        # Update popup log if it exists
        if popup_log_window:
            popup_log_window["-POPUP-LOG-"].print(line.strip())
        return True, popup_log_window
    
    # Event not handled
    return False, popup_log_window 