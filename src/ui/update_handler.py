"""
Update Handler Module

This module provides integration between the core update functionality 
and the BTC-AI Training Interface UI.
"""

import os
import sys
import threading
import time
import logging
import PySimpleGUI as sg
from typing import Dict, Any, Optional, Callable

# Set up logger
try:
    from src.utils.log_manager import get_logger
    logger = get_logger('update_handler')
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('update_handler')

# Import platform utils
try:
    from src.utils.platform_utils import is_executable
except ImportError:
    logger.warning("Platform utilities not available. Using fallback functionality.")
    def is_executable():
        return getattr(sys, 'frozen', False)

# Import update dialog functions
try:
    from src.ui.update_dialog import (
        check_for_updates_in_background, 
        show_update_notification, 
        show_update_dialog
    )
    update_dialog_available = True
except ImportError:
    update_dialog_available = False
    logger.warning("Update dialog not available. Update functionality will be limited.")
    
    # Define stub functions
    def check_for_updates_in_background(app_dir=None, version=None, callback=None):
        if callback:
            callback(False, None, "Update dialog not available")
    
    def show_update_notification(parent_window, update_info):
        pass
    
    def show_update_dialog(parent_window=None):
        return False


class UpdateHandler:
    """
    Handler for the application update system.
    Manages update checking, notification, and integration with the UI.
    """
    
    def __init__(self, app_state=None):
        """
        Initialize the update handler.
        
        Args:
            app_state: Application state reference
        """
        self.app_state = app_state
        self.update_available = False
        self.update_info = None
        self.update_check_scheduled = False
        
        logger.info("Update handler initialized")
    
    def check_for_updates(self, window):
        """
        Check for updates and show notification if updates are available
        
        Args:
            window: The main application window
        """
        if not update_dialog_available:
            logger.warning("Update system not available")
            sg.popup_error("Update functionality is not available.\n\nThe update module could not be loaded.",
                       title="Update Error")
            return
        
        # Define callback for update check
        def update_callback(update_available, update_info, error):
            if error:
                logger.error(f"Error checking for updates: {error}")
                sg.popup_error(f"Error checking for updates: {error}", title="Update Error")
                return
            
            if update_available and update_info:
                logger.info(f"Update available: v{update_info.get('latest_version', 'unknown')}")
                # Show notification about available update
                show_update_notification(window, update_info)
            else:
                logger.info("No updates available")
                sg.popup_ok("Your application is up to date.", title="No Updates Available")
        
        # Start background check
        logger.info("Checking for updates...")
        check_for_updates_in_background(callback=update_callback)
    
    def schedule_update_check(self, window, delay=5):
        """
        Schedule a background update check after a delay.
        
        Args:
            window: The main application window
            delay: Delay in seconds before checking
        """
        if not update_dialog_available or self.update_check_scheduled:
            return
        
        # Only schedule update checks for executable builds
        if not is_executable():
            logger.debug("Not running as executable, skipping auto update check")
            return
        
        self.update_check_scheduled = True
        
        # Define delayed update check function
        def delayed_update_check():
            # Wait for application to initialize
            logger.debug(f"Waiting {delay} seconds before checking for updates")
            time.sleep(delay)
            
            # Define callback for silent update check
            def silent_update_callback(update_available, update_info, error):
                if error:
                    logger.error(f"Background update check error: {error}")
                    return
                
                if update_available and update_info:
                    self.update_available = True
                    self.update_info = update_info
                    logger.info(f"Update available: v{update_info.get('latest_version', 'unknown')}")
                    
                    # Notify UI about available update
                    if window:
                        window.write_event_value("-UPDATE-AVAILABLE-", update_info)
            
            # Check for updates
            logger.info("Performing background update check")
            check_for_updates_in_background(callback=silent_update_callback)
        
        # Start background thread for update check
        update_thread = threading.Thread(target=delayed_update_check, daemon=True)
        update_thread.start()
        logger.info("Update check scheduled")
    
    def handle_update_event(self, window, event, values):
        """
        Handle update-related events.
        
        Args:
            window: The main application window
            event: Event string
            values: Event values
            
        Returns:
            bool: True if event was handled, False otherwise
        """
        if not update_dialog_available:
            return False
        
        # Check for update notification event
        if event == "-UPDATE-AVAILABLE-":
            update_info = values[event]
            show_update_notification(window, update_info)
            return True
        
        # Check for update menu command
        elif event == "Check for Updates":
            self.check_for_updates(window)
            return True
        
        return False
    
    def enhance_about_dialog(self, about_window, version):
        """
        Enhance the about dialog with update functionality.
        
        Args:
            about_window: About dialog window
            version: Application version
        """
        if not update_dialog_available:
            return
        
        try:
            # Add update button to about dialog
            about_window.extend_layout(about_window['-BUTTON-CONTAINER-'], [
                [sg.Button("Check for Updates", key="-CHECK-UPDATES-")]
            ])
        except Exception as e:
            logger.error(f"Error enhancing about dialog: {e}")


def get_update_handler(app_state=None):
    """
    Get or create the update handler singleton instance.
    
    Args:
        app_state: Application state reference
        
    Returns:
        UpdateHandler: Update handler instance
    """
    global _update_handler_instance
    
    if '_update_handler_instance' not in globals() or _update_handler_instance is None:
        _update_handler_instance = UpdateHandler(app_state)
    
    return _update_handler_instance 