"""
Update Dialog for BTC-AI Application

This module provides a UI dialog for checking, downloading, and installing updates
for the BTC-AI application using PySimpleGUI.
"""

import os
import sys
import time
import threading
import PySimpleGUI as sg
from typing import Optional, Dict, Any, Callable

# Import logging
try:
    from src.utils.log_manager import get_logger
    logger = get_logger('update_dialog')
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('update_dialog')

# Import update manager
try:
    from src.utils.update_manager import get_manager, UpdateManager
except ImportError:
    logger.error("Failed to import update_manager module")
    UpdateManager = None
    get_manager = None

# Constants
WINDOW_TITLE = "BTC-AI Updates"
THEME = "DarkBlue"  # Match main application theme

class UpdateDialog:
    """Dialog for managing application updates"""
    
    def __init__(self, parent_window=None, app_dir=None, version=None):
        """
        Initialize the update dialog
        
        Args:
            parent_window: Parent window reference (for modal behavior)
            app_dir: Application directory
            version: Current application version
        """
        self.parent = parent_window
        self.window = None
        self.update_thread = None
        self.download_thread = None
        self.progress_value = 0
        self.progress_max = 100
        self.status_message = "Ready to check for updates"
        self.update_available = False
        self.update_info = None
        self.download_complete = False
        self.download_path = None
        
        # Initialize update manager
        if get_manager:
            if app_dir or version:
                from src.utils.update_manager import initialize
                self.update_manager = initialize(app_dir=app_dir, version=version)
            else:
                self.update_manager = get_manager()
        else:
            self.update_manager = None
            logger.error("Update manager not available")
    
    def create_layout(self):
        """Create the window layout"""
        sg.theme(THEME)
        
        if not self.update_manager:
            # Simple layout for when update manager is not available
            layout = [
                [sg.Text("Update functionality is not available.", font=("Helvetica", 12))],
                [sg.Text("Please check your installation.", font=("Helvetica", 10))],
                [sg.Button("Close", key="-CLOSE-", size=(10, 1))]
            ]
            return layout
        
        # Current version display
        current_version = self.update_manager.get_current_version()
        
        # Main layout with update functionality
        layout = [
            [sg.Text(f"Current Version: {current_version}", font=("Helvetica", 12), key="-CURRENT_VERSION-")],
            [sg.Text("", font=("Helvetica", 12), key="-LATEST_VERSION-", visible=False)],
            [sg.Text(self.status_message, key="-STATUS-", size=(50, 1))],
            [sg.ProgressBar(100, orientation='h', size=(40, 20), key='-PROGRESS-', visible=False)],
            [sg.Multiline("", key="-CHANGELOG-", size=(50, 10), disabled=True, visible=False)],
            [
                sg.Button("Check for Updates", key="-CHECK-", size=(15, 1)),
                sg.Button("Download", key="-DOWNLOAD-", size=(15, 1), disabled=True),
                sg.Button("Install", key="-INSTALL-", size=(15, 1), disabled=True),
                sg.Button("Close", key="-CLOSE-", size=(10, 1))
            ]
        ]
        
        return layout
    
    def show(self):
        """Show the update dialog window"""
        layout = self.create_layout()
        self.window = sg.Window(
            WINDOW_TITLE, 
            layout, 
            finalize=True, 
            modal=self.parent is not None,
            keep_on_top=True,
            icon=None  # Add app icon here if available
        )
        
        # Start the event loop
        self.run()
    
    def run(self):
        """Run the event loop for the dialog"""
        if not self.window:
            logger.error("Window not created")
            return
        
        while True:
            event, values = self.window.read(timeout=100)
            
            if event in (sg.WIN_CLOSED, "-CLOSE-"):
                break
            
            # Check for updates
            if event == "-CHECK-":
                self._start_update_check()
            
            # Download update
            elif event == "-DOWNLOAD-":
                self._start_download()
            
            # Install update
            elif event == "-INSTALL-":
                self._install_update()
            
            # Update UI based on progress
            self._update_ui()
        
        # Clean up threads
        self._cleanup()
        self.window.close()
    
    def _start_update_check(self):
        """Start the update check in a separate thread"""
        if not self.update_manager:
            self.status_message = "Update manager not available"
            return
        
        # Disable check button during check
        self.window["-CHECK-"].update(disabled=True)
        self.status_message = "Checking for updates..."
        
        # Start check in a thread
        self.update_thread = threading.Thread(target=self._check_for_updates)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def _check_for_updates(self):
        """Check for updates in background thread"""
        try:
            update_available, update_info = self.update_manager.check_for_updates()
            
            if update_available and update_info:
                self.update_available = True
                self.update_info = update_info
                self.status_message = f"Update available: v{update_info.get('latest_version', 'unknown')}"
                
                # Prepare changelog text
                changelog = update_info.get('changelog', 'No changelog available')
                
                # Update UI in the main thread
                if self.window:
                    self.window.write_event_value("-UPDATE_AVAILABLE-", {
                        "version": update_info.get('latest_version'),
                        "changelog": changelog
                    })
            else:
                self.status_message = "You have the latest version"
                if self.window:
                    self.window.write_event_value("-NO_UPDATE-", None)
        
        except Exception as e:
            logger.error(f"Error checking for updates: {str(e)}")
            self.status_message = f"Error checking for updates: {str(e)}"
            if self.window:
                self.window.write_event_value("-CHECK_ERROR-", str(e))
    
    def _start_download(self):
        """Start downloading the update in a background thread"""
        if not self.update_manager or not self.update_info:
            self.status_message = "No update information available"
            return
        
        # Update UI
        self.window["-DOWNLOAD-"].update(disabled=True)
        self.window["-PROGRESS-"].update(visible=True)
        self.progress_value = 0
        self.progress_max = 100
        self.status_message = "Downloading update..."
        
        # Start download in a thread
        self.download_thread = threading.Thread(target=self._download_update)
        self.download_thread.daemon = True
        self.download_thread.start()
    
    def _download_update(self):
        """Download the update in background thread"""
        try:
            # Define progress callback
            def progress_callback(current, total):
                self.progress_value = current
                self.progress_max = total
                if self.window:
                    self.window.write_event_value("-DOWNLOAD_PROGRESS-", {"current": current, "total": total})
            
            # Start download
            download_path = self.update_manager.download_update(progress_callback)
            
            if download_path:
                self.download_complete = True
                self.download_path = download_path
                self.status_message = "Download complete - ready to install"
                if self.window:
                    self.window.write_event_value("-DOWNLOAD_COMPLETE-", download_path)
            else:
                self.status_message = "Download failed"
                if self.window:
                    self.window.write_event_value("-DOWNLOAD_FAILED-", None)
        
        except Exception as e:
            logger.error(f"Error downloading update: {str(e)}")
            self.status_message = f"Error downloading update: {str(e)}"
            if self.window:
                self.window.write_event_value("-DOWNLOAD_ERROR-", str(e))
    
    def _install_update(self):
        """Install the downloaded update"""
        if not self.update_manager or not self.download_path:
            self.status_message = "No update package available"
            return
        
        # Ask for confirmation
        confirm = sg.popup_yes_no(
            "Are you sure you want to install this update?\n\n"
            "The application will be closed and restarted after the update.",
            title="Confirm Update Installation",
            keep_on_top=True
        )
        
        if confirm != "Yes":
            return
        
        # Update UI
        self.window["-INSTALL-"].update(disabled=True)
        self.status_message = "Installing update..."
        
        # Create a task to apply the update - this should be run in a way
        # that won't be affected if the main application closes
        try:
            # In a real application, you might:
            # 1. Write a script to apply the update after the main app closes
            # 2. Launch that script as a separate process
            # 3. Close the main application
            
            # For demonstration, we'll just call apply_update
            result = self.update_manager.apply_update(self.download_path)
            
            if result:
                # Ask user if they want to restart now
                restart = sg.popup_yes_no(
                    "Update installed successfully!\n\n"
                    "The application needs to be restarted to apply the changes.\n"
                    "Restart now?",
                    title="Update Complete",
                    keep_on_top=True
                )
                
                if restart == "Yes":
                    # Close the application and restart
                    # This is application-specific - here's a generic approach:
                    self._restart_application()
                else:
                    self.status_message = "Update installed - please restart the application"
            else:
                self.status_message = "Update installation failed"
                sg.popup_error(
                    "Failed to install the update.\n"
                    "Please try again or contact support.",
                    title="Update Failed",
                    keep_on_top=True
                )
        
        except Exception as e:
            logger.error(f"Error installing update: {str(e)}")
            self.status_message = f"Error installing update: {str(e)}"
            sg.popup_error(
                f"Error installing update: {str(e)}",
                title="Update Error",
                keep_on_top=True
            )
        
        finally:
            # Re-enable button in case user wants to try again
            if self.window and not self.window.was_closed():
                self.window["-INSTALL-"].update(disabled=False)
    
    def _restart_application(self):
        """Restart the application after update"""
        if not self.window:
            return
        
        # Close the dialog
        self.window.close()
        self.window = None
        
        # Tell the parent application to restart
        # This is application-specific and will need to be implemented
        # based on your application's architecture
        if self.parent:
            self.parent.write_event_value("-RESTART_APPLICATION-", None)
        else:
            # If no parent window, try to restart directly
            # This is a generic approach that may not work for all applications
            logger.info("Attempting to restart application")
            
            try:
                import subprocess
                
                # Get the path to the executable
                if getattr(sys, 'frozen', False):
                    # Running as compiled executable
                    application_path = sys.executable
                else:
                    # Running as script
                    application_path = sys.argv[0]
                
                # Launch the application again
                subprocess.Popen([application_path] + sys.argv[1:])
                
                # Exit the current instance
                sys.exit(0)
                
            except Exception as e:
                logger.error(f"Failed to restart application: {str(e)}")
                sg.popup_error(
                    "The update was installed successfully, but the application could not be restarted automatically.\n"
                    "Please close and restart the application manually.",
                    title="Restart Failed",
                    keep_on_top=True
                )
    
    def _update_ui(self):
        """Update the UI based on various events"""
        if not self.window or self.window.was_closed():
            return
        
        # Update status message
        self.window["-STATUS-"].update(self.status_message)
        
        # Update progress bar if visible
        if self.window["-PROGRESS-"].visible:
            # Ensure we don't divide by zero
            if self.progress_max > 0:
                percentage = int((self.progress_value / self.progress_max) * 100)
                self.window["-PROGRESS-"].update(current_count=percentage)
        
        # Handle custom events from threads
        event, values = self.window.read(timeout=0)
        
        if event == "-UPDATE_AVAILABLE-":
            # Update version information
            new_version = values[event].get("version", "unknown")
            changelog = values[event].get("changelog", "")
            
            self.window["-LATEST_VERSION-"].update(f"Latest Version: {new_version}")
            self.window["-LATEST_VERSION-"].update(visible=True)
            self.window["-CHANGELOG-"].update(changelog)
            self.window["-CHANGELOG-"].update(visible=True)
            self.window["-CHECK-"].update(disabled=False)
            self.window["-DOWNLOAD-"].update(disabled=False)
        
        elif event == "-NO_UPDATE-":
            self.window["-CHECK-"].update(disabled=False)
            self.window["-PROGRESS-"].update(visible=False)
        
        elif event == "-CHECK_ERROR-":
            self.window["-CHECK-"].update(disabled=False)
            self.window["-PROGRESS-"].update(visible=False)
        
        elif event == "-DOWNLOAD_PROGRESS-":
            current = values[event].get("current", 0)
            total = values[event].get("total", 100)
            
            if total > 0:
                percentage = int((current / total) * 100)
                self.window["-PROGRESS-"].update(current_count=percentage)
        
        elif event == "-DOWNLOAD_COMPLETE-":
            self.window["-DOWNLOAD-"].update(disabled=True)
            self.window["-INSTALL-"].update(disabled=False)
        
        elif event == "-DOWNLOAD_FAILED-" or event == "-DOWNLOAD_ERROR-":
            self.window["-DOWNLOAD-"].update(disabled=False)
    
    def _cleanup(self):
        """Clean up resources before closing"""
        # Stop any running threads
        if self.update_thread and self.update_thread.is_alive():
            # We can't really stop a thread, but we can wait for it
            # with a short timeout
            self.update_thread.join(timeout=0.5)
        
        if self.download_thread and self.download_thread.is_alive():
            self.download_thread.join(timeout=0.5)


def check_for_updates_in_background(app_dir=None, version=None, callback=None):
    """
    Check for updates in the background without showing a UI
    
    Args:
        app_dir: Application directory
        version: Current application version
        callback: Function to call with update information
    """
    if not get_manager:
        if callback:
            callback(False, None, "Update manager not available")
        return
    
    # Initialize update manager
    if app_dir or version:
        from src.utils.update_manager import initialize
        update_manager = initialize(app_dir=app_dir, version=version)
    else:
        update_manager = get_manager()
    
    # Check in a separate thread
    def check_thread():
        try:
            update_available, update_info = update_manager.check_for_updates()
            if callback:
                callback(update_available, update_info, None)
        except Exception as e:
            logger.error(f"Error checking for updates: {str(e)}")
            if callback:
                callback(False, None, str(e))
    
    thread = threading.Thread(target=check_thread)
    thread.daemon = True
    thread.start()


def show_update_notification(parent_window, update_info):
    """
    Show a notification when an update is available
    
    Args:
        parent_window: Parent window reference
        update_info: Update information dictionary
    """
    if not update_info:
        return
    
    version = update_info.get('latest_version', 'unknown')
    
    layout = [
        [sg.Text(f"A new version (v{version}) is available!", font=("Helvetica", 12))],
        [sg.Text("Would you like to view the update details?")],
        [
            sg.Button("View Details", key="-VIEW-"),
            sg.Button("Remind Me Later", key="-LATER-"),
            sg.Button("Ignore", key="-IGNORE-")
        ]
    ]
    
    window = sg.Window(
        "BTC-AI Update Available", 
        layout,
        modal=parent_window is not None,
        keep_on_top=True,
        finalize=True
    )
    
    while True:
        event, values = window.read()
        
        if event in (sg.WIN_CLOSED, "-LATER-"):
            window.close()
            return False
        
        elif event == "-VIEW-":
            window.close()
            # Show update dialog
            dialog = UpdateDialog(parent_window)
            dialog.update_available = True
            dialog.update_info = update_info
            dialog.show()
            return True
        
        elif event == "-IGNORE-":
            window.close()
            # TODO: Store this version as "ignored" in settings
            return True
    
    window.close()


def show_update_dialog(parent_window=None):
    """
    Show the update dialog
    
    Args:
        parent_window: Parent window reference
    """
    dialog = UpdateDialog(parent_window)
    dialog.show()
    
    # Return True if updates were installed
    return dialog.update_available and dialog.download_complete


if __name__ == "__main__":
    # Test the dialog when run directly
    show_update_dialog() 