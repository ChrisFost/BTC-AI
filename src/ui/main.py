"""
Main Application Module

This is the main entry point for the BTC AI Training Interface.
It provides a GUI for configuring, training, and managing models.
"""

import os
import sys
import json
import logging
import threading
import torch
import PySimpleGUI as sg
from datetime import datetime
from typing import Dict, Any, List, Optional
import time

# Set up project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, "logs", "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import modularized components
try:
    from src.ui.app_state import AppState
    from src.ui.layouts import create_layout
    from src.ui.notes_manager import load_notes, save_notes, handle_notes_event
    from src.ui.log_window_manager import create_log_window, handle_log_events
    from src.ui.training_manager import TrainingManager
    from src.ui.bucket_manager import BucketManager, handle_bucket_change
    from src.ui.comparison_manager import ComparisonManager
    from src.ui.theme_manager import ThemeManager, get_theme_manager
    from src.ui.error_handler import handle_error, ErrorSeverity
    from src.utils.persistent_logger import log_persistent_error
    from src.utils.platform_utils import is_executable, check_required_files, get_app_version
    from src.ui.update_handler import get_update_handler
    from src.ui.backtesting_integration import initialize_backtesting_integration
    from src.ui.performance_monitor import get_performance_monitor
    
    error_handling_available = True
    platform_utils_available = True
    update_handler_available = True
    backtesting_available = True
    performance_monitor_available = True
except ImportError as e:
    error_handling_available = False
    platform_utils_available = False
    update_handler_available = False
    backtesting_available = False
    performance_monitor_available = False
    logger.warning(f"Some utilities not available: {str(e)}. Some features will be limited.")
    
    # Define stub functions if error handling is not available
    def handle_error(error, context="", window=None, retry_func=None, additional_context=None):
        if isinstance(error, Exception):
            logger.error(f"Error in {context}: {str(error)}")
        return {"handled": False}
    
    class ErrorSeverity:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        FATAL = "fatal"
    
    def log_persistent_error(error, context="", severity="medium", additional_info=None):
        logger.error(f"Error in {context}: {str(error)}")
    
    def log_exception(e, context=""):
        logger.error(f"Exception in {context}: {str(e)}")
        import traceback
        if hasattr(e, "__traceback__"):
            logger.error(traceback.format_exception(type(e), e, e.__traceback__))
    
    # Define stub functions for platform utils
    def is_executable():
        return getattr(sys, 'frozen', False)
    
    def check_required_files():
        return True, ""
    
    def get_app_version():
        return "1.0.0"
    
    # Define stub function for backtesting
    def initialize_backtesting_integration():
        pass

# Try to import preset system
try:
    from src.ui.preset_handlers import (
        handle_preset_tab_events, 
        initialize_preset_handlers,
        refresh_preset_list
    )
    preset_system_available = True
    logger.info("Preset system initialized successfully")
except ImportError as e:
    preset_system_available = False
    logger.warning(f"Preset system not available: {str(e)}. Some features will be disabled.")
    
    # Define stub function
    def handle_preset_tab_events(window, event, values):
        return False
    
    def initialize_preset_handlers():
        return False
    
    def refresh_preset_list(window, bucket):
        pass

# Global variables
app_state = None
training_manager = None
bucket_manager = None
comparison_manager = None
theme_manager = None
logs_window = None
notes_window = None
process = None

def initialize_app():
    """Initialize the application components."""
    global app_state, training_manager, bucket_manager, comparison_manager, theme_manager
    
    # Check required files if platform utils available
    if platform_utils_available:
        success, error_msg = check_required_files()
        if not success:
            logger.error(f"Required files check failed: {error_msg}")
            sg.popup_error(f"Missing required files or directories: {error_msg}\n\nPlease reinstall the application.")
            return None
    
    # Create application state
    app_state = AppState()
    
    # Initialize theme manager and apply theme
    theme_manager = get_theme_manager(app_state)
    theme_manager.apply_theme()
    
    # Initialize preset system first if available
    if preset_system_available:
        initialize_preset_handlers()
        logger.info("Preset handlers initialized")
        
    # Create specialized managers
    training_manager = TrainingManager(app_state)
    bucket_manager = BucketManager(app_state)
    comparison_manager = ComparisonManager(app_state, bucket_manager)
    
    # Initialize backtesting integration if available
    if backtesting_available:
        try:
            initialize_backtesting_integration()
            logger.info("Backtesting integration initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing backtesting integration: {e}")
    
    # Initialize performance monitor if available
    if performance_monitor_available:
        performance_monitor = get_performance_monitor(app_state)
        logger.info("Performance monitor initialized")
    
    logger.info("Application initialized successfully")
    
    return app_state

def init_main_tab(window):
    """
    Initialize components specific to the main tab
    
    Args:
        window: The PySimpleGUI window
    """
    # Update bucket goals based on current bucket
    if "BUCKET" in window.AllKeysDict:
        current_bucket = window["BUCKET"].get()
        handle_bucket_change(window, {"BUCKET": current_bucket})
    
    # Initialize other main tab components as needed
    logger.info("Main tab initialized")

def update_window_from_state(window, values):
    """
    Update the window elements from application state
    
    Args:
        window: The PySimpleGUI window
        values: Current values dictionary
    """
    try:
        # Apply any state to window elements
        for key, value in app_state.config.items():
            if key in window.AllKeysDict:
                window[key].update(value)
        
        # Update notes
        if "-NOTES-" in window.AllKeysDict:
            notes_content = load_notes()
            window["-NOTES-"].update(notes_content)
        
        # Initialize bucket settings
        if "BUCKET" in window.AllKeysDict and "BUCKET" in values:
            bucket_manager.handle_bucket_change(window, values)
        
        # Update training control buttons based on current state
        is_training, recovery_data = training_manager.check_training_in_progress(
            values.get("BUCKET", "Scalping")
        )
        
        if is_training:
            window["START_TRAINING"].update(disabled=True)
            window["STOP_TRAINING"].update(disabled=False)
            window["PAUSE_TRAINING"].update(disabled=False)
            window["RESUME_TRAINING"].update(disabled=True)
        else:
            window["START_TRAINING"].update(disabled=False)
            window["STOP_TRAINING"].update(disabled=True)
            window["PAUSE_TRAINING"].update(disabled=True)
            window["RESUME_TRAINING"].update(disabled=True)
        
        logger.info("Window updated from state")
    except Exception as e:
        logger.error(f"Error updating window from state: {e}")
        if error_handling_available:
            handle_error(
                e,
                context="Updating window from application state",
                window=window
            )

def main():
    """Main application entry point"""
    global app_state, training_manager, bucket_manager, comparison_manager, theme_manager
    global logs_window, notes_window, process
    
    # For system status updates
    last_status_update = time.time()
    
    try:
        # Initialize application
        app_state = initialize_app()
        if not app_state:
            return
        
        # Create main window
        window = app_state.create_window()
        
        # Update window elements from state
        update_window_from_state(window, window.read(timeout=0)[1])
        
        # Schedule update check if running as executable
        if update_handler_available:
            update_handler = get_update_handler(app_state)
            update_handler.schedule_update_check(window)
        
        # Start live log if available
        current_bucket = window["BUCKET"].get() if "BUCKET" in window.AllKeysDict else "Scalping"
        training_manager.start_live_log(window, current_bucket)
        
        # Initialize main tab
        init_main_tab(window)
        
        # Start performance monitoring
        if performance_monitor_available:
            performance_monitor = get_performance_monitor(app_state)
            performance_monitor.start_monitoring(window)
        
        # Main event loop
        while True:
            # Read events with a timeout to allow for background updates
            event, values = window.read(timeout=100)
            
            # Window closed event
            if event == sg.WIN_CLOSED:
                if logs_window:
                    logs_window.close()
                if notes_window:
                    # Save notes from popup window before closing
                    if values and "-POPUP-NOTES-" in values:
                        notes_content = values["-POPUP-NOTES-"]
                        save_notes(notes_content)
                        # Update main window notes
                        window["-NOTES-"].update(notes_content)
                    notes_window.close()
                
                # Main window closed - save all state before exit
                app_state.save_all_state(values)
                # Stop training if in progress
                training_manager.stop_training(window)
                break
            
            # Timeout event - use for background updates if needed
            if event == "__TIMEOUT__":
                continue
            
            # Handle preset events if available
            if preset_system_available and handle_preset_tab_events(window, event, values):
                # If the event was handled by preset_handlers, continue to next iteration
                continue
            
            # Handle menu item events from the File menu
            if event == "Save Settings":
                # Menu item "Save Settings" triggers same behavior as the button
                window.write_event_value("SAVE_SETTINGS", None)
                continue
                
            elif event == "Load Preset":
                # Menu item "Load Preset" triggers same behavior as the button
                window.write_event_value("LOAD_PRESET", None)
                continue
            
            # Handle menu item events from the Training menu
            elif event == "Start Training":
                window.write_event_value("START_TRAINING", None)
                continue
                
            elif event == "Stop Training":
                window.write_event_value("STOP_TRAINING", None)
                continue
                
            elif event == "Pause Training":
                window.write_event_value("PAUSE_TRAINING", None)
                continue
                
            elif event == "Resume Training":
                window.write_event_value("RESUME_TRAINING", None)
                continue
                
            elif event == "Restart Training":
                window.write_event_value("RESTART_TRAINING", None)
                continue
            
            # Handle menu item events from the Tools menu
            elif event == "Run Comparison":
                window.write_event_value("RUN_COMPARISON", None)
                continue
                
            elif event == "Clear Log":
                window.write_event_value("CLEAR_LOG", None)
                continue
                
            elif event == "Save Log":
                window.write_event_value("SAVE_LOG", None)
                continue
                
            elif event == "Create Recovery Checkpoint":
                window.write_event_value("-CREATE-CHECKPOINT-", None)
                continue
            
            # Handle menu item events from the Help menu
            elif event == "Help":
                window.write_event_value("HELP", None)
                continue
                
            elif event == "About":
                window.write_event_value("ABOUT", None)
                continue
                
            elif event == "Theme Settings":
                theme_manager.show_theme_selector(window)
                continue
                
            elif event == "Toggle Dark Mode":
                theme_manager.toggle_dark_mode()
                # Relaunch the window to apply the new theme
                window.close()
                window = app_state.create_window()
                update_window_from_state(window, window.read(timeout=0)[1])
                continue
            
            elif event == "Check for Updates" and update_handler_available:
                update_handler = get_update_handler()
                update_handler.check_for_updates(window)
                continue
            
            elif event == "-UPDATE-AVAILABLE-" and update_handler_available:
                update_handler = get_update_handler()
                update_handler.handle_update_event(window, event, values)
                continue
            
            # Handle notes events
            elif event.startswith("-NOTES-") or event == "SAVE_NOTES" or event == "POP_OUT_NOTES":
                notes_window = handle_notes_event(window, event, values, notes_window)
                continue
            
            # Handle log events
            elif event.startswith("-LOG-") or event == "CLEAR_LOG" or event == "SAVE_LOG" or event == "POP_OUT_LOG":
                logs_window = handle_log_events(window, event, values, logs_window)
                continue
            
            # Handle bucket change
            elif event == "BUCKET":
                bucket_manager.handle_bucket_change(window, values)
                continue
            
            # --- Handle Manual Tab Click in Bucket Goals --- 
            elif event == "-GOALS-TABGROUP-":
                selected_tab_key = values["-GOALS-TABGROUP-"]
                logger.info(f"Bucket goal tab manually selected: {selected_tab_key}")
                
                tab_to_bucket_map = {
                    "-TAB-SCALPING-": "Scalping",
                    "-TAB-SHORT-": "Short",
                    "-TAB-MEDIUM-": "Medium",
                    "-TAB-LONG-": "Long"
                }
                
                target_bucket = tab_to_bucket_map.get(selected_tab_key)
                
                # Check if the dropdown needs updating
                if target_bucket and "BUCKET" in values and values["BUCKET"] != target_bucket:
                    logger.info(f"Updating BUCKET dropdown to match selected tab: {target_bucket}")
                    window["BUCKET"].update(value=target_bucket)
                    # Note: Updating dropdown triggers 'BUCKET' event, ensuring consistency.
                # Add a continue to match the pattern
                continue
            
            # Handle training control events
            elif event == "START_TRAINING":
                # Save all state before starting training
                app_state.save_all_state(values)
                
                # Start training
                training_manager.start_training(window, values)
                
            elif event == "STOP_TRAINING":
                training_manager.stop_training(window)
                
            elif event == "PAUSE_TRAINING":
                training_manager.pause_training(window)
                
            elif event == "RESUME_TRAINING":
                training_manager.resume_training(window)
            
            # Handle comparison events
            elif event == "RUN_COMPARISON":
                # Save current config
                app_state.save_all_state(values)
                
                # Run comparison between current settings and default settings
                try:
                    comparison_process = comparison_manager.run_comparison(window, values)
                except Exception as e:
                    logger.error(f"Error running comparison: {str(e)}")
                    sg.popup_error(f"Error running comparison: {str(e)}")
            
            elif event == "-COMPARISON-UPDATE-":
                # Update log with comparison progress
                window["-LOG-"].print(f"[Comparison] {values[event]}")
                
            elif event == "-COMPARISON-COMPLETE-":
                # Comparison finished successfully
                comparison_manager.handle_comparison_complete(window, values, event)
            
            elif event == "-COMPARISON-ERROR-":
                # Comparison error
                window["RUN_COMPARISON"].update(disabled=False)
                window["-STATUS-"].update("Status: Comparison failed")
                sg.popup_error(f"Comparison failed: {values[event]}")
            
            # Handle training monitoring events
            elif event == "-TRAINING-COMPLETE-":
                # Training completed successfully
                window["START_TRAINING"].update(disabled=False)
                window["STOP_TRAINING"].update(disabled=True)
                window["PAUSE_TRAINING"].update(disabled=True)
                window["RESUME_TRAINING"].update(disabled=True)
                
                window["-STATUS-"].update("Status: Training completed")
                window["-LOG-"].print(f"[{datetime.now()}] Training completed successfully")
                
                # Show metrics if available
                if values[event]:
                    metrics_text = ""
                    for metric, value in values[event].items():
                        if isinstance(value, (int, float)):
                            if "profit" in metric:
                                metrics_text += f"{metric.replace('_', ' ').title()}: ${value:.2f}\n"
                            elif "rate" in metric:
                                metrics_text += f"{metric.replace('_', ' ').title()}: {value*100:.1f}%\n"
                            else:
                                metrics_text += f"{metric.replace('_', ' ').title()}: {value}\n"
                    
                    if metrics_text:
                        sg.popup("Training Completed", metrics_text, title="Training Results")
            
            elif event == "-TRAINING-ERROR-":
                # Training error
                window["START_TRAINING"].update(disabled=False)
                window["STOP_TRAINING"].update(disabled=True)
                window["PAUSE_TRAINING"].update(disabled=True)
                window["RESUME_TRAINING"].update(disabled=True)
                
                window["-STATUS-"].update("Status: Training failed")
                window["-LOG-"].print(f"[{datetime.now()}] Training failed: {values[event]}")
                
                sg.popup_error(f"Training failed: {values[event]}")
            
            elif event == "-LOG_UPDATE-":
                # Update from live log
                if "-LOG-" in window.AllKeysDict:
                    window["-LOG-"].print(values[event], end="")
            
            # Handle save settings
            elif event == "SAVE_SETTINGS":
                app_state.save_all_state(values)
                window["-STATUS-"].update("Status: Settings saved")
                window["-LOG-"].print(f"[{datetime.now()}] Settings saved")
            
            elif event == "-QUICK-SAVE-":
                # Quick save without popup confirmation
                app_state.save_all_state(values)
                window["-STATUS-"].update("Status: Settings saved")
            
            # Handle window resize event
            elif event == "-WINDOW-SIZE-CHANGE-":
                # Handle window resize if needed
                pass
            
            # Handle About dialog
            elif event == "ABOUT":
                show_about()
            
            # Handle Exit
            elif event == "Exit":
                # Save all state before exiting
                app_state.save_all_state(values)
                # Stop training if in progress
                training_manager.stop_training(window)
                break
            
            # Handle performance monitor updates
            elif event == "-PERFORMANCE-UPDATE-" and performance_monitor_available:
                # Just use this opportunity to update system status
                # Check if the training_manager exists and its process attribute is not None
                is_training = (training_manager and training_manager.process is not None)
                performance_monitor = get_performance_monitor()
                last_status_update = performance_monitor.update_system_status(
                    window, 
                    is_training, 
                    getattr(app_state, 'training_start_time', None),
                    last_status_update
                )
                continue
            
    except Exception as e:
        if error_handling_available:
            # Handle error with persistent logging and UI notification
            error_details = handle_error(
                e,
                context="Processing UI event", 
                window=window if 'window' in locals() else None,
                additional_context={
                    "event": event if 'event' in locals() else "Unknown",
                }
            )
           
           # Log error persistently
            log_persistent_error(
               e,
               context="Main event loop error",
               severity=ErrorSeverity.HIGH,
               additional_info={
                   "event": event if 'event' in locals() else "Unknown"
               }
            )
           
            # If error wasn't handled by the dialog, show a basic error message
            if not error_details.get("handled", False):
                sg.popup_error(f"An error occurred: {str(e)}\n\nThe application will continue, but some functionality may be limited.")
                
                # Give the user a chance to see the error before continuing
                time.sleep(1)
        else:
            # Basic error handling
            logger.error(f"Unhandled error in main event loop: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            sg.popup_error(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        if 'logs_window' in locals() and logs_window:
            logs_window.close()
        if 'notes_window' in locals() and notes_window:
            notes_window.close()
        if 'window' in locals() and window:
            window.close()
        
        # Stop performance monitoring
        if performance_monitor_available:
            try:
                performance_monitor = get_performance_monitor()
                performance_monitor.stop_monitoring()
                logger.info("Performance monitoring stopped")
            except Exception as e:
                logger.error(f"Error stopping performance monitor: {e}")

def show_about():
    """Show about dialog with version and update information"""
    try:
        # Get application version
        version = get_app_version() if platform_utils_available else "1.0.0"
        
        # Get system info
        system_info = f"System: {sys.platform}\n"
        system_info += f"Python: {sys.version.split()[0]}\n"
        system_info += f"PySimpleGUI: {sg.__version__}\n"
        system_info += f"PyTorch: {torch.__version__}\n"
        
        # Check for CUDA
        cuda_available = torch.cuda.is_available()
        cuda_info = f"CUDA available: {cuda_available}\n"
        if cuda_available:
            cuda_info += f"CUDA version: {torch.version.cuda}\n"
            cuda_info += f"GPU: {torch.cuda.get_device_name(0)}\n"
        
        # Create layout for the about dialog
        layout = [
            [sg.Text(f"BTC AI Training Interface v{version}", font=("Helvetica", 16))],
            [sg.Text("A tool for training and managing trading models")],
            [sg.Text("")],
            [sg.Text("System Information:", font=("Helvetica", 10, "bold"))],
            [sg.Text(system_info)],
            [sg.Text("CUDA Information:", font=("Helvetica", 10, "bold"))],
            [sg.Text(cuda_info)],
            [sg.Text("")],
            [sg.Text("© 2023-2024 BTC AI Team")],
            [sg.Column([[sg.Button("OK", key="OK")]], key="-BUTTON-CONTAINER-")]
        ]
        
        # Create the about window
        about_window = sg.Window("About", layout, modal=True, finalize=True)
        
        # Enhance with update functionality if available
        if update_handler_available:
            update_handler = get_update_handler()
            update_handler.enhance_about_dialog(about_window, version)
        
        # Event loop
        while True:
            event, values = about_window.read()
            if event in (sg.WIN_CLOSED, "OK"):
                break
            elif event == "-CHECK-UPDATES-" and update_handler_available:
                about_window.close()
                update_handler = get_update_handler()
                update_handler.check_for_updates(None)
                return
        
        about_window.close()
        
    except Exception as e:
        logger.error(f"Error showing about dialog: {str(e)}")
        sg.popup_error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 