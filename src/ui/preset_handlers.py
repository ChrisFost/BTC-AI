import PySimpleGUI as sg
import logging
import os
from typing import Dict, Any, List, Tuple
import datetime

# Import preset manager
try:
    from src.ui.preset_manager import (
        list_presets,
        load_preset,
        save_preset,
        delete_preset,
        cleanup_temp_presets,
        update_suggestion_list,
        show_performance_history,
        PRESET_DIR
    )
except ImportError:
    sg.popup_error("Error importing preset_manager. Preset functionality will not be available.")
    logging.error("Failed to import preset_manager module")

# Try to import logging utilities
try:
    from src.utils.log_manager import LogManager
    logger = LogManager.get_logger("preset_handlers")
except ImportError:
    # Fallback if imports fail
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("preset_handlers")

# Global variables
preset_id_map = {}  # Maps preset list indices to preset IDs
suggestion_id_map = {}  # Maps suggestion list indices to preset IDs
current_preset_id = None  # Currently loaded preset ID

def initialize_preset_handlers():
    """Initialize preset directories and cleanup old temporary presets"""
    # Import here to avoid circular imports
    from src.ui.preset_manager import ensure_preset_directories, initialize_default_presets
    
    # Initialize directories and default presets
    ensure_preset_directories()
    initialize_default_presets()
    
    # Clean up old temporary presets (older than 7 days)
    deleted_count = cleanup_temp_presets(7)
    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} temporary presets older than 7 days")
    
    return True

def handle_preset_tab_events(window, event, values):
    """
    Handle events for the preset tab
    
    Args:
        window: The PySimpleGUI window
        event: The event that occurred
        values: The current values dictionary
        
    Returns:
        bool: True if the event was handled, False otherwise
    """
    global preset_id_map, suggestion_id_map, current_preset_id
    
    # Refresh preset list when bucket changes
    if event == "-PRESET-BUCKET-":
        refresh_preset_list(window, values["-PRESET-BUCKET-"])
        update_suggestion_list(window, values["-PRESET-BUCKET-"], 
                              get_filter_type(values))
        return True
    
    # Load default presets
    elif event in ["-LOAD-DEFAULT-SCALPING-", "-LOAD-DEFAULT-SHORT-", 
                  "-LOAD-DEFAULT-MEDIUM-", "-LOAD-DEFAULT-LONG-"]:
        bucket = event.replace("-LOAD-DEFAULT-", "").replace("-", "")
        return load_default_preset(window, bucket)
    
    # Preset list selection
    elif event == "-PRESET-LIST-":
        if values["-PRESET-LIST-"]:
            window["-LOAD-PRESET-"].update(disabled=False)
            window["-DELETE-PRESET-"].update(disabled=False)
            window["-QUICK-VIEW-PERFORMANCE-"].update(disabled=False)
        else:
            window["-LOAD-PRESET-"].update(disabled=True)
            window["-DELETE-PRESET-"].update(disabled=True)
            window["-QUICK-VIEW-PERFORMANCE-"].update(disabled=True)
        return True
    
    # Load selected preset
    elif event == "-LOAD-PRESET-":
        if values["-PRESET-LIST-"]:
            selected_idx = values["-PRESET-LIST-"].index(values["-PRESET-LIST-"][0])
            if selected_idx in preset_id_map:
                preset_id = preset_id_map[selected_idx]
                return load_preset_into_ui(window, preset_id)
        return True
    
    # Delete selected preset
    elif event == "-DELETE-PRESET-":
        if values["-PRESET-LIST-"]:
            selected_idx = values["-PRESET-LIST-"].index(values["-PRESET-LIST-"][0])
            if selected_idx in preset_id_map:
                preset_id = preset_id_map[selected_idx]
                return delete_preset_from_ui(window, preset_id, values["-PRESET-BUCKET-"])
        return True
    
    # Refresh presets list
    elif event == "-REFRESH-PRESETS-":
        refresh_preset_list(window, values["-PRESET-BUCKET-"])
        return True
    
    # Save current settings as preset
    elif event == "-SAVE-PRESET-":
        return save_current_settings(window, values)
    
    # Suggestion filtering
    elif event in ["-FILTER-OVERALL-", "-FILTER-PROFIT-", "-FILTER-RISK-"]:
        update_suggestion_list(window, values["-PRESET-BUCKET-"], 
                              get_filter_type(values))
        return True
    
    # Suggestion list selection
    elif event == "-SUGGESTION-LIST-":
        if values["-SUGGESTION-LIST-"]:
            window["-LOAD-SUGGESTION-"].update(disabled=False)
        else:
            window["-LOAD-SUGGESTION-"].update(disabled=True)
        return True
    
    # Load selected suggestion
    elif event == "-LOAD-SUGGESTION-":
        if values["-SUGGESTION-LIST-"]:
            selected_idx = values["-SUGGESTION-LIST-"].index(values["-SUGGESTION-LIST-"][0])
            if selected_idx in suggestion_id_map:
                preset_id = suggestion_id_map[selected_idx]
                return load_preset_into_ui(window, preset_id)
        return True
    
    # Clear suggestion selection
    elif event == "-CLEAR-SUGGESTION-":
        window["-SUGGESTION-LIST-"].update(set_to_index=[])
        window["-LOAD-SUGGESTION-"].update(disabled=True)
        return True
    
    # Refresh suggestions
    elif event == "-REFRESH-SUGGESTIONS-":
        update_suggestion_list(window, values["-PRESET-BUCKET-"], 
                              get_filter_type(values))
        return True
    
    # Quick View Performance
    elif event == "-QUICK-VIEW-PERFORMANCE-":
        if values["-PRESET-LIST-"]:
            selected_idx = values["-PRESET-LIST-"].index(values["-PRESET-LIST-"][0])
            if selected_idx in preset_id_map:
                preset_id = preset_id_map[selected_idx]
                from src.ui.preset_manager import show_simplified_performance_history
                show_simplified_performance_history(preset_id)
        return True
    
    # Show performance history for current preset
    elif event == "-SHOW-PERFORMANCE-":
        if current_preset_id:
            show_performance_history(current_preset_id)
        return True
        
    # List temporary presets
    elif event == "-LIST-TEMP-PRESETS-":
        list_temporary_presets(window)
        return True
        
    # Keep selected preset (move from temporary to permanent)
    elif event == "-KEEP-PRESET-":
        if values["-PRESET-LIST-"]:
            selected_idx = values["-PRESET-LIST-"].index(values["-PRESET-LIST-"][0])
            if selected_idx in preset_id_map:
                preset_id = preset_id_map[selected_idx]
                keep_temporary_preset(window, preset_id, values["-PRESET-BUCKET-"])
        return True
        
    # Clean old temporary presets now
    elif event == "-CLEAN-TEMP-PRESETS-":
        clean_temporary_presets(window, values["-PRESET-BUCKET-"])
        return True
    
    return False

def get_filter_type(values):
    """Get the current filter type from values"""
    if values.get("-FILTER-PROFIT-"):
        return "profit"
    elif values.get("-FILTER-RISK-"):
        return "risk"
    else:
        return "overall"

def refresh_preset_list(window, bucket):
    """
    Refresh the preset list for a specific bucket
    
    Args:
        window: The PySimpleGUI window
        bucket: The bucket to show presets for
    """
    global preset_id_map
    
    # Get presets for this bucket
    presets = list_presets(bucket=bucket)
    
    # Format for display
    formatted_presets = []
    preset_id_map = {}
    
    for i, preset in enumerate(presets):
        preset_name = preset.get("name", "Unknown")
        preset_type = preset.get("type", "user").capitalize()
        preset_desc = preset.get("description", "")
        
        # Format string
        display_text = f"{preset_name} ({preset_type})"
        if preset_desc:
            display_text += f" - {preset_desc}"
        
        formatted_presets.append(display_text)
        preset_id_map[i] = preset.get("id")
    
    # Update the listbox
    if "-PRESET-LIST-" in window.AllKeysDict:
        window["-PRESET-LIST-"].update(formatted_presets)
        window["-LOAD-PRESET-"].update(disabled=True)
        window["-DELETE-PRESET-"].update(disabled=True)
    
    return formatted_presets

def load_default_preset(window, bucket):
    """
    Load a default preset for a bucket
    
    Args:
        window: The PySimpleGUI window
        bucket: The bucket to load default for
        
    Returns:
        bool: True if successful
    """
    # Import here to avoid circular imports
    from src.ui.preset_manager import DEFAULT_PRESETS_DIR
    
    # Find the default preset for this bucket
    bucket_dir = os.path.join(DEFAULT_PRESETS_DIR, bucket)
    if not os.path.exists(bucket_dir):
        sg.popup_error(f"No default presets found for {bucket} bucket")
        return False
    
    # Get the first preset in this directory
    preset_files = [os.path.join(bucket_dir, f) for f in os.listdir(bucket_dir) 
                   if f.endswith('.json')]
    
    if not preset_files:
        sg.popup_error(f"No default presets found for {bucket} bucket")
        return False
    
    # Load the first preset
    return load_preset_into_ui(window, preset_files[0])

def load_preset_into_ui(window, preset_id):
    """
    Load a preset's parameters into the UI
    
    Args:
        window: The PySimpleGUI window
        preset_id: ID of the preset to load
        
    Returns:
        bool: True if successful
    """
    global current_preset_id
    
    # Load the preset
    preset_data = load_preset(preset_id)
    if not preset_data:
        sg.popup_error(f"Failed to load preset: {preset_id}")
        return False
    
    # Get params
    params = preset_data.get("params", {})
    if not params:
        sg.popup_error(f"Preset has no parameters: {preset_id}")
        return False
    
    # Update UI elements with preset values
    for key, value in params.items():
        if key in window.AllKeysDict:
            window[key].update(value)
    
    # Set current bucket from preset if available
    if "BUCKET" in params:
        if "BUCKET" in window.AllKeysDict:
            window["BUCKET"].update(params["BUCKET"])
    
    # Update the current preset ID
    current_preset_id = preset_id
    
    # Update the window title to include preset name
    preset_name = preset_data.get("name", os.path.basename(preset_id).replace('.json', ''))
    window.set_title(f"Trading System - Preset: {preset_name}")
    
    # Update status
    if "-STATUS-" in window.AllKeysDict:
        window["-STATUS-"].update(f"Loaded preset: {preset_name}")
    
    return True

def delete_preset_from_ui(window, preset_id, current_bucket):
    """
    Delete a preset and refresh the UI
    
    Args:
        window: The PySimpleGUI window
        preset_id: ID of the preset to delete
        current_bucket: Current bucket selected
        
    Returns:
        bool: True if successful
    """
    global current_preset_id
    
    # Confirm deletion
    preset_data = load_preset(preset_id)
    preset_name = preset_data.get("name", os.path.basename(preset_id).replace('.json', ''))
    
    if not sg.popup_yes_no(f"Are you sure you want to delete preset: {preset_name}?"):
        return False
    
    # Delete the preset
    success = delete_preset(preset_id)
    if not success:
        sg.popup_error(f"Failed to delete preset: {preset_id}")
        return False
    
    # If this is the current preset, clear current preset ID
    if current_preset_id == preset_id:
        current_preset_id = None
        # Restore original window title
        window.set_title("Trading System")
    
    # Refresh preset list
    refresh_preset_list(window, current_bucket)
    
    # Update status
    if "-STATUS-" in window.AllKeysDict:
        window["-STATUS-"].update(f"Deleted preset: {preset_name}")
    
    return True

def save_current_settings(window, values):
    """
    Save current settings as a preset
    
    Args:
        window: The PySimpleGUI window
        values: The current values dictionary
        
    Returns:
        bool: True if successful
    """
    global current_preset_id
    
    # Get bucket, name, and description
    bucket = values.get("-PRESET-BUCKET-", values.get("BUCKET", "Scalping"))
    name = values.get("-PRESET-NAME-", "")
    description = values.get("-PRESET-DESC-", "")
    is_temporary = values.get("-TEMP-PRESET-", False)
    
    # Validate name
    if not name:
        sg.popup_error("Please enter a name for the preset")
        return False
    
    # Collect current parameters
    params = {}
    
    # Add bucket
    params["BUCKET"] = bucket
    
    # Add bucket-specific parameters
    if bucket == "Scalping":
        if "monthly_target_min" in values:
            params["monthly_target_min"] = float(values["monthly_target_min"])
        if "monthly_target_max" in values:
            params["monthly_target_max"] = float(values["monthly_target_max"])
    
    elif bucket == "Short":
        if "yearly_target_min" in values:
            params["yearly_target_min"] = float(values["yearly_target_min"])
        if "yearly_target_max" in values:
            params["yearly_target_max"] = float(values["yearly_target_max"])
    
    elif bucket in ["Medium", "Long"]:
        # Keys are specific to bucket type
        gain_min_key = f"min_gain_per_holding_{bucket.lower()}"
        gain_max_key = f"max_gain_per_holding_{bucket.lower()}"
        bonus_key = f"bonus_multiplier_{bucket.lower()}"
        
        if gain_min_key in values:
            params[gain_min_key] = float(values[gain_min_key])
        if gain_max_key in values:
            params[gain_max_key] = float(values[gain_max_key])
        if bonus_key in values:
            params[bonus_key] = float(values[bonus_key])
    
    # Add additional params if they exist in values
    for key in ["use_advanced_features", "risk_tolerance", "trade_frequency"]:
        if key in values:
            params[key] = values[key]
    
    # Save the preset
    preset_id = save_preset(
        bucket=bucket,
        name=name,
        params=params,
        description=description,
        is_temporary=is_temporary
    )
    
    if not preset_id:
        sg.popup_error("Failed to save preset")
        return False
    
    # Update current preset ID
    current_preset_id = preset_id
    
    # Refresh preset list
    refresh_preset_list(window, bucket)
    
    # Update status
    if "-STATUS-" in window.AllKeysDict:
        window["-STATUS-"].update(f"Saved preset: {name}")
    
    # Update window title
    window.set_title(f"Trading System - Preset: {name}")
    
    # Show confirmation
    sg.popup(f"Preset '{name}' saved successfully", title="Preset Saved")
    
    return True

def get_current_preset_id():
    """Get the ID of the currently loaded preset"""
    global current_preset_id
    return current_preset_id 

def list_temporary_presets(window):
    """
    List temporary presets in the UI
    
    Args:
        window: The PySimpleGUI window
    """
    # Import needed functions
    from src.ui.preset_manager import list_presets, TEMP_PRESETS_DIR
    
    # Get all temporary presets
    temp_presets = list_presets(include_defaults=False, include_user=False, include_temp=True)
    
    if not temp_presets:
        sg.popup_quick_message("No temporary presets found", auto_close_duration=2)
        return
    
    # Format presets for display
    formatted_presets = []
    temp_preset_map = {}
    
    for i, preset in enumerate(temp_presets):
        preset_name = preset.get("name", "Unknown")
        bucket = preset.get("bucket", "Unknown")
        created = preset.get("created", "Unknown")
        
        # Try to parse and format the date
        try:
            date_obj = datetime.datetime.fromisoformat(created)
            date_str = date_obj.strftime("%Y-%m-%d %H:%M")
        except:
            date_str = created
        
        # Format string
        display_text = f"{preset_name} (Bucket: {bucket}) - Created: {date_str}"
        
        formatted_presets.append(display_text)
        temp_preset_map[i] = preset.get("id")
    
    # Show in a popup list
    layout = [
        [sg.Text("Temporary Presets", font=("Helvetica", 14))],
        [sg.Listbox(values=formatted_presets, size=(70, 10), key="-TEMP-PRESETS-LIST-", enable_events=True)],
        [sg.Button("Load"), sg.Button("Keep", key="KEEP"), sg.Button("Delete"), sg.Button("Close")]
    ]
    
    temp_window = sg.Window("Temporary Presets", layout, modal=True, finalize=True)
    
    # Event loop
    while True:
        event, values = temp_window.read()
        
        if event in (sg.WIN_CLOSED, "Close"):
            break
        
        elif event == "-TEMP-PRESETS-LIST-":
            # Enable buttons if a preset is selected
            selected = len(values["-TEMP-PRESETS-LIST-"]) > 0
            temp_window["Load"].update(disabled=not selected)
            temp_window["Keep"].update(disabled=not selected)
            temp_window["Delete"].update(disabled=not selected)
        
        elif event == "Load" and values["-TEMP-PRESETS-LIST-"]:
            # Get selected preset
            selected_idx = values["-TEMP-PRESETS-LIST-"].index(values["-TEMP-PRESETS-LIST-"][0])
            if selected_idx in temp_preset_map:
                preset_id = temp_preset_map[selected_idx]
                
                # Load this preset
                loaded = load_preset_into_ui(window, preset_id)
                if loaded:
                    sg.popup_quick_message("Preset loaded successfully", auto_close_duration=2)
                    temp_window.close()
                    return
                else:
                    sg.popup_error("Failed to load preset")
        
        elif event == "Keep" and values["-TEMP-PRESETS-LIST-"]:
            # Get selected preset
            selected_idx = values["-TEMP-PRESETS-LIST-"].index(values["-TEMP-PRESETS-LIST-"][0])
            if selected_idx in temp_preset_map:
                preset_id = temp_preset_map[selected_idx]
                
                # Keep this preset
                kept = keep_temporary_preset(window, preset_id)
                if kept:
                    # Remove from list
                    temp_presets = [p for i, p in enumerate(temp_presets) if i != selected_idx]
                    formatted_presets = [p for i, p in enumerate(formatted_presets) if i != selected_idx]
                    temp_window["-TEMP-PRESETS-LIST-"].update(formatted_presets)
        
        elif event == "Delete" and values["-TEMP-PRESETS-LIST-"]:
            # Get selected preset
            selected_idx = values["-TEMP-PRESETS-LIST-"].index(values["-TEMP-PRESETS-LIST-"][0])
            if selected_idx in temp_preset_map:
                preset_id = temp_preset_map[selected_idx]
                
                # Confirm deletion
                if sg.popup_yes_no(f"Are you sure you want to delete this preset?") == "Yes":
                    # Delete preset
                    from src.ui.preset_manager import delete_preset
                    deleted = delete_preset(preset_id)
                    
                    if deleted:
                        sg.popup_quick_message("Preset deleted", auto_close_duration=2)
                        
                        # Remove from list
                        temp_presets = [p for i, p in enumerate(temp_presets) if i != selected_idx]
                        formatted_presets = [p for i, p in enumerate(formatted_presets) if i != selected_idx]
                        temp_window["-TEMP-PRESETS-LIST-"].update(formatted_presets)
                    else:
                        sg.popup_error("Failed to delete preset")
    
    temp_window.close()

def keep_temporary_preset(window, preset_id, current_bucket=None):
    """
    Move a temporary preset to the permanent user presets
    
    Args:
        window: The PySimpleGUI window
        preset_id: ID of the preset to keep
        current_bucket: Current bucket selected
        
    Returns:
        bool: True if successful
    """
    # Import needed functions
    from src.ui.preset_manager import load_preset, save_preset, delete_preset
    
    # Load the preset data
    preset_data = load_preset(preset_id)
    if not preset_data:
        sg.popup_error(f"Failed to load preset: {preset_id}")
        return False
    
    # Extract preset info
    name = preset_data.get("name", "Unknown")
    params = preset_data.get("params", {})
    description = preset_data.get("description", "")
    bucket = params.get("BUCKET", current_bucket or "Scalping")
    
    # Save as permanent preset
    new_preset_id = save_preset(
        bucket=bucket,
        name=name,
        params=params,
        description=description,
        is_temporary=False  # Save as permanent
    )
    
    if not new_preset_id:
        sg.popup_error("Failed to save preset as permanent")
        return False
    
    # Delete the temporary preset
    deleted = delete_preset(preset_id)
    if not deleted:
        logger.warning(f"Failed to delete temporary preset {preset_id} after keeping it")
    
    # Refresh preset list
    if window and current_bucket:
        refresh_preset_list(window, current_bucket)
    
    # Show confirmation
    sg.popup_quick_message(f"Preset '{name}' is now permanent", auto_close_duration=2)
    
    return True

def clean_temporary_presets(window, current_bucket=None):
    """
    Clean up old temporary presets
    
    Args:
        window: The PySimpleGUI window
        current_bucket: Current bucket selected
        
    Returns:
        bool: True if successful
    """
    # Import needed function
    from src.ui.preset_manager import cleanup_temp_presets
    
    # Ask for days threshold
    days = sg.popup_get_text("Delete temporary presets older than how many days?", 
                           default_text="7",
                           title="Clean Temporary Presets")
    
    if not days:
        return False
    
    try:
        days = int(days)
        if days <= 0:
            raise ValueError("Days must be positive")
    except ValueError:
        sg.popup_error("Please enter a valid positive number of days")
        return False
    
    # Perform cleanup
    deleted_count = cleanup_temp_presets(days)
    
    # Show confirmation
    if deleted_count > 0:
        sg.popup_quick_message(f"Cleaned up {deleted_count} temporary presets older than {days} days", 
                             auto_close_duration=2)
    else:
        sg.popup_quick_message(f"No temporary presets found older than {days} days",
                             auto_close_duration=2)
    
    # Refresh preset list
    if window and current_bucket:
        refresh_preset_list(window, current_bucket)
    
    return True 