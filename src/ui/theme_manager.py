"""
Theme Manager Module

This module handles functionality related to UI theming,
including light/dark mode switching and color scheme customization.
"""

import os
import sys
import json
import logging
import PySimpleGUI as sg
from typing import Dict, Any, List, Optional, Tuple

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

# Set up logger
logger = logging.getLogger(__name__)

# Constants for theme management
DEFAULT_THEME = "DarkBlue3"
DEFAULT_THEME_MODE = "dark"
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config"))
THEME_CONFIG_FILE = os.path.join(CONFIG_DIR, "theme_config.json")

# Light and dark theme definitions
THEME_DEFINITIONS = {
    "light": {
        "name": "LightTheme",
        "colors": {
            "BACKGROUND": "#f0f0f0",
            "TEXT": "#000000",
            "INPUT": "#ffffff",
            "TEXT_INPUT": "#000000",
            "SCROLL": "#c7e0f0",
            "BUTTON": ("#000000", "#e1e1e1"),
            "PROGRESS": ("#d7d7d7", "#c2d5e3"),
            "BORDER": 1,
            "SLIDER_DEPTH": 0,
            "PROGRESS_DEPTH": 0,
            "COLOR_LIST": ["#f0f0f0", "#f0f0f0", "#f0f0f0", "#e1e1e1", "#000000", "#5e95c8", "#dddddd", "#333333"]
        },
        "button_color": "#e1e1e1",
        "text_color": "#000000",
        "background_color": "#f0f0f0",
        "input_background_color": "#ffffff",
        "input_text_color": "#000000"
    },
    "dark": {
        "name": "DarkTheme",
        "colors": {
            "BACKGROUND": "#1e1e1e",
            "TEXT": "#e0e0e0",
            "INPUT": "#343434",
            "TEXT_INPUT": "#e0e0e0",
            "SCROLL": "#4c5b6e",
            "BUTTON": ("#e0e0e0", "#2d2d2d"),
            "PROGRESS": ("#404040", "#4c5b6e"),
            "BORDER": 1,
            "SLIDER_DEPTH": 0,
            "PROGRESS_DEPTH": 0,
            "COLOR_LIST": ["#1e1e1e", "#1e1e1e", "#1e1e1e", "#2d2d2d", "#e0e0e0", "#5e95c8", "#444444", "#dddddd"]
        },
        "button_color": "#2d2d2d",
        "text_color": "#e0e0e0",
        "background_color": "#1e1e1e",
        "input_background_color": "#343434",
        "input_text_color": "#e0e0e0"
    }
}

# PySimpleGUI built-in themes mapping
PYSG_THEMES = {
    "light": ["Default", "DefaultNoMoreNagging", "Material1", "Material2", "Reddit", "Topanga", "LightGrey", "LightGreen", "LightBlue"],
    "dark": ["DarkBlue3", "DarkAmber", "DarkBrown", "DarkGreen", "DarkGrey", "DarkTeal", "DarkPurple", "DarkBlack", "DarkBlue"]
}

class ThemeManager:
    """Class to manage UI theming"""
    
    def __init__(self, app_state=None):
        """
        Initialize the ThemeManager.
        
        Args:
            app_state: Optional reference to the application state
        """
        self.app_state = app_state
        self.current_mode = DEFAULT_THEME_MODE
        self.current_theme = DEFAULT_THEME
        self.custom_theme = None
        self.custom_colors = {}
        
        # Load theme configuration
        self._load_theme_config()
        
        # Apply the theme immediately after loading
        self.apply_theme()
        
        logger.info(f"ThemeManager initialized with '{self.current_theme}' theme")
    
    def _load_theme_config(self):
        """
        Load theme configuration from settings file.
        """
        try:
            # Try to load from app_state if available
            if self.app_state and hasattr(self.app_state, "config") and "theme_config" in self.app_state.config:
                config = self.app_state.config["theme_config"]
                self.current_mode = config.get("mode", DEFAULT_THEME_MODE)
                self.current_theme = config.get("theme", DEFAULT_THEME)
                self.custom_theme = config.get("custom_theme", None)
                self.custom_colors = config.get("custom_colors", {})
                return
            
            # Try to load from config file
            if os.path.exists(THEME_CONFIG_FILE):
                with open(THEME_CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.current_mode = config.get("mode", DEFAULT_THEME_MODE)
                    self.current_theme = config.get("theme", DEFAULT_THEME)
                    self.custom_theme = config.get("custom_theme", None)
                    self.custom_colors = config.get("custom_colors", {})
            else:
                # Use defaults
                self.current_mode = DEFAULT_THEME_MODE
                self.current_theme = DEFAULT_THEME
                self.custom_theme = None
                self.custom_colors = {}
                
                # Save default configuration
                self.save_theme_config()
                
        except Exception as e:
            logger.error(f"Error loading theme configuration: {e}")
            # Use defaults
            self.current_mode = DEFAULT_THEME_MODE
            self.current_theme = DEFAULT_THEME
            self.custom_theme = None
            self.custom_colors = {}
    
    def save_theme_config(self):
        """
        Save theme configuration to settings file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Save to app_state if available
            if self.app_state and hasattr(self.app_state, "config"):
                self.app_state.config["theme_config"] = {
                    "mode": self.current_mode,
                    "theme": self.current_theme,
                    "custom_theme": self.custom_theme,
                    "custom_colors": self.custom_colors
                }
                
                # If app_state has save_config method, use it
                if hasattr(self.app_state, "save_config"):
                    self.app_state.save_config()
            
            # Also save to config file
            config = {
                "mode": self.current_mode,
                "theme": self.current_theme,
                "custom_theme": self.custom_theme,
                "custom_colors": self.custom_colors
            }
            
            os.makedirs(CONFIG_DIR, exist_ok=True)
            with open(THEME_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.info("Theme configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving theme configuration: {e}")
            if error_handling_available:
                log_persistent_error(
                    e,
                    context="Saving theme configuration",
                    severity=ErrorSeverity.MEDIUM
                )
            return False
    
    def apply_theme(self):
        """
        Apply the current theme to PySimpleGUI.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # If using a custom theme, register it first
            if self.custom_theme:
                self._register_custom_theme()
                sg.theme(self.custom_theme)
            else:
                # Use a built-in theme
                sg.theme(self.current_theme)
            
            logger.info(f"Applied theme: {self.current_theme if not self.custom_theme else self.custom_theme}")
            return True
        except Exception as e:
            logger.error(f"Error applying theme: {e}")
            if error_handling_available:
                log_persistent_error(
                    e,
                    context="Applying theme",
                    severity=ErrorSeverity.LOW
                )
            # Try to fall back to default theme
            try:
                sg.theme(DEFAULT_THEME)
                logger.info(f"Fell back to default theme: {DEFAULT_THEME}")
                return True
            except:
                return False
    
    def _register_custom_theme(self):
        """
        Register a custom theme with PySimpleGUI.
        """
        try:
            # Use a predefined theme as base if custom_theme is a string from THEME_DEFINITIONS
            if self.custom_theme in THEME_DEFINITIONS:
                theme_def = THEME_DEFINITIONS[self.custom_theme]
            else:
                # Use dark theme as base for custom themes
                theme_def = THEME_DEFINITIONS["dark"].copy()
                
                # Override with custom colors if provided
                for key, value in self.custom_colors.items():
                    if key in theme_def["colors"]:
                        theme_def["colors"][key] = value
            
            # Register the theme
            sg.LOOK_AND_FEEL_TABLE[self.custom_theme] = theme_def["colors"]
            logger.info(f"Registered custom theme: {self.custom_theme}")
        except Exception as e:
            logger.error(f"Error registering custom theme: {e}")
            if error_handling_available:
                log_persistent_error(
                    e,
                    context="Registering custom theme",
                    severity=ErrorSeverity.LOW
                )
    
    def set_theme(self, theme_name):
        """
        Set a new theme by name.
        
        Args:
            theme_name: Name of the theme to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if it's a valid theme
            if theme_name in sg.theme_list():
                self.current_theme = theme_name
                self.custom_theme = None  # Not using custom theme
                
                # Determine mode based on theme name
                if theme_name in PYSG_THEMES["light"]:
                    self.current_mode = "light"
                elif theme_name in PYSG_THEMES["dark"]:
                    self.current_mode = "dark"
                
                # Apply the theme
                self.apply_theme()
                
                # Save the configuration
                self.save_theme_config()
                
                logger.info(f"Set theme to {theme_name}")
                return True
            else:
                logger.warning(f"Invalid theme name: {theme_name}")
                return False
        except Exception as e:
            logger.error(f"Error setting theme: {e}")
            if error_handling_available:
                log_persistent_error(
                    e,
                    context="Setting theme",
                    severity=ErrorSeverity.LOW
                )
            return False
    
    def toggle_dark_mode(self):
        """
        Toggle between light and dark mode.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Toggle the mode
            new_mode = "light" if self.current_mode == "dark" else "dark"
            
            # Choose an appropriate theme for the new mode
            if self.custom_theme:
                # Toggle between light and dark custom themes
                self.custom_theme = new_mode
            else:
                # Use the first theme in the list for the new mode
                self.current_theme = PYSG_THEMES[new_mode][0]
            
            # Update current mode
            self.current_mode = new_mode
            
            # Apply the theme
            self.apply_theme()
            
            # Save the configuration
            self.save_theme_config()
            
            logger.info(f"Toggled to {new_mode} mode with theme {self.current_theme}")
            return True
        except Exception as e:
            logger.error(f"Error toggling dark mode: {e}")
            if error_handling_available:
                log_persistent_error(
                    e,
                    context="Toggling dark mode",
                    severity=ErrorSeverity.LOW
                )
            return False
    
    def set_custom_theme(self, name, colors):
        """
        Set a custom theme with specific colors.
        
        Args:
            name: Name for the custom theme
            colors: Dictionary of color definitions
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.custom_theme = name
            self.custom_colors = colors
            
            # Register and apply the theme
            self._register_custom_theme()
            self.apply_theme()
            
            # Save the configuration
            self.save_theme_config()
            
            logger.info(f"Applied custom theme: {name}")
            return True
        except Exception as e:
            logger.error(f"Error setting custom theme: {e}")
            if error_handling_available:
                log_persistent_error(
                    e,
                    context="Setting custom theme",
                    severity=ErrorSeverity.LOW
                )
            return False
    
    def get_current_theme(self):
        """
        Get the current theme name.
        
        Returns:
            str: Current theme name
        """
        return self.custom_theme if self.custom_theme else self.current_theme
    
    def get_theme_mode(self):
        """
        Get the current theme mode.
        
        Returns:
            str: "light" or "dark"
        """
        return self.current_mode
    
    def is_dark_mode(self):
        """
        Check if dark mode is active.
        
        Returns:
            bool: True if dark mode is active, False otherwise
        """
        return self.current_mode == "dark"
    
    def get_available_themes(self, mode=None):
        """
        Get a list of available themes.
        
        Args:
            mode: Optional filter by mode ("light" or "dark")
            
        Returns:
            list: List of theme names
        """
        if mode:
            return PYSG_THEMES.get(mode, [])
        else:
            # Return all themes
            return sg.theme_list()
    
    def show_theme_selector(self, window=None):
        """
        Show a theme selector dialog.
        
        Args:
            window: Optional parent window
            
        Returns:
            str or None: Selected theme name or None if cancelled
        """
        # Get all available themes
        all_themes = sg.theme_list()
        
        # Create theme previews
        light_themes = [[sg.Text(theme), sg.Button("Preview", key=f"-PREVIEW-{theme}-"), 
                       sg.Button("Select", key=f"-SELECT-{theme}-")] 
                      for theme in PYSG_THEMES["light"]]
        
        dark_themes = [[sg.Text(theme), sg.Button("Preview", key=f"-PREVIEW-{theme}-"), 
                      sg.Button("Select", key=f"-SELECT-{theme}-")] 
                     for theme in PYSG_THEMES["dark"]]
        
        # Create the layout
        layout = [
            [sg.Text("Theme Selector", font=("Helvetica", 16))],
            [sg.TabGroup([
                [sg.Tab("Light Themes", light_themes, key="-LIGHT-THEMES-")],
                [sg.Tab("Dark Themes", dark_themes, key="-DARK-THEMES-")]
            ])],
            [sg.Button("Toggle Dark Mode", key="-TOGGLE-DARK-"), sg.Button("Cancel", key="-CANCEL-")]
        ]
        
        # Create the window
        selector_window = sg.Window("Theme Selector", layout, modal=True, finalize=True)
        
        # Event loop
        selected_theme = None
        while True:
            event, values = selector_window.read()
            
            if event == sg.WIN_CLOSED or event == "-CANCEL-":
                break
                
            elif event == "-TOGGLE-DARK-":
                self.toggle_dark_mode()
                if window:
                    window.TKroot.update()  # Refresh the parent window
                
                # Also refresh the selector window
                selector_window.close()
                selected_theme = self.show_theme_selector(window)
                break
                
            elif event.startswith("-PREVIEW-"):
                # Extract theme name from the event
                theme_name = event.replace("-PREVIEW-", "").replace("-", "")
                
                # Apply the theme temporarily
                original_theme = self.get_current_theme()
                self.set_theme(theme_name)
                
                # Show a preview window
                preview_layout = [
                    [sg.Text(f"Preview of {theme_name} theme", font=("Helvetica", 12))],
                    [sg.Input("Sample input field", key="-SAMPLE-INPUT-")],
                    [sg.Combo(["Option 1", "Option 2", "Option 3"], default_value="Option 1", key="-SAMPLE-COMBO-")],
                    [sg.Slider(range=(1, 100), default_value=50, orientation="h", key="-SAMPLE-SLIDER-")],
                    [sg.Button("Sample Button"), sg.Button("Close Preview")]
                ]
                
                preview_window = sg.Window(f"Theme Preview: {theme_name}", preview_layout, modal=True, finalize=True)
                
                # Simple event loop for preview
                while True:
                    preview_event, _ = preview_window.read()
                    if preview_event in (sg.WIN_CLOSED, "Close Preview"):
                        break
                
                preview_window.close()
                
                # Restore original theme
                self.set_theme(original_theme)
                
            elif event.startswith("-SELECT-"):
                # Extract theme name from the event
                theme_name = event.replace("-SELECT-", "").replace("-", "")
                
                # Set the selected theme
                self.set_theme(theme_name)
                selected_theme = theme_name
                
                # Refresh parent window if provided
                if window:
                    window.TKroot.update()
                
                break
        
        selector_window.close()
        return selected_theme
    
    def get_theme_color(self, color_key):
        """
        Get a specific color from the current theme.
        
        Args:
            color_key: The color key to retrieve
            
        Returns:
            str: The color value or None if not found
        """
        try:
            if self.custom_theme in THEME_DEFINITIONS:
                return THEME_DEFINITIONS[self.custom_theme]["colors"].get(color_key)
            elif self.custom_theme:
                return self.custom_colors.get(color_key)
            else:
                # For built-in themes, return None as we don't have direct access to their color values
                return None
        except Exception as e:
            logger.error(f"Error getting theme color: {e}")
            return None

# Global instance for easier access
_theme_manager = None

def get_theme_manager(app_state=None):
    """
    Get the global theme manager instance.
    
    Args:
        app_state: Optional application state instance
        
    Returns:
        ThemeManager: Global theme manager instance
    """
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager(app_state)
    return _theme_manager

def set_theme(theme_name):
    """
    Set a theme by name.
    
    Args:
        theme_name: Name of the theme to use
        
    Returns:
        bool: True if successful, False otherwise
    """
    theme_manager = get_theme_manager()
    return theme_manager.set_theme(theme_name)

def toggle_dark_mode():
    """
    Toggle between light and dark mode.
    
    Returns:
        bool: True if successful, False otherwise
    """
    theme_manager = get_theme_manager()
    return theme_manager.toggle_dark_mode()

def get_current_theme():
    """
    Get the current theme name.
    
    Returns:
        str: Current theme name
    """
    theme_manager = get_theme_manager()
    return theme_manager.get_current_theme()

def is_dark_mode():
    """
    Check if dark mode is active.
    
    Returns:
        bool: True if dark mode is active, False otherwise
    """
    theme_manager = get_theme_manager()
    return theme_manager.is_dark_mode()

def show_theme_selector(window=None):
    """
    Show a theme selector dialog.
    
    Args:
        window: Optional parent window
        
    Returns:
        str or None: Selected theme name or None if cancelled
    """
    theme_manager = get_theme_manager()
    return theme_manager.show_theme_selector(window) 