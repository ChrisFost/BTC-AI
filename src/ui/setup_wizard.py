import PySimpleGUI as sg
import os
import json
import sys
import base64
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Get the base directory (project root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Add project root to path so imports work correctly
sys.path.insert(0, project_root)

from src.utils.log_manager import LogManager

# Initialize logger
logger = LogManager.get_logger("setup_wizard")

# Define paths
CONFIG_FILE = os.path.join(project_root, "configs", "config.json")
WIZARD_COMPLETE_FILE = os.path.join(project_root, "configs", "wizard_complete.json")

# Load the blue partyhat icon directly
ICON_PATH = os.path.join(os.path.dirname(__file__), "icons", "blue_partyhat.png")
HAS_ICON = os.path.exists(ICON_PATH)
if HAS_ICON:
    try:
        with open(ICON_PATH, 'rb') as f:
            BLUE_PHAT_ICON = base64.b64encode(f.read())
        logger.info("Blue partyhat icon loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load blue partyhat icon: {e}")
        HAS_ICON = False
        BLUE_PHAT_ICON = None
else:
    logger.warning("Blue partyhat icon file not found")
    BLUE_PHAT_ICON = None

class RSTheme:
    """RuneScape style theming for PySimpleGUI"""
    
    # Colors based on OSRS interface
    OSRS_BROWN = '#3E3529'
    OSRS_DARK_BROWN = '#2E2921'
    OSRS_LIGHT_BROWN = '#6B5F4D'
    OSRS_GOLD = '#DAA520'
    OSRS_TEXT = '#FFC34D'
    OSRS_BUTTON = '#47422F'
    OSRS_BUTTON_TEXT = '#FFD700'
    
    @staticmethod
    def apply_theme():
        """Apply OSRS theme to PySimpleGUI"""
        sg.theme_background_color(RSTheme.OSRS_BROWN)
        sg.theme_text_element_background_color(RSTheme.OSRS_BROWN)
        sg.theme_button_color((RSTheme.OSRS_BUTTON_TEXT, RSTheme.OSRS_BUTTON))
        sg.theme_text_color(RSTheme.OSRS_TEXT)
        sg.theme_input_background_color(RSTheme.OSRS_DARK_BROWN)
        sg.theme_input_text_color(RSTheme.OSRS_TEXT)
    
    @staticmethod
    def get_rs_button(text):
        """Create a RuneScape style button"""
        return sg.Button(
            text,
            button_color=(RSTheme.OSRS_BUTTON_TEXT, RSTheme.OSRS_BUTTON),
            border_width=2
        )
    
    @staticmethod
    def get_rs_text(text, size=(None, None), font_size=12, bold=False):
        """Create a RuneScape style text element"""
        return sg.Text(
            text,
            background_color=RSTheme.OSRS_BROWN,
            text_color=RSTheme.OSRS_TEXT,
            size=size
        )
    
    @staticmethod
    def get_rs_input(default_text="", key=None, size=(20, 1)):
        """Create a RuneScape style input element"""
        return sg.Input(
            default_text=default_text,
            key=key,
            background_color=RSTheme.OSRS_DARK_BROWN,
            text_color=RSTheme.OSRS_TEXT,
            size=size,
            border_width=2
        )

def create_welcome_layout():
    """Create the welcome page layout"""
    layout = [
        [sg.Column([
            [RSTheme.get_rs_text("Welcome to BTC Trading AI", font_size=16, bold=True)],
            [RSTheme.get_rs_text("First Time Setup Wizard", font_size=14)],
            [sg.HorizontalSeparator(color=RSTheme.OSRS_GOLD, pad=(5, 15))],
            [RSTheme.get_rs_text("Greetings, adventurer! I shall be your guide through")],
            [RSTheme.get_rs_text("the setup of your Bitcoin trading journey.")],
            [RSTheme.get_rs_text("")],
            [RSTheme.get_rs_text("This wizard will help you configure your trading")],
            [RSTheme.get_rs_text("system for optimal performance.")],
            [RSTheme.get_rs_text("")],
            [RSTheme.get_rs_text("Are you ready to begin your quest?")],
            [sg.Column([[RSTheme.get_rs_button("Begin Adventure")]], justification='right', pad=(0, 20))]
        ], background_color=RSTheme.OSRS_BROWN, justification='center', pad=(20, 20))]
    ]
    return layout

def create_trading_style_layout():
    """Create the trading style selection layout"""
    layout = [
        [sg.Column([
            [RSTheme.get_rs_text("Choose Your Trading Style", font_size=16, bold=True)],
            [sg.HorizontalSeparator(color=RSTheme.OSRS_GOLD, pad=(5, 15))],
            [RSTheme.get_rs_text("Select which trading style suits you best:")],
            [RSTheme.get_rs_text("")],
            [sg.Radio("Scalping - Quick trades, small profits", "TRADING_STYLE", key="-SCALPING-", 
                      background_color=RSTheme.OSRS_BROWN, text_color=RSTheme.OSRS_TEXT, font=('Runescape UF', 11),
                      default=True)],
            [RSTheme.get_rs_text("Like a nimble thief, you make quick moves for small gains.")],
            [RSTheme.get_rs_text("")],
            [sg.Radio("Short-Term - Hours to days", "TRADING_STYLE", key="-SHORT-", 
                      background_color=RSTheme.OSRS_BROWN, text_color=RSTheme.OSRS_TEXT, font=('Runescape UF', 11))],
            [RSTheme.get_rs_text("You're a skilled hunter, patient enough for the right opportunity.")],
            [RSTheme.get_rs_text("")],
            [sg.Radio("Medium-Term - Days to weeks", "TRADING_STYLE", key="-MEDIUM-", 
                      background_color=RSTheme.OSRS_BROWN, text_color=RSTheme.OSRS_TEXT, font=('Runescape UF', 11))],
            [RSTheme.get_rs_text("Like a master craftsman, you build your wealth with care.")],
            [RSTheme.get_rs_text("")],
            [sg.Radio("Long-Term - Weeks to months", "TRADING_STYLE", key="-LONG-", 
                      background_color=RSTheme.OSRS_BROWN, text_color=RSTheme.OSRS_TEXT, font=('Runescape UF', 11))],
            [RSTheme.get_rs_text("A sage investor with the patience of the elves.")],
            [RSTheme.get_rs_text("")],
            [sg.Column([
                [RSTheme.get_rs_button("« Back"), sg.Push(), RSTheme.get_rs_button("Next »")]
            ], background_color=RSTheme.OSRS_BROWN)]
        ], background_color=RSTheme.OSRS_BROWN, justification='left', pad=(20, 20))]
    ]
    return layout

def create_risk_layout():
    """Create the risk tolerance layout"""
    layout = [
        [sg.Column([
            [RSTheme.get_rs_text("Risk Tolerance Settings", font_size=16, bold=True)],
            [sg.HorizontalSeparator(color=RSTheme.OSRS_GOLD, pad=(5, 15))],
            [RSTheme.get_rs_text("How much of your treasure are you willing to risk?")],
            [RSTheme.get_rs_text("")],
            [RSTheme.get_rs_text("Maximum BTC per position:")],
            [RSTheme.get_rs_input("0.1", key="-MAX_BTC-", size=(10, 1)), 
             RSTheme.get_rs_text("BTC", size=(4, 1))],
            [RSTheme.get_rs_text("Maximum USD per position:")],
            [RSTheme.get_rs_input("1000", key="-MAX_USD-", size=(10, 1)), 
             RSTheme.get_rs_text("USD", size=(4, 1))],
            [RSTheme.get_rs_text("")],
            [RSTheme.get_rs_text("Monthly Profit Target:")],
            [RSTheme.get_rs_text("Minimum:"), RSTheme.get_rs_input("5", key="-MIN_PROFIT-", size=(5, 1)), 
             RSTheme.get_rs_text("%", size=(2, 1))],
            [RSTheme.get_rs_text("Maximum:"), RSTheme.get_rs_input("20", key="-MAX_PROFIT-", size=(5, 1)), 
             RSTheme.get_rs_text("%", size=(2, 1))],
            [RSTheme.get_rs_text("")],
            [RSTheme.get_rs_text("Risk Level:")],
            [sg.Slider(range=(1, 10), default_value=5, orientation='h', key="-RISK_LEVEL-",
                      background_color=RSTheme.OSRS_BROWN, text_color=RSTheme.OSRS_TEXT,
                      trough_color=RSTheme.OSRS_DARK_BROWN)],
            [RSTheme.get_rs_text("Low Risk", size=(8, 1)), sg.Push(), RSTheme.get_rs_text("High Risk", size=(8, 1))],
            [RSTheme.get_rs_text("")],
            [sg.Column([
                [RSTheme.get_rs_button("« Back"), sg.Push(), RSTheme.get_rs_button("Next »")]
            ], background_color=RSTheme.OSRS_BROWN)]
        ], background_color=RSTheme.OSRS_BROWN, justification='left', pad=(20, 20))]
    ]
    return layout

def create_gpu_layout():
    """Create the GPU settings layout"""
    layout = [
        [sg.Column([
            [RSTheme.get_rs_text("Magical Powers (GPU Settings)", font_size=16, bold=True)],
            [sg.HorizontalSeparator(color=RSTheme.OSRS_GOLD, pad=(5, 15))],
            [RSTheme.get_rs_text("Configure your magical powers (GPU acceleration):")],
            [RSTheme.get_rs_text("")],
            [sg.Checkbox("Enable GPU Acceleration", key="-USE_GPU-", default=True, 
                        background_color=RSTheme.OSRS_BROWN, text_color=RSTheme.OSRS_TEXT, 
                        font=('Runescape UF', 11))],
            [RSTheme.get_rs_text("Use the mystical forces of your graphics rune to")],
            [RSTheme.get_rs_text("accelerate training. Recommended if available.")],
            [RSTheme.get_rs_text("")],
            [RSTheme.get_rs_text("GPU Target Utilization:")],
            [RSTheme.get_rs_text("Low:")], 
            [RSTheme.get_rs_input("30", key="-GPU_LOW-", size=(5, 1)), RSTheme.get_rs_text("%", size=(2, 1))],
            [RSTheme.get_rs_text("High:")], 
            [RSTheme.get_rs_input("70", key="-GPU_HIGH-", size=(5, 1)), RSTheme.get_rs_text("%", size=(2, 1))],
            [RSTheme.get_rs_text("")],
            [sg.Checkbox("Enable Mixed Precision", key="-MIXED_PRECISION-", default=True, 
                        background_color=RSTheme.OSRS_BROWN, text_color=RSTheme.OSRS_TEXT, 
                        font=('Runescape UF', 11))],
            [RSTheme.get_rs_text("Use a special training technique to save memory.")],
            [RSTheme.get_rs_text("")],
            [sg.Column([
                [RSTheme.get_rs_button("« Back"), sg.Push(), RSTheme.get_rs_button("Next »")]
            ], background_color=RSTheme.OSRS_BROWN)]
        ], background_color=RSTheme.OSRS_BROWN, justification='left', pad=(20, 20))]
    ]
    return layout

def create_final_layout():
    """Create the final setup page layout"""
    layout = [
        [sg.Column([
            [RSTheme.get_rs_text("Quest Complete!", font_size=16, bold=True)],
            [sg.HorizontalSeparator(color=RSTheme.OSRS_GOLD, pad=(5, 15))],
            [sg.Image(data=base64.b64decode(BLUE_PHAT_ICON), background_color=RSTheme.OSRS_BROWN)],
            [RSTheme.get_rs_text("")],
            [RSTheme.get_rs_text("Congratulations, brave adventurer!")],
            [RSTheme.get_rs_text("Your BTC Trading AI has been configured.")],
            [RSTheme.get_rs_text("")],
            [RSTheme.get_rs_text("You've earned:")],
            [RSTheme.get_rs_text("• 1x Blue Partyhat (displayed on application)")],
            [RSTheme.get_rs_text("• 1000 Trading Experience")],
            [RSTheme.get_rs_text("• Bitcoin Trading Knowledge")],
            [RSTheme.get_rs_text("")],
            [RSTheme.get_rs_text("Are you ready to begin your trading journey?")],
            [RSTheme.get_rs_text("")],
            [sg.Column([
                [RSTheme.get_rs_button("« Back"), sg.Push(), RSTheme.get_rs_button("Begin Trading Adventure!")]
            ], background_color=RSTheme.OSRS_BROWN)]
        ], background_color=RSTheme.OSRS_BROWN, justification='center', pad=(20, 20))]
    ]
    return layout

def run_setup_wizard() -> Optional[Dict[str, Any]]:
    """
    Run the setup wizard and return the configuration
    
    Returns:
        Dict[str, Any] or None: Configuration dictionary if completed, None if canceled
    """
    # Apply RuneScape theme
    RSTheme.apply_theme()
    
    # Create the different pages of the wizard
    layouts = {
        "welcome": create_welcome_layout(),
        "trading_style": create_trading_style_layout(),
        "risk": create_risk_layout(),
        "gpu": create_gpu_layout(),
        "final": create_final_layout()
    }
    
    # Start with the welcome page
    current_page = "welcome"
    
    # Create the window with optional icon
    window_args = {
        "title": "BTC-AI Setup Wizard",
        "layout": layouts[current_page],
        "finalize": True,
        "element_justification": 'center',
        "keep_on_top": True
    }
    
    if HAS_ICON:
        window_args["icon"] = base64.b64decode(BLUE_PHAT_ICON)
    
    window = sg.Window(**window_args)
    
    # Wizard configuration data
    config = {}
    
    # Store page history for back button
    page_history = []
    
    try:
        # Main event loop
        while True:
            event, values = window.read()
            
            if event == sg.WIN_CLOSED:
                logger.info("Setup wizard canceled by user")
                config = None
                break
            
            # Handle navigation
            if event == "Begin Adventure":
                page_history.append(current_page)
                current_page = "trading_style"
                window.close()
                window = sg.Window(
                    "BTC-AI Setup Wizard - Trading Style", 
                    layouts[current_page],
                    finalize=True,
                    icon=base64.b64decode(BLUE_PHAT_ICON),
                    element_justification='center',
                    font=('Runescape UF', 11),
                    keep_on_top=True
                )
            
            elif event == "« Back":
                if page_history:
                    current_page = page_history.pop()
                    window.close()
                    window = sg.Window(
                        f"BTC-AI Setup Wizard - {current_page.replace('_', ' ').title()}", 
                        layouts[current_page],
                        finalize=True,
                        icon=base64.b64decode(BLUE_PHAT_ICON),
                        element_justification='center',
                        font=('Runescape UF', 11),
                        keep_on_top=True
                    )
            
            elif event == "Next »":
                # Save values from current page
                if current_page == "trading_style":
                    if values["-SCALPING-"]:
                        config["BUCKET"] = "Scalping"
                    elif values["-SHORT-"]:
                        config["BUCKET"] = "Short"
                    elif values["-MEDIUM-"]:
                        config["BUCKET"] = "Medium"
                    elif values["-LONG-"]:
                        config["BUCKET"] = "Long"
                    
                    # Navigate to risk page
                    page_history.append(current_page)
                    current_page = "risk"
                
                elif current_page == "risk":
                    try:
                        # Save risk settings
                        config["MAX_BTC_PER_POSITION"] = float(values["-MAX_BTC-"])
                        config["MAX_USD_PER_POSITION"] = float(values["-MAX_USD-"])
                        config["monthly_target_min"] = float(values["-MIN_PROFIT-"])
                        config["monthly_target_max"] = float(values["-MAX_PROFIT-"])
                        config["RISK_LEVEL"] = int(values["-RISK_LEVEL-"])
                        
                        # Navigate to GPU page
                        page_history.append(current_page)
                        current_page = "gpu"
                    except ValueError:
                        sg.popup_error("Please enter valid numbers for all fields", title="Input Error")
                        continue
                
                elif current_page == "gpu":
                    # Save GPU settings
                    config["USE_GPU"] = values["-USE_GPU-"]
                    try:
                        config["GPU_TARGET_UTILIZATION_LOW"] = float(values["-GPU_LOW-"])
                        config["GPU_TARGET_UTILIZATION_HIGH"] = float(values["-GPU_HIGH-"])
                    except ValueError:
                        sg.popup_error("Please enter valid numbers for GPU utilization", title="Input Error")
                        continue
                    
                    config["USE_MIXED_PRECISION"] = values["-MIXED_PRECISION-"]
                    
                    # Navigate to final page
                    page_history.append(current_page)
                    current_page = "final"
                
                # Update the window to show the new page
                window.close()
                window = sg.Window(
                    f"BTC-AI Setup Wizard - {current_page.replace('_', ' ').title()}", 
                    layouts[current_page],
                    finalize=True,
                    icon=base64.b64decode(BLUE_PHAT_ICON),
                    element_justification='center',
                    font=('Runescape UF', 11),
                    keep_on_top=True
                )
            
            elif event == "Begin Trading Adventure!":
                # Complete the wizard
                logger.info("Setup wizard completed successfully")
                
                # Mark wizard as complete
                with open(WIZARD_COMPLETE_FILE, 'w') as f:
                    json.dump({"completed": True, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f)
                
                break
    
    except Exception as e:
        logger.error(f"Error in setup wizard: {str(e)}")
        if window:
            window.close()
        return None
    
    finally:
        if window:
            window.close()
    
    return config

def is_first_run() -> bool:
    """
    Check if this is the first run by looking for the wizard_complete.json file
    
    Returns:
        bool: True if this is the first run, False otherwise
    """
    return not os.path.exists(WIZARD_COMPLETE_FILE)

def save_config(config: Dict[str, Any]) -> bool:
    """
    Save the configuration to the config file
    
    Args:
        config: Configuration dictionary
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure config directory exists
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        
        # If a config file already exists, load it and update it
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                existing_config = json.load(f)
            
            # Update with new values
            existing_config.update(config)
            config = existing_config
        
        # Save the config
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}")
        return False

def load_config() -> Optional[Dict[str, Any]]:
    """
    Load the configuration from the config file
    
    Returns:
        Dict[str, Any] or None: Configuration dictionary if successful, None otherwise
    """
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config
        else:
            logger.warning(f"Config file not found: {CONFIG_FILE}")
            return None
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return None

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        ValueError: If numeric values are invalid
        KeyError: If required fields are missing
    """
    # Check required fields
    required_fields = ["MAX_BTC_PER_POSITION", "MAX_USD_PER_POSITION"]
    for field in required_fields:
        if field not in config:
            raise KeyError(f"Missing required field: {field}")
    
    # Validate numeric values
    float_fields = ["MAX_BTC_PER_POSITION", "MAX_USD_PER_POSITION"]
    for field in float_fields:
        if field in config:
            value = config[field]
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(f"Invalid value for {field}: {value}")
            
            if value <= 0:
                raise ValueError(f"{field} must be positive")
    
    # Check risk level is in range
    if "RISK_LEVEL" in config:
        risk_level = config["RISK_LEVEL"]
        if isinstance(risk_level, str):
            try:
                risk_level = int(risk_level)
            except ValueError:
                raise ValueError(f"Invalid risk level: {risk_level}")
        
        if not (1 <= risk_level <= 10):
            raise ValueError(f"Risk level must be between 1 and 10")
    
    return True

def mark_wizard_complete() -> bool:
    """
    Mark the wizard as complete by creating the wizard_complete.json file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(WIZARD_COMPLETE_FILE), exist_ok=True)
        
        # Write the completion file
        with open(WIZARD_COMPLETE_FILE, 'w') as f:
            json.dump({
                "completed": True,
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=4)
        
        return True
    except Exception as e:
        logger.error(f"Error marking wizard as complete: {str(e)}")
        return False

if __name__ == "__main__":
    # For testing the wizard directly
    try:
        # Force the wizard to run regardless of whether it's the first run
        if len(sys.argv) > 1 and sys.argv[1] == "--force":
            # Delete the wizard complete file to force it to run
            if os.path.exists(WIZARD_COMPLETE_FILE):
                os.remove(WIZARD_COMPLETE_FILE)
        
        if is_first_run():
            print("Running setup wizard...")
            config = run_setup_wizard()
            if config:
                print("Wizard completed, saving config...")
                if save_config(config):
                    print("Config saved successfully!")
                else:
                    print("Failed to save config.")
            else:
                print("Wizard was canceled or encountered an error.")
        else:
            print("Not first run, skipping wizard.")
            print("Use --force to run the wizard anyway.")
    except Exception as e:
        print(f"Error: {str(e)}") 