import os
import base64

def load_icon(filename):
    """Load an icon file and return its base64 encoded data"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(current_dir, filename)
    
    if not os.path.exists(icon_path):
        raise FileNotFoundError(f"Icon file not found: {icon_path}")
    
    with open(icon_path, 'rb') as f:
        return base64.b64encode(f.read())

# Load the blue partyhat icon
try:
    BLUE_PHAT_ICON = load_icon('blue_partyhat.png')
except Exception as e:
    print(f"Warning: Could not load blue partyhat icon: {e}")
    BLUE_PHAT_ICON = None 