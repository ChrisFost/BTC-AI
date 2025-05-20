#!/usr/bin/env python3
"""
Path Updater for BTC-AI Project

This script updates hardcoded file paths in Python files to reflect the new directory structure.
"""

import os
import re
import sys
from pathlib import Path

# Define path updates (old paths to new paths)
PATH_UPDATES = {
    r"Scripts[\\/]newest stuff[\\/]menu_script v2\.py": "src/ui/main.py",
    r"Scripts[\\/]newest stuff[\\/]Agent V2\.py": "src/agent/agent.py",
    r"Scripts[\\/]newest stuff[\\/]models v2\.py": "src/models/models.py",
    r"Scripts[\\/]newest stuff[\\/]environment[\\/]env_base\.py": "src/environment/env_base.py",
    r"Scripts[\\/]newest stuff[\\/]environment[\\/]env_risk\.py": "src/environment/env_risk.py",
    r"Scripts[\\/]newest stuff[\\/]visualize\.py": "src/utils/visualization.py",
    r"Scripts[\\/]newest stuff[\\/]reasoning_analyzer\.py": "src/utils/reasoning.py",
    r"Scripts[\\/]newest stuff[\\/]training\.py": "src/training/training.py",
    r"Scripts[\\/]newest stuff[\\/]performance_optimizer\.py": "src/training/optimizer.py",
    r"utils_v2\.py": "src/utils/utils.py",
    r"config v2\.py": "src/training/src/training/config.py",
    r"config\.py": "src/training/src/training/config.py",
    r"from src.utils.utils": "from src.utils.utils",
    r"from src.environment.env_base": "from src.environment.env_base",
    r"import src.agent.agent": "import src.agent.agent",
    r"importlib\.import_module\(\"Agent V2\"\)": "importlib.import_module(\"src.agent.agent\")",
    r"from src.agent.agent": "from src.agent.agent"
}

def update_file_paths(file_path):
    """Update file paths in a single file"""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        try:
            content = file.read()
        except UnicodeDecodeError:
            print(f"Warning: Could not read {file_path} due to encoding issues. Skipping.")
            return
    
    # Make a copy of the original content
    updated_content = content
    
    # Replace paths
    for old_path_pattern, new_path in PATH_UPDATES.items():
        # Use regex to handle both forward and backslash variations
        pattern = re.compile(old_path_pattern)
        updated_content = pattern.sub(new_path, updated_content)
    
    # Write the updated content back to the file if changes were made
    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
        print(f"Updated paths in {file_path}")
    else:
        print(f"No path changes needed in {file_path}")

def process_directory(directory):
    """Process all Python files in a directory and its subdirectories"""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.py', '.json', '.md')):  # Include config and docs files
                file_path = os.path.join(root, file)
                try:
                    update_file_paths(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Get the directory to process from command line argument or use default
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print(f"Updating paths in {directory}...")
    process_directory(directory)
    print("Path update completed!") 