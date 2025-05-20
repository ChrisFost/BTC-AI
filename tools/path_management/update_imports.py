#!/usr/bin/env python3
"""
Import Path Updater for BTC-AI Project

This script updates import statements in Python files to reflect the new directory structure.
"""

import os
import re
import sys
from pathlib import Path
import fileinput

# Define the old-to-new module mappings
MODULE_MAPPINGS = {
    # Core files
    "Agent V2": "src.agent.agent",
    "models v2": "src.models.models",
    "utils v2": "src.utils.utils",
    "menu_script v2": "src.ui.main",
    "bucket_goals": "src.utils.bucket_goals",
    "visualize": "src.utils.visualization",
    "reasoning_analyzer": "src.utils.reasoning",
    "dataframe": "src.utils.dataframe",
    "tensor_utils v3": "src.utils.tensor_utils",
    "progressive_visualizer": "src.utils.progressive_visualizer",
    
    # Environment modules
    "env_base": "src.environment.env_base",
    "env_risk": "src.environment.env_risk",
    "env_interfaces": "src.environment.env_interfaces",
    "env_market": "src.environment.env_market",
    "env_utils": "src.environment.env_utils",
    "env_rewards": "src.environment.env_rewards",
    "env_observation": "src.environment.env_observation",
    
    # Training modules
    "training": "src.training.training",
    "progressive_training": "src.training.progressive",
    "performance_optimizer": "src.training.optimizer",
    "backtesting_v2": "src.training.backtesting_v2",
    "mock_training": "src.training.mock_training",
    "realtime_inference": "src.training.realtime_inference",
    "evaluate": "src.training.evaluate"
}

# Simple function imports (ones without 'from')
def update_simple_imports(line):
    """Update simple import statements like 'import module'"""
    pattern = r"import\s+([A-Za-z0-9_]+(?:\s*,\s*[A-Za-z0-9_]+)*)"
    
    def replace_module(match):
        modules = [m.strip() for m in match.group(1).split(',')]
        updated_modules = []
        
        for module in modules:
            if module in MODULE_MAPPINGS:
                updated_modules.append(MODULE_MAPPINGS[module])
            else:
                updated_modules.append(module)
        
        return f"import {', '.join(updated_modules)}"
    
    return re.sub(pattern, replace_module, line)

# From imports
def update_from_imports(line):
    """Update 'from module import stuff' statements"""
    pattern = r"from\s+([A-Za-z0-9_]+)\s+import\s+(.+)"
    
    def replace_from_import(match):
        module = match.group(1)
        imports = match.group(2)
        
        if module in MODULE_MAPPINGS:
            return f"from {MODULE_MAPPINGS[module]} import {imports}"
        return match.group(0)
    
    return re.sub(pattern, replace_from_import, line)

# Helper function to check if line is likely a docstring or comment
def is_docstring_or_comment(line):
    """Check if a line is likely part of a docstring or comment"""
    line = line.strip()
    return line.startswith('#') or line.startswith('"""') or line.startswith("'''")

# Main function to update imports in a file
def update_file_imports(file_path):
    """Update import statements in a single file"""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the file content into lines
    lines = content.split('\n')
    updated_lines = []
    
    for line in lines:
        # Skip docstrings and comments
        if is_docstring_or_comment(line):
            updated_lines.append(line)
            continue
        
        # Update simple imports
        updated_line = update_simple_imports(line)
        
        # Update from imports
        updated_line = update_from_imports(updated_line)
        
        updated_lines.append(updated_line)
    
    # Join the lines back together
    updated_content = '\n'.join(updated_lines)
    
    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)
    
    print(f"Updated imports in {file_path}")

def process_directory(directory):
    """Process all Python files in a directory and its subdirectories"""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                update_file_imports(file_path)

if __name__ == "__main__":
    # Get the directory to process from command line argument or use default
    directory = sys.argv[1] if len(sys.argv) > 1 else "src"
    
    print(f"Updating imports in {directory}...")
    process_directory(directory)
    print("Import update completed!") 