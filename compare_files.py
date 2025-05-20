#!/usr/bin/env python
"""
Script to compare files between 'Scripts/newest stuff' and 'src' directories,
identifying missing functions, class properties, and initializations.
"""

import os
import re
import ast
import difflib
from pathlib import Path

# Define paths
NEWEST_STUFF_DIR = "Scripts/newest stuff"
SRC_DIR = "src"

# Define file mappings (original -> target)
file_mappings = {
    "Agent V2.py": "agent/agent.py",
    "models/models v2.py": "models/models.py", 
    "utils v2.py": "utils/utils.py",
    "environment/env_base.py": "environment/env_base.py",
    "environment/env_risk.py": "environment/env_risk.py",
    "utils/tensor_utils_v3.py": "utils/tensor_utils.py",
    "utils/reasoning_analyzer.py": "utils/reasoning_analyzer.py"
}

# Update the file mappings to correctly identify the environment files
corrected_file_mappings = {}
for orig, target in file_mappings.items():
    # Check if the original file exists
    orig_path = os.path.join(NEWEST_STUFF_DIR, orig)
    if not os.path.exists(orig_path):
        print(f"Original file not found: {orig_path}")
        # Try to find alternate locations for environment files
        if "env_" in orig:
            alt_path = os.path.join(NEWEST_STUFF_DIR, orig.replace("environment/", ""))
            if os.path.exists(alt_path):
                print(f"Found alternate location: {alt_path}")
                corrected_file_mappings[orig.replace("environment/", "")] = target
            else:
                print(f"Could not find alternate location for {orig}")
        else:
            print(f"Skipping {orig}")
    else:
        corrected_file_mappings[orig] = target

# Replace the original mappings with corrected ones
file_mappings = corrected_file_mappings

# Patterns to look for
FUNCTION_DEF_PATTERN = re.compile(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
CLASS_DEF_PATTERN = re.compile(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]')
PROPERTY_INIT_PATTERN = re.compile(r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=')
CONFIG_GET_PATTERN = re.compile(r'self\.config\.get\("([A-Z_]+)"')

def extract_functions(content):
    """Extract function definitions from content."""
    # Match "def function_name(..." patterns
    function_pattern = re.compile(r'^\s*def\s+(\w+)\s*\(', re.MULTILINE)
    return set(function_pattern.findall(content))

def extract_class_properties(content):
    """Extract class property initializations from content."""
    # Match "self.property_name = ..." patterns in __init__ methods
    property_pattern = re.compile(r'^\s*self\.(\w+)\s*=', re.MULTILINE)
    return set(property_pattern.findall(content))

def extract_config_params(content):
    """Extract configuration parameters from content."""
    # Match patterns like "CONFIG_PARAM = value" or "self.CONFIG_PARAM = value"
    config_pattern = re.compile(r'^\s*((?:self\.)?\w+_[A-Z_]+)\s*=', re.MULTILINE)
    return set(config_pattern.findall(content))

def analyze_files(original_path, target_path):
    """Compare original file with target file and generate a report."""
    try:
        # Check if files exist
        if not os.path.exists(original_path):
            return f"Error: Original file {original_path} not found."
        if not os.path.exists(target_path):
            return f"Error: Target file {target_path} not found."
        
        # Read file contents
        with open(original_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()
        with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
            target_content = f.read()
        
        # Generate diff stats
        diff = list(difflib.unified_diff(
            original_content.splitlines(),
            target_content.splitlines(),
            n=0
        ))
        original_only_lines = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
        target_only_lines = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        
        # Extract function definitions
        original_functions = extract_functions(original_content)
        target_functions = extract_functions(target_content)
        missing_functions = original_functions - target_functions
        
        # Extract class properties
        original_properties = extract_class_properties(original_content)
        target_properties = extract_class_properties(target_content)
        missing_properties = original_properties - target_properties
        
        # Extract configuration parameters
        original_configs = extract_config_params(original_content)
        target_configs = extract_config_params(target_content)
        missing_configs = original_configs - target_configs
        
        # Generate report
        report = []
        report.append(f"# Comparison Report: {os.path.basename(original_path)} vs {os.path.basename(target_path)}\n")
        
        if not missing_functions and not missing_properties and not missing_configs:
            report.append("No significant differences found. All key elements appear to be present in the target file.\n")
            report.append(f"Diff stats: {original_only_lines} lines present in original but missing in target, {target_only_lines} lines present in target but not in original.")
            return "\n".join(report)
        
        if missing_functions:
            report.append(f"## Missing Functions/Methods ({len(missing_functions)})")
            for func in sorted(missing_functions):
                report.append(f"- `{func}`")
            report.append("")
        
        if missing_properties:
            report.append(f"## Missing Class Properties ({len(missing_properties)})")
            for prop in sorted(missing_properties):
                report.append(f"- `self.{prop}`")
            report.append("")
        
        if missing_configs:
            report.append(f"## Missing Configuration Parameters ({len(missing_configs)})")
            for config in sorted(missing_configs):
                report.append(f"- `{config}`")
            report.append("")
        
        report.append(f"Diff stats: {original_only_lines} lines present in original but missing in target, {target_only_lines} lines present in target but not in original.")
        
        return "\n".join(report)
    except Exception as e:
        return f"Error analyzing files: {str(e)}"

def main():
    """Main function to compare files and generate reports."""
    print("Starting file comparison...")
    
    # Create output directory
    output_dir = "comparison_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare each file pair
    for original_rel_path, target_rel_path in file_mappings.items():
        print(f"Comparing {original_rel_path} to {target_rel_path}...")
        
        original_path = os.path.join(NEWEST_STUFF_DIR, original_rel_path)
        target_path = os.path.join(SRC_DIR, target_rel_path)
        
        report = analyze_files(original_path, target_path)
        
        # Save report
        original_filename = os.path.basename(original_rel_path).replace(" ", "_")
        report_path = os.path.join(output_dir, f"report_{original_filename}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to {report_path}")
    
    print("All comparisons complete!")

if __name__ == "__main__":
    main() 