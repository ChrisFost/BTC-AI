import os
import re
import difflib

# Define paths
NEWEST_STUFF_DIR = "Scripts/newest stuff"
SRC_DIR = "src"

# Define file mappings (original -> target)
# Expanded based on imports and references in test scripts
file_mappings = {
    # Core files
    "Agent V2.py": "agent/agent.py",
    "utils v2.py": "utils/utils.py",
    "environment/env_base.py": "environment/env_base.py",
    "environment/env_risk.py": "environment/env_risk.py",
    "main.py": "ui/main.py",
    "training.py": "training/training.py",
    "backtesting_v2.py": "training/backtesting.py",
    "mock_training.py": "training/mock_training.py",
    "config v2.py": "utils/config.py"
}

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

def extract_imports(content):
    """Extract import statements from content."""
    # Match both "import X" and "from X import Y" patterns
    import_pattern = re.compile(r'^\s*(?:from\s+(\S+)\s+import|import\s+(\S+))', re.MULTILINE)
    imports = []
    for match in import_pattern.finditer(content):
        if match.group(1):  # from X import Y
            imports.append(match.group(1))
        else:  # import X
            imports.append(match.group(2))
    return set(imports)

def analyze_files(original_path, target_path):
    """Compare original file with target file and generate a report."""
    try:
        # Check if files exist
        if not os.path.exists(original_path):
            return f"Error: Original file {original_path} not found."
        if not os.path.exists(target_path):
            return f"Error: Target file {target_path} not found."
        
        print(f"Analyzing: {original_path} -> {target_path}")
        
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
        
        # Extract imports
        original_imports = extract_imports(original_content)
        target_imports = extract_imports(target_content)
        missing_imports = original_imports - target_imports
        
        # Generate report
        report = []
        report.append(f"# Comparison Report: {os.path.basename(original_path)} vs {os.path.basename(target_path)}\n")
        
        if not (missing_functions or missing_properties or missing_configs or missing_imports):
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
            
        if missing_imports:
            report.append(f"## Missing Imports ({len(missing_imports)})")
            for imp in sorted(missing_imports):
                report.append(f"- `{imp}`")
            report.append("")
        
        report.append(f"Diff stats: {original_only_lines} lines present in original but missing in target, {target_only_lines} lines present in target but not in original.")
        
        return "\n".join(report)
    except Exception as e:
        return f"Error analyzing files: {str(e)}"

def main():
    """Main function to compare files and generate reports."""
    print("Starting file comparison...")
    
    # Create output directory
    output_dir = "comparison_reports_final"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which files exist before comparing
    valid_mappings = {}
    for original_rel_path, target_rel_path in file_mappings.items():
        original_path = os.path.join(NEWEST_STUFF_DIR, original_rel_path)
        target_path = os.path.join(SRC_DIR, target_rel_path)
        
        if os.path.exists(original_path) and os.path.exists(target_path):
            valid_mappings[original_rel_path] = target_rel_path
        else:
            if not os.path.exists(original_path):
                print(f"Warning: Original file not found: {original_path}")
            if not os.path.exists(target_path):
                print(f"Warning: Target file not found: {target_path}")
    
    # Compare each file pair
    for original_rel_path, target_rel_path in valid_mappings.items():
        print(f"Comparing {original_rel_path} to {target_rel_path}...")
        
        original_path = os.path.join(NEWEST_STUFF_DIR, original_rel_path)
        target_path = os.path.join(SRC_DIR, target_rel_path)
        
        report = analyze_files(original_path, target_path)
        
        # Save report with the proper filename
        # For paths like environment/env_base.py, use env_base.py as the filename base
        original_filename = os.path.basename(original_rel_path).replace(" ", "_")
        report_path = os.path.join(output_dir, f"report_{original_filename}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to {report_path}")
    
    print("All comparisons complete!")

if __name__ == "__main__":
    main() 