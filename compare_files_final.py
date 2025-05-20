import os
import re
import difflib
import glob

# Define paths
NEWEST_STUFF_DIR = "Scripts/newest stuff"
SRC_DIR = "src"

# Function to find all Python files in a directory
def find_all_python_files(directory):
    """Find all Python files in a directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                python_files.append(rel_path.replace('\\', '/'))
    return python_files

# Find potential target mapping for file in src directory
def find_target_file(original_file):
    """Try to find a matching target file in src directory."""
    # Extract base filename without version numbers and spaces
    base_name = os.path.basename(original_file)
    base_name = re.sub(r'[\s_]?[vV]\d+\.py$', '.py', base_name)
    
    # Find all Python files in src directory
    src_files = find_all_python_files(SRC_DIR)
    
    # First try: exact match on basename
    for src_file in src_files:
        if os.path.basename(src_file) == base_name:
            return src_file
    
    # Second try: similar name match
    for src_file in src_files:
        src_base = os.path.basename(src_file)
        # Remove extensions for comparison
        src_base_no_ext = os.path.splitext(src_base)[0]
        orig_base_no_ext = os.path.splitext(base_name)[0]
        
        # Check if name is similar (ignoring case and version numbers)
        if src_base_no_ext.lower() == orig_base_no_ext.lower():
            return src_file
    
    # Third try: check subfolders based on logical naming
    if 'agent' in base_name.lower():
        for src_file in src_files:
            if 'agent' in src_file.lower():
                return src_file
    if 'model' in base_name.lower():
        for src_file in src_files:
            if 'model' in src_file.lower():
                return src_file
    if 'env' in base_name.lower():
        for src_file in src_files:
            if 'env' in src_file.lower() and not 'environ' in src_file.lower():
                return src_file
    if 'config' in base_name.lower():
        for src_file in src_files:
            if 'config' in src_file.lower():
                return src_file
    if 'train' in base_name.lower():
        for src_file in src_files:
            if 'train' in src_file.lower():
                return src_file
    if 'backtest' in base_name.lower():
        for src_file in src_files:
            if 'backtest' in src_file.lower():
                return src_file
    
    # Not found
    return None

def extract_functions(content):
    """Extract function definitions from content."""
    # Match "def function_name(..." patterns
    function_pattern = re.compile(r'^\s*def\s+(\w+)\s*\(', re.MULTILINE)
    return set(function_pattern.findall(content))

def extract_class_definitions(content):
    """Extract class definitions from content."""
    class_pattern = re.compile(r'^\s*class\s+(\w+)[\(:]', re.MULTILINE)
    return set(class_pattern.findall(content))

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
        
        # Extract class definitions
        original_classes = extract_class_definitions(original_content)
        target_classes = extract_class_definitions(target_content)
        missing_classes = original_classes - target_classes
        
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
        
        # Determine if there are significant differences
        has_important_differences = (
            missing_functions or missing_classes or 
            missing_properties or missing_configs
        )
        
        if not has_important_differences:
            report.append("No significant differences found. All key elements appear to be present in the target file.\n")
            report.append(f"Diff stats: {original_only_lines} lines present in original but missing in target, {target_only_lines} lines present in target but not in original.")
            return "\n".join(report), False
        
        # Flag as having important differences
        has_major_issues = False
        
        if missing_classes:
            report.append(f"## Missing Classes ({len(missing_classes)})")
            for cls in sorted(missing_classes):
                report.append(f"- `{cls}`")
            report.append("")
            has_major_issues = True
        
        if missing_functions:
            report.append(f"## Missing Functions/Methods ({len(missing_functions)})")
            for func in sorted(missing_functions):
                report.append(f"- `{func}`")
            report.append("")
            if len(missing_functions) > 3:
                has_major_issues = True
        
        if missing_properties:
            report.append(f"## Missing Class Properties ({len(missing_properties)})")
            for prop in sorted(missing_properties):
                report.append(f"- `self.{prop}`")
            report.append("")
            if len(missing_properties) > 5:
                has_major_issues = True
        
        if missing_configs:
            report.append(f"## Missing Configuration Parameters ({len(missing_configs)})")
            for config in sorted(missing_configs):
                report.append(f"- `{config}`")
            report.append("")
            if len(missing_configs) > 3:
                has_major_issues = True
            
        if missing_imports:
            report.append(f"## Missing Imports ({len(missing_imports)})")
            for imp in sorted(missing_imports):
                report.append(f"- `{imp}`")
            report.append("")
        
        report.append(f"Diff stats: {original_only_lines} lines present in original but missing in target, {target_only_lines} lines present in target but not in original.")
        
        return "\n".join(report), has_major_issues
    except Exception as e:
        return f"Error analyzing files: {str(e)}", False

def main():
    """Main function to compare files and generate reports."""
    print("Starting comprehensive file comparison...")
    
    # Create output directories
    all_reports_dir = "comparison_reports_all"
    major_issues_dir = "comparison_reports_critical"
    os.makedirs(all_reports_dir, exist_ok=True)
    os.makedirs(major_issues_dir, exist_ok=True)
    
    # Find all Python files in newest stuff directory
    original_files = find_all_python_files(NEWEST_STUFF_DIR)
    print(f"Found {len(original_files)} Python files in {NEWEST_STUFF_DIR}")
    
    # Track files with major issues
    files_with_major_issues = []
    
    # Process each file
    for original_rel_path in original_files:
        # Skip certain files that are testing, debugging, or obsolete
        if any(x in original_rel_path.lower() for x in ['obsolete', 'test_', 'debug_', 'legacy_', 'old_']):
            print(f"Skipping likely test/debug file: {original_rel_path}")
            continue
            
        # Try to find a corresponding file in src
        target_rel_path = find_target_file(original_rel_path)
        
        if target_rel_path:
            print(f"Comparing {original_rel_path} to {target_rel_path}...")
            
            original_path = os.path.join(NEWEST_STUFF_DIR, original_rel_path)
            target_path = os.path.join(SRC_DIR, target_rel_path)
            
            report, has_major_issues = analyze_files(original_path, target_path)
            
            # Save report
            original_filename = os.path.basename(original_rel_path).replace(" ", "_")
            report_path = os.path.join(all_reports_dir, f"report_{original_filename}.md")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"Report saved to {report_path}")
            
            # If file has major issues, create a report in the critical directory and track it
            if has_major_issues:
                critical_report_path = os.path.join(major_issues_dir, f"CRITICAL_{original_filename}.md")
                with open(critical_report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                files_with_major_issues.append((original_rel_path, target_rel_path))
        else:
            print(f"No matching target file found for {original_rel_path}")
    
    # Generate summary report
    if files_with_major_issues:
        summary_path = os.path.join(major_issues_dir, "00_SUMMARY.md")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# Critical Missing Functionality Summary\n\n")
            f.write("The following files have significant missing functionality that should be addressed:\n\n")
            
            for orig, target in files_with_major_issues:
                f.write(f"- {orig} -> {target}\n")
        
        print(f"\n⚠️ FOUND {len(files_with_major_issues)} FILES WITH MAJOR ISSUES!")
        print(f"Summary saved to {summary_path}")
    else:
        print("\n✅ No files with major issues were found.")
    
    print("\nAll comparisons complete!")

if __name__ == "__main__":
    main() 