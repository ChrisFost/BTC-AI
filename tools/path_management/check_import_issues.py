import os
import re
import importlib.util
import sys

def list_python_files(directory):
    """Find all Python files in a directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_imports(file_path):
    """Check for potential import issues in a file."""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Look for potential issues
    
    # 1. Check for references to utils_v2
    if re.search(r'utils_v2', content):
        issues.append("References to utils_v2 found (should be utils)")
    
    # 2. Check for references to Agent V2
    if re.search(r'Agent\s+V2', content):
        issues.append("References to Agent V2 found (should be agent)")
    
    # 3. Check for references to models v2
    if re.search(r'models\s+v2', content):
        issues.append("References to models v2 found (should be models)")
    
    # 4. Check for references to backtesting_v2
    if re.search(r'backtesting_v2', content):
        issues.append("References to backtesting_v2 found (should be backtesting)")
    
    # 5. Check for bare imports from training
    if re.search(r'from\s+training\s+import', content):
        issues.append("Bare import from training found (should be src.training.training)")
    
    # 6. Check for improper relative imports
    if re.search(r'from\s+\.\.\s+import', content) and 'src/' in file_path:
        issues.append("Relative import with parent directory reference found (may need fixing)")
    
    return issues

def main():
    """Main function to check for import issues in Python files."""
    src_dir = "src"
    python_files = list_python_files(src_dir)
    
    print(f"Checking {len(python_files)} Python files for import issues...")
    
    files_with_issues = 0
    for file_path in python_files:
        issues = check_imports(file_path)
        if issues:
            files_with_issues += 1
            rel_path = os.path.relpath(file_path, start=os.getcwd())
            print(f"\n{rel_path}:")
            for issue in issues:
                print(f"  - {issue}")
    
    if files_with_issues > 0:
        print(f"\nFound {files_with_issues} files with potential import issues.")
    else:
        print("\nNo files with import issues found!")
    
    # Check for circular imports
    print("\nChecking for circular imports (this may take a moment)...")
    sys.path.insert(0, os.getcwd())
    
    # Try to import key modules
    key_modules = [
        'src.agent.agent',
        'src.models.models',
        'src.environment.env_base',
        'src.training.training',
        'src.training.progressive_training',
        'src.utils.utils'
    ]
    
    for module_name in key_modules:
        try:
            print(f"Attempting to import {module_name}...")
            module = __import__(module_name, fromlist=[''])
            print(f"  Success!")
        except ImportError as e:
            print(f"  Failed: {e}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()