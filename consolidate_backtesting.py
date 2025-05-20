import os
import shutil
import re

def backup_file(file_path):
    """Create a backup of a file."""
    backup_path = file_path + ".bak"
    if os.path.exists(file_path):
        shutil.copy2(file_path, backup_path)
        print(f"Created backup at {backup_path}")
    return backup_path

def check_references():
    """Check for any remaining references to backtesting_v2 in the codebase."""
    references = []
    
    for root, _, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                if "backtesting_v2" in content:
                    references.append(file_path)
    
    return references

def update_setup_py():
    """Update setup.py to fix entry points referencing backtesting_v2."""
    setup_path = "setup.py"
    if not os.path.exists(setup_path):
        print(f"Warning: {setup_path} not found")
        return False
    
    with open(setup_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix entry point references
    updated_content = re.sub(
        r'src\.training\.backtesting_v2', 
        r'src.training.backtesting', 
        content
    )
    
    if content != updated_content:
        backup_file(setup_path)
        with open(setup_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Updated {setup_path}")
        return True
    
    return False

def update_test_files():
    """Update test files referencing backtesting_v2."""
    updated_files = 0
    
    for root, _, files in os.walk("tests"):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Update import statements
                updated_content = re.sub(
                    r'(from|import)\s+([.\w]+\.)?backtesting_v2', 
                    r'\1 \2backtesting', 
                    content
                )
                
                # Update patch statements
                updated_content = re.sub(
                    r"patch\(['\"]([.\w]+\.)?backtesting_v2", 
                    r"patch('\1backtesting", 
                    updated_content
                )
                
                if content != updated_content:
                    backup_file(file_path)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    print(f"Updated {file_path}")
                    updated_files += 1
    
    return updated_files

def main():
    """Consolidate backtesting files by removing backtesting_v2.py."""
    backtesting_path = "src/training/backtesting.py"
    backtesting_v2_path = "src/training/backtesting_v2.py"
    
    if not os.path.exists(backtesting_path) or not os.path.exists(backtesting_v2_path):
        print("Error: One or both backtesting files not found")
        return
    
    # Check if files are identical
    with open(backtesting_path, 'r', encoding='utf-8') as f1, \
         open(backtesting_v2_path, 'r', encoding='utf-8') as f2:
        content1 = f1.read()
        content2 = f2.read()
    
    if content1 != content2:
        print("Warning: The files are not identical. Please review them manually.")
        print("Proceeding with consolidation anyway...")
    else:
        print("Confirmed: Files are identical.")
    
    # Create backup of backtesting_v2.py before removal
    backup_v2 = backup_file(backtesting_v2_path)
    
    # Update setup.py
    update_setup_py()
    
    # Update test files
    updated_test_files = update_test_files()
    print(f"Updated {updated_test_files} test files")
    
    # Check for any remaining references
    remaining_refs = check_references()
    if remaining_refs:
        print("\nWarning: Found remaining references to backtesting_v2 in these files:")
        for ref in remaining_refs:
            print(f"  - {ref}")
        print("Please update these references manually.")
    
    # Delete backtesting_v2.py
    os.remove(backtesting_v2_path)
    print(f"Removed {backtesting_v2_path}")
    
    print(f"\nConsolidation complete. A backup of backtesting_v2.py was created at {backup_v2}")
    print("You can delete the backup once you've verified everything works correctly.")

if __name__ == "__main__":
    main() 