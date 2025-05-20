import os
import re
import shutil

# Define paths
NEWEST_STUFF_DIR = "Scripts/newest stuff"
SRC_DIR = "src"

def ensure_directory_exists(file_path):
    """Ensure the directory for the specified file exists."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def adjust_imports(content, file_path):
    """Adjust import statements to match the new directory structure."""
    # Convert relative imports that might be referring to the root directory
    if "environment/" in file_path:
        # For environment files
        content = re.sub(r'from\s+env_', r'from .env_', content)
        content = re.sub(r'import\s+env_', r'from . import env_', content)
    
    # Fix any other imports that need adjustment
    content = re.sub(r'from\s+src\.utils\.utils_v2\s+import', r'from src.utils.utils import', content)
    content = re.sub(r'from\s+Agent\s+V2', r'from src.agent.agent', content)
    content = re.sub(r'from\s+models\s+v2', r'from src.models.models', content)
    content = re.sub(r'from\s+config\s+v2', r'from src.utils.config', content)
    
    return content

def copy_file(source_file, dest_file, is_environment=False):
    """Copy a file from source to destination with import adjustments."""
    source_path = os.path.join(NEWEST_STUFF_DIR, source_file)
    
    if not os.path.exists(source_path):
        print(f"Error: Source file not found: {source_path}")
        return False
    
    # Ensure the destination directory exists
    ensure_directory_exists(dest_file)
    
    # Read the source file
    with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Adjust imports
    adjusted_content = adjust_imports(content, dest_file)
    
    # Write to destination
    with open(dest_file, 'w', encoding='utf-8') as f:
        f.write(adjusted_content)
    
    print(f"Copied with adjustments: {source_path} -> {dest_file}")
    return True

def check_environment_files():
    """Check if all environment files are present and copy if missing."""
    # Define important environment files to check
    env_files = [
        "env_base.py",
        "env_interfaces.py",
        "env_market.py",
        "env_observation.py",
        "env_rewards.py",
        "env_risk.py",
        "env_tensor.py",
        "env_utils.py",
        "__init__.py"
    ]
    
    source_env_dir = os.path.join(NEWEST_STUFF_DIR, "environment")
    target_env_dir = os.path.join(SRC_DIR, "environment")
    
    # Ensure the target environment directory exists
    if not os.path.exists(target_env_dir):
        os.makedirs(target_env_dir)
        print(f"Created directory: {target_env_dir}")
    
    copied_files = 0
    for env_file in env_files:
        source_path = os.path.join(source_env_dir, env_file)
        target_path = os.path.join(target_env_dir, env_file)
        
        # Check if the file already exists in the target directory
        if not os.path.exists(target_path):
            print(f"Missing environment file: {env_file}")
            if os.path.exists(source_path):
                if copy_file(os.path.join("environment", env_file), target_path, is_environment=True):
                    copied_files += 1
            else:
                print(f"Warning: Source file not found: {source_path}")
        else:
            print(f"Environment file already exists: {env_file}")
    
    print(f"\nCopied {copied_files} missing environment files.")

def check_core_files():
    """Check if all core files from the newest stuff directory are present in src."""
    # Define core files to check (excluding environment files which are handled separately)
    core_files = [
        ("Agent V2.py", "agent/agent.py"),
        ("models v2.py", "models/models.py"),
        ("utils v2.py", "utils/utils.py"),
        ("training.py", "training/training.py"),
        ("backtesting_v2.py", "training/backtesting.py"),
        ("progressive_training.py", "training/progressive_training.py"),
        ("config v2.py", "utils/config.py"),
        ("tensor_utils v3.py", "utils/tensor_utils.py")
    ]
    
    copied_files = 0
    for source_file, target_rel_path in core_files:
        target_path = os.path.join(SRC_DIR, target_rel_path)
        
        # Check if the file already exists in the target directory
        if not os.path.exists(target_path):
            print(f"Missing core file: {target_rel_path}")
            if copy_file(source_file, target_path):
                copied_files += 1
        else:
            print(f"Core file already exists: {target_rel_path}")
    
    print(f"\nCopied {copied_files} missing core files.")

def main():
    """Check for missing files and copy them from the original directory."""
    print("Starting file copy process...\n")
    
    print("Checking environment files:")
    check_environment_files()
    
    print("\nChecking core files:")
    check_core_files()
    
    print("\nFile copy process complete. You may need to manually check the files for proper import paths.")

if __name__ == "__main__":
    main() 