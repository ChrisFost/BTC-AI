import os
import re
import shutil

# Define paths
NEWEST_STUFF_DIR = "Scripts/newest stuff"
SRC_DIR = "src"

# Define the critical files to copy
files_to_copy = [
    # Format: (source file, destination file)
    ("progressive_training.py", "src/training/progressive_training.py"),
    ("environment/__init__.py", "src/environment/__init__.py")
]

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
    elif "training/" in file_path:
        # For training files, adjust imports based on new structure
        content = re.sub(r'from\s+Agent\s+V2', r'from src.agent.agent', content)
        content = re.sub(r'from\s+models\s+v2', r'from src.models.models', content)
        content = re.sub(r'from\s+utils\s+v2', r'from src.utils.utils', content)
        content = re.sub(r'from\s+config\s+v2', r'from src.utils.config', content)
        content = re.sub(r'from\s+backtesting_v2', r'from src.training.backtesting', content)
    
    return content

def copy_file(source_file, dest_file):
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

def main():
    """Main function to copy the critical files."""
    print("Starting file copy process...")
    
    successful_copies = 0
    
    for source_file, dest_file in files_to_copy:
        if copy_file(source_file, dest_file):
            successful_copies += 1
    
    print(f"\nCopied {successful_copies} of {len(files_to_copy)} files.")
    print("You may need to manually check the files for proper import paths and other adjustments.")

if __name__ == "__main__":
    main() 