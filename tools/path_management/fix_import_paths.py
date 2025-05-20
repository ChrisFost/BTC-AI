import os
import re

def list_python_files(directory):
    """Find all Python files in a directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def fix_file_imports(file_path):
    """Fix import paths in a single file."""
    print(f"Checking imports in: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    original_content = content
    
    # Fix common import patterns with both space and underscore variations
    # 1. Fix references to utils (both v2 and v2)
    content = re.sub(r'from\s+(?:src\.|\.)?utils(?:\s+v2|_v2)\s+import', r'from src.utils.utils import', content)
    
    # 2. Fix references to training module
    content = re.sub(r'from\s+(?:src\.|\.)?training\s+import', r'from src.training.training import', content)
    
    # 3. Fix references to config (both v2 and v2)
    content = re.sub(r'from\s+(?:src\.|\.)?config(?:\s+v2|_v2)\s+import', r'from src.utils.config import', content)
    
    # 4. Fix references to models (both v2 and v2)
    content = re.sub(r'from\s+(?:src\.|\.)?models(?:\s+v2|_v2)\s+import', r'from src.models.models import', content)
    
    # 5. Fix references to backtesting (both v2 and v2)
    content = re.sub(r'from\s+(?:src\.|\.)?backtesting(?:\s+v2|_v2)\s+import', r'from src.training.backtesting import', content)
    
    # 6. Fix references to Agent (both V2 and V2)
    content = re.sub(r'from\s+(?:src\.|\.)?Agent(?:\s+V2|_V2)\s+import', r'from src.agent.agent import', content)
    
    # 7. Fix environment module imports (handle relative imports)
    content = re.sub(r'from\s+(?:src\.|\.)?env_base\s+import', r'from src.environment.env_base import', content)
    content = re.sub(r'from\s+(?:src\.|\.)?env_risk\s+import', r'from src.environment.env_risk import', content)
    content = re.sub(r'from\s+(?:src\.|\.)?env_tensor\s+import', r'from src.environment.env_tensor import', content)
    content = re.sub(r'from\s+(?:src\.|\.)?env_observation\s+import', r'from src.environment.env_observation import', content)
    content = re.sub(r'from\s+(?:src\.|\.)?env_rewards\s+import', r'from src.environment.env_rewards import', content)
    content = re.sub(r'from\s+(?:src\.|\.)?env_market\s+import', r'from src.environment.env_market import', content)
    content = re.sub(r'from\s+(?:src\.|\.)?env_utils\s+import', r'from src.environment.env_utils import', content)
    content = re.sub(r'from\s+(?:src\.|\.)?env_interfaces\s+import', r'from src.environment.env_interfaces import', content)
    
    # 8. Fix tensor utilities (both v3 and v3)
    content = re.sub(r'from\s+(?:src\.|\.)?tensor_utils(?:\s+v3|_v3)\s+import', r'from src.utils.tensor_utils import', content)
    
    # 9. Fix progressive training imports
    content = re.sub(r'from\s+(?:src\.|\.)?progressive_training\s+import', r'from src.training.progressive import', content)
    
    # 10. Fix monitor training imports
    content = re.sub(r'from\s+(?:src\.|\.)?monitor_training\s+import', r'from src.ui.monitor_training import', content)
    
    # 11. Fix visualization imports
    content = re.sub(r'from\s+(?:src\.|\.)?visualize\s+import', r'from src.utils.prediction_visualizer import', content)
    
    # 12. Fix reasoning analyzer imports
    content = re.sub(r'from\s+(?:src\.|\.)?reasoning_analyzer\s+import', r'from src.utils.reasoning import', content)
    
    # 13. Fix bucket goals imports
    content = re.sub(r'from\s+(?:src\.|\.)?bucket_goals\s+import', r'from src.utils.bucket_goals import', content)
    
    # 14. Fix menu script imports (both v2 and v2)
    content = re.sub(r'from\s+(?:src\.|\.)?menu_script(?:\s+v2|_v2)\s+import', r'from src.ui.main import', content)
    
    # 15. Fix dataframe imports
    content = re.sub(r'from\s+(?:src\.|\.)?dataframe\s+import', r'from src.utils.dataframe import', content)
    
    # 16. Fix evaluate imports
    content = re.sub(r'from\s+(?:src\.|\.)?evaluate\s+import', r'from src.utils.evaluate import', content)
    
    # 17. Fix bare imports (no from)
    content = re.sub(r'import\s+(?:src\.)?utils(?:\s+v2|_v2)', r'import src.utils.utils', content)
    content = re.sub(r'import\s+(?:src\.)?config(?:\s+v2|_v2)', r'import src.utils.config', content)
    content = re.sub(r'import\s+(?:src\.)?models(?:\s+v2|_v2)', r'import src.models.models', content)
    content = re.sub(r'import\s+(?:src\.)?backtesting(?:\s+v2|_v2)', r'import src.training.backtesting', content)
    content = re.sub(r'import\s+(?:src\.)?Agent(?:\s+V2|_V2)', r'import src.agent.agent', content)
    content = re.sub(r'import\s+(?:src\.)?tensor_utils(?:\s+v3|_v3)', r'import src.utils.tensor_utils', content)
    content = re.sub(r'import\s+(?:src\.)?menu_script(?:\s+v2|_v2)', r'import src.ui.main', content)
    
    # Save changes if there were any
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Fixed imports in: {file_path}")
        return True
    else:
        print(f"  No import fixes needed for: {file_path}")
        return False

def main():
    """Main function to fix import paths in all Python files in the src directory."""
    src_dir = "src"
    python_files = list_python_files(src_dir)
    
    print(f"Found {len(python_files)} Python files to check")
    
    fixed_files = 0
    for file_path in python_files:
        if fix_file_imports(file_path):
            fixed_files += 1
    
    print(f"\nFixed imports in {fixed_files} of {len(python_files)} files")
    
    # Check specifically for critical files
    critical_files = [
        os.path.join(src_dir, "training", "progressive_training.py"),
        os.path.join(src_dir, "environment", "__init__.py"),
        os.path.join(src_dir, "environment", "env_tensor.py"),
        os.path.join(src_dir, "environment", "env_observation.py"),
        os.path.join(src_dir, "environment", "env_risk.py"),
        os.path.join(src_dir, "environment", "env_base.py"),
        os.path.join(src_dir, "environment", "env_rewards.py"),
        os.path.join(src_dir, "environment", "env_market.py"),
        os.path.join(src_dir, "environment", "env_utils.py"),
        os.path.join(src_dir, "environment", "env_interfaces.py"),
        os.path.join(src_dir, "utils", "tensor_utils.py"),
        os.path.join(src_dir, "utils", "dataframe.py"),
        os.path.join(src_dir, "utils", "evaluate.py"),
        os.path.join(src_dir, "utils", "visualization.py"),
        os.path.join(src_dir, "utils", "reasoning.py"),
        os.path.join(src_dir, "utils", "bucket_goals.py"),
        os.path.join(src_dir, "ui", "monitor_training.py"),
        os.path.join(src_dir, "ui", "main.py")
    ]
    
    print("\nChecking critical files again:")
    for file_path in critical_files:
        if os.path.exists(file_path):
            fix_file_imports(file_path)
        else:
            print(f"Critical file not found: {file_path}")

if __name__ == "__main__":
    main() 