import os
import re
import sys

# Files with known import issues
problem_files = [
    "src/agent/agent.py",
    "src/training/backtesting.py",
    "gradient_verification.py",
    "test_calibration_flow.py",
    "test_calibration.py"
]

# Module path fixes for specific import errors
module_path_fixes = {
    "tensor_utils v3": "src.utils.tensor_utils",
    "utils v2": "src.utils.utils",
    "Agent V2": "src.agent.agent",
    "models v2": "src.models.models"
}

def preserve_dynamic_imports(file_path):
    """Fix dynamic imports while preserving their functionality."""
    print(f"Fixing dynamic imports in: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    original_content = content
    fixed_content = content
    
    # Check if the file contains importlib usage
    if "importlib" in content and "import_module" in content:
        # Check gradient_verification.py and test_calibration_flow.py
        if os.path.basename(file_path) in ["gradient_verification.py", "test_calibration_flow.py"]:
            # Fix the models import to use proper dynamic import
            fixed_content = re.sub(
                r'from src\.models\.models import ActorCritic,.*?\n(ActorCritic = models_module\.ActorCritic)',
                r'# Use dynamic import for models\nmodels_module = importlib.import_module("src.models.models")\n\1',
                fixed_content
            )
            
            # If missing importlib, add it
            if "import importlib" not in fixed_content:
                fixed_content = re.sub(
                    r'import sys',
                    r'import sys\nimport importlib',
                    fixed_content
                )
    
    # Check backtesting.py for dynamic imports
    if os.path.basename(file_path) == "backtesting.py":
        # Preserve fallback mechanism for utilities
        fallback_utils = r'''try:
    from src.utils.utils import (
        log, validate_dataframe, calculate_metrics, format_metrics,
        optimize_memory, visualize_metrics
    )
except ImportError:
    # Fallback for older code structure
    print("Warning: Could not import from src.utils.utils, using fallback imports")
    try:
        utils_module = importlib.import_module("utils v2")
        log = utils_module.log
        validate_dataframe = utils_module.validate_dataframe
        calculate_metrics = utils_module.calculate_metrics
        format_metrics = utils_module.format_metrics
        optimize_memory = utils_module.optimize_memory
        visualize_metrics = utils_module.visualize_metrics
    except ImportError:
        print("Error: Could not import utilities, backtesting may not function correctly")'''
            
        if "utils_module = importlib.import_module" not in fixed_content:
            fixed_content = re.sub(
                r'try:\s+from src\.utils\.utils import.*?except ImportError:.*?print\("Error: Could not import utilities, backtesting may not function correctly"\)',
                fallback_utils,
                fixed_content, 
                flags=re.DOTALL
            )
            
        # Preserve fallback mechanism for tensor utils
        fallback_tensor = r'''try:
    from src.utils.tensor_utils import (
        compute_fractal_dimension_tensor, detect_elliott_wave_pattern_tensor,
        compute_market_fractals_tensor, compute_timeframe_wavelet_features
    )
except ImportError:
    # Fallback for older code structure
    print("Warning: Could not import from src.utils.tensor_utils, using fallback imports")
    try:
        tensor_utils_module = importlib.import_module("tensor_utils v3")
        compute_fractal_dimension_tensor = tensor_utils_module.compute_fractal_dimension_tensor
        detect_elliott_wave_pattern_tensor = tensor_utils_module.detect_elliott_wave_pattern_tensor
        compute_market_fractals_tensor = tensor_utils_module.compute_market_fractals_tensor
        compute_timeframe_wavelet_features = tensor_utils_module.compute_timeframe_wavelet_features
    except ImportError:
        print("Error: Could not import tensor utilities, advanced features will be disabled")'''
            
        if "tensor_utils_module = importlib.import_module" not in fixed_content:
            fixed_content = re.sub(
                r'try:\s+from src\.utils\.tensor_utils import.*?except ImportError:.*?print\("Error: Could not import tensor utilities, advanced features will be disabled"\)',
                fallback_tensor,
                fixed_content, 
                flags=re.DOTALL
            )
            
    if fixed_content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        return True
    else:
        print(f"  No dynamic import changes needed for: {file_path}")
        return False

def fix_specific_file(file_path):
    """Apply specialized fixes for specific files while preserving dynamic imports."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    basename = os.path.basename(file_path)
    
    # Special case for agent.py
    if basename == "agent.py":
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Add importlib for dynamic imports if needed
        if "import importlib" not in content and ("importlib" in content or "import_module" in content):
            content = re.sub(r'import os', r'import os\nimport importlib', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Applied specialized fixes for {basename}")
        return True
    
    return False

def main():
    """Main function to fix import issues in Python files."""
    print("Starting import fixes with dynamic path preservation...")
    
    # Fix each problem file
    fixed_files = 0
    for file_path in problem_files:
        if os.path.exists(file_path):
            # Apply dynamic import fixes
            if preserve_dynamic_imports(file_path):
                fixed_files += 1
            
            # Apply specialized fixes
            fix_specific_file(file_path)
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nFixed imports in {fixed_files} files while preserving dynamic path functionality.")
    print("\nImport fixes complete.")

if __name__ == "__main__":
    main() 