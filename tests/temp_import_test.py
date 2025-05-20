import sys
import os

# Get the current working directory
cwd = os.getcwd()

# Add project root based on the assumption that this script is in tests/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Go up one level from tests/

# Add both CWD and calculated project root to sys.path if not already present
if cwd not in sys.path:
    sys.path.insert(0, cwd)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("--- sys.path --- DUMP BEGIN ---")
for p in sys.path:
    print(p)
print("--- sys.path --- DUMP END ---")


# Path to the file in question
training_file_path = os.path.join(project_root, "src", "training", "training.py")

print(f"\nChecking file content directly: {training_file_path}")

if not os.path.exists(training_file_path):
    print(f"ERROR: File not found at {training_file_path}")
else:
    try:
        with open(training_file_path, "rb") as f:
            file_bytes = f.read()
        print("Successfully read file bytes.")

        # Check for null byte
        null_byte = b'\x00'
        if null_byte in file_bytes:
            print(f"*** FOUND NULL BYTE (b'\x00') in {training_file_path}! ***")
            # Optionally, find the position
            try:
                position = file_bytes.index(null_byte)
                print(f"Null byte found at byte offset: {position}")
            except ValueError:
                pass # Should not happen if 'in' check passed
        else:
            print("No literal null byte (b'\x00') found.")

        # Attempt to decode as UTF-8
        try:
            decoded_content = file_bytes.decode('utf-8')
            print("Successfully decoded file content as UTF-8.")
        except UnicodeDecodeError as e_decode:
            print(f"*** FAILED TO DECODE file as UTF-8: {e_decode} ***")
            print("This is likely the cause of the 'source code string cannot contain null bytes' error.")

    except Exception as e_read:
        print(f"ERROR reading file {training_file_path}: {e_read}")


print("\n--- Import Test (for comparison) ---")
try:
    # Try importing the specific function first, as in progressive.py
    from src.training.training import train_model
    print("Import successful using 'from ... import ...'")
except ValueError as e:
    print(f"Caught ValueError during 'from ... import ...': {e}")
except ImportError as e_import:
    print(f"Caught ImportError during initial 'from ... import ...': {e_import}")
except Exception as e_initial:
    print(f"Caught other exception during initial 'from ... import ...': {e_initial}")


print("\nTemporary check finished.") 