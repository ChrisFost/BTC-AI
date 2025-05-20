#!/usr/bin/env python
"""
Fix else:: typo in agent.py
"""

file_path = 'src/agent/agent.py'

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace else:: with else:
fixed_content = content.replace('else::', 'else:')

# Check if any changes were made
if fixed_content != content:
    # Write the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    print(f"Fixed 'else::' typo in {file_path}")
else:
    print("No 'else::' typo found in the file.") 