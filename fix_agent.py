#!/usr/bin/env python
"""
Fix duplicate line causing indentation error in agent.py
"""

import re

file_path = 'agent.py'

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the duplicate line and indentation issue
pattern = r'(# Weight total adjustment by strongest regime[\s\n]+regime_adjustment = trending_factor \+ ranging_factor \+ volatile_factor \+ mixed_factor)[\s\n]+regime_adjustment = trending_factor \+ ranging_factor \+ volatile_factor \+ else:'
replacement = r'\1\n        else:'

fixed_content = re.sub(pattern, replacement, content)

# Write the fixed content back to the file
if fixed_content != content:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    print(f"Fixed duplicate line in {file_path}")
else:
    print("No matching pattern found, no changes made.") 