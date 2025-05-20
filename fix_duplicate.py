#!/usr/bin/env python
"""
Fix duplicate regime_adjustment line in agent.py
"""

file_path = 'src/agent/agent.py'

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Look for the problematic sequence
fixed = False
for i in range(len(lines) - 1):
    if 'trending_factor + ranging_factor + volatile_factor + mixed_factor' in lines[i] and 'trending_factor + ranging_factor + volatile_factor + else:' in lines[i+1]:
        # Found the issue - duplicate line with partial "else:"
        print(f"Found duplicate line issue at lines {i+1}-{i+2}")
        print(f"Line {i+1}: {lines[i].strip()}")
        print(f"Line {i+2}: {lines[i+1].strip()}")
        
        # Fix the issue by replacing the corrupt line with a proper else statement
        indent_level = len(lines[i]) - len(lines[i].lstrip())
        proper_indent = ' ' * indent_level
        lines[i+1] = proper_indent + 'else:\n'
        
        fixed = True
        print(f"Fixed corrupt line at line {i+2}")
        break

# Write the file if changes were made
if fixed:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"Successfully updated {file_path}")
else:
    print("No duplicate regime_adjustment line found.") 