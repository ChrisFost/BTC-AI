#!/usr/bin/env python
"""
Fix indentation in agent.py
"""

import re

file_path = 'src/agent/agent.py'

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
    
# Fix the first issue (with torch.no_grad indentation)
pattern = re.compile(r'(\s+if state is not None and hasattr\(self\.model, \'market_regime_head\'\):)\s+(\1)(with torch\.no_grad\(\):)')
fixed_content = pattern.sub(r'\1\n\1    \3', content)

# Check if first issue was fixed
first_issue_fixed = fixed_content != content

# Write the changes back to a temporary content
content = fixed_content

# Convert to lines for the second fix
lines = content.splitlines(True)  # Keep line endings

# Look for and fix unexpected indentation around line 412
second_issue_fixed = False
for i in range(407, 417):
    if i < len(lines) and lines[i].strip() == 'else:':
        # Check if it's unexpectedly indented
        if i > 0 and 'mixed_factor' in lines[i-1]:
            # Get the proper indentation from surrounding lines
            proper_indent = '        '  # Default to 8 spaces
            
            # Try to find matching 'if' statement
            for j in range(max(0, i-10), i):
                if 'if' in lines[j] and not lines[j].strip().startswith('#'):
                    # Extract the indentation
                    indent_match = re.match(r'^(\s+)', lines[j])
                    if indent_match:
                        proper_indent = indent_match.group(1)
                        break
            
            # Fix the indentation
            lines[i] = proper_indent + 'else:\n'
            second_issue_fixed = True
            print(f"Fixed unexpected indentation at line {i+1}")
            break

# Also check for duplicate line
duplicate_fixed = False
for i in range(len(lines)-1):
    if i > 0 and 'regime_adjustment = trending_factor + ranging_factor + volatile_factor + ' in lines[i]:
        if 'mixed_factor' in lines[i] and 'trending_factor + ranging_factor + volatile_factor + else:' in lines[i+1]:
            # Found duplicate/corrupted line, fix it
            lines[i+1] = lines[i].replace('mixed_factor', 'else:\n')
            duplicate_fixed = True
            print(f"Fixed corrupted line at line {i+2}")
            break

# Write the file if changes were made
if first_issue_fixed or second_issue_fixed or duplicate_fixed:
    with open(file_path, 'w', encoding='utf-8') as f:
        if second_issue_fixed or duplicate_fixed:
            # Write lines if we did line-by-line changes
            f.writelines(lines)
        else:
            # Otherwise write the content with regex changes
            f.write(content)
    
    print(f"Successfully updated {file_path}")
    if first_issue_fixed:
        print("Fixed indentation for torch.no_grad block")
    if second_issue_fixed:
        print("Fixed unexpected indentation for else statement")
    if duplicate_fixed:
        print("Fixed duplicate/corrupted line")
else:
    print("No indentation issues found in the agent.py file.")
