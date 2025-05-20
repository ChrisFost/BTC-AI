#!/usr/bin/env python
"""
Diagnose and fix the specific line in agent.py causing the indentation error
"""

import re

file_path = 'src/agent/agent.py'

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
    
# Examine content around "mixed_factor" and "else:"
mixed_factor_pos = content.find('mixed_factor')
if mixed_factor_pos > 0:
    # Print context around mixed_factor
    print("Context around mixed_factor:")
    context_start = max(0, mixed_factor_pos - 200)
    context_end = min(len(content), mixed_factor_pos + 200)
    print(repr(content[context_start:context_end]))
    print("\n" + "-"*80 + "\n")

# Find the problematic line with trailing "else:"
pattern = re.compile(r'(regime_adjustment\s*=\s*trending_factor\s*\+\s*ranging_factor\s*\+\s*volatile_factor\s*\+\s*mixed_factor).*?(\s*regime_adjustment\s*=\s*trending_factor\s*\+\s*ranging_factor\s*\+\s*volatile_factor\s*\+\s*else)', re.DOTALL)
match = pattern.search(content)

if match:
    # Found the problematic section
    print("Found problematic duplicate line pattern.")
    print(f"Match group 1: {repr(match.group(1))}")
    print(f"Match group 2: {repr(match.group(2))}")
    
    # Fix by replacing the corrupted line
    start_pos = match.start(2)
    end_pos = match.end(2)
    
    # Calculate proper indentation level
    line_start = content.rfind('\n', 0, start_pos) + 1
    indent = content[line_start:start_pos].split('regime_adjustment')[0]
    
    # Create replacement line with proper indentation
    replacement = f"{indent}else:"
    
    # Apply the fix
    fixed_content = content[:start_pos] + replacement + content[end_pos:]
    
    # Write the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed corrupted line in {file_path}")
    
else:
    # Try an alternative approach - look for duplicate lines by line
    lines = content.splitlines()
    fixed = False
    
    for i in range(len(lines) - 1):
        if 'mixed_factor' in lines[i] and 'regime_adjustment' in lines[i]:
            if 'regime_adjustment' in lines[i+1] and 'else:' in lines[i+1]:
                # Found the issue - duplicate line with partial "else:"
                print(f"Found duplicate line issue at lines {i+1}-{i+2}")
                print(f"Line {i+1}: {repr(lines[i])}")
                print(f"Line {i+2}: {repr(lines[i+1])}")
                
                # Calculate indentation by counting leading spaces
                indent_level = len(lines[i]) - len(lines[i].lstrip())
                proper_indent = ' ' * indent_level
                
                # Replace line with properly indented else
                fixed_lines = lines.copy()
                fixed_lines[i+1] = proper_indent + 'else:'
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(fixed_lines))
                
                fixed = True
                print(f"Fixed corrupt line at line {i+2}")
                break
    
    if not fixed:
        print("No issues found with standard approaches. Manual examination needed.")
        
        # Print lines in the area where the issue is expected
        print("\nLines in expected problem area (405-415):")
        for i, line in enumerate(lines[405:415], 406):
            print(f"Line {i}: {repr(line)}") 