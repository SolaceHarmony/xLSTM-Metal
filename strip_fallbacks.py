#!/usr/bin/env python3
"""
Strip all fallback patterns from MLX code.
"""
import re
import os
from pathlib import Path

def remove_fallback_patterns(file_path):
    """Remove all try/except fallback patterns from a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: try/except with silent pass or return default
    patterns = [
        # try: ... except: pass
        r'try:\s*\n(.*?)\n\s*except[^:]*:\s*\n\s*pass',
        # try: ... except: return default
        r'try:\s*\n(.*?)\n\s*except[^:]*:\s*\n\s*return[^\\n]*',
        # try: ... except Exception: ...
        r'try:\s*\n(.*?)\n\s*except Exception:\s*\n(.*?)(?=\n\S|\n$)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, content, re.DOTALL | re.MULTILINE)
        for match in reversed(list(matches)):
            # Extract the try block content
            try_content = match.group(1)
            # Replace the entire try/except with just the try content
            content = content[:match.start()] + try_content + content[match.end():]
    
    # More aggressive: remove any line with "except" that doesn't re-raise
    lines = content.split('\n')
    new_lines = []
    skip_until_dedent = False
    indent_level = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip lines in except blocks that don't re-raise
        if skip_until_dedent:
            current_indent = len(line) - len(line.lstrip())
            if line.strip() == '' or current_indent > indent_level:
                i += 1
                continue
            else:
                skip_until_dedent = False
        
        # Check for except blocks
        if 'except' in line and ':' in line:
            # Look ahead to see if it re-raises
            j = i + 1
            has_raise = False
            block_indent = len(line) - len(line.lstrip())
            
            while j < len(lines):
                next_line = lines[j]
                if next_line.strip() == '':
                    j += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= block_indent:
                    break
                if 'raise' in next_line:
                    has_raise = True
                    break
                j += 1
            
            if not has_raise:
                # Skip this except block
                skip_until_dedent = True
                indent_level = block_indent
                i += 1
                continue
        
        # Also remove try: lines that are orphaned
        if line.strip() == 'try:':
            # Check if next non-empty line is except
            j = i + 1
            has_except = False
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            if j < len(lines) and 'except' in lines[j]:
                # Skip the try line if we're going to skip the except
                i += 1
                continue
        
        new_lines.append(line)
        i += 1
    
    content = '\n'.join(new_lines)
    
    # Clean up extra blank lines
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Stripped fallbacks from: {file_path}")
        return True
    return False

def main():
    # Find all Python files in the MLX source
    mlx_src = Path("xlstm-solace-mlx/src/xlstm_solace_mlx")
    
    for py_file in mlx_src.rglob("*.py"):
        if remove_fallback_patterns(py_file):
            print(f"Modified: {py_file}")

if __name__ == "__main__":
    main()
