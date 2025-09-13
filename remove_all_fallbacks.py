#!/usr/bin/env python3
"""
Remove ALL fallback patterns - no conditions, no exceptions, no fallbacks.
"""
import re
from pathlib import Path

def remove_all_fallbacks(file_path):
    """Remove ALL fallback patterns from a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Remove all "is None" checks and their else blocks
    content = re.sub(r'if\s+[^=]*\s+is\s+None:\s*\n.*?(?=\n\s*[a-zA-Z_]|\n\n|\Z)', '', content, flags=re.DOTALL)
    content = re.sub(r'if\s+[^=]*\s+is\s+not\s+None:\s*\n(.*?)(?=\n\s*[a-zA-Z_]|\n\n|\Z)', r'\1', content, flags=re.DOTALL)
    
    # Remove all try/except blocks
    content = re.sub(r'try:\s*\n(.*?)\nexcept[^:]*:.*?(?=\n\s*[a-zA-Z_]|\n\n|\Z)', r'\1', content, flags=re.DOTALL)
    
    # Clean up extra blank lines
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Stripped ALL fallbacks from: {file_path}")
        return True
    return False

def main():
    # Target the kernels file specifically
    kernel_file = Path("xlstm-solace-mlx/src/xlstm_mlx/kernels/gemm_kernels.py")
    if kernel_file.exists():
        remove_all_fallbacks(kernel_file)

if __name__ == "__main__":
    main()
