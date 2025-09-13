#!/usr/bin/env python3
"""
Nuclear option: Remove ALL exception handlers from ALL Python files.
No mercy. No fallbacks. Fail fast and loud.
"""
import re
import os
from pathlib import Path

def remove_all_exceptions(file_path):
    """Remove ALL exception handling from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Remove all except blocks that just pass or return defaults
    patterns = [
        # except Exception: pass
        r'except\s+[^:]*:\s*\n\s*pass\s*\n',
        # except Exception: return default
        r'except\s+[^:]*:\s*\n\s*return[^\n]*\n',
        # except Exception: <single statement>
        r'except\s+[^:]*:\s*\n\s*[^#\n][^\n]*\n',
    ]
    
    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.MULTILINE)
    
    # Remove orphaned try: statements
    content = re.sub(r'try:\s*\n(?=\s*[a-zA-Z_])', '', content, flags=re.MULTILINE)
    
    # Clean up extra blank lines
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Nuked exceptions in: {file_path}")
        return True
    return False

def main():
    # Target files with high exception counts
    targets = [
        "tools/mlx_tuning.py",
        "tools/telem/aggregate.py", 
        "xlstm-solace-mlx/src/xlstm_mlx/kernels/gemm_kernels.py",
        "scripts/optimizer/optimize_mps.py",
        "scripts/judge_with_ollama.py",
    ]
    
    for target in targets:
        target_path = Path(target)
        if target_path.exists():
            print(f"Processing {target}...")
            remove_all_exceptions(target_path)

if __name__ == "__main__":
    main()
