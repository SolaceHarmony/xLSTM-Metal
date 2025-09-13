#!/usr/bin/env python3
"""
Script to systematically remove all try/except Exception: pass blocks
"""

import re
from pathlib import Path

def remove_exception_blocks(file_path: Path):
    """Remove try/except Exception: blocks from a file"""
    content = file_path.read_text()
    
    # Pattern to match try/except Exception: pass/return patterns
    patterns = [
        # Pattern 1: try: ... except Exception: pass
        r'(\s*)try:\s*\n((?:\1    .*\n)*)\1except Exception:\s*\n\1    pass\n',
        # Pattern 2: try: ... except Exception: return/continue
        r'(\s*)try:\s*\n((?:\1    .*\n)*)\1except Exception:\s*\n\1    (?:return|continue).*\n',
        # Pattern 3: if something: try: ... except Exception: pass
        r'(\s*)if .*:\s*\n\1    try:\s*\n((?:\1        .*\n)*)\1    except Exception:\s*\n\1        pass\n',
    ]
    
    for pattern in patterns:
        while True:
            matches = list(re.finditer(pattern, content, re.MULTILINE))
            if not matches:
                break
            
            for match in reversed(matches):  # Process from end to avoid offset issues
                indent = match.group(1)
                try_content = match.group(2)
                
                # Remove one level of indentation from try content
                dedented_content = re.sub(f'^{indent}    ', indent, try_content, flags=re.MULTILINE)
                
                # Replace the entire try/except block with just the dedented content
                content = content[:match.start()] + dedented_content + content[match.end():]
    
    file_path.write_text(content)
    print(f"Fixed {file_path}")

def main():
    files_to_fix = [
        "xlstm-solace-mlx/src/xlstm_mlx/kernels/gemm_kernels.py",
        "tools/mlx_tuning.py",
        "scripts/build/build_metal_extension.py",
        "tests/test_xlstm.py"
    ]
    
    root = Path(".")
    for file_path in files_to_fix:
        full_path = root / file_path
        if full_path.exists():
            remove_exception_blocks(full_path)
        else:
            print(f"File not found: {full_path}")

if __name__ == "__main__":
    main()
