#!/usr/bin/env python3
"""
Script to refactor voicechatengine to voxengine throughout the codebase.
"""

import os
import re
import sys

def update_file(filepath):
    """Update a single file replacing voicechatengine with voxengine."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False
    
    # Store original content for comparison
    original_content = content
    
    # Replace patterns
    replacements = [
        # Package imports
        (r'from voxengine\b', 'from voxengine'),
        (r'import voxengine\b', 'import voxengine'),
        
        # Module paths in strings
        (r'voicechatengine\.', 'voxengine.'),
        
        # Package name in strings
        (r'"voxengine"', '"voxengine"'),
        (r"'voxengine'", "'voxengine'"),
        
        # Comments and docstrings
        (r'voicechatengine\s*-\s*Modern', 'voxengine - Modern'),
        (r'# here is voxengine/', '# here is voxengine/'),
        (r'# voxengine\b', '# voxengine'),
        
        # Test commands
        (r'python -m\s+voicechatengine', 'python -m voxengine'),
        
        # Directory references
        (r'/voxengine/', '/voxengine/'),
        (r'\\\voxengine\\\', r'\\voxengine\\'),
        
        # Package descriptions
        (r'VoxEngine', 'VoxEngine'),
        (r'Vox Engine', 'Vox Engine'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Only write if content changed
    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Updated: {filepath}")
            return True
        except Exception as e:
            print(f"✗ Error writing {filepath}: {e}")
            return False
    else:
        return False

def find_files_to_update(root_dir):
    """Find all files that need updating."""
    files_to_update = []
    
    # File extensions to process
    extensions = {'.py', '.md', '.txt', '.yml', '.yaml', '.json', '.toml', '.cfg', '.ini'}
    
    # Special files without extensions
    special_files = {'README', 'LICENSE', 'MANIFEST', 'Makefile'}
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden directories and __pycache__
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
        
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            
            # Check if file should be processed
            if any(filename.endswith(ext) for ext in extensions) or filename in special_files:
                files_to_update.append(filepath)
    
    return files_to_update

def main():
    """Main refactoring function."""
    print("=== VoxEngine → VoxEngine Refactoring ===\n")
    
    # Get project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Find all files to update
    print("Scanning for files to update...")
    files = find_files_to_update(project_root)
    print(f"Found {len(files)} files to check\n")
    
    # Update files
    updated_count = 0
    for filepath in files:
        if update_file(filepath):
            updated_count += 1
    
    print(f"\n✓ Updated {updated_count} files")
    
    # Special case: setup.py
    setup_path = os.path.join(project_root, 'setup.py')
    if os.path.exists(setup_path):
        print("\n⚠️  Don't forget to update setup.py package name!")
    
    print("\n=== Refactoring Complete ===")
    print("\nNext steps:")
    print("1. Review the changes")
    print("2. Run tests to ensure everything works")
    print("3. Update any external documentation or references")
    print("4. Commit the changes")

if __name__ == "__main__":
    main()