#!/usr/bin/env python3
"""
Setup the fraud detection project structure.
"""

import os
import shutil
from pathlib import Path


def create_project_structure():
    """Create the complete project structure."""
    base_dir = Path('fraud-detection')
    
    # Directories to create
    directories = [
        '.vscode',
        '.github/workflows',
        'data/raw',
        'data/processed',
        'notebooks',
        'src',
        'tests',
        'models',
        'scripts',
        'reports'
    ]
    
    print("Creating project structure...")
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {dir_path}")
    
    # Create __init__.py files
    init_files = [
        'notebooks/__init__.py',
        'src/__init__.py',
        'tests/__init__.py',
        'scripts/__init__.py',
        'reports/__init__.py'
    ]
    
    for init_file in init_files:
        file_path = base_dir / init_file
        file_path.touch()
        print(f"  Created: {file_path}")
    
    # Create .gitkeep files for empty directories
    gitkeep_files = [
        'data/raw/.gitkeep',
        'data/processed/.gitkeep',
        'models/.gitkeep',
        'reports/.gitkeep'
    ]
    
    for gitkeep in gitkeep_files:
        file_path = base_dir / gitkeep
        file_path.touch()
        print(f"  Created: {file_path}")
    
    print("\nProject structure created successfully!")
    
    # Create instructions file
    instructions = base_dir / 'INSTRUCTIONS.md'
    with open(instructions, 'w') as f:
        f.write("""# Fraud Detection Project - Setup Instructions

## 1. Download Datasets
Download these files and place them in `data/raw/`:
1. Fraud_Data.csv
2. IpAddress_to_Country.csv
3. creditcard.csv

## 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate""")