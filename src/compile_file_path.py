import os
from pathlib import Path


def get_file_path(filepath: str) -> str:
    """
    Get the absolute file path for a file outside the src directory.
    """
    # src/this_script.py → go one folder up → project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, filepath)
