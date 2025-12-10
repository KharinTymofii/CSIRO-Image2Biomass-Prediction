from pathlib import Path
import sys

# Add parent directory to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

try:
    from src.utils.config import *
    print("Available variables:", [var for var in dir() if not var.startswith('_')])
    from src.utils.logging_config import setup_logging
    from src.utils.loggers import CustomLogger
except ImportError as e:
    print(f"Import error: {e}")