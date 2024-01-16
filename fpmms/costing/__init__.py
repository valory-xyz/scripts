import sys
from pathlib import Path


file_path = Path(__file__).resolve().parent.parent
sys.path.append(f"{file_path}")
