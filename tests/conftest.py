import sys
from pathlib import Path

# Repo root on sys.path so `from app...` / `from evals...` imports work
# regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()
