"""
Pytest configuration for Multi-Agent Orchestration tests.
Ensures proper import paths.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import src and config
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
