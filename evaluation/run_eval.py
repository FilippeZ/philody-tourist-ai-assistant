"""
Evaluation Runner wrapper for compatibility with `python evaluation/run_eval.py`.
Loads evaluation_dataset.json and executes all deterministic test assertions.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from evaluate_dataset import main

if __name__ == "__main__":
    main()
