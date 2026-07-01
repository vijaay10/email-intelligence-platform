"""
Test runner — discovers and runs the whole suite.

Run:
    python tests/run_tests.py
    ./run.sh test

Tests that require optional packages (scikit-learn for the ML model) are
skipped automatically when those packages are absent, so the suite stays green
on a minimal install. Install everything (`pip install -r requirements.txt`) to
exercise the full ML pipeline too.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.dirname(__file__),
                            pattern="test_*.py", top_level_dir=ROOT)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
