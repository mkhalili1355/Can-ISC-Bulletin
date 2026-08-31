"""Run the whole test suite without pytest: python tests/run_tests.py"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import harness

MODULES = ["test_geometry", "test_stats", "test_isc_io", "test_analysis"]


def main():
    total_passed = total_failed = 0
    for name in MODULES:
        if not os.path.exists(os.path.join(HERE, name + ".py")):
            continue
        print("\n%s" % name)
        module = importlib.import_module(name)
        passed, failed = harness.run_module(module)
        total_passed += passed
        total_failed += failed
    print("\n%d passed, %d failed" % (total_passed, total_failed))
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
