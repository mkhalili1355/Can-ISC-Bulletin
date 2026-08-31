"""
Dependency-free test harness.

The tests are also valid pytest tests, so `python -m pytest tests` works if
pytest is installed. If it is not, `python tests/run_tests.py` runs them with
no third-party dependency, so that checking the analysis code requires no
installation.
"""

import traceback


class _Approx(object):
    def __init__(self, expected, abs_tol=1e-9, rel_tol=1e-9):
        self.expected = float(expected)
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)

    def __eq__(self, other):
        other = float(other)
        tolerance = max(self.abs_tol, self.rel_tol * abs(self.expected))
        return abs(other - self.expected) <= tolerance

    def __repr__(self):
        return "approx(%r +/- %g)" % (self.expected, self.abs_tol)


def approx(expected, abs=None, rel=None):
    kwargs = {}
    if abs is not None:
        kwargs["abs_tol"] = abs
    if rel is not None:
        kwargs["rel_tol"] = rel
    if not kwargs:
        kwargs = {"abs_tol": 1e-9, "rel_tol": 1e-9}
    return _Approx(expected, **kwargs)


class raises(object):
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError("expected %s, none raised"
                                 % self.exc_type.__name__)
        return issubclass(exc_type, self.exc_type)


def run_module(module):
    """Run every callable named test_* in `module`. Returns (passed, failed)."""
    names = sorted(name for name in dir(module) if name.startswith("test_"))
    passed, failed = 0, 0
    for name in names:
        try:
            getattr(module, name)()
            print("  pass  %s" % name)
            passed += 1
        except Exception:
            print("  FAIL  %s" % name)
            print("        " + traceback.format_exc().replace("\n", "\n        "))
            failed += 1
    return passed, failed
