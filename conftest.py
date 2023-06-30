"""Configuration for pytest.

It's needed here to add the option `--runslow`, which is mainly used in the lattice
subpackage.
"""
import pytest


def pytest_addoption(parser):
    """Add option '--runslow' and '--rungmsh'."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--rungmsh", action="store_true", default=False, help="run tests needing gmsh"
    )


def pytest_configure(config):
    """Add marker 'slow' and 'gmsh'."""
    config.addinivalue_line("markers", "gmsh: test needs gmsh")
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Skip tests without necessary options."""
    if not config.getoption("--runslow"):
        # --runslow not given in cli: skip slow tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--rungmsh"):
        # --rungmsh not given in cli: skip gmsh tests
        skip_slow = pytest.mark.skip(reason="need --rungmsh option to run")
        for item in items:
            if "gmsh" in item.keywords:
                item.add_marker(skip_slow)
