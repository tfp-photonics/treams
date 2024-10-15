"""Configuration for pytest.

It's needed here to add the option `--runslow`, which is mainly used in the lattice
subpackage.
"""
import pytest
import pathlib

def pytest_addoption(parser):
    """Add option '--runslow' and '--rungmsh'."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--rungmsh", action="store_true", default=False, help="run tests needing gmsh"
    )
    parser.addini("datadir", "my own datadir for pytest-regressions")
    parser.addini("original_datadir", "my own original_datadir for pytest-regressions")


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

@pytest.fixture()
def original_datadir(request) -> pathlib.Path:
    config = request.config
    return config.rootpath / config.getini('datadir')


@pytest.fixture()
def datadir(request) -> pathlib.Path:
    config = request.config
    return config.rootpath / config.getini('datadir')