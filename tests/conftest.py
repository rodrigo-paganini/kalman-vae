import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-stability", action="store_true", default=False, help="run slow tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "integration: mark test as integration test")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--no-stability"):
        skip_stability = pytest.mark.skip(reason="need --no-stability option to run")
        for item in items:
            if "stability" in item.module.__name__:
                item.add_marker(skip_stability)
