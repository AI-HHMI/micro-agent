import pytest

from micro_agent.registry import Registry


@pytest.fixture(scope="session")
def registry() -> Registry:
    return Registry()
