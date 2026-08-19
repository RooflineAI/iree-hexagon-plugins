from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runtime-so",
        required=True,
        help="path to the DSP runtime shared object to inspect",
    )


@pytest.fixture
def runtime_so(request: pytest.FixtureRequest) -> Path:
    return Path(request.config.getoption("--runtime-so"))
