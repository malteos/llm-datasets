from pathlib import Path
import pytest
import random
import os

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
RANDOM_SEED = 0

random.seed(RANDOM_SEED)


if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


@pytest.fixture(scope="module")
def raw_datasets_dir():
    return FIXTURES_DIR / "raw_datasets"
