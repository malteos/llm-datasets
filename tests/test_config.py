import os

from llm_datasets.utils.config import Config, get_config_from_paths

from tests.conftest import FIXTURES_DIR


def test_config_from_yamls():
    config: Config = get_config_from_paths(
        [
            os.path.join(FIXTURES_DIR, "configs", "dummy_config.yml"),
        ]
    )
    assert config.seed == 42
