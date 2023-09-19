import os
import logging
from enum import Enum
from typing import List


logger = logging.getLogger(__name__)


class System(Enum):
    DEFAULT = "default"
    PEGASUS = "pegasus"
    TEST_RUNNER = "test_runner"


def get_current_system(allow_default: bool = True) -> System:
    if os.path.exists("/netscratch/"):
        return System.PEGASUS
    elif os.path.exists("/home/runner/"):
        return System.TEST_RUNNER
    elif allow_default:
        return System.DEFAULT
    else:
        raise ValueError("Cannot determine system")


def get_path_by_system(possible_paths: List[str], allow_default: bool = True, default_path: str = "/dev/null") -> str:
    current_system = get_current_system(allow_default)

    for raw_path in possible_paths:
        system_name, path = raw_path.split(":", 2)

        if system_name == current_system.value:
            return path

    if allow_default:
        logger.warning("Using default path because current system cannot be matched!")

        return default_path
    else:
        raise ValueError(f"Cannot assign possible paths to current system: {possible_paths} {current_system=}")
