import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
from generate_test_clip import generate_test_clip


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("clips") / "test_clip.mp4"
    return generate_test_clip(str(out))
