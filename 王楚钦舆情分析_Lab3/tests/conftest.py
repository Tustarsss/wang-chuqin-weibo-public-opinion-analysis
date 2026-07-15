from __future__ import annotations

import sys
from pathlib import Path

import pytest


LAB3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB3_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return LAB3_ROOT.parent


@pytest.fixture(scope="session")
def project_data(repo_root: Path):
    from lab3.data_loader import load_project_data

    return load_project_data(repo_root)
