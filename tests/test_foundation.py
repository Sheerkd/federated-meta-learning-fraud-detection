from pathlib import Path

import pytest
from pydantic import ValidationError

from fraud_detection.config import AppConfig, Paths, initialize_storage, load_config


@pytest.mark.parametrize("path", ["../outside", "C:/outside", "", "."])
def test_storage_rejects_unsafe_paths(path):
    with pytest.raises(ValidationError):
        Paths(data=path)


def test_rejects_misspelled_config_and_duplicate_clients():
    with pytest.raises(ValidationError):
        AppConfig(sead=42)
    with pytest.raises(ValidationError):
        AppConfig(client_ids=["bank_a"] * 4)


def test_storage_is_local_and_idempotent(tmp_path):
    config = AppConfig()
    first = initialize_storage(tmp_path, config)
    assert first == initialize_storage(tmp_path, config)
    assert len(first) == 18
    assert all(path.is_dir() and path.is_relative_to(tmp_path) for path in first)


def test_default_config():
    root = Path(__file__).resolve().parents[1]
    assert load_config(root / "configs/default.toml").device == "cpu"


def test_dashboard_renders_saved_work():
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "src/fraud_detection/dashboard/app.py")).run(timeout=30)
    assert not app.exception
    assert any("Scores are research outputs" in item.value for item in app.info)
