import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.dashboard import service
from fraud_detection.data.pipeline import FEATURES
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec


def _csv(frame):
    return frame.to_csv(index=False).encode("utf-8")


def test_upload_schema_and_bank_scaling(tmp_path, monkeypatch):
    frame = pd.DataFrame(np.zeros((2, len(FEATURES))), columns=FEATURES)
    frame.loc[0, "Time"] = 12
    frame.loc[0, "Amount"] = 30
    frame.loc[1, "Time"] = 10
    frame.loc[1, "Amount"] = 20
    assert len(service.parse_upload(_csv(frame))) == 2
    with pytest.raises(ValueError, match="columns"):
        service.parse_upload(_csv(frame.drop(columns=["V1"])))
    bad = frame.copy()
    bad.loc[0, "V1"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        service.parse_upload(_csv(bad))
    bad = frame.copy()
    bad.loc[0, "Amount"] = -1
    with pytest.raises(ValueError, match="negative"):
        service.parse_upload(_csv(bad))
    assert len(service.parse_upload(_csv(frame))) == 2

    preprocessing = tmp_path / "clients/bank_a/preprocessing"
    preprocessing.mkdir(parents=True)
    (preprocessing / "mild_scaler.json").write_text(json.dumps({
        "client_id": "bank_a", "setting": "mild", "features": ["Time", "Amount"],
        "mean": [10, 20], "scale": [2, 5]}))
    model = FraudMLP(ModelSpec())
    model.eval()
    captured = {}

    def fake_score(_model, values):
        captured["values"] = values.copy()
        return np.array([0.8, 0.2])

    monkeypatch.setattr(service, "selected_model", lambda *a, **kw: (model, 0.5, "hash"))
    monkeypatch.setattr(service, "score_model", fake_score)
    scored = service.predict_upload(tmp_path, bank="bank_a", setting="mild", seed=42,
                                    method="FedAvg", file_bytes=_csv(frame))
    assert captured["values"][0, FEATURES.index("Time")] == 1
    assert captured["values"][0, FEATURES.index("Amount")] == 2
    assert list(scored["Flagged"]) == [True, False]


def test_background_job_refuses_existing_run_and_tracks_failure(tmp_path, monkeypatch):
    existing = tmp_path / "runs/federated/mild/seed_42"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        service.launch_run(tmp_path, method="FedAvg", setting="mild", seed=42)

    called = {}

    def fake_popen(command, **kwargs):
        called["command"] = command
        return object()

    monkeypatch.setattr(service.subprocess, "Popen", fake_popen)
    path = service.launch_run(tmp_path, method="FedMAML", setting="mild", seed=43)
    assert "fraud_detection.dashboard.worker" in called["command"]
    assert (path / "job.json").is_file()
    job = json.loads((path / "job.json").read_text())
    job.update({"status": "failed", "error": "synthetic failure"})
    (path / "job.json").write_text(json.dumps(job))
    rows = service.job_rows(tmp_path)
    assert rows.iloc[0]["Status"] == "failed"
    assert rows.iloc[0]["Details"] == "synthetic failure"


def test_real_dashboard_prediction_and_explanation_buttons_if_artifacts_exist():
    root = Path(__file__).resolve().parents[1]
    if not (root / "runs/meta/mild/seed_42/results.json").is_file():
        pytest.skip("Prepared research artifacts are unavailable")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(root / "src/fraud_detection/dashboard/app.py")).run(timeout=30)
    assert not app.exception
    next(button for button in app.button if button.label == "Score selected example").click()
    app.run(timeout=30)
    assert not app.exception
    assert any(metric.label == "Fraud score" for metric in app.metric), (
        [item.value for item in app.warning], [item.value for item in app.error],
        [(item.label, item.value) for item in app.selectbox])
    next(button for button in app.button if button.label == "Create or load explanation").click()
    app.run(timeout=30)
    assert not app.exception
    assert any(metric.label == "Explained fraud score" for metric in app.metric)


def test_comparison_uses_saved_test_results_and_missing_seed_is_safe():
    root = Path(__file__).resolve().parents[1]
    result_path = root / "runs/federated/mild/seed_42/results.json"
    if not result_path.is_file():
        pytest.skip("Prepared research artifacts are unavailable")
    expected = json.loads(result_path.read_text())["macro_test"]["f1"]
    rows = service.comparison_rows(root, setting="mild", seed=42)
    actual = rows.loc[rows["Method"] == "FedAvg", "F1"].mean()
    assert actual == pytest.approx(expected)
    assert service.available_methods(root, setting="mild", seed=99999) == []
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(root / "src/fraud_detection/dashboard/app.py")).run(timeout=30)
    saved_seed = next(item for item in app.number_input if item.label == "Saved run seed")
    saved_seed.set_value(99999)
    app.run(timeout=30)
    assert not app.exception
    assert any("No completed model" in item.value for item in app.warning)
