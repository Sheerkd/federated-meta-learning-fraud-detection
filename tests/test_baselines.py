import json

import numpy as np
import pandas as pd
import pytest
import torch

from fraud_detection.data.pipeline import FEATURES, prepare_ulb
from fraud_detection.evaluation.metrics import choose_f1_threshold, classification_metrics
from fraud_detection.models.fraud_model import FraudMLP
from fraud_detection.training.baselines import train_setting
from fraud_detection.training.core import TrainSpec, fit_model, score_model
from fraud_detection.training.verify import verify_baseline_setting


def test_threshold_and_metrics_on_rare_events():
    labels = np.array([0, 0, 0, 1, 1], dtype=np.int8)
    scores = np.array([0.05, 0.1, 0.8, 0.7, 0.9])
    threshold, selection = choose_f1_threshold(labels, scores)
    assert threshold == pytest.approx(0.7)
    assert selection["validation_f1"] == pytest.approx(0.8)
    metrics = classification_metrics(labels, scores, threshold)
    assert metrics["confusion"] == {"tn": 2, "fp": 1, "fn": 0, "tp": 2}
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)
    assert metrics["average_precision"] is not None
    one_class = classification_metrics(np.zeros(3), np.array([0.1, 0.2, 0.3]), 0.5)
    assert one_class["roc_auc"] is None and one_class["undefined_reason"]


def test_fitting_selects_checkpoint_from_validation_and_predicts_after_reload(tmp_path):
    rng = np.random.default_rng(4)
    train_x = rng.normal(size=(96, 30)).astype(np.float32)
    train_y = np.array([0] * 72 + [1] * 24, dtype=np.float32)
    valid_x = rng.normal(size=(32, 30)).astype(np.float32)
    valid_y = np.array([0] * 24 + [1] * 8, dtype=np.float32)
    spec = TrainSpec(epochs=2, batch_size=24)
    model, report = fit_model(train_x, train_y, valid_x, valid_y, spec, seed=5)
    assert isinstance(model, FraudMLP)
    assert report["positive_class_weight"] == 3.0
    assert len(report["history"]) == 2
    assert 1 <= report["best_epoch"] <= 2
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = FraudMLP(spec.model_spec())
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    np.testing.assert_array_equal(score_model(model, valid_x), score_model(restored, valid_x))


def test_complete_small_baseline_run(tmp_path):
    rng = np.random.default_rng(11)
    count = 2_160
    frame = pd.DataFrame({name: rng.normal(size=count) for name in FEATURES})
    frame["Time"] = rng.uniform(0, 10_000, size=count)
    frame["Amount"] = rng.uniform(0, 500, size=count)
    labels = np.array([0] * 2_000 + [1] * 160)
    rng.shuffle(labels)
    frame["Class"] = labels
    source = tmp_path / "creditcard.csv"
    frame.to_csv(source, index=False)
    prepare_ulb(source, tmp_path, seed=17, require_reference=False, min_legitimate_groups=50)
    path = train_setting(tmp_path, setting="mild", seed=17,
                         spec=TrainSpec(epochs=1, batch_size=256))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert len(report["local"]) == 4
    assert len(report["centralized"]["clients"]) == 4
    assert all(row["test"]["fraud"] > 0 for row in report["local"].values())
    assert all((tmp_path / row["bundle"]).is_file() for row in report["local"].values())
    assert verify_baseline_setting(tmp_path, setting="mild", seed=17)["status"] == "passed"
    with pytest.raises(FileExistsError):
        train_setting(tmp_path, setting="mild", seed=17,
                      spec=TrainSpec(epochs=1, batch_size=256))
