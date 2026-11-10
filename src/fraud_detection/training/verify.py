"""Recompute baseline predictions and metrics from saved model bundles."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from fraud_detection.data.pipeline import BANKS, FEATURES, SETTINGS, file_hash
from fraud_detection.evaluation.metrics import choose_f1_threshold, classification_metrics, macro_average
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec
from fraud_detection.training.core import read_split, score_model


def _load_model(project_root: Path, bundle: dict) -> FraudMLP:
    checkpoint = project_root / bundle["checkpoint"]
    if file_hash(checkpoint) != bundle["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {checkpoint}")
    model = FraudMLP(ModelSpec(**bundle["model_spec"]))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _check_metrics(actual: dict, saved: dict) -> None:
    for field in ("rows", "fraud", "confusion", "undefined_reason"):
        if actual[field] != saved[field]:
            raise ValueError(f"Saved {field} differs from recomputed result")
    for field in ("fraud_prevalence", "threshold", "precision", "recall", "f1",
                  "false_positive_rate", "roc_auc", "average_precision"):
        if actual[field] is None or saved[field] is None:
            if actual[field] != saved[field]:
                raise ValueError(f"Saved {field} differs from recomputed result")
        else:
            np.testing.assert_allclose(actual[field], saved[field], rtol=0, atol=1e-12)


def verify_baseline_setting(project_root: Path, *, setting: str, seed: int,
                            clients_dir: str = "clients", data_dir: str = "data",
                            runs_dir: str = "runs") -> dict:
    if setting not in SETTINGS:
        raise ValueError(f"Unknown setting: {setting}")
    run_dir = project_root / runs_dir / "baselines" / setting / f"seed_{seed}"
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if result["status"] != "completed" or result["setting"] != setting or result["seed"] != seed:
        raise ValueError("Run status, setting, or seed is inconsistent")
    partition = json.loads((project_root / data_dir / "processed" / setting / "metadata.json").read_text(encoding="utf-8"))
    if (result["partition_manifest_sha256"] != partition["manifest_sha256"]
            or result["dataset_sha256"] != partition["dataset"]["sha256"]):
        raise ValueError("Run and partition source mismatch")
    local_metrics = {}
    for bank in BANKS:
        row = result["local"][bank]
        bundle = json.loads((project_root / row["bundle"]).read_text(encoding="utf-8"))
        if (bundle["client_id"] != bank or bundle["setting"] != setting
                or bundle["partition_manifest_sha256"] != partition["manifest_sha256"]
                or bundle["feature_order"] != FEATURES
                or file_hash(project_root / bundle["scaler"]) != bundle["scaler_sha256"]):
            raise ValueError(f"Local bundle mismatch: {bank}")
        model = _load_model(project_root, bundle)
        valid_x, valid_y = read_split(project_root, clients_dir, bank, setting, "validation")
        valid_scores = score_model(model, valid_x)
        threshold, selection = choose_f1_threshold(valid_y, valid_scores)
        if abs(threshold - bundle["threshold"]) > 1e-12:
            raise ValueError(f"Threshold mismatch: {bank}")
        _check_metrics(classification_metrics(valid_y, valid_scores, threshold), row["validation"])
        np.testing.assert_allclose(average_precision_score(valid_y, valid_scores),
                                   row["training"]["best_validation_average_precision"],
                                   rtol=0, atol=1e-12)
        best_record = row["training"]["history"][row["training"]["best_epoch"] - 1]
        np.testing.assert_allclose(best_record["validation_average_precision"],
                                   row["training"]["best_validation_average_precision"],
                                   rtol=0, atol=1e-12)
        if selection["rule"] != bundle["threshold_selection"]["rule"]:
            raise ValueError(f"Threshold rule mismatch: {bank}")
        test_x, test_y = read_split(project_root, clients_dir, bank, setting, "test")
        _check_metrics(classification_metrics(test_y, score_model(model, test_x), threshold), row["test"])
        local_metrics[bank] = row["test"]
    central_bundle = json.loads((project_root / result["centralized"]["bundle"]).read_text(encoding="utf-8"))
    if (central_bundle["partition_manifest_sha256"] != partition["manifest_sha256"]
            or central_bundle["feature_order"] != FEATURES):
        raise ValueError("Central bundle source mismatch")
    central_model = _load_model(project_root, central_bundle)
    central_metrics = {}
    pooled_valid_labels = []
    pooled_valid_scores = []
    for bank in BANKS:
        row = result["centralized"]["clients"][bank]
        expected_scaler = (project_root / clients_dir / bank / "preprocessing" /
                           f"{setting}_scaler.json").relative_to(project_root)
        if central_bundle["client_scalers"][bank] != str(expected_scaler) or not (project_root / expected_scaler).is_file():
            raise ValueError(f"Central scaler reference mismatch: {bank}")
        valid_x, valid_y = read_split(project_root, clients_dir, bank, setting, "validation")
        valid_scores = score_model(central_model, valid_x)
        threshold, _ = choose_f1_threshold(valid_y, valid_scores)
        if abs(threshold - row["threshold"]) > 1e-12 or abs(threshold - central_bundle["client_thresholds"][bank]) > 1e-12:
            raise ValueError(f"Central threshold mismatch: {bank}")
        _check_metrics(classification_metrics(valid_y, valid_scores, threshold), row["validation"])
        pooled_valid_labels.append(valid_y)
        pooled_valid_scores.append(valid_scores)
        test_x, test_y = read_split(project_root, clients_dir, bank, setting, "test")
        _check_metrics(classification_metrics(test_y, score_model(central_model, test_x), threshold), row["test"])
        central_metrics[bank] = row["test"]
    np.testing.assert_allclose(average_precision_score(np.concatenate(pooled_valid_labels),
                                                        np.concatenate(pooled_valid_scores)),
                               result["centralized"]["training"]["best_validation_average_precision"],
                               rtol=0, atol=1e-12)
    for method, metrics in (("local", local_metrics), ("centralized", central_metrics)):
        actual = macro_average(metrics)
        for field, value in actual.items():
            np.testing.assert_allclose(value, result["macro_test"][method][field], rtol=0, atol=1e-12)
    return {"status": "passed", "setting": setting, "seed": seed,
            "clients": 4, "models_reloaded": 5,
            "thresholds_recomputed": 8, "test_metric_sets_recomputed": 8,
            "macro_test": result["macro_test"]}
