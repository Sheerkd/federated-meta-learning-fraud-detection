"""Local and centralized reference experiments over the same saved partitions."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fraud_detection.data.pipeline import BANKS, FEATURES, SETTINGS, file_hash
from fraud_detection.evaluation.metrics import (
    choose_f1_threshold, classification_metrics, macro_average,
)
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec
from fraud_detection.training.core import TrainSpec, fit_model, read_split, score_model


def save_model(model: FraudMLP, path: Path, *, spec: ModelSpec,
               verification_x: np.ndarray, device: str) -> str:
    """Save weights atomically and prove a fresh model reproduces predictions."""
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, path)
    reloaded = FraudMLP(spec).to(device)
    reloaded.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    original = score_model(model, verification_x, device=device)
    restored = score_model(reloaded, verification_x, device=device)
    np.testing.assert_allclose(original, restored, rtol=0, atol=0)
    return file_hash(path)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _metric_record(method: str, bank: str, metrics: dict) -> dict:
    return {"method": method, "client_id": bank, **{key: value for key, value in metrics.items()
            if key not in ("confusion", "undefined_reason")}, **metrics["confusion"]}


def train_setting(project_root: Path, *, setting: str, seed: int,
                  spec: TrainSpec, clients_dir: str = "clients",
                  data_dir: str = "data", runs_dir: str = "runs",
                  device: str = "cpu") -> Path:
    if setting not in SETTINGS:
        raise ValueError(f"Unknown setting: {setting}")
    partition_path = project_root / data_dir / "processed" / setting / "metadata.json"
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    manifest_path = partition_path.parent / "manifest.parquet"
    if file_hash(manifest_path) != partition["manifest_sha256"]:
        raise ValueError(f"Partition manifest checksum changed for {setting}")
    run_dir = project_root / runs_dir / "baselines" / setting / f"seed_{seed}"
    if run_dir.exists():
        raise FileExistsError(f"Baseline run already exists: {run_dir}")
    for bank in BANKS:
        local_path = project_root / clients_dir / bank / "models" / setting / f"local_seed_{seed}.pt"
        if local_path.exists():
            raise FileExistsError(f"Local model already exists: {local_path}")
    run_dir.mkdir(parents=True)
    model_spec = spec.model_spec()
    local_results = {}
    central_results = {}
    records = []
    for bank in BANKS:
        train_x, train_y = read_split(project_root, clients_dir, bank, setting, "train")
        valid_x, valid_y = read_split(project_root, clients_dir, bank, setting, "validation")
        model, training = fit_model(train_x, train_y, valid_x, valid_y, spec, seed=seed, device=device)
        valid_scores = score_model(model, valid_x, device=device)
        threshold, threshold_selection = choose_f1_threshold(valid_y, valid_scores)
        valid_metrics = classification_metrics(valid_y, valid_scores, threshold)
        checkpoint = project_root / clients_dir / bank / "models" / setting / f"local_seed_{seed}.pt"
        checkpoint_sha = save_model(model, checkpoint, spec=model_spec,
                                    verification_x=valid_x[:128], device=device)
        # Test data is loaded only after training, checkpoint selection and threshold selection.
        test_x, test_y = read_split(project_root, clients_dir, bank, setting, "test")
        test_scores = score_model(model, test_x, device=device)
        test_metrics = classification_metrics(test_y, test_scores, threshold)
        scaler = project_root / clients_dir / bank / "preprocessing" / f"{setting}_scaler.json"
        bundle = {
            "schema_version": 1, "method": "local", "client_id": bank,
            "setting": setting, "seed": seed, "model_spec": model_spec.as_dict(),
            "feature_order": FEATURES, "checkpoint": str(checkpoint.relative_to(project_root)),
            "checkpoint_sha256": checkpoint_sha,
            "scaler": str(scaler.relative_to(project_root)),
            "scaler_sha256": file_hash(scaler),
            "threshold": threshold, "threshold_selection": threshold_selection,
            "partition_manifest_sha256": partition["manifest_sha256"],
            "dataset_sha256": partition["dataset"]["sha256"],
        }
        _write_json(checkpoint.with_suffix(".json"), bundle)
        local_results[bank] = {"bundle": str(checkpoint.with_suffix(".json").relative_to(project_root)),
                               "training": training, "validation": valid_metrics, "test": test_metrics}
        records.append(_metric_record("local", bank, test_metrics))
        del train_x, train_y, valid_x, valid_y, test_x, test_y, model

    train_parts = [read_split(project_root, clients_dir, bank, setting, "train") for bank in BANKS]
    valid_parts = [read_split(project_root, clients_dir, bank, setting, "validation") for bank in BANKS]
    train_x = np.concatenate([part[0] for part in train_parts])
    train_y = np.concatenate([part[1] for part in train_parts])
    valid_x = np.concatenate([part[0] for part in valid_parts])
    valid_y = np.concatenate([part[1] for part in valid_parts])
    del train_parts, valid_parts
    model, central_training = fit_model(train_x, train_y, valid_x, valid_y,
                                        spec, seed=seed, device=device)
    central_checkpoint = run_dir / "centralized.pt"
    central_sha = save_model(model, central_checkpoint, spec=model_spec,
                             verification_x=valid_x[:128], device=device)
    del train_x, train_y, valid_x, valid_y
    for bank in BANKS:
        bank_valid_x, bank_valid_y = read_split(project_root, clients_dir, bank, setting, "validation")
        bank_valid_scores = score_model(model, bank_valid_x, device=device)
        threshold, threshold_selection = choose_f1_threshold(bank_valid_y, bank_valid_scores)
        valid_metrics = classification_metrics(bank_valid_y, bank_valid_scores, threshold)
        bank_test_x, bank_test_y = read_split(project_root, clients_dir, bank, setting, "test")
        test_metrics = classification_metrics(bank_test_y,
                                              score_model(model, bank_test_x, device=device), threshold)
        central_results[bank] = {"threshold": threshold,
                                 "threshold_selection": threshold_selection,
                                 "validation": valid_metrics, "test": test_metrics}
        records.append(_metric_record("centralized", bank, test_metrics))
    central_bundle = {
        "schema_version": 1, "method": "centralized", "setting": setting,
        "seed": seed, "model_spec": model_spec.as_dict(), "feature_order": FEATURES,
        "checkpoint": str(central_checkpoint.relative_to(project_root)),
        "checkpoint_sha256": central_sha,
        "client_thresholds": {bank: central_results[bank]["threshold"] for bank in BANKS},
        "client_scalers": {bank: str((project_root / clients_dir / bank / "preprocessing" /
                                     f"{setting}_scaler.json").relative_to(project_root)) for bank in BANKS},
        "partition_manifest_sha256": partition["manifest_sha256"],
        "dataset_sha256": partition["dataset"]["sha256"],
    }
    _write_json(run_dir / "centralized.json", central_bundle)
    result = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed", "setting": setting, "seed": seed,
        "device": device, "train_spec": spec.as_dict(), "model_spec": model_spec.as_dict(),
        "partition_manifest_sha256": partition["manifest_sha256"],
        "dataset_sha256": partition["dataset"]["sha256"],
        "selection": "Best epoch by validation average precision; per-bank threshold by validation F1",
        "loss": "BCEWithLogitsLoss with training-only negative/positive class weight",
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "numpy", "pandas", "scikit-learn")},
        "local": local_results,
        "centralized": {"bundle": str((run_dir / "centralized.json").relative_to(project_root)),
                        "training": central_training, "clients": central_results},
        "macro_test": {
            "local": macro_average({bank: value["test"] for bank, value in local_results.items()}),
            "centralized": macro_average({bank: value["test"] for bank, value in central_results.items()}),
        },
    }
    _write_json(run_dir / "results.json", result)
    pd.DataFrame(records).to_csv(run_dir / "test_metrics.csv", index=False)
    return run_dir / "results.json"


def train_baselines(project_root: Path, *, settings: list[str], seed: int,
                    config_path: Path, clients_dir: str = "clients",
                    data_dir: str = "data", runs_dir: str = "runs",
                    device: str = "cpu") -> list[Path]:
    spec = TrainSpec.from_toml(config_path)
    return [train_setting(project_root, setting=setting, seed=seed, spec=spec,
                          clients_dir=clients_dir, data_dir=data_dir,
                          runs_dir=runs_dir, device=device) for setting in settings]
