"""Verify every saved FedAvg round and recompute final public-benchmark metrics."""

import json
from pathlib import Path

import numpy as np
import torch

from fraud_detection.data.pipeline import BANKS, FEATURES, file_hash
from fraud_detection.evaluation.metrics import choose_f1_threshold, classification_metrics, macro_average
from fraud_detection.federated.protocol import model_bytes, state_fingerprint, validate_state
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec
from fraud_detection.training.core import read_split, score_model
from fraud_detection.training.verify import _check_metrics


def verify_fedavg(project_root: Path, *, setting: str, seed: int,
                  clients_dir: str = "clients", data_dir: str = "data",
                  runs_dir: str = "runs") -> dict:
    run_dir = project_root / runs_dir / "federated" / setting / f"seed_{seed}"
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    partition = json.loads((project_root / data_dir / "processed" / setting / "metadata.json").read_text(encoding="utf-8"))
    if (run["status"] != "completed" or result["status"] != "completed"
            or run["setting"] != setting or result["setting"] != setting
            or run["seed"] != seed or result["seed"] != seed
            or run["partition_manifest_sha256"] != partition["manifest_sha256"]
            or result["partition_manifest_sha256"] != partition["manifest_sha256"]
            or file_hash(project_root / data_dir / "processed" / setting / "manifest.parquet") != partition["manifest_sha256"]):
        raise ValueError("Federated run, result, or partition metadata mismatch")
    spec = ModelSpec(**run["model_spec"])
    first_path = run_dir / "round_0.pt"
    if file_hash(first_path) != run["initial_checkpoint_sha256"]:
        raise ValueError("Initial model checksum mismatch")
    previous = torch.load(first_path, map_location="cpu", weights_only=True)
    validate_state(previous, spec)
    previous_hash = state_fingerprint(previous)
    if previous_hash != run["initial_model_sha256"]:
        raise ValueError("Initial model tensor hash mismatch")
    reports = []
    for number in range(1, run["rounds_completed"] + 1):
        report = json.loads((run_dir / f"round_{number}.json").read_text(encoding="utf-8"))
        if (report["round"] != number or report["status"] != "completed"
                or report["base_sha256"] != previous_hash
                or not report["aggregation_verified"]
                or set(report["clients"]) != set(BANKS)
                or set(report["validation"]) != set(BANKS)):
            raise ValueError(f"Round {number} failed participation or lineage checks")
        state_path = run_dir / report["checkpoint"]
        if file_hash(state_path) != report["checkpoint_sha256"]:
            raise ValueError(f"Round {number} checkpoint checksum mismatch")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        validate_state(state, spec)
        actual_hash = state_fingerprint(state)
        if actual_hash != report["aggregate_sha256"] or actual_hash == previous_hash:
            raise ValueError(f"Round {number} tensor hash or update mismatch")
        for bank in BANKS:
            counts = partition["partition"]["counts"][bank]
            if (report["clients"][bank]["train_rows"] != counts["train"]["rows"]
                    or report["validation"][bank]["validation_rows"] != counts["validation"]["rows"]):
                raise ValueError(f"Round {number} client count mismatch: {bank}")
        tensor_bytes = model_bytes(state)
        if (report["train_weight_sum"] != sum(partition["partition"]["counts"][bank]["train"]["rows"] for bank in BANKS)
                or report["train_broadcast_tensor_bytes"] != 4 * tensor_bytes
                or report["train_update_tensor_bytes"] != 4 * tensor_bytes
                or report["validation_broadcast_tensor_bytes"] != 4 * tensor_bytes):
            raise ValueError(f"Round {number} communication accounting mismatch")
        previous_hash = actual_hash
        previous = state
        reports.append(report)
    if (len(reports) != run["rounds_requested"] or result["rounds"] != len(reports)
            or result["final_model_sha256"] != previous_hash
            or run["final_model_sha256"] != previous_hash
            or result["final_checkpoint_sha256"] != reports[-1]["checkpoint_sha256"]
            or result["model_spec"] != spec.as_dict()):
        raise ValueError("Final federated result does not match completed rounds")
    model = FraudMLP(spec)
    model.load_state_dict(previous)
    model.eval()
    actual_metrics = {}
    for bank in BANKS:
        valid_x, valid_y = read_split(project_root, clients_dir, bank, setting, "validation")
        valid_scores = score_model(model, valid_x)
        threshold, _ = choose_f1_threshold(valid_y, valid_scores)
        stored = result["clients"][bank]
        if abs(threshold - stored["test"]["threshold"]) > 1e-12:
            raise ValueError(f"Final threshold mismatch: {bank}")
        _check_metrics(classification_metrics(valid_y, valid_scores, threshold), stored["validation"])
        test_x, test_y = read_split(project_root, clients_dir, bank, setting, "test")
        actual = classification_metrics(test_y, score_model(model, test_x), threshold)
        _check_metrics(actual, stored["test"])
        actual_metrics[bank] = actual
    macro = macro_average(actual_metrics)
    for key, value in macro.items():
        np.testing.assert_allclose(value, result["macro_test"][key], rtol=0, atol=1e-12)
    return {"status": "passed", "setting": setting, "seed": seed,
            "rounds": len(reports), "clients_per_round": 4,
            "checkpoints_verified": len(reports) + 1,
            "test_metric_sets_recomputed": 4,
            "model_tensor_bytes": result["model_tensor_bytes"],
            "macro_test": macro}
