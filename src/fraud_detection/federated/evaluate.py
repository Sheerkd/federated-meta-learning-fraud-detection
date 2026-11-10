"""Offline public-benchmark evaluation after the federated ServerApp finishes."""

import json
import os
from pathlib import Path

import pandas as pd
import torch

from fraud_detection.data.pipeline import BANKS, file_hash
from fraud_detection.evaluation.metrics import choose_f1_threshold, classification_metrics, macro_average
from fraud_detection.federated.protocol import model_bytes, state_fingerprint, validate_state
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec
from fraud_detection.training.core import read_split, score_model


def evaluate_federated(project_root: Path, *, setting: str, seed: int,
                       clients_dir: str = "clients", data_dir: str = "data",
                       runs_dir: str = "runs") -> Path:
    run_dir = project_root / runs_dir / "federated" / setting / f"seed_{seed}"
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run["status"] not in ("federated_complete", "completed"):
        raise ValueError(f"Training has not completed: {run['status']}")
    partition = json.loads((project_root / data_dir / "processed" / setting / "metadata.json").read_text(encoding="utf-8"))
    if run["partition_manifest_sha256"] != partition["manifest_sha256"]:
        raise ValueError("Source partition changed")
    rounds = int(run["rounds_completed"])
    reports = []
    for number in range(1, rounds + 1):
        report = json.loads((run_dir / f"round_{number}.json").read_text(encoding="utf-8"))
        if (report["status"] != "completed" or report["round"] != number
                or set(report["clients"]) != set(BANKS) or set(report["validation"]) != set(BANKS)
                or not report["aggregation_verified"]
                or file_hash(run_dir / report["checkpoint"]) != report["checkpoint_sha256"]):
            raise ValueError(f"Incomplete or altered federated round {number}")
        reports.append(report)
    final = reports[-1]
    if final["aggregate_sha256"] != run["final_model_sha256"]:
        raise ValueError("Final model hash mismatch")
    state = torch.load(run_dir / final["checkpoint"], map_location="cpu", weights_only=True)
    spec = ModelSpec(**run["model_spec"])
    validate_state(state, spec)
    if state_fingerprint(state) != final["aggregate_sha256"]:
        raise ValueError("Final checkpoint tensors changed")
    model = FraudMLP(spec)
    model.load_state_dict(state)
    model.eval()
    by_bank = {}
    flat = []
    for bank in BANKS:
        valid_x, valid_y = read_split(project_root, clients_dir, bank, setting, "validation")
        valid_scores = score_model(model, valid_x)
        threshold, selection = choose_f1_threshold(valid_y, valid_scores)
        test_x, test_y = read_split(project_root, clients_dir, bank, setting, "test")
        test_metrics = classification_metrics(test_y, score_model(model, test_x), threshold)
        by_bank[bank] = {"threshold_selection": selection,
                         "validation": classification_metrics(valid_y, valid_scores, threshold),
                         "test": test_metrics}
        flat.append({"client_id": bank, **{k: v for k, v in test_metrics.items()
                                           if k not in ("confusion", "undefined_reason")},
                     **test_metrics["confusion"]})
    tensor_bytes = model_bytes(state)
    result = {
        "schema_version": 1, "status": "completed", "method": "fedavg",
        "setting": setting, "seed": seed, "rounds": rounds,
        "model_spec": spec.as_dict(),
        "dataset_sha256": run["dataset_sha256"],
        "partition_manifest_sha256": run["partition_manifest_sha256"],
        "final_checkpoint": final["checkpoint"],
        "final_checkpoint_sha256": final["checkpoint_sha256"],
        "final_model_sha256": final["aggregate_sha256"],
        "model_tensor_bytes": tensor_bytes,
        "approximate_communication": {
            "train_broadcast_tensor_bytes": sum(r["train_broadcast_tensor_bytes"] for r in reports),
            "train_update_tensor_bytes": sum(r["train_update_tensor_bytes"] for r in reports),
            "validation_broadcast_tensor_bytes": sum(r["validation_broadcast_tensor_bytes"] for r in reports),
            "scope": "model tensor payload only; excludes protocol, authentication, and metadata overhead",
        },
        "evaluation_scope": "Offline public-benchmark evaluator on the simulation host; not the Flower server",
        "threshold_rule": "per-client validation F1, then recall, then lower threshold",
        "clients": by_bank,
        "macro_test": macro_average({bank: item["test"] for bank, item in by_bank.items()}),
    }
    result_path = run_dir / "results.json"
    if result_path.exists():
        raise FileExistsError(f"Federated evaluation already exists: {result_path}")
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, result_path)
    pd.DataFrame(flat).to_csv(run_dir / "test_metrics.csv", index=False)
    run["status"] = "completed"
    temp_run = run_path.with_suffix(".json.tmp")
    temp_run.write_text(json.dumps(run, indent=2), encoding="utf-8")
    os.replace(temp_run, run_path)
    return result_path
