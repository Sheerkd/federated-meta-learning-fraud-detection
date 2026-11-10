"""Offline public-benchmark personalization assessment after Flower training."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from fraud_detection.data.pipeline import BANKS, file_hash
from fraud_detection.evaluation.metrics import choose_f1_threshold, classification_metrics, macro_average
from fraud_detection.federated.protocol import state_fingerprint, validate_state
from fraud_detection.meta.learning import adapt
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec
from fraud_detection.training.core import read_split, score_model


def _support_indices(y: np.ndarray, budget: int, seed: int) -> np.ndarray:
    if budget > len(y):
        raise ValueError("Support budget exceeds training rows")
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    n_positive = min(len(positive), budget // 2)
    if n_positive == 0 or len(negative) < budget - n_positive:
        raise ValueError("Insufficient support examples")
    selected = np.concatenate((rng.choice(positive, n_positive, replace=False),
                               rng.choice(negative, budget - n_positive, replace=False)))
    rng.shuffle(selected)
    return selected


def evaluate_meta(project_root: Path, *, setting: str, seed: int,
                  clients_dir: str = "clients", data_dir: str = "data",
                  runs_dir: str = "runs") -> Path:
    run_dir = project_root / runs_dir / "meta" / setting / f"seed_{seed}"
    status_path = run_dir / "run.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] != "federated_complete":
        raise ValueError("Meta training did not complete")
    partition = json.loads((project_root / data_dir / "processed" / setting / "metadata.json").read_text(encoding="utf-8"))
    if partition["manifest_sha256"] != status["partition_manifest_sha256"]:
        raise ValueError("Partition changed")
    rounds = int(status["rounds_completed"])
    previous = status["initial_model_sha256"]
    for number in range(1, rounds + 1):
        report = json.loads((run_dir / f"round_{number}.json").read_text(encoding="utf-8"))
        if (report["status"] != "completed" or report["round"] != number
                or report["base_sha256"] != previous or set(report["clients"]) != set(BANKS)
                or set(report["validation"]) != set(BANKS)
                or file_hash(run_dir / report["checkpoint"]) != report["checkpoint_sha256"]):
            raise ValueError(f"Meta round {number} incomplete or altered")
        previous = report["aggregate_sha256"]
    if previous != status["final_model_sha256"]:
        raise ValueError("Final meta hash mismatch")
    state = torch.load(run_dir / status["final_checkpoint"], map_location="cpu", weights_only=True)
    spec = ModelSpec(**status["model_spec"])
    validate_state(state, spec)
    if state_fingerprint(state) != previous:
        raise ValueError("Final meta tensors changed")
    by_bank = {}
    for index, bank in enumerate(BANKS):
        train_x, train_y = read_split(project_root, clients_dir, bank, setting, "train")
        valid_x, valid_y = read_split(project_root, clients_dir, bank, setting, "validation")
        test_x, test_y = read_split(project_root, clients_dir, bank, setting, "test")
        model = FraudMLP(spec)
        model.load_state_dict(state)
        before_threshold, _ = choose_f1_threshold(valid_y, score_model(model, valid_x))
        before = classification_metrics(test_y, score_model(model, test_x), before_threshold)
        adaptations = []
        for budget in (32, 128, 512):
            support = _support_indices(train_y, budget, seed + 10000 * index + budget)
            for steps in (1, 3, 5):
                personalized = FraudMLP(spec)
                personalized.load_state_dict(state)
                adapt(personalized, train_x[support], train_y[support], steps=steps,
                      learning_rate=float(status["run_config"]["inner-lr"]),
                      seed=seed + 10000 * index + budget + steps)
                threshold, selection = choose_f1_threshold(valid_y, score_model(personalized, valid_x))
                metrics = classification_metrics(test_y, score_model(personalized, test_x), threshold)
                adaptations.append({"support_budget": budget, "support_fraud": int(train_y[support].sum()),
                                    "steps": steps, "threshold_selection": selection,
                                    "model_sha256": state_fingerprint(personalized.state_dict()),
                                    "test": metrics})
        by_bank[bank] = {"before": before, "adaptations": adaptations}
    result = {"schema_version": 1, "status": "completed", "method": "fedmaml",
              "setting": setting, "seed": seed, "rounds": rounds,
              "adaptation_torch_threads": torch.get_num_threads(),
              "final_model_sha256": previous,
              "partition_manifest_sha256": status["partition_manifest_sha256"],
              "episode": {"support_size": status["run_config"]["support-size"],
                          "query_size": status["run_config"]["query-size"],
                          "class_sampling": "balanced support and query, disjoint, sampled without replacement"},
              "personalization": {"support_sampling": "enriched positive class; up to half budget, limited by available fraud rows",
                                  "threshold_rule": "validation F1, then recall, then lower threshold",
                                  "evaluation_scope": "offline public-benchmark evaluator, outside the Flower server"},
              "clients": by_bank,
              "macro_before_test": macro_average({bank: value["before"] for bank, value in by_bank.items()}),
              "macro_after_test": {f"budget_{budget}_steps_{steps}": macro_average({
                  bank: next(item["test"] for item in value["adaptations"]
                             if item["support_budget"] == budget and item["steps"] == steps)
                  for bank, value in by_bank.items()})
                  for budget in (32, 128, 512) for steps in (1, 3, 5)}}
    output = run_dir / "results.json"
    if output.exists():
        raise FileExistsError("Meta evaluation already exists")
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    status["status"] = "completed"
    temporary = status_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(temporary, status_path)
    return output
