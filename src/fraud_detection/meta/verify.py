"""Recompute saved first-order meta rounds and personalization metrics."""

import json
from pathlib import Path

import numpy as np
import torch

from fraud_detection.data.pipeline import BANKS, file_hash
from fraud_detection.evaluation.metrics import choose_f1_threshold, classification_metrics, macro_average
from fraud_detection.federated.protocol import state_fingerprint, validate_state
from fraud_detection.meta.evaluate import _support_indices
from fraud_detection.meta.learning import adapt, episode_indices, query_gradient
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec
from fraud_detection.training.core import read_split, score_model


def verify_meta(project_root: Path, *, setting: str, seed: int,
                clients_dir: str = "clients", data_dir: str = "data",
                runs_dir: str = "runs") -> dict:
    run_dir = project_root / runs_dir / "meta" / setting / f"seed_{seed}"
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    partition = json.loads((project_root / data_dir / "processed" / setting / "metadata.json").read_text(encoding="utf-8"))
    if (run["status"] != result["status"] or run["status"] != "completed"
            or run["algorithm"] != result["method"] or run["algorithm"] != "fedmaml"
            or run["partition_manifest_sha256"] != partition["manifest_sha256"]
            or result["partition_manifest_sha256"] != partition["manifest_sha256"]):
        raise ValueError("Meta run metadata mismatch")
    spec = ModelSpec(**run["model_spec"])
    train_data = {bank: read_split(project_root, clients_dir, bank, setting, "train") for bank in BANKS}
    previous = None
    for number in range(0, int(run["rounds_completed"]) + 1):
        path = run_dir / f"round_{number}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        validate_state(state, spec)
        if number == 0:
            if (file_hash(path) != run["initial_checkpoint_sha256"]
                    or state_fingerprint(state) != run["initial_model_sha256"]):
                raise ValueError("Initial meta checkpoint changed")
        else:
            report = json.loads((run_dir / f"round_{number}.json").read_text(encoding="utf-8"))
            if (report["status"] != "completed" or report["base_sha256"] != state_fingerprint(previous)
                    or report["aggregate_sha256"] != state_fingerprint(state)
                    or file_hash(path) != report["checkpoint_sha256"]
                    or set(report["clients"]) != set(BANKS)
                    or set(report["validation"]) != set(BANKS)):
                raise ValueError(f"Meta round {number} changed")
            gradients = []
            for index, bank in enumerate(BANKS):
                x, y = train_data[bank]
                support, query = episode_indices(
                    y, support_size=int(run["run_config"]["support-size"]),
                    query_size=int(run["run_config"]["query-size"]),
                    seed=seed + 1000 * number + index)
                gradient, loss = query_gradient(
                    previous, spec, x, y, support, query,
                    steps=int(run["run_config"]["inner-steps"]),
                    inner_lr=float(run["run_config"]["inner-lr"]),
                    seed=seed + 1000 * number + index)
                observation = report["clients"][bank]
                if (state_fingerprint(gradient) != observation["gradient_sha256"]
                        or not np.isclose(loss, observation["query_loss"], rtol=1e-5)):
                    raise ValueError(f"Meta gradient changed: round {number}, {bank}")
                gradients.append(gradient)
            beta = float(run["run_config"]["outer-lr"])
            for name in state:
                manual = previous[name] - beta * torch.stack([grad[name] for grad in gradients]).mean(0)
                torch.testing.assert_close(state[name], manual, rtol=1e-5, atol=1e-7)
        previous = state
    final_hash = state_fingerprint(previous)
    if final_hash != run["final_model_sha256"] or final_hash != result["final_model_sha256"]:
        raise ValueError("Final meta model changed")
    before_by_bank = {}
    after_by_bank = {key: {} for key in result["macro_after_test"]}
    for index, bank in enumerate(BANKS):
        x_train, y_train = train_data[bank]
        x_valid, y_valid = read_split(project_root, clients_dir, bank, setting, "validation")
        x_test, y_test = read_split(project_root, clients_dir, bank, setting, "test")
        model = FraudMLP(spec)
        model.load_state_dict(previous)
        threshold, _ = choose_f1_threshold(y_valid, score_model(model, x_valid))
        before = classification_metrics(y_test, score_model(model, x_test), threshold)
        if before != result["clients"][bank]["before"]:
            raise ValueError(f"Before-adaptation metrics changed: {bank}")
        before_by_bank[bank] = before
        for item in result["clients"][bank]["adaptations"]:
            budget, steps = int(item["support_budget"]), int(item["steps"])
            support = _support_indices(y_train, budget, seed + 10000 * index + budget)
            personalized = FraudMLP(spec)
            personalized.load_state_dict(previous)
            adapt(personalized, x_train[support], y_train[support], steps=steps,
                  learning_rate=float(run["run_config"]["inner-lr"]),
                  seed=seed + 10000 * index + budget + steps)
            threshold, _ = choose_f1_threshold(y_valid, score_model(personalized, x_valid))
            actual = classification_metrics(y_test, score_model(personalized, x_test), threshold)
            if (actual != item["test"] or int(y_train[support].sum()) != item["support_fraud"]
                    or state_fingerprint(personalized.state_dict()) != item["model_sha256"]):
                raise ValueError(f"Adaptation changed: {bank}, {budget}, {steps}")
            after_by_bank[f"budget_{budget}_steps_{steps}"][bank] = actual
    if macro_average(before_by_bank) != result["macro_before_test"]:
        raise ValueError("Macro before metrics changed")
    for key, values in after_by_bank.items():
        if macro_average(values) != result["macro_after_test"][key]:
            raise ValueError(f"Macro adaptation metrics changed: {key}")
    return {"status": "passed", "setting": setting, "seed": seed,
            "rounds": int(run["rounds_completed"]), "clients_per_round": 4,
            "adaptation_metric_sets_recomputed": sum(len(values) for values in after_by_bank.values()),
            "macro_before_test": result["macro_before_test"]}
