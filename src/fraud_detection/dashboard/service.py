"""Read-only dashboard queries plus explicitly requested background runs."""

import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from fraud_detection.config import load_config
from fraud_detection.data.pipeline import BANKS, FEATURES, file_hash
from fraud_detection.explainability.local import load_bank_model
from fraud_detection.federated.protocol import state_fingerprint, validate_state
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec
from fraud_detection.training.core import score_model


METHODS = ("Local", "Centralized", "FedAvg", "FedMAML", "Personalized FedMAML")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_path(root: Path, runs_dir: str, method: str, setting: str, seed: int) -> Path:
    family = {"Local": "baselines", "Centralized": "baselines",
              "FedAvg": "federated", "FedMAML": "meta",
              "Personalized FedMAML": "meta"}[method]
    return root / runs_dir / family / setting / f"seed_{seed}" / "results.json"


def available_methods(root: Path, *, setting: str, seed: int, runs_dir: str = "runs") -> list[str]:
    available = []
    for method in METHODS:
        path = _result_path(root, runs_dir, method, setting, seed)
        if path.is_file():
            try:
                result = _json(path)
                if result["status"] == "completed" and result["setting"] == setting and result["seed"] == seed:
                    available.append(method)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
    return available


def dataset_summary(root: Path, *, setting: str, data_dir: str = "data") -> dict:
    path = root / data_dir / "processed" / setting / "metadata.json"
    metadata = _json(path)
    return {"source_rows": metadata["dataset"]["rows"],
            "source_fraud": metadata["dataset"]["fraud"],
            "source_prevalence": metadata["dataset"]["fraud_prevalence"],
            "banks": {bank: metadata["partition"]["counts"][bank] for bank in BANKS}}


def comparison_rows(root: Path, *, setting: str, seed: int,
                    runs_dir: str = "runs") -> pd.DataFrame:
    rows = []
    for method in available_methods(root, setting=setting, seed=seed, runs_dir=runs_dir):
        result = _json(_result_path(root, runs_dir, method, setting, seed))
        if method == "Local":
            by_bank = {bank: result["local"][bank]["test"] for bank in BANKS}
        elif method == "Centralized":
            by_bank = {bank: result["centralized"]["clients"][bank]["test"] for bank in BANKS}
        elif method == "FedAvg":
            by_bank = {bank: result["clients"][bank]["test"] for bank in BANKS}
        elif method == "FedMAML":
            by_bank = {bank: result["clients"][bank]["before"] for bank in BANKS}
        else:
            by_bank = {bank: next(item["test"] for item in result["clients"][bank]["adaptations"]
                                  if item["support_budget"] == 32 and item["steps"] == 5)
                       for bank in BANKS}
        for bank, metrics in by_bank.items():
            rows.append({"Method": method, "Bank": bank, "Test rows": metrics["rows"],
                         "Fraud": metrics["fraud"], "Prevalence": metrics["fraud_prevalence"],
                         "Precision": metrics["precision"], "Recall": metrics["recall"],
                         "F1": metrics["f1"], "False positive rate": metrics["false_positive_rate"],
                         "ROC AUC": metrics["roc_auc"], "Average precision": metrics["average_precision"]})
    return pd.DataFrame(rows)


def round_rows(root: Path, *, setting: str, seed: int, method: str,
               runs_dir: str = "runs") -> pd.DataFrame:
    if method not in ("FedAvg", "FedMAML"):
        raise ValueError("Round progress requires a federated method")
    family = "federated" if method == "FedAvg" else "meta"
    run_dir = root / runs_dir / family / setting / f"seed_{seed}"
    reports = []
    for path in sorted(run_dir.glob("round_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        report = _json(path)
        clients = report.get("clients", {})
        reports.append({"Round": report["round"], "Status": report["status"],
                        "Banks": len(clients), "Duration (s)": report.get("round_total_seconds"),
                        "Mean local loss": float(np.mean([item.get("train_loss", item.get("query_loss"))
                                                           for item in clients.values()])) if clients else None,
                        "Tensor bytes": report.get("train_broadcast_tensor_bytes", 0)
                        + report.get("train_update_tensor_bytes", report.get("gradient_update_tensor_bytes", 0))
                        + report.get("validation_broadcast_tensor_bytes", 0)})
    return pd.DataFrame(reports)


def sample_options(root: Path, *, bank: str, setting: str,
                   clients_dir: str = "clients") -> list[dict]:
    if bank not in BANKS or setting not in ("mild", "strong"):
        raise ValueError("Unknown bank or setting")
    path = root / clients_dir / bank / "data" / setting / "test.parquet"
    frame = pd.read_parquet(path, columns=["row_id", "Class"])
    selected = pd.concat([frame.loc[frame["Class"] == 1].head(10),
                          frame.loc[frame["Class"] == 0].head(10)], ignore_index=True)
    return [{"row_id": int(row.row_id), "label": int(row.Class)} for row in selected.itertuples()]


def _baseline_model(root: Path, *, bank: str, setting: str, seed: int,
                    method: str, clients_dir: str, runs_dir: str) -> tuple[FraudMLP, float, str]:
    result = _json(_result_path(root, runs_dir, method, setting, seed))
    if result["status"] != "completed":
        raise ValueError("Baseline run is incomplete")
    bundle_path = (root / clients_dir / bank / "models" / setting / f"local_seed_{seed}.json"
                   if method == "Local" else
                   root / runs_dir / "baselines" / setting / f"seed_{seed}" / "centralized.json")
    bundle = _json(bundle_path)
    checkpoint = root / bundle["checkpoint"]
    if (file_hash(checkpoint) != bundle["checkpoint_sha256"]
            or bundle["feature_order"] != FEATURES):
        raise ValueError("Baseline model bundle changed")
    spec = ModelSpec(**bundle["model_spec"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_state(state, spec)
    model = FraudMLP(spec)
    model.load_state_dict(state)
    model.eval()
    if method == "Local":
        threshold = float(bundle["threshold"])
    else:
        threshold = float(bundle["client_thresholds"][bank])
    return model, threshold, state_fingerprint(state)


def selected_model(root: Path, *, bank: str, setting: str, seed: int, method: str,
                   clients_dir: str = "clients", runs_dir: str = "runs") -> tuple[FraudMLP, float, str]:
    if bank not in BANKS or method not in METHODS or setting not in ("mild", "strong"):
        raise ValueError("Unknown bank, method, or setting")
    if method not in available_methods(root, setting=setting, seed=seed, runs_dir=runs_dir):
        raise FileNotFoundError("Selected saved model is unavailable")
    if method in ("Local", "Centralized"):
        return _baseline_model(root, bank=bank, setting=setting, seed=seed,
                               method=method, clients_dir=clients_dir, runs_dir=runs_dir)
    model, threshold, info = load_bank_model(
        root, bank=bank, setting=setting, seed=seed,
        method="fedavg" if method == "FedAvg" else "fedmaml",
        support_budget=32 if method == "Personalized FedMAML" else None,
        steps=5, clients_dir=clients_dir, runs_dir=runs_dir)
    return model, threshold, info["model_sha256"]


def predict_sample(root: Path, *, bank: str, setting: str, seed: int,
                   method: str, row_id: int, clients_dir: str = "clients",
                   runs_dir: str = "runs") -> dict:
    model, threshold, model_hash = selected_model(root, bank=bank, setting=setting,
                                                   seed=seed, method=method,
                                                   clients_dir=clients_dir, runs_dir=runs_dir)
    frame = pd.read_parquet(root / clients_dir / bank / "data" / setting / "test.parquet")
    selected = frame.loc[frame["row_id"] == row_id]
    if len(selected) != 1:
        raise ValueError("Selected bank-local test row is unavailable")
    features = selected[FEATURES].to_numpy(dtype=np.float32, copy=True)
    score = float(score_model(model, features)[0])
    return {"bank": bank, "setting": setting, "method": method, "row_id": row_id,
            "score": score, "threshold": threshold, "flagged": score >= threshold,
            "held_out_label": int(selected.iloc[0]["Class"]),
            "model_sha256": model_hash}


def parse_upload(file_bytes: bytes, *, max_rows: int = 1000) -> pd.DataFrame:
    if len(file_bytes) > 2_000_000:
        raise ValueError("CSV exceeds 2 MB")
    try:
        frame = pd.read_csv(io.BytesIO(file_bytes))
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError("Could not read CSV") from exc
    if list(frame.columns) != FEATURES:
        raise ValueError("CSV columns must be Time, V1 through V28, Amount in that order")
    if not 1 <= len(frame) <= max_rows:
        raise ValueError("CSV needs 1 to 1000 rows")
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("CSV features must be numeric") from exc
    if not np.isfinite(values).all() or (frame[["Time", "Amount"]] < 0).any().any():
        raise ValueError("CSV contains missing, nonfinite, or negative Time/Amount values")
    return frame


def predict_upload(root: Path, *, bank: str, setting: str, seed: int,
                   method: str, file_bytes: bytes, clients_dir: str = "clients",
                   runs_dir: str = "runs") -> pd.DataFrame:
    frame = parse_upload(file_bytes)
    model, threshold, model_hash = selected_model(root, bank=bank, setting=setting,
                                                   seed=seed, method=method,
                                                   clients_dir=clients_dir, runs_dir=runs_dir)
    scaler_path = root / clients_dir / bank / "preprocessing" / f"{setting}_scaler.json"
    scaler = _json(scaler_path)
    if scaler["client_id"] != bank or scaler["setting"] != setting:
        raise ValueError("Bank preprocessing reference mismatch")
    values = frame.to_numpy(dtype=np.float32, copy=True)
    for column, mean, scale in zip(scaler["features"], scaler["mean"], scaler["scale"]):
        values[:, FEATURES.index(column)] = (values[:, FEATURES.index(column)] - mean) / scale
    if not np.isfinite(values).all():
        raise ValueError("Transformed CSV contains invalid values")
    scores = score_model(model, values)
    return pd.DataFrame({"Row": np.arange(1, len(scores) + 1),
                         "Fraud score": scores, "Threshold": threshold,
                         "Flagged": scores >= threshold,
                         "Model": method, "Model hash": model_hash})


def launch_run(root: Path, *, method: str, setting: str, seed: int,
               runs_dir: str = "runs") -> Path:
    if method not in ("FedAvg", "FedMAML") or setting not in ("mild", "strong"):
        raise ValueError("Choose a federated method and setting")
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must fit in 32 bits")
    family = "federated" if method == "FedAvg" else "meta"
    if (root / runs_dir / family / setting / f"seed_{seed}").exists():
        raise FileExistsError("A saved run already uses this setting and seed")
    job_dir = root / runs_dir / "dashboard_jobs" / f"{family}_{setting}_seed_{seed}"
    job_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "fraud_detection.dashboard.worker",
               "--root", str(root), "--method", method,
               "--setting", setting, "--seed", str(seed),
               "--job-dir", str(job_dir)]
    job_path = job_dir / "job.json"
    job_path.write_text(json.dumps({"method": method, "setting": setting,
                                    "seed": seed, "status": "starting",
                                    "run_dir": str(root / runs_dir / family / setting / f"seed_{seed}")},
                                   indent=2), encoding="utf-8")
    log = job_dir / "worker.log"
    with log.open("w", encoding="utf-8") as stream:
        try:
            subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                             stdout=stream, stderr=subprocess.STDOUT,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            job = _json(job_path)
            job.update({"status": "failed", "error": str(exc)})
            job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
            raise
    return job_dir


def job_rows(root: Path, *, runs_dir: str = "runs") -> pd.DataFrame:
    records = []
    for path in (root / runs_dir / "dashboard_jobs").glob("*/job.json"):
        job = _json(path)
        run_dir = Path(job["run_dir"])
        status = _json(run_dir / "run.json")["status"] if (run_dir / "run.json").is_file() else job["status"]
        if job["status"] in ("completed", "failed"):
            status = job["status"]
        records.append({"Method": job["method"], "Setting": job["setting"],
                        "Seed": job["seed"], "Status": status,
                        "Details": job.get("error", ""), "Log": str(path.parent / "worker.log")})
    return pd.DataFrame(records)
