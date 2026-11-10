"""Bank-local probability SHAP explanations with artifact-bound cache keys."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

from fraud_detection.data.pipeline import BANKS, FEATURES, file_hash
from fraud_detection.federated.protocol import state_fingerprint, validate_state
from fraud_detection.meta.evaluate import _support_indices
from fraud_detection.meta.learning import adapt
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec
from fraud_detection.training.core import read_split, score_model


def _hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_bank_model(root: Path, *, bank: str, setting: str, seed: int,
                    method: str = "fedmaml", support_budget: int | None = 32,
                    steps: int = 5, clients_dir: str = "clients",
                    runs_dir: str = "runs") -> tuple[FraudMLP, float, dict]:
    if bank not in BANKS or setting not in ("mild", "strong"):
        raise ValueError("Unknown bank or setting")
    if method not in ("fedmaml", "fedavg"):
        raise ValueError("Unknown model method")
    run_dir = root / runs_dir / ("meta" if method == "fedmaml" else "federated") / setting / f"seed_{seed}"
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if run["status"] != "completed" or result["status"] != "completed":
        raise ValueError("Selected model run is incomplete")
    spec = ModelSpec(**run["model_spec"])
    checkpoint = run_dir / run["final_checkpoint"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_state(state, spec)
    if (state_fingerprint(state) != run["final_model_sha256"]
            or state_fingerprint(state) != result["final_model_sha256"]):
        raise ValueError("Saved model hash mismatch")
    model = FraudMLP(spec)
    model.load_state_dict(state)
    adaptation = None
    if method == "fedmaml" and support_budget is not None:
        matches = [item for item in result["clients"][bank]["adaptations"]
                   if item["support_budget"] == support_budget and item["steps"] == steps]
        if len(matches) != 1:
            raise ValueError("Unknown personalization setting")
        train_x, train_y = read_split(root, clients_dir, bank, setting, "train")
        index = BANKS.index(bank)
        support = _support_indices(train_y, support_budget, seed + 10000 * index + support_budget)
        previous_threads = torch.get_num_threads()
        target_threads = int(result.get("adaptation_torch_threads", run.get("torch_num_threads", previous_threads)))
        if target_threads < 1:
            raise ValueError("Invalid saved adaptation thread count")
        try:
            torch.set_num_threads(target_threads)
            adapt(model, train_x[support], train_y[support], steps=steps,
                  learning_rate=float(run["run_config"]["inner-lr"]),
                  seed=seed + 10000 * index + support_budget + steps)
        finally:
            torch.set_num_threads(previous_threads)
        adaptation = {"support_budget": support_budget, "steps": steps,
                      "support_fraud": int(train_y[support].sum())}
        if state_fingerprint(model.state_dict()) != matches[0]["model_sha256"]:
            raise ValueError("Personalized model differs from saved evaluation")
        threshold = float(matches[0]["test"]["threshold"])
    elif method == "fedmaml":
        threshold = float(result["clients"][bank]["before"]["threshold"])
    else:
        if support_budget is not None:
            raise ValueError("FedAvg has no personalized support budget")
        threshold = float(result["clients"][bank]["test"]["threshold"])
    model.eval()
    return model, threshold, {"method": method, "model_sha256": state_fingerprint(model.state_dict()),
                              "checkpoint_sha256": file_hash(checkpoint), "adaptation": adaptation,
                              "run_dir": str(run_dir)}


def _render_plots(values: np.ndarray, features: np.ndarray, expected: float,
                  output_dir: Path) -> None:
    import plotly.graph_objects as go

    labels = FEATURES
    waterfall = go.Figure(go.Waterfall(
        orientation="h", measure=["relative"] * len(labels),
        y=labels, x=[float(value) for value in values],
        text=[f"{value:+.4f}" for value in values],
        connector={"line": {"color": "#6b7280"}}))
    waterfall.update_layout(title=f"Fraud score contributions (baseline {expected:.4f})",
                            xaxis_title="Change in predicted probability", height=850)
    waterfall.write_html(output_dir / "waterfall.html", include_plotlyjs=True)
    order = np.argsort(np.abs(values))[-15:]
    bar = go.Figure(go.Bar(x=values[order], y=[labels[i] for i in order], orientation="h",
                           marker_color=["#d97706" if values[i] > 0 else "#2563eb" for i in order]))
    bar.update_layout(title="Largest local contributions", xaxis_title="Probability contribution",
                      height=650)
    bar.write_html(output_dir / "bar.html", include_plotlyjs=True)


def explain_bank_record(root: Path, *, bank: str, setting: str, seed: int,
                        row_id: int | None = None, method: str = "fedmaml",
                        support_budget: int | None = 32, steps: int = 5,
                        background_size: int = 32, nsamples: int = 128,
                        clients_dir: str = "clients", runs_dir: str = "runs") -> Path:
    if not 1 <= background_size <= 100 or nsamples < 2 * len(FEATURES) + 2:
        raise ValueError("Invalid SHAP background or sample budget")
    model, threshold, model_info = load_bank_model(
        root, bank=bank, setting=setting, seed=seed, method=method,
        support_budget=support_budget, steps=steps,
        clients_dir=clients_dir, runs_dir=runs_dir)
    local_root = root / clients_dir / bank
    test_path = local_root / "data" / setting / "test.parquet"
    frame = pd.read_parquet(test_path)
    if list(frame.columns) != ["row_id", *FEATURES, "Class"]:
        raise ValueError("Unexpected test feature schema")
    if row_id is None:
        selected = frame.loc[frame["Class"] == 1].head(1)
    else:
        selected = frame.loc[frame["row_id"] == row_id]
    if len(selected) != 1:
        raise ValueError("Requested row must identify exactly one bank-local test record")
    row_id = int(selected.iloc[0]["row_id"])
    feature_row = selected[FEATURES].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(feature_row).all():
        raise ValueError("Selected transaction contains nonfinite features")
    train_x, _ = read_split(root, clients_dir, bank, setting, "train")
    if background_size > len(train_x):
        raise ValueError("Background exceeds training rows")
    rng = np.random.default_rng(seed + 20000 + BANKS.index(bank))
    background = train_x[rng.choice(len(train_x), background_size, replace=False)]
    scaler_path = local_root / "preprocessing" / f"{setting}_scaler.json"
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    if scaler["client_id"] != bank or scaler["setting"] != setting:
        raise ValueError("Bank preprocessing reference mismatch")
    cache_contract = {"schema_version": 1, "bank": bank, "setting": setting,
                      "seed": seed,
                      "row_id": row_id, "model_sha256": model_info["model_sha256"],
                      "preprocessing_sha256": file_hash(scaler_path),
                      "background_sha256": _hash_array(background),
                      "transaction_sha256": _hash_array(feature_row),
                      "nsamples": nsamples, "shap_version": shap.__version__,
                      "output_scale": "predicted fraud probability"}
    cache_key = hashlib.sha256(json.dumps(cache_contract, sort_keys=True).encode()).hexdigest()
    output_dir = local_root / "explanations" / setting / method / cache_key
    output_path = output_dir / "explanation.json"
    if output_path.is_file():
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("cache_contract") == cache_contract and all(
                (output_dir / name).is_file() for name in ("waterfall.html", "bar.html")):
            return output_path
        raise ValueError("Existing explanation cache is incomplete or altered")
    output_dir.mkdir(parents=True, exist_ok=False)
    predictor = lambda values: score_model(model, np.array(values, dtype=np.float32, copy=True))
    explainer = shap.KernelExplainer(predictor, background, feature_names=FEATURES)
    values = np.asarray(explainer.shap_values(feature_row, nsamples=nsamples,
                                             l1_reg=0, silent=True)).reshape(-1)
    if values.shape != (len(FEATURES),) or not np.isfinite(values).all():
        raise ValueError("Invalid SHAP feature contributions")
    expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    score = float(predictor(feature_row)[0])
    reconstructed = expected + float(values.sum())
    tolerance = 1e-3
    if not np.isfinite(expected) or abs(reconstructed - score) > tolerance:
        raise ValueError("SHAP contributions failed probability reconstruction")
    _render_plots(values, feature_row[0], expected, output_dir)
    artifact = {"schema_version": 1, "cache_contract": cache_contract,
                "model": {key: value for key, value in model_info.items() if key != "run_dir"},
                "method": method, "bank": bank, "setting": setting, "row_id": row_id,
                "score": score, "threshold": threshold, "flagged": bool(score >= threshold),
                "expected_probability": expected,
                "feature_contributions": {name: float(value) for name, value in zip(FEATURES, values)},
                "feature_values": {name: float(value) for name, value in zip(FEATURES, feature_row[0])},
                "reconstructed_probability": reconstructed,
                "reconstruction_error": abs(reconstructed - score),
                "reconstruction_tolerance": tolerance, "background_rows": background_size,
                "approximation_samples": nsamples,
                "plots": {"waterfall": "waterfall.html", "bar": "bar.html"},
                "caveat": "ULB V1-V28 are PCA components; SHAP describes model associations, not causality."}
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path
