"""Independently verify bank-local explanation references and reconstruction."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.data.pipeline import BANKS, FEATURES, file_hash
from fraud_detection.explainability.local import _hash_array, load_bank_model
from fraud_detection.training.core import read_split, score_model


def verify_explanations(root: Path, *, setting: str, seed: int,
                        clients_dir: str = "clients", runs_dir: str = "runs") -> dict:
    if setting not in ("mild", "strong"):
        raise ValueError("Unknown setting")
    count = 0
    by_bank = {}
    for index, bank in enumerate(BANKS):
        local_root = root / clients_dir / bank
        base = local_root / "explanations" / setting
        if not base.is_dir():
            raise ValueError(f"No explanations for {bank}, {setting}")
        train_x, _ = read_split(root, clients_dir, bank, setting, "train")
        test = pd.read_parquet(local_root / "data" / setting / "test.parquet")
        bank_count = 0
        for path in base.glob("*/?*/explanation.json"):
            artifact = json.loads(path.read_text(encoding="utf-8"))
            contract = artifact["cache_contract"]
            artifact_seed = int(contract.get("seed", seed))
            if (contract["bank"] != bank or contract["setting"] != setting
                    or artifact["bank"] != bank or artifact["setting"] != setting
                    or artifact["row_id"] != contract["row_id"]
                    or path.parent.name != hashlib.sha256(json.dumps(
                        contract, sort_keys=True).encode()).hexdigest()):
                raise ValueError(f"Explanation cache contract changed: {path}")
            model_details = artifact["model"]
            adaptation = model_details["adaptation"]
            model, threshold, info = load_bank_model(
                root, bank=bank, setting=setting, seed=artifact_seed,
                method=artifact["method"],
                support_budget=adaptation["support_budget"] if adaptation else None,
                steps=adaptation["steps"] if adaptation else 5,
                clients_dir=clients_dir, runs_dir=runs_dir)
            if (info["model_sha256"] != contract["model_sha256"]
                    or info["model_sha256"] != model_details["model_sha256"]
                    or info["checkpoint_sha256"] != model_details["checkpoint_sha256"]
                    or threshold != artifact["threshold"]):
                raise ValueError(f"Explanation model changed: {path}")
            scaler = local_root / "preprocessing" / f"{setting}_scaler.json"
            if file_hash(scaler) != contract["preprocessing_sha256"]:
                raise ValueError(f"Explanation preprocessing changed: {path}")
            selected = test.loc[test["row_id"] == int(artifact["row_id"])]
            if len(selected) != 1:
                raise ValueError(f"Explained record is missing: {path}")
            row = selected[FEATURES].to_numpy(dtype=np.float32, copy=True)
            if (_hash_array(row) != contract["transaction_sha256"]
                    or [float(value) for value in row[0]] != [artifact["feature_values"][name] for name in FEATURES]):
                raise ValueError(f"Explained features changed: {path}")
            rng = np.random.default_rng(artifact_seed + 20000 + index)
            background = train_x[rng.choice(len(train_x), int(artifact["background_rows"]), replace=False)]
            if _hash_array(background) != contract["background_sha256"]:
                raise ValueError(f"Explanation background changed: {path}")
            score = float(score_model(model, row)[0])
            expected = float(np.mean(score_model(model, background)))
            contributions = artifact["feature_contributions"]
            if list(contributions) != FEATURES:
                raise ValueError(f"Explanation feature order changed: {path}")
            reconstruction = expected + sum(float(contributions[name]) for name in FEATURES)
            if (not np.isclose(score, artifact["score"], rtol=0, atol=1e-7)
                    or not np.isclose(expected, artifact["expected_probability"], rtol=0, atol=1e-7)
                    or abs(reconstruction - score) > artifact["reconstruction_tolerance"]
                    or bool(score >= threshold) != artifact["flagged"]):
                raise ValueError(f"Explanation probability reconstruction failed: {path}")
            if not all((path.parent / name).is_file() for name in ("waterfall.html", "bar.html")):
                raise ValueError(f"Explanation plot missing: {path}")
            count += 1
            bank_count += 1
        by_bank[bank] = bank_count
    if not count or any(value == 0 for value in by_bank.values()):
        raise ValueError("Expected at least one local explanation per bank")
    return {"status": "passed", "setting": setting, "explanations_verified": count,
            "by_bank": by_bank}
