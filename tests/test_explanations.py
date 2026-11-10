import json

import numpy as np
import pandas as pd
import pytest
import torch

from fraud_detection.data.pipeline import FEATURES
from fraud_detection.explainability import local
from fraud_detection.federated.protocol import state_fingerprint
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec


def _fixture_data(tmp_path, monkeypatch):
    bank_root = tmp_path / "clients" / "bank_a"
    data_dir = bank_root / "data" / "mild"
    data_dir.mkdir(parents=True)
    rng = np.random.default_rng(9)
    values = rng.normal(size=(48, len(FEATURES))).astype(np.float32)
    train = pd.DataFrame(values, columns=FEATURES)
    train.insert(0, "row_id", np.arange(48))
    train["Class"] = np.array([0] * 40 + [1] * 8)
    train.to_parquet(data_dir / "train.parquet", index=False)
    test = train.iloc[[42]].copy()
    test["row_id"] = 999
    test.to_parquet(data_dir / "test.parquet", index=False)
    preprocessing = bank_root / "preprocessing"
    preprocessing.mkdir()
    (preprocessing / "mild_scaler.json").write_text(
        json.dumps({"client_id": "bank_a", "setting": "mild", "mean": [0, 0]}))
    torch.manual_seed(9)
    model = FraudMLP(ModelSpec())
    model.eval()

    def load_model(*args, **kwargs):
        return model, 0.5, {"model_sha256": state_fingerprint(model.state_dict()),
                            "checkpoint_sha256": "test", "adaptation": None,
                            "method": "fedmaml", "run_dir": "test"}

    monkeypatch.setattr(local, "load_bank_model", load_model)
    return bank_root, model


def test_local_shap_reconstruction_cache_and_preprocessing_invalidation(tmp_path, monkeypatch):
    bank_root, model = _fixture_data(tmp_path, monkeypatch)
    kwargs = dict(bank="bank_a", setting="mild", seed=42, row_id=999,
                  background_size=8, nsamples=64)
    first = local.explain_bank_record(tmp_path, **kwargs)
    artifact = json.loads(first.read_text())
    assert artifact["reconstruction_error"] <= 1e-3
    assert len(artifact["feature_contributions"]) == len(FEATURES)
    assert (first.parent / "waterfall.html").is_file()
    assert (first.parent / "bar.html").is_file()
    assert local.explain_bank_record(tmp_path, **kwargs) == first
    scaler = bank_root / "preprocessing" / "mild_scaler.json"
    scaler.write_text(json.dumps({"client_id": "bank_a", "setting": "mild", "mean": [1, 0]}))
    second = local.explain_bank_record(tmp_path, **kwargs)
    assert second != first
    third = local.explain_bank_record(tmp_path, **{**kwargs, "background_size": 9})
    assert third not in (first, second)
    with torch.no_grad():
        next(model.parameters()).add_(0.001)
    fourth = local.explain_bank_record(tmp_path, **kwargs)
    assert fourth not in (first, second, third)
    test_path = bank_root / "data" / "mild" / "test.parquet"
    changed = pd.read_parquet(test_path)
    changed.loc[changed.index[0], "V1"] += 0.1
    changed.to_parquet(test_path, index=False)
    fifth = local.explain_bank_record(tmp_path, **kwargs)
    assert fifth not in (first, second, third, fourth)
    with pytest.raises(ValueError, match="exactly one"):
        local.explain_bank_record(tmp_path, **{**kwargs, "row_id": 123456})
