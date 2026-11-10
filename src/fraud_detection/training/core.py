"""Shared data loading and PyTorch training for baseline experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import torch

from fraud_detection.data.pipeline import FEATURES
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec


@dataclass(frozen=True)
class TrainSpec:
    epochs: int = 5
    batch_size: int = 2048
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    hidden_1: int = 64
    hidden_2: int = 32
    dropout: float = 0.2
    max_grad_norm: float = 5.0

    @classmethod
    def from_toml(cls, path: Path) -> "TrainSpec":
        with path.open("rb") as stream:
            settings = tomllib.load(stream)
        unknown = set(settings) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown training settings: {sorted(unknown)}")
        spec = cls(**settings)
        if (spec.epochs < 1 or spec.batch_size < 1 or spec.learning_rate <= 0
                or spec.weight_decay < 0 or spec.hidden_1 < 1 or spec.hidden_2 < 1
                or not 0 <= spec.dropout < 1 or spec.max_grad_norm <= 0):
            raise ValueError("Invalid training settings")
        return spec

    def model_spec(self) -> ModelSpec:
        return ModelSpec(input_dim=len(FEATURES), hidden_1=self.hidden_1,
                         hidden_2=self.hidden_2, dropout=self.dropout)

    def as_dict(self) -> dict:
        return asdict(self)


def read_split(project_root: Path, clients_dir: str, bank: str,
               setting: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = project_root / clients_dir / bank / "data" / setting / f"{split}.parquet"
    frame = pd.read_parquet(path)
    if list(frame.columns) != ["row_id", *FEATURES, "Class"]:
        raise ValueError(f"Unexpected feature order in {path}")
    x = frame[FEATURES].to_numpy(dtype=np.float32, copy=True)
    y = frame["Class"].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError(f"Invalid model data in {path}")
    return x, y


def fit_model(train_x: np.ndarray, train_y: np.ndarray,
              valid_x: np.ndarray, valid_y: np.ndarray,
              spec: TrainSpec, *, seed: int, device: str = "cpu") -> tuple[FraudMLP, dict]:
    """Train only on train_x; choose checkpoint using validation average precision."""
    from sklearn.metrics import average_precision_score

    if len(train_x) != len(train_y) or len(valid_x) != len(valid_y):
        raise ValueError("Feature/label length mismatch")
    if not (np.isin(train_y, [0, 1]).all() and len(np.unique(train_y)) == 2
            and np.isin(valid_y, [0, 1]).all() and len(np.unique(valid_y)) == 2):
        raise ValueError("Train and validation data need both classes")
    if len(train_x) == 0 or len(valid_x) == 0:
        raise ValueError("Empty training or validation data")
    if train_x.shape[1] != len(FEATURES) or valid_x.shape[1] != len(FEATURES):
        raise ValueError("Unexpected feature dimension")
    if not np.isfinite(train_x).all() or not np.isfinite(valid_x).all():
        raise ValueError("Nonfinite model inputs")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.manual_seed(seed)
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model = FraudMLP(spec.model_spec()).to(device)
    positive = int(train_y.sum())
    negative = len(train_y) - positive
    pos_weight = negative / positive
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate,
                                 weight_decay=spec.weight_decay)
    features = torch.from_numpy(train_x)
    labels = torch.from_numpy(train_y)
    generator = torch.Generator().manual_seed(seed)
    history = []
    best_ap = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(1, spec.epochs + 1):
        model.train()
        order = torch.randperm(len(features), generator=generator)
        total_loss = 0.0
        for batch_ids in order.split(spec.batch_size):
            batch_x = features[batch_ids].to(device)
            batch_y = labels[batch_ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_ids)
        scores = score_model(model, valid_x, device=device)
        ap = float(average_precision_score(valid_y, scores))
        history.append({"epoch": epoch, "train_weighted_loss": total_loss / len(features),
                        "validation_average_precision": ap})
        if ap > best_ap + 1e-12:
            best_ap = ap
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No valid model checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_validation_average_precision": best_ap,
                   "positive_class_weight": pos_weight, "train_rows": len(train_y),
                   "train_fraud": positive, "validation_rows": len(valid_y),
                   "validation_fraud": int(valid_y.sum()), "history": history}


def score_model(model: FraudMLP, x: np.ndarray, *, device: str = "cpu",
                batch_size: int = 8192) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            values = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            parts.append(torch.sigmoid(model(values)).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64) if parts else np.empty(0, dtype=np.float64)
