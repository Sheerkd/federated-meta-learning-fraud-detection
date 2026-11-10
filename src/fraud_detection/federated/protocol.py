"""FedAvg message and model-schema checks shared by the server and clients."""

import hashlib

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, RecordDict

from fraud_detection.models.fraud_model import FraudMLP, ModelSpec


ALGORITHM = "fedavg"
SCHEMA_VERSION = 1


def validate_state(state: dict[str, torch.Tensor], spec: ModelSpec) -> None:
    expected = FraudMLP(spec).state_dict()
    if list(state) != list(expected):
        raise ValueError("Model tensor names or order differ from the agreed schema")
    for name, tensor in state.items():
        reference = expected[name]
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValueError(f"Tensor schema mismatch: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite tensor: {name}")


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_bytes(state: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())


def reply_payload(model: FraudMLP, *, bank: str, round_number: int,
                  base_hash: str, num_examples: int, train_loss: float) -> RecordDict:
    """Only model parameters and aggregate metadata can enter a train reply."""
    if not np.isfinite(train_loss) or num_examples <= 0:
        raise ValueError("Invalid aggregate client metrics")
    state = model.state_dict()
    validate_state(state, model.spec)
    return RecordDict({
        "arrays": ArrayRecord(state),
        "metrics": MetricRecord({"num-examples": int(num_examples),
                                 "train-loss": float(train_loss)}),
        "audit": ConfigRecord({"algorithm": ALGORITHM, "schema-version": SCHEMA_VERSION,
                               "bank-id": bank, "server-round": int(round_number),
                               "base-sha256": base_hash}),
    })
