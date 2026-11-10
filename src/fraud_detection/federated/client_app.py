"""Flower ClientApp: each SuperNode reads only its assigned bank directory."""

import os
from pathlib import Path

import numpy as np
import torch
from flwr.app import ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import average_precision_score

from fraud_detection.data.pipeline import BANKS
from fraud_detection.federated.local_training import train_local
from fraud_detection.federated.protocol import ALGORITHM, SCHEMA_VERSION, reply_payload, state_fingerprint, validate_state
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec
from fraud_detection.training.core import read_split, score_model


app = ClientApp()


def _bank_and_root(context: Context) -> tuple[str, Path]:
    bank = str(context.node_config["bank-id"])
    if bank not in BANKS:
        raise ValueError(f"Unknown simulated bank: {bank}")
    root = Path(os.environ["FRAUD_PROJECT_ROOT"]).resolve()
    return bank, root


def _model_from_message(message: Message) -> tuple[FraudMLP, str]:
    model = FraudMLP(ModelSpec())
    state = message.content["arrays"].to_torch_state_dict()
    validate_state(state, model.spec)
    model.load_state_dict(state)
    return model, state_fingerprint(state)


@app.train()
def train(message: Message, context: Context) -> Message:
    bank, root = _bank_and_root(context)
    config = message.content["config"]
    round_number = int(config["server-round"])
    model, received_hash = _model_from_message(message)
    if (config["algorithm"] != ALGORITHM or config["schema-version"] != SCHEMA_VERSION
            or received_hash != config["base-sha256"] or round_number < 1):
        raise ValueError("Stale, malformed, or incompatible model initialization")
    setting = str(config["setting"])
    x, y = read_split(root, str(config["clients-dir"]), bank, setting, "train")
    loss = train_local(model, x, y,
                       epochs=int(config["local-epochs"]),
                       batch_size=int(config["batch-size"]),
                       learning_rate=float(config["learning-rate"]),
                       weight_decay=float(config["weight-decay"]),
                       max_grad_norm=float(config["max-grad-norm"]),
                       seed=int(config["seed"]) + 1000 * round_number + BANKS.index(bank))
    payload = reply_payload(model, bank=bank, round_number=round_number,
                            base_hash=received_hash, num_examples=len(y), train_loss=loss)
    return Message(content=payload, reply_to=message)


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    bank, root = _bank_and_root(context)
    config = message.content["config"]
    model, _ = _model_from_message(message)
    x, y = read_split(root, str(config["clients-dir"]), bank, str(config["setting"]), "validation")
    scores = score_model(model, x)
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(y)).item()
    ap = float(average_precision_score(y, scores))
    if not np.isfinite(ap) or not np.isfinite(loss):
        raise ValueError("Nonfinite validation metrics")
    payload = RecordDict({
        "metrics": MetricRecord({"num-examples": len(y), "validation-loss": loss,
                                 "validation-average-precision": ap}),
        "audit": ConfigRecord({"algorithm": ALGORITHM, "schema-version": SCHEMA_VERSION,
                               "bank-id": bank, "server-round": int(config["server-round"])}),
    })
    return Message(content=payload, reply_to=message)
