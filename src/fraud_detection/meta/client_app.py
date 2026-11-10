"""Bank-local first-order MAML episodes over Flower."""

import os
from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
import torch

from fraud_detection.data.pipeline import BANKS
from fraud_detection.federated.protocol import state_fingerprint, validate_state
from fraud_detection.meta.learning import ALGORITHM, episode_indices, query_gradient
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec
from fraud_detection.training.core import read_split, score_model
from sklearn.metrics import average_precision_score


app = ClientApp()


def _input(message: Message, context: Context):
    bank = str(context.node_config["bank-id"])
    if bank not in BANKS:
        raise ValueError("Unknown bank")
    config = message.content["config"]
    if config["algorithm"] != ALGORITHM or int(config["schema-version"]) != 1:
        raise ValueError("Incorrect meta-learning message type")
    state = message.content["arrays"].to_torch_state_dict()
    spec = ModelSpec()
    validate_state(state, spec)
    return bank, config, state, spec


@app.train()
def train(message: Message, context: Context) -> Message:
    bank, config, state, spec = _input(message, context)
    round_number = int(config["server-round"])
    base_hash = state_fingerprint(state)
    if round_number < 1 or base_hash != config["base-sha256"]:
        raise ValueError("Stale meta-learning initialization")
    root = Path(os.environ["FRAUD_PROJECT_ROOT"]).resolve()
    x, y = read_split(root, str(config["clients-dir"]), bank, str(config["setting"]), "train")
    episode_seed = int(config["seed"]) + 1000 * round_number + BANKS.index(bank)
    support, query = episode_indices(
        y, support_size=int(config["support-size"]), query_size=int(config["query-size"]),
        seed=episode_seed)
    gradients, loss = query_gradient(
        state, spec, x, y, support, query, steps=int(config["inner-steps"]),
        inner_lr=float(config["inner-lr"]), seed=episode_seed)
    payload = RecordDict({
        "arrays": ArrayRecord(gradients),
        "metrics": MetricRecord({"num-examples": len(query), "query-loss": loss,
                                 "support-examples": len(support)}),
        "audit": ConfigRecord({"algorithm": ALGORITHM, "schema-version": 1,
                               "bank-id": bank, "server-round": round_number,
                               "base-sha256": base_hash, "payload-kind": "query-gradient"}),
    })
    return Message(content=payload, reply_to=message)


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    bank, config, state, spec = _input(message, context)
    root = Path(os.environ["FRAUD_PROJECT_ROOT"]).resolve()
    model = FraudMLP(spec)
    model.load_state_dict(state)
    x, y = read_split(root, str(config["clients-dir"]), bank, str(config["setting"]), "validation")
    scores = score_model(model, x)
    with torch.no_grad():
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(torch.from_numpy(x)), torch.from_numpy(y)).item()
    payload = RecordDict({
        "metrics": MetricRecord({"num-examples": len(y), "validation-loss": loss,
                                 "validation-average-precision": float(average_precision_score(y, scores))}),
        "audit": ConfigRecord({"algorithm": ALGORITHM, "schema-version": 1,
                               "bank-id": bank, "server-round": int(config["server-round"]),
                               "base-sha256": state_fingerprint(state)}),
    })
    return Message(content=payload, reply_to=message)
