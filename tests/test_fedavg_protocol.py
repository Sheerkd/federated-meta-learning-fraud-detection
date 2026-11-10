import json
import time

import numpy as np
import pytest
import torch
from flwr.app import Message, Metadata

from fraud_detection.data.pipeline import BANKS
from fraud_detection.federated.local_training import train_local
from fraud_detection.federated.protocol import reply_payload, state_fingerprint, validate_state
from fraud_detection.federated.server_app import StrictFedAvg
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec


def _reply(bank, value, count, base):
    model = FraudMLP(ModelSpec())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)
    content = reply_payload(model, bank=bank, round_number=1, base_hash=base,
                            num_examples=count, train_loss=float(value))
    metadata = Metadata(run_id=1, message_id=f"reply-{bank}", src_node_id=BANKS.index(bank) + 1,
                        dst_node_id=0, reply_to_message_id="request", group_id="1",
                        created_at=time.time(), ttl=60, message_type="train")
    return Message(content=content, metadata=metadata)


def test_four_client_weighted_aggregation_and_audit(tmp_path):
    base = state_fingerprint(FraudMLP(ModelSpec()).state_dict())
    strategy = StrictFedAvg(run_dir=tmp_path, model_spec=ModelSpec(),
                            expected_train_counts={bank: index for index, bank in enumerate(BANKS, 1)},
                            expected_valid_counts={bank: 2 for bank in BANKS})
    strategy.base_hashes[1] = base
    strategy.round_times[1] = 0.0
    replies = [_reply(bank, float(index), index, base) for index, bank in enumerate(BANKS, 1)]
    assert all(set(message.content) == {"arrays", "metrics", "audit"} for message in replies)
    aggregated, _ = strategy.aggregate_train(1, replies)
    state = aggregated.to_torch_state_dict()
    validate_state(state, ModelSpec())
    assert all(torch.allclose(tensor, torch.full_like(tensor, 3.0)) for tensor in state.values())
    report = json.loads((tmp_path / "round_1.json").read_text())
    assert report["aggregation_verified"] and report["train_weight_sum"] == 10
    assert report["train_update_tensor_bytes"] > 0


def test_duplicate_or_stale_client_reply_is_rejected(tmp_path):
    base = state_fingerprint(FraudMLP(ModelSpec()).state_dict())
    strategy = StrictFedAvg(run_dir=tmp_path, model_spec=ModelSpec(),
                            expected_train_counts={bank: 1 for bank in BANKS},
                            expected_valid_counts={bank: 1 for bank in BANKS})
    strategy.base_hashes[1] = base
    replies = [_reply(bank, 1.0, 1, base) for bank in BANKS]
    replies[3].content["audit"]["bank-id"] = "bank_a"
    with pytest.raises(ValueError, match="Duplicate"):
        strategy.aggregate_train(1, replies)


def test_local_update_uses_training_rows_and_changes_model():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(64, 30)).astype(np.float32)
    y = np.array([0] * 48 + [1] * 16, dtype=np.float32)
    model = FraudMLP(ModelSpec())
    before = [value.clone() for value in model.parameters()]
    loss = train_local(model, x, y, epochs=1, batch_size=16,
                       learning_rate=0.001, weight_decay=0.0001,
                       max_grad_norm=5.0, seed=42)
    assert np.isfinite(loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
