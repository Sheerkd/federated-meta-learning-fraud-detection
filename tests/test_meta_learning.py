import numpy as np
import pytest
import torch
import time
from flwr.app import ArrayRecord, ConfigRecord, Message, Metadata, MetricRecord, RecordDict

from fraud_detection.data.pipeline import BANKS
from fraud_detection.federated.protocol import state_fingerprint
from fraud_detection.meta.learning import episode_indices, outer_update, query_gradient
from fraud_detection.meta.server_app import StrictMetaStrategy
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec


def test_disjoint_balanced_episode_and_immutable_base():
    labels = np.array([0] * 30 + [1] * 30, dtype=np.float32)
    support, query = episode_indices(labels, support_size=12, query_size=10, seed=7)
    assert not set(support) & set(query)
    assert labels[support].sum() == 6
    assert labels[query].sum() == 5
    spec = ModelSpec(input_dim=30, dropout=0)
    state = FraudMLP(spec).state_dict()
    before = {name: value.clone() for name, value in state.items()}
    features = np.random.default_rng(1).normal(size=(60, 30)).astype(np.float32)
    gradient, loss = query_gradient(state, spec, features, labels, support, query,
                                    steps=2, inner_lr=0.01)
    assert np.isfinite(loss)
    assert all(torch.equal(state[name], before[name]) for name in state)
    assert any(torch.count_nonzero(tensor) for tensor in gradient.values())
    with pytest.raises(ValueError, match="overlap"):
        query_gradient(state, spec, features, labels, support, support,
                       steps=1, inner_lr=0.01)


def test_outer_update_matches_manual_mean_and_rejects_bad_gradient():
    spec = ModelSpec()
    state = FraudMLP(spec).state_dict()
    one = {name: torch.ones_like(tensor) for name, tensor in state.items()}
    three = {name: torch.full_like(tensor, 3) for name, tensor in state.items()}
    result = outer_update(state, {"a": one, "b": three}, beta=0.1, spec=spec)
    for name in state:
        torch.testing.assert_close(result[name], state[name] - 0.2, rtol=0, atol=1e-7)
    bad = dict(one)
    bad[next(iter(bad))] = torch.full_like(next(iter(bad.values())), float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        outer_update(state, {"bad": bad}, beta=0.1, spec=spec)


def _meta_reply(bank, base_hash, gradient_value=1.0, round_number=1):
    spec = ModelSpec()
    state = FraudMLP(spec).state_dict()
    gradient = {name: torch.full_like(value, gradient_value) for name, value in state.items()}
    content = RecordDict({
        "arrays": ArrayRecord(gradient),
        "metrics": MetricRecord({"num-examples": 4, "support-examples": 4, "query-loss": 0.5}),
        "audit": ConfigRecord({"algorithm": "fedmaml", "schema-version": 1,
                               "bank-id": bank, "server-round": round_number,
                               "base-sha256": base_hash, "payload-kind": "query-gradient"}),
    })
    metadata = Metadata(run_id=1, message_id=f"meta-{bank}", src_node_id=BANKS.index(bank) + 1,
                        dst_node_id=0, reply_to_message_id="request", group_id="1",
                        created_at=time.time(), ttl=60, message_type="train")
    return Message(content=content, metadata=metadata)


def test_meta_server_applies_equal_task_gradients_and_rejects_stale(tmp_path):
    spec = ModelSpec()
    base = FraudMLP(spec).state_dict()
    strategy = StrictMetaStrategy(run_dir=tmp_path, spec=spec,
                                  train_counts={bank: 20 for bank in BANKS},
                                  valid_counts={bank: 10 for bank in BANKS},
                                  support_size=4, query_size=4, beta=0.1)
    strategy.base_states[1] = base
    strategy.round_times[1] = time.monotonic()
    replies = [_meta_reply(bank, state_fingerprint(base), value)
               for bank, value in zip(BANKS, (1, 2, 3, 4))]
    aggregated, _ = strategy.aggregate_train(1, replies)
    for name, value in aggregated.to_torch_state_dict().items():
        torch.testing.assert_close(value, base[name] - 0.25, atol=1e-7, rtol=0)
    strategy.base_states[2] = base
    replies[0].content["audit"]["server-round"] = 2
    with pytest.raises(ValueError, match="stale"):
        strategy.aggregate_train(2, replies)
