"""Flower ServerApp with strict four-bank FedAvg and auditable round artifacts."""

from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from flwr.app import ArrayRecord, ConfigRecord, Context, Message
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import torch

from fraud_detection.config import load_config
from fraud_detection.data.pipeline import BANKS, file_hash
from fraud_detection.federated.protocol import ALGORITHM, SCHEMA_VERSION, model_bytes, state_fingerprint, validate_state
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec


app = ServerApp()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _save_weights(path: Path, state: dict[str, torch.Tensor]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    return file_hash(path)


class StrictFedAvg(FedAvg):
    """Reject missing, duplicated, stale, or malformed client updates."""

    def __init__(self, *, run_dir: Path, model_spec: ModelSpec,
                 expected_train_counts: dict[str, int], expected_valid_counts: dict[str, int]):
        super().__init__(fraction_train=1.0, fraction_evaluate=1.0,
                         min_train_nodes=4, min_evaluate_nodes=4,
                         min_available_nodes=4, weighted_by_key="num-examples")
        self.run_dir = run_dir
        self.model_spec = model_spec
        self.expected_train_counts = expected_train_counts
        self.expected_valid_counts = expected_valid_counts
        self.base_hashes: dict[int, str] = {}
        self.round_times: dict[int, float] = {}
        self.completed_rounds: list[int] = []

    def configure_train(self, server_round: int, arrays: ArrayRecord,
                        config: ConfigRecord, grid: Grid):
        state = arrays.to_torch_state_dict()
        validate_state(state, self.model_spec)
        base_hash = state_fingerprint(state)
        config["base-sha256"] = base_hash
        self.base_hashes[server_round] = base_hash
        self.round_times[server_round] = time.monotonic()
        messages = list(super().configure_train(server_round, arrays, config, grid))
        if len(messages) != 4:
            raise RuntimeError("FedAvg requires exactly four client messages")
        return messages

    def aggregate_train(self, server_round: int, replies):
        received = list(replies)
        if len(received) != 4 or any(message.has_error() for message in received):
            raise RuntimeError(f"Round {server_round}: expected four successful client updates")
        observations = {}
        received_states = {}
        expected_base = self.base_hashes[server_round]
        for message in received:
            content = message.content
            if set(content) != {"arrays", "metrics", "audit"}:
                raise ValueError("Client reply contains unexpected records")
            audit = content["audit"]
            bank = str(audit["bank-id"])
            if (bank not in BANKS or bank in observations or audit["algorithm"] != ALGORITHM
                    or audit["schema-version"] != SCHEMA_VERSION
                    or audit["server-round"] != server_round
                    or audit["base-sha256"] != expected_base):
                raise ValueError("Duplicate, stale, or incompatible client update")
            metrics = content["metrics"]
            examples = metrics["num-examples"]
            loss = metrics["train-loss"]
            if examples != self.expected_train_counts[bank] or not math.isfinite(loss):
                raise ValueError(f"Invalid aggregate client metrics: {bank}")
            state = content["arrays"].to_torch_state_dict()
            validate_state(state, self.model_spec)
            received_states[bank] = state
            observations[bank] = {"train_rows": int(examples),
                                  "train_loss": float(loss),
                                  "update_bytes": model_bytes(state),
                                  "update_sha256": state_fingerprint(state)}
        aggregated, metrics = super().aggregate_train(server_round, received)
        if aggregated is None:
            raise RuntimeError("FedAvg produced no aggregate")
        state = aggregated.to_torch_state_dict()
        validate_state(state, self.model_spec)
        total_examples = sum(self.expected_train_counts.values())
        for name, tensor in state.items():
            independently_weighted = sum(
                received_states[bank][name].to(torch.float64) * self.expected_train_counts[bank]
                for bank in BANKS
            ) / total_examples
            torch.testing.assert_close(tensor, independently_weighted.to(tensor.dtype),
                                       rtol=1e-5, atol=1e-7)
        checkpoint = self.run_dir / f"round_{server_round}.pt"
        checkpoint_sha = _save_weights(checkpoint, state)
        payload_bytes = model_bytes(state)
        round_report = {
            "round": server_round, "status": "trained", "clients": observations,
            "base_sha256": expected_base, "aggregate_sha256": state_fingerprint(state),
            "checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_sha,
            "weighted_by": "original local training rows",
            "train_weight_sum": sum(value["train_rows"] for value in observations.values()),
            "aggregation_verified": True,
            "train_broadcast_tensor_bytes": 4 * payload_bytes,
            "train_update_tensor_bytes": sum(value["update_bytes"] for value in observations.values()),
            "round_train_seconds": time.monotonic() - self.round_times[server_round],
        }
        _write_json(self.run_dir / f"round_{server_round}.json", round_report)
        self.completed_rounds.append(server_round)
        return aggregated, metrics

    def configure_evaluate(self, server_round: int, arrays: ArrayRecord,
                           config: ConfigRecord, grid: Grid):
        messages = list(super().configure_evaluate(server_round, arrays, config, grid))
        if len(messages) != 4:
            raise RuntimeError("FedAvg requires exactly four evaluation messages")
        return messages

    def aggregate_evaluate(self, server_round: int, replies):
        received = list(replies)
        if len(received) != 4 or any(message.has_error() for message in received):
            raise RuntimeError(f"Round {server_round}: expected four successful validation replies")
        observations = {}
        for message in received:
            content = message.content
            if set(content) != {"metrics", "audit"}:
                raise ValueError("Validation reply contains unexpected records")
            audit = content["audit"]
            bank = str(audit["bank-id"])
            if (bank not in BANKS or bank in observations or audit["algorithm"] != ALGORITHM
                    or audit["schema-version"] != SCHEMA_VERSION
                    or audit["server-round"] != server_round):
                raise ValueError("Duplicate, stale, or incompatible validation reply")
            metrics = content["metrics"]
            if metrics["num-examples"] != self.expected_valid_counts[bank]:
                raise ValueError(f"Validation row count mismatch: {bank}")
            loss = float(metrics["validation-loss"])
            ap = float(metrics["validation-average-precision"])
            if not math.isfinite(loss) or not 0 <= ap <= 1:
                raise ValueError(f"Invalid validation metrics: {bank}")
            observations[bank] = {"validation_rows": int(metrics["num-examples"]),
                                  "validation_loss": loss, "validation_average_precision": ap}
        aggregate = super().aggregate_evaluate(server_round, received)
        round_path = self.run_dir / f"round_{server_round}.json"
        round_report = json.loads(round_path.read_text(encoding="utf-8"))
        round_report["validation"] = observations
        round_report["status"] = "completed"
        round_report["validation_broadcast_tensor_bytes"] = 4 * model_bytes(
            torch.load(self.run_dir / round_report["checkpoint"], map_location="cpu", weights_only=True))
        round_report["round_total_seconds"] = time.monotonic() - self.round_times[server_round]
        _write_json(round_path, round_report)
        return aggregate


@app.main()
def main(grid: Grid, context: Context) -> None:
    root = Path(os.environ["FRAUD_PROJECT_ROOT"]).resolve()
    app_config = load_config(root / "configs" / "default.toml")
    run_config = context.run_config
    setting = str(run_config["setting"])
    if setting not in ("mild", "strong"):
        raise ValueError("Unknown data setting")
    seed = int(run_config["seed"])
    rounds = int(run_config["num-server-rounds"])
    if rounds < 2 or rounds > 100:
        raise ValueError("Federated run requires 2 to 100 rounds")
    partition_path = root / app_config.paths.data / "processed" / setting / "metadata.json"
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    if file_hash(partition_path.parent / "manifest.parquet") != partition["manifest_sha256"]:
        raise ValueError("Partition manifest checksum changed")
    run_dir = root / app_config.paths.runs / "federated" / setting / f"seed_{seed}"
    if run_dir.exists():
        raise FileExistsError(f"Federated run already exists and cannot be resumed automatically: {run_dir}")
    run_dir.mkdir(parents=True)
    status = {
        "schema_version": 1, "status": "training", "algorithm": ALGORITHM,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "setting": setting, "seed": seed, "rounds_requested": rounds,
        "partition_manifest_sha256": partition["manifest_sha256"],
        "dataset_sha256": partition["dataset"]["sha256"],
        "expected_clients": list(BANKS),
        "run_config": dict(run_config),
        "flower_version": importlib.metadata.version("flwr"),
        "torch_version": importlib.metadata.version("torch"),
        "transport": "localhost plaintext for simulated research deployment",
        "resume_policy": "Refuse existing run directories; incomplete runs require explicit investigation",
    }
    _write_json(run_dir / "run.json", status)
    try:
        torch.manual_seed(seed)
        model_spec = ModelSpec()
        initial = FraudMLP(model_spec)
        initial_state = initial.state_dict()
        validate_state(initial_state, model_spec)
        status["model_spec"] = model_spec.as_dict()
        status["initial_checkpoint_sha256"] = _save_weights(run_dir / "round_0.pt", initial_state)
        status["initial_model_sha256"] = state_fingerprint(initial_state)
        _write_json(run_dir / "run.json", status)
        expected_train = {bank: partition["partition"]["counts"][bank]["train"]["rows"] for bank in BANKS}
        expected_valid = {bank: partition["partition"]["counts"][bank]["validation"]["rows"] for bank in BANKS}
        strategy = StrictFedAvg(run_dir=run_dir, model_spec=model_spec,
                                expected_train_counts=expected_train,
                                expected_valid_counts=expected_valid)
        train_config = ConfigRecord({
            "algorithm": ALGORITHM, "schema-version": SCHEMA_VERSION,
            "setting": setting, "seed": seed,
            "clients-dir": app_config.paths.clients,
            "local-epochs": int(run_config["local-epochs"]),
            "batch-size": int(run_config["batch-size"]),
            "learning-rate": float(run_config["learning-rate"]),
            "weight-decay": float(run_config["weight-decay"]),
            "max-grad-norm": float(run_config["max-grad-norm"]),
        })
        eval_config = ConfigRecord({"algorithm": ALGORITHM, "schema-version": SCHEMA_VERSION,
                                    "setting": setting, "clients-dir": app_config.paths.clients})
        result = strategy.start(grid=grid, initial_arrays=ArrayRecord(initial_state),
                                train_config=train_config, evaluate_config=eval_config,
                                num_rounds=rounds, timeout=180)
        if strategy.completed_rounds != list(range(1, rounds + 1)):
            raise RuntimeError("Federated run did not complete every requested round")
        final_state = result.arrays.to_torch_state_dict()
        validate_state(final_state, model_spec)
        last = json.loads((run_dir / f"round_{rounds}.json").read_text(encoding="utf-8"))
        if state_fingerprint(final_state) != last["aggregate_sha256"]:
            raise RuntimeError("Final Flower result differs from last checkpoint")
        status.update({"status": "federated_complete", "rounds_completed": rounds,
                       "final_model_sha256": last["aggregate_sha256"],
                       "final_checkpoint": last["checkpoint"],
                       "completed_utc": datetime.now(timezone.utc).isoformat()})
        _write_json(run_dir / "run.json", status)
    except Exception as exc:
        status.update({"status": "failed", "error": str(exc),
                       "failed_utc": datetime.now(timezone.utc).isoformat()})
        _write_json(run_dir / "run.json", status)
        raise
