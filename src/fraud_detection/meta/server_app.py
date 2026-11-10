"""Strict four-bank first-order MAML Flower ServerApp."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import torch

from fraud_detection.config import load_config
from fraud_detection.data.pipeline import BANKS, file_hash
from fraud_detection.federated.protocol import model_bytes, state_fingerprint, validate_state
from fraud_detection.federated.server_app import _save_weights, _write_json
from fraud_detection.meta.learning import ALGORITHM, outer_update
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec


app = ServerApp()


class StrictMetaStrategy(FedAvg):
    def __init__(self, *, run_dir: Path, spec: ModelSpec, train_counts: dict,
                 valid_counts: dict, support_size: int, query_size: int, beta: float):
        super().__init__(fraction_train=1.0, fraction_evaluate=1.0,
                         min_train_nodes=4, min_evaluate_nodes=4, min_available_nodes=4,
                         weighted_by_key="num-examples")
        self.run_dir, self.spec = run_dir, spec
        self.train_counts, self.valid_counts = train_counts, valid_counts
        self.support_size, self.query_size, self.beta = support_size, query_size, beta
        self.base_states = {}
        self.round_times = {}
        self.completed = []

    def configure_train(self, server_round, arrays, config, grid):
        state = arrays.to_torch_state_dict()
        validate_state(state, self.spec)
        self.base_states[server_round] = state
        config["base-sha256"] = state_fingerprint(state)
        self.round_times[server_round] = time.monotonic()
        messages = list(super().configure_train(server_round, arrays, config, grid))
        if len(messages) != 4:
            raise RuntimeError("Meta round requires exactly four banks")
        return messages

    def aggregate_train(self, server_round, replies):
        received = list(replies)
        if len(received) != 4 or any(reply.has_error() for reply in received):
            raise RuntimeError("Meta round missing successful bank replies")
        base = self.base_states[server_round]
        base_hash = state_fingerprint(base)
        gradients, observations = {}, {}
        for reply in received:
            content = reply.content
            if set(content) != {"arrays", "metrics", "audit"}:
                raise ValueError("Unexpected meta reply records")
            audit = content["audit"]
            bank = str(audit["bank-id"])
            if (bank not in BANKS or bank in gradients or audit["algorithm"] != ALGORITHM
                    or audit["schema-version"] != 1 or audit["server-round"] != server_round
                    or audit["base-sha256"] != base_hash
                    or audit["payload-kind"] != "query-gradient"):
                raise ValueError("Duplicate, stale, or incompatible meta gradient")
            metrics = content["metrics"]
            if (metrics["num-examples"] != self.query_size
                    or metrics["support-examples"] != self.support_size
                    or self.train_counts[bank] < self.support_size + self.query_size
                    or not math.isfinite(float(metrics["query-loss"]))):
                raise ValueError("Invalid episode metrics")
            gradient = content["arrays"].to_torch_state_dict()
            validate_state(gradient, self.spec)
            gradients[bank] = gradient
            observations[bank] = {"query_loss": float(metrics["query-loss"]),
                                  "support_examples": self.support_size,
                                  "query_examples": self.query_size,
                                  "gradient_sha256": state_fingerprint(gradient)}
        if set(gradients) != set(BANKS):
            raise ValueError("Missing bank gradient")
        state = outer_update(base, gradients, beta=self.beta, spec=self.spec)
        checkpoint = self.run_dir / f"round_{server_round}.pt"
        checkpoint_hash = _save_weights(checkpoint, state)
        report = {"round": server_round, "status": "trained", "clients": observations,
                  "algorithm": ALGORITHM, "base_sha256": base_hash,
                  "aggregate_sha256": state_fingerprint(state),
                  "checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_hash,
                  "outer_rule": "theta minus beta times unweighted mean of four query gradients",
                  "beta": self.beta, "aggregation_verified": True,
                  "train_broadcast_tensor_bytes": 4 * model_bytes(base),
                  "gradient_update_tensor_bytes": 4 * model_bytes(base),
                  "round_train_seconds": time.monotonic() - self.round_times[server_round]}
        _write_json(self.run_dir / f"round_{server_round}.json", report)
        self.completed.append(server_round)
        return ArrayRecord(state), MetricRecord({"query-loss": sum(
            item["query_loss"] for item in observations.values()) / 4})

    def configure_evaluate(self, server_round, arrays, config, grid):
        config["base-sha256"] = state_fingerprint(arrays.to_torch_state_dict())
        messages = list(super().configure_evaluate(server_round, arrays, config, grid))
        if len(messages) != 4:
            raise RuntimeError("Meta validation requires four banks")
        return messages

    def aggregate_evaluate(self, server_round, replies):
        received = list(replies)
        if len(received) != 4 or any(reply.has_error() for reply in received):
            raise RuntimeError("Meta validation missing bank replies")
        report_path = self.run_dir / f"round_{server_round}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        observed = {}
        for reply in received:
            if set(reply.content) != {"metrics", "audit"}:
                raise ValueError("Unexpected meta validation reply")
            audit, metrics = reply.content["audit"], reply.content["metrics"]
            bank = str(audit["bank-id"])
            if (bank not in BANKS or bank in observed or audit["algorithm"] != ALGORITHM
                    or audit["server-round"] != server_round
                    or audit["base-sha256"] != report["aggregate_sha256"]
                    or metrics["num-examples"] != self.valid_counts[bank]):
                raise ValueError("Invalid meta validation reply")
            loss, ap = float(metrics["validation-loss"]), float(metrics["validation-average-precision"])
            if not math.isfinite(loss) or not 0 <= ap <= 1:
                raise ValueError("Nonfinite validation metrics")
            observed[bank] = {"validation_rows": int(metrics["num-examples"]),
                              "validation_loss": loss, "validation_average_precision": ap}
        report["validation"] = observed
        report["status"] = "completed"
        report["validation_broadcast_tensor_bytes"] = 4 * model_bytes(
            torch.load(self.run_dir / report["checkpoint"], weights_only=True))
        report["round_total_seconds"] = time.monotonic() - self.round_times[server_round]
        _write_json(report_path, report)
        return super().aggregate_evaluate(server_round, received)


@app.main()
def main(grid: Grid, context: Context) -> None:
    root = Path(os.environ["FRAUD_PROJECT_ROOT"]).resolve()
    config = load_config(root / "configs/default.toml")
    setting = str(context.run_config["setting"])
    if setting not in ("mild", "strong"):
        raise ValueError("Invalid setting")
    seed = int(context.run_config["seed"])
    rounds = int(context.run_config["num-server-rounds"])
    support_size = int(context.run_config["support-size"])
    query_size = int(context.run_config["query-size"])
    steps = int(context.run_config["inner-steps"])
    alpha = float(context.run_config["inner-lr"])
    beta = float(context.run_config["outer-lr"])
    if not (2 <= rounds <= 100 and 0 < alpha <= 1 and 0 < beta <= 1
            and 1 <= steps <= 20 and 2 <= support_size <= 512
            and 2 <= query_size <= 512 and support_size % 2 == query_size % 2 == 0):
        raise ValueError("Invalid meta optimization settings")
    partition_path = root / config.paths.data / "processed" / setting / "metadata.json"
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    if file_hash(partition_path.parent / "manifest.parquet") != partition["manifest_sha256"]:
        raise ValueError("Partition manifest changed")
    run_dir = root / config.paths.runs / "meta" / setting / f"seed_{seed}"
    if run_dir.exists():
        raise FileExistsError("Meta run already exists; inspect before retrying")
    run_dir.mkdir(parents=True)
    status = {"schema_version": 1, "status": "training", "algorithm": ALGORITHM,
              "setting": setting, "seed": seed, "rounds_requested": rounds,
              "partition_manifest_sha256": partition["manifest_sha256"],
              "dataset_sha256": partition["dataset"]["sha256"],
              "torch_num_threads": torch.get_num_threads(),
              "run_config": dict(context.run_config),
              "started_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(run_dir / "run.json", status)
    try:
        torch.manual_seed(seed)
        spec = ModelSpec()
        initial = FraudMLP(spec).state_dict()
        validate_state(initial, spec)
        status["model_spec"] = spec.as_dict()
        status["initial_checkpoint_sha256"] = _save_weights(run_dir / "round_0.pt", initial)
        status["initial_model_sha256"] = state_fingerprint(initial)
        _write_json(run_dir / "run.json", status)
        counts = partition["partition"]["counts"]
        strategy = StrictMetaStrategy(
            run_dir=run_dir, spec=spec,
            train_counts={bank: counts[bank]["train"]["rows"] for bank in BANKS},
            valid_counts={bank: counts[bank]["validation"]["rows"] for bank in BANKS},
            support_size=support_size, query_size=query_size, beta=beta)
        train_config = ConfigRecord({"algorithm": ALGORITHM, "schema-version": 1,
                                     "setting": setting, "seed": seed,
                                     "clients-dir": config.paths.clients,
                                     "support-size": support_size, "query-size": query_size,
                                     "inner-steps": steps, "inner-lr": alpha})
        eval_config = ConfigRecord({"algorithm": ALGORITHM, "schema-version": 1,
                                    "setting": setting, "clients-dir": config.paths.clients})
        result = strategy.start(grid=grid, initial_arrays=ArrayRecord(initial),
                                train_config=train_config, evaluate_config=eval_config,
                                num_rounds=rounds, timeout=180)
        if strategy.completed != list(range(1, rounds + 1)):
            raise RuntimeError("Incomplete meta rounds")
        final = result.arrays.to_torch_state_dict()
        final_report = json.loads((run_dir / f"round_{rounds}.json").read_text(encoding="utf-8"))
        if state_fingerprint(final) != final_report["aggregate_sha256"]:
            raise RuntimeError("Final model mismatch")
        status.update({"status": "federated_complete", "rounds_completed": rounds,
                       "final_model_sha256": final_report["aggregate_sha256"],
                       "final_checkpoint": final_report["checkpoint"],
                       "completed_utc": datetime.now(timezone.utc).isoformat()})
        _write_json(run_dir / "run.json", status)
    except Exception as exc:
        status.update({"status": "failed", "error": str(exc)})
        _write_json(run_dir / "run.json", status)
        raise
