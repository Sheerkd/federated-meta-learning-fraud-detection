"""Native Windows Flower transport diagnostic, not a fraud experiment.

Uses the retained legacy gRPC API only to verify real process communication.
Production training will use ClientApp/ServerApp infrastructure in the federation milestone.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def serve(address: str, output: Path) -> None:
    import flwr as fl
    import numpy as np

    class CheckedFedAvg(fl.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            if failures or len(results) != 4:
                raise RuntimeError("Expected four successful clients")
            parameters, metrics = super().aggregate_fit(server_round, results, failures)
            actual = fl.common.parameters_to_ndarrays(parameters)[0]
            # Four clients return constant values 1..4 with weights 1..4.
            np.testing.assert_allclose(actual, [3.0], atol=1e-6)
            output.write_text(json.dumps({"status": "passed", "clients": 4, "rounds": 1,
                                         "aggregate": actual.tolist(), "expected": [3.0],
                                         "data": "synthetic constants; no fraud model"}, indent=2), encoding="utf-8")
            return parameters, metrics

    strategy = CheckedFedAvg(min_fit_clients=4, min_available_clients=4,
                             fraction_evaluate=0.0, accept_failures=False,
                             initial_parameters=fl.common.ndarrays_to_parameters([np.zeros(1, dtype=np.float32)]))
    fl.server.start_server(server_address=address, config=fl.server.ServerConfig(num_rounds=1, round_timeout=60), strategy=strategy)


def connect(address: str, client_id: int) -> None:
    import flwr as fl
    import numpy as np

    class ProbeClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [np.zeros(1, dtype=np.float32)]

        def fit(self, parameters, config):
            return [np.array([float(client_id)], dtype=np.float32)], client_id, {}

        def evaluate(self, parameters, config):
            return 0.0, client_id, {}

    fl.client.start_client(server_address=address, client=ProbeClient().to_client(),
                           insecure=True, max_wait_time=60)


def run_smoke(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "result.json"
    output.unlink(missing_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    address = f"127.0.0.1:{port}"
    processes = []
    streams = []
    env = dict(os.environ, FLWR_TELEMETRY_ENABLED="0")
    deadline = time.monotonic() + 120
    try:
        for role, client_id in [("server", 0)] + [("client", i) for i in range(1, 5)]:
            stream = (output_dir / f"{role}_{client_id}.log").open("w", encoding="utf-8")
            streams.append(stream)
            process = subprocess.Popen(
                [sys.executable, "-m", "fraud_detection.federated.smoke", role,
                 "--address", address, "--client-id", str(client_id), "--output", str(output)],
                stdout=stream, stderr=subprocess.STDOUT, env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            processes.append(process)
        for process in processes:
            process.wait(timeout=max(1, deadline - time.monotonic()))
            if process.returncode != 0:
                raise RuntimeError(f"Flower process failed; inspect {output_dir}")
        if not output.exists():
            raise RuntimeError(f"Flower did not complete aggregation; inspect {output_dir}")
        return json.loads(output.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Flower smoke timed out; inspect {output_dir}") from exc
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["server", "client"])
    parser.add_argument("--address", required=True)
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.role == "server":
        serve(args.address, args.output)
    else:
        connect(args.address, args.client_id)
