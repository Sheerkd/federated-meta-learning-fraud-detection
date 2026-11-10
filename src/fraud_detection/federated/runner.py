"""Launch a real local Flower deployment with four SuperNodes on Windows."""

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from fraud_detection.config import load_config
from fraud_detection.federated.evaluate import evaluate_federated


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_port(port: int, process: subprocess.Popen, *, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Flower process exited before port {port} was ready")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"Flower port {port} did not become ready")


def _stop_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=15, check=False)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_local_deployment(project_root: Path, *, setting: str, seed: int,
                         method: str = "fedavg") -> Path:
    if method not in ("fedavg", "fedmaml"):
        raise ValueError("Unknown federated method")
    root = project_root.resolve()
    config = load_config(root / "configs" / "default.toml")
    run_dir = root / config.paths.runs / ("federated" if method == "fedavg" else "meta") / setting / f"seed_{seed}"
    if run_dir.exists():
        raise FileExistsError(f"Federated run already exists; inspect it before a new run: {run_dir}")
    runtime = root / config.paths.runs / "flower_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    app_dir = runtime / ("fraud-flower" if method == "fedavg" else "fraud-meta-flower")
    # Flower rejects the user's workspace name because it contains a space.
    # Stage only source code and pyproject under a valid ignored folder name.
    if not app_dir.resolve().is_relative_to(runtime.resolve()):
        raise ValueError("Flower package staging escaped the runtime directory")
    if app_dir.exists():
        shutil.rmtree(app_dir)
    app_dir.mkdir()
    project_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    if method == "fedmaml":
        project_text = project_text.replace(
            'serverapp = "fraud_detection.federated.server_app:app"',
            'serverapp = "fraud_detection.meta.server_app:app"').replace(
            'clientapp = "fraud_detection.federated.client_app:app"',
            'clientapp = "fraud_detection.meta.client_app:app"')
        project_text += '\nsupport-size = 32\nquery-size = 32\ninner-steps = 1\ninner-lr = 0.01\nouter-lr = 0.001\n'
    (app_dir / "pyproject.toml").write_text(project_text, encoding="utf-8")
    shutil.copytree(root / "src" / "fraud_detection", app_dir / "src" / "fraud_detection",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    ports = set()
    while len(ports) < 5:
        ports.add(_free_port())
    superlink_port, *node_ports = list(ports)
    (runtime / "config.toml").write_text(
        "[superlink.local-deployment]\n"
        f'address = "127.0.0.1:{superlink_port}"\n'
        "insecure = true\n", encoding="utf-8")
    env = dict(os.environ)
    env.update({"FLWR_HOME": str(runtime), "FRAUD_PROJECT_ROOT": str(root),
                "FLWR_TELEMETRY_ENABLED": "0", "PYTHONIOENCODING": "utf-8"})
    env["PYTHONPATH"] = str(app_dir / "src") + os.pathsep + env.get("PYTHONPATH", "")
    scripts = Path(sys.executable).parent
    env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
    processes = []
    streams = []
    log_dir = runtime / "logs" / f"{method}_{setting}_seed_{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    def start(name: str, arguments: list[str]) -> subprocess.Popen:
        stream = (log_dir / f"{name}.log").open("w", encoding="utf-8")
        streams.append(stream)
        process = subprocess.Popen(arguments, cwd=root, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, creationflags=flags)
        processes.append(process)
        return process

    try:
        superlink = start("superlink", [str(scripts / "flower-superlink.exe"),
                                          "--insecure", "--host", "127.0.0.1",
                                          "--port", str(superlink_port),
                                          "--disable-runtime-dependency-installation"])
        _wait_for_port(superlink_port, superlink)
        for index, bank in enumerate(config.client_ids):
            node = start(bank, [str(scripts / "flower-supernode.exe"), "--insecure",
                                "--superlink", f"127.0.0.1:{superlink_port}",
                                "--host", "127.0.0.1", "--port", str(node_ports[index]),
                                "--node-config", f"bank-id='{bank}'"])
            _wait_for_port(node_ports[index], node)
        time.sleep(2)
        cli_log = log_dir / "flwr_run.log"
        with cli_log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                [str(scripts / "flwr.exe"), "run", str(app_dir), "local-deployment", "--stream",
                 "--run-config", f'setting="{setting}" seed={seed}'],
                cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=900, creationflags=flags, check=False)
        if completed.returncode != 0:
            tail = cli_log.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(f"Flower run failed (exit {completed.returncode}). Log: {cli_log}\n{tail}")
        status_path = run_dir / "run.json"
        if not status_path.is_file():
            raise RuntimeError(f"Flower returned without server result; inspect {cli_log}")
        import json
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["status"] != "federated_complete":
            raise RuntimeError(f"Flower training did not complete: {status.get('error', status['status'])}")
        if method == "fedmaml":
            from fraud_detection.meta.evaluate import evaluate_meta
            return evaluate_meta(root, setting=setting, seed=seed,
                                 clients_dir=config.paths.clients, data_dir=config.paths.data,
                                 runs_dir=config.paths.runs)
        return evaluate_federated(root, setting=setting, seed=seed,
                                  clients_dir=config.paths.clients, data_dir=config.paths.data,
                                  runs_dir=config.paths.runs)
    finally:
        for process in reversed(processes):
            _stop_tree(process)
        for stream in streams:
            stream.close()
