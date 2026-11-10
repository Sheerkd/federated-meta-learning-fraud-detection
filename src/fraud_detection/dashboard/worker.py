"""Detached training worker with persistent completion and failure status."""

import argparse
import json
import os
from pathlib import Path

from fraud_detection.federated.runner import run_local_deployment


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=["FedAvg", "FedMAML"], required=True)
    parser.add_argument("--setting", choices=["mild", "strong"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    job_path = args.job_dir / "job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if (job["method"] != args.method or job["setting"] != args.setting
            or job["seed"] != args.seed):
        raise ValueError("Background job parameters changed")
    job.update({"status": "running", "pid": os.getpid()})
    _write(job_path, job)
    try:
        result = run_local_deployment(args.root, setting=args.setting, seed=args.seed,
                                      method="fedavg" if args.method == "FedAvg" else "fedmaml")
        job.update({"status": "completed", "result": str(result)})
        _write(job_path, job)
        return 0
    except Exception as exc:
        job.update({"status": "failed", "error": str(exc)})
        _write(job_path, job)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
