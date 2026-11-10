"""Command line entry points for the integrated research application."""

import argparse
import json
import logging
from pathlib import Path

from fraud_detection.config import initialize_storage, load_config
from fraud_detection.logging_utils import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Personalized fraud detection research tools")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project directory")
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--dataset", type=Path, help="ULB creditcard.csv path for data commands")
    parser.add_argument("--source-url", default="local file", help="Recorded data source URL")
    parser.add_argument("--setting", choices=["mild", "strong", "all"], default="all",
                        help="Dataset setting for training and verification")
    parser.add_argument("--bank", choices=["bank_a", "bank_b", "bank_c", "bank_d"])
    parser.add_argument("--row-id", type=int, help="Bank-local test row ID; defaults to first fraud row")
    parser.add_argument("--method", choices=["fedmaml", "fedavg"], default="fedmaml")
    parser.add_argument("--support-budget", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--background-size", type=int, default=32)
    parser.add_argument("--shap-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, help="Override model seed for a new federated run")
    parser.add_argument("command", choices=["init", "check-config", "doctor", "flower-smoke", "dashboard", "prepare-data", "verify-data", "train-baselines", "verify-baselines", "run-fedavg", "evaluate-fedavg", "verify-fedavg", "run-fedmaml", "evaluate-fedmaml", "verify-fedmaml", "explain", "verify-explanations", "build-study", "verify-study", "verify-study-sources"])
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        config = load_config(args.config if args.config.is_absolute() else root / args.config)
        if args.seed is not None and not 0 <= args.seed < 2**32:
            raise ValueError("Seed must be between 0 and 2^32 - 1")
        if args.command == "check-config":
            print(config.model_dump_json(indent=2))
            return 0
        initialize_storage(root, config)
        configure_logging(root / config.paths.runs / "logs", config.log_level)
        if args.command == "init":
            print(f"Project storage ready at {root}")
        elif args.command == "doctor":
            from fraud_detection.diagnostics import check_environment
            result = check_environment()
            output = root / config.paths.runs / "environment.json"
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2))
        elif args.command == "flower-smoke":
            from fraud_detection.federated.smoke import run_smoke
            print(json.dumps(run_smoke(root / config.paths.runs / "flower_smoke"), indent=2))
        elif args.command == "dashboard":
            import subprocess
            import sys
            app = Path(__file__).parent / "dashboard" / "app.py"
            return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app),
                                    "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"], cwd=root)
        elif args.command == "prepare-data":
            from fraud_detection.data.pipeline import prepare_ulb
            dataset = args.dataset or root / config.paths.data / "raw" / "creditcard.csv"
            paths = prepare_ulb(dataset.resolve(), root, seed=config.seed,
                                source_url=args.source_url, data_dir=config.paths.data,
                                clients_dir=config.paths.clients)
            print(json.dumps([str(path) for path in paths], indent=2))
        elif args.command == "verify-data":
            from fraud_detection.data.verify import verify_artifacts
            dataset = args.dataset or root / config.paths.data / "raw" / "creditcard.csv"
            print(json.dumps(verify_artifacts(root, dataset.resolve(),
                                              data_dir=config.paths.data,
                                              clients_dir=config.paths.clients), indent=2))
        elif args.command == "train-baselines":
            from fraud_detection.training.baselines import train_baselines
            settings = ["mild", "strong"] if args.setting == "all" else [args.setting]
            outputs = train_baselines(root, settings=settings,
                                      seed=args.seed if args.seed is not None else config.seed,
                                      config_path=root / "configs" / "baseline.toml",
                                      clients_dir=config.paths.clients,
                                      data_dir=config.paths.data, runs_dir=config.paths.runs,
                                      device=config.device)
            print(json.dumps([str(path) for path in outputs], indent=2))
        elif args.command == "verify-baselines":
            from fraud_detection.training.verify import verify_baseline_setting
            settings = ["mild", "strong"] if args.setting == "all" else [args.setting]
            print(json.dumps([verify_baseline_setting(root, setting=setting,
                                                      seed=args.seed if args.seed is not None else config.seed,
                                                      clients_dir=config.paths.clients,
                                                      data_dir=config.paths.data,
                                                      runs_dir=config.paths.runs)
                              for setting in settings], indent=2))
        elif args.command == "run-fedavg":
            from fraud_detection.federated.runner import run_local_deployment
            settings = ["mild", "strong"] if args.setting == "all" else [args.setting]
            print(json.dumps([str(run_local_deployment(root, setting=setting,
                                                      seed=args.seed if args.seed is not None else config.seed))
                              for setting in settings], indent=2))
        elif args.command == "run-fedmaml":
            from fraud_detection.federated.runner import run_local_deployment
            settings = ["mild", "strong"] if args.setting == "all" else [args.setting]
            print(json.dumps([str(run_local_deployment(root, setting=setting,
                                                      seed=args.seed if args.seed is not None else config.seed,
                                                      method="fedmaml"))
                              for setting in settings], indent=2))
        elif args.command == "evaluate-fedmaml":
            from fraud_detection.meta.evaluate import evaluate_meta
            settings = ["mild", "strong"] if args.setting == "all" else [args.setting]
            print(json.dumps([str(evaluate_meta(root, setting=setting,
                                                seed=args.seed if args.seed is not None else config.seed,
                                                clients_dir=config.paths.clients,
                                                data_dir=config.paths.data,
                                                runs_dir=config.paths.runs))
                              for setting in settings], indent=2))
        elif args.command == "verify-fedmaml":
            from fraud_detection.meta.verify import verify_meta
            settings = ["mild", "strong"] if args.setting == "all" else [args.setting]
            print(json.dumps([verify_meta(root, setting=setting,
                                          seed=args.seed if args.seed is not None else config.seed,
                                          clients_dir=config.paths.clients,
                                          data_dir=config.paths.data,
                                          runs_dir=config.paths.runs)
                              for setting in settings], indent=2))
        elif args.command == "explain":
            from fraud_detection.explainability.local import explain_bank_record
            if args.setting == "all" or args.bank is None:
                raise ValueError("Explain requires one --setting and --bank")
            support_budget = args.support_budget if args.method == "fedmaml" else None
            output = explain_bank_record(root, bank=args.bank, setting=args.setting,
                                         seed=args.seed if args.seed is not None else config.seed,
                                         row_id=args.row_id,
                                         method=args.method, support_budget=support_budget,
                                         steps=args.steps, background_size=args.background_size,
                                         nsamples=args.shap_samples,
                                         clients_dir=config.paths.clients,
                                         runs_dir=config.paths.runs)
            print(str(output))
        elif args.command == "verify-explanations":
            from fraud_detection.explainability.verify import verify_explanations
            settings = ["mild", "strong"] if args.setting == "all" else [args.setting]
            print(json.dumps([verify_explanations(root, setting=setting, seed=config.seed,
                                                  clients_dir=config.paths.clients,
                                                  runs_dir=config.paths.runs)
                              for setting in settings], indent=2))
        elif args.command == "build-study":
            from fraud_detection.evaluation.study import build_study
            print(str(build_study(root, clients_dir=config.paths.clients,
                                  data_dir=config.paths.data, runs_dir=config.paths.runs)))
        elif args.command == "verify-study":
            from fraud_detection.evaluation.study import verify_study
            print(json.dumps(verify_study(root, clients_dir=config.paths.clients,
                                          data_dir=config.paths.data,
                                          runs_dir=config.paths.runs), indent=2))
        elif args.command == "verify-study-sources":
            from fraud_detection.evaluation.study import verify_source_runs
            print(str(verify_source_runs(root, clients_dir=config.paths.clients,
                                         data_dir=config.paths.data,
                                         runs_dir=config.paths.runs)))
        elif args.command == "evaluate-fedavg":
            from fraud_detection.federated.evaluate import evaluate_federated
            settings = ["mild", "strong"] if args.setting == "all" else [args.setting]
            print(json.dumps([str(evaluate_federated(root, setting=setting,
                                                    seed=args.seed if args.seed is not None else config.seed,
                                                    clients_dir=config.paths.clients,
                                                    data_dir=config.paths.data,
                                                    runs_dir=config.paths.runs))
                              for setting in settings], indent=2))
        elif args.command == "verify-fedavg":
            from fraud_detection.federated.verify import verify_fedavg
            settings = ["mild", "strong"] if args.setting == "all" else [args.setting]
            print(json.dumps([verify_fedavg(root, setting=setting,
                                           seed=args.seed if args.seed is not None else config.seed,
                                           clients_dir=config.paths.clients,
                                           data_dir=config.paths.data,
                                           runs_dir=config.paths.runs)
                              for setting in settings], indent=2))
        logging.getLogger("fraud_detection").info("Command completed: %s", args.command)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
