"""Matched-split, multi-seed analysis of saved research runs."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from fraud_detection.data.pipeline import BANKS, file_hash
from fraud_detection.evaluation.metrics import macro_average


SEEDS = (42, 43, 44)
SETTINGS = ("mild", "strong")
METHODS = ("Local", "Centralized", "FedAvg", "FedMAML", "Personalized FedMAML")
METRICS = ("precision", "recall", "f1", "false_positive_rate", "roc_auc", "average_precision")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_result(root: Path, runs_dir: str, setting: str, seed: int, family: str) -> dict:
    path = root / runs_dir / family / setting / f"seed_{seed}" / "results.json"
    result = _read(path)
    if result["status"] != "completed" or result["setting"] != setting or result["seed"] != seed:
        raise ValueError(f"Incomplete or mismatched study source: {path}")
    return result


def _metrics_by_bank(baseline: dict, fedavg: dict, meta: dict, method: str) -> dict:
    if method == "Local":
        return {bank: baseline["local"][bank]["test"] for bank in BANKS}
    if method == "Centralized":
        return {bank: baseline["centralized"]["clients"][bank]["test"] for bank in BANKS}
    if method == "FedAvg":
        return {bank: fedavg["clients"][bank]["test"] for bank in BANKS}
    if method == "FedMAML":
        return {bank: meta["clients"][bank]["before"] for bank in BANKS}
    return {bank: next(item["test"] for item in meta["clients"][bank]["adaptations"]
                       if item["support_budget"] == 32 and item["steps"] == 5)
            for bank in BANKS}


def _explanation_identity(root: Path, clients_dir: str, *, setting: str,
                          seed: int, bank: str, expected_hash: str) -> bool:
    folder = root / clients_dir / bank / "explanations" / setting / "fedmaml"
    for path in folder.glob("*/explanation.json"):
        item = _read(path)
        adaptation = item["model"]["adaptation"]
        if (item["model"]["model_sha256"] == expected_hash
                and item["cache_contract"]["model_sha256"] == expected_hash
                and item["cache_contract"].get("seed", 42) == seed
                and adaptation is not None and adaptation["support_budget"] == 32
                and adaptation["steps"] == 5):
            return True
    return False


def _mean_std(rows: list[dict], group_keys: tuple[str, ...], value_keys: tuple[str, ...]) -> list[dict]:
    frame = pd.DataFrame(rows)
    output = []
    for group, part in frame.groupby(list(group_keys), sort=True):
        item = dict(zip(group_keys, group if isinstance(group, tuple) else (group,)))
        item["n_seeds"] = len(part)
        for key in value_keys:
            values = part[key].dropna().to_numpy(dtype=float)
            item[f"{key}_mean"] = float(np.mean(values)) if len(values) else None
            item[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        output.append(item)
    return output


def collect_study(root: Path, *, seeds: tuple[int, ...] = SEEDS,
                  settings: tuple[str, ...] = SETTINGS, clients_dir: str = "clients",
                  data_dir: str = "data", runs_dir: str = "runs",
                  require_explanations: bool = True) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or not settings:
        raise ValueError("Study needs unique seeds and at least one setting")
    per_run, per_bank, adaptation, communication, budgets = [], [], [], [], []
    manifests = {}
    method_configs = {}
    explanation_matches = 0
    for setting in settings:
        if setting not in SETTINGS:
            raise ValueError("Unknown data setting")
        metadata = _read(root / data_dir / "processed" / setting / "metadata.json")
        manifest = root / data_dir / "processed" / setting / "manifest.parquet"
        if file_hash(manifest) != metadata["manifest_sha256"]:
            raise ValueError("Partition manifest changed")
        manifests[setting] = metadata["manifest_sha256"]
        bank_counts = metadata["partition"]["counts"]
        train_total = sum(bank_counts[bank]["train"]["rows"] for bank in BANKS)
        for seed in seeds:
            baseline = _source_result(root, runs_dir, setting, seed, "baselines")
            fedavg = _source_result(root, runs_dir, setting, seed, "federated")
            meta = _source_result(root, runs_dir, setting, seed, "meta")
            meta_run = _read(root / runs_dir / "meta" / setting / f"seed_{seed}" / "run.json")
            sources = (baseline, fedavg, meta)
            if any(result["partition_manifest_sha256"] != manifests[setting] for result in sources):
                raise ValueError("Methods used different data partitions")
            if any(value != metadata["dataset"]["sha256"] for value in
                   (baseline["dataset_sha256"], fedavg["dataset_sha256"],
                    meta_run["dataset_sha256"])):
                raise ValueError("Methods used different source datasets")
            if len({json.dumps(spec, sort_keys=True) for spec in
                    (baseline["model_spec"], fedavg["model_spec"], meta_run["model_spec"])}) != 1:
                raise ValueError("Methods used different model architectures")
            for name, configuration in (
                    ("Local/Centralized", baseline["train_spec"]),
                    ("FedAvg", {key: value for key, value in _read(root / runs_dir / "federated" /
                        setting / f"seed_{seed}" / "run.json")["run_config"].items()
                        if key not in ("setting", "seed")}),
                    ("FedMAML", {key: value for key, value in meta_run["run_config"].items()
                        if key not in ("setting", "seed")})):
                encoded = json.dumps(configuration, sort_keys=True)
                if name in method_configs and method_configs[name] != encoded:
                    raise ValueError(f"{name} hyperparameters changed between seeds or settings")
                method_configs[name] = encoded
            if int(baseline["train_spec"]["epochs"]) != 5 or fedavg["rounds"] != 5 or meta["rounds"] != 5:
                raise ValueError("Study cycle budget differs from the declared five cycles")
            for method in METHODS:
                by_bank = _metrics_by_bank(baseline, fedavg, meta, method)
                macro = macro_average(by_bank)
                source_macro = (baseline["macro_test"]["local"] if method == "Local" else
                                baseline["macro_test"]["centralized"] if method == "Centralized" else
                                fedavg["macro_test"] if method == "FedAvg" else
                                meta["macro_before_test"] if method == "FedMAML" else
                                meta["macro_after_test"]["budget_32_steps_5"])
                for key in METRICS:
                    if not np.isclose(macro[key], source_macro[key], rtol=0, atol=1e-12):
                        raise ValueError(f"Saved macro metric differs: {setting}, {seed}, {method}, {key}")
                per_run.append({"setting": setting, "seed": seed, "method": method, **macro})
                for bank, metric in by_bank.items():
                    expected = bank_counts[bank]["test"]
                    if (metric["rows"] != expected["rows"] or metric["fraud"] != expected["fraud"]
                            or not np.isclose(metric["fraud_prevalence"], expected["fraud_prevalence"])
                            or not 0 <= metric["threshold"] <= 1):
                        raise ValueError("Per-bank test counts, prevalence, or threshold differ")
                    per_bank.append({"setting": setting, "seed": seed, "method": method,
                                     "bank": bank, "rows": metric["rows"],
                                     "fraud": metric["fraud"], "prevalence": metric["fraud_prevalence"],
                                     "threshold": metric["threshold"], **{
                                         key: metric[key] for key in METRICS}})
            for bank in BANKS:
                expected_hash = next(item["model_sha256"] for item in meta["clients"][bank]["adaptations"]
                                     if item["support_budget"] == 32 and item["steps"] == 5)
                found = _explanation_identity(root, clients_dir, setting=setting,
                                              seed=seed, bank=bank, expected_hash=expected_hash)
                if require_explanations and not found:
                    raise ValueError(f"Explained E4 model hash missing: {setting}, seed {seed}, {bank}")
                explanation_matches += int(found)
                for item in meta["clients"][bank]["adaptations"]:
                    adaptation.append({"setting": setting, "seed": seed, "bank": bank,
                                       "support_budget": item["support_budget"],
                                       "support_fraud": item["support_fraud"],
                                       "steps": item["steps"],
                                       **{key: item["test"][key] for key in METRICS}})
            for family, method in (("federated", "FedAvg"), ("meta", "FedMAML")):
                run_dir = root / runs_dir / family / setting / f"seed_{seed}"
                reports = [_read(run_dir / f"round_{number}.json") for number in range(1, 6)]
                if any(report["status"] != "completed" or len(report["clients"]) != 4 for report in reports):
                    raise ValueError("Incomplete federated communication record")
                broadcast = sum(report["train_broadcast_tensor_bytes"] for report in reports)
                updates = sum(report.get("train_update_tensor_bytes",
                                         report.get("gradient_update_tensor_bytes", 0)) for report in reports)
                validation = sum(report["validation_broadcast_tensor_bytes"] for report in reports)
                communication.append({"setting": setting, "seed": seed, "method": method,
                                      "train_broadcast_bytes": broadcast,
                                      "train_update_bytes": updates,
                                      "validation_broadcast_bytes": validation,
                                      "tensor_bytes_total": broadcast + updates + validation,
                                      "round_seconds_total": sum(report["round_total_seconds"] for report in reports)})
            budgets.extend([
                {"setting": setting, "seed": seed, "method": "Local", "cycles": 5,
                 "train_examples_processed_approx": 5 * train_total,
                 "note": "Five local full-data epochs across four banks"},
                {"setting": setting, "seed": seed, "method": "Centralized", "cycles": 5,
                 "train_examples_processed_approx": 5 * train_total,
                 "note": "Five pooled full-data epochs"},
                {"setting": setting, "seed": seed, "method": "FedAvg", "cycles": 5,
                 "train_examples_processed_approx": 5 * train_total,
                 "note": "Five rounds with one local full-data epoch each"},
                {"setting": setting, "seed": seed, "method": "FedMAML", "cycles": 5,
                 "train_examples_processed_approx": 5 * 4 * (32 + 32) + 4 * 32 * 5,
                 "note": "Five rounds of balanced 32-support/32-query episodes plus 32-support five-step adaptation; repeated support steps counted"},
            ])
    expected_runs = len(settings) * len(seeds) * len(METHODS)
    if len(per_run) != expected_runs:
        raise RuntimeError("Study result matrix is incomplete")
    adaptation_macro = []
    adaptation_frame = pd.DataFrame(adaptation)
    for group, frame in adaptation_frame.groupby(["setting", "seed", "support_budget", "steps"]):
        if set(frame["bank"]) != set(BANKS):
            raise ValueError("Adaptation grid is missing a bank")
        item = dict(zip(("setting", "seed", "support_budget", "steps"), group))
        item.update({key: float(frame[key].mean()) for key in METRICS})
        adaptation_macro.append(item)
    return {"schema_version": 1, "status": "complete", "seeds": list(seeds),
            "settings": list(settings),
            "training_methods": ["Local", "Centralized", "FedAvg", "FedMAML"],
            "evaluation_variants": list(METHODS),
            "partition_manifest_sha256": manifests,
            "fixed_method_configs": {name: json.loads(value) for name, value in method_configs.items()},
            "comparison_contract": "same public ULB partition, model architecture, validation/test split, threshold rule, and seed list; five training cycles for each method; record exposure differs",
            "selected_personalization": {"support_budget": 32, "steps": 5,
                "policy": "Fixed before this multi-seed summary; other adaptation grid entries are exploratory"},
            "explanation_identity": {"matched_models": explanation_matches,
                                     "expected_models": len(settings) * len(seeds) * len(BANKS),
                                     "training_method_count": 4},
            "per_run": per_run, "per_bank": per_bank, "adaptation": adaptation,
            "adaptation_macro": adaptation_macro,
            "communication": communication, "budgets": budgets,
            "method_summary": _mean_std(per_run, ("setting", "method"), METRICS),
            "bank_summary": _mean_std(per_bank, ("setting", "method", "bank"), METRICS),
            "communication_summary": _mean_std(communication, ("setting", "method"),
                ("tensor_bytes_total", "round_seconds_total")),
            "adaptation_summary": _mean_std(adaptation_macro,
                ("setting", "support_budget", "steps"), METRICS)}


def build_study(root: Path, *, output_dir: Path | None = None,
                seeds: tuple[int, ...] = SEEDS,
                settings: tuple[str, ...] = SETTINGS,
                clients_dir: str = "clients", data_dir: str = "data",
                runs_dir: str = "runs") -> Path:
    output_dir = output_dir or root / runs_dir / "study"
    if output_dir.exists():
        raise FileExistsError(f"Study output already exists: {output_dir}")
    study = collect_study(root, seeds=seeds, settings=settings,
                          clients_dir=clients_dir, data_dir=data_dir, runs_dir=runs_dir)
    output_dir.mkdir(parents=True)
    output = output_dir / "study.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(study, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    for key in ("per_run", "per_bank", "adaptation", "adaptation_macro", "communication", "budgets",
                "method_summary", "bank_summary", "communication_summary", "adaptation_summary"):
        pd.DataFrame(study[key]).to_csv(output_dir / f"{key}.csv", index=False)
    runs = pd.DataFrame(study["per_run"])
    summary = pd.DataFrame(study["method_summary"])
    px.bar(summary, x="method", y="f1_mean", error_y="f1_sd", color="method",
           facet_col="setting", title="Held-out macro F1 across three seeds").write_html(
               output_dir / "macro_f1.html", include_plotlyjs=True)
    px.scatter(runs, x="f1", y="average_precision", color="method", symbol="seed",
               facet_col="setting", title="Each seed: macro F1 and average precision").write_html(
                   output_dir / "seed_variation.html", include_plotlyjs=True)
    px.box(pd.DataFrame(study["per_bank"]), x="bank", y="f1", color="method",
           facet_col="setting", points="all", title="Per-bank held-out F1 across seeds").write_html(
               output_dir / "bank_f1.html", include_plotlyjs=True)
    adapt = pd.DataFrame(study["adaptation_summary"])
    px.line(adapt, x="steps", y="f1_mean", color="support_budget", facet_col="setting",
            markers=True, title="Personalization sensitivity: macro F1").write_html(
                output_dir / "adaptation.html", include_plotlyjs=True)
    traffic = pd.DataFrame(study["communication"])
    px.bar(traffic, x="method", y="tensor_bytes_total", color="method", facet_col="setting",
           title="Approximate model tensor traffic by run").write_html(
               output_dir / "communication.html", include_plotlyjs=True)
    return output


def verify_study(root: Path, *, output_dir: Path | None = None,
                 clients_dir: str = "clients", data_dir: str = "data",
                 runs_dir: str = "runs") -> dict:
    output_dir = output_dir or root / runs_dir / "study"
    saved = _read(output_dir / "study.json")
    recomputed = collect_study(root, seeds=tuple(saved["seeds"]),
                               settings=tuple(saved["settings"]),
                               clients_dir=clients_dir, data_dir=data_dir,
                               runs_dir=runs_dir)
    if saved != recomputed:
        raise ValueError("Saved study differs from source artifacts")
    for key in ("per_run", "per_bank", "adaptation", "adaptation_macro", "communication", "budgets",
                "method_summary", "bank_summary", "communication_summary", "adaptation_summary"):
        path = output_dir / f"{key}.csv"
        if not path.is_file():
            raise ValueError(f"Study table missing or incomplete: {key}")
        try:
            expected = pd.DataFrame(saved[key]).replace({None: np.nan})
            pd.testing.assert_frame_equal(pd.read_csv(path), expected,
                                          check_dtype=False, rtol=1e-12, atol=1e-12)
        except AssertionError as exc:
            raise ValueError(f"Study table differs from source artifacts: {key}") from exc
    for name in ("macro_f1", "seed_variation", "bank_f1", "adaptation", "communication"):
        if not (output_dir / f"{name}.html").is_file():
            raise ValueError(f"Study figure missing: {name}")
    return {"status": "passed", "seeds": saved["seeds"],
            "settings": saved["settings"], "method_seed_setting_rows": len(saved["per_run"]),
            "bank_result_rows": len(saved["per_bank"]),
            "explained_e4_hash_matches": saved["explanation_identity"]["matched_models"]}


def verify_source_runs(root: Path, *, clients_dir: str = "clients",
                       data_dir: str = "data", runs_dir: str = "runs") -> Path:
    """Replay all saved trainers and bank-local explanations for the study matrix."""
    from fraud_detection.explainability.verify import verify_explanations
    from fraud_detection.federated.verify import verify_fedavg
    from fraud_detection.meta.verify import verify_meta
    from fraud_detection.training.verify import verify_baseline_setting

    results = []
    for setting in SETTINGS:
        for seed in SEEDS:
            results.append({"method": "Local and Centralized", **verify_baseline_setting(
                root, setting=setting, seed=seed, clients_dir=clients_dir,
                data_dir=data_dir, runs_dir=runs_dir)})
            results.append({"method": "FedAvg", **verify_fedavg(
                root, setting=setting, seed=seed, clients_dir=clients_dir,
                data_dir=data_dir, runs_dir=runs_dir)})
            results.append({"method": "FedMAML", **verify_meta(
                root, setting=setting, seed=seed, clients_dir=clients_dir,
                data_dir=data_dir, runs_dir=runs_dir)})
    explanations = [verify_explanations(root, setting=setting, seed=42,
                   clients_dir=clients_dir, runs_dir=runs_dir) for setting in SETTINGS]
    report = {"status": "passed", "source_run_checks": len(results),
              "run_checks": results, "explanation_checks": explanations}
    path = root / runs_dir / "study" / "source_verification.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path
