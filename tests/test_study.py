from pathlib import Path

import pytest

from fraud_detection.evaluation.study import build_study, collect_study, verify_study


def test_one_seed_study_matches_saved_methods_and_explanation_identity():
    root = Path(__file__).resolve().parents[1]
    if not (root / "runs/meta/mild/seed_42/results.json").is_file():
        pytest.skip("Prepared study artifacts are unavailable")
    study = collect_study(root, seeds=(42,), settings=("mild",))
    assert len(study["per_run"]) == 5
    assert len(study["per_bank"]) == 20
    assert len(study["adaptation_macro"]) == 9
    assert study["explanation_identity"]["matched_models"] == 4
    assert {row["n_seeds"] for row in study["method_summary"]} == {1}
    budgets = {row["method"]: row["train_examples_processed_approx"] for row in study["budgets"]}
    assert budgets["Local"] == budgets["Centralized"] == budgets["FedAvg"]
    assert budgets["FedMAML"] < budgets["FedAvg"]
    fedavg = next(row for row in study["per_run"] if row["method"] == "FedAvg")
    assert fedavg["f1"] == pytest.approx(0.8312295751633987)


def test_study_refuses_missing_seed_and_duplicate_seed():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="unique"):
        collect_study(root, seeds=(42, 42), settings=("mild",))
    with pytest.raises(FileNotFoundError):
        collect_study(root, seeds=(999999,), settings=("mild",))


def test_study_exports_tables_and_figures_then_rechecks_sources(tmp_path):
    root = Path(__file__).resolve().parents[1]
    if not (root / "runs/meta/mild/seed_42/results.json").is_file():
        pytest.skip("Prepared study artifacts are unavailable")
    output = tmp_path / "study"
    path = build_study(root, output_dir=output, seeds=(42,), settings=("mild",))
    assert path.is_file()
    assert (output / "per_bank.csv").is_file()
    assert (output / "macro_f1.html").is_file()
    assert (output / "bank_f1.html").is_file()
    assert verify_study(root, output_dir=output)["status"] == "passed"
    table = output / "per_bank.csv"
    original = table.read_text(encoding="utf-8")
    table.write_text(original.replace("bank_a", "altered", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="table differs"):
        verify_study(root, output_dir=output)
