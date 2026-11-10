import json

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.pipeline import (
    EXPECTED_COLUMNS, FEATURES, build_partition, duplicate_groups, load_ulb, prepare_ulb,
)
from fraud_detection.data.verify import verify_artifacts


def make_sample(path):
    rng = np.random.default_rng(9)
    count = 2_160
    frame = pd.DataFrame({name: rng.normal(size=count) for name in FEATURES})
    frame["Time"] = rng.uniform(0, 200_000, size=count)
    frame["Amount"] = rng.uniform(0, 1_000, size=count)
    labels = np.array([0] * 2_000 + [1] * 160, dtype=np.int8)
    rng.shuffle(labels)
    frame["Class"] = labels
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame = frame[EXPECTED_COLUMNS]
    frame.to_csv(path, index=False)
    return frame


def test_rejects_altered_reference_and_nonfinite_values(tmp_path):
    path = tmp_path / "sample.csv"
    frame = make_sample(path)
    with pytest.raises(ValueError, match="MD5"):
        load_ulb(path)
    frame.loc[0, "V1"] = np.nan
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="nonfinite"):
        load_ulb(path, require_reference=False)


def test_partition_reproduces_and_keeps_duplicates_together(tmp_path):
    path = tmp_path / "sample.csv"
    make_sample(path)
    frame, _ = load_ulb(path, require_reference=False)
    groups = duplicate_groups(frame)
    assert groups[3]["extra_duplicate_rows"] == 1
    first, report = build_partition(frame, setting="strong", seed=23, group_info=groups,
                                    min_legitimate_groups=50)
    second, _ = build_partition(frame, setting="strong", seed=23, group_info=groups,
                                min_legitimate_groups=50)
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, ["client_id", "split", "feature_group"]].tolist() == first.loc[len(first)-1, ["client_id", "split", "feature_group"]].tolist()
    assert first.row_id.nunique() == len(frame)
    assert all(report["counts"][bank][split]["fraud"] > 0
               for bank in report["counts"] for split in ("train", "validation", "test"))


def test_written_partitions_verify_train_only_scaling(tmp_path):
    path = tmp_path / "sample.csv"
    make_sample(path)
    output = tmp_path / "project"
    output.mkdir()
    metadata = prepare_ulb(path, output, seed=23, require_reference=False,
                           min_legitimate_groups=50, data_dir="source_data",
                           clients_dir="institutions")
    assert len(metadata) == 2
    result = verify_artifacts(output, path, require_reference=False,
                              data_dir="source_data", clients_dir="institutions")
    assert result["status"] == "passed"
    scaler_path = output / "institutions/bank_a/preprocessing/mild_scaler.json"
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    scaler["mean"][0] += 10
    scaler_path.write_text(json.dumps(scaler), encoding="utf-8")
    with pytest.raises(AssertionError):
        verify_artifacts(output, path, require_reference=False,
                         data_dir="source_data", clients_dir="institutions")
