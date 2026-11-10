# Research application release

## Included in this working copy

The project is one Python application with a shared 30-feature fraud model, verified ULB data preparation, local and centralized references, real four-client Flower FedAvg and first-order FedMAML training, bank-local personalization, SHAP explanations, and a Streamlit dashboard. The same package and saved artifacts support command-line checks and the dashboard. Dataset files, bank-local records, checkpoints, and generated reports remain local and are excluded from version control.

The saved study covers four training methods, two bank distributions, and three training seeds on fixed partitions. It includes per-bank prevalence and validation thresholds, mean and sample standard deviation across seeds, adaptation sensitivity, communication estimates, and interactive figures. The tested meta-learning configuration underperformed the established references; the [study report](STUDY_REPORT.md) states the observed values and comparison limits.

## Reproducibility evidence

- The verified public ULB copy contains 284,807 transactions, including 492 fraud cases. Its source checksum and every prepared bank split are recorded in the [dataset report](DATASET_REPORT.md).
- The study summary reproduces 30 evaluation-variant rows and 120 bank rows from the saved runs. Eighteen independent source-run checks passed, and 24 required personalized explanation model hashes matched the corresponding evaluated models.
- The dashboard scored a real held-out row and displayed its bank-local explanation. A background FedAvg run completed through the dashboard worker and was independently verified.
- A new project-local `.venv-release` installed all pinned dependencies from `requirements.lock`; `pip check` found no broken requirements. The clean environment passed all 29 project tests, functional library checks, and a synthetic four-client Flower round. Its saved-results check verified the dataset, study, all 18 source runs, and the local explanations.
- The clean-environment dashboard demo scored held-out mild bank A row 6336 at 0.5543 against threshold 0.6997 and loaded the matching 0.5543 explanation. The live localhost dashboard health endpoint returned HTTP 200 and `ok`.
- An installable wheel was built under `dist/` and inspected: 40 Python application modules, with no dataset, bank, or run entries. The source handoff archive is built from an explicit allowlist and includes a SHA-256 manifest for its files.

The [usage guide](USAGE_GUIDE.md) gives the installation commands and a repeatable prediction-to-explanation demonstration. [ARCHITECTURE.md](ARCHITECTURE.md) describes the actual components and privacy boundaries. Focused reports remain available as detailed evidence for each subsystem.

## Limits

The four banks are simulations on one public dataset and one host, with localhost plaintext Flower transport. The dashboard's bank switcher is a demonstration control. There is no production access-control, TLS deployment, secure aggregation, differential privacy, calibration for live decisions, or external-bank validation. The three seeds share fixed data partitions, and FedMAML uses far fewer training examples than the full-data references. The measured performance differences should be read with those limits.
