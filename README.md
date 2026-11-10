# Personalized Credit Card Fraud Detection

A research prototype for four simulated financial institutions using federated learning, first-order MAML personalization, and local SHAP explanations.

## Project documentation

- [System architecture](docs/ARCHITECTURE.md): components, data boundaries, training algorithms, interfaces, evaluation, and deployment.
- [Usage guide](docs/USAGE_GUIDE.md): start the application and reproduce the demonstration.
- [Study report](docs/STUDY_REPORT.md): measured three-seed results and limits.

The application includes data preparation, four training methods, bank-local personalization and explanations, a dashboard, and reproducible evaluation. The measured meta-learning setting did not outperform the established baselines.

## Setup on Windows

Use 64-bit Python 3.13. Open PowerShell in this project directory:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File scripts\verify.ps1
powershell -ExecutionPolicy Bypass -File scripts\verify-results.ps1
```

Setup creates an isolated `.venv`, installs exact dependencies from `requirements.lock`, installs this package, and creates local storage directories. `verify.ps1` checks the installed application; `verify-results.ps1` checks the prepared dataset and saved three-seed study in this working copy. The execution policy option affects only that command; it does not change your system policy. No activation is needed.

## Available commands

```powershell
.\.venv\Scripts\python.exe -m fraud_detection check-config
.\.venv\Scripts\python.exe -m fraud_detection init
.\.venv\Scripts\python.exe -m fraud_detection doctor
.\.venv\Scripts\python.exe -m fraud_detection flower-smoke
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
```

The dashboard opens on localhost. It shows dataset and saved-run summaries, scores bank-local held-out examples or validated CSV files, displays local explanations, tracks federated rounds, and compares saved models. Stop it with Ctrl+C.

`doctor` checks a PyTorch backward pass, Flower serialization, SHAP reconstruction, basic data and metrics operations, and Streamlit rendering. `flower-smoke` launches one actual Flower server and four client processes using synthetic constants, verifies a weighted aggregation result of 3.0, then stops them. This is an environment test, not model training or a scientific result.

The Flower transport diagnostic uses the retained gRPC interface. Actual federated training uses Flower ClientApp and ServerApp with a localhost SuperLink and four SuperNodes. WSL2 is not required for this tested route.

## Prepared ULB dataset

The public ULB `creditcard.csv` was downloaded from [Zenodo](https://zenodo.org/records/7395559) and verified against the checksum published there. The raw file and all generated bank data are stored locally and ignored by version control. See [dataset report](docs/DATASET_REPORT.md) for the observed counts and partition method.

To verify the existing preparation:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection verify-data
```

For a fresh project with an identical downloaded ULB CSV, place `creditcard.csv` in `data/raw`, then run:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection --source-url "https://zenodo.org/records/7395559" prepare-data
.\.venv\Scripts\python.exe -m fraud_detection verify-data
```

The preparation command refuses to overwrite an existing partition. It creates mild and strong four-bank simulations, with separate local training, validation, and test files. Validation and test keep their natural prevalence within each simulated bank. The coordinator does not receive transaction files. The bank names are simulations, not labels from ULB.

## Baseline fraud models

Four local models and one centralized reference were trained for each of the mild and strong client settings. They share one PyTorch MLP architecture and the same split manifests. Model checkpoints, per-bank thresholds, test metrics, and exact training settings are saved locally. The results and their limits are in [baseline report](docs/BASELINE_REPORT.md).

To independently reload all saved models and recompute validation choices and test metrics:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection verify-baselines
```

On a prepared project with no baseline run for the configured seed, train with:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection train-baselines
```

The command trains both settings by default; `--setting mild` or `--setting strong` selects one. It refuses to overwrite existing runs. Training uses the parameters in `configs/baseline.toml` and the seed in `configs/default.toml`. Test data is loaded after model selection and threshold selection. These are research scores from public data; the sigmoid outputs have not been calibrated as real-world fraud probabilities.

## Federated baseline

The FedAvg experiment completed five rounds with four simulated banks in each mild and strong setting. Flower clients train on their own prepared files and send model tensors plus aggregate counts and losses. The server saves each round and checks the weighted model average. A separate offline evaluator computes public-benchmark test metrics after training. See the [FedAvg report](docs/FEDAVG_REPORT.md) for results and limitations.

To verify the saved runs:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection verify-fedavg
```

To launch and evaluate a new run for a setting with an unused seed:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection --setting mild --seed 45 run-fedavg
```

The run refuses to overwrite existing results. The local Flower transport is plaintext over localhost and is intended for this simulated research setup, not multi-host banking deployment.

## Meta-learning and personalization

Four banks completed five first-order MAML rounds in each setting. Each client adapts on a disjoint support set, sends a query gradient, and the coordinator applies an equal-bank outer update. The saved global initialization was evaluated before and after separate bank-local adaptation with 32, 128, and 512 training examples and 1, 3, and 5 steps. See the [meta-learning report](docs/META_REPORT.md) for the measured results and limitations.

```powershell
.\.venv\Scripts\python.exe -m fraud_detection verify-fedmaml
```

For an unused seed, `--setting mild --seed 45 run-fedmaml` trains and evaluates one setting. `evaluate-fedmaml` evaluates an otherwise completed training run whose evaluation is missing. Existing runs are protected from overwrite.

## Bank-local explanations

The explanation service uses a bank's saved model and training-only background to explain a selected held-out transaction on the predicted-probability scale. It saves SHAP contributions and interactive waterfall and bar charts under that bank's `clients` directory. The [explanations report](docs/EXPLANATIONS_REPORT.md) records the checks and limits.

```powershell
.\.venv\Scripts\python.exe -m fraud_detection --setting mild --bank bank_a explain
.\.venv\Scripts\python.exe -m fraud_detection verify-explanations
```

`--row-id` selects another test record. The default selects the first fraud test row for that bank. The default explains the bank's 32-row, five-step personalized FedMAML model; `--method fedavg` explains the global FedAvg model. Changing the model, preprocessing, background, transaction, or SHAP sample budget creates a new cache key. These explanations describe model associations, not causes of fraud.

## Dashboard workflow

Launch with `powershell -ExecutionPolicy Bypass -File scripts\start.ps1`, then open the localhost address printed by Streamlit. Choose a bank, data setting, and saved seed in the sidebar. In **Detection**, select a saved model and held-out example, then select **Score selected example**. In **Explanation**, choose the same bank, model, and row and select **Create or load explanation**. The page displays its probability-scale waterfall and bar charts. **Federated rounds** shows saved or running round records; **Comparison** reads the held-out metrics saved by the trainers.

For CSV scoring, upload original-scale ULB feature columns in the exact order `Time,V1,...,V28,Amount`. The bank's saved training scaler transforms Time and Amount. The upload accepts up to 1,000 rows and 2 MB, rejects missing or nonnumeric values, and displays scores and bank-specific decisions. Uploaded rows are processed locally and are not sent to Flower.

The **Training** tab starts a new FedAvg or FedMAML run in a separate process when you select an unused seed and click **Start background run**. The saved status and log stay visible after page refreshes. Existing runs are protected from overwrite. See the [dashboard report](docs/DASHBOARD_REPORT.md) for verification and limitations.

## Scientific evaluation

The [three-seed study report](docs/STUDY_REPORT.md) compares four training methods on both bank distributions, reports means and variation across seeds, examines adaptation budgets and communication, and states the limits of the comparison. The prespecified [evaluation protocol](docs/EVALUATION_PROTOCOL.md) records the common data and five-cycle schedule and the methods' unequal training-row exposure. Saved tables and interactive figures are in `runs/study/`.

```powershell
.\.venv\Scripts\python.exe -m fraud_detection verify-study
.\.venv\Scripts\python.exe -m fraud_detection verify-study-sources
```

The first command recomputes summary tables from saved results. The second replays every saved source run and checks the local explanations. To regenerate the study in a fresh workspace after the three-seed runs and explanations are present, use `build-study`; it refuses to overwrite existing output.

## Configuration and artifacts

Edit `configs/default.toml` to change the default random seed, device, logging, or project-local storage directories. Unknown settings and invalid client lists are rejected. Defaults use CPU and four simulated clients. Application logs go to `runs/logs`; environment evidence goes to `runs/environment.json`; the diagnostic goes to `runs/flower_smoke`, and federated process logs go to `runs/flower_runtime/logs`.

Raw data, local client artifacts, model files, generated runs, environments, and secrets are excluded from version control by `.gitignore`. No dataset credentials are needed for foundation verification.

The exact dependency snapshot targets Windows x64 and Python 3.13. Revalidate and regenerate it for other platforms. To verify a second clean environment, run setup and verification with `-Environment .venv-check`.

## Handoff package

The installable application wheel is under `dist/`. To create a source handoff archive with setup scripts, configuration, documentation, and tests, run `.\.venv\Scripts\python.exe scripts\package_release.py`. Its allowlist excludes local dataset, bank, and run artifacts, and its `SOURCE_MANIFEST.json` records file checksums. The [release notes](docs/RELEASE_NOTES.md) record verification evidence and known limits.
