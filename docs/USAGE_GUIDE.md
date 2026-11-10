# Application usage guide

## Install and check

Use Windows x64 and Python 3.13. In PowerShell, open the project directory and run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File scripts\verify.ps1
powershell -ExecutionPolicy Bypass -File scripts\verify-results.ps1
```

The first command creates `.venv`, installs the pinned dependencies, and initializes local folders. The second checks dependencies, runs the tests, checks the main libraries, and completes a synthetic four-client Flower connectivity test. The third verifies the prepared dataset and saved study in this working copy. None of these commands retrains the fraud models.

The public ULB benchmark file is not bundled with the source. The current working copy already contains the verified `data/raw/creditcard.csv` and prepared bank artifacts. In a new copy, download the matching [Zenodo ULB CSV](https://zenodo.org/records/7395559) to `data/raw/creditcard.csv`, then run:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection --source-url "https://zenodo.org/records/7395559" prepare-data
.\.venv\Scripts\python.exe -m fraud_detection verify-data
```

The command checks the source checksum and refuses to overwrite existing partitions. Prepared data and trained model artifacts are local and excluded from version control; a source-only copy must train new models before it can show saved predictions.

## Demonstrate a prediction and explanation

Start the dashboard:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
```

Open the localhost address shown in PowerShell. Select **mild**, **bank_a**, and saved seed **42** in the sidebar. In **Detection**, select **Personalized FedMAML** and held-out row **6336**, then click **Score selected example**. The saved research result is a fraud score around **0.5543** and a bank threshold around **0.6997**, so this configuration does **not** flag that known fraud case. This is an intentional demonstration of a model error, not a claim of correct detection.

In **Explanation**, keep **Personalized FedMAML** and row **6336**, then click **Create or load explanation**. Review the waterfall, top-contribution bar chart, and feature table. The explanation is saved under that bank's `clients/bank_a/explanations/` folder. `V1` through `V28` are anonymized PCA features; their contributions describe the model, not a business reason for a transaction. The **Comparison** view shows per-bank held-out scores for all saved methods. The [study report](STUDY_REPORT.md) provides the three-seed comparison and its limits.

## Score a CSV

The **Detection** view accepts a CSV of one to 1,000 original-scale transactions, up to 2 MB. Its header must be exactly `Time,V1,V2,...,V28,Amount` in that order, with finite numeric values. Use the bank selector to choose the corresponding local scaler. The upload produces fraud scores, thresholds, and decisions; it does not need a `Class` column. Malformed or missing values are rejected. Uploaded rows are processed in this local dashboard process and are not sent to the Flower coordinator.

## Run or verify experiments

The **Training** view starts a FedAvg or FedMAML run only after you click **Start background run** with an unused seed. The saved worker status and log remain visible after refresh. Training uses the existing prepared bank partitions, so changing the model seed does not create a new partition. Runs with an existing setting and seed are protected from overwrite.

For command-line verification of the included study artifacts:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection verify-study
.\.venv\Scripts\python.exe -m fraud_detection verify-study-sources
```

`verify-study` recomputes the summary and checks the exported CSV tables. `verify-study-sources` reloads all saved model runs and checks local explanations. To train from an unprepared source-only copy, follow the data step above, then use `train-baselines`, `run-fedavg`, and `run-fedmaml` for a chosen unused seed and setting. The current three-seed study requires seeds 42, 43, and 44 in both settings plus the bank-local explanations. Exact configuration and budget details are in [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).

## What the results mean

These are experiments on one public dataset split into simulated banks on a single machine. The tested first-order meta-learning setting underperformed the local, centralized, and FedAvg references. The scores are not calibrated for live fraud decisions. The local plaintext Flower transport, shared simulation host, and bank switcher are not a production privacy or security design.
