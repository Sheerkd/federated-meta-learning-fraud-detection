# Integrated research dashboard

## Completed workflow

The localhost Streamlit dashboard has six views: overview, detection, explanation, federated rounds, comparison, and training. It reads the saved dataset and run artifacts rather than displaying placeholder values. A bank, partition setting, and saved seed are selected in the sidebar. Detection supports the local, centralized, FedAvg, global FedMAML, and recorded 32-row/five-step personalized FedMAML models where completed artifacts are available.

The demonstrated path selects mild bank A and held-out row 6336, scores it with the saved personalized FedMAML model, then opens its bank-local SHAP explanation and probability-scale plots. The displayed score was 0.5543 with a bank threshold of 0.6997, so the model did not flag it. The dashboard also reads round duration, bank participation, aggregate loss, and approximate tensor traffic from real per-round reports. The comparison table reads per-bank test metrics from completed baseline, FedAvg, and FedMAML result files and calculates equal-bank macro means.

CSV uploads require the 30 original-scale ULB feature columns in exact order, one to 1,000 rows, at most 2 MB, and finite numeric values. The chosen bank's training-only scaler transforms Time and Amount. Invalid schemas, missing values, and negative Time or Amount are rejected. The bank selector is a local research convenience, not an access-control boundary.

## Background run evidence

Training starts only after the user clicks the run button. A separate worker process records starting, running, completed, or failed status in `runs/dashboard_jobs`, with a log and result path. A fresh mild FedAvg seed 43 run was started through this worker and completed five rounds with four banks. Its checkpoint lineage, validation choices, and test metrics passed the independent FedAvg verifier. The worker recorded `completed`; the dashboard read all five completed round records. Existing run directories are refused rather than overwritten.

## Verification and limits

The dashboard served a localhost health request successfully (`200 OK`). Streamlit interaction tests clicked the score and explanation controls and observed the corresponding real values without application exceptions. Tests also checked the comparison value against a saved FedAvg macro result, missing-model behavior, invalid CSVs, bank scaling, and worker status. A full-suite test exposed sensitivity of personalized-model reconstruction to PyTorch's CPU thread count; the evaluator now records that setting and the dashboard restores it while adapting. The full project test suite passed 25 tests, and both saved meta runs and nine explanation artifacts passed independent verification afterward.

The interface uses one local process with access to all four simulated banks for demonstration. It is not a secure multi-bank deployment. Uploaded CSV rows are scored in the local dashboard process and are not sent to Flower. The comparison page shows the selected seed; the [study report](STUDY_REPORT.md) gives the three-seed analysis.
