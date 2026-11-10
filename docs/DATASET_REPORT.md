# ULB dataset preparation report

Completed 8 November 2026. The dataset is the public ULB European credit card fraud benchmark obtained from the [Zenodo record](https://zenodo.org/records/7395559), which identifies the Kaggle ULB dataset as its source. The record's published MD5 matches the downloaded file.

| Property | Observed value |
| --- | ---: |
| Raw CSV bytes | 150,828,752 |
| MD5 | `e90efcb83d69faf99fcab8b0255024de` |
| SHA-256 | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| Transactions | 284,807 |
| Fraud | 492 |
| Legitimate | 284,315 |
| Fraud prevalence | 0.1727% |
| Feature columns | Time, V1 through V28, Amount |
| Label | Class, with 1 for fraud |
| Exact feature groups | 283,726 |
| Extra rows in duplicate groups | 1,081 |

The validator checked the exact ordered schema, numeric and finite values, nonnegative Time and Amount, binary labels, source checksum, and expected original counts. Exact feature duplicates remain in one client and one split. Identical feature rows with conflicting labels would stop preparation. No rows were dropped or resampled.

## Simulated client distributions

The source does not contain bank identity. Bank A through Bank D are controlled simulations. Groups of identical feature rows were assigned by label-specific Dirichlet draws. The mild setting used concentration 10.0 and the strong setting used 0.5. Both accepted the requested seed 42 on the first attempt. Allocation required at least 12 distinct fraud groups and 1,000 legitimate groups in each client so every split remained evaluable. Each bank's groups were then stratified into approximately 70% training, 15% validation, and 15% test.

| Setting | Simulated client | Transactions | Fraud | Fraud prevalence |
| --- | --- | ---: | ---: | ---: |
| Mild | Bank A | 66,290 | 133 | 0.2006% |
| Mild | Bank B | 74,562 | 115 | 0.1542% |
| Mild | Bank C | 58,329 | 141 | 0.2417% |
| Mild | Bank D | 85,626 | 103 | 0.1203% |
| Strong | Bank A | 208,421 | 86 | 0.0413% |
| Strong | Bank B | 20,373 | 69 | 0.3387% |
| Strong | Bank C | 15,105 | 157 | 1.0394% |
| Strong | Bank D | 40,908 | 180 | 0.4400% |

Each setting covers all 284,807 original rows and all 492 fraud cases exactly once. The strong setting intentionally produces a large size and prevalence imbalance. It represents simulated heterogeneity, not real institutions or known customer categories. No feature-distribution grouping was applied; the controlled difference is label prevalence and client size.

## Split and transformation checks

| Setting | Training rows and fraud | Validation rows and fraud | Test rows and fraud |
| --- | ---: | ---: | ---: |
| Mild | 199,341 / 343 | 42,713 / 73 | 42,753 / 76 |
| Strong | 199,368 / 347 | 42,714 / 71 | 42,725 / 74 |

For each bank, `StandardScaler` was fit only on its training Time and Amount values. The same fitted mean and scale were applied to that bank's validation and test rows. V1 through V28 retained their source values, converted to float32 for model input. The independent artifact check recalculated training means and scales from the raw rows, reproduced the full assignment from seed 42, checked all row and label counts, verified duplicate boundaries, and compared each stored transformed file with the raw source and scaler. It passed for both settings. Eleven project tests passed, including synthetic tests for invalid source data, deterministic grouping, and scaler tampering.

The raw CSV is at `data/raw/creditcard.csv`. Offline manifests and reports are under `data/processed/<setting>`. Each bank's transformed splits are under `clients/<bank>/data/<setting>` and scaler metadata under `clients/<bank>/preprocessing`. These directories are ignored by version control. Detailed machine-readable verification results are in `runs/data_verification.json`.

The ULB benchmark covers only two days, and V1 through V28 are anonymized PCA components. Their values support modeling but not business descriptions. A separate temporal holdout remains an optional later experiment; this preparation uses stratified random splits.
