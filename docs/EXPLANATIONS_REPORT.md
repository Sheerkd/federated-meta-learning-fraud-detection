# Bank-local explanations

## Completed workflow

The service loads a saved FedMAML or FedAvg checkpoint and verifies its model hash. For FedMAML, it reconstructs the requested personalized bank model from the recorded training support subset and confirms the resulting hash matches the saved evaluation. It reads only that bank's prepared data and local preprocessing reference. A user can select a bank-local test row by `row_id`; the default demonstration selects the first fraud test row.

SHAP KernelExplainer uses up to 100 randomly sampled training rows from the selected bank as its background. The default is 32 background rows and 128 approximation samples. Its prediction function returns the model's sigmoid fraud score, so the expected value and each feature contribution are on the predicted-probability scale. Dropout is disabled for explanation and prediction. The service checks that expected probability plus all 30 contributions reconstructs the score within 0.001, then saves a JSON explanation and interactive waterfall and bar HTML plots.

The cache key includes the model hash, local preprocessing file hash, background hash, transaction feature hash, SHAP version, and approximation budget. Individual explanations and plots live under `clients/<bank>/explanations/`; the federated coordinator does not receive them. A changed model, preprocessing reference, background, or transaction creates a different cache entry.

## Real-data verification

One held-out fraud transaction from every bank in each mild and strong setting was explained using that bank's 32-support-row, five-step personalized FedMAML model: eight explanations. One additional mild bank A explanation used the saved FedAvg model. All nine produced both plots. Independent verification reloaded their model, local test transaction, preprocessing reference, and training background; checked the prediction and expected probability; and checked the contribution sum. The maximum observed reconstruction error was 0.0 at saved numerical precision. The local cache and invalidation behavior were also tested with synthetic data. The complete project suite passed 21 tests.

Run `python -m fraud_detection verify-explanations` to check the saved artifacts. Use `python -m fraud_detection --setting mild --bank bank_a explain` for the default example; add `--row-id <id>` to select another local test record.

## Interpretation and limits

The ULB `V1` through `V28` columns are PCA components, so the plots keep those names and do not invent business meanings. `Time` and `Amount` are bank-standardized model inputs in these artifacts. SHAP approximates contributions for this model and background; it does not prove why a transaction occurred or whether it is fraudulent. A reconstructed sum can be exact even when individual approximate contributions vary with the SHAP sampling budget. This is a local research demonstration on a public dataset, not an access-control boundary for real banking records.
