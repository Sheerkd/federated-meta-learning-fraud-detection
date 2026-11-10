# Three-seed scientific evaluation

## Scope and protocol

Four training methods were evaluated on the public ULB dataset: bank-local models, a centralized reference, FedAvg, and first-order FedMAML. FedMAML is shown both before and after a fixed personalization setting of 32 bank training support rows and five SGD steps; these are two evaluations of one training method. Its SHAP explanations use that same personalized model, not a fifth training method or an added predictive improvement. The study uses seeds 42, 43, and 44 in both mild and strong four-bank settings. The bank partition, validation rows, and untouched test rows are fixed across seeds, so the standard deviations below measure training randomness conditional on these two partitions.

All methods use the same 30-feature MLP and bank-specific thresholds selected from validation data. Local and centralized models each train for five full-data epochs. FedAvg uses five rounds with one local full-data epoch per round. FedMAML uses five rounds of one class-balanced 32-support/32-query episode per bank, then the fixed 32-row/five-step adaptation. The five-cycle schedule is matched, and the bank test data is identical. **Training-row exposure is not matched:** local, centralized, and FedAvg process about 996,700 rows per setting across cycles; the first-order meta-learning and personalization path processes about 1,920 episode/support examples as counted in the budget table. This difference prevents attributing a performance gap solely to the learning algorithm. See [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) for the prespecified comparison rules.

## Held-out results

Values are equal-bank macro test means across the three seeds, shown as mean ± sample standard deviation. Average precision (AP) is averaged per bank; it is not pooled AP. The personalized row uses the 32-support/five-step setting fixed before compiling this study.

| Setting | Evaluation variant | Macro F1 | Macro AP | Macro recall | Macro false-positive rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Mild | Local | 0.8113 ± 0.0119 | 0.7838 ± 0.0060 | 0.7935 | 0.0003 |
| Mild | Centralized | 0.8300 ± 0.0079 | 0.7957 ± 0.0024 | 0.8284 | 0.0003 |
| Mild | FedAvg | 0.8324 ± 0.0046 | 0.7633 ± 0.0027 | 0.8227 | 0.0003 |
| Mild | FedMAML before adaptation | 0.0108 ± 0.0101 | 0.0034 ± 0.0017 | 0.2123 | 0.1985 |
| Mild | FedMAML after adaptation | 0.2095 ± 0.0434 | 0.1632 ± 0.0307 | 0.1976 | 0.0010 |
| Strong | Local | 0.7185 ± 0.0069 | 0.6210 ± 0.0088 | 0.6734 | 0.0003 |
| Strong | Centralized | 0.7338 ± 0.0087 | 0.6396 ± 0.0041 | 0.6982 | 0.0003 |
| Strong | FedAvg | 0.7132 ± 0.0087 | 0.6221 ± 0.0029 | 0.6843 | 0.0005 |
| Strong | FedMAML before adaptation | 0.0108 ± 0.0039 | 0.0102 ± 0.0037 | 0.0462 | 0.0266 |
| Strong | FedMAML after adaptation | 0.2335 ± 0.0541 | 0.2329 ± 0.0600 | 0.3465 | 0.0067 |

The small mild FedAvg F1 difference from the centralized reference does not establish superiority with three seeds. FedMAML improves relative to its weak global initialization after adaptation, but remains well below local, centralized, and FedAvg results in this configuration. In strong skew, personalized FedMAML's bank A mean F1 was 0 across the three seeds, while FedAvg bank A mean F1 was 0.4615. Bank-level prevalence, thresholds, F1, and AP for all 120 bank/seed/variant combinations are in `runs/study/per_bank.csv`; `bank_summary.csv` gives their means and standard deviations.

## Adaptation and communication

The full prespecified exploratory grid contains support budgets 32, 128, and 512 and adaptation steps 1, 3, and 5. For the 32-row budget, macro F1 changed from 0.0259 → 0.0909 → 0.2095 in mild skew and from 0.0577 → 0.1348 → 0.2335 in strong skew as steps increased from 1 → 3 → 5. These are descriptive test outcomes, not a basis to select a new setting on test data. The complete grid and its across-seed summaries are in `adaptation.csv`, `adaptation_macro.csv`, and `adaptation_summary.csv`.

Both federated methods transferred approximately 983,280 model-tensor bytes per five-round run in either setting: training broadcasts, returned weights or query gradients, and validation broadcasts. This excludes Flower protocol, transport, and metadata overhead. Sum of recorded round durations averaged 111.7 seconds for mild FedAvg, 115.4 for mild FedMAML, 112.1 for strong FedAvg, and 113.6 for strong FedMAML. These are round timings, not full launch-to-exit times. Per-run and across-seed details are in `communication.csv` and `communication_summary.csv`.

## Evidence and reproducibility

The saved study under `runs/study/` contains `study.json`, ten CSV tables, and standalone interactive figures: `macro_f1.html`, `seed_variation.html`, `bank_f1.html`, `adaptation.html`, and `communication.html`. `python -m fraud_detection verify-study` recomputes the study from source artifacts and checks the CSV tables. `python -m fraud_detection verify-study-sources` independently reloads and checks all saved local/centralized, FedAvg, and FedMAML runs and bank-local explanations. The latter passed 18 source-run checks and verified 25 explanation artifacts. The 24 required personalized E4 explanation-to-model hash matches passed across four banks, two settings, and three seeds. The full project suite passed 28 tests.

The fixed partitions and public dataset do not establish generalization to unseen banks or production fraud streams. The sigmoid scores have not been calibrated for deployment, and these simulations do not provide real-bank privacy or security guarantees. No statistical significance claim is made from three seeds on fixed partitions.
