# Local and centralized fraud model baselines

Completed 8 November 2026 using the verified ULB dataset and the four simulated client partitions. These are measured single-seed baseline results, not a conclusion about federated learning or meta-learning.

## Training and selection

Every model uses a 30-input MLP with hidden layers of 64 and 32 ReLU units, dropout 0.2, and one raw logit. Training uses Adam, `BCEWithLogitsLoss`, a positive-class weight calculated only from the training rows, batch size 2,048, learning rate 0.001, and a five-epoch budget. The epoch with the highest validation average precision is retained. Each bank's threshold maximizes its own validation F1, with recall and then a lower threshold breaking ties. Test rows are read only after both selections.

The four local models each train on one bank's training rows. The centralized reference trains on the pooled training rows of the four public-data simulations. All models use the same bank-specific transforms and evaluate on identical bank test rows. This makes the centralized model a comparison reference, not a privacy-preserving deployment design or guaranteed upper bound. This report records the initial seed; the [three-seed study](STUDY_REPORT.md) reports variation across training seeds.

## Test results

Macro values average the four bank metrics equally. Average precision is the ranking metric used for rare fraud; F1, precision, and recall use the validation-selected threshold. False positive rate is the fraction of genuine transactions flagged.

| Client setting | Method | Macro precision | Macro recall | Macro F1 | Macro average precision | Macro ROC-AUC | Macro false positive rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mild | Local | 0.8433 | 0.7942 | 0.8156 | 0.7866 | 0.9763 | 0.000259 |
| Mild | Centralized | 0.8237 | 0.8455 | 0.8308 | 0.7969 | 0.9790 | 0.000334 |
| Strong | Local | 0.7713 | 0.6734 | 0.7106 | 0.6135 | 0.9594 | 0.000293 |
| Strong | Centralized | 0.7910 | 0.6943 | 0.7331 | 0.6435 | 0.9649 | 0.000310 |

| Client setting | Method | Simulated client | Test fraud / rows | F1 | Average precision |
| --- | --- | --- | ---: | ---: | ---: |
| Mild | Local | Bank A | 22 / 9,942 | 0.8571 | 0.8362 |
| Mild | Local | Bank B | 18 / 11,193 | 0.8235 | 0.8024 |
| Mild | Local | Bank C | 21 / 8,748 | 0.7692 | 0.6884 |
| Mild | Local | Bank D | 15 / 12,870 | 0.8125 | 0.8195 |
| Mild | Centralized | Bank A | 22 / 9,942 | 0.8511 | 0.8362 |
| Mild | Centralized | Bank B | 18 / 11,193 | 0.8235 | 0.8209 |
| Mild | Centralized | Bank C | 21 / 8,748 | 0.8000 | 0.6965 |
| Mild | Centralized | Bank D | 15 / 12,870 | 0.8485 | 0.8340 |
| Strong | Local | Bank A | 12 / 31,277 | 0.4138 | 0.2410 |
| Strong | Local | Bank B | 10 / 3,055 | 0.7059 | 0.5622 |
| Strong | Local | Bank C | 23 / 2,258 | 0.9302 | 0.9144 |
| Strong | Local | Bank D | 29 / 6,135 | 0.7925 | 0.7364 |
| Strong | Centralized | Bank A | 12 / 31,277 | 0.5185 | 0.2718 |
| Strong | Centralized | Bank B | 10 / 3,055 | 0.7059 | 0.6024 |
| Strong | Centralized | Bank C | 23 / 2,258 | 0.9302 | 0.9554 |
| Strong | Centralized | Bank D | 29 / 6,135 | 0.7778 | 0.7443 |

Only 10 to 29 fraud cases appear in each strong-setting client test split. Consequently, a few decisions can substantially change a bank's score. These single-seed results cannot establish that one method consistently wins. Multi-seed and federated comparisons remain in the final evaluation stage. Sigmoid scores have not been calibrated as real-world probabilities.

## Saved artifacts and verification

The complete per-bank metrics, confusion counts, threshold values, model histories, configuration, package versions, dataset checksum, and partition checksum are in `runs/baselines/<setting>/seed_42/results.json` and `test_metrics.csv`. Local model bundles are under `clients/<bank>/models/<setting>`; centralized bundles are in the corresponding run directory. These files remain local and are ignored by version control.

The independent verifier reloaded all ten checkpoints, checked their SHA-256 hashes and feature order, recalculated all eight bank thresholds per setting from validation data, and reproduced all sixteen bank test metric sets across the two settings. It also checked each stored macro result. Fourteen project tests passed, including a complete synthetic baseline run and model reload. Machine-readable verification evidence is in `runs/baseline_verification.json`.
