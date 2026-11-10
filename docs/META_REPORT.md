# Federated meta-learning and personalization

## Completed implementation

Four Flower clients completed five first-order MAML rounds in each of the mild and strong ULB partitions with seed 42. Each bank sampled a disjoint, class-balanced support set of 32 and query set of 32 from its training rows, without replacement. A private model copy took one SGD support step at learning rate 0.01. Each bank sent the gradient of its query loss at the adapted parameters. The coordinator applied `theta = theta - 0.001 × mean(four query gradients)`. The first-order approximation omits differentiation through adaptation.

Messages explicitly identify query gradients and carry the algorithm, bank, round, starting model hash, schema, and aggregate episode metrics. The server rejected duplicate, stale, malformed, or nonfinite updates. It saved the initial model and every completed round. The local Flower package contained code only; no transaction rows were transmitted in replies. Runs on the same seed and setting refuse to overwrite saved results.

After training, an offline evaluator froze the final initialization. For each bank it measured the untouched test set before adaptation, then adapted a separate copy with support budgets 32, 128, and 512 and step counts 1, 3, and 5. Support comes only from that bank's training split. Sampling enriches fraud examples up to half the budget, limited by available fraud rows. Thresholds are selected on that bank's validation split and then applied to test data. The offline evaluator's access to public benchmark partitions is outside the Flower server.

## Preliminary held-out results

These are macro means across four banks. The displayed five-step configurations are members of the planned grid, not configurations selected on test outcomes.

| Partition and adaptation | Macro test F1 | Macro test average precision |
| --- | ---: | ---: |
| Mild, before adaptation | 0.0018 | 0.0015 |
| Mild, 32 support rows, 5 steps | 0.2595 | 0.1987 |
| Mild, 128 support rows, 5 steps | 0.2596 | 0.1719 |
| Mild, 512 support rows, 5 steps | 0.0127 | 0.0075 |
| Strong, before adaptation | 0.0075 | 0.0065 |
| Strong, 32 support rows, 5 steps | 0.2813 | 0.3008 |
| Strong, 128 support rows, 5 steps | 0.2110 | 0.2904 |
| Strong, 512 support rows, 5 steps | 0.0868 | 0.0705 |

The initialization and several adapted settings perform poorly. This implementation verifies the method and provides a measurable starting point; it does not demonstrate superiority to FedAvg or the baseline models. These are single-seed results; the [three-seed study](STUDY_REPORT.md) reports variation across training seeds and the unequal training-row exposure. Larger enriched support sets can reduce the minority-class fraction and change adaptation behavior.

## Verification

`python -m fraud_detection verify-fedmaml` independently replays all 5 rounds per setting from their saved initial checkpoints and original training rows. It checks every client's query-gradient hash, numerical outer update, checkpoint lineage, validation reply presence, final model hash, and all 72 personalized test result sets. The full project suite passes 20 tests, including manual outer-update arithmetic, disjoint support/query sampling, no mutation of the global initialization, and stale-message rejection.

Saved results are in `runs/meta/<setting>/seed_42/results.json`; per-round evidence and model checkpoints are beside them. The runs use localhost plaintext transport and simulated banks derived from one public dataset. They do not establish production privacy or generalization to unseen banks.
