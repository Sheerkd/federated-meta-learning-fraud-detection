# Federated averaging baseline

## Completed experiment

Four simulated banks completed five Flower FedAvg rounds on each of the mild and strong ULB partitions, using seed 42, one local epoch per round, and the shared PyTorch MLP. Training ran through Flower ClientApp and ServerApp with a localhost SuperLink and four SuperNodes. The server required all four banks for every training and validation round.

Each client read only its bank's prepared train and validation files during Flower training. Replies carried model tensors, aggregate metrics, and audit metadata; no transaction rows were included. The server checked client identity, round, starting model hash, sample counts, tensor schema, and the numerical weighted average before saving each checkpoint. The Flower application bundle contained no dataset, client files, or run artifacts.

## Held-out results

Thresholds were selected on each bank's validation set and applied unchanged to its test set. A separate offline evaluator loaded the saved global model and the public benchmark test partitions. The table reports unweighted means of four bank test metrics; it is not pooled average precision.

| Partition | Macro precision | Macro recall | Macro F1 | Macro false positive rate | Macro ROC-AUC | Macro average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mild | 0.8378 | 0.8289 | 0.8312 | 0.000284 | 0.9793 | 0.7658 |
| Strong | 0.7684 | 0.6843 | 0.7182 | 0.000493 | 0.9465 | 0.6254 |

| Partition | Bank A F1 / AP | Bank B F1 / AP | Bank C F1 / AP | Bank D F1 / AP |
| --- | ---: | ---: | ---: | ---: |
| Mild | 0.8889 / 0.8146 | 0.8235 / 0.8113 | 0.8000 / 0.6622 | 0.8125 / 0.7751 |
| Strong | 0.4615 / 0.2813 | 0.7059 / 0.5568 | 0.9130 / 0.9496 | 0.7925 / 0.7137 |

The strong partition exposes substantial bank variation. These are one-seed results and do not establish a statistically reliable improvement over the local or centralized baselines in [BASELINE_REPORT.md](BASELINE_REPORT.md).

## Run evidence and boundaries

- Each setting saved an initial checkpoint and five round checkpoints. Independent verification reproduced checkpoint lineage, four client replies per round, final validation threshold choices, and all four test metric sets.
- The model tensors occupy 16,388 bytes. Approximate tensor traffic per setting was 327,760 bytes for training broadcasts, 327,760 for returned updates, and 327,760 for validation broadcasts: 983,280 bytes total. This excludes Flower protocol and transport overhead.
- An existing seed and setting run is refused rather than overwritten. Interrupted runs are not resumed automatically.
- Transport is plaintext on localhost. These simulated banks share one machine and originate from a public dataset. The offline evaluator intentionally reads public client partitions, outside the federated server; this setup is not a production privacy or security guarantee.

Reproduce verification with `python -m fraud_detection verify-fedavg`. Raw run evidence is under `runs/federated/<setting>/seed_42/`; the consolidated verification result is `runs/fedavg_verification.json`.
