# Foundation verification

Completed 8 November 2026 on Windows 11, native PowerShell and 64-bit Python 3.13.5.

| Check | Observed result |
| --- | --- |
| Machine | 16 logical processors; 15.6 GiB RAM; 4.8 GiB available at inspection; no CUDA device; WSL2 not installed |
| Clean initial environment | New project-local `.venv` created; core packages installed successfully |
| Dependency snapshot | 89 exact package pins in `requirements.lock`; `pip check` reported no broken requirements |
| Setup script | Completed installation check and project storage initialization |
| Project tests | 8 passed |
| PyTorch | Finite gradient produced by a backward pass |
| Flower serialization | Tensor round-trip preserved values |
| SHAP | Synthetic explanation reconstructed its model output |
| Streamlit | AppTest rendered without exceptions; local health endpoint returned HTTP 200 and `ok` |
| Flower transport | One actual local server and four independent client processes completed one FedAvg round; weighted result 3.0 matched the expected value |
| Verification script | `scripts/verify.ps1` completed with exit code 0 |

The Flower probe uses synthetic constants solely to verify process connectivity and aggregation. Its result is an environment check, separate from the fraud model results reported in the dataset, baseline, federated, and study reports.

The installed package snapshot is specific to this Python and Windows platform. Flower's diagnostic transport API is deprecated and isolated; the fraud training application uses the supported ClientApp and ServerApp deployment route.

Evidence generated locally and excluded from version control: `runs/environment.json`, `runs/flower_smoke/result.json`, Flower process logs, and application logs. Reproduce checks with the PowerShell commands in `README.md`.
