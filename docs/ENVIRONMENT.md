# Development environment

## Selected route

Use native Windows x64 with Python 3.13 in a project-local virtual environment. The machine exposes 16 logical processors and approximately 15.6 GiB physical RAM. WSL2 is not installed. Initial configuration uses CPU; GPU acceleration is not required.

The foundation verifies a real Flower server and four independent client processes over localhost gRPC, using synthetic constant arrays and a known weighted mean. The subsequent fraud experiment uses Flower ClientApp, ServerApp, SuperLink, and four SuperNodes; five real FedAvg rounds completed in both partition settings.

## Compatibility boundary

Flower's retained `start_server` and `start_client` calls are deprecated. They are restricted to the foundation transport probe in `federated/smoke.py`. The supported ClientApp/ServerApp deployment route was verified for actual training on this Windows environment.

The diagnostic disables Flower telemetry in child processes, binds to 127.0.0.1, uses an available temporary port, and closes all child processes in a finally block. Plaintext transport is limited to this local synthetic diagnostic. Cross-host training requires a separate TLS configuration.

## Reproducibility

`requirements.lock` is an exact pip dependency snapshot, including transitive packages and build tooling, for the tested platform. `pyproject.toml` describes direct requirements; the lock controls the tested installation. It is not a hash-verified, universal multi-platform lock. Setup installs the snapshot before installing the application with dependency resolution and build isolation disabled.

`scripts/verify.ps1` checks dependency consistency, project tests, functional library operations, and four-client transport. Its output and generated run files provide repeatable evidence. Dataset and model work remain separate acceptance gates.

Foundation verification passed on Windows 11 with Python 3.13.5, PyTorch 2.15.0, Flower 1.43.0, SHAP 0.54.1, and Streamlit 1.67.0. The complete version snapshot is in `requirements.lock`; detailed observed checks are in `FOUNDATION_REPORT.md`.

## Sources checked during setup

- Flower package and Python support: https://pypi.org/project/flwr/
- Flower client API deprecation: https://flower.ai/docs/framework/1.34/en/ref-api/flwr.client.start_client.html
- Flower installation guidance: https://flower.ai/docs/framework/main/en/how-to-install-flower.html
- MAML and the project architecture are documented separately in ARCHITECTURE.md.
