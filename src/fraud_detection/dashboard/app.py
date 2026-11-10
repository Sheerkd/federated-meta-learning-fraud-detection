"""Dashboard for saved models, bank-local predictions, and experiment progress."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fraud_detection.config import load_config
from fraud_detection.dashboard.service import (available_methods, comparison_rows,
    dataset_summary, job_rows, launch_run, parse_upload, predict_sample,
    predict_upload, round_rows, sample_options)
from fraud_detection.data.pipeline import BANKS, FEATURES
from fraud_detection.explainability.local import explain_bank_record


st.set_page_config(page_title="Personalized Fraud Detection", page_icon="🔎", layout="wide")
st.title("Personalized Credit Card Fraud Detection")
st.caption("Research prototype · Four simulated banks · Public ULB credit card data")
root = Path.cwd()
try:
    config = load_config(root / "configs/default.toml")
except (OSError, ValueError) as exc:
    st.error(f"Project configuration is unavailable: {exc}")
    st.stop()

with st.sidebar:
    st.header("Research setting")
    setting = st.selectbox("Bank distribution", ["mild", "strong"])
    bank = st.selectbox("Simulated bank", list(BANKS))
    seed = st.number_input("Saved run seed", min_value=0, max_value=2**32 - 1,
                           value=config.seed, step=1)
    st.caption("The bank selector is a demonstration control, not a real access boundary.")

methods = available_methods(root, setting=setting, seed=int(seed), runs_dir=config.paths.runs)
overview, detection, explanation, rounds_tab, comparison_tab, training = st.tabs(
    ["Overview", "Detection", "Explanation", "Federated rounds", "Comparison", "Training"])

with overview:
    st.subheader("Dataset and saved work")
    try:
        summary = dataset_summary(root, setting=setting, data_dir=config.paths.data)
        c1, c2, c3 = st.columns(3)
        c1.metric("Source transactions", f"{summary['source_rows']:,}")
        c2.metric("Source fraud cases", f"{summary['source_fraud']:,}")
        c3.metric("Fraud prevalence", f"{summary['source_prevalence']:.3%}")
        st.dataframe(pd.DataFrame([
            {"Bank": client, "Split": split, "Transactions": values["rows"],
             "Fraud": values["fraud"], "Prevalence": values["fraud_prevalence"]}
            for client, splits in summary["banks"].items() for split, values in splits.items()]),
            hide_index=True, width="stretch")
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Prepared dataset is unavailable: {exc}")
    st.write("Available saved models:", ", ".join(methods) if methods else "none for this seed")
    st.info("Scores are research outputs from simulated banks, not calibrated real-world fraud probabilities.")

with detection:
    st.subheader("Score a bank-local transaction")
    if not methods:
        st.warning("No completed model is available for this setting and seed.")
    else:
        method = st.selectbox("Saved model", methods,
                              index=methods.index("Personalized FedMAML")
                              if "Personalized FedMAML" in methods else 0)
        try:
            examples = sample_options(root, bank=bank, setting=setting,
                                      clients_dir=config.paths.clients)
            choice = st.selectbox("Held-out example", examples,
                                  format_func=lambda item: f"Row {item['row_id']} · known {'fraud' if item['label'] else 'legitimate'}")
            if st.button("Score selected example"):
                st.session_state["prediction"] = predict_sample(
                    root, bank=bank, setting=setting, seed=int(seed), method=method,
                    row_id=choice["row_id"], clients_dir=config.paths.clients,
                    runs_dir=config.paths.runs)
            saved = st.session_state.get("prediction")
            if saved and (saved["bank"], saved["setting"], saved["method"], saved["row_id"]) == (
                    bank, setting, method, choice["row_id"]):
                a, b, c = st.columns(3)
                a.metric("Fraud score", f"{saved['score']:.4f}")
                b.metric("Bank threshold", f"{saved['threshold']:.4f}")
                c.metric("Decision", "Flagged" if saved["flagged"] else "Not flagged")
                st.caption(f"Model {saved['model_sha256'][:16]}… · Held-out label "
                           f"{'fraud' if saved['held_out_label'] else 'legitimate'}")
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f"Held-out prediction is unavailable: {exc}")
        st.divider()
        st.subheader("Score an uploaded CSV")
        st.caption("Upload original-scale ULB features in order: Time, V1–V28, Amount. "
                   "Time and Amount use this bank's training-only scaler. Limit: 1,000 rows, 2 MB.")
        uploaded = st.file_uploader("Transaction CSV", type="csv")
        if uploaded is not None:
            try:
                raw = uploaded.getvalue()
                preview = parse_upload(raw)
                st.write(f"Validated {len(preview)} transaction(s).")
                if st.button("Score uploaded transactions"):
                    output = predict_upload(root, bank=bank, setting=setting, seed=int(seed),
                                            method=method, file_bytes=raw,
                                            clients_dir=config.paths.clients,
                                            runs_dir=config.paths.runs)
                    st.dataframe(output, hide_index=True, width="stretch")
                    st.download_button("Download scores", output.to_csv(index=False),
                                       file_name=f"{bank}_{setting}_scores.csv", mime="text/csv")
            except (OSError, ValueError, KeyError) as exc:
                st.error(f"CSV could not be scored: {exc}")

with explanation:
    st.subheader("Explain a saved prediction")
    st.caption("Explanations stay in this bank's local files. V1–V28 are PCA components; "
               "contributions describe model behavior, not causes.")
    explain_methods = [value for value in ("Personalized FedMAML", "FedMAML", "FedAvg") if value in methods]
    if not explain_methods:
        st.warning("A completed federated model is needed for an explanation.")
    else:
        explain_method = st.selectbox("Model to explain", explain_methods)
        try:
            examples = sample_options(root, bank=bank, setting=setting,
                                      clients_dir=config.paths.clients)
            choice = st.selectbox("Transaction to explain", examples,
                                  format_func=lambda item: f"Row {item['row_id']} · known {'fraud' if item['label'] else 'legitimate'}")
            if st.button("Create or load explanation"):
                path = explain_bank_record(root, bank=bank, setting=setting, seed=int(seed),
                    row_id=choice["row_id"],
                    method="fedavg" if explain_method == "FedAvg" else "fedmaml",
                    support_budget=32 if explain_method == "Personalized FedMAML" else None,
                    steps=5, clients_dir=config.paths.clients, runs_dir=config.paths.runs)
                st.session_state["explanation_path"] = str(path)
            saved_path = st.session_state.get("explanation_path")
            if saved_path and Path(saved_path).is_file():
                path = Path(saved_path)
                artifact = json.loads(path.read_text(encoding="utf-8"))
                expected_method = "fedavg" if explain_method == "FedAvg" else "fedmaml"
                expected_budget = 32 if explain_method == "Personalized FedMAML" else None
                if (artifact["bank"] == bank and artifact["setting"] == setting
                        and artifact["row_id"] == choice["row_id"]
                        and artifact["method"] == expected_method
                        and (artifact["model"]["adaptation"] or {}).get("support_budget") == expected_budget):
                    st.metric("Explained fraud score", f"{artifact['score']:.4f}")
                    st.write(f"Background score {artifact['expected_probability']:.4f} · "
                             f"Reconstruction error {artifact['reconstruction_error']:.2g}")
                    values = np.array([artifact["feature_contributions"][name] for name in FEATURES])
                    waterfall = go.Figure(go.Waterfall(orientation="h",
                        measure=["relative"] * len(FEATURES), y=FEATURES, x=values))
                    waterfall.update_layout(title="Contributions to predicted fraud probability", height=850)
                    st.plotly_chart(waterfall, width="stretch")
                    top = np.argsort(np.abs(values))[-15:]
                    bars = go.Figure(go.Bar(x=values[top], y=[FEATURES[i] for i in top], orientation="h"))
                    bars.update_layout(title="Largest feature contributions", height=600)
                    st.plotly_chart(bars, width="stretch")
                    st.dataframe(pd.DataFrame({"Feature": FEATURES,
                        "Value used by model": [artifact["feature_values"][name] for name in FEATURES],
                        "Probability contribution": values}), hide_index=True, width="stretch")
                    st.download_button("Download explanation JSON", path.read_bytes(),
                        file_name=f"{bank}_{setting}_{artifact['row_id']}_explanation.json",
                        mime="application/json")
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            st.error(f"Explanation is unavailable: {exc}")

with rounds_tab:
    st.subheader("Federated round progress")
    round_methods = [value for value, family in (("FedAvg", "federated"), ("FedMAML", "meta"))
                     if (root / config.paths.runs / family / setting / f"seed_{int(seed)}" / "run.json").is_file()]
    if not round_methods:
        st.warning("No federated run has started for this seed.")
    else:
        round_method = st.selectbox("Federated method", round_methods)
        try:
            rounds = round_rows(root, setting=setting, seed=int(seed), method=round_method,
                                runs_dir=config.paths.runs)
            st.dataframe(rounds, hide_index=True, width="stretch")
            if not rounds.empty:
                st.line_chart(rounds.set_index("Round")[["Mean local loss"]])
                st.caption("Tensor bytes exclude network protocol overhead.")
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f"Round history is unavailable: {exc}")

with comparison_tab:
    st.subheader("Held-out model comparison")
    try:
        table = comparison_rows(root, setting=setting, seed=int(seed),
                                runs_dir=config.paths.runs)
        if table.empty:
            st.warning("No completed comparison results are available.")
        else:
            macro = table.groupby("Method", sort=False)[["Precision", "Recall", "F1",
                "False positive rate", "ROC AUC", "Average precision"]].mean().reset_index()
            st.caption("Macro means give each bank equal weight. Personalized FedMAML uses "
                       "32 support rows and five steps. Single-seed results are preliminary.")
            st.dataframe(macro, hide_index=True, width="stretch")
            st.plotly_chart(px.bar(table, x="Bank", y="F1", color="Method", barmode="group",
                                   title="Test F1 by bank"), width="stretch")
            st.dataframe(table, hide_index=True, width="stretch")
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        st.warning(f"Comparison results are unavailable: {exc}")

with training:
    st.subheader("Start a research run")
    st.caption("Training starts only when requested and continues in a separate process. "
               "Existing settings and seeds are protected from overwrite.")
    run_method = st.selectbox("Training method", ["FedAvg", "FedMAML"])
    run_seed = st.number_input("New run seed", min_value=0, max_value=2**32 - 1,
                               value=min(int(seed) + 1, 2**32 - 1), step=1)
    if st.button("Start background run"):
        try:
            job_dir = launch_run(root, method=run_method, setting=setting,
                                 seed=int(run_seed), runs_dir=config.paths.runs)
            st.success(f"Run started. Progress is recorded in {job_dir}.")
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Run could not start: {exc}")
    if st.button("Refresh run status"):
        st.rerun()
    jobs = job_rows(root, runs_dir=config.paths.runs)
    if not jobs.empty:
        st.dataframe(jobs, hide_index=True, width="stretch")
