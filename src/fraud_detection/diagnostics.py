"""Small functional checks; synthetic data here is never research evidence."""

import importlib.metadata
import os
import platform
import sys


def check_environment() -> dict:
    import numpy as np
    import pandas as pd
    import torch
    import shap
    import flwr
    from sklearn.metrics import average_precision_score
    from streamlit.testing.v1 import AppTest

    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    x = torch.tensor([[0., 0.], [1., 1.]])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).ravel(), torch.tensor([0., 1.]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    arrays = [np.array([1., 2.], dtype=np.float32)]
    restored = flwr.common.parameters_to_ndarrays(flwr.common.ndarrays_to_parameters(arrays))
    np.testing.assert_array_equal(arrays[0], restored[0])
    predict = lambda rows: np.asarray(rows).sum(axis=1)
    background = np.zeros((2, 2))
    explainer = shap.KernelExplainer(predict, background)
    values = explainer.shap_values(np.array([[1., 2.]]), nsamples=4, silent=True)
    np.testing.assert_allclose(np.asarray(values).sum() + explainer.expected_value, 3., atol=1e-5)
    assert average_precision_score([0, 1], [0.1, 0.9]) == 1.0
    assert len(pd.DataFrame({"synthetic": [1, 2]})) == 2
    app = AppTest.from_string('import streamlit as st\nst.title("Environment check")').run()
    assert not app.exception
    return {
        "status": "passed", "python": sys.version, "platform": platform.platform(),
        "logical_cpus": os.cpu_count(), "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "flwr", "shap", "streamlit", "pandas", "numpy", "scikit-learn", "plotly", "pydantic")},
        "checks": ["torch backward", "Flower tensor serialization", "SHAP reconstruction", "metrics", "dataframe", "Streamlit render"],
    }
