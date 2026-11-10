"""One FedAvg client update from a received initialization."""

import numpy as np
import torch

from fraud_detection.models.fraud_model import FraudMLP


def train_local(model: FraudMLP, x: np.ndarray, y: np.ndarray, *, epochs: int,
                batch_size: int, learning_rate: float, weight_decay: float,
                max_grad_norm: float, seed: int) -> float:
    if len(x) != len(y) or len(x) == 0 or len(np.unique(y)) != 2:
        raise ValueError("Client training requires aligned data and both classes")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0 or max_grad_norm <= 0:
        raise ValueError("Invalid local training configuration")
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    features = torch.from_numpy(x)
    labels = torch.from_numpy(y)
    weight = (len(y) - int(y.sum())) / int(y.sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    sum_loss = 0.0
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(features), generator=generator)
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features[indices]), labels[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            sum_loss += float(loss.detach()) * len(indices)
    model.eval()
    return sum_loss / (len(y) * epochs)
