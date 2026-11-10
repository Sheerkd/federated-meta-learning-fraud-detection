"""First-order MAML client episode and transparent outer update."""

import numpy as np
import torch

from fraud_detection.federated.protocol import validate_state
from fraud_detection.models.fraud_model import FraudMLP, ModelSpec


ALGORITHM = "fedmaml"


def episode_indices(labels: np.ndarray, *, support_size: int, query_size: int,
                    seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw disjoint class-balanced support/query subsets from training rows."""
    if support_size < 2 or query_size < 2 or support_size % 2 or query_size % 2:
        raise ValueError("Episode sizes must be positive even numbers")
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        pool = np.flatnonzero(labels == label)
        needed = (support_size + query_size) // 2
        if len(pool) < needed:
            raise ValueError(f"Insufficient training examples of class {label} for episode")
        selected.append(rng.choice(pool, needed, replace=False))
    support = np.concatenate([part[:support_size // 2] for part in selected])
    query = np.concatenate([part[support_size // 2:] for part in selected])
    rng.shuffle(support)
    rng.shuffle(query)
    if np.intersect1d(support, query).size:
        raise RuntimeError("Support and query overlap")
    return support, query


def adapt(model: FraudMLP, x: np.ndarray, y: np.ndarray, *, steps: int,
          learning_rate: float, max_grad_norm: float = 5.0,
          seed: int = 0) -> None:
    if steps < 1 or learning_rate <= 0:
        raise ValueError("Invalid inner optimization settings")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        features = torch.as_tensor(x, dtype=torch.float32)
        labels = torch.as_tensor(y, dtype=torch.float32)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


def query_gradient(base_state: dict[str, torch.Tensor], spec: ModelSpec,
                   x: np.ndarray, y: np.ndarray, support: np.ndarray,
                   query: np.ndarray, *, steps: int, inner_lr: float,
                   max_grad_norm: float = 5.0, seed: int = 0) -> tuple[dict[str, torch.Tensor], float]:
    validate_state(base_state, spec)
    if np.intersect1d(support, query).size:
        raise ValueError("Support and query overlap")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = FraudMLP(spec)
        model.load_state_dict(base_state)
        adapt(model, x[support], y[support], steps=steps, learning_rate=inner_lr,
              max_grad_norm=max_grad_norm, seed=seed)
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(torch.as_tensor(x[query], dtype=torch.float32))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.as_tensor(y[query], dtype=torch.float32))
        loss.backward()
        gradients = {name: param.grad.detach().cpu().clone()
                     for name, param in model.named_parameters()}
    validate_state(gradients, spec)
    return gradients, float(loss.detach())


def outer_update(base_state: dict[str, torch.Tensor],
                 gradients: dict[str, dict[str, torch.Tensor]], *, beta: float,
                 spec: ModelSpec) -> dict[str, torch.Tensor]:
    validate_state(base_state, spec)
    if not gradients or beta <= 0:
        raise ValueError("Missing gradients or invalid outer learning rate")
    for client_gradient in gradients.values():
        validate_state(client_gradient, spec)
    return {name: base_state[name] - beta * torch.stack(
        [client_gradient[name] for client_gradient in gradients.values()]).mean(0)
            for name in base_state}
