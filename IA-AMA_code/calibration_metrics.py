
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS


def expected_calibration_error(probs: np.ndarray,
                               labels: np.ndarray,
                               n_bins: int = 15) -> float:

    assert probs.ndim == 2 and labels.ndim == 1 and probs.shape[0] == labels.shape[0]
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        prop = in_bin.mean()
        ece += prop * abs(accuracies[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def maximum_calibration_error(probs: np.ndarray,
                              labels: np.ndarray,
                              n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        gap = abs(accuracies[in_bin].mean() - confidences[in_bin].mean())
        mce = max(mce, gap)
    return float(mce)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:

    N, C = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), labels] = 1.0
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == labels).mean())


class TemperatureScaler(nn.Module):
    def __init__(self, init_T: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * float(init_T))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self,
            logits_val: torch.Tensor,
            labels_val: torch.Tensor,
            max_iter: int = 100) -> float:

        self.train()
        logits_val = logits_val.detach().float()
        labels_val = labels_val.detach().long()
        nll = nn.CrossEntropyLoss()
        opt = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = nll(self.forward(logits_val), labels_val)
            loss.backward()
            return loss

        opt.step(closure)
        self.eval()
        return float(self.temperature.item())


def reliability_diagram_data(probs: np.ndarray,
                             labels: np.ndarray,
                             n_bins: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidences > lo) & (confidences <= hi)
        cnt[i] = mask.sum()
        if cnt[i] > 0:
            acc[i] = accuracies[mask].mean()
    return centers, acc, cnt


def risk_coverage_curve(probs: np.ndarray,
                        labels: np.ndarray,
                        n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float32)

    order = np.argsort(-confidences)
    correct_sorted = correct[order]
    N = len(labels)

    cum_correct = np.cumsum(correct_sorted)
    k = np.arange(1, N + 1)
    coverages = k / N
    risks = 1.0 - cum_correct / k

    if n_points is not None and n_points < N:
        idxs = np.linspace(0, N - 1, n_points).astype(int)
        coverages = coverages[idxs]
        risks = risks[idxs]
    return coverages, risks


def risk_coverage_auc(probs: np.ndarray, labels: np.ndarray) -> float:

    cov, risk = risk_coverage_curve(probs, labels, n_points=None)
    # numpy 2.x renamed trapz -> trapezoid; keep both paths working.
    _trap = getattr(np, "trapezoid", None) or np.trapz
    return float(_trap(risk, cov))


def selective_f1_at_coverage(probs: np.ndarray,
                             labels: np.ndarray,
                             target_coverage: float = 0.85) -> Tuple[float, float]:
    from sklearn.metrics import f1_score
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    N = len(labels)
    k = max(1, int(round(target_coverage * N)))
    order = np.argsort(-confidences)[:k]
    tau = float(confidences[order].min())
    f1 = float(f1_score(labels[order], predictions[order], average="macro"))
    return f1, tau


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, C = 500, 4
    logits = rng.normal(size=(N, C)) * 3.0
    labels = rng.integers(0, C, size=N)
    flip = rng.random(N) < 0.2
    labels[flip] = (labels[flip] + 1) % C
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    print("[synthetic sanity test]")
    print(f"  acc  = {accuracy(probs, labels):.4f}")
    print(f"  ECE  = {expected_calibration_error(probs, labels):.4f}")
    print(f"  MCE  = {maximum_calibration_error(probs, labels):.4f}")
    print(f"  Brier= {brier_score(probs, labels):.4f}")
    print(f"  RCauc= {risk_coverage_auc(probs, labels):.4f}")

    ts = TemperatureScaler(init_T=1.5)
    T = ts.fit(torch.tensor(logits, dtype=torch.float32),
               torch.tensor(labels, dtype=torch.long))
    print(f"  learned T = {T:.4f}")
    logits_cal = logits / T
    probs_cal = np.exp(logits_cal - logits_cal.max(axis=1, keepdims=True))
    probs_cal = probs_cal / probs_cal.sum(axis=1, keepdims=True)
    print(f"  ECE(+TS) = {expected_calibration_error(probs_cal, labels):.4f}")

    f1, tau = selective_f1_at_coverage(probs_cal, labels, 0.8)
    print(f"  sel-F1@0.8 = {f1:.4f}, tau={tau:.4f}")
