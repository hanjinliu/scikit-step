from __future__ import annotations
import numpy as np


def normalize_sigma(sigma: float | None, data: np.ndarray) -> float:
    if sigma is not None:
        if sigma < 0.0:
            raise ValueError("sigma must be positive.")
        return sigma
    else:
        from scipy.stats import norm

        p = norm.cdf(1)  # = sigma for standard normal distribution.
        return np.quantile(np.diff(data), p) / np.sqrt(2)


def normalize_prob(prob: float | None, ndata: int) -> float:
    if prob is not None:
        if not 0.0 < prob < 0.5:
            raise ValueError("prob must be in range 0.0 < prob < 0.5.")
        return prob
    else:
        return 1 / (1 + np.sqrt(ndata))
