from __future__ import annotations

import numpy as np
from numpy.typing import NDArray, ArrayLike


class SamplerBase:
    pass


class GaussSampler(SamplerBase):
    def __init__(
        self,
        ends: ArrayLike,
        means: ArrayLike,
    ):
        self._ends = np.asarray(ends)
        self._means = np.asarray(means)
        if self._ends.ndim != 1:
            raise ValueError("ends must be 1d")
        if self._means.ndim != 1:
            raise ValueError("means must be 1d")
        if self._ends.dtype.kind not in "ui":
            raise ValueError("ends must be integer")
        if not np.all(np.diff(ends) > 0):
            raise ValueError("ends must be monotonically increasing")

        if self._ends.size != self._means.size:
            raise ValueError("length of ends and means must be the same")

    def sample(
        self, sigma: float = 1.0, seed: int | None = None
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        out = np.empty(self._ends[-1], dtype=np.float64)
        pos = np.concatenate([[0], self._ends], axis=0)
        for i in range(self._ends.size):
            p0, p1 = pos[i], pos[i + 1]
            out[p0:p1] = rng.normal(self._means[i], sigma, size=p1 - p0)
        return out


class PoissonSampler(SamplerBase):
    def __init__(
        self,
        ends: ArrayLike,
        means: ArrayLike,
    ):
        self._ends = np.asarray(ends)
        self._means = np.asarray(means)
        if self._ends.ndim != 1:
            raise ValueError("ends must be 1d")
        if self._means.ndim != 1:
            raise ValueError("means must be 1d")
        if self._ends.dtype.kind not in "ui":
            raise ValueError("ends must be integer")
        if not np.all(np.diff(ends) > 0):
            raise ValueError("ends must be monotonically increasing")

        if self._ends.size != self._means.size:
            raise ValueError("length of ends and means must be the same")

    def sample(self, seed: int | None = None) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        out = np.empty(self._ends[-1], dtype=np.float64)
        pos = np.concatenate([[0], self._ends], axis=0)
        for i in range(self._ends.size):
            p0, p1 = pos[i], pos[i + 1]
            out[p0:p1] = rng.poisson(self._means[i], size=p1 - p0)
        return out
