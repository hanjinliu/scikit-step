from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import NDArray
from ._base import RecursiveStepFinder
from ._moments import RecursiveMoment
from ._utils import normalize_sigma


class TtestMoment(RecursiveMoment):
    def __init__(
        self,
        fw: NDArray[np.number],
        bw: NDArray[np.number],
        total: NDArray[np.number],
        alpha: float,
        sigma: float,
    ):
        from scipy.stats import t

        super().__init__(fw, bw, total)
        self._alpha = alpha
        self._sigma = sigma
        self._t = t

    def get_optimal_splitter(self):
        n = np.arange(1, len(self))
        tk = np.abs(self.fw[0] / n - self.bw[0] / n[::-1]) / np.sqrt(
            1 / n + 1 / (n[::-1])
        )
        x = int(np.argmax(tk))
        return tk[x], x + 1

    def with_fw_bw(
        self,
        fw: NDArray[np.number],
        bw: NDArray[np.number],
        total: NDArray[np.number],
    ) -> TtestMoment:
        """Return a new Moment object with the given array."""
        return self.__class__(fw, bw, total, self._alpha, self._sigma)

    def _continue(self, tk: Any) -> bool:
        t_cri = self._t.ppf(1 - self._alpha / 2, len(self))
        return t_cri < tk / self._sigma


class TtestStepFinder(RecursiveStepFinder):
    """
    T-test based step finding.

    Reference
    ---------
    Shuang, B., Cooper, D., Taylor, J. N., Kisley, L., Chen, J., Wang, W., ... & Landes,
    C. F. (2014). Fast step transition and state identification (STaSI) for discrete
    single-molecule data analysis. The journal of physical chemistry letters, 5(18), 3157-3161.
    https://doi.org/10.1021/jz501435p
    """

    def __init__(self, alpha: float = 0.05, sigma: float | None = None):
        from scipy.stats import t as student_t

        if not 0 < alpha < 0.5:
            raise ValueError("alpha must be in range 0.0 < alpha < 0.5.")
        self._alpha = alpha
        self._student_t = student_t
        self._sigma = sigma

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def sigma(self) -> float | None:
        return self._sigma

    def get_params(self) -> dict[str, float | None]:
        return {"alpha": self.alpha, "sigma": self.sigma}

    def _moment_from_array(self, data):
        sigma = normalize_sigma(self._sigma, data)
        alpha = self._alpha
        return TtestMoment(*TtestMoment.calculate_fw_bw(data), alpha, sigma)
