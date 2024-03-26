from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import NDArray
from skstep._utils import calculate_penalty
from skstep._moments import RecursiveMoment
from skstep._base import RecursiveStepFinder, TransitionProbabilityMixin

_EPS = 1e-12


class PoissonMoment(RecursiveMoment):
    def __init__(
        self,
        fw: NDArray[np.number],
        bw: NDArray[np.number],
        total: NDArray[np.number],
        penalty: float,
    ):
        super().__init__(fw, bw, total)
        self._penalty = penalty

    @property
    def slogm(self):
        return self.total[0] * np.log((self.total[0] + _EPS) / len(self))

    def get_optimal_splitter(self):
        n = np.arange(1, len(self))
        slogm_fw = self.fw[0] * np.log((self.fw[0] + _EPS) / n)
        slogm_bw = self.bw[0] * np.log((self.bw[0] + _EPS) / n[::-1])
        slogm = slogm_fw + slogm_bw
        x = np.argmax(slogm)
        return slogm[x] - self.slogm, x + 1

    def _continue(self, dlogL: Any) -> bool:
        return self._penalty + dlogL > 0

    def with_fw_bw(
        self,
        fw: NDArray[np.number],
        bw: NDArray[np.number],
        total: NDArray[np.number],
    ) -> PoissonMoment:
        """Return a new Moment object with the given array."""
        return self.__class__(fw, bw, total, self._penalty)


class PoissonStepFinder(TransitionProbabilityMixin, RecursiveStepFinder):
    """Poisson distribution step finding."""

    def __init__(self, prob: float | None = None):
        self._prob = prob

    @property
    def prob(self) -> float | None:
        return self._prob

    def _moment_from_array(self, data):
        penalty = calculate_penalty(data, self._prob)
        return PoissonMoment(*PoissonMoment.calculate_fw_bw(data), penalty)


class BayesianPoissonMoment(RecursiveMoment):
    def __init__(
        self,
        fw: NDArray[np.number],
        bw: NDArray[np.number],
        total: NDArray[np.number],
        skept: float,
    ):
        super().__init__(fw, bw, total)
        self._skept = skept

    def with_fw_bw(
        self,
        fw: NDArray[np.number],
        bw: NDArray[np.number],
        total: NDArray[np.number],
    ) -> BayesianPoissonMoment:
        """Return a new Moment object with the given array."""
        return self.__class__(fw, bw, total, self._skept)

    def get_optimal_splitter(self):
        from scipy.special import gammaln, logsumexp

        n = np.arange(1, len(self))
        g1 = gammaln(self.fw[0] + 1)
        g2 = gammaln(self.bw[0] + 1)
        logprob = (
            g1
            + g2
            - (self.fw[0] + 1) * np.log(n)
            - (self.bw[0] + 1) * np.log(n[::-1])
            - np.log((self.fw[0] / n) ** 2 + (self.bw[0] / n[::-1]) ** 2)
        )
        logC = (
            np.log(2 / np.pi / (len(self) - 1))
            - gammaln(self.total[0])
            + self.total[0] * np.log(len(self))
        )
        logBayesFactor = logC + logsumexp(logprob)
        return logBayesFactor, np.argmax(logprob) + 1

    def _continue(self, logbf: float) -> bool:
        return np.log(self._skept) < logbf


class BayesianPoissonStepFinder(RecursiveStepFinder):
    """
    Poisson distribution step finding in a Bayesian method.

    Reference
    ---------
    Ensign, D. L., & Pande, V. S. (2010). Bayesian detection of intensity changes in
    single molecule and molecular dynamics trajectories. Journal of Physical Chemistry
    B, 114(1), 280-292. https://doi.org/10.1021/jp906786b
    """

    def __init__(self, skept: float = 4.0):
        if skept <= 0:
            raise ValueError(f"`skept` must be larger than 0, but got {skept}")
        self._skept = skept

    @property
    def skept(self) -> float:
        return self._skept

    def get_params(self) -> dict[str, float]:
        return {"skept": self.skept}

    def _moment_from_array(self, data):
        return BayesianPoissonMoment(
            *BayesianPoissonMoment.calculate_fw_bw(data),
            self._skept,
        )
