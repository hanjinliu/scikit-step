from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln, logsumexp
from ._base import MomentBase, RecursiveStepFinder, TransitionProbabilityMixin

_EPS = 1e-12


class PoissonMoment(MomentBase):
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


class PoissonStepFinder(TransitionProbabilityMixin, RecursiveStepFinder):
    """
    Poisson distribution step finding.
    """

    _MOMENT_CLASS = PoissonMoment

    def __init__(self, data: ArrayLike, prob: float | None = None):
        super().__init__(data)
        if not np.issubdtype(self.data.dtype, np.integer):
            raise TypeError("In PoissonStep, non-integer data type is forbidden.")

        self._init_probability(prob)

    def _continue(self, dlogL):
        return self.penalty + dlogL > 0


class BayesianPoissonMoment(MomentBase):
    def get_optimal_splitter(self):
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


class BayesianPoissonStepFinder(RecursiveStepFinder):
    """
    Poisson distribution step finding in a Bayesian method.

    Reference
    ---------
    Ensign, D. L., & Pande, V. S. (2010). Bayesian detection of intensity changes in
    single molecule and molecular dynamics trajectories. Journal of Physical Chemistry
    B, 114(1), 280-292. https://doi.org/10.1021/jp906786b
    """

    _MOMENT_CLASS = BayesianPoissonMoment

    def __init__(self, data: ArrayLike, skept: float = 4):
        super().__init__(data)
        if skept <= 0:
            raise ValueError(f"`skept` must be larger than 0, but got {skept}")
        self.skept = skept
        if not np.issubdtype(self.data.dtype, np.integer):
            raise TypeError("In PoissonStep, non-integer data type is forbidden.")

    def get_params(self) -> dict[str, float]:
        return {"skept": self.skept}

    def _continue(self, logbf):
        return np.log(self.skept) < logbf
