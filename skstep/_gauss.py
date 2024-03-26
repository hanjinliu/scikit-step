from __future__ import annotations
from typing import Tuple
import numpy as np
import heapq

from numpy.typing import NDArray, ArrayLike
from ._base import (
    StepFinderBase,
    RecursiveStepFinder,
    TransitionProbabilityMixin,
)
from skstep._moments import MomentBase, RecursiveMoment
from skstep._utils import normalize_sigma, calculate_penalty
from skstep._results import FitResult

_HeapItem = Tuple[float, int, int, "GaussMoment"]


class Heap:
    """
    Priorty queue. I wrapped heapq because it is not intuitive.
    """

    def __init__(self):
        self.heap = []

    def push(self, item: _HeapItem):
        heapq.heappush(self.heap, item)

    def pop(self) -> _HeapItem:
        return heapq.heappop(self.heap)


class GaussMoment(MomentBase):
    MAX_ORDER = 2

    @property
    def chi2(self):
        return self.total[1] - self.total[0] ** 2 / len(self)

    def get_optimal_splitter(self):
        n = np.arange(1, len(self))
        chi2_fw = self.fw[1] - self.fw[0] ** 2 / n
        chi2_bw = self.bw[1] - self.bw[0] ** 2 / n[::-1]
        chi2 = chi2_fw + chi2_bw
        x = int(np.argmin(chi2))
        return chi2[x] - self.chi2, x + 1


class GaussStepFinder(TransitionProbabilityMixin, StepFinderBase):
    """
    Kalafut-Visscher's step finding algorithm on data with Gaussian noise.

    Reference
    ---------
    Kalafut, B., & Visscher, K. (2008). An objective, model-independent method for
    detection of non-uniform steps in noisy signals. Computer Physics Communications,
    179(10), 716-723. https://doi.org/10.1016/j.cpc.2008.06.008
    """

    def __init__(self, prob: float | None = None):
        """
        Parameters
        ----------
        prob : float, optional
            Probability of transition (signal change). If not in a proper range 0<p<0.5,
            then This algorithm will be identical to the original Kalafut-Visscher's.
        """
        self._prob = prob

    @property
    def prob(self) -> float | None:
        return self._prob

    def _moment_from_array(self, data):
        return GaussMoment(*GaussMoment.calculate_fw_bw(data))

    def fit(self, data: ArrayLike):
        data = np.asarray(data)
        g = self._moment_from_array(data)
        chi2 = g.chi2  # initialize total chi^2
        heap = Heap()  # chi^2 change (<0), dx, x0, GaussMoment object of the step
        heap.push(g.get_optimal_splitter() + (0, g))
        ndata = data.size
        penalty = calculate_penalty(data, self._prob)
        step_positions = [0, data.size]

        while True:
            dchi2, dx, x0, g = heap.pop()
            dlogL = penalty - ndata / 2 * np.log(1 + dchi2 / chi2)

            if dlogL > 0:
                x = x0 + dx
                g1, g2 = g.split(dx)
                if len(g1) > 2:
                    heap.push(g1.get_optimal_splitter() + (x0, g1))
                if len(g2) > 2:
                    heap.push(g2.get_optimal_splitter() + (x, g2))
                step_positions.append(x)
                chi2 += dchi2
            else:
                break

        step_positions.sort()
        return FitResult(data, step_positions)


class SDFixedGaussMoment(RecursiveMoment):
    def __init__(
        self,
        fw: NDArray[np.number],
        bw: NDArray[np.number],
        total: NDArray[np.number],
        penalty: float,
        sigma: float,
    ):
        super().__init__(fw, bw, total)
        self._penalty = penalty
        self._sigma = sigma

    def with_fw_bw(
        self,
        fw: NDArray[np.number],
        bw: NDArray[np.number],
        total: NDArray[np.number],
    ) -> SDFixedGaussMoment:
        """Return a new Moment object with the given array."""
        return self.__class__(fw, bw, total, self._penalty, self._sigma)

    @property
    def sq(self):
        return (2 - 1 / len(self)) / len(self) * self.total[0] ** 2

    def get_optimal_splitter(self):
        n = np.arange(1, len(self))
        sq_fw = (2 - 1 / n) / n * self.fw[0] ** 2
        sq_bw = (2 - 1 / n[::-1]) / n[::-1] * self.bw[0] ** 2
        sq = sq_fw + sq_bw
        x = int(np.argmax(sq))
        return sq[x] - self.sq, x + 1

    def _continue(self, sq) -> bool:
        return self._penalty + sq / (2 * self._sigma**2) > 0


class SDFixedGaussStepFinder(TransitionProbabilityMixin, RecursiveStepFinder):
    """
    Gauss-distribution step finding with fixed standard deviation of noise

    If standard deviation of noise is unknown then it will be estimated by
    wavelet method. Compared to GaussStep, this algorithm detects more steps
    in some cases and less in others.
    """

    def __init__(self, prob: float | None, sigma: float | None = None):
        self._prob = prob
        self._sigma = sigma

    @property
    def prob(self) -> float | None:
        return self._prob

    @property
    def sigma(self) -> float | None:
        return self._sigma

    def _moment_from_array(self, data):
        penalty = calculate_penalty(data, self._prob)
        sigma = normalize_sigma(self._sigma, data)
        return SDFixedGaussMoment(
            *SDFixedGaussMoment.calculate_fw_bw(data),
            penalty,
            sigma,
        )

    def get_params(self) -> dict[str, float | None]:
        return {"prob": self.prob, "sigma": self.sigma}
