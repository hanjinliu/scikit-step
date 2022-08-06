from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from typing_extensions import Self

# Ideally we should prepare `n = np.arange(1, len(data))` first, and view
# it many times in get_optimal_splitter like n[:len(self.fw)], but this
# did not improve efficiency so much.


class MomentBase(ABC):
    """
    This class aims at splitting arrays in a way that moment can be calculated very
    fast.

    fw          bw
     0 o ++++++ 0
     1 oo +++++ 1
     : :      : :
     n oooooo + n

    When ndarray ``data`` is given by ``self.init(data)``, ``self.fw[i,k]`` means
    ``np.sum(data[:k] ** (i+1))``, while ``self.bw[i,k]`` means
    ``np.sum(data[-k:]**(i+1))``. ``self.total[i]`` means ``np.sum(data**(i+1))``.
    """

    MAX_ORDER = 1

    def __init__(
        self,
        fw: np.ndarray,
        bw: np.ndarray,
        total: np.ndarray,
    ):
        self.fw = fw
        self.bw = bw
        self.total = total

    def __len__(self):
        """
        Get the length of the constant region
        """
        return self.fw.shape[1] + 1

    def split(self, i: int) -> tuple[Self, Self]:
        """
        Split a Moment object into to Moment objects at position i.
        This means :i-1 will be the former, and i: will be the latter.
        """

        # fw          bw
        #  0 o ++++++++++++ 0
        #  1 oo +++++++++++ 1
        #  : :            : :
        #    ooooooooooo ++
        #  n oooooooooooo + n

        # becomes

        # fw      bw=None
        #  0 o
        #  1 oo
        #  : :
        #  k oooooo

        #           +++++++
        #            ++++++
        #                 :
        #                 +
        #       fw=None   bw

        if self.fw is None or self.bw is None:
            raise RuntimeError("Moment object is not initialized")

        border = i - 1
        total1 = self.fw[:, border]
        fw1 = self.fw[:, :border]
        bw1 = _complement_bw(fw1, total1)
        m1 = self.__class__(fw1, bw1, total1)

        total2 = self.bw[:, border]
        bw2 = self.bw[:, i:]
        fw2 = _complement_fw(bw2, total2)
        m2 = self.__class__(fw2, bw2, total2)

        return m1, m2

    @classmethod
    def from_array(cls, data: np.ndarray) -> Self:
        """
        Initialization using all the data.

        Parameters
        ----------
        data : array
            The input data.
        """
        orders = np.arange(1, cls.MAX_ORDER + 1)
        fw = np.vstack([np.cumsum(data[:-1] ** o) for o in orders])
        total = np.array(fw[:, -1] + data[-1] ** orders)
        rv = _complement_bw(fw, total)
        return cls(fw, rv, total)

    @abstractmethod
    def get_optimal_splitter(self) -> tuple[float, int]:
        """Return the best index to split, and the loss at the split point."""


def _complement_fw(bw: np.ndarray, total: np.ndarray) -> np.ndarray:
    return total.reshape(-1, 1) - bw  # type: ignore


def _complement_bw(fw: np.ndarray, total: np.ndarray) -> np.ndarray:
    return total.reshape(-1, 1) - fw  # type: ignore


class StepFinderBase(ABC):
    _MOMENT_CLASS = MomentBase

    def __init__(self, data: ArrayLike):
        self.data = np.asarray(data)
        self.ndata = self.data.size
        self._step_positions = [0, self.ndata]
        self._caches: dict[str, Any] = {}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"

    def new(self, data: ArrayLike):
        """Construct a new StepFinder object with new data."""
        return self.__class__(data, **self.get_params())

    @abstractmethod
    def fit(self) -> Self:
        """Run step-finding algorithm and store all the information."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return parameters of the step-finding algorithm as a dictionary."""

    @property
    def step_positions(self) -> list[int]:
        return self._step_positions

    @property
    def nsteps(self) -> int:
        """Number of steps."""
        return len(self.step_positions) - 1

    @property
    def means(self) -> np.ndarray:
        """Mean of each step."""
        if (out := self._caches.get("means", None)) is None:
            out = self._caches["means"] = np.empty(self.nsteps)
            for i in range(self.nsteps):
                out[i] = np.mean(
                    self.data[self.step_positions[i] : self.step_positions[i + 1]]
                )
        return out

    @property
    def lengths(self) -> np.ndarray:
        """Length of each step."""
        if (out := self._caches.get("lengths", None)) is None:
            out = self._caches["lengths"] = np.diff(self.step_positions)
        return out

    @property
    def step_sizes(self) -> np.ndarray:
        """Array of step sizes (means[i+1] - means[i])."""
        if (out := self._caches.get("step_sizes", None)) is None:
            out = self._caches["step_sizes"] = np.diff(self.means)
        return out

    @property
    def data_fit(self) -> np.ndarray:
        """The fitting data."""
        if (out := self._caches.get("data_fit", None)) is None:
            out = self._caches["data_fit"] = np.empty(self.data.size)
            means = self.means
            for i in range(self.nsteps):
                out[self.step_positions[i] : self.step_positions[i + 1]] = means[i]

        return out

    def plot(self, range: tuple[int, int] | None = None):
        import matplotlib.pyplot as plt

        if range is None:
            sl = slice(None)
        else:
            sl = slice(*range)

        plt.plot(self.data[sl], color="lightgray", label="raw data")
        plt.plot(self.data_fit[sl], color="red", label="fit")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

        return plt.gca()

    def fit_chunkwise(
        self,
        chunksize: int = 50000,
        overlap: int = 2500,
        scheduler="threads",
    ) -> Self:
        if self.ndata < chunksize:
            return self.fit()

        from dask import array as da

        darr: da.core.Array = da.from_array(self.data, chunks=chunksize)  # type: ignore

        out = darr.map_overlap(
            _fit_chunk_data,
            depth=overlap,
            boundary="none",
            cls=self.__class__,
            overlap=overlap,
            chunksize=chunksize,
            **self.get_params(),
            dtype=np.uint64,
            trim=False,
        ).compute(scheduler=scheduler)
        self._step_positions = list(out)
        return self


def _fit_chunk_data(
    arr: np.ndarray,
    cls: type[StepFinderBase],
    block_info: dict = {},
    overlap: int = 0,
    chunksize: int = 0,
    **kwargs,
) -> np.ndarray:
    self = cls(arr, **kwargs)
    self.fit()
    block_0 = block_info[0]
    chunk_loc = block_0["chunk-location"][0]
    nchunks = block_0["num-chunks"][0]
    pos = np.array(self.step_positions, dtype=np.uint64)
    if 0 < chunk_loc < nchunks - 1:
        pos = pos[(overlap <= pos) & (pos < (arr.size - overlap))]
        return pos - overlap + chunksize * chunk_loc
    elif chunk_loc == 0:
        pos = pos[(pos < (arr.size - overlap))]
        return pos
    else:
        pos = pos[(overlap <= pos)]
        return pos - overlap + chunksize * chunk_loc


class RecursiveStepFinder(StepFinderBase):
    """Step finders that can be recursively fitted."""

    def _append_steps(self, mom: MomentBase, x0: int = 0):
        if len(mom) < 3:
            return None
        s, dx = mom.get_optimal_splitter()
        if self._continue(s):
            self.step_positions.append(x0 + dx)
            mom1, mom2 = mom.split(dx)
            self._append_steps(mom1, x0=x0)
            self._append_steps(mom2, x0=x0 + dx)

        return None

    @abstractmethod
    def _continue(self, s) -> bool:
        """Check if recursive step needs continue."""

    def fit(self) -> Self:
        self._caches.clear()
        mom = self._MOMENT_CLASS.from_array(self.data)
        self._append_steps(mom)
        self.step_positions.sort()
        return self


class TransitionProbabilityMixin(StepFinderBase):
    """Step finders that take transition probability as a initial parameter."""

    prob: float
    penalty: float

    def _init_probability(self, prob: float | None):
        if prob is not None:
            if not 0.0 < prob < 0.5:
                raise ValueError("prob must be in range 0.0 < prob < 0.5.")
            self.prob = prob
        else:
            self.prob = 1 / (1 + np.sqrt(self.ndata))
        self.penalty = np.log(self.prob / (1 - self.prob))

    def get_params(self) -> dict[str, float]:
        return {"prob": self.prob}
