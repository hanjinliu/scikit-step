from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray
from abc import ABC, abstractmethod
from skstep._moments import MomentBase, RecursiveMoment
from skstep._results import FitResult

# Ideally we should prepare `n = np.arange(1, len(data))` first, and view
# it many times in get_optimal_splitter like n[:len(self.fw)], but this
# did not improve efficiency so much.


class StepFinderBase(ABC):
    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"

    @abstractmethod
    def _moment_from_array(self, data: NDArray[np.number]) -> MomentBase:
        """Initialize a Moment object from data."""

    @abstractmethod
    def fit(self, data: ArrayLike) -> FitResult:
        """Run step-finding algorithm and store all the information."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return parameters of the step-finding algorithm as a dictionary."""

    def fit_chunkwise(
        self,
        data: ArrayLike,
        chunksize: int = 50000,
        overlap: int = 2500,
        scheduler="threads",
    ) -> FitResult:
        data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError("data must be 1-dimensional.")
        if data.size < chunksize:
            return self.fit(data)

        from dask import array as da

        darr: da.Array = da.from_array(data, chunks=chunksize)  # type: ignore

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
        return FitResult(data, out)


def _fit_chunk_data(
    arr: np.ndarray,
    cls: type[StepFinderBase],
    block_info: dict = {},
    overlap: int = 0,
    chunksize: int = 0,
    **kwargs,
) -> np.ndarray:
    self = cls(**kwargs)
    result = self.fit(arr)
    block_0 = block_info[0]
    chunk_loc = block_0["chunk-location"][0]
    nchunks = block_0["num-chunks"][0]
    pos = np.array(result.step_positions, dtype=np.uint64)
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

    @abstractmethod
    def _moment_from_array(self, data: NDArray[np.number]) -> RecursiveMoment:
        """Initialize a Moment object from data."""

    def fit(self, data: ArrayLike) -> FitResult:
        data = np.asarray(data)
        mom = self._moment_from_array(data)
        step_positions = sorted([0, data.size] + mom.append_steps())
        return FitResult(data, step_positions)


class TransitionProbabilityMixin(StepFinderBase):
    """Step finders that take transition probability as a initial parameter."""

    prob: float

    def get_params(self) -> dict[str, float]:
        return {"prob": self.prob}
