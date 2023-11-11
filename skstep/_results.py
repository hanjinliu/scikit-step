from __future__ import annotations

from functools import cached_property
import numpy as np
from numpy.typing import NDArray


class FitResult:
    """Fitting result of step finders."""

    def __init__(self, data: NDArray[np.number], step_positions: list[int]):
        self._data = data
        self._step_positions = np.array(step_positions, dtype=np.uint32)
        self._step_positions.flags.writeable = False

    def __repr__(self) -> str:
        cn = type(self).__name__
        ndata = self._data.size
        nstep = self.nsteps
        return f"<{cn} of {ndata} data and {nstep} steps>"

    @property
    def step_positions(self) -> NDArray[np.uint32]:
        """Step positions (including the start and end points) as an array."""
        return self._step_positions

    @property
    def data(self) -> NDArray[np.number]:
        return self._data

    @property
    def nsteps(self) -> int:
        """Number of steps."""
        return len(self.step_positions) - 1

    @cached_property
    def means(self) -> NDArray[np.floating]:
        """Mean of each step."""
        out = np.empty(self.nsteps)
        pos = self.step_positions
        for i in range(self.nsteps):
            subset = self.data[pos[i] : pos[i + 1]]
            out[i] = subset.mean()
        return out

    @cached_property
    def lengths(self) -> NDArray[np.integer]:
        """Length of each step."""
        return np.diff(self.step_positions)

    @cached_property
    def step_sizes(self) -> NDArray[np.integer]:
        """Array of step sizes (means[i+1] - means[i])."""
        return np.diff(self.means)

    @cached_property
    def data_fit(self) -> NDArray[np.floating]:
        """The fitted data."""
        out = np.empty(self.data.size)
        means = self.means
        for i in range(self.nsteps):
            out[self.step_positions[i] : self.step_positions[i + 1]] = means[i]
        return out

    def plot(self, range: tuple[int, int] | None = None):
        """Plot the fitting result."""
        import matplotlib.pyplot as plt

        if range is None:
            sl = slice(None)
        else:
            sl = slice(*range)

        plt.plot(self.data[sl], color="lightgray", label="raw data")
        plt.plot(self.data_fit[sl], color="red", label="fit")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

        return plt.gca()
