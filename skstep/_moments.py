from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from typing_extensions import Self


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
        fw: NDArray[np.number],
        bw: NDArray[np.number],
        total: NDArray[np.number],
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
        m1 = self.with_fw_bw(fw1, bw1, total1)

        total2 = self.bw[:, border]
        bw2 = self.bw[:, i:]
        fw2 = _complement_fw(bw2, total2)
        m2 = self.with_fw_bw(fw2, bw2, total2)

        return m1, m2

    @classmethod
    def calculate_fw_bw(cls, data: NDArray[np.number]):
        orders = np.arange(1, cls.MAX_ORDER + 1)
        fw = np.vstack([np.cumsum(data[:-1] ** o) for o in orders])
        total = np.array(fw[:, -1] + data[-1] ** orders)
        rv = _complement_bw(fw, total)
        return fw, rv, total

    def with_fw_bw(
        self,
        fw: NDArray[np.number],
        bw: NDArray[np.number],
        total: NDArray[np.number],
    ) -> Self:
        """Return a new Moment object with the given array."""
        return self.__class__(fw, bw, total)

    @abstractmethod
    def get_optimal_splitter(self) -> tuple[float, int]:
        """Return the best index to split, and the loss at the split point."""


def _complement_fw(
    bw: NDArray[np.number], total: NDArray[np.number]
) -> NDArray[np.number]:
    return total.reshape(-1, 1) - bw


def _complement_bw(
    fw: NDArray[np.number], total: NDArray[np.number]
) -> NDArray[np.number]:
    return total.reshape(-1, 1) - fw


class RecursiveMoment(MomentBase):
    def append_steps(self, x0: int = 0) -> list[int]:
        steps = []
        if len(self) < 3:
            return steps
        s, dx = self.get_optimal_splitter()
        if self._continue(s):
            steps.append(x0 + dx)
            mom1, mom2 = self.split(dx)
            steps += mom1.append_steps(x0=x0)
            steps += mom2.append_steps(x0=x0 + dx)

        return steps

    @abstractmethod
    def _continue(self, s: Any) -> bool:
        """Check if recursive step needs continue."""
