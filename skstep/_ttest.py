from __future__ import annotations
import numpy as np
from ._base import MomentBase, RecursiveStepFinder
from ._utils import normalize_sigma


class TtestMoment(MomentBase):
    def get_optimal_splitter(self):
        n = np.arange(1, len(self))
        tk = np.abs(self.fw[0] / n - self.bw[0] / n[::-1]) / np.sqrt(
            1 / n + 1 / (n[::-1])
        )
        x = int(np.argmax(tk))
        return tk[x], x + 1


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

    _MOMENT_CLASS = TtestMoment

    def __init__(self, data, alpha: float = 0.05, sigma: float | None = None):
        from scipy.stats import t as student_t

        super().__init__(data)
        if not 0 < alpha < 0.5:
            alpha = 0.05
        self.alpha = alpha
        self.sigma = normalize_sigma(sigma, data)
        self._student_t = student_t

    def _append_steps(self, mom: TtestMoment, x0: int = 0):
        if len(mom) < 3:
            return None
        tk, dx = mom.get_optimal_splitter()
        t_cri = self._student_t.ppf(1 - self.alpha / 2, len(mom))
        if t_cri < tk / self.sigma:
            self.step_list.append(x0 + dx)
            mom1, mom2 = mom.split(dx)
            self._append_steps(mom1, x0=x0)
            self._append_steps(mom2, x0=x0 + dx)
        else:
            pass
        return None

    def _continue(self, s):
        # just for now
        pass
