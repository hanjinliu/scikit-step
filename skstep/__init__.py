__version__ = "0.0.1"

r"""
    An efficient, extended version of step finding algorithm original in [1].
    Although similar summation of data is repeated many times during a single run, the
original algorithm did not make full use of the results obtained in previous iterations.
In short, before calculating such as square deviation, the 'generator' of it is stored and
passed to next iteration. The n-th-order moment E[X_i^n] is very useful for in this purpose,
that's why `Moment` class is defined in moment.py. A `Moment` object corresponds to one constant
region (between two signal change points) and has n-th order moment with any step position as
attributes. Here in step finding classes, a corresponding `Moment` object is split into two
fragments and they are passed to newly generated two steps.
    In this optimized implementation, both GaussStep and PoissonStep can analyze 100,000-point
data within 1 sec!

Notes
-----
`x0` and `dx` always denote positions shown below:

       x0+dx
 0  x0   v  oooooooooo
 v   v   ooo    <- new step (2)
 oooo
     .......    <- old step
     oooo       <- new step (1)

"""

from ._gauss import GaussStepFinder, SDFixedGaussStepFinder
from ._poisson import PoissonStepFinder, BayesianPoissonStepFinder
from ._ttest import TtestStepFinder

from ._base import StepFinderBase

__all__ = [
    "GaussStepFinder",
    "SDFixedGaussStepFinder",
    "PoissonStepFinder",
    "BayesianPoissonStepFinder",
    "TtestStepFinder",
    "StepFinderBase",
]
