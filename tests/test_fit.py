from __future__ import annotations
import skstep as sks
import numpy as np
import pytest


@pytest.mark.parametrize(
    ["step_finder", "param"], [
        (sks.GaussStepFinder, 0.01),
        (sks.SDFixedGaussStepFinder, 0.01),
        (sks.TtestStepFinder, 0.001),
    ]
)
def test_fit_float(
    step_finder: type[sks.GaussStepFinder],
    param: float,
):
    rng = np.random.default_rng(0)
    data = np.concatenate([
        rng.normal(0.0, 1.0, size=322),
        rng.normal(4.8, 1.0, size=678)
    ])

    f = step_finder(data, param)

    f.fit()

    assert f.step_positions == [0, 322, 1000]
    assert f.nsteps == 2

@pytest.mark.parametrize(
    ["step_finder", "param"], [
        (sks.PoissonStepFinder, 0.01),
        (sks.BayesianPoissonStepFinder, 100),
    ],
)
def test_fit_integer(step_finder: type[sks.PoissonStepFinder], param: float):
    rng = np.random.default_rng(0)
    data = np.concatenate([
        rng.poisson(9.7, size=322),
        rng.poisson(52.2, size=678)
    ])

    f = step_finder(data, param)

    f.fit()

    assert f.step_positions == [0, 322, 1000]
    assert f.nsteps == 2
