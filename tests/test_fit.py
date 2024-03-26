from __future__ import annotations
import skstep as sks
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
    data = sks.GaussSampler([322, 1000], [0.0, 4.8]).sample(1.0, seed=0)

    f = step_finder(param)

    res = f.fit(data)

    assert list(res.step_positions) == [0, 322, 1000]
    assert res.nsteps == 2

@pytest.mark.parametrize(
    ["step_finder", "param"], [
        (sks.PoissonStepFinder, 0.01),
        (sks.BayesianPoissonStepFinder, 100),
    ],
)
def test_fit_integer(step_finder: type[sks.PoissonStepFinder], param: float):
    data = sks.PoissonSampler([322, 1000], [9.7, 52.2]).sample(seed=0)

    f = step_finder(param)

    res = f.fit(data)

    assert list(res.step_positions) == [0, 322, 1000]
    assert res.nsteps == 2
