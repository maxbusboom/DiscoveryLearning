"""Tests for the error-analysis tooling in error_analysis.py.

Run with:  python3 -m pytest
"""

from __future__ import annotations

import numpy as np
import pytest

import error_analysis as E
import heat9 as H


@pytest.fixture(autouse=True)
def _restore_boundary():
    yield
    E.set_boundary("homogeneous")


# ---------------------------------------------------------------------------
# exact error budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("boundary", ["homogeneous", "constant"])
@pytest.mark.parametrize("stencil", [H.LAPLACIAN_9PT, H.LAPLACIAN_5PT],
                         ids=lambda s: s.name)
@pytest.mark.parametrize("N, nsteps", [(16, 3), (32, 40)])
def test_budget_parts_sum_to_the_total(boundary, stencil, N, nsteps):
    """space + time + roundoff == total, as signed fields."""
    E.set_boundary(boundary)
    b = E.error_budget(H.Grid(N, N), nsteps, stencil)
    resid = b["space"] + b["time"] + b["roundoff"] - b["total"]
    assert np.abs(resid).max() < 1e-12


@pytest.mark.parametrize("N, nsteps", [(16, 1), (32, 82), (64, 8), (128, 1),
                                       (128, 16), (128, 1311)])
def test_budget_roundoff_is_tiny_and_below_the_floor_estimate(N, nsteps):
    """The closed forms really are the discrete solutions: what is left is
    roundoff, and the a priori floor bounds it."""
    grid = H.Grid(N, N)
    U = E.run(grid, nsteps)
    r = np.abs(E.error_budget(grid, nsteps, U=U)["roundoff"]).max()
    assert r < 1e-12
    assert r <= E.roundoff_floor(grid, nsteps, H.LAPLACIAN_9PT, U)


def test_budget_parts_have_the_expected_orders():
    """Space part ~ h^2 at fixed dt; time part ~ dt^2 at fixed h."""
    nsteps = 8
    sp = [np.abs(E.error_budget(H.Grid(N, N), nsteps)["space"]).max()
          for N in (32, 64)]
    assert abs(np.log2(sp[0] / sp[1]) - 2.0) < 0.05
    grid = H.Grid(64, 64)
    tm = [np.abs(E.error_budget(grid, ns)["time"]).max() for ns in (16, 32)]
    assert abs(np.log2(tm[0] / tm[1]) - 2.0) < 0.05


def test_budget_refuses_analytic_boundary_data():
    E.set_boundary("analytic")
    with pytest.raises(ValueError):
        E.error_budget(H.Grid(8, 8), 2)


def test_constant_boundary_matches_homogeneous_to_roundoff():
    grid = H.Grid(32, 32)
    E.set_boundary("constant")
    Uc = E.run(grid, 20)
    E.set_boundary("homogeneous")
    Uh = E.run(grid, 20)
    assert np.abs(Uc - Uh - E.CONSTANT).max() < 1e-13


# ---------------------------------------------------------------------------
# roundoff floor, orders and the asymptotic window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stencil, smin", [(H.LAPLACIAN_9PT, 16.0 / 3.0),
                                           (H.LAPLACIAN_5PT, 8.0)],
                         ids=["9-point", "5-point"])
def test_kappa_bound(stencil, smin):
    grid = H.Grid(16, 16)
    dt = 0.01
    assert E.kappa_A(grid, dt, stencil) == pytest.approx(
        1.0 + 0.5 * E.ALPHA * dt * smin / grid.h ** 2)


def test_orders_flag_errors_at_the_floor():
    errors = [1e-2, 2.5e-3, 1e-14, 1e-14]
    floors = [1e-15] * 4
    o = E.orders(errors, floors)
    assert o[0] == pytest.approx(2.0)
    assert o[1] is None and o[2] is None
    assert E.orders(errors)[1] is not None        # no floors -> no flagging


def test_asymptotic_window_skips_preasymptotic_and_floor_levels():
    # orders: 2.6, 2.02, 2.00, 2.01, then the error hits the floor
    e = [1.0]
    for p in (2.6, 2.02, 2.00, 2.01):
        e.append(e[-1] / 2 ** p)
    e.append(e[-1] * 0.9)                          # roundoff plateau
    floors = [0.0] * 5 + [e[-1]]
    assert E.asymptotic_window(e, floors) == (1, 5)


def test_asymptotic_window_prefers_finer_levels_on_ties():
    e = [1.0, 0.25, 0.25 / 2 ** 3, 0.25 / 2 ** 3 / 4]   # orders 2, 3, 2
    assert E.asymptotic_window(e) == (2, 4)


def test_asymptotic_window_none_when_everything_is_at_the_floor():
    assert E.asymptotic_window([1e-15, 1e-15, 1e-15], [1e-15] * 3) is None


def test_slope_over_window():
    hs = [1 / 8, 1 / 16, 1 / 32, 1 / 64]
    e = [5.0 * h ** 2 for h in hs]
    e[0] *= 3.0                                     # pre-asymptotic coarse level
    s, win = E.fit(hs, e)
    assert win == (1, 4)
    assert s == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# solver time bookkeeping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["cn", "fe"])
def test_step_times_are_exact_multiples_of_dt(method):
    grid = H.Grid(8, 8)
    nsteps, T = 49, 0.007
    times = []
    H.solve_heat(grid, lambda x, y: 0 * x, T, nsteps=nsteps, method=method,
                 callback=lambda n, t, U: times.append(t))
    dt = T / nsteps
    assert times == [(n + 1) * dt for n in range(nsteps)]
