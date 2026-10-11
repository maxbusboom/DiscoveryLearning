"""Verification tests for :mod:`heat9`.

Covers, in order:

1. the 9-point stencil reproduces  lap(u) + (h^2/12) lap^2(u)  exactly;
2. second-order convergence in the max norm for both stencils;
3. the discrete Fourier symbol  h^2 lambda = -4 (s + c - (2/3) s c);
4. stability -- Crank-Nicolson unconditional, forward Euler at mu = 3/8 (9pt)
   and mu = 1/4 (5pt), and the Crank-Nicolson checkerboard sign flip;
5. isotropy -- the 9-point stencil diffuses a Gaussian blob far more
   symmetrically than the 5-point one.
"""

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

import heat9 as H


ALPHA = 1.0


# ---------------------------------------------------------------------------
# polynomials used by the stencil check
# ---------------------------------------------------------------------------


def _poly(entries, shape=(6, 6)):
    C = np.zeros(shape)
    for (i, j), c in entries.items():
        C[i, j] = c
    return C


def _random_quintic(seed=20240917):
    rng = np.random.default_rng(seed)
    C = np.zeros((6, 6))
    for i in range(6):
        for j in range(6 - i):
            C[i, j] = rng.standard_normal()
    return C


POLYS = {
    "x^2": _poly({(2, 0): 1.0}),
    "y^2": _poly({(0, 2): 1.0}),
    "x^2 y^2": _poly({(2, 2): 1.0}),
    "quintic": _poly({(5, 0): 1.0, (3, 2): 2.0, (1, 4): -3.0,
                      (2, 2): 1.0, (1, 0): 1.0, (0, 1): -1.0, (0, 0): 1.0}),
    "random quintic": _random_quintic(),
}


def _apply_matrix(grid, stencil, C):
    """Interior values of ``L U + b`` with Dirichlet data taken from the polynomial."""
    L = H.build_laplacian(grid, stencil)
    g = lambda x, y, t: H.poly_eval(C, x, y)
    b = H.boundary_vector(grid, g, 0.0, stencil)
    X, Y = grid.interior_nodes()
    U = grid.ravel(H.poly_eval(C, X, Y))
    return grid.unravel(L @ U + b)


# ---------------------------------------------------------------------------
# 1. stencil check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(POLYS))
@pytest.mark.parametrize("N", [8, 10])
def test_9pt_stencil_equals_lap_plus_h2_over_12_bilaplacian(name, N):
    """L9{u} == lap(u) + (h^2/12) lap^2(u), exactly, for degree <= 5."""
    C = POLYS[name]
    grid = H.Grid(N, N)
    lap = H.poly_laplacian(C)
    bih = H.poly_laplacian(lap)
    X, Y = grid.interior_nodes()
    expected = H.poly_eval(lap, X, Y) + (grid.h ** 2 / 12.0) * H.poly_eval(bih, X, Y)

    got = _apply_matrix(grid, H.LAPLACIAN_9PT, C)
    scale = max(1.0, np.abs(expected).max())
    assert np.abs(got - expected).max() < 1e-10 * scale

    # the same must hold for the bare stencil applied to the full node array
    F = H.poly_eval(C, *grid.nodes())
    assert np.abs(H.LAPLACIAN_9PT.apply(F, grid.h) - expected).max() < 1e-10 * scale


@pytest.mark.parametrize("name", list(POLYS))
def test_5pt_stencil_equals_lap_plus_h2_over_12_uxxxx_uyyyy(name):
    """The 5-point truncation error is the *anisotropic* h^2/12 (u_xxxx + u_yyyy)."""
    C = POLYS[name]
    grid = H.Grid(8, 8)
    lap = H.poly_laplacian(C)
    aniso = H.poly_add(H.poly_deriv(C, 0, 4), H.poly_deriv(C, 1, 4))
    X, Y = grid.interior_nodes()
    expected = H.poly_eval(lap, X, Y) + (grid.h ** 2 / 12.0) * H.poly_eval(aniso, X, Y)
    got = _apply_matrix(grid, H.LAPLACIAN_5PT, C)
    assert np.abs(got - expected).max() < 1e-10 * max(1.0, np.abs(expected).max())


def test_5pt_is_not_isotropic_to_order_h2():
    """Sanity: for x^2 y^2 the 5-point stencil misses the h^2/12 lap^2 term."""
    C = POLYS["x^2 y^2"]
    grid = H.Grid(8, 8)
    lap = H.poly_laplacian(C)
    bih = H.poly_laplacian(lap)
    X, Y = grid.interior_nodes()
    isotropic = H.poly_eval(lap, X, Y) + (grid.h ** 2 / 12.0) * H.poly_eval(bih, X, Y)
    got = _apply_matrix(grid, H.LAPLACIAN_5PT, C)
    # lap^2(x^2 y^2) = 8, so the discrepancy is exactly h^2/12 * 8 = 2 h^2 / 3
    assert np.allclose(isotropic - got, 2.0 * grid.h ** 2 / 3.0)


@pytest.mark.parametrize("stencil", [H.LAPLACIAN_9PT, H.LAPLACIAN_5PT])
def test_matrix_rows_and_symmetry(stencil):
    grid = H.Grid(12, 12)
    L = H.build_laplacian(grid, stencil).tocsr()
    assert np.diff(L.indptr).max() <= len(stencil.weights)
    assert abs((L - L.T)).max() == 0.0                      # symmetric
    assert np.all(L.diagonal() < 0)                          # negative diagonal
    # row sums vanish except next to the boundary (constants are in the kernel
    # once the boundary vector is added back)
    b = H.boundary_vector(grid, lambda x, y, t: np.ones_like(x), 0.0, stencil)
    ones = np.ones(grid.n_interior)
    assert np.abs(L @ ones + b).max() < 1e-10 / grid.h ** 2


# ---------------------------------------------------------------------------
# 2. convergence
# ---------------------------------------------------------------------------


def _exact_sine(x, y, t):
    return np.exp(-2.0 * np.pi ** 2 * ALPHA * t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def _convergence(stencil, Ns=(8, 16, 32, 64), T=0.1, steps0=None):
    """Refine with dt proportional to h and return (h, dt, error) triples."""
    if steps0 is None:
        steps0 = max(1, int(np.ceil(T / (1.0 / Ns[0]))))
    out = []
    for k, N in enumerate(Ns):
        grid = H.Grid(N, N)
        nsteps = steps0 * 2 ** k
        U, info = H.solve_heat(grid, lambda x, y: _exact_sine(x, y, 0.0), T,
                               ALPHA, nsteps=nsteps, stencil=stencil)
        X, Y = grid.interior_nodes()
        err = np.abs(grid.unravel(U) - _exact_sine(X, Y, T)).max()
        out.append((grid.h, info["dt"], err))
    return out


@pytest.mark.parametrize("stencil", [H.LAPLACIAN_9PT, H.LAPLACIAN_5PT])
def test_second_order_convergence(stencil):
    rows = _convergence(stencil)
    ratios = [r[1] / r[0] for r in rows]
    assert np.allclose(ratios, ratios[0])                    # dt/h held fixed
    orders = [np.log2(rows[i - 1][2] / rows[i][2]) for i in range(1, len(rows))]
    assert all(1.6 < o < 2.6 for o in orders), orders        # all roughly 2
    assert 1.85 < orders[-1] < 2.2, orders                   # asymptotic order 2


@pytest.mark.parametrize("stencil", [H.LAPLACIAN_9PT, H.LAPLACIAN_5PT])
def test_second_order_convergence_with_nonzero_dirichlet(stencil):
    """cos(pi x) cos(pi y) exp(-2 pi^2 t) is nonzero on the boundary."""
    exact = lambda x, y, t: (np.exp(-2.0 * np.pi ** 2 * ALPHA * t)
                             * np.cos(np.pi * x) * np.cos(np.pi * y))
    T = 0.1
    errs = []
    # dt/h = 0.4 here rather than 0.8: this solution's error is dominated by the
    # time discretisation, and at dt = 0.1 or 0.05 the step is still outside the
    # Crank-Nicolson asymptotic range (|lambda| dt ~ 1-2), which flatters or
    # spoils the measured order.  Halving dt/h puts all three levels inside it.
    for k, N in enumerate((16, 32, 64)):
        grid = H.Grid(N, N)
        U, _ = H.solve_heat(grid, lambda x, y: exact(x, y, 0.0), T, ALPHA,
                            nsteps=4 * 2 ** k, stencil=stencil, g=exact)
        X, Y = grid.interior_nodes()
        errs.append(np.abs(grid.unravel(U) - exact(X, Y, T)).max())
    orders = [np.log2(errs[i - 1] / errs[i]) for i in range(1, len(errs))]
    assert all(1.8 < o < 2.4 for o in orders), (errs, orders)


@pytest.mark.parametrize("stencil", [H.LAPLACIAN_9PT, H.LAPLACIAN_5PT])
def test_spatial_order_alone_with_nonzero_dirichlet(stencil):
    """Freeze the time error (400 steps) and refine h only: the Dirichlet
    boundary vector must not degrade the second-order spatial accuracy."""
    exact = lambda x, y, t: (np.exp(-2.0 * np.pi ** 2 * ALPHA * t)
                             * np.cos(np.pi * x) * np.cos(np.pi * y))
    errs = []
    for N in (16, 32, 64):
        grid = H.Grid(N, N)
        U, _ = H.solve_heat(grid, lambda x, y: exact(x, y, 0.0), 0.1, ALPHA,
                            nsteps=400, stencil=stencil, g=exact)
        X, Y = grid.interior_nodes()
        errs.append(np.abs(grid.unravel(U) - exact(X, Y, 0.1)).max())
    orders = [np.log2(errs[i - 1] / errs[i]) for i in range(1, len(errs))]
    assert all(1.9 < o < 2.1 for o in orders), (errs, orders)


@pytest.mark.parametrize("stencil", [H.LAPLACIAN_9PT, H.LAPLACIAN_5PT])
@pytest.mark.parametrize("mu", [0.25, 50.0])
def test_constant_is_preserved_exactly(stencil, mu):
    """u = C on the boundary with u(.,.,0) = C is a steady state.

    The Laplacian annihilates a constant, so the scheme must hold it to
    round-off for any dt.  Unlike homogeneous data this keeps the Dirichlet
    boundary vector nonzero, so it checks that L U + b really does vanish on
    a constant -- i.e. that the boundary coefficients match the matrix ones.
    """
    C = 2.5
    grid = H.Grid(32, 32)
    g = lambda x, y, t: np.full(np.broadcast(np.asarray(x), np.asarray(y)).shape, C)
    dt = mu * grid.h ** 2 / ALPHA
    U, _ = H.solve_heat(grid, lambda x, y: np.full(x.shape, C), 40 * dt, ALPHA,
                        nsteps=40, stencil=stencil, g=g)
    assert np.abs(U - C).max() < 1e-12 * C

    b = H.boundary_vector(grid, g, 0.0, stencil)
    assert np.abs(b).max() > 0.0                      # the path is exercised
    assert np.abs(L_times(grid, stencil) @ np.full(grid.n_interior, C) + b).max() \
        < 1e-10 / grid.h ** 2


def L_times(grid, stencil):
    return H.build_laplacian(grid, stencil)


def test_constant_shift_leaves_the_error_unchanged():
    """Adding a constant to both the solution and the boundary data changes
    nothing: v = u - C solves the same problem with homogeneous data."""
    C = 3.0
    grid = H.Grid(32, 32)
    base = lambda x, y, t: _exact_sine(x, y, t)
    shifted = lambda x, y, t: _exact_sine(x, y, t) + C
    U0, _ = H.solve_heat(grid, lambda x, y: base(x, y, 0.0), 0.1, ALPHA,
                         nsteps=20, stencil=H.LAPLACIAN_9PT)
    U1, _ = H.solve_heat(grid, lambda x, y: shifted(x, y, 0.0), 0.1, ALPHA,
                         nsteps=20, stencil=H.LAPLACIAN_9PT,
                         g=lambda x, y, t: np.full(
                             np.broadcast(np.asarray(x), np.asarray(y)).shape, C))
    assert np.abs((U1 - C) - U0).max() < 1e-12


def test_homogeneous_dirichlet_decays_to_zero():
    grid = H.Grid(32, 32)
    U, _ = H.solve_heat(grid, lambda x, y: _exact_sine(x, y, 0.0), 1.0, ALPHA,
                        nsteps=50, stencil=H.LAPLACIAN_9PT)
    assert np.abs(U).max() < 1e-6


# ---------------------------------------------------------------------------
# 3. Fourier symbol
# ---------------------------------------------------------------------------


MODES = [(0.0, 0.0), (np.pi / 4, 0.0), (np.pi / 2, np.pi / 2),
         (np.pi / 4, 3 * np.pi / 4), (np.pi, np.pi / 2), (np.pi, np.pi)]


@pytest.mark.parametrize("theta,phi", MODES)
def test_symbol_from_periodic_matrix_9pt(theta, phi):
    """Applying L to exp(i(p theta + q phi)) on a periodic grid reproduces lambda."""
    N, h = 8, 0.125                      # all MODES are multiples of 2 pi / 8
    L = H.build_laplacian_periodic(N, N, h, H.LAPLACIAN_9PT)
    P, Q = np.meshgrid(np.arange(N), np.arange(N), indexing="xy")
    v = np.exp(1j * (P * theta + Q * phi)).ravel()
    lam = (L @ v.real) + 1j * (L @ v.imag)
    expected = H.symbol_9pt(theta, phi) / h ** 2
    assert np.abs(lam - expected * v).max() < 1e-12 / h ** 2
    assert abs(H.fourier_symbol(H.LAPLACIAN_9PT, theta, phi)
               - H.symbol_9pt(theta, phi)) < 1e-14


@pytest.mark.parametrize("theta,phi", MODES)
def test_symbol_from_periodic_matrix_5pt(theta, phi):
    N, h = 8, 0.125
    L = H.build_laplacian_periodic(N, N, h, H.LAPLACIAN_5PT)
    P, Q = np.meshgrid(np.arange(N), np.arange(N), indexing="xy")
    v = np.exp(1j * (P * theta + Q * phi)).ravel()
    lam = (L @ v.real) + 1j * (L @ v.imag)
    assert np.abs(lam - (H.symbol_5pt(theta, phi) / h ** 2) * v).max() < 1e-12 / h ** 2


def test_symbol_closed_form_matches_stencil_everywhere():
    th, ph = np.meshgrid(np.linspace(-np.pi, np.pi, 37),
                         np.linspace(-np.pi, np.pi, 41), indexing="ij")
    assert np.abs(H.fourier_symbol(H.LAPLACIAN_9PT, th, ph) - H.symbol_9pt(th, ph)).max() < 1e-13
    assert np.abs(H.fourier_symbol(H.LAPLACIAN_5PT, th, ph) - H.symbol_5pt(th, ph)).max() < 1e-13


def test_checkerboard_symbol_is_minus_16_over_3():
    assert H.symbol_9pt(np.pi, np.pi) == pytest.approx(-16.0 / 3.0, rel=1e-14)
    assert H.fourier_symbol(H.LAPLACIAN_9PT, np.pi, np.pi) == pytest.approx(-16.0 / 3.0, rel=1e-14)
    assert H.symbol_5pt(np.pi, np.pi) == pytest.approx(-8.0, rel=1e-14)
    # and it is the extreme value of the symbol
    th, ph = np.meshgrid(np.linspace(-np.pi, np.pi, 401),
                         np.linspace(-np.pi, np.pi, 401), indexing="ij")
    assert H.symbol_9pt(th, ph).min() == pytest.approx(-16.0 / 3.0, rel=1e-9)


def test_checkerboard_is_an_eigenvector_of_the_periodic_matrix():
    N, h = 16, 1.0 / 16
    L = H.build_laplacian_periodic(N, N, h, H.LAPLACIAN_9PT)
    P, Q = np.meshgrid(np.arange(N), np.arange(N), indexing="xy")
    cb = ((-1.0) ** (P + Q)).ravel()
    assert np.abs(L @ cb - (-16.0 / 3.0 / h ** 2) * cb).max() < 1e-9 / h ** 2


# ---------------------------------------------------------------------------
# 4. stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stencil", [H.LAPLACIAN_9PT, H.LAPLACIAN_5PT])
@pytest.mark.parametrize("mu", [0.1, 1.0, 10.0, 100.0])
def test_crank_nicolson_is_unconditionally_stable(stencil, mu):
    grid = H.Grid(32, 32)
    L = H.build_laplacian(grid, stencil)
    rng = np.random.default_rng(7)
    U0 = rng.standard_normal(grid.n_interior)     # rough data: all modes excited
    dt = mu * grid.h ** 2 / ALPHA
    norms, sups = [], []

    def cb(n, t, U):
        norms.append(np.linalg.norm(U))
        sups.append(np.abs(U).max())

    H.crank_nicolson_solve(L, U0, ALPHA, dt, 200, callback=cb)
    n0 = np.linalg.norm(U0)
    assert norms[0] <= n0 * (1 + 1e-12)
    assert all(norms[i + 1] <= norms[i] * (1 + 1e-12) for i in range(len(norms) - 1))
    assert max(sups) <= n0                        # ||U||_inf <= ||U||_2 <= ||U0||_2
    assert np.isfinite(norms[-1])


@pytest.mark.parametrize("stencil,mu_ok,mu_bad,limit", [
    (H.LAPLACIAN_9PT, 0.37, 0.39, 3.0 / 8.0),
    (H.LAPLACIAN_5PT, 0.24, 0.26, 1.0 / 4.0),
])
def test_forward_euler_stability_limit(stencil, mu_ok, mu_bad, limit):
    grid = H.Grid(32, 32)
    L = H.build_laplacian(grid, stencil)
    rng = np.random.default_rng(11)
    U0 = rng.standard_normal(grid.n_interior)
    sup0 = np.abs(U0).max()

    stable = H.forward_euler_solve(L, U0, ALPHA, mu_ok * grid.h ** 2, 600)
    assert np.abs(stable).max() < sup0

    blown = H.forward_euler_solve(L, U0, ALPHA, mu_bad * grid.h ** 2, 600)
    assert np.abs(blown).max() > 1e6 * sup0

    # the limit is set by the most negative point of the symbol
    worst = H.fourier_symbol(stencil, np.pi, np.pi)
    assert abs(2.0 / abs(worst) - limit) < 1e-14


def test_forward_euler_limits_bracket_the_two_stencils():
    """3/8 for the 9-point stencil, 1/4 for the 5-point one."""
    assert 2.0 / abs(H.symbol_9pt(np.pi, np.pi)) == pytest.approx(3.0 / 8.0)
    assert 2.0 / abs(H.symbol_5pt(np.pi, np.pi)) == pytest.approx(1.0 / 4.0)


@pytest.mark.parametrize("mu,flips", [(0.1, False), (0.2, False), (0.3, False),
                                      (0.4, True), (1.0, True), (10.0, True)])
def test_crank_nicolson_checkerboard_sign_flip(mu, flips):
    """CN amplification is (1 - 8 mu/3)/(1 + 8 mu/3): negative exactly for mu > 3/8."""
    N, h = 16, 1.0 / 16
    L = H.build_laplacian_periodic(N, N, h, H.LAPLACIAN_9PT)
    P, Q = np.meshgrid(np.arange(N), np.arange(N), indexing="xy")
    cb = ((-1.0) ** (P + Q)).ravel()
    signs = []
    H.crank_nicolson_solve(L, cb, ALPHA, mu * h ** 2, 6,
                           callback=lambda n, t, U: signs.append(np.sign(U[0])))
    expected = [(-1.0) ** (n + 1) if flips else 1.0 for n in range(6)]
    assert signs == expected
    g = (1.0 - 8.0 * mu / 3.0) / (1.0 + 8.0 * mu / 3.0)
    assert (g < 0) is flips
    assert abs(g) <= 1.0                       # bounded either way


# ---------------------------------------------------------------------------
# 5. isotropy
# ---------------------------------------------------------------------------


def _diffuse_gaussian(stencil, N=64, sigma0_in_h=1.5, sigma_end_in_h=5.0, mu=0.25):
    grid = H.Grid(N, N)
    h = grid.h
    s0 = sigma0_in_h * h
    T = ((sigma_end_in_h * h) ** 2 - s0 ** 2) / (2.0 * ALPHA)
    u0 = lambda x, y: np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * s0 ** 2))
    nsteps = max(1, int(round(T / (mu * h ** 2))))
    U, _ = H.solve_heat(grid, u0, T, ALPHA, nsteps=nsteps, stencil=stencil)
    return grid, grid.full_field(U), np.sqrt(s0 ** 2 + 2 * ALPHA * T)


def _anisotropy(grid, F, rmax=0.3):
    """(max |u_axis - u_diag| / peak, r_half along x, r_half along the diagonal).

    Quintic interpolation keeps the sampling artifact near 1e-7, two orders of
    magnitude below the smallest signal measured here.
    """
    f = RegularGridInterpolator((grid.y, grid.x), F, method="quintic")
    c, s2 = 0.5, 1.0 / np.sqrt(2.0)
    ray = lambda r, ux, uy: float(f([[c + uy * r, c + ux * r]])[0])
    level = 0.5 * ray(0.0, 1.0, 0.0)
    r_ax = brentq(lambda r: ray(r, 1.0, 0.0) - level, 1e-6, rmax, xtol=1e-13)
    r_dg = brentq(lambda r: ray(r, s2, s2) - level, 1e-6, rmax, xtol=1e-13)
    r = np.linspace(0.0, rmax, 1201)
    pa = f(np.stack([np.full_like(r, c), c + r], axis=1))
    pd = f(np.stack([c + r * s2, c + r * s2], axis=1))
    return np.abs(pa - pd).max() / F.max(), r_ax, r_dg


def test_measurement_floor_is_negligible():
    """An exactly radial Gaussian sampled on the grid must read as isotropic."""
    grid, _, sigma = _diffuse_gaussian(H.LAPLACIAN_9PT)
    X, Y = grid.nodes()
    F = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * sigma ** 2))
    prof, r_ax, r_dg = _anisotropy(grid, F)
    assert prof < 1e-6
    assert abs(r_dg - r_ax) / r_ax < 1e-6


def test_9pt_is_more_isotropic_than_5pt():
    res = {}
    for stencil in (H.LAPLACIAN_5PT, H.LAPLACIAN_9PT):
        grid, F, _ = _diffuse_gaussian(stencil)
        prof, r_ax, r_dg = _anisotropy(grid, F)
        res[stencil.name] = (prof, abs(r_dg - r_ax) / r_ax)
    prof5, spread5 = res["5-point"]
    prof9, spread9 = res["9-point"]
    assert prof9 < prof5 / 20.0, res          # amplitude anisotropy
    assert spread9 < spread5 / 20.0, res      # axis-vs-diagonal spread
    assert prof9 > 1e-6                       # still above the measurement floor


if __name__ == "__main__":
    # so that hitting Run on this file in an editor actually runs the suite
    raise SystemExit(pytest.main([__file__, "-v"]))
