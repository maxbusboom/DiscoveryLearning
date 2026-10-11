"""Verification summary for :mod:`heat9` -- prints every check and writes isotropy.png.

Sections
--------
1. stencil check          L9{u} == lap(u) + (h^2/12) lap^2(u) for degree <= 5
2. convergence            max-norm error and observed order, 9-point vs 5-point
3. Fourier symbol         h^2 lambda = -4 (s + c - (2/3) s c), checkerboard = -16/3
4. stability              Crank-Nicolson vs forward Euler, mu = alpha dt / h^2
5. isotropy               Gaussian blob, axis vs diagonal, and isotropy.png

Run with ``python3 run_demo.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

import heat9 as H

ALPHA = 1.0
RULE = "-" * 78
_checks = []


def check(label, ok, detail=""):
    _checks.append((label, bool(ok)))
    print("   [%s] %s%s" % ("PASS" if ok else "FAIL", label,
                            ("  (%s)" % detail) if detail else ""))


def header(title):
    print("\n" + RULE + "\n" + title + "\n" + RULE)


# ---------------------------------------------------------------------------
# 1. stencil check
# ---------------------------------------------------------------------------


def section_stencil():
    header("1. STENCIL CHECK   L{u} vs  lap(u) + (h^2/12) lap^2(u)   [h = 1/8]")
    rng = np.random.default_rng(20240917)
    Crand = np.zeros((6, 6))
    for i in range(6):
        for j in range(6 - i):
            Crand[i, j] = rng.standard_normal()
    polys = {
        "x^2": {(2, 0): 1.0},
        "y^2": {(0, 2): 1.0},
        "x^2 y^2": {(2, 2): 1.0},
        "x^5 + 2x^3y^2 - 3xy^4 + x^2y^2 + x - y + 1":
            {(5, 0): 1.0, (3, 2): 2.0, (1, 4): -3.0, (2, 2): 1.0,
             (1, 0): 1.0, (0, 1): -1.0, (0, 0): 1.0},
    }
    grid = H.Grid(8, 8)
    X, Y = grid.interior_nodes()
    print("%-44s %14s %14s" % ("polynomial", "9-point", "5-point"))
    worst9 = 0.0
    for name, entries in list(polys.items()) + [("random degree-5", None)]:
        C = Crand if entries is None else np.array(
            [[entries.get((i, j), 0.0) for j in range(6)] for i in range(6)])
        lap = H.poly_laplacian(C)
        ref = H.poly_eval(lap, X, Y) + (grid.h ** 2 / 12.0) * H.poly_eval(
            H.poly_laplacian(lap), X, Y)
        devs = []
        for st in (H.LAPLACIAN_9PT, H.LAPLACIAN_5PT):
            L = H.build_laplacian(grid, st)
            b = H.boundary_vector(grid, lambda x, y, t: H.poly_eval(C, x, y), 0.0, st)
            U = grid.ravel(H.poly_eval(C, X, Y))
            devs.append(np.abs(grid.unravel(L @ U + b) - ref).max())
        worst9 = max(worst9, devs[0])
        print("%-44s %14.3e %14.3e" % (name[:44], devs[0], devs[1]))
    print("\n   (the 5-point column is *supposed* to be nonzero: its truncation")
    print("    error is the anisotropic (h^2/12)(u_xxxx + u_yyyy), not (h^2/12) lap^2)")
    check("9-point reproduces lap + (h^2/12) lap^2 to round-off",
          worst9 < 1e-10, "max deviation %.2e" % worst9)


# ---------------------------------------------------------------------------
# 2. convergence
# ---------------------------------------------------------------------------


def exact_sine(x, y, t):
    return np.exp(-2.0 * np.pi ** 2 * ALPHA * t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def section_convergence(T=0.1, Ns=(8, 16, 32, 64)):
    header("2. CONVERGENCE   u = exp(-2 pi^2 t) sin(pi x) sin(pi y),  T = %.2f" % T)
    steps0 = max(1, int(np.ceil(T * Ns[0])))
    print("   Crank-Nicolson, dt proportional to h (dt/h = %.2f, held fixed so the"
          % (T / steps0 * Ns[0]))
    print("   O(dt^2) and O(h^2) errors refine together); dt = h exactly would need")
    print("   %.1f steps on the coarsest grid, so dt is the nearest divisor of T.\n"
          % (T * Ns[0]))
    print("%-10s %10s %7s | %12s %7s | %12s %7s"
          % ("h", "dt", "steps", "err 9-pt", "order", "err 5-pt", "order"))
    prev = {}
    orders = {"9-point": [], "5-point": []}
    for k, N in enumerate(Ns):
        grid = H.Grid(N, N)
        nsteps = steps0 * 2 ** k
        row = []
        for st in (H.LAPLACIAN_9PT, H.LAPLACIAN_5PT):
            U, info = H.solve_heat(grid, lambda x, y: exact_sine(x, y, 0.0), T,
                                   ALPHA, nsteps=nsteps, stencil=st)
            X, Y = grid.interior_nodes()
            err = np.abs(grid.unravel(U) - exact_sine(X, Y, T)).max()
            p = np.log2(prev[st.name] / err) if st.name in prev else np.nan
            if np.isfinite(p):
                orders[st.name].append(p)
            prev[st.name] = err
            row += [err, p]
        print("1/%-8d %10.6f %7d | %12.4e %7s | %12.4e %7s"
              % (N, T / nsteps, nsteps, row[0],
                 "  --  " if np.isnan(row[1]) else "%6.3f" % row[1], row[2],
                 "  --  " if np.isnan(row[3]) else "%6.3f" % row[3]))
    for name in ("9-point", "5-point"):
        check("%s converges at order ~2 (asymptotic order %.3f)"
              % (name, orders[name][-1]), 1.85 < orders[name][-1] < 2.2)

    # nonzero Dirichlet data, space refinement only
    print("\n   Nonzero Dirichlet data g = exp(-2 pi^2 t) cos(pi x) cos(pi y),")
    print("   400 time steps so the time error is frozen and only h refines:")
    exact = lambda x, y, t: (np.exp(-2.0 * np.pi ** 2 * ALPHA * t)
                             * np.cos(np.pi * x) * np.cos(np.pi * y))
    print("   %-8s %12s %7s %12s %7s" % ("h", "err 9-pt", "order", "err 5-pt", "order"))
    prev, last = {}, {}
    for N in (16, 32, 64):
        grid = H.Grid(N, N)
        cells = []
        for st in (H.LAPLACIAN_9PT, H.LAPLACIAN_5PT):
            U, _ = H.solve_heat(grid, lambda x, y: exact(x, y, 0.0), 0.1, ALPHA,
                                nsteps=400, stencil=st, g=exact)
            X, Y = grid.interior_nodes()
            err = np.abs(grid.unravel(U) - exact(X, Y, 0.1)).max()
            p = np.log2(prev[st.name] / err) if st.name in prev else np.nan
            prev[st.name] = err
            last[st.name] = p
            cells += [err, p]
        print("   1/%-6d %12.4e %7s %12.4e %7s"
              % (N, cells[0], "  --  " if np.isnan(cells[1]) else "%6.3f" % cells[1],
                 cells[2], "  --  " if np.isnan(cells[3]) else "%6.3f" % cells[3]))
    check("nonzero Dirichlet boundary vector keeps order 2",
          all(1.9 < last[k] < 2.1 for k in last),
          "orders %.3f / %.3f" % (last["9-point"], last["5-point"]))


# ---------------------------------------------------------------------------
# 3. Fourier symbol
# ---------------------------------------------------------------------------


def section_symbol():
    header("3. FOURIER SYMBOL   h^2 lambda = -4 (s + c - (2/3) s c),"
           "  s = sin^2(th/2), c = sin^2(ph/2)")
    N, h = 8, 0.125
    L = H.build_laplacian_periodic(N, N, h, H.LAPLACIAN_9PT)
    P, Q = np.meshgrid(np.arange(N), np.arange(N), indexing="xy")
    modes = [(0, 0), (1, 0), (2, 0), (2, 2), (1, 3), (4, 2), (4, 4)]
    print("%14s %14s %16s %16s %12s"
          % ("theta", "phi", "L applied to mode", "closed form", "|diff|"))
    worst = 0.0
    for j, m in modes:
        th, ph = 2 * np.pi * j / N, 2 * np.pi * m / N
        v = np.exp(1j * (P * th + Q * ph)).ravel()
        lam = ((L @ v.real) + 1j * (L @ v.imag)) / v        # exact eigenvector
        num = h ** 2 * lam.mean().real
        closed = H.symbol_9pt(th, ph)
        worst = max(worst, abs(num - closed))
        print("%14s %14s %16.10f %16.10f %12.2e"
              % ("%d*pi/4" % j, "%d*pi/4" % m, num, closed, abs(num - closed)))
    cb = H.symbol_9pt(np.pi, np.pi)
    print("\n   checkerboard (theta = phi = pi):  h^2 lambda = %.12f   (-16/3 = %.12f)"
          % (cb, -16.0 / 3.0))
    print("   5-point checkerboard:             h^2 lambda = %.12f   (-8)"
          % H.symbol_5pt(np.pi, np.pi))
    check("periodic operator matches the closed-form symbol", worst < 1e-12,
          "max |diff| %.2e" % worst)
    check("checkerboard symbol is -16/3", abs(cb + 16.0 / 3.0) < 1e-14)


# ---------------------------------------------------------------------------
# 4. stability
# ---------------------------------------------------------------------------


def section_stability():
    header("4. STABILITY   mu = alpha dt / h^2")
    grid = H.Grid(32, 32)
    h = grid.h
    rng = np.random.default_rng(7)
    U0 = rng.standard_normal(grid.n_interior)

    print("Crank-Nicolson, 200 steps from random (rough) data -- unconditionally stable:")
    print("   %-9s %6s %14s %14s %14s" % ("stencil", "mu", "||U0||_inf",
                                          "max_n ||U||_inf", "2-norm decays"))
    cn_ok = True
    for st in (H.LAPLACIAN_9PT, H.LAPLACIAN_5PT):
        L = H.build_laplacian(grid, st)
        for mu in (0.1, 1.0, 10.0, 100.0):
            sups, nrms = [], []
            H.crank_nicolson_solve(
                L, U0, ALPHA, mu * h ** 2, 200,
                callback=lambda n, t, U: (sups.append(np.abs(U).max()),
                                          nrms.append(np.linalg.norm(U))))
            mono = (nrms[0] <= np.linalg.norm(U0) * (1 + 1e-12) and
                    all(nrms[i + 1] <= nrms[i] * (1 + 1e-12) for i in range(len(nrms) - 1)))
            bounded = max(sups) <= np.linalg.norm(U0) and np.isfinite(nrms[-1])
            cn_ok &= bool(mono and bounded)
            print("   %-9s %6g %14.4f %14.4f %14s"
                  % (st.name, mu, np.abs(U0).max(), max(sups), "yes" if mono else "NO"))
    check("Crank-Nicolson bounded for mu = 0.1, 1, 10, 100", cn_ok)

    print("\nForward Euler, 600 steps -- limit mu <= 3/8 (9-point), mu <= 1/4 (5-point):")
    print("   %-9s %6s %16s %12s" % ("stencil", "mu", "||U||_inf after", "verdict"))
    fe_ok = True
    for st, mus, lim in ((H.LAPLACIAN_9PT, (0.37, 0.39), 3.0 / 8.0),
                         (H.LAPLACIAN_5PT, (0.24, 0.26), 1.0 / 4.0)):
        L = H.build_laplacian(grid, st)
        for mu in mus:
            U = H.forward_euler_solve(L, U0, ALPHA, mu * h ** 2, 600)
            sup = np.abs(U).max()
            grew = not (sup < np.abs(U0).max())
            fe_ok &= (grew == (mu > lim))
            print("   %-9s %6g %16.4e %12s"
                  % (st.name, mu, sup, "blows up" if grew else "stable"))
        print("      theoretical limit 2/|h^2 lambda_min| = 2/%.6f = %.6f"
              % (abs(H.fourier_symbol(st, np.pi, np.pi)),
                 2.0 / abs(H.fourier_symbol(st, np.pi, np.pi))))
    check("forward Euler: 9-point stable at mu=0.37 / blows up at 0.39; "
          "5-point at 0.24 / 0.26", fe_ok)

    print("\nCrank-Nicolson on an exact checkerboard (periodic 16x16):")
    print("   amplification g = (1 - 8mu/3)/(1 + 8mu/3), negative exactly for mu > 3/8")
    Np, hp = 16, 1.0 / 16
    Lp = H.build_laplacian_periodic(Np, Np, hp, H.LAPLACIAN_9PT)
    P, Q = np.meshgrid(np.arange(Np), np.arange(Np), indexing="xy")
    cb = ((-1.0) ** (P + Q)).ravel()
    print("   %6s %12s %28s %10s" % ("mu", "g", "sign of U[0], steps 1..6", "flips"))
    flip_ok = True
    for mu in (0.1, 0.2, 0.3, 0.375, 0.4, 1.0, 10.0):
        signs = []
        H.crank_nicolson_solve(Lp, cb, ALPHA, mu * hp ** 2, 6,
                               callback=lambda n, t, U: signs.append(int(np.sign(U[0]))))
        g = (1 - 8 * mu / 3) / (1 + 8 * mu / 3)
        flips = any(s < 0 for s in signs)
        flip_ok &= (flips == (mu > 3.0 / 8.0))
        print("   %6g %12.5f %28s %10s"
              % (mu, g, " ".join("%+d" % s for s in signs), "yes" if flips else "no"))
    check("checkerboard flips sign under CN exactly when mu > 3/8", flip_ok)


# ---------------------------------------------------------------------------
# 5. isotropy
# ---------------------------------------------------------------------------


def diffuse_gaussian(stencil, N, sigma0_in_h, sigma_end_in_h, mu=0.25):
    grid = H.Grid(N, N)
    h = grid.h
    s0 = sigma0_in_h * h
    T = ((sigma_end_in_h * h) ** 2 - s0 ** 2) / (2.0 * ALPHA)
    u0 = lambda x, y: np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * s0 ** 2))
    nsteps = max(1, int(round(T / (mu * h ** 2))))
    U, _ = H.solve_heat(grid, u0, T, ALPHA, nsteps=nsteps, stencil=stencil)
    return grid, grid.full_field(U), np.sqrt(s0 ** 2 + 2 * ALPHA * T), nsteps


def rays(grid, F, rmax):
    """Quintic interpolant plus the axis/diagonal profiles and half-max radii."""
    f = RegularGridInterpolator((grid.y, grid.x), F, method="quintic")
    c, s2 = 0.5, 1.0 / np.sqrt(2.0)
    ray = lambda r, ux, uy: float(f([[c + uy * r, c + ux * r]])[0])
    level = 0.5 * ray(0.0, 1.0, 0.0)
    r_ax = brentq(lambda r: ray(r, 1.0, 0.0) - level, 1e-9, rmax)
    r_dg = brentq(lambda r: ray(r, s2, s2) - level, 1e-9, rmax)
    r = np.linspace(0.0, rmax, 1201)
    pa = f(np.stack([np.full_like(r, c), c + r], axis=1))
    pd = f(np.stack([c + r * s2, c + r * s2], axis=1))
    return f, r, pa, pd, r_ax, r_dg


def azimuthal_deviation(f, sigma, nr=220, nth=481):
    """u - (azimuthal mean of u), normalised by the peak: the anisotropic part."""
    r = np.linspace(1e-6, 4.0 * sigma, nr)
    th = np.linspace(0.0, 2 * np.pi, nth)
    R, TH = np.meshgrid(r, th, indexing="ij")
    X, Y = 0.5 + R * np.cos(TH), 0.5 + R * np.sin(TH)
    U = f(np.stack([Y.ravel(), X.ravel()], axis=1)).reshape(R.shape)
    mean_r = U[:, :-1].mean(axis=1, keepdims=True)
    return X, Y, U, U - mean_r


def section_isotropy(N=96, sigma0_in_h=1.0, sigma_end_in_h=3.0):
    header("5. ISOTROPY   narrow Gaussian at the domain centre, h = 1/%d" % N)
    print("   sigma_0 = %.1f h diffused to sigma = %.1f h; the exact solution stays"
          % (sigma0_in_h, sigma_end_in_h))
    print("   exactly radial, so any axis-vs-diagonal difference is numerical.\n")
    data = {}
    for st in (H.LAPLACIAN_5PT, H.LAPLACIAN_9PT):
        grid, F, sigma, ns = diffuse_gaussian(st, N, sigma0_in_h, sigma_end_in_h)
        f, r, pa, pd, r_ax, r_dg = rays(grid, F, 4.0 * sigma)
        data[st.name] = dict(grid=grid, F=F, sigma=sigma, nsteps=ns, f=f, r=r,
                             pa=pa, pd=pd, r_ax=r_ax, r_dg=r_dg,
                             prof=np.abs(pa - pd).max() / F.max(),
                             spread=abs(r_dg - r_ax) / r_ax)
    # measurement floor: the same pipeline applied to an exactly radial Gaussian
    grid = data["9-point"]["grid"]
    sigma = data["9-point"]["sigma"]
    X, Y = grid.nodes()
    Fb = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * sigma ** 2))
    _, _, pab, pdb, rab, rdb = rays(grid, Fb, 4.0 * sigma)
    floor_prof = np.abs(pab - pdb).max() / Fb.max()
    floor_spread = abs(rdb - rab) / rab

    print("%-22s %16s %16s %16s" % ("", "5-point", "9-point", "ratio 5/9"))
    print("%-22s %16d %16d" % ("time steps", data["5-point"]["nsteps"],
                               data["9-point"]["nsteps"]))
    print("%-22s %16.6f %16.6f" % ("half-max r, axis",
                                   data["5-point"]["r_ax"], data["9-point"]["r_ax"]))
    print("%-22s %16.6f %16.6f" % ("half-max r, diagonal",
                                   data["5-point"]["r_dg"], data["9-point"]["r_dg"]))
    print("%-22s %16.3e %16.3e %16.1f"
          % ("|dr|/r  (spread)", data["5-point"]["spread"], data["9-point"]["spread"],
             data["5-point"]["spread"] / data["9-point"]["spread"]))
    print("%-22s %16.3e %16.3e %16.1f"
          % ("max|u_ax - u_dg|/peak", data["5-point"]["prof"], data["9-point"]["prof"],
             data["5-point"]["prof"] / data["9-point"]["prof"]))
    print("%-22s %16.3e %16.3e" % ("measurement floor", floor_prof, floor_prof))
    print("\n   (the floor is the same pipeline applied to an exactly radial Gaussian;")
    print("    both signals sit well above it, so both numbers are real.)")
    check("9-point anisotropy is far smaller than 5-point",
          data["9-point"]["prof"] < data["5-point"]["prof"] / 20.0 and
          data["9-point"]["spread"] < data["5-point"]["spread"] / 20.0)
    check("both measurements exceed the interpolation floor",
          data["9-point"]["prof"] > 10 * floor_prof and floor_spread < 1e-5)
    make_plot(data, sigma)
    return data


def make_plot(data, sigma, path="isotropy.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    # sequential: one hue, light -> dark (never a rainbow)
    seq = LinearSegmentedColormap.from_list("seq", ["#f2f6fc", "#9cc1ec", BLUE, "#123a6b"])
    # diverging: two hues with a neutral midpoint
    div = LinearSegmentedColormap.from_list(
        "div", ["#8c3a18", ORANGE, "#f6cdbb", "#eceae4", "#bcd6f3", BLUE, "#163f73"])

    fig = plt.figure(figsize=(9.6, 10.8), dpi=170, facecolor=SURF)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.72],
                          hspace=0.46, wspace=0.20,
                          left=0.085, right=0.875, top=0.855, bottom=0.065)
    names = ["5-point", "9-point"]
    w = 3.2 * sigma

    # --- row 1: the solutions themselves -----------------------------------
    for j, nm in enumerate(names):
        ax = fig.add_subplot(gs[0, j])
        d = data[nm]
        X, Y = d["grid"].nodes()
        Fn = d["F"] / d["F"].max()
        ax.contourf(X, Y, Fn, levels=np.linspace(0, 1, 11), cmap=seq)
        ax.contour(X, Y, Fn, levels=[0.1, 0.25, 0.5, 0.75],
                   colors="white", linewidths=1.0)
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(0.5 + d["r_ax"] * np.cos(th), 0.5 + d["r_ax"] * np.sin(th),
                color=INK, lw=1.0, ls=(0, (4, 3)))
        ax.set_xlim(0.5 - w, 0.5 + w); ax.set_ylim(0.5 - w, 0.5 + w)
        ax.set_aspect("equal")
        ax.set_title("%s Laplacian" % nm, color=INK, fontsize=11, pad=7)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#d8d6cf")
    fig.text(0.5, 0.965, "Diffusing a narrow Gaussian: 5-point vs compact 9-point",
             ha="center", color=INK, fontsize=14)
    fig.text(0.5, 0.940,
             "Same grid (h = 1/%d), same time step, same final width. "
             "Dashed ring: half-max radius on the x-axis."
             % data["5-point"]["grid"].Nx,
             ha="center", color=INK2, fontsize=9)
    fig.text(0.5, 0.922,
             "Both panels in each row share one colour scale.",
             ha="center", color=INK2, fontsize=9)

    # --- row 2: deviation from the azimuthal mean (shared scale) -----------
    devs = {}
    for nm in names:
        Xp, Yp, _, D = azimuthal_deviation(data[nm]["f"], sigma)
        devs[nm] = (Xp, Yp, D / data[nm]["F"].max())
    vmax = max(np.abs(devs[nm][2]).max() for nm in names)
    levels = np.linspace(-vmax, vmax, 21)
    for j, nm in enumerate(names):
        ax = fig.add_subplot(gs[1, j])
        Xp, Yp, D = devs[nm]
        cf = ax.contourf(Xp, Yp, D, levels=levels, cmap=div, extend="both")
        ax.set_xlim(0.5 - w, 0.5 + w); ax.set_ylim(0.5 - w, 0.5 + w)
        ax.set_aspect("equal")
        ax.set_title("%s: deviation from radial mean\npeak %.2e of max(u)"
                     % (nm, np.abs(D).max()), color=INK, fontsize=10, pad=7)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#d8d6cf")
        if j == 1:
            box = ax.get_position()
            cax = fig.add_axes([0.895, box.y0, 0.018, box.height])
            cb = fig.colorbar(cf, cax=cax, ticks=[-vmax, 0, vmax])
            cb.ax.set_yticklabels(["%.0e" % -vmax, "0", "%.0e" % vmax])
            cb.ax.tick_params(colors=INK2, labelsize=8)
            cb.outline.set_edgecolor("#d8d6cf")

    # --- row 3: axis vs diagonal profile difference ------------------------
    ax = fig.add_subplot(gs[2, :])
    for nm, col in ((names[0], ORANGE), (names[1], BLUE)):
        d = data[nm]
        ax.semilogy(d["r"] / sigma, np.maximum(np.abs(d["pa"] - d["pd"]) / d["F"].max(),
                                               1e-16),
                    color=col, lw=2.0, label="%s" % nm)
        k = int(0.45 * len(d["r"]))
        ax.annotate(nm, (d["r"][k] / sigma,
                         np.abs(d["pa"] - d["pd"])[k] / d["F"].max()),
                    color=col, fontsize=9, xytext=(4, 2), textcoords="offset points")
    ax.set_ylim(1e-8, 3e-2)
    ax.set_xlabel("radius  r / sigma", color=INK2, fontsize=9)
    ax.set_ylabel("|u(axis) - u(diagonal)| / max u", color=INK2, fontsize=9)
    ax.set_title("Axis-vs-diagonal disagreement at equal radius (lower is more isotropic)",
                 color=INK, fontsize=10, pad=7)
    ax.grid(True, which="major", color="#e8e6df", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#d8d6cf")

    fig.savefig(path, facecolor=SURF)
    plt.close(fig)
    print("\n   wrote %s" % path)


# ---------------------------------------------------------------------------


def main():
    print(RULE)
    print("heat9 -- compact 9-point Laplacian + Crank-Nicolson, verification summary")
    print(RULE)
    print("stencil:  1/(6h^2) * [[1, 4, 1], [4, -20, 4], [1, 4, 1]]")
    print("CN step:  (I - (alpha dt/2) L) U^{n+1} = (I + (alpha dt/2) L) U^n"
          " + (alpha dt/2)(b^{n+1} + b^n)")
    section_stencil()
    section_convergence()
    section_symbol()
    section_stability()
    section_isotropy()
    header("SUMMARY")
    for label, ok in _checks:
        print("   [%s] %s" % ("PASS" if ok else "FAIL", label))
    bad = [l for l, ok in _checks if not ok]
    print("\n   %d/%d checks passed" % (len(_checks) - len(bad), len(_checks)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
