"""Error analysis for the compact 9-point / Crank-Nicolson heat solver.

Exact solution used
-------------------
Every term  c * exp(-alpha (a^2 + b^2) t) * sin(a x + p) * sin(b y + q)
solves  u_t = alpha (u_xx + u_yy)  exactly, and so does any superposition of
them.  We use a four-mode superposition with generic phases, so that

  * several length scales are present at once (a, b run from pi to 3 pi),
    rather than the single grid-aligned eigenmode that a one-mode test uses;
  * the phases make u nonzero and time dependent on all four sides of the
    square, so the Dirichlet boundary vector b(t) is exercised too.

The three studies
-----------------
Because the scheme has two discretisation parameters, a single refinement
cannot reveal both orders.  Each study isolates one:

  A. SPACE   refine h with dt = 0.25 h^2, so the O(dt^2) time error is
             O(h^4) and the measured slope is the spatial order alone.
  B. TIME    fix h = 1/128 and refine dt, measured against a dt-converged
             solution on that same grid.  Subtracting a reference computed
             on the same mesh removes the spatial error exactly, so the
             slope is the time order alone.  (Measured instead against the
             analytic solution, the curve would flatten onto the spatial
             error floor -- the script prints that version too.)
  C. BOTH    refine h with dt proportional to h, the way the solver is
             actually run.  This is the headline number.

All errors are max-norm (discrete L-infinity) over the interior nodes at the
final time.  Each study reports pairwise observed orders
log2(e_{k-1}/e_k) and a least-squares slope of log(error) against
log(parameter) -- the slope of the log-log plot.
"""

from __future__ import annotations

import numpy as np

import heat9 as H

ALPHA = 1.0
T_FINAL = 0.02

#: (coefficient, a, b, phase_x, phase_y).  The wavenumbers are integer
#: multiples of pi, so with ZERO phases every term is a Dirichlet
#: eigenfunction of the unit square and vanishes identically on the whole
#: boundary.  Turning the phases on shifts each sinusoid off the boundary
#: nodes and makes g nonzero and time dependent -- same wavenumbers, same
#: decay rates, so the two settings are directly comparable.
_BASE_MODES = [
    (1.00, 1.0 * np.pi, 1.0 * np.pi, 0.0, 0.0),
    (0.60, 2.0 * np.pi, 1.0 * np.pi, 0.3, 0.7),
    (-0.40, 1.0 * np.pi, 3.0 * np.pi, 1.1, 0.2),
    (0.25, 3.0 * np.pi, 2.0 * np.pi, 0.5, 1.3),
]

#: "homogeneous" -> u = 0 on all of dD (phases dropped, b(t) identically zero)
#: "constant"    -> u = CONSTANT on all of dD, the same value everywhere and
#:                  for all time.  Because the Laplacian annihilates a constant
#:                  and a constant has no time derivative, v = u - CONSTANT
#:                  solves the SAME heat equation with zero boundary data: so
#:                  the exact solution is just the homogeneous one shifted up by
#:                  CONSTANT, and u relaxes to the flat state u == CONSTANT.
#: "analytic"    -> u = g(x, y, t) on dD, nonzero and time dependent
BOUNDARY = "homogeneous"

#: the boundary value used when BOUNDARY == "constant"
CONSTANT = 1.0

MODES = [(c, a, b, 0.0, 0.0) for c, a, b, p, q in _BASE_MODES]


def set_boundary(mode):
    """Select homogeneous (u = 0 on dD) or analytic Dirichlet data."""
    global BOUNDARY, MODES
    if mode not in ("homogeneous", "constant", "analytic"):
        raise ValueError("boundary must be 'homogeneous', 'constant' or 'analytic'")
    BOUNDARY = mode
    MODES = (list(_BASE_MODES) if mode == "analytic"
             else [(c, a, b, 0.0, 0.0) for c, a, b, p, q in _BASE_MODES])


def bc():
    """The g handed to the solver for the selected boundary condition."""
    if BOUNDARY == "homogeneous":
        return None                      # b(t) is identically zero
    if BOUNDARY == "constant":
        return lambda x, y, t: np.full(
            np.broadcast(np.asarray(x), np.asarray(y)).shape, CONSTANT)
    return exact


def exact(x, y, t):
    """u(x, y, t) = sum_j c_j exp(-alpha (a^2+b^2) t) sin(a x + p) sin(b y + q)."""
    out = np.zeros(np.broadcast(x, y).shape, dtype=float)
    for c, a, b, p, q in MODES:
        out = out + c * np.exp(-ALPHA * (a * a + b * b) * t) \
            * np.sin(a * x + p) * np.sin(b * y + q)
    if BOUNDARY == "constant":
        out = out + CONSTANT             # v = u - CONSTANT solves the same PDE
    return out


def max_error(grid, U, t):
    X, Y = grid.interior_nodes()
    return float(np.abs(grid.unravel(U) - exact(X, Y, t)).max())


def slope(params, errors):
    """Least-squares slope of log(error) vs log(parameter): the log-log slope."""
    return float(np.polyfit(np.log(np.asarray(params)), np.log(np.asarray(errors)), 1)[0])


def orders(errors):
    """Pairwise observed orders for a sequence that halves the parameter."""
    e = np.asarray(errors)
    return [float(np.log2(e[i - 1] / e[i])) for i in range(1, len(e))]


def _table(param_name, params, results, extra=None):
    names = list(results)
    head = "%-12s" % param_name
    for n in names:
        head += " | %13s %7s" % ("err " + n, "order")
    print(head)
    print("-" * len(head))
    ords = {n: [np.nan] + orders(results[n]) for n in names}
    for i, p in enumerate(params):
        line = "%-12s" % (extra[i] if extra else "%.6g" % p)
        for n in names:
            o = ords[n][i]
            line += " | %13.4e %7s" % (results[n][i],
                                       "  --  " if np.isnan(o) else "%6.3f" % o)
        print(line)
    print()


# ---------------------------------------------------------------------------
# A. spatial order:  dt = 0.25 h^2  =>  time error is O(h^4)
# ---------------------------------------------------------------------------


def study_space(Ns=(8, 16, 32, 64, 128), mu=0.25):
    print("=" * 78)
    print("A. SPATIAL ORDER   dt = %.2f h^2, so the O(dt^2) time error is O(h^4)" % mu)
    print("=" * 78)
    hs, res, labels = [], {"9-point": [], "5-point": []}, []
    for N in Ns:
        grid = H.Grid(N, N)
        nsteps = max(1, int(round(T_FINAL / (mu * grid.h ** 2))))
        hs.append(grid.h)
        labels.append("1/%-4d (%d)" % (N, nsteps))
        for st in (H.LAPLACIAN_9PT, H.LAPLACIAN_5PT):
            U, _ = H.solve_heat(grid, lambda x, y: exact(x, y, 0.0), T_FINAL,
                                ALPHA, nsteps=nsteps, stencil=st, g=bc())
            res[st.name].append(max_error(grid, U, T_FINAL))
    _table("h (steps)", hs, res, labels)
    sl = {n: slope(hs, res[n]) for n in res}
    for n in sl:
        print("   least-squares slope, %s:  %.4f" % (n, sl[n]))
    print()
    return hs, res, sl


# ---------------------------------------------------------------------------
# B. temporal order: fixed grid, reference solution on that same grid
# ---------------------------------------------------------------------------


def study_time(N=128, nss=(1, 2, 4, 8, 16, 32, 64, 128, 256), ns_ref=4096,
               fit=(1, 7)):
    print("=" * 78)
    print("B. TEMPORAL ORDER   h = 1/%d fixed; error measured against a "
          "dt-converged" % N)
    print("   solution on the SAME grid (%d steps), which removes the spatial error"
          % ns_ref)
    print("=" * 78)
    grid = H.Grid(N, N)
    u0 = lambda x, y: exact(x, y, 0.0)
    Uref, _ = H.solve_heat(grid, u0, T_FINAL, ALPHA, nsteps=ns_ref,
                           stencil=H.LAPLACIAN_9PT, g=bc())
    dts, res, labels = [], {"vs reference": [], "vs analytic": []}, []
    for ns in nss:
        U, info = H.solve_heat(grid, u0, T_FINAL, ALPHA, nsteps=ns,
                               stencil=H.LAPLACIAN_9PT, g=bc())
        dts.append(info["dt"])
        labels.append("%.3e (%d)" % (info["dt"], ns))
        res["vs reference"].append(float(np.abs(U - Uref).max()))
        res["vs analytic"].append(max_error(grid, U, T_FINAL))
    _table("dt (steps)", dts, res, labels)
    lo, hi = fit
    sl = slope(dts[lo:hi], res["vs reference"][lo:hi])
    print("   least-squares slope, vs reference (dt = %.1e .. %.1e):  %.4f"
          % (dts[hi - 1], dts[lo], sl))
    print("   the 'vs analytic' column stops improving once the time error drops")
    print("   below the spatial error of this grid -- that floor is the O(h^2) term.")
    print()
    return dts, res, sl


# ---------------------------------------------------------------------------
# C. combined refinement, dt proportional to h -- how the solver is really run
# ---------------------------------------------------------------------------


def study_combined(Ns=(8, 16, 32, 64, 128)):
    print("=" * 78)
    print("C. COMBINED   dt proportional to h (dt/h held fixed) -- the headline rate")
    print("=" * 78)
    hs, res, labels = [], {"9-point": [], "5-point": []}, []
    for k, N in enumerate(Ns):
        grid = H.Grid(N, N)
        nsteps = 2 ** k
        hs.append(grid.h)
        labels.append("1/%-4d (%d)" % (N, nsteps))
        for st in (H.LAPLACIAN_9PT, H.LAPLACIAN_5PT):
            U, info = H.solve_heat(grid, lambda x, y: exact(x, y, 0.0), T_FINAL,
                                   ALPHA, nsteps=nsteps, stencil=st, g=bc())
            res[st.name].append(max_error(grid, U, T_FINAL))
        ratio = info["dt"] / grid.h
    print("   dt/h = %.3f on every level\n" % ratio)
    _table("h (steps)", hs, res, labels)
    sl = {n: slope(hs, res[n]) for n in res}
    for n in sl:
        print("   least-squares slope, %s:  %.4f" % (n, sl[n]))
    print()
    return hs, res, sl


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def make_plot(space, time, combined, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = {"homogeneous": "reports/error_analysis.png",
            "constant": "reports/error_analysis_constant_bc.png",
            "analytic": "reports/error_analysis_analytic_bc.png"}[BOUNDARY]
    INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
    BLUE, ORANGE, GUIDE = "#2a78d6", "#eb6834", "#a8a69c"
    hs, sres, ssl = space
    dts, tres, tsl = time
    hc, cres, csl = combined

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.2), dpi=170, facecolor=SURF)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.145, wspace=0.26)

    def guide(ax, xs, ys, order, where=0.45, label=None):
        """Dashed reference line of the given slope, anchored under the data."""
        xs, ys = np.asarray(xs), np.asarray(ys)
        x0 = np.exp(np.log(xs).mean())
        y0 = np.exp(np.log(ys).mean()) * where
        xl = np.array([xs.min(), xs.max()])
        ax.plot(xl, y0 * (xl / x0) ** order, color=GUIDE, lw=1.4,
                ls=(0, (5, 4)), zorder=1)
        ax.annotate(label or "slope %d" % order,
                    (xl[1], y0 * (xl[1] / x0) ** order), color=INK2, fontsize=9,
                    xytext=(-6, -14), textcoords="offset points", ha="right")

    def style(ax, xlabel, title, subtitle):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
        ax.set_ylabel("max-norm error at t = %.2f" % T_FINAL, color=INK2, fontsize=10)
        ax.set_title(title, color=INK, fontsize=11.5, pad=16)
        ax.annotate(subtitle, (0.5, 1.012), xycoords="axes fraction",
                    ha="center", color=INK2, fontsize=9)
        ax.grid(True, which="major", color="#e4e2db", lw=0.8)
        ax.grid(True, which="minor", color="#f0eee8", lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=9)
        for s in ax.spines.values():
            s.set_color("#d8d6cf")

    # --- A: space -----------------------------------------------------------
    ax = axes[0]
    for name, col in (("9-point", BLUE), ("5-point", ORANGE)):
        ax.plot(hs, sres[name], color=col, lw=2.0, marker="o", ms=6,
                label="%s   slope %.3f" % (name, ssl[name]), zorder=3)
    guide(ax, hs, sres["9-point"], 2)
    style(ax, "grid spacing  h", "A.  Spatial order",
          "dt = 0.25 h$^2$, so the time error is O(h$^4$)")
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left")

    # --- B: time ------------------------------------------------------------
    ax = axes[1]
    ax.plot(dts, tres["vs reference"], color=BLUE, lw=2.0, marker="o", ms=6,
            label="vs dt-converged ref   slope %.3f" % tsl, zorder=3)
    ax.plot(dts, tres["vs analytic"], color=ORANGE, lw=2.0, marker="s", ms=5.5,
            label="vs analytic solution", zorder=3)
    ax.axhline(tres["vs analytic"][-1], color=GUIDE, lw=1.4, ls=(0, (2, 3)))
    ax.annotate("spatial error floor", (dts[-1], tres["vs analytic"][-1]),
                color=INK2, fontsize=9, xytext=(4, -15), textcoords="offset points")
    guide(ax, dts, tres["vs reference"], 2, where=0.25)
    style(ax, "time step  dt", "B.  Temporal order",
          "h = 1/128 fixed")
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left")

    # --- C: combined --------------------------------------------------------
    ax = axes[2]
    for name, col in (("9-point", BLUE), ("5-point", ORANGE)):
        ax.plot(hc, cres[name], color=col, lw=2.0, marker="o", ms=6,
                label="%s   slope %.3f" % (name, csl[name]), zorder=3)
    guide(ax, hc, cres["9-point"], 2)
    style(ax, "grid spacing  h  (dt proportional to h)", "C.  Combined refinement",
          "how the solver is actually run")
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left")

    fig.text(0.5, 0.955, "Error analysis: compact 9-point Laplacian + Crank-Nicolson",
             ha="center", color=INK, fontsize=15)
    fig.text(0.5, 0.905,
             "2D heat equation, four-mode exact solution, %s.  "
             "Both axes logarithmic, so the slope is the order."
             % {"homogeneous": "homogeneous Dirichlet u = 0 on all of the boundary",
                "constant": "constant Dirichlet u = %g on all of the boundary" % CONSTANT,
                "analytic": "nonzero time-dependent Dirichlet data"}[BOUNDARY],
             ha="center", color=INK2, fontsize=9.5)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)
    print("wrote %s\n" % path)


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if "--bc" in argv:
        set_boundary(argv[argv.index("--bc") + 1])
    print("Exact solution: sum of %d modes "
          "c exp(-alpha(a^2+b^2)t) sin(ax+p) sin(by+q)" % len(MODES))
    print("alpha = %g, final time T = %g, domain [0,1]^2" % (ALPHA, T_FINAL))
    if BOUNDARY == "homogeneous":
        print("boundary: HOMOGENEOUS Dirichlet, u = 0 on all of dD "
              "(phases zero, b(t) == 0)\n")
    elif BOUNDARY == "constant":
        print("boundary: CONSTANT Dirichlet, u = %g on all of dD, "
              "same value for all t" % CONSTANT)
        print("          exact solution = %g + (the homogeneous solution)\n" % CONSTANT)
    else:
        print("boundary: analytic Dirichlet, nonzero and time dependent\n")
    space = study_space()
    time_ = study_time()
    combined = study_combined()
    make_plot(space, time_, combined)
    print("=" * 78)
    print("SLOPES (the big-O exponents)")
    print("=" * 78)
    print("   A. space only, 9-point        error = O(h^%.2f)" % space[2]["9-point"])
    print("   A. space only, 5-point        error = O(h^%.2f)" % space[2]["5-point"])
    print("   B. time only  (9-point)       error = O(dt^%.2f)" % time_[2])
    print("   C. combined,   9-point        error = O(h^%.2f)" % combined[2]["9-point"])
    print("   C. combined,   5-point        error = O(h^%.2f)" % combined[2]["5-point"])
    print("\n   Both discretisations are second order, so the scheme is")
    print("   O(h^2 + dt^2); with dt proportional to h that is O(h^2) overall.")


if __name__ == "__main__":
    main()
