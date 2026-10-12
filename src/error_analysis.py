"""Error analysis for the compact 9-point / Crank-Nicolson heat solver.

Exact solution used
-------------------
Every term  c * exp(-alpha (a^2 + b^2) t) * sin(a x + p) * sin(b y + q)
solves  u_t = alpha (u_xx + u_yy)  exactly, and so does any superposition of
them.  We use a four-mode superposition, with a, b running from pi to 3 pi,
so several length scales are present at once.

The boundary condition decides what the test exercises:

  * "homogeneous" (default): the phases are zero, so u = 0 on the boundary
    and b(t) == 0.  Every mode is then an EXACT eigenvector of the discrete
    Dirichlet operator, so the error is pure eigenvalue error -- and it can be
    split exactly into space, time and roundoff parts (see error_budget).
  * "analytic": generic phases make u nonzero and time dependent on all four
    sides, so the Dirichlet boundary vector b(t) is exercised and the modes
    are no longer grid eigenvectors.  This is the more demanding test.
  * "constant": u = CONSTANT on the boundary.  The stencil rows sum to zero,
    so the discrete solution is the homogeneous one shifted by CONSTANT; the
    error matches the homogeneous case to within roundoff.  It is a
    consistency and roundoff check (LU and b are each O(1/h^2) and cancel),
    not a separate accuracy test.

The studies
-----------
Because the scheme has two discretisation parameters, a single refinement
cannot reveal both orders.  Each study isolates one:

  A. SPACE   refine h with dt = 0.25 h^2, so the O(dt^2) time error is
             O(h^4) and the measured slope is the spatial order alone.
  B. TIME    fix h = 1/128 and refine dt, measured against a dt-converged
             solution on that same grid.  Subtracting a reference computed
             on the same mesh removes the spatial error, so the slope is the
             time order alone.  Measured against the analytic solution the
             error is the SIGNED sum of the time and space parts, which can
             cancel -- the "vs analytic" column dips below the spatial error
             before settling onto it.
  C. BOTH    refine h with dt proportional to h, the way the solver is
             actually run.  The space and time parts are printed separately,
             because at dt/h = 0.16 the time part is the larger one.
  D. BUDGET  (homogeneous and constant only) the exact split of the error
             into space truncation, time truncation and roundoff.

All errors are max-norm (discrete L-infinity) over the interior nodes at the
final time.  Each study reports pairwise observed orders
log2(e_{k-1}/e_k), a roundoff floor, and a least-squares slope of
log(error) against log(parameter) fitted only over the asymptotic window:
the longest run of levels whose pairwise orders agree to within ORDER_TOL
and sit well above the roundoff floor.

In VS Code the  # %%  markers make each section a runnable cell.
"""

# %% setup

from __future__ import annotations

import numpy as np

import heat9 as H

ALPHA = 1.0
T_FINAL = 0.02

#: machine epsilon for float64
EPS = float(np.finfo(float).eps)

#: an error within this factor of its roundoff floor gives no reliable order
FLOOR_MARGIN = 100.0

#: pairwise orders within this spread count as converged (asymptotic window)
ORDER_TOL = 0.05

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
    """Select homogeneous (u = 0 on dD), constant or analytic Dirichlet data."""
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


def u0(x, y):
    return exact(x, y, 0.0)


def run(grid, nsteps, stencil=H.LAPLACIAN_9PT):
    """Solve to T_FINAL in ``nsteps`` Crank-Nicolson steps; return U."""
    U, _ = H.solve_heat(grid, u0, T_FINAL, ALPHA, nsteps=nsteps,
                        stencil=stencil, g=bc())
    return U


def max_error(grid, U, t):
    X, Y = grid.interior_nodes()
    return float(np.abs(grid.unravel(U) - exact(X, Y, t)).max())


# %% roundoff floor and exact error budget


def kappa_A(grid, dt, stencil=H.LAPLACIAN_9PT):
    """Upper bound on the 2-norm condition number of A = I - (alpha dt/2) L.

    L is symmetric negative definite with spectrum inside
    [h^-2 * min symbol, 0), and the most negative symbol value sits at the
    checkerboard (-16/3 for the 9-point stencil, -8 for the 5-point one).
    """
    smin = float(H.fourier_symbol(stencil, np.pi, np.pi))
    return 1.0 + 0.5 * ALPHA * dt * abs(smin) / grid.h ** 2


def abs_norm_B(grid, dt, stencil=H.LAPLACIAN_9PT):
    """Max row sum of |B|, B = I + (alpha dt/2) L: the size of the terms the
    matvec B U adds up before they cancel to O(1)."""
    wsum = sum(abs(w) for w in stencil.weights.values())
    return 1.0 + 0.5 * ALPHA * dt * stencil.scale * wsum / grid.h ** 2


def roundoff_floor(grid, nsteps, stencil, U):
    """A priori roundoff estimate

        eps * (1 + nsteps * (kappa(A) + || |B| ||_inf)) * max|U|.

    Each step is a matvec B U whose O(mu) terms cancel to O(1) (error
    ~ eps * ||B||; A^-1 has norm <= 1, so it passes through) plus a
    backward-stable LU solve (error ~ eps * kappa(A)); the 1 is the sampling
    of the initial data.  Crank-Nicolson does not damp the stiff modes that
    carry this error (|R| -> 1), so it adds up over the steps.  Used as the
    floor below which observed orders mean nothing.
    """
    dt = T_FINAL / nsteps
    per_step = kappa_A(grid, dt, stencil) + abs_norm_B(grid, dt, stencil)
    return EPS * (1.0 + nsteps * per_step) * float(np.abs(U).max())


def error_budget(grid, nsteps, stencil=H.LAPLACIAN_9PT, U=None):
    """Exact split of the error into space, time and roundoff parts.

    Needs BOUNDARY "homogeneous" or "constant", where each mode
    v = sin(a x) sin(b y) is an exact eigenvector of the discrete operator
    with eigenvalue lam_h = symbol(a h, b h) / h^2.  Two answers then have
    closed forms:

        U_ode = sum c exp(alpha lam_h T) v      exact in time
        U_cn  = sum c R(z)^N v,  R(z) = (1 + z/2)/(1 - z/2),  z = alpha dt lam_h

    and the total error telescopes:

        U - u = (U - U_cn) + (U_cn - U_ode) + (U_ode - u)
                 roundoff   time truncation   space truncation

    Returns the four SIGNED fields (shape of the interior) in a dict with keys
    "total", "space", "time", "roundoff".
    """
    if BOUNDARY == "analytic":
        raise ValueError("error_budget needs homogeneous or constant boundary data:"
                         " with analytic data the modes are not grid eigenvectors")
    if U is None:
        U = run(grid, nsteps, stencil)
    X, Y = grid.interior_nodes()
    h, dt = grid.h, T_FINAL / nsteps
    shift = CONSTANT if BOUNDARY == "constant" else 0.0
    ode = np.full(X.shape, shift)
    cn = np.full(X.shape, shift)
    for c, a, b, _, _ in MODES:
        lam = float(H.fourier_symbol(stencil, a * h, b * h)) / h ** 2
        z = ALPHA * dt * lam
        v = c * np.sin(a * X) * np.sin(b * Y)
        ode = ode + np.exp(ALPHA * lam * T_FINAL) * v
        cn = cn + ((1.0 + 0.5 * z) / (1.0 - 0.5 * z)) ** nsteps * v
    u = exact(X, Y, T_FINAL)
    U = grid.unravel(U)
    return {"total": U - u, "space": ode - u, "time": cn - ode, "roundoff": U - cn}


# %% orders, slopes and tables


def orders(errors, floors=None, margin=FLOOR_MARGIN):
    """Pairwise observed orders for a sequence that halves the parameter.

    An order is None when the finer error is within ``margin`` of its
    roundoff floor: there the ratio measures roundoff, not truncation.
    """
    e = np.asarray(errors, dtype=float)
    f = np.zeros_like(e) if floors is None else np.asarray(floors, dtype=float)
    return [None if e[i] < margin * f[i] else float(np.log2(e[i - 1] / e[i]))
            for i in range(1, len(e))]


def asymptotic_window(errors, floors=None, tol=ORDER_TOL):
    """Levels ``(lo, hi)`` (a slice) of the asymptotic range.

    The longest run of consecutive pairwise orders that are all valid (above
    the roundoff floor) and agree to within ``tol``; ties go to the finer
    levels.  Returns None if no pairwise order is valid.
    """
    o = orders(errors, floors)
    best = None
    for i in range(len(o)):
        if o[i] is None:
            continue
        lo_o = hi_o = o[i]
        for j in range(i, len(o)):
            if o[j] is None:
                break
            lo_o, hi_o = min(lo_o, o[j]), max(hi_o, o[j])
            if hi_o - lo_o > tol:
                break
            if best is None or (j - i) >= (best[1] - best[0]):
                best = (i, j)
    return None if best is None else (best[0], best[1] + 2)


def slope(params, errors, window=None):
    """Least-squares slope of log(error) vs log(parameter) over ``window``."""
    lo, hi = window if window is not None else (0, len(params))
    return float(np.polyfit(np.log(np.asarray(params[lo:hi])),
                            np.log(np.asarray(errors[lo:hi])), 1)[0])


def fit(params, errors, floors=None):
    """Slope over the asymptotic window; returns ``(slope, window)``."""
    win = asymptotic_window(errors, floors)
    if win is None:
        return float("nan"), None
    return slope(params, errors, win), win


def _table(param_name, params, results, extra=None, floors=None):
    """Print errors and pairwise orders, plus one roundoff-floor column.

    ``floors`` maps a result name to its floors; orders whose errors are
    within FLOOR_MARGIN of the floor print as "floor".
    """
    floors = floors or {}
    names = list(results)
    head = "%-16s" % param_name
    for n in names:
        head += " | %13s %7s" % ("err " + n, "order")
    head += " | %10s" % "roundoff"
    print(head)
    print("-" * len(head))
    ords = {n: [False] + orders(results[n], floors.get(n)) for n in names}
    for i, p in enumerate(params):
        line = "%-16s" % (extra[i] if extra else "%.6g" % p)
        for n in names:
            o = ords[n][i]
            ostr = "  --  " if o is False else (" floor" if o is None else "%6.3f" % o)
            line += " | %13.4e %7s" % (results[n][i], ostr)
        fl = max((floors[n][i] for n in floors), default=float("nan"))
        line += " | %10.1e" % fl
        print(line)
    print()


def _h_label(h):
    return "h = 1/%d" % round(1.0 / h)


def _dt_label(dt):
    return "dt = %.2e" % dt


def _print_fit(label, params, sl, win, fmt=_h_label):
    if win is None:
        print("   %-30s no valid orders (all at the roundoff floor)" % label)
        return
    lo, hi = win
    print("   %-30s %.4f   (fitted over %s .. %s)"
          % (label, sl, fmt(params[lo]), fmt(params[hi - 1])))


# %% A. spatial order:  dt = 0.25 h^2  =>  time error is O(h^4)


def _space_steps(grid, mu):
    return max(1, int(round(T_FINAL / (mu * grid.h ** 2))))


def study_space(Ns=(8, 16, 32, 64, 128), mu=0.25):
    print("=" * 78)
    print("A. SPATIAL ORDER   dt = %.2f h^2, so the O(dt^2) time error is O(h^4)" % mu)
    print("=" * 78)
    hs, labels = [], []
    res = {"9-point": [], "5-point": []}
    flo = {"9-point": [], "5-point": []}
    for N in Ns:
        grid = H.Grid(N, N)
        nsteps = _space_steps(grid, mu)
        hs.append(grid.h)
        labels.append("1/%-4d (%d)" % (N, nsteps))
        for st in (H.LAPLACIAN_9PT, H.LAPLACIAN_5PT):
            U = run(grid, nsteps, st)
            res[st.name].append(max_error(grid, U, T_FINAL))
            flo[st.name].append(roundoff_floor(grid, nsteps, st, U))
    _table("h (steps)", hs, res, labels, flo)
    sl = {}
    for n in res:
        sl[n], win = fit(hs, res[n], flo[n])
        _print_fit("least-squares slope, %s:" % n, hs, sl[n], win)
    print()
    return hs, res, sl


# %% B. temporal order: fixed grid, reference solution on that same grid


def study_time(N=128, nss=(1, 2, 4, 8, 16, 32, 64, 128, 256), ns_ref=4096):
    print("=" * 78)
    print("B. TEMPORAL ORDER   h = 1/%d fixed; error measured against a "
          "dt-converged" % N)
    print("   solution on the SAME grid (%d steps), which removes the spatial error"
          % ns_ref)
    print("=" * 78)
    grid = H.Grid(N, N)
    Uref = run(grid, ns_ref)
    ref_floor = roundoff_floor(grid, ns_ref, H.LAPLACIAN_9PT, Uref)
    space_err = max_error(grid, Uref, T_FINAL)
    dts, labels = [], []
    res = {"vs reference": [], "vs analytic": []}
    flo = {"vs reference": []}
    for ns in nss:
        U = run(grid, ns)
        dts.append(T_FINAL / ns)
        labels.append("%.3e (%d)" % (T_FINAL / ns, ns))
        res["vs reference"].append(float(np.abs(U - Uref).max()))
        res["vs analytic"].append(max_error(grid, U, T_FINAL))
        flo["vs reference"].append(
            roundoff_floor(grid, ns, H.LAPLACIAN_9PT, U) + ref_floor)
    _table("dt (steps)", dts, res, labels, flo)
    sl, win = fit(dts, res["vs reference"], flo["vs reference"])
    _print_fit("least-squares slope, vs ref:", dts, sl, win, _dt_label)
    print("   spatial error on this grid, |Uref - u|:  %.4e" % space_err)
    dips = [i for i, e in enumerate(res["vs analytic"]) if e < space_err]
    if dips:
        i = dips[0]
        print("   'vs analytic' dips BELOW the spatial error at dt = %.2e (%.4e):"
              % (dts[i], res["vs analytic"][i]))
        print("   the time and space errors have opposite signs and partly cancel,")
        print("   then the column settles onto the spatial error as dt -> 0.")
    print()
    return dts, res, sl, space_err


# %% C. combined refinement, dt proportional to h -- how the solver is really run


def study_combined(Ns=(8, 16, 32, 64, 128), ref_factor=32):
    """dt/h fixed.  A same-grid reference with ``ref_factor`` times as many
    steps splits each 9-point error into its time part |U - Uref| and its
    space part |Uref - u| (the reference's own time error is ref_factor^2
    times smaller)."""
    print("=" * 78)
    print("C. COMBINED   dt proportional to h (dt/h held fixed) -- the headline rate")
    print("=" * 78)
    hs, labels = [], []
    res = {"9-point": [], "5-point": []}
    flo = {"9-point": [], "5-point": []}
    parts = []
    for k, N in enumerate(Ns):
        grid = H.Grid(N, N)
        nsteps = 2 ** k
        hs.append(grid.h)
        labels.append("1/%-4d (%d)" % (N, nsteps))
        for st in (H.LAPLACIAN_9PT, H.LAPLACIAN_5PT):
            U = run(grid, nsteps, st)
            res[st.name].append(max_error(grid, U, T_FINAL))
            flo[st.name].append(roundoff_floor(grid, nsteps, st, U))
            if st is H.LAPLACIAN_9PT:
                Uref = run(grid, nsteps * ref_factor, st)
                parts.append((max_error(grid, Uref, T_FINAL),
                              float(np.abs(U - Uref).max())))
    ratio = (T_FINAL / nsteps) / grid.h
    print("   dt/h = %.3f on every level\n" % ratio)
    _table("h (steps)", hs, res, labels, flo)
    sl = {}
    for n in res:
        sl[n], win = fit(hs, res[n], flo[n])
        _print_fit("least-squares slope, %s:" % n, hs, sl[n], win)
    print()
    print("   9-point error split with a same-grid reference (%dx the steps):" % ref_factor)
    print("   %-14s %12s %12s %12s" % ("h (steps)", "space", "time", "time/space"))
    for lab, (s, t) in zip(labels, parts):
        print("   %-14s %12.4e %12.4e %12.2f" % (lab, s, t, t / s))
    s_fine, t_fine = parts[-1]
    if t_fine > s_fine:
        print("   the time part is the larger one, so the headline error mostly")
        print("   measures Crank-Nicolson, not the stencil.")
    else:
        print("   the space part is the larger one, so the headline error mostly")
        print("   measures the stencil.")
    print()
    return hs, res, sl


# %% D. exact error budget (homogeneous / constant boundary data only)


def study_budget(Ns=(8, 16, 32, 64, 128), mu=0.25):
    print("=" * 78)
    print("D. EXACT ERROR BUDGET   (9-point; max-norm of each signed part)")
    print("=" * 78)
    if BOUNDARY == "analytic":
        print("   skipped: with analytic boundary data the modes are not grid")
        print("   eigenvectors, so there is no closed-form split.\n")
        return None
    head = "%-18s %11s %11s %11s %11s %11s" % (
        "case", "total", "space", "time", "roundoff", "floor est.")
    print(head)
    print("-" * len(head))
    rows = []
    cases = ([("A 1/%d (%d)" % (N, _space_steps(H.Grid(N, N), mu)), N,
               _space_steps(H.Grid(N, N), mu)) for N in Ns]
             + [("C 1/%d (%d)" % (N, 2 ** k), N, 2 ** k) for k, N in enumerate(Ns)])
    for lab, N, ns in cases:
        grid = H.Grid(N, N)
        U = run(grid, ns)
        b = {k: float(np.abs(v).max()) for k, v in error_budget(grid, ns, U=U).items()}
        fl = roundoff_floor(grid, ns, H.LAPLACIAN_9PT, U)
        rows.append((lab, b, fl))
        print("%-18s %11.3e %11.3e %11.3e %11.3e %11.1e"
              % (lab, b["total"], b["space"], b["time"], b["roundoff"], fl))
    gap = min(np.log10(b["total"] / max(b["roundoff"], EPS)) for _, b, _ in rows)
    under = all(b["roundoff"] <= fl for _, b, fl in rows)
    print("\n   roundoff sits at least %.0f orders of magnitude below the total" % gap)
    print("   error on every grid used; the floor estimate")
    print("   eps*(1 + nsteps*(kappa(A) + ||B||))*max|U| %s the measured roundoff.\n"
          % ("is above" if under else "UNDERESTIMATES"))
    return rows


# %% plot


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
    dts, tres, tsl, space_err = time
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
    ax.axhline(space_err, color=GUIDE, lw=1.4, ls=(0, (2, 3)))
    ax.annotate("spatial error |Uref - u|", (dts[0], space_err),
                color=INK2, fontsize=9, xytext=(-4, -15), textcoords="offset points",
                ha="right")
    guide(ax, dts, tres["vs reference"], 2, where=0.25)
    style(ax, "time step  dt", "B.  Temporal order",
          "h = 1/128 fixed; the dip is space/time cancellation")
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
             "Slopes fitted over the asymptotic range only."
             % {"homogeneous": "homogeneous Dirichlet u = 0 on all of the boundary",
                "constant": "constant Dirichlet u = %g on all of the boundary" % CONSTANT,
                "analytic": "nonzero time-dependent Dirichlet data"}[BOUNDARY],
             ha="center", color=INK2, fontsize=9.5)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)
    print("wrote %s\n" % path)


# %% main


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
        print("          exact solution = %g + (the homogeneous solution);"
              % CONSTANT)
        print("          a roundoff/consistency check, not a new accuracy test\n")
    else:
        print("boundary: analytic Dirichlet, nonzero and time dependent\n")
    space = study_space()
    time_ = study_time()
    combined = study_combined()
    study_budget()
    make_plot(space, time_, combined)
    print("=" * 78)
    print("SLOPES (the big-O exponents, fitted over the asymptotic range)")
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
