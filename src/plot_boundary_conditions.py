"""Picture of the three Dirichlet boundary conditions used in error_analysis.

For each setting ("homogeneous", "constant", "analytic") the figure shows

  * left:   the exact solution u(x, y, 0) on the unit square, with the
            boundary drawn as a ring coloured by its value g, and
  * right:  g unrolled along the perimeter (bottom -> right -> top -> left)
            at several times up to T_FINAL.

    python src/plot_boundary_conditions.py   ->  reports/boundary_conditions.png
"""

from __future__ import annotations

import numpy as np

import error_analysis as E

MODES = ("homogeneous", "constant", "analytic")
TIMES = (0.0, 0.002, 0.005, 0.01, E.T_FINAL)


def perimeter(n=400):
    """Points (x, y) along the boundary, counter-clockwise from (0, 0), and the
    arc length s in [0, 4]."""
    s = np.linspace(0.0, 4.0, 4 * n + 1)
    side, r = np.minimum(s.astype(int), 3), s - np.minimum(s.astype(int), 3)
    x = np.select([side == 0, side == 1, side == 2, side == 3], [r, 1.0, 1.0 - r, 0.0])
    y = np.select([side == 0, side == 1, side == 2, side == 3], [0.0, r, 1.0, 1.0 - r])
    return s, x, y


def g_on(x, y, t):
    """The boundary data the solver sees for the current mode."""
    g = E.bc()
    return np.zeros_like(x) if g is None else np.asarray(g(x, y, t), dtype=float)


def main(path="reports/boundary_conditions.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
    cmap = plt.get_cmap("RdBu_r")
    tcols = plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(TIMES)))

    fig, axes = plt.subplots(len(MODES), 2, figsize=(12.5, 11.5), dpi=150,
                             facecolor=SURF,
                             gridspec_kw={"width_ratios": [1, 1.9]})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.875, bottom=0.06,
                        hspace=0.42, wspace=0.18)

    xs = np.linspace(0.0, 1.0, 201)
    X, Y = np.meshgrid(xs, xs)
    s, bx, by = perimeter()

    titles = {
        "homogeneous": "homogeneous   u = 0 on the boundary  (default)",
        "constant": "constant   u = %g on the boundary" % E.CONSTANT,
        "analytic": "analytic   u = g(x, y, t), nonzero and time dependent",
    }

    for row, mode in enumerate(MODES):
        E.set_boundary(mode)
        U0 = E.exact(X, Y, 0.0)
        centre = E.CONSTANT if mode == "constant" else 0.0
        vmax = max(np.abs(U0 - centre).max(), 1e-12)
        norm = plt.Normalize(centre - vmax, centre + vmax)

        # --- left: initial field with the boundary ring ---------------------
        ax = axes[row, 0]
        im = ax.pcolormesh(X, Y, U0, cmap=cmap, norm=norm, shading="auto")
        g0 = g_on(bx, by, 0.0)
        pts = np.column_stack([0.5 + 1.06 * (bx - 0.5), 0.5 + 1.06 * (by - 0.5)]).reshape(-1, 1, 2)
        ring = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1),
                              cmap=cmap, norm=norm, linewidths=7)
        ring.set_array(0.5 * (g0[:-1] + g0[1:]))
        ax.add_collection(ring)
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=INK, lw=0.6)
        for lab, (px, py) in (("bottom", (0.5, -0.11)), ("right", (1.11, 0.5)),
                              ("top", (0.5, 1.09)), ("left", (-0.11, 0.5))):
            ax.text(px, py, lab, ha="center", va="center", color=INK2, fontsize=8,
                    rotation=90 if lab in ("left", "right") else 0)
        ax.set_xlim(-0.16, 1.16); ax.set_ylim(-0.16, 1.14)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("u(x, y, 0); outer ring = boundary value g", color=INK2, fontsize=9.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=8, colors=INK2)
        cb.outline.set_edgecolor("#d8d6cf")

        # --- right: g unrolled along the perimeter --------------------------
        ax = axes[row, 1]
        for t, col in zip(TIMES, tcols):
            ax.plot(s, g_on(bx, by, t), color=col, lw=1.8, label="t = %g" % t)
        for k in range(1, 4):
            ax.axvline(k, color="#d8d6cf", lw=0.9, zorder=0)
        for k, lab in enumerate(("bottom  y=0", "right  x=1", "top  y=1", "left  x=0")):
            ax.text(k + 0.5, 1.02, lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", color=INK2, fontsize=8.5)
        ax.set_xlim(0, 4)
        lo, hi = ax.get_ylim()
        if hi - lo < 0.2:
            ax.set_ylim(centre - 0.5, centre + 0.5)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(["(0,0)", "(1,0)", "(1,1)", "(0,1)", "(0,0)"])
        ax.set_xlabel("position along the boundary (counter-clockwise)",
                      color=INK2, fontsize=9)
        ax.set_ylabel("g", color=INK2, fontsize=10)
        ax.grid(True, color="#ecebe5", lw=0.7)
        ax.tick_params(colors=INK2, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color("#d8d6cf")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=5,
                  loc="upper center", bbox_to_anchor=(0.5, -0.17))

        fig.text(0.06, axes[row, 0].get_position().y1 + 0.035,
                 "%d.  %s" % (row + 1, titles[mode]), color=INK, fontsize=12.5)

    fig.text(0.5, 0.972, "Boundary conditions used in the error analysis",
             ha="center", color=INK, fontsize=15)
    fig.text(0.5, 0.95,
             "Same four modes in every case; wavenumbers are multiples of pi.  "
             "Zero phases make every mode vanish on the boundary; "
             "the analytic case turns the phases on.",
             ha="center", color=INK2, fontsize=9)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)
    E.set_boundary("homogeneous")
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
