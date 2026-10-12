"""Compact 9-point / Crank-Nicolson solver for the 2D heat equation.

Model problem
-------------
    u_t = alpha * (u_xx + u_yy)      on  (0, Lx) x (0, Ly),  t in (0, T]
    u   = g(x, y, t)                 on the boundary  (g == 0 by default)
    u(x, y, 0) = u0(x, y)

Grid
----
Uniform, with the *same* spacing h in both directions (dx = dy = h):

    x_p = p*h,  p = 0 .. Nx          y_q = q*h,  q = 0 .. Ny

The unknowns are the interior nodes only, ordered row by row,

    k = (p - 1) + (q - 1) * (Nx - 1),     p = 1..Nx-1,  q = 1..Ny-1,

so that ``U.reshape(Ny - 1, Nx - 1)`` is the field with y running down the
rows and x along the columns.  Boundary nodes (p = 0, Nx or q = 0, Ny) are
not unknowns: a neighbour sitting on the boundary contributes to the
boundary vector b(t), never to the matrix.

Spatial operator (9-point, finite volumes + Simpson's rule)
-----------------------------------------------------------
                            1      |  1    4   1 |
    L{u}_{p,q}  =  ---------------  |  4  -20   4 |
                         6 h^2      |  1    4   1 |

    L{u}_{p,q} = 1/(6 h^2) * [ 4*(u_{p+1,q} + u_{p-1,q} + u_{p,q+1} + u_{p,q-1})
                               +  (u_{p+1,q+1} + u_{p+1,q-1}
                                   + u_{p-1,q+1} + u_{p-1,q-1})
                               - 20 * u_{p,q} ]

Its truncation error is *isotropic*,

    L{u} = lap(u) + (h^2/12) * lap(lap(u)) + O(h^4),

and the O(h^4) remainder vanishes identically for polynomials of degree <= 5.
The standard 5-point Laplacian instead carries the anisotropic error
(h^2/12) * (u_xxxx + u_yyyy), which is why it visibly distorts a radially
symmetric blob while the 9-point stencil does not.

Fourier symbol of the 9-point operator, for the grid mode exp(i(p*theta + q*phi)):

    h^2 * lambda_hat = -4 * (s + c - (2/3) * s * c),
    s = sin^2(theta/2),  c = sin^2(phi/2),

whose most negative value is -16/3 at the checkerboard theta = phi = pi
(the 5-point symbol is -4*(s + c), minimum -8).

Time stepping (Crank-Nicolson)
------------------------------
    (U^{n+1} - U^n) / dt = (alpha/2) * (L U^{n+1} + L U^n) + boundary terms

which is solved in the rearranged form

    (I - (alpha*dt/2) L) U^{n+1} = (I + (alpha*dt/2) L) U^n
                                   + (alpha*dt/2) * (b^{n+1} + b^n).

A = I - (alpha*dt/2)*L is factored *once* with scipy.sparse.linalg.splu and
the factorisation is reused at every step.  Forward Euler,

    U^{n+1} = U^n + alpha*dt * (L U^n + b^n),

is provided as an option; it is stable only for mu = alpha*dt/h^2 <= 3/8
(9-point) or mu <= 1/4 (5-point), while Crank-Nicolson is unconditionally
stable.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "Stencil",
    "LAPLACIAN_9PT",
    "LAPLACIAN_5PT",
    "Grid",
    "build_laplacian",
    "build_laplacian_9pt",
    "build_laplacian_5pt",
    "build_laplacian_periodic",
    "boundary_vector",
    "make_boundary_fn",
    "crank_nicolson_solve",
    "forward_euler_solve",
    "solve_heat",
    "fourier_symbol",
    "symbol_9pt",
    "symbol_5pt",
    "poly_eval",
    "poly_deriv",
    "poly_add",
    "poly_laplacian",
]


# ---------------------------------------------------------------------------
# Stencils
# ---------------------------------------------------------------------------


class Stencil:
    """A compact difference stencil for the Laplacian.

    The coefficient attached to offset ``(dp, dq)`` is ``scale * w / h**2``.
    """

    def __init__(self, name, weights, scale):
        self.name = str(name)
        self.weights = dict(weights)
        self.scale = float(scale)

    @property
    def radius(self):
        return max(max(abs(dp), abs(dq)) for dp, dq in self.weights)

    def coefficients(self, h):
        """Offset -> actual matrix coefficient on a grid of spacing ``h``."""
        f = self.scale / (h * h)
        return {off: f * w for off, w in self.weights.items()}

    def apply(self, F, h):
        """Apply the stencil to a full node array ``F[q, p]``.

        ``F`` has shape ``(Ny + 1, Nx + 1)`` (all nodes, boundary included);
        the returned array has shape ``(Ny - 1, Nx - 1)`` and holds the
        operator evaluated at every interior node.
        """
        F = np.asarray(F, dtype=float)
        ny, nx = F.shape[0] - 1, F.shape[1] - 1
        out = np.zeros((ny - 1, nx - 1), dtype=float)
        for (dp, dq), coeff in self.coefficients(h).items():
            out += coeff * F[1 + dq:dq + ny, 1 + dp:dp + nx]
        return out

    def __repr__(self):
        return "Stencil(%r, %d points)" % (self.name, len(self.weights))


#: compact 9-point Laplacian, 1/(6 h^2) * [[1, 4, 1], [4, -20, 4], [1, 4, 1]]
LAPLACIAN_9PT = Stencil(
    "9-point",
    {
        (0, 0): -20.0,
        (1, 0): 4.0, (-1, 0): 4.0, (0, 1): 4.0, (0, -1): 4.0,
        (1, 1): 1.0, (1, -1): 1.0, (-1, 1): 1.0, (-1, -1): 1.0,
    },
    1.0 / 6.0,
)

#: standard 5-point Laplacian, 1/h^2 * [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
LAPLACIAN_5PT = Stencil(
    "5-point",
    {(0, 0): -4.0, (1, 0): 1.0, (-1, 0): 1.0, (0, 1): 1.0, (0, -1): 1.0},
    1.0,
)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


class Grid:
    """Uniform square grid on ``[0, Lx] x [0, Ly]`` with dx = dy = h."""

    def __init__(self, Nx, Ny, Lx=1.0, Ly=1.0):
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        hx = self.Lx / self.Nx
        hy = self.Ly / self.Ny
        if abs(hx - hy) > 1e-12 * max(hx, hy):
            raise ValueError("grid must be square: dx=%g != dy=%g" % (hx, hy))
        self.h = hx

    @classmethod
    def from_h(cls, h, Lx=1.0, Ly=1.0):
        """Grid with spacing (approximately) ``h``; Lx/h and Ly/h must be integers."""
        Nx = int(round(Lx / h))
        Ny = int(round(Ly / h))
        if abs(Nx * h - Lx) > 1e-10 or abs(Ny * h - Ly) > 1e-10:
            raise ValueError("h=%g does not divide the domain" % h)
        return cls(Nx, Ny, Lx, Ly)

    # -- node coordinates ---------------------------------------------------
    @property
    def x(self):
        """All x node coordinates, shape (Nx + 1,)."""
        return self.h * np.arange(self.Nx + 1)

    @property
    def y(self):
        """All y node coordinates, shape (Ny + 1,)."""
        return self.h * np.arange(self.Ny + 1)

    @property
    def n_interior(self):
        return (self.Nx - 1) * (self.Ny - 1)

    @property
    def interior_shape(self):
        """Shape of the unknown field as a 2-D array: (Ny - 1, Nx - 1)."""
        return (self.Ny - 1, self.Nx - 1)

    def nodes(self):
        """Meshgrid ``(X, Y)`` of *all* nodes, shape (Ny + 1, Nx + 1)."""
        return np.meshgrid(self.x, self.y, indexing="xy")

    def interior_nodes(self):
        """Meshgrid ``(X, Y)`` of the interior nodes, shape (Ny - 1, Nx - 1)."""
        return np.meshgrid(self.x[1:-1], self.y[1:-1], indexing="xy")

    def interior_index(self):
        """``(P, Q, K)`` arrays of shape (Ny - 1, Nx - 1).

        ``K = (P - 1) + (Q - 1) * (Nx - 1)`` is the unknown number.
        """
        P, Q = np.meshgrid(
            np.arange(1, self.Nx), np.arange(1, self.Ny), indexing="xy"
        )
        K = (P - 1) + (Q - 1) * (self.Nx - 1)
        return P, Q, K

    def ravel(self, F):
        """Interior 2-D field -> unknown vector U."""
        return np.asarray(F, dtype=float).reshape(self.n_interior)

    def unravel(self, U):
        """Unknown vector U -> interior 2-D field, shape (Ny - 1, Nx - 1)."""
        return np.asarray(U, dtype=float).reshape(self.interior_shape)

    def full_field(self, U, g=None, t=0.0):
        """Assemble the full node array (shape (Ny+1, Nx+1)) from U plus the
        boundary data ``g(x, y, t)`` (zero when ``g`` is None)."""
        F = np.zeros((self.Ny + 1, self.Nx + 1), dtype=float)
        F[1:-1, 1:-1] = self.unravel(U)
        if g is not None:
            X, Y = self.nodes()
            G = np.asarray(g(X, Y, t), dtype=float)
            F[0, :] = G[0, :]
            F[-1, :] = G[-1, :]
            F[:, 0] = G[:, 0]
            F[:, -1] = G[:, -1]
        return F

    def sample(self, f, *args):
        """Evaluate ``f(X, Y, *args)`` at the interior nodes and ravel it."""
        X, Y = self.interior_nodes()
        return self.ravel(np.broadcast_to(np.asarray(f(X, Y, *args), dtype=float), X.shape))

    def __repr__(self):
        return "Grid(Nx=%d, Ny=%d, h=%g)" % (self.Nx, self.Ny, self.h)


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------


def build_laplacian(grid, stencil=LAPLACIAN_9PT, format="csr"):
    """Sparse matrix of ``stencil`` acting on the interior unknowns.

    Dirichlet rows are *not* modified: a neighbour that falls on the boundary
    simply drops out of the matrix, and its contribution is accounted for by
    :func:`boundary_vector`.  Every row therefore has at most 9 (resp. 5)
    nonzeros.
    """
    P, Q, K = grid.interior_index()
    rows, cols, vals = [], [], []
    for (dp, dq), coeff in stencil.coefficients(grid.h).items():
        PP, QQ = P + dp, Q + dq
        inside = (PP >= 1) & (PP <= grid.Nx - 1) & (QQ >= 1) & (QQ <= grid.Ny - 1)
        KK = (PP - 1) + (QQ - 1) * (grid.Nx - 1)
        rows.append(K[inside])
        cols.append(KK[inside])
        vals.append(np.full(int(inside.sum()), coeff))
    n = grid.n_interior
    L = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    return L.asformat(format)


def build_laplacian_9pt(grid, format="csr"):
    """Compact 9-point Laplacian, 1/(6h^2)*[[1,4,1],[4,-20,4],[1,4,1]]."""
    return build_laplacian(grid, LAPLACIAN_9PT, format)


def build_laplacian_5pt(grid, format="csr"):
    """Standard 5-point Laplacian, 1/h^2*[[0,1,0],[1,-4,1],[0,1,0]]."""
    return build_laplacian(grid, LAPLACIAN_5PT, format)


def build_laplacian_periodic(Nx, Ny, h, stencil=LAPLACIAN_9PT, format="csr"):
    """Same stencil on a doubly periodic Nx-by-Ny grid.

    Every node is an unknown, numbered ``k = p + q*Nx`` with p = 0..Nx-1 and
    q = 0..Ny-1.  Used for the Fourier-symbol and checkerboard checks, where
    the grid modes exp(i(p*theta + q*phi)) are exact eigenvectors.
    """
    P, Q = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="xy")
    K = P + Q * Nx
    rows, cols, vals = [], [], []
    for (dp, dq), coeff in stencil.coefficients(h).items():
        KK = (P + dp) % Nx + ((Q + dq) % Ny) * Nx
        rows.append(K.ravel())
        cols.append(KK.ravel())
        vals.append(np.full(Nx * Ny, coeff))
    L = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(Nx * Ny, Nx * Ny),
    )
    return L.asformat(format)


def boundary_vector(grid, g, t=0.0, stencil=LAPLACIAN_9PT):
    """Dirichlet contribution ``b(t)`` of the stencil's boundary neighbours.

    For every interior node, each neighbour that lies on the boundary adds
    ``coeff * g(x_neighbour, y_neighbour, t)`` to that node's entry, so that

        (L U + b)_k  ==  stencil applied to the full field.

    ``g`` may be None, in which case b = 0 (homogeneous Dirichlet).
    """
    b = np.zeros(grid.n_interior)
    if g is None:
        return b
    P, Q, K = grid.interior_index()
    x, y = grid.x, grid.y
    for (dp, dq), coeff in stencil.coefficients(grid.h).items():
        if (dp, dq) == (0, 0):
            continue
        PP, QQ = P + dp, Q + dq
        on_boundary = (PP == 0) | (PP == grid.Nx) | (QQ == 0) | (QQ == grid.Ny)
        if not on_boundary.any():
            continue
        pb, qb = PP[on_boundary], QQ[on_boundary]
        b[K[on_boundary]] += coeff * np.asarray(
            g(x[pb], y[qb], t), dtype=float
        )
    return b


def make_boundary_fn(grid, g, stencil=LAPLACIAN_9PT):
    """Return ``b(t)`` as a callable, or None for homogeneous Dirichlet data."""
    if g is None:
        return None
    return lambda t: boundary_vector(grid, g, t, stencil)


# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------


def _bvec(bfun, t, n):
    return np.zeros(n) if bfun is None else np.asarray(bfun(t), dtype=float)


def crank_nicolson_solve(L, u0, alpha, dt, nsteps, bfun=None, t0=0.0,
                         callback=None):
    """Advance ``nsteps`` Crank-Nicolson steps of size ``dt``.

    The update actually solved is

        (I - (alpha*dt/2) L) U^{n+1} = (I + (alpha*dt/2) L) U^n
                                       + (alpha*dt/2) * (b^{n+1} + b^n),

    with ``A = I - (alpha*dt/2) L`` factored once by ``splu`` and reused for
    every step.  ``bfun(t)`` supplies the Dirichlet boundary vector (None for
    homogeneous data).  ``callback(n, t, U)`` is called after each step.

    Unconditionally stable: L is symmetric negative semi-definite, so the
    amplification factor (1 + z/2)/(1 - z/2) has modulus <= 1 for every
    eigenvalue z = alpha*dt*lambda <= 0.
    """
    U = np.array(u0, dtype=float, copy=True)
    n = U.size
    half = 0.5 * alpha * dt
    I = sp.identity(n, format="csc")
    A = (I - half * L).tocsc()
    B = (I + half * L).tocsr()
    lu = spla.splu(A)
    t0 = float(t0)
    t = t0
    b_now = _bvec(bfun, t, n)
    for step in range(nsteps):
        b_next = _bvec(bfun, t0 + (step + 1) * dt, n)
        rhs = B @ U + half * (b_now + b_next)
        U = lu.solve(rhs)
        t = t0 + (step + 1) * dt     # not t += dt: no accumulated rounding
        b_now = b_next
        if callback is not None:
            callback(step + 1, t, U)
    return U


def forward_euler_solve(L, u0, alpha, dt, nsteps, bfun=None, t0=0.0,
                        callback=None):
    """Advance ``nsteps`` explicit Euler steps,

        U^{n+1} = U^n + alpha*dt * (L U^n + b^n).

    Stable only for mu = alpha*dt/h^2 <= 3/8 (9-point) or <= 1/4 (5-point).
    """
    U = np.array(u0, dtype=float, copy=True)
    n = U.size
    t0 = float(t0)
    t = t0
    for step in range(nsteps):
        U = U + alpha * dt * (L @ U + _bvec(bfun, t, n))
        t = t0 + (step + 1) * dt
        if callback is not None:
            callback(step + 1, t, U)
    return U


def solve_heat(grid, u0, T, alpha=1.0, dt=None, nsteps=None,
               stencil=LAPLACIAN_9PT, g=None, method="cn", callback=None):
    """Convenience driver: integrate from t = 0 to ``T`` and return ``(U, info)``.

    ``u0`` is either a callable ``u0(x, y)`` or a vector of interior values.
    Exactly one of ``dt`` / ``nsteps`` is required; ``dt`` is adjusted to
    ``T / nsteps`` so the last step lands exactly on ``T``.
    """
    if (dt is None) == (nsteps is None):
        raise ValueError("give exactly one of dt, nsteps")
    if nsteps is None:
        nsteps = max(1, int(round(T / dt)))
    dt = T / nsteps
    if callable(u0):
        U0 = grid.sample(lambda X, Y: u0(X, Y))
    else:
        U0 = np.asarray(u0, dtype=float)
    L = build_laplacian(grid, stencil)
    bfun = make_boundary_fn(grid, g, stencil)
    solver = {"cn": crank_nicolson_solve, "fe": forward_euler_solve}[method]
    U = solver(L, U0, alpha, dt, nsteps, bfun=bfun, callback=callback)
    info = {"dt": dt, "nsteps": nsteps, "mu": alpha * dt / grid.h ** 2,
            "stencil": stencil.name, "method": method}
    return U, info


# ---------------------------------------------------------------------------
# Fourier symbols
# ---------------------------------------------------------------------------


def fourier_symbol(stencil, theta, phi):
    """``h^2 * lambda_hat`` for the grid mode ``exp(i(p*theta + q*phi))``.

    Computed straight from the stencil weights; the result is real because
    every stencil here is symmetric under (dp, dq) -> (-dp, -dq).
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    total = np.zeros(np.broadcast(theta, phi).shape, dtype=complex)
    for (dp, dq), w in stencil.weights.items():
        total = total + w * np.exp(1j * (dp * theta + dq * phi))
    return (stencil.scale * total).real


def symbol_9pt(theta, phi):
    """Closed form for the 9-point stencil: -4*(s + c - (2/3) s c)."""
    s = np.sin(np.asarray(theta, dtype=float) / 2.0) ** 2
    c = np.sin(np.asarray(phi, dtype=float) / 2.0) ** 2
    return -4.0 * (s + c - (2.0 / 3.0) * s * c)


def symbol_5pt(theta, phi):
    """Closed form for the 5-point stencil: -4*(s + c)."""
    s = np.sin(np.asarray(theta, dtype=float) / 2.0) ** 2
    c = np.sin(np.asarray(phi, dtype=float) / 2.0) ** 2
    return -4.0 * (s + c)


# ---------------------------------------------------------------------------
# Exact 2-D polynomial helpers (used by the stencil verification)
# ---------------------------------------------------------------------------


def poly_eval(C, X, Y):
    """Evaluate sum_ij C[i, j] * x**i * y**j."""
    C = np.asarray(C, dtype=float)
    out = np.zeros(np.broadcast(X, Y).shape, dtype=float)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] != 0.0:
                out = out + C[i, j] * X ** i * Y ** j
    return out


def poly_deriv(C, axis, order=1):
    """Differentiate the coefficient array ``order`` times along x (0) or y (1)."""
    C = np.asarray(C, dtype=float)
    for _ in range(order):
        if axis == 0:
            if C.shape[0] <= 1:
                return np.zeros((1, C.shape[1]))
            C = C[1:, :] * np.arange(1, C.shape[0])[:, None]
        else:
            if C.shape[1] <= 1:
                return np.zeros((C.shape[0], 1))
            C = C[:, 1:] * np.arange(1, C.shape[1])[None, :]
    return C


def poly_add(A, B):
    """Add two coefficient arrays of possibly different shapes."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = max(A.shape[0], B.shape[0])
    m = max(A.shape[1], B.shape[1])
    out = np.zeros((n, m))
    out[:A.shape[0], :A.shape[1]] += A
    out[:B.shape[0], :B.shape[1]] += B
    return out


def poly_laplacian(C):
    """Coefficient array of ``u_xx + u_yy``."""
    return poly_add(poly_deriv(C, 0, 2), poly_deriv(C, 1, 2))


# ---------------------------------------------------------------------------
# Running this file directly does a quick self-check; run_demo.py is the full
# verification summary and test/test_heat9.py is the pytest suite.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print(__doc__.strip().splitlines()[0])
    print()
    for _st in (LAPLACIAN_9PT, LAPLACIAN_5PT):
        print("%-9s stencil, h^2*lambda at the checkerboard = %+.6f"
              % (_st.name, fourier_symbol(_st, np.pi, np.pi)))

    _grid = Grid(64, 64)
    _exact = lambda x, y, t: (np.exp(-2.0 * np.pi ** 2 * t)
                              * np.sin(np.pi * x) * np.sin(np.pi * y))
    _U, _info = solve_heat(_grid, lambda x, y: _exact(x, y, 0.0), 0.1,
                           alpha=1.0, nsteps=8, stencil=LAPLACIAN_9PT)
    _X, _Y = _grid.interior_nodes()
    _err = np.abs(_grid.unravel(_U) - _exact(_X, _Y, 0.1)).max()
    print("\nCrank-Nicolson on %r, %d steps of dt=%g (mu = %.3f)"
          % (_grid, _info["nsteps"], _info["dt"], _info["mu"]))
    print("max-norm error vs exp(-2 pi^2 t) sin(pi x) sin(pi y) at T=0.1: %.4e" % _err)
    print("\nFor the full verification summary run:  python3 src/run_demo.py")
    print("For the test suite run:                 python3 -m pytest")
