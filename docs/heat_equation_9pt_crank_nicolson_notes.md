# 2D Heat Equation: Integral Form, Off-Axis 9-Point Stencil, and Crank–Nicolson

Working notes from the DLA FDTD project. They cover the derivation from the divergence theorem through a verified 9-point (axial + diagonal) finite-volume stencil, Crank–Nicolson time stepping, von Neumann stability, verification code, and an implementation prompt.

---

## 1. Problem and notation

$$u_t = \alpha\,\nabla^2 u = \alpha\,(u_{xx} + u_{yy}) = \nabla\cdot(\alpha\nabla u), \qquad u = u(x,y,t),\ \alpha \in \mathbb{R}$$

| Symbol | Meaning |
|---|---|
| $A$, $\partial A$ | control area (region in the $xy$-plane) and its boundary curve |
| $\hat{\mathbf n} = (n_x, n_y)$ | **outward unit** normal on $\partial A$ |
| $s$, $ds$ | arc length along $\partial A$ (always positive) |
| $h$ | grid spacing, $h = \Delta x = \Delta y$ (square grid) |
| $u^n_{i,j}$ | $u(x_i, y_j, t^n)$: subscripts are **space**, the superscript is **time** |
| $x_{i\pm 1/2}$ | $x_i \pm h/2$: a **half grid spacing in space** (not a half time step) |
| $\Delta t$ | time step |
| $\mu = \alpha\Delta t/h^2$ | mesh ratio |

(In the Fourier analysis, grid indices are written $p, q$ so that $\mathrm{i} = \sqrt{-1}$ is free.)

---

## 2. Integral form via the divergence theorem

### 2.1 Option 1: 2D control area (used throughout)

The 2D divergence theorem (flux form of Green's theorem) is

$$\iint_A \nabla\cdot\mathbf F\,dA = \oint_{\partial A} \mathbf F\cdot\hat{\mathbf n}\,ds$$

(The 3D statement $\iiint_V \nabla\cdot\mathbf F\,dV = \oiint \mathbf F\cdot\hat{\mathbf n}\,dS$ is a different theorem; don't mix the two up.)

Integrate the PDE over a fixed region $A$ and take $\mathbf F = \alpha\nabla u$:

$$\frac{d}{dt}\iint_A u\,dA = \iint_A u_t\,dA = \alpha\oint_{\partial A}\frac{\partial u}{\partial n}\,ds$$

- Moving $d/dt$ inside the integral is justified because **$A$ does not depend on $t$** and $u_t$ is continuous (Leibniz rule), not by linearity alone.
- Physically, the rate of change of heat stored in $A$ equals the diffusive flux through its boundary.

An equivalent form uses $\hat{\mathbf n}\,ds = (dy, -dx)$ when $\partial A$ is traversed counterclockwise:

$$\iint_A u_t\,dA = \alpha\oint_{\partial A}\big(u_x\,dy - u_y\,dx\big)$$

Traversal direction matters **only** in this $dy,dx$ form. In the $\partial u/\partial n\,ds$ form all the sign information is carried by $\hat{\mathbf n}$.

### 2.2 Option 2: space-time volume (true 3D divergence theorem in $x,y,t$)

Take $\mathbf F = (-\alpha u_x,\,-\alpha u_y,\,u)$. Then $\nabla_{x,y,t}\cdot\mathbf F = u_t - \alpha\nabla^2u = 0$.

The volume is the prism $V = A\times[t^n, t^{n+1}]$, and $\oiint_{\partial V}\mathbf F\cdot\mathbf N\,dS = 0$ gives

$$\iint_A\big[u(t^{n+1}) - u(t^n)\big]dA = \alpha\int_{t^n}^{t^{n+1}}\oint_{\partial A}\frac{\partial u}{\partial n}\,ds\,dt$$

Its faces contribute as follows:

- **Top face** ($\mathbf N = +\hat t$): $+\iint u^{n+1}$
- **Bottom face** ($\mathbf N = -\hat t$): $-\iint u^n$
- **Side walls**: the time-integrated boundary flux

This is Option 1 integrated over one time step. The choice of **time quadrature** for the right-hand side is what fixes the time-stepping scheme (Section 8).

---

## 3. The normal derivative

$$\frac{\partial u}{\partial n} \equiv \nabla u\cdot\hat{\mathbf n} = u_x n_x + u_y n_y$$

This is the directional derivative in the direction $\hat{\mathbf n}$. From the definition and the chain rule, with $g(\varepsilon) = u(x+\varepsilon d_x,\, y+\varepsilon d_y)$:

$$\frac{\partial u}{\partial \mathbf d} = g'(0) = u_x d_x + u_y d_y = \nabla u\cdot\mathbf d$$

$\hat{\mathbf n}$ **must be a unit vector**; otherwise the result is scaled by $|\mathbf n|$.

| Face | $\hat{\mathbf n}$ | $\partial u/\partial n$ |
|---|---|---|
| right | $(1,0)$ | $+u_x$ |
| left | $(-1,0)$ | $-u_x$ |
| top | $(0,1)$ | $+u_y$ |
| bottom | $(0,-1)$ | $-u_y$ |
| diamond NE face | $(1,1)/\sqrt2$ | $(u_x+u_y)/\sqrt2$ (diagonal derivative) |

---

## 4. Why the divergence theorem lowers the derivative order

$\nabla^2 u = \nabla\cdot(\nabla u)$ is two first-order operators. The divergence theorem trades the **outer** derivative for the area integral; the **inner** one (the gradient) remains, evaluated on the boundary:

$$\iint_A\nabla^2u\,dA = \oint_{\partial A}\nabla u\cdot\hat{\mathbf n}\,ds$$

In 1D this is the fundamental theorem of calculus: $\int_a^b u_{xx}\,dx = u_x(b) - u_x(a)$.

The second derivative is not lost. It reappears as a **difference of fluxes on opposite faces**:

$$\frac{1}{h}\left[\frac{u_{i+1}-u_i}{h} - \frac{u_i-u_{i-1}}{h}\right] = \frac{u_{i+1}-2u_i+u_{i-1}}{h^2}$$

Benefits of the integral form:

- $u$ needs only one derivative, so it handles kinks, interfaces and variable $\alpha$.
- It is **conservative** by construction: the flux out of one cell is exactly the flux into its neighbour.
- The stencil is determined by which faces you choose and how you approximate the flux across each one. That is the source of the freedom for off-axis schemes.

---

## 5. Rectangular control area: edge by edge

Rectangle $[x_0,x_1]\times[y_0,y_1]$ with $x_0<x_1$ and $y_0<y_1$:

| Edge | $\hat{\mathbf n}$ | $\partial u/\partial n$ | $ds$ | Contribution |
|---|---|---|---|---|
| right, $x=x_1$ | $(1,0)$ | $+u_x$ | $dy$ | $+\int_{y_0}^{y_1}u_x(x_1,y)\,dy$ |
| left, $x=x_0$ | $(-1,0)$ | $-u_x$ | $dy$ | $-\int_{y_0}^{y_1}u_x(x_0,y)\,dy$ |
| top, $y=y_1$ | $(0,1)$ | $+u_y$ | $dx$ | $+\int_{x_0}^{x_1}u_y(x,y_1)\,dx$ |
| bottom, $y=y_0$ | $(0,-1)$ | $-u_y$ | $dx$ | $-\int_{x_0}^{x_1}u_y(x,y_0)\,dx$ |

Summing the four edges:

$$\boxed{\oint_{\partial A}\alpha\frac{\partial u}{\partial n}ds = \alpha\left[\int_{y_0}^{y_1}\!\big(u_x(x_1,y)-u_x(x_0,y)\big)dy + \int_{x_0}^{x_1}\!\big(u_y(x,y_1)-u_y(x,y_0)\big)dx\right]}$$

**Rule:** (far wall − near wall), integrated from the low limit to the high limit. Each bracket equals $\iint u_{xx}\,dA$ or $\iint u_{yy}\,dA$ by the FTC, so the total is $\iint\nabla^2u\,dA$ exactly.

### Net flux is not zero in general

The net flux equals $\iint_A\nabla^2u\,dA$:

- It is zero only when $\nabla^2 u$ integrates to zero over $A$, e.g. linear or harmonic $u$, or steady state.
- **$u = y$:** $\nabla^2u = 0$, so the net flux is $0$. Heat enters one face and leaves the opposite one at the same rate.
- **$u = x^2$:** the net flux is $(2x_1-2x_0)\Delta y = 2\Delta x\Delta y = \iint 2\,dA$.
- **Sign:** positive net flux means the cell warms (a local "valley"); negative means it cools (a hot spot).

### Common mistakes (caught while reviewing the handwritten derivation)

1. **Double negation.** On the bottom and left edges, $\hat{\mathbf n}$ already supplies the minus sign. Don't add a second minus for traversal direction.
2. **Merging values from different walls.** $u_y$ on the top and on the bottom are different functions evaluated at different places, so they can't be combined into "$2\Delta x\,u_y$".
3. **Reversed limits.** With an origin at the upper-right corner and $x_1<x_0$, $y_1<y_0$ (standard axes), writing $\int_{x_0}^{x_1}$ flips the sign of the whole result. Either integrate from low to high, or keep those limits and swap the integrand order. The convention-independent correct line is:

$$\alpha\left(\int_{x_0}^{x_1}\big[u_y(x,y_1)-u_y(x,y_0)\big]dx + \int_{y_0}^{y_1}\big[u_x(x_1,y)-u_x(x_0,y)\big]\,dy\right)$$

   (Note the $dy$ in the second integral.)
4. **Sanity check:** a correct formula must give zero net flux for linear $u$ such as $u = y$, and $+2\Delta x\Delta y$ for $u = x^2$.
5. **Notation:** $u$ is a scalar (no vector arrow); avoid naming side lengths $x$ and $y$ (use $\Delta x$, $\Delta y$, or $h$).

---

## 6. Where the nodes go: cell-centered grid and half offsets

Node $(i,j)$ sits at the **cell center** $(x_i,y_j)$. The walls are at $x_{i\pm1/2}$ and $y_{j\pm1/2}$, half a grid spacing away. Mapping to the handwritten diagram's labels (right $=x_0$, left $=x_1$, top $=y_0$, bottom $=y_1$):

| Label | Grid notation |
|---|---|
| right wall $x_0$ | $x_{i+1/2}$ |
| left wall $x_1$ | $x_{i-1/2}$ |
| top wall $y_0$ | $y_{j+1/2}$ |
| bottom wall $y_1$ | $y_{j-1/2}$ |
| center | $(x_i,y_j)$, where $u_{i,j}$ lives |

### Why $\pm\tfrac12$ and not integer offsets?

The integral form holds for any cell size, so integer offsets are legal. They cost you, though.

**Walls at $x_{i\pm1}$ (cell width $2h$).** Each wall midpoint lands **on** an axial node, where only $u$ is stored, not $u_x$. You then have two bad options:

- **Centered difference across the node:** $(u_{i+2}-u_i)/(2h)$ gives $(u_{i+2}-2u_i+u_{i-2})/(4h^2)$. That has **4× the error** ($h^2/3$ vs $h^2/12$) and **odd–even decoupling**: a sawtooth $(-1)^i$ looks constant to the stencil and is never damped.
- **One-sided difference:** $(u_{i+1}-u_i)/h$ evaluated at $x_{i+1}$ is off-center and only first order.

The overlapping cells also lose exact conservation.

**Half offsets give all three properties at once:**

1. The cells tile the domain exactly, one per node.
2. Each wall is shared by two cells, so conservation is automatic.
3. Each wall midpoint lies halfway between two nodes, giving compact, centered, second-order fluxes.

---

## 7. Spatial discretization

Everything in this section is **spatial**: it approximates integrals and derivatives in $x,y$ at a fixed instant. The result is a semi-discrete ODE system (method of lines).

### 7.1 Midpoint rule → 5-point Laplacian

**Why the midpoint rule.** It is second-order with a single sample, $\int f\,ds = h f(\text{mid}) + \tfrac{h^3}{24}f''$. The wall midpoint is also exactly where the two-node difference $(u_{i+1,j}-u_{i,j})/h$ is centered, and therefore $O(h^2)$. Quadrature and difference are accurate at the same point and to the same order.

For each wall, the procedure is:

1. Midpoint rule: $\int_{\text{wall}} u_x\,dy \approx h\,u_x|_{i+1/2,j}$.
2. Centered difference: $u_x|_{i+1/2,j}\approx(u_{i+1,j}-u_{i,j})/h$.
3. Left side: $\iint_A u_t\,dA\approx h^2\,du_{ij}/dt$ (2D midpoint rule).

$$\frac{du_{ij}}{dt} = \frac{\alpha}{h^2}\big(u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{ij}\big)$$

The $-4$ counts walls: each of the 4 walls contributes $(\text{neighbour} - u_{ij})$.

### 7.2 Simpson's rule → compact 9-point Laplacian

**Simpson's rule** is the blend of midpoint and trapezoid that cancels their leading errors:

$$S = \tfrac23 M + \tfrac13 T = \tfrac h6\big[f(a)+4f(\text{mid})+f(b)\big]$$

The midpoint error is $+\tfrac{h^3}{24}f''$ and the trapezoid error is $-\tfrac{h^3}{12}f''$; the 2:1 weighting cancels $f''$. Simpson's rule is exact for cubics.

#### Corner flux by averaging

The corner $(x_{i+1/2},y_{j+1/2})$ lies on no row of nodes; it sits at the center of the $2\times2$ block $(i,j),(i+1,j),(i,j+1),(i+1,j+1)$.

1. Define the row differences $D_k = u_{i+1,k}-u_{i,k}$. $D_k/h$ is a centered estimate of $u_x$ at $(x_{i+1/2},y_k)$.
2. Rows $j$ and $j+1$ sit $h/2$ below and $h/2$ above the corner. Average them:

$$u_x\big|_{i+\frac12,j+\frac12}\approx\frac{D_j+D_{j+1}}{2h},\qquad u_x\big|_{i+\frac12,j-\frac12}\approx\frac{D_{j-1}+D_j}{2h}$$

3. The symmetric average is second-order: $\tfrac12[g(y-\tfrac h2)+g(y+\tfrac h2)] = g(y)+\tfrac{h^2}{8}g''+\cdots$
4. Equivalently, average each column in $y$ first and then take the centered difference; you get the same formula.

#### Right-wall stencil

$$\frac h6\cdot\frac1h\left[\frac{D_{j-1}+D_j}2+4D_j+\frac{D_j+D_{j+1}}2\right] = \frac16\Big[5D_j+\tfrac12D_{j+1}+\tfrac12D_{j-1}\Big]$$

The 5 is $4+\tfrac12+\tfrac12$, because $D_j$ is used at the midpoint and in both corner averages. $D_{j\pm1}$ bring in the diagonal nodes.

#### The other walls by rotation

Because the cell is square and the operator is isotropic, each wall is the right wall rotated about the center node. In offsets $(a,b) = (\Delta i,\Delta j)$:

| Wall | Rotation | Offset map |
|---|---|---|
| top | $\pi/2$ | $(a,b)\to(-b,a)$ |
| left | $\pi$ | $(a,b)\to(-a,-b)$ |
| bottom | $3\pi/2$ | $(a,b)\to(b,-a)$ |

For the left wall, the reflection $i+1\to i-1$ gives the same result as the $\pi$ rotation. Normal signs need no manual handling, because every flux is written as (outside node − inside node).

If $\Delta x\neq\Delta y$ or $\alpha$ is anisotropic, the rotation argument fails and each wall must be computed separately.

#### Per-wall grids (×1/6; rows $j+1, j, j-1$; columns $i-1, i, i+1$)

```
Right            Top              Left             Bottom
 0  -½   ½        ½   5   ½        ½  -½   0        0   0   0
 0  -5   5       -½  -5  -½        5  -5   0       -½  -5  -½
 0  -½   ½        0   0   0        ½  -½   0        ½   5   ½
```

Each wall's grid sums to zero, which is a quick check that the rotation was done correctly.

#### Written out

- **Right:** $\tfrac16[5(u_{i+1,j}-u_{ij})+\tfrac12(u_{i+1,j+1}-u_{i,j+1})+\tfrac12(u_{i+1,j-1}-u_{i,j-1})]$
- **Top:** $\tfrac16[5(u_{i,j+1}-u_{ij})+\tfrac12(u_{i+1,j+1}-u_{i+1,j})+\tfrac12(u_{i-1,j+1}-u_{i-1,j})]$
- **Left:** $\tfrac16[5(u_{i-1,j}-u_{ij})+\tfrac12(u_{i-1,j+1}-u_{i,j+1})+\tfrac12(u_{i-1,j-1}-u_{i,j-1})]$
- **Bottom:** $\tfrac16[5(u_{i,j-1}-u_{ij})+\tfrac12(u_{i+1,j-1}-u_{i+1,j})+\tfrac12(u_{i-1,j-1}-u_{i-1,j})]$

#### Sum over the four walls

- **Corner (diagonal) node:** $\tfrac12+\tfrac12 = 1$
- **Axial node:** $5-\tfrac12-\tfrac12 = 4$
- **Center:** $4\times(-5) = -20$

$$\boxed{L\{u\}_{i,j} = \frac1{6h^2}\Big[4\big(u_{i\pm1,j}+u_{i,j\pm1}\big)+u_{i\pm1,j\pm1}-20\,u_{ij}\Big]}
\qquad
\frac1{6h^2}\begin{bmatrix}1&4&1\\4&-20&4\\1&4&1\end{bmatrix}$$

Here $u_{i\pm1,j\pm1}$ is shorthand for the sum over all four sign combinations. The coefficients sum to zero ($16+4-20$), so a constant field produces no flux.

### 7.3 Properties of the 9-point stencil

- **It is an exact blend:** 9-point $= \tfrac23$(axial 5-point) $+\tfrac13$(diagonal 5-point). The midpoint part alone gives the 5-point stencil. The trapezoid part alone gives the rotated stencil $\tfrac1{2h^2}[u_{i\pm1,j\pm1}-4u_{ij}]$, because its axial terms cancel ($+\tfrac12-\tfrac14-\tfrac14$). This is the "Mehrstellen / Patra–Karttunen" isotropic stencil with diagonal weight $\gamma = 1/3$.
- **Truncation error is isotropic:**

$$L\{u\} = \nabla^2u+\frac{h^2}{12}\nabla^4u+O(h^4)$$

  compared with the 5-point stencil's $\tfrac{h^2}{12}(u_{xxxx}+u_{yyyy})$, which favours the axes. So diffusion spreads in near-circular contours.
- **It is still second order**, because the corner averages and the area integral are $O(h^2)$.
- **Fourth order** is reachable via the Mehrstellen correction: apply the same weighted averaging to the $u_t$ (identity) term instead of the midpoint rule.
- **Quadrature as a design lever:** the quadrature rule chosen along the walls controls how much off-axis information enters the stencil. Midpoint is the minimal choice.

### 7.4 Diamond (rotated) control cell

Take vertices at $(i\pm1,j)$ and $(i,j\pm1)$ and outward normals $(\pm1,\pm1)/\sqrt2$:

- Each wall carries a diagonal derivative $(\pm u_x\pm u_y)/\sqrt2$.
- The wall midpoints $(i\pm\tfrac12,j\pm\tfrac12)$ lie halfway between the center and the diagonal neighbours, so the diagonal differences are centered. These are the "half offsets along the diagonal."
- Parametrizing the NE wall gives $u_x\,dy-u_y\,dx = (u_x+u_y)\,d\tau$.
- Result: the rotated 5-point stencil $\tfrac1{2h^2}[u_{i\pm1,j\pm1}-4u_{ij}]$.

---
