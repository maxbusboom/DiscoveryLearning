# Finite Difference Notes: S24, Heat Equation, and Off-Axis Stencils

Working notes for deriving an off-axis finite difference (FDTD) scheme. They build from a 1D fourth-order staggered (S24) stencil, through time integration and Fourier analysis, to isotropic 2D off-axis Laplacians on 3×3 and 5×5 grids.

**Notation**

- $h$: grid spacing ($\Delta x = \Delta y = h$ unless stated)
- $\Delta t$: time step
- $\kappa$: diffusivity. Not $k$, which is reserved for the wavenumber.
- $\theta = kh$: phase change per cell. In 2D, $\theta_x = k_x h$ and $\theta_y = k_y h$.
- $r = \kappa\Delta t / h^2$: diffusion number
- $u_{i\pm m}$: shorthand for $u_{i+m} + u_{i-m}$ inside stencils

---

## 1. General derivation recipe

1. **Discretize.** Choose the grid and decide where each variable lives (collocated or staggered).
2. **Replace derivatives** with difference quotients built from Taylor expansions.
3. **Consistency.** Taylor-expand the scheme back out and read off the truncation error.
4. **Stability (von Neumann).** Substitute $u_i^n = G^n e^{\mathrm{i}kx_i}$ and require $|G| \le 1$ for all $\theta$.
5. **Dispersion.** Set $G = e^{-\mathrm{i}\omega\Delta t}$, get the discrete $\omega(k)$, and compare it with the exact one.

Dispersion matters most for off-axis work: the point of off-axis schemes is to fix how the error depends on propagation direction.

### Suggested warm-up sequence

1. 1D advection: upwind and Lax–Wendroff; CFL.
2. 1D heat equation: explicit and Crank–Nicolson.
3. 1D wave equation: leapfrog. At $c\Delta t = \Delta x$ ("magic time step") it is dispersion-free.
4. 1D Maxwell on a staggered Yee grid: the actual FDTD.
5. 2D wave / TM Maxwell: 2D CFL, anisotropic dispersion.
6. 2D advection at an angle: a first off-axis differencing exercise.
7. Rotated / 9-point Laplacian: the key off-axis exercise (Section 8).

---

## 2. The S24 scheme (FDTD(2,4))

S24 is second-order in time (Yee leapfrog) and fourth-order in space, on a staggered grid (Fang's scheme).

### First-derivative stencil

$$\partial_x f\big|_i \approx \frac{27\big(f_{i+\frac12} - f_{i-\frac12}\big) - \big(f_{i+\frac32} - f_{i-\frac32}\big)}{24h}$$

Equivalently, the weights are $\tfrac98$ on the inner pair and $-\tfrac1{24}$ on the outer pair, each over $h$.

### Deriving the coefficients

Taylor-expand about $x_i$:

- $f_{i+\frac12} - f_{i-\frac12} = h f' + \frac{h^3}{24} f''' + \frac{h^5}{1920} f^{(5)} + \dots$
- $f_{i+\frac32} - f_{i-\frac32} = 3h f' + \frac{27h^3}{24} f''' + \frac{243h^5}{1920} f^{(5)} + \dots$

Take $a\cdot(\text{first}) + b\cdot(\text{second})$ and require:

- $f'$ coefficient equal to 1: $a + 3b = 1$
- $f'''$ coefficient zero: $a + 27b = 0$

This gives $a = 9/8$ and $b = -1/24$.

### Truncation error

$$f'(x_i) = \frac{27(\cdot) - (\cdot)}{24h} + \frac{3}{640}h^4 f^{(5)}(\xi)$$

Equivalently, $+\left(\tfrac h2\right)^4 \tfrac{3}{40} f^{(5)}$. (The handwritten "$1/13.3$" is $3/40$.) The error is evaluated at $x_i$, or strictly at some $\xi$ near $x_i$, not at 0.

In operator form: $D = \partial_x - \tfrac{3}{640}h^4\partial_x^5 + O(h^6)$.

### Properties

**1D dispersion relation:**
$$\frac{1}{c\Delta t}\sin\frac{\omega\Delta t}{2} = \frac{1}{\Delta x}\left[\frac98\sin\frac{k\Delta x}{2} - \frac1{24}\sin\frac{3k\Delta x}{2}\right]$$

**Stability.** The coefficient magnitudes sum to $\tfrac98 + \tfrac1{24} = \tfrac76$, which tightens the CFL limits:

| | Yee (2,2) | S24 |
|---|---|---|
| 1D | $c\Delta t/\Delta x \le 1$ | $\le 6/7$ |
| 2D, square cells | $\le 1/\sqrt2$ | $\le 6/(7\sqrt2)$ |
| 3D, cubic cells | $\le 1/\sqrt3$ | $\le 6/(7\sqrt3)$ |

**Other notes:**

- **No magic time step.** Temporal and spatial errors no longer cancel along an axis.
- **Boundaries.** The stencil reaches $\pm\tfrac32$ cells, so it needs one-sided or reduced-order stencils near PEC, PML, and material interfaces.
- **S24 is not off-axis.** The half-integer nodes come from staggering, and the $\pm\tfrac32$ nodes make the stencil *extended* along the same axis.
- **M24 (Hadi & Piket-May)** adds weighted extended and off-axis terms to S24, tuned to cancel phase error.

---

## 3. 1D heat equation with the S24 spatial operator

$$u_t = \kappa\, u_{xx}$$

### Flux form

The heat equation is first-order in time, so S24's leapfrog time stepping does not apply (Section 5). The spatial operator can still be used by writing the equation as a staggered system:

$$u_t = -\partial_x q, \qquad q = -\kappa\,\partial_x u$$

$u$ lives at integer points and $q$ at half points. Applying the S24 first derivative twice gives the second-derivative operator. This form also handles variable $\kappa(x)$ and material interfaces.

### Error terms during the derivation

Don't carry them. Drop the error term, compose the stencils, and get the error from the operator form:

$$D^2 = \left(\partial_x - \tfrac{3}{640}h^4\partial_x^5\right)^2 = \partial_x^2 - \tfrac{3}{320}h^4\partial_x^6 + O(h^6)$$

You can confirm this afterwards by Taylor-expanding the final stencil directly.

### Composition

The four inner first derivatives, each over $24h$:

- At $i+\tfrac12$: $27(f_{i+1} - f_i) - (f_{i+2} - f_{i-1})$
- At $i-\tfrac12$: $27(f_i - f_{i-1}) - (f_{i+1} - f_{i-2})$
- At $i+\tfrac32$: $27(f_{i+2} - f_{i+1}) - (f_{i+3} - f_i)$
- At $i-\tfrac32$: $27(f_{i-1} - f_{i-2}) - (f_i - f_{i-3})$

The outer derivative combines these as $\big[27(\cdot_{+1/2} - \cdot_{-1/2}) - (\cdot_{+3/2} - \cdot_{-3/2})\big]/24h$.

Intermediate results:

- $(\cdot_{+1/2}) - (\cdot_{-1/2})$, numerator: $28 f_{i\pm1} - 54 f_i - f_{i\pm2}$
- $(\cdot_{+3/2}) - (\cdot_{-3/2})$, numerator: $27 f_{i\pm2} - 27 f_{i\pm1} - f_{i\pm3} + 2f_i$

### Result: 7-point fourth-order second derivative

$$\boxed{\;\partial_{xx} u\big|_i \approx \frac{(u_{i+3}+u_{i-3}) - 54(u_{i+2}+u_{i-2}) + 783(u_{i+1}+u_{i-1}) - 1460\,u_i}{576\,h^2}\;}$$

Leading error: $-\tfrac{3}{320}h^4 u^{(6)}$.

### Sanity checks

- **Coefficients sum to zero:** $2(1 - 54 + 783) = 1460$.
- **Symmetry:** the coefficient of $f_{i+m}$ equals that of $f_{i-m}$. This catches index and sign slips that the sum check misses.

Mistakes found in the hand derivation:

- An index error in the $i+\tfrac32$ term ($f_{i+1}$ instead of $f_i$)
- A sign not distributed through all four terms
- A missing outer $1/24h$

---

## 4. Shift operators

### Definition

$$(E f)_i = f_{i+1}, \qquad E^m f_i = f_{i+m}, \qquad E^{1/2} f_i = f_{i+\frac12}, \qquad E^0 = I$$

### Uses

**Stencils become polynomials in $E$:**

- Forward difference: $f_{i+1} - f_i = (E - I)f_i$
- Centered second difference: $(E - 2I + E^{-1})f_i$
- S24 first derivative: $\dfrac{1}{24h}\big[27(E^{1/2} - E^{-1/2}) - (E^{3/2} - E^{-3/2})\big]$

**Composition is multiplication.** Squaring the S24 polynomial (using $E^aE^b = E^{a+b}$) reproduces the 7-point stencil directly:

$$\frac{1}{576h^2}\big[27(E^{1/2} - E^{-1/2}) - (E^{3/2} - E^{-3/2})\big]^2$$

**Taylor link.** $E = e^{hD}$ with $D = d/dx$. Finite difference design is approximating $D = \tfrac1h\ln E$ by a polynomial in $E$.

**Multi-D.** Use $E_x$ and $E_y$. Off-axis nodes appear as mixed products like $E_xE_y$.

---

## 5. Time integration for the heat equation

### Centered time difference (Richardson): unconditionally unstable

$$\frac{u_i^{n+1} - u_i^{n-1}}{2\Delta t} = \kappa\, D^2 u_i^n$$

With the spatial symbol $-\sigma/h^2$ (Section 6) the amplification equation is

$$G - \frac1G = -2r\sigma \;\Longrightarrow\; G^2 + 2r\sigma G - 1 = 0$$

The root product is $-1$, so one root has $|G| > 1$ for every $r > 0$. Numerically the growth factor is already about 1.056 per step at $r = 0.01$.

Centered time differences suit reversible (wave) problems, but diffusion is irreversible.

### DuFort–Frankel: also fails with this stencil

DuFort–Frankel replaces the center term $u_i^n$ by $\tfrac12(u_i^{n+1} + u_i^{n-1})$. It needs the center weight to dominate the off-center weights, but here $1460 < 2(1 + 54 + 783) = 1676$. It was verified numerically to be unstable for all $r$.

### Schemes that work

| Method | Time order | Stability |
|---|---|---|
| Forward Euler | 1 | $r \le 18/49 \approx 0.367$ |
| RK2 (Heun) | 2 | $r \le 18/49$ |
| Crank–Nicolson | 2 | unconditional |

For comparison, the standard 3-point stencil allows $r \le 0.5$ with forward Euler.

**Order matching.** Explicit stability forces $\Delta t \propto h^2$. A second-order time error $O(\Delta t^2)$ is then $O(h^4)$, matching the spatial order. Forward Euler only gives $O(h^2)$ overall.

### Crank–Nicolson

Average the spatial operator between levels $n$ and $n+1$. This centers the scheme at $n+\tfrac12$ (trapezoidal rule):

$$\frac{u^{n+1} - u^n}{\Delta t} = \frac{\kappa}{2}\Big(D^2u^{n+1} + D^2u^n\Big)$$

$$\Big(I - \tfrac{\kappa\Delta t}{2}D^2\Big)u^{n+1} = \Big(I + \tfrac{\kappa\Delta t}{2}D^2\Big)u^n$$

At node $i$, with $\beta = \dfrac{\kappa\Delta t}{1152\,h^2}$:

$$-\beta\,u_{i\pm3}^{n+1} + 54\beta\,u_{i\pm2}^{n+1} - 783\beta\,u_{i\pm1}^{n+1} + (1+1460\beta)\,u_i^{n+1}$$
$$= \beta\,u_{i\pm3}^{n} - 54\beta\,u_{i\pm2}^{n} + 783\beta\,u_{i\pm1}^{n} + (1-1460\beta)\,u_i^{n}$$

**Solving.** The left-side matrix is symmetric, has 7 diagonals, and is the same every step. Factor it once and reuse it (`scipy.linalg.solve_banded` or `scipy.sparse.linalg.splu`).

**Amplification factor:**
$$G = \frac{1 - \tfrac r2\sigma}{1 + \tfrac r2\sigma}, \qquad |G| \le 1 \;\; \forall r$$

**Caveat.** At large $r$, high-frequency modes have $G \to -1$. They decay slowly and alternate sign, which shows up as sawtooth noise with sharp initial data. To avoid it, keep $r$ moderate or take a few backward-Euler steps at the start (Rannacher startup).

**Boundaries.** The stencil reaches 3 points each way, so the first three rows at each end need ghost points or one-sided / lower-order stencils.

**Verification test.** On a periodic domain, start from $u_0 = \sin(2\pi x)$. The exact solution is $e^{-4\pi^2\kappa t}\sin(2\pi x)$, and the error should scale like $O(\Delta t^2 + h^4)$.

---

## 6. Fourier symbols and von Neumann analysis

### The idea

A pure mode $e^{\mathrm{i}kx}$ is an eigenfunction of every linear, shift-invariant operator. The multiplier it picks up is the operator's **symbol**. The exact $\partial_x^2$ has symbol $-k^2$.

### The core substitution

$$E^m \;\to\; e^{\mathrm{i}m\theta}$$

| Stencil piece | Symbol |
|---|---|
| $f_{i+m} + f_{i-m}$ | $2\cos m\theta$ |
| $f_{i+m} - f_{i-m}$ | $2\mathrm{i}\sin m\theta$ |
| $f_{i+\frac12}$ | $e^{\mathrm{i}\theta/2}$ |
| exact $\partial_x$ | $\mathrm{i}k$ |
| exact $\partial_x^2$ | $-k^2$ |
| 2D shift $f_{i+m,\,j+n}$ | $e^{\mathrm{i}(m\theta_x + n\theta_y)}$ |

### Time levels

$u^n \to G^n$:

- Two-level schemes give $G$ explicitly.
- Three-level schemes give a quadratic in $G$; check both roots.
- Systems (Maxwell $E$, $H$) give a matrix symbol; $G$ values are its eigenvalues.

### What to read off

- **Accuracy:** expand the symbol for small $\theta$ and compare it with the exact symbol.
- **Stability:** $|G(\theta)| \le 1$ for all $\theta \in [-\pi, \pi]$. The worst case is usually $\theta = \pi$.
- **Dispersion / dissipation:** set $G = e^{-\mathrm{i}\omega\Delta t}$. $|G| < 1$ is damping, and the phase of $G$ gives the phase velocity. In 2D, plot phase velocity against angle.

### Procedure

1. Write the scheme with explicit shifts.
2. Substitute $u_i^n = G^n e^{\mathrm{i}kx_i}$ and divide out the common factor.
3. Simplify with $\cos$/$\sin$ pairs.
4. Expand for accuracy.
5. Bound $|G|$ for stability.
6. Solve for $\omega(k)$.

### Limits

The analysis is exact only for linear, constant-coefficient problems on periodic or infinite grids.

- **Variable coefficients:** apply it "frozen" at local values, which is a guide rather than a proof.
- **Boundaries:** these can cause instabilities it won't catch. GKS theory is the rigorous tool; in practice, test numerically.

### Symbol of the 7-point stencil

$$\frac{1}{576h^2}\Big[2\cos3\theta - 108\cos2\theta + 1566\cos\theta - 1460\Big] = -\frac{\sigma(\theta)}{h^2}$$

$$\sigma(\theta) = \frac{1460 - 1566\cos\theta + 108\cos2\theta - 2\cos3\theta}{576} = 4\left[\tfrac98\sin\tfrac\theta2 - \tfrac1{24}\sin\tfrac{3\theta}2\right]^2 \ge 0$$

- **Accuracy:** $\sigma = \theta^2 - \tfrac{3}{320}\theta^6 + \dots$, versus the exact $\theta^2$.
- **Stability:** $\sigma_{\max} = 49/9$ at $\theta = \pi$. Forward Euler needs $r\sigma \le 2$, so $r \le 18/49$.
- **Comparison:** the 3-point stencil has $\sigma = 4\sin^2(\theta/2)$.

### Exercise

For 1D Yee Maxwell, treated as a $2\times2$ system, derive $\sin\tfrac{\omega\Delta t}{2} = \tfrac{c\Delta t}{\Delta x}\sin\tfrac\theta2$. For S24, replace $\sin\tfrac\theta2$ with $\tfrac98\sin\tfrac\theta2 - \tfrac1{24}\sin\tfrac{3\theta}2$.

---

## 7. What "off-axis" means

An off-axis stencil for $\partial_x$ at $(i,j)$ also uses nodes displaced in the **transverse** direction. A simple second-order staggered example:

$$\partial_x f\big|_{i,j} \approx \alpha\,\frac{f_{i+\frac12,j} - f_{i-\frac12,j}}{h_x} + \frac\beta2\left[\frac{f_{i+\frac12,j+1} - f_{i-\frac12,j+1}}{h_x} + \frac{f_{i+\frac12,j-1} - f_{i-\frac12,j-1}}{h_x}\right]$$

- **Consistency:** $\alpha + \beta = 1$.
- **What the off-axis bracket adds:** it equals the on-axis difference plus $\tfrac{h_y^2}{2}\partial_x\partial_y^2 f$.
- **Effect:** this cross term doesn't change the order of accuracy. It changes the **angular** dependence of the error.
- **Tuning:** choosing $\beta$ can cancel anisotropic dispersion, for example making the phase velocity exactly $c$ along the beam axis. This is the basis of accelerator low-dispersion schemes (Kärkkäinen / Gjonaj / Weiland type) and of the off-axis terms in M24.

---

## 8. 2D off-axis Laplacian on a 3×3 grid

### Where the distance metric enters

| Family | Nodes | Distance |
|---|---|---|
| Axial | $(i\pm1, j)$, $(i, j\pm1)$ | $h$ |
| Diagonal | $(i\pm1, j\pm1)$ | $\sqrt2\,h$ |

Along a diagonal direction $\xi = (x+y)/\sqrt2$ the spacing is $\sqrt2 h$, so the second difference has denominator $(\sqrt2h)^2 = 2h^2$. That is where the Euclidean metric belongs.

**Don't weight nodes by $1/r$ directly.** Doing so gives a stencil that isn't consistent with $\nabla^2$.

By rotation invariance, $\partial_\xi^2 + \partial_\eta^2 = \nabla^2$, so there are two Laplacians:

$$L_+ = \frac{U_{i+1,j} + U_{i-1,j} + U_{i,j+1} + U_{i,j-1} - 4U_{i,j}}{h^2}$$

$$L_\times = \frac{U_{i+1,j+1} + U_{i-1,j-1} + U_{i+1,j-1} + U_{i-1,j+1} - 4U_{i,j}}{2h^2}$$

### Taylor expansions

- $L_+ = \nabla^2u + \tfrac{h^2}{12}(u_{xxxx} + u_{yyyy}) + \dots$
- $L_\times = \nabla^2u + \tfrac{h^2}{12}(u_{xxxx} + 6u_{xxyy} + u_{yyyy}) + \dots$

### Blend $(1-\gamma)L_+ + \gamma L_\times$

The leading error is $\tfrac{h^2}{12}(u_{xxxx} + u_{yyyy} + 6\gamma u_{xxyy})$. It is isotropic, meaning proportional to $\nabla^4 u = u_{xxxx} + 2u_{xxyy} + u_{yyyy}$, when $\gamma = \tfrac13$.

### Isotropic 9-point Laplacian

$$\nabla_h^2 U = \frac{1}{6h^2}\Big[4(\text{axial}) + (\text{diagonal}) - 20\,U_{i,j}\Big] \qquad \text{leading error } \tfrac{h^2}{12}\nabla^4u$$

### Shift-operator (nested) form

With $\delta_x^2 = (E_x - 2 + E_x^{-1})/h^2$:

$$\nabla_h^2 = \delta_x^2 + \delta_y^2 + \frac{h^2}{6}\,\delta_x^2\delta_y^2$$

The off-axis correction is the **product** of 1D operators. The same pattern appears in 3D and for Maxwell, with terms like $\delta_x\delta_y^2$.

### Symbol and stability

With $s_x = \sin(\theta_x/2)$ and $s_y = \sin(\theta_y/2)$:

$$-\frac{4}{h^2}\Big[s_x^2 + s_y^2 - \tfrac23 s_x^2s_y^2\Big]$$

- **Forward Euler:** the worst mode is at $\theta_x = \theta_y = \pi$, giving $r \le 3/8$. The 5-point stencil gives $r \le 1/4$.
- **Order:** 3×3 can only reach second order. Fourth order needs 5×5 or wider.

---

## 9. 2D off-axis Laplacian on a 5×5 grid (fourth order, isotropic)

### Neighbor families

| Family | Offset | Distance | # nodes | Weight |
|---|---|---|---|---|
| axial | (1,0) | $h$ | 4 | $a$ |
| diagonal | (1,1) | $\sqrt2h$ | 4 | $b$ |
| far axial | (2,0) | $2h$ | 4 | $c$ |
| knight | (2,1) | $\sqrt5h$ | 8 | $d$ |
| far diagonal | (2,2) | $2\sqrt2h$ | 4 | $e$ |
| center | (0,0) | 0 | 1 | $c_0$ |

The stencil is $\nabla_h^2U = \tfrac{1}{h^2}\big[c_0U_{i,j} + \sum_f w_f\sum_{\text{nodes in }f}U\big]$.

### Moment equations (2D Taylor expansion)

| Condition | Equation |
|---|---|
| Constant gives zero | $c_0 + 4a + 4b + 4c + 8d + 4e = 0$ |
| $\nabla^2$ coefficient = 1 | $a + 2b + 4c + 10d + 8e = 1$ |
| $u_{xxxx}$ term vanishes | $2a + 4b + 32c + 68d + 64e = 0$ |
| $u_{xxyy}$ term vanishes | $b + 8d + 16e = 0$ |
| Isotropic $h^4$ error | $B = 3A$ |

where

$$A = \tfrac{a}{360} + \tfrac{b}{180} + \tfrac{8c}{45} + \tfrac{13d}{36} + \tfrac{16e}{45} \;\;(u_{xxxxxx}), \qquad B = \tfrac{b}{12} + \tfrac{5d}{3} + \tfrac{16e}{3} \;\;(u_{xxxxyy})$$

The isotropy condition makes the $h^4$ error proportional to $\nabla^6u = u_{xxxxxx} + 3u_{xxxxyy} + 3u_{xxyyyy} + u_{yyyyyy}$.

### One-parameter family

$$a = \tfrac{13}{15} - 24e,\quad b = \tfrac4{15} + 16e,\quad c = 6e - \tfrac1{60},\quad d = -\tfrac1{30} - 4e,\quad c_0 = 36e - \tfrac{21}5$$

Every member has leading error $-\tfrac{h^4}{90}\nabla^6u$. **Sixth order is impossible on 5×5.**

### Knight stencil ($e = 0$)

$$\nabla_h^2U = \frac{1}{60h^2}\Big[52(\text{axial}) + 16(\text{diag}) - (\text{far axial}) - 2(\text{knight}) - 252\,U_{i,j}\Big]$$

### Corner stencil ($d = 0$, knights neglected)

This forces $e = -\tfrac1{120}$, so the stencil is unique.

$$\frac{1}{120h^2}\begin{bmatrix} -1 & 0 & -8 & 0 & -1 \\ 0 & 16 & 128 & 16 & 0 \\ -8 & 128 & -540 & 128 & -8 \\ 0 & 16 & 128 & 16 & 0 \\ -1 & 0 & -8 & 0 & -1 \end{bmatrix}$$

### Comparison

Phase error is the relative error at 10 points per wavelength, swept from 0° to 45°.

| Stencil | Nodes | Forward Euler limit $r$ | Phase error over angle |
|---|---|---|---|
| Plain 4th-order + (axis only) | 9 | 0.1875 | −0.043% to −0.167% (4× variation) |
| Knight ($e=0$) | 21 | 0.3125 | −0.166% to −0.167% (flat) |
| Corner ($d=0$) | 17 | 0.234 | −0.163% to −0.167% |

- **Isotropy, not size.** Off-axis stencils make the error the same in every direction rather than smaller overall.
- **Crank–Nicolson.** All three symbols are $\le 0$, so CN is unconditionally stable with any of them.
- **Recommendation.** The knight stencil is the most isotropic and allows the largest explicit time step. The corner stencil is sparser, slightly less isotropic at $h^6$, and has a smaller explicit time step.

### Dropping both knights and far corners ($d = e = 0$)

The $u_{xxyy}$ equation then forces $b = 0$, leaving the plain axis-only fourth-order stencil, which is anisotropic. **A fourth-order isotropic 5×5 stencil needs at least one far off-axis family.**

### Relation to S24

- **Axial reduction.** Along an axis, the 5×5 stencils reduce to the standard $(-1, 16, -30, 16, -1)/12h^2$.
- **Width.** The S24-derived 7-point stencil needs a 7×7 grid.
- **Extending to S24.** Use the same method, adding the (3,0), (3,1), (3,2), and (3,3) families and more equations.

---

## 10. Next steps

1. Derive the 5×5 moment equations by hand from a 2D Taylor expansion of $U_{i+m,j+n}$, then verify the knight weights.
2. Implement 2D Crank–Nicolson heat with the 9-point and 5×5 stencils. Check convergence and plot angular phase error.
3. Move to 2D TM Maxwell. Put off-axis terms in the staggered first derivatives (Section 7), derive the 2D dispersion relation, and tune coefficients for dispersion-free propagation along the beam axis.
4. Combine with S24's fourth-order axial stencil (7×7 support) and handle boundaries and material interfaces.

## References (for further reading)

- J. Fang, PhD thesis (UC Berkeley, 1989): the (2,4) staggered FDTD scheme.
- M. F. Hadi and M. Piket-May, *IEEE Trans. Antennas Propag.* (1997): the modified (2,4) "M24" scheme.
- M. Kärkkäinen, E. Gjonaj, T. Lau, T. Weiland: low-dispersion wake-field / off-axis FDTD schemes for accelerators.
