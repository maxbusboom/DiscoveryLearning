# Error Analysis Review: Findings and Improvement Plan

A review of how `src/error_analysis.py` measures the error of the compact 9-point / Crank–Nicolson heat solver (`src/heat9.py`). It looks at whether truncation, roundoff and number-representation errors are accounted for, and at any other error sources, then lays out a plan to improve the analysis.

Unless stated otherwise, numbers are for the 9-point stencil, $\alpha = 1$, $T = 0.02$, max-norm over interior nodes.

---

## 1. Summary

| Error source | Accounted for? | Size at $h = 1/128$ | Verdict |
|---|---|---|---|
| Space truncation, $O(h^2)$ | Yes (Study A) | $1.2\times10^{-4}$ | Measured correctly |
| Time truncation, $O(\Delta t^2)$ | Yes (Study B) | $3\times10^{-8}$ to $10^{-1}$ depending on $\Delta t$ | Measured correctly |
| Roundoff | **No** | $\sim10^{-13}$ | Negligible, but nothing in the code shows that |
| Number representation | **No** | $\lesssim10^{-16}$ relative | Negligible |
| Linear-solver tolerance | Not applicable | 0 | Direct LU solve, so no iteration error |

The convergence orders the script reports (2 in space, 2 in time) are correct. The gaps are:

- roundoff is never estimated;
- the totals hide how the space and time errors interact;
- the default test is weaker than its docstring says.

---

## 2. What the current code measures

`max_error` ([error_analysis.py:105](../src/error_analysis.py)) computes

$$e = \max_{\text{interior nodes}} \big|U_{p,q}^N - u(x_p, y_q, T)\big|,$$

which is the **total** error at one time $T$. Three refinement studies separate its parts by their *orders*:

- **A. Space:** refine $h$ with $\Delta t = 0.25h^2$, so time error is $O(h^4)$.
- **B. Time:** fix $h = 1/128$ and compare against a 4096-step solution on the same grid.
- **C. Combined:** refine $h$ with $\Delta t/h = 0.16$.

Each study reports pairwise orders $\log_2(e_{k-1}/e_k)$ and a least-squares log–log slope.

---

## 3. Exact error split (method)

In the default homogeneous case, every mode $\sin(ax)\sin(by)$ (with $a, b$ integer multiples of $\pi$) is an **exact eigenvector** of the discrete Dirichlet operator, for both stencils. The discrete operator therefore only changes each mode's decay rate, from $\lambda = -(a^2+b^2)$ to

$$\lambda_h = \frac{\hat\sigma(ah,\,bh)}{h^2}, \qquad \hat\sigma = \texttt{fourier\_symbol(stencil, } ah,\ bh\texttt{)}.$$

That gives closed forms for two intermediate answers:

| Answer | Formula (per mode, coefficient $c$) |
|---|---|
| True solution | $c\,e^{\lambda T}\,v$ |
| Perfect ODE answer (exact in time) | $c\,e^{\lambda_h T}\,v$ |
| Perfect CN answer (exact arithmetic) | $c\,R(z)^N v,\quad R(z) = \dfrac{1+z/2}{1-z/2},\ z = \alpha\Delta t\,\lambda_h$ |
| Solver output (float64) | $U^N$ |

Taking differences between neighbours in this chain splits the total error exactly:

$$U^N - u = \underbrace{(U^N - U_{\text{CN}})}_{\text{roundoff}} + \underbrace{(U_{\text{CN}} - U_{\text{ODE}})}_{\text{time truncation}} + \underbrace{(U_{\text{ODE}} - u)}_{\text{space truncation}}$$

---

## 4. Findings

### 4.1 Measured error budget

| Case | Total | Space | Time | Roundoff |
|---|---|---|---|---|
| A, $N=8$, 5 steps | 2.84e-2 | 3.01e-2 | 1.79e-3 | 5.6e-16 |
| A, $N=32$, 82 steps | 1.90e-3 | 1.91e-3 | 8.7e-6 | 8.1e-15 |
| A, $N=128$, 1311 steps | 1.19e-4 | 1.19e-4 | 3.4e-8 | 2.1e-13 |
| B, $N=128$, 1 step | 9.67e-2 | 1.19e-4 | 9.68e-2 | 9.6e-14 |
| B, $N=128$, 16 steps | 1.58e-4 | 1.19e-4 | 2.30e-4 | 1.0e-13 |
| B, $N=128$, 64 steps | 1.07e-4 | 1.19e-4 | 1.43e-5 | 8.5e-14 |
| B, $N=128$, 4096 steps (the reference) | 1.19e-4 | 1.19e-4 | 3.5e-9 | 1.4e-13 |
| C, $N=8$, 1 step | 5.14e-2 | 3.01e-2 | 6.64e-2 | 6.7e-16 |
| C, $N=128$, 16 steps | 1.58e-4 | 1.19e-4 | 2.30e-4 | 1.0e-13 |

(Each column is a max-norm, so the three parts need not add up to the total.)

### 4.2 Truncation error: handled well, with caveats

1. **The studies isolate each order correctly.**
   - In Study A, time error is 6% of the total at $N=8$ and 0.03% at $N=128$.
   - In Study B, the reference solution's own time error ($3.5\times10^{-9}$) is 0.4% of the smallest error measured against it ($8.9\times10^{-7}$).
2. **Space and time errors have opposite signs and partly cancel.**
   - At 64 steps in Study B, the total ($1.07\times10^{-4}$) is *below* the space error ($1.19\times10^{-4}$). The "vs analytic" column dips to $7.1\times10^{-5}$ at 32 steps and then rises back.
   - The script's explanation ([error_analysis.py:196](../src/error_analysis.py)) says this column "stops improving", which misses the cancellation.
3. **Study C is mostly time error.** At $\Delta t/h = 0.16$, the time error is about 1.9× the space error. The slope of 2 is right, but the error sizes in C mainly measure Crank–Nicolson error, not the stencil. This includes the 9-point vs 5-point comparison.
4. **The least-squares slopes include coarse levels that haven't converged yet.**
   - Study C at $N=8$ has pairwise order 2.27, which pulls the reported slope to 2.076.
   - Study B's fit window `fit=(1, 7)` is picked by hand and leaves out two clean points (orders 2.001 and 2.004).
5. **The 9-point stencil is still $O(h^2)$.** Its leading error $(h^2/12)\nabla^4 u$ is isotropic but never cancelled. For these modes it is about 1.4× the 5-point error. Fourth order would need the Mehrstellen correction to $u_t$. That would change the scheme, not the analysis, so it is out of scope here.

### 4.3 Roundoff: not accounted for

The code has no roundoff estimate, no floor check and no guard in `orders()`. If the error ever reached roundoff level, $\log_2(e_{k-1}/e_k)$ would quietly report meaningless orders.

Roundoff sources in the solver:

- **Conditioning of $A$.** $A = I - \tfrac{\alpha\Delta t}{2}L$ has $\kappa(A) \approx 1 + \tfrac{8}{3}\mu$ for the 9-point stencil, which is about 870 for Study B with 1 step ($\mu \approx 327$).
- **Cancellation in $B U$.** In $B U = (I + \tfrac{\alpha\Delta t}{2}L)U$, terms of size $O(\mu)$ cancel down to $O(1)$.
- **Accumulation** over as many as 4096 steps.
- **No damping of stiff modes.** Crank–Nicolson is not L-stable: for stiff modes $R(z) \to -1$, so roundoff in those modes is carried forward, not damped out.
- **Constant boundary data.** With $u = 1$ on the boundary, $LU$ and $b$ are each $O(1/h^2)$ and cancel. I measured the constant-BC solution minus (homogeneous solution + 1) at $1.8\times10^{-13}$.

**Measured:** roundoff is about $10^{-13}$ at $h = 1/128$ and grows roughly like $\varepsilon/h^2$ (×4.5 per refinement in Study A). Truncation shrinks like $h^2$, so the two would only meet near $N \approx 10^4$. Roundoff is therefore negligible for every grid used, but the analysis doesn't currently show that.

### 4.4 Number representation

- **Time accumulation.** `t += dt` ([heat9.py:398](../src/heat9.py)) adds up rounding error, and $\Delta t = 0.02/n$ is generally not exactly representable. This only affects $b(t)$ in the analytic-BC case and is about $10^{-16}$ relative. Writing `t = t0 + (step + 1) * dt` would remove it.
- **Exact quantities.** $h = 1/2^k$ and the node coordinates $p h$ are exact in binary.
- **Rounding of $\pi$.** $\sin(k\pi\cdot 1) \approx 10^{-16}$ rather than 0. This doesn't matter, because the homogeneous case never samples the boundary.
- **Evaluating the exact solution** has about $\varepsilon$ relative error.

### 4.5 Other limitations of the error measure

- **It is a nodal error.** It is the max over interior nodes, not the continuous $L^\infty$ error. Error between nodes (an extra $O(h^2)$ if the field is interpolated) is not measured.
- **It is measured at one time only.** Error at $t < T$ is never checked.
- **It is absolute, not relative.** That's fine for an amplitude of about 1. In constant-BC mode the $+1$ offset changes what a relative error would mean.
- **The default test is weaker than documented.** The module docstring (lines 9–12) says the test avoids grid-aligned eigenmodes and exercises $b(t)$. The default `BOUNDARY = "homogeneous"` drops the phases, so all four modes *are* exact discrete eigenvectors and $b \equiv 0$. Only `--bc analytic` matches the description.
- **The constant BC adds no new information.** Its error equals the homogeneous error to within roundoff.
- **Mesh ratio drift.** Rounding `nsteps` in Study A means $\mu$ isn't exactly 0.25 (0.256 at $N=8$). This is harmless.

---

## 5. Improvement plan

**Status:** Steps 1–5 are implemented in `src/error_analysis.py` (new Study D prints the exact budget), `src/heat9.py` (step 5's time fix), and `test/test_error_analysis.py`. Step 6 is still open.

Steps are in priority order. Steps 1–4 change only `src/error_analysis.py`; step 5 touches `src/heat9.py`.

### Step 1: Add an exact error split (homogeneous BC)

- Add `error_budget(grid, nsteps, stencil)` returning `(total, space, time, roundoff)` using the closed forms in §3. The ingredients are `H.fourier_symbol`, the amplification factor $R(z)$, and `MODES`.
- Add a budget table after Studies A–C. This turns "the orders are 2" into actual sizes for each part.
- Add a test in `test/test_heat9.py` that the three parts sum to the total (as signed fields, not max-norms) to within $10^{-12}$.

### Step 2: Estimate the roundoff floor everywhere

- Homogeneous BC: use the roundoff part from step 1.
- Analytic BC: there is no closed form, so use the estimate $\varepsilon\,\big(1 + N_{\text{steps}}\,(\kappa(A) + \||B|\|_\infty)\big)\,\|U\|_\infty$. $\kappa(A)$ is known from the symbol: $1 + \tfrac{\alpha\Delta t}{2}\,|\hat\sigma_{\min}|/h^2$, where $\hat\sigma_{\min} = -16/3$ for the 9-point stencil. The $\||B|\|_\infty$ term covers the cancellation in the matvec $BU$; without it the estimate fell below the measured roundoff at large $\mu$ (e.g. $h = 1/64$, 8 steps).
- Print the floor as a column next to each error table.
- In `orders()`, mark any order whose errors are within 100× the floor as unreliable instead of printing a number.

### Step 3: Report the space and time parts in Studies B and C

- In Study B, plot the space and time parts (from step 1) next to "vs analytic" so the cancellation dip is explained.
- Replace the text at [error_analysis.py:196](../src/error_analysis.py) with a description of the cancellation.
- In Study C, print the time-to-space error ratio. Optionally add a second $\Delta t/h$ value (e.g. 0.04) where space error dominates, so the 9-point vs 5-point comparison measures the stencils.

### Step 4: Fit slopes only where the orders have settled

- Replace the hand-picked `fit=(1, 7)` with an automatic window: the longest run of levels whose pairwise orders are within 0.05 of each other and above the roundoff floor.
- Print the chosen window with each slope, and apply the same rule to Studies A and C.

### Step 5: Fix the documentation and small precision issues

- Make the module docstring match the default, or make `analytic` the default since it is the more demanding test.
- Relabel the `constant` mode as a roundoff and consistency check rather than a separate accuracy test.
- In `crank_nicolson_solve` and `forward_euler_solve`, compute `t = t0 + (step + 1) * dt` instead of accumulating `t += dt`.

### Step 6 (optional): Broaden the error measure

- Track $\max_n \|U^n - u(t^n)\|_\infty$ through the `callback` hook, to cover error over the whole run.
- Report a discrete $L^2$ norm next to the max-norm.
- Check the measured space error against the leading-order prediction $\tfrac{h^2}{12}(a^2+b^2)^2\,T\,e^{\lambda T}$ per mode, which confirms the error constant as well as the order.

### Out of scope

- **Fourth-order Mehrstellen time derivative**, $(I + \tfrac{h^2}{12}L_5)\,U_t = \alpha L_9 U$. It would change the scheme, not the error analysis, and deserves its own plan.

---

## 6. Reproducing the numbers

The figures in §4 came from small scratch scripts that call `error_analysis.study_*` and compute the split in §3 directly. Step 1 of the plan makes that split a permanent part of `src/error_analysis.py`.
