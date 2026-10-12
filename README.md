# DiscoveryLearning

Compact 9-point Laplacian with Crank-Nicolson time stepping for the 2D heat
equation (`heat9.py`), plus verification and error-analysis scripts.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
uv run pytest                                      # test suite
uv run python src/run_demo.py                      # verification summary + reports/isotropy.png
uv run python src/error_analysis.py                # convergence study + reports/error_analysis.png
uv run python src/error_analysis.py --bc constant  # alternate boundary condition
```
