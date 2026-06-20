# CF-Machine

A research tool for discovering algebraic relationships between **Polynomial Continued Fractions (PCFs)** using PSLQ integer-relation detection — the same mathematical primitive used by the [Ramanujan Machine](https://www.ramanujanmachine.com/).

## Goal

Given two (or more) generalized continued fractions defined by polynomial coefficient forms, detect algebraic relationships between their **values** — especially Möbius/projective transformations and general polynomial identities — without any prior knowledge of what constants the CFs represent.

## The Math

### Generalized Continued Fraction

```
b0 + a(1) / (b(1) + a(2) / (b(2) + a(3) / (b(3) + ⋯)))
```

A **PCF** specifies `a(n)` and `b(n)` as polynomials in `n` with integer coefficients, plus a starting term `b0`.

### Evaluation via Matrix Recurrence

Working at `dps + 15` guard digits throughout:

```
h[-1]=1,  h[0]=b0
k[-1]=0,  k[0]=1

h[n] = b(n)·h[n-1] + a(n)·h[n-2]
k[n] = b(n)·k[n-1] + a(n)·k[n-2]

value ≈ h[n]/k[n]
```

Iterate until `|h[n]/k[n] − h[n-1]/k[n-1]| < 10^(−dps)` or 200 000 iterations.  Some PCFs (e.g. Brouncker's formula for π) converge linearly and require many terms — this is expected.

### Relation Detection

1. **Möbius basis** (degree 1 + cross term):  
   `[v1·v2, v1, v2, 1]`  
   A returned vector `[r, s, −p, −q]` means `r·v1·v2 + s·v1 − p·v2 − q = 0`, i.e.  
   `v1 = (p·v2 + q) / (r·v2 + s)`.

2. **General polynomial basis** up to total degree `d`:  
   All monomials `v1^i · v2^j` with `i+j ≤ d`.

3. **Re-verification** — every PSLQ candidate is checked at `2×dps` precision before being reported, guarding against spurious low-precision hits.

## Dependencies

| Package | Role |
|---------|------|
| `mpmath` | Arbitrary-precision arithmetic + `pslq` implementation |
| `gmpy2`  | Optional GMP/MPFR backend — makes mpmath **~5–10× faster** per core |

### Installing gmpy2 on Ubuntu/WSL2

```bash
sudo apt install -y libgmp-dev libmpfr-dev libmpc-dev
pip install gmpy2
```

On most systems the manylinux wheel installs without compiling.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo.py
# Higher precision (slower but more confident):
python demo.py --dps 80
```

## Architecture

```
pcf.py        — PCFForm dataclass + evaluate() pure function (picklable)
relations.py  — find_relation(), PSLQ wrappers, pretty-printers
library.py    — named catalog of PCFs
demo.py       — evaluation + pairwise relation search
```

The evaluation and relation-finding layers are **process-pool ready**: `evaluate` is a top-level picklable function with no closure state, safe for `concurrent.futures.ProcessPoolExecutor` under both `fork` and `spawn` start methods.  The parallel search engine (enumerate forms → evaluate → hash & group → PSLQ-verify) is a planned future phase.

## Demo Output

The demo should discover — with **no prior knowledge** that either CF involves π — that:

```
v_brouncker × v_pi_pcf = 4
```

because Brouncker's formula gives `4/π` and the pi-PCF gives `π`.

It also confirms a **negative control** (golden ratio vs silver ratio → no relation, as they live in different quadratic fields) and an **equivalence-transform identity** (two PCFs with the same value report `v1 − v2 = 0`).
