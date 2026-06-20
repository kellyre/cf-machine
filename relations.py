"""
Algebraic relation detection between PCF values using PSLQ.

`find_relation(values, max_degree, dps)` builds a monomial basis over the
given values and calls mpmath.pslq to find an integer relation vector.  Any
candidate is re-verified at 2× precision before being returned as confirmed.

Pretty-printers handle:
  - Möbius/bilinear form (degree-1 basis with cross term): [v1*v2, v1, v2, 1]
  - General polynomial in 2 variables up to total degree *d*.
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

import mpmath


# ---------------------------------------------------------------------------
# Basis helpers
# ---------------------------------------------------------------------------

# Exponents for the 2-variable Möbius basis [v1*v2, v1, v2, 1].
_MOBIUS_EXPS: list[tuple[int, int]] = [(1, 1), (1, 0), (0, 1), (0, 0)]
_MOBIUS_LABELS: list[str] = ["v1*v2", "v1", "v2", "1"]


def _poly_basis_and_exps(
    values: Sequence["mpmath.mpf"],
    max_degree: int,
) -> tuple[list["mpmath.mpf"], list[tuple[int, ...]], list[str]]:
    """
    Build the full polynomial monomial basis up to total degree *max_degree*.

    Returns (basis_values, exponent_tuples, labels).
    """
    k = len(values)
    basis_vals: list["mpmath.mpf"] = []
    basis_exps: list[tuple[int, ...]] = []
    labels: list[str] = []

    for exponents in product(range(max_degree + 1), repeat=k):
        if sum(exponents) > max_degree:
            continue
        mono = mpmath.mpf(1)
        for v, e in zip(values, exponents):
            mono *= v ** e
        basis_vals.append(mono)
        basis_exps.append(exponents)
        parts = [
            f"v{i+1}^{e}" if e > 1 else (f"v{i+1}" if e == 1 else "")
            for i, e in enumerate(exponents)
        ]
        labels.append("*".join(p for p in parts if p) or "1")

    return basis_vals, basis_exps, labels


def _all_vars_present(vector: list[int], exps: list[tuple]) -> bool:
    """
    Return True only if every variable (v1, v2, …) appears in at least one
    non-zero term of the relation.

    Filters out trivial single-variable identities (e.g. φ²−φ−1=0) that
    PSLQ may find when searching a multi-variable polynomial basis.
    """
    k = len(exps[0]) if exps else 0
    active: set[int] = set()
    for c, exp in zip(vector, exps):
        if c != 0:
            for i, e in enumerate(exp):
                if e > 0:
                    active.add(i)
    return len(active) == k


# ---------------------------------------------------------------------------
# Core relation finder
# ---------------------------------------------------------------------------

def find_relation(
    values: Sequence["mpmath.mpf"],
    max_degree: int = 1,
    dps: int = 50,
    *,
    verify_dps: int | None = None,
    maxcoeff: int = 1000,
) -> dict | None:
    """
    Search for an integer relation among *values*.

    Parameters
    ----------
    values      : sequence of mpmath.mpf (at least 2).
    max_degree  : total-degree bound for the monomial basis.
                  1 → Möbius basis [v1*v2, v1, v2, 1] for two values.
                  ≥2 → full polynomial basis up to that degree.
    dps         : working precision for the initial PSLQ call.
    verify_dps  : precision used for re-verification (default: 2*dps).
    maxcoeff    : PSLQ maxcoeff guard against trivial/giant-coefficient hits.

    Returns
    -------
    A dict with keys:
      'vector'    : list[int] — the integer relation coefficients
      'basis_desc': list[str] — human-readable monomial labels
      'residual'  : mpmath.mpf — |∑ c_i * m_i| at verify_dps
    or None if no relation was found (or verification / cross-variable check
    failed).

    Note: relations that only involve a strict subset of the input variables
    (e.g. the minimal polynomial of one value) are silently discarded.
    """
    if verify_dps is None:
        verify_dps = 2 * dps

    if len(values) < 2:
        raise ValueError("Need at least 2 values.")

    k = len(values)
    use_mobius = (k == 2 and max_degree == 1)

    # --- build basis exponents and labels (once, precision-independent) ---
    if use_mobius:
        basis_exps: list[tuple] = _MOBIUS_EXPS  # type: ignore[assignment]
        labels = _MOBIUS_LABELS
    else:
        _, basis_exps, labels = _poly_basis_and_exps(values, max_degree)

    def _make_basis(vals: Sequence["mpmath.mpf"]) -> list["mpmath.mpf"]:
        if use_mobius:
            v1, v2 = vals[0], vals[1]
            return [v1 * v2, v1, v2, mpmath.mpf(1)]
        result = []
        for exps in basis_exps:
            mono = mpmath.mpf(1)
            for v, e in zip(vals, exps):
                mono *= v ** e
            result.append(mono)
        return result

    # --- PSLQ at working precision ---
    with mpmath.workdps(dps):
        basis = _make_basis(values)
        rel = mpmath.pslq(basis, maxcoeff=maxcoeff)

    if rel is None:
        return None

    # --- Cross-variable check: reject single-variable identities ---
    if not _all_vars_present(list(rel), basis_exps):
        return None

    # --- Re-verify at higher precision ---
    # Threshold is based on evaluation precision, not verify_dps, because
    # the input *values* are only accurate to ~dps digits regardless of the
    # working precision used here.
    with mpmath.workdps(verify_dps):
        hi_basis = _make_basis(values)
        residual = abs(sum(mpmath.mpf(c) * m for c, m in zip(rel, hi_basis)))
        threshold = mpmath.mpf(10) ** (-(dps // 2))

    if residual > threshold:
        return None  # spurious relation; discard

    return {
        "vector": list(rel),
        "basis_desc": labels,
        "residual": residual,
    }


# ---------------------------------------------------------------------------
# Pretty-printers
# ---------------------------------------------------------------------------

def _coeff_str(c: int, label: str, first: bool) -> str:
    """Format one term  c * label  for a polynomial expression."""
    if label == "1":
        return ("" if first else "+ ") + str(c) if c >= 0 else f"- {-c}"
    if c == 1:
        return ("" if first else "+ ") + label
    if c == -1:
        return "- " + label
    if c > 0:
        return ("" if first else "+ ") + f"{c}*{label}"
    return f"- {-c}*{label}"


def format_polynomial(result: dict) -> str:
    """
    Return a human-readable polynomial = 0 string from a find_relation result.
    """
    terms = []
    for c, label in zip(result["vector"], result["basis_desc"]):
        if c != 0:
            terms.append((c, label))

    if not terms:
        return "0 = 0"

    parts = []
    for i, (c, label) in enumerate(terms):
        parts.append(_coeff_str(c, label, i == 0))

    return " ".join(parts) + " = 0"


def format_mobius(result: dict, name1: str = "v1", name2: str = "v2") -> str | None:
    """
    If *result* came from a Möbius / degree-1 basis on two variables, return
    the solved form  v1 = (p*v2 + q) / (r*v2 + s).

    Basis order: [v1*v2, v1, v2, 1]  →  r*v1*v2 + s*v1 - p*v2 - q = 0
    (coefficients [r, s, -p, -q])
    """
    vec = result["vector"]
    if len(vec) != 4:
        return None

    r, s, neg_p, neg_q = vec
    p, q = -neg_p, -neg_q

    # Build numerator and denominator strings
    def _linear(a: int, var: str, b: int) -> str:
        parts = []
        if a != 0:
            parts.append(("" if a == 1 else ("-" if a == -1 else str(a) + "*")) + var)
        if b != 0:
            if b > 0 and parts:
                parts.append(f"+ {b}")
            else:
                parts.append(str(b))
        return " ".join(parts) if parts else "0"

    numer = _linear(p, name2, q)
    denom = _linear(r, name2, s)

    if denom == "1":
        return f"{name1} = {numer}"
    if denom == "-1":
        return f"{name1} = -({numer})"
    return f"{name1} = ({numer}) / ({denom})"


def print_relation(
    result: dict,
    names: Sequence[str] | None = None,
    label: str = "",
) -> None:
    """Print a relation result with both polynomial and Möbius forms."""
    prefix = f"[{label}] " if label else ""
    poly_str = format_polynomial(result)

    # Substitute actual CF names into the label string
    display = poly_str
    if names:
        for i, name in enumerate(names):
            display = display.replace(f"v{i+1}", name)

    print(f"{prefix}RELATION FOUND:  {display}")
    print(f"  residual at 2× dps: {float(result['residual']):.2e}")

    if len(result["vector"]) == 4 and names and len(names) >= 2:
        mob = format_mobius(result, names[0], names[1])
        if mob:
            mob_display = mob
            print(f"  Möbius form:      {mob_display}")

    print()
