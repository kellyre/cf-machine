"""
Named catalog of Polynomial Continued Fractions.

Each entry is a (name, description, PCFForm) triple.  The catalog is a plain
dict so it can be iterated, filtered, and sent to worker processes without
unpickling class instances.
"""

from __future__ import annotations

from pcf import PCFForm

# ---------------------------------------------------------------------------
# Helper: build coefficient tuples for common polynomial patterns
# ---------------------------------------------------------------------------

def _const(c: int) -> tuple[int, ...]:
    """Constant polynomial c."""
    return (c,)


def _linear(a: int, b: int) -> tuple[int, ...]:
    """Polynomial a + b*n  (coeffs in ascending degree)."""
    return (a, b)


def _quad(a: int, b: int, c: int) -> tuple[int, ...]:
    """Polynomial a + b*n + c*n^2."""
    return (a, b, c)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
#
# BROUNCKER / PI-PCF
#   Both share  a(n) = (2n-1)^2 = 4n^2 - 4n + 1
#
# The PSLQ demo should discover:  v_brouncker * v_pi = 4
#

CATALOG: dict[str, tuple[str, PCFForm]] = {
    # ------------------------------------------------------------------ π
    "brouncker": (
        "Brouncker's formula: 4/π = 1 + 1²/(2 + 3²/(2 + 5²/(2 + ...)))",
        PCFForm(
            a_coeffs=_quad(1, -4, 4),   # (2n-1)^2 = 4n^2 - 4n + 1
            b_coeffs=_const(2),
            b0=1,
        ),
    ),
    "pi_pcf": (
        "π via  3 + 1²/(6 + 3²/(6 + 5²/(6 + ...)))",
        PCFForm(
            a_coeffs=_quad(1, -4, 4),   # (2n-1)^2
            b_coeffs=_const(6),
            b0=3,
        ),
    ),

    # ------------------------------------------------------------------ φ / √5 / δ
    "golden": (
        "Golden ratio φ = (1+√5)/2  via  1 + 1/(1 + 1/(1 + ...))",
        PCFForm(
            a_coeffs=_const(1),
            b_coeffs=_const(1),
            b0=1,
        ),
    ),
    "sqrt5": (
        "√5  via  2 + 1/(4 + 1/(4 + 1/(4 + ...)))  — relates to φ by 2φ = 1 + √5",
        PCFForm(
            a_coeffs=_const(1),
            b_coeffs=_const(4),
            b0=2,
        ),
    ),
    "silver": (
        "Silver ratio δ = 1+√2  via  2 + 1/(2 + 1/(2 + ...))",
        PCFForm(
            a_coeffs=_const(1),
            b_coeffs=_const(2),
            b0=2,
        ),
    ),

    # ------------------------------------------------------------------ equivalence-transform demo
    # Take golden ratio and apply the equivalence transform with c(n) = n+1:
    #   a'(n) = c(n)*c(n-1)*a(n) = (n+1)*n*1 = n^2 + n
    #   b'(n) = c(n)*b(n)        = (n+1)*1   = n + 1
    #   b0 is unchanged (the transform only rescales partial quotients ≥ 1)
    # c(n) = n+1 avoids c(0)=0, which would collapse a'(1)=0 and terminate the CF.
    # Convergents: 2, 3/2, 5/3, 8/5, 13/8, … (Fibonacci ratios) → φ.
    # The two CFs must have the same value, so v_golden_equiv - v_golden = 0.
    "golden_equiv": (
        "Equivalence transform of golden ratio (same value, different coefficients)",
        PCFForm(
            a_coeffs=_quad(0, 1, 1),    # n^2 + n
            b_coeffs=_linear(1, 1),      # n + 1
            b0=1,
        ),
    ),
}


def get(name: str) -> tuple[str, PCFForm]:
    """Return (description, PCFForm) by name."""
    return CATALOG[name]


def all_names() -> list[str]:
    return list(CATALOG.keys())
