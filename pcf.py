"""
Polynomial Continued Fraction (PCF) representation and evaluation.

A PCF is a generalized CF:  b0 + a(1)/(b(1) + a(2)/(b(2) + ...))
where a(n) and b(n) are polynomials in n with integer coefficients.

Evaluation uses the standard matrix recurrence:
    h[-1]=1,  h[0]=b0
    k[-1]=0,  k[0]=1
    h[n] = b(n)*h[n-1] + a(n)*h[n-2]
    k[n] = b(n)*k[n-1] + a(n)*k[n-2]
    value ≈ h[n]/k[n]

`evaluate` is a top-level, picklable pure function — safe for ProcessPoolExecutor
under both fork and spawn start methods.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Sequence

import mpmath


@dataclass(frozen=True)
class PCFForm:
    """
    Immutable description of a polynomial continued fraction.

    The CF is:  b0 + a(1)/(b(1) + a(2)/(b(2) + ...))

    Each of a(n) and b(n) can be specified as either a polynomial (via
    *_coeffs in ascending-degree order) or a raw integer sequence (via
    *_seq, overrides *_coeffs when set).  The sequence is 0-indexed in the
    tuple but provides 1-indexed CF inputs: a_seq[k-1] is a(k).

    a_coeffs / b_coeffs : polynomial coefficients, ascending degree.
                          e.g. (0, 0, 1) → n².  Ignored when the
                          corresponding *_seq is not None.
    b0                  : the leading constant term.
    a_seq / b_seq       : raw integer sequences (e.g. from OEIS).
                          Evaluation stops when the index exceeds len(*_seq).
    a_seq_id / b_seq_id : human-readable label for display (e.g. 'A000045').
    """
    a_coeffs: tuple[int, ...]    # poly for a(n), ascending degree
    b_coeffs: tuple[int, ...]    # poly for b(n), ascending degree
    b0: int = 1                  # leading term
    a_seq: tuple[int, ...] | None = None   # raw sequence for a(n); overrides a_coeffs
    b_seq: tuple[int, ...] | None = None   # raw sequence for b(n); overrides b_coeffs
    a_seq_id: str = ""
    b_seq_id: str = ""

    def __post_init__(self) -> None:
        # Normalise everything to tuples so the object is hashable/picklable.
        object.__setattr__(self, "a_coeffs", tuple(self.a_coeffs))
        object.__setattr__(self, "b_coeffs", tuple(self.b_coeffs))
        if self.a_seq is not None:
            object.__setattr__(self, "a_seq", tuple(self.a_seq))
        if self.b_seq is not None:
            object.__setattr__(self, "b_seq", tuple(self.b_seq))


def _poly_eval(coeffs: tuple[int, ...], n: "mpmath.mpf") -> "mpmath.mpf":
    """Horner evaluation of a polynomial with integer coefficients."""
    result = mpmath.mpf(0)
    for c in reversed(coeffs):
        result = result * n + c
    return result


def _wynn_epsilon(seq: list, depth: int) -> "mpmath.mpf":
    """
    Wynn epsilon extrapolation of a slowly converging sequence.

    Uses the last 2*depth+1 elements of *seq* and applies 2*depth rounds of
    the epsilon algorithm, returning the ε_{2*depth}(0) extrapolate.

    For a sequence converging as L + C₁/n + C₂/n² + …, each pair of ε steps
    cancels one more asymptotic term, dramatically accelerating convergence.

    The algorithm:
        ε_{-1}(n) = 0  (auxiliary)
        ε_0(n)    = seq[n]
        ε_{k+1}(n) = ε_{k-1}(n+1) + 1 / (ε_k(n+1) − ε_k(n))
    """
    needed = 2 * depth + 1
    s = seq[-needed:]

    eps_prev = [mpmath.mpf(0)] * len(s)   # ε_{-1} row (all zeros)
    eps_curr = [mpmath.mpf(v) for v in s]  # ε_0 row (the convergents)

    for _ in range(2 * depth):
        if len(eps_curr) < 2:
            break
        eps_next = []
        for i in range(len(eps_curr) - 1):
            d = eps_curr[i + 1] - eps_curr[i]
            if d == 0:
                eps_next.append(eps_prev[i + 1])
            else:
                eps_next.append(eps_prev[i + 1] + mpmath.mpf(1) / d)
        eps_prev = eps_curr
        eps_curr = eps_next

    return eps_curr[0] if eps_curr else mpmath.mpf(seq[-1])


# ---------------------------------------------------------------------------
# Top-level picklable evaluation function
# ---------------------------------------------------------------------------

_GUARD_DIGITS: int = 15
_MAX_ITER: int = 200_000


def evaluate(form: PCFForm, dps: int, *, epsilon_depth: int = 0) -> "mpmath.mpf":
    """
    Evaluate *form* to *dps* decimal places of precision.

    Parameters
    ----------
    form          : PCFForm describing the continued fraction.
    dps           : target decimal places of precision.
    epsilon_depth : if > 0 and the CF does not converge naturally within
                    _MAX_ITER iterations, apply Wynn epsilon extrapolation
                    at this depth to the last 2*epsilon_depth+1 convergents.
                    Useful for slowly-converging CFs (e.g. Brouncker's formula).
                    Each unit of depth cancels one more asymptotic term.
                    Values of 6–10 are typically sufficient for dps ≤ 100.

    Works at dps + _GUARD_DIGITS + 2*epsilon_depth internally to leave room
    for the epsilon algorithm's intermediate arithmetic.

    This function is a module-level callable — it can be pickled and sent to
    a worker process via concurrent.futures.ProcessPoolExecutor.
    """
    working_dps = dps + _GUARD_DIGITS + 2 * epsilon_depth

    # Sliding window of recent convergents for Wynn epsilon (kept small).
    window = 2 * epsilon_depth + 11 if epsilon_depth > 0 else 0

    with mpmath.workdps(working_dps):
        b0 = mpmath.mpf(form.b0)
        threshold = mpmath.mpf(10) ** (-dps)

        # Recurrence initialisation
        h_prev = mpmath.mpf(1)   # h[-1]
        h_curr = b0               # h[0]
        k_prev = mpmath.mpf(0)   # k[-1]
        k_curr = mpmath.mpf(1)   # k[0]

        value = h_curr / k_curr  # initial estimate

        recent: collections.deque | None = (
            collections.deque(maxlen=window) if window else None
        )

        a_seq = form.a_seq
        b_seq = form.b_seq
        a_len = len(a_seq) if a_seq is not None else _MAX_ITER
        b_len = len(b_seq) if b_seq is not None else _MAX_ITER

        converged = False
        for n in range(1, _MAX_ITER + 1):
            if n > a_len or n > b_len:
                break   # sequence(s) exhausted
            n_mp = mpmath.mpf(n)
            an = mpmath.mpf(a_seq[n - 1]) if a_seq is not None else _poly_eval(form.a_coeffs, n_mp)
            bn = mpmath.mpf(b_seq[n - 1]) if b_seq is not None else _poly_eval(form.b_coeffs, n_mp)

            h_new = bn * h_curr + an * h_prev
            k_new = bn * k_curr + an * k_prev

            new_value = h_new / k_new
            if recent is not None:
                recent.append(new_value)
            if abs(new_value - value) < threshold:
                value = new_value
                converged = True
                break

            value = new_value
            h_prev, h_curr = h_curr, h_new
            k_prev, k_curr = k_curr, k_new

        # If the CF did not converge naturally, try Wynn epsilon extrapolation.
        if not converged and recent is not None and len(recent) >= 2 * epsilon_depth + 1:
            value = _wynn_epsilon(list(recent), epsilon_depth)

        # Return the value captured inside the workdps context so it retains
        # the working precision (mpmath mpf carries its own precision).
        return +value  # unary + forces a copy at current working precision
