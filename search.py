"""
Parallel PCF search engine.

Pipeline
--------
  1. Enumerate  — generate PCFForm candidates within coefficient/degree bounds
  2. Evaluate   — compute each CF value in parallel (ProcessPoolExecutor)
  3. Cluster    — group forms that evaluate to the same constant
  4. PSLQ       — search every pair of distinct values for algebraic relations

Usage
-----
    python search.py                              # fast default (max_coeff=2, a_deg=1)
    python search.py --max-coeff 3 --a-degree 2  # wider search
    python search.py --workers 28 --dps 60       # explicit parallelism / precision
    python search.py --max-degree 2              # include degree-2 polynomial relations
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from fractions import Fraction
from itertools import combinations, product

import mpmath

from pcf import PCFForm, evaluate
from relations import find_relation, format_polynomial, format_mobius


# ─────────────────────────────────────────────────────────────────────────────
# Worker functions — must be module-level to be picklable
# ─────────────────────────────────────────────────────────────────────────────

def _eval_worker(form: PCFForm, dps: int) -> tuple[PCFForm, "mpmath.mpf | None"]:
    """Evaluate one PCF; return (form, None) on divergence, overflow, or trivial zero."""
    try:
        val = evaluate(form, dps)
        if not mpmath.isfinite(val) or val == 0:
            return form, None
        return form, val
    except Exception:
        return form, None


def _is_b0_additive(f1: PCFForm, f2: PCFForm, result: dict) -> bool:
    """
    Return True when the relation is trivially explained by a b0 offset.

    Two CFs that share a(n) and b(n) but differ only in b0 always satisfy
    v1 - v2 = b0_1 - b0_2.  PSLQ faithfully finds this additive relation, but
    it carries no mathematical content beyond 'we varied a free parameter.'

    The Möbius vector for a purely additive relation is [0, ±1, ∓1, c].
    """
    if f1.a_coeffs != f2.a_coeffs or f1.b_coeffs != f2.b_coeffs:
        return False
    vec = result["vector"]
    if len(vec) != 4:
        return False
    r, s, neg_p, _ = vec
    return r == 0 and s == -neg_p


def _is_simple_rational(val: "mpmath.mpf", dps: int, max_denominator: int) -> bool:
    """
    Return True if val is numerically indistinguishable from a rational p/q
    with |q| <= max_denominator.

    Strategy: convert to a decimal string, find the best rational approximation
    via Fraction.limit_denominator, then check the residual against the working
    precision.  Uses Decimal as intermediary so scientific-notation strings
    (e.g. "1.23e-8") parse correctly.

    A CF that evaluates to a simple rational (e.g. -1, 1/2, 7/3) produces
    Möbius relations of the form  CF2*(CF1 - p/q) = 0  which are vacuous —
    they just restate that CF1 = p/q.
    """
    try:
        s = mpmath.nstr(val, dps // 2)
        frac = Fraction(Decimal(s)).limit_denominator(max_denominator)
        residual = abs(val - mpmath.mpf(frac.numerator) / mpmath.mpf(frac.denominator))
        return residual < mpmath.mpf(10) ** (-(dps // 3))
    except Exception:
        return False


def _pslq_worker(
    val1: "mpmath.mpf",
    val2: "mpmath.mpf",
    dps: int,
    max_degree: int,
) -> "dict | None":
    """Search for an integer relation between val1 and val2."""
    return find_relation([val1, val2], max_degree=max_degree, dps=dps)


# ─────────────────────────────────────────────────────────────────────────────
# Form enumeration
# ─────────────────────────────────────────────────────────────────────────────

def _b_stays_positive(b_coeffs: tuple[int, ...], n_max: int = 20) -> bool:
    """b(n) = Σ b_coeffs[i]·nⁱ must be > 0 for all n in 1..n_max."""
    for n in range(1, n_max + 1):
        if sum(c * n**i for i, c in enumerate(b_coeffs)) <= 0:
            return False
    return True


def enumerate_forms(
    max_coeff: int,
    a_degree: int,
    b_degree: int,
    b0_min: int,
    b0_max: int,
) -> list[PCFForm]:
    """
    All PCFForm candidates with polynomial coefficients in [-max_coeff, max_coeff].

    Filters applied:
      · a(1) ≠ 0  — a(1)=0 collapses the CF to the trivial value b0
      · a(n) not identically zero
      · b(n) > 0 for n = 1..20  — denominator recurrence must stay positive
    """
    cr = range(-max_coeff, max_coeff + 1)
    forms: list[PCFForm] = []
    for a_coeffs in product(cr, repeat=a_degree + 1):
        # a(1) = sum of all coefficients  (since 1^i = 1 for all i)
        if sum(a_coeffs) == 0 or all(c == 0 for c in a_coeffs):
            continue
        for b_coeffs in product(cr, repeat=b_degree + 1):
            if not _b_stays_positive(b_coeffs):
                continue
            for b0 in range(b0_min, b0_max + 1):
                forms.append(PCFForm(a_coeffs=a_coeffs, b_coeffs=b_coeffs, b0=b0))
    return forms


# ─────────────────────────────────────────────────────────────────────────────
# Value clustering
# ─────────────────────────────────────────────────────────────────────────────

def _cluster(
    form_vals: dict[PCFForm, "mpmath.mpf"],
    key_digits: int,
) -> list[list[tuple[PCFForm, "mpmath.mpf"]]]:
    """
    Group forms whose values agree to key_digits significant digits.
    Returns a list of groups; each group is a list of (form, value) pairs.
    """
    buckets: dict[str, list] = {}
    for form, val in form_vals.items():
        key = mpmath.nstr(val, key_digits)
        buckets.setdefault(key, []).append((form, val))
    return list(buckets.values())


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def _pcf_str(form: PCFForm) -> str:
    return f"a={list(form.a_coeffs)}  b={list(form.b_coeffs)}  b0={form.b0}"


def _bar(title: str, width: int = 62) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _progress(done: int, total: int, extra: str = "") -> None:
    w = len(str(total))
    print(f"\r  {done:{w}d}/{total}{extra}", end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main search pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_search(
    max_coeff: int = 2,
    a_degree: int = 1,
    b_degree: int = 1,
    b0_min: int = 0,
    b0_max: int = 3,
    dps: int = 50,
    max_degree: int = 1,
    workers: int | None = None,
    show_trivial: bool = False,
    max_rational_denom: int = 10_000_000,
) -> None:
    """
    Run the full enumerate → evaluate → cluster → PSLQ pipeline.

    Parameters
    ----------
    max_coeff  : absolute coefficient bound for a(n) and b(n) polynomials
    a_degree   : polynomial degree of the CF numerator a(n)
    b_degree   : polynomial degree of the CF denominator b(n)
    b0_min/max : range of the leading constant b0
    dps        : decimal-place precision for both evaluation and PSLQ
    max_degree   : max total monomial degree in the PSLQ basis (1 = Möbius)
    workers      : parallel worker processes (default: os.cpu_count())
    show_trivial      : include b0-additive relations in output (default: False)
    max_rational_denom: skip CFs whose value is p/q with |q| <= this bound;
                        0 disables the filter (default: 10_000_000)
    """
    if workers is None:
        workers = os.cpu_count() or 4

    # ── Phase 1: Enumerate ──────────────────────────────────────────────────
    _bar("PHASE 1  —  ENUMERATE")
    t0 = time.perf_counter()
    forms = enumerate_forms(max_coeff, a_degree, b_degree, b0_min, b0_max)
    print(
        f"  {len(forms):,} candidate forms  "
        f"(max_coeff={max_coeff}, a_deg={a_degree}, b_deg={b_degree}, "
        f"b0=[{b0_min},{b0_max}])  [{time.perf_counter()-t0:.2f}s]"
    )

    if not forms:
        print("  No forms generated — try relaxing the filters.")
        return

    # ── Phase 2: Evaluate in parallel ───────────────────────────────────────
    _bar("PHASE 2  —  EVALUATE")
    print(f"  {len(forms):,} forms  ·  dps={dps}  ·  {workers} workers")
    t0 = time.perf_counter()

    form_vals: dict[PCFForm, mpmath.mpf] = {}
    n_failed = 0
    tick = max(1, len(forms) // 40)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_worker, f, dps): f for f in forms}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                form, val = fut.result()
            except Exception:
                n_failed += 1
                continue
            if val is not None:
                form_vals[form] = val
            else:
                n_failed += 1
            if done % tick == 0 or done == len(forms):
                _progress(done, len(forms),
                          f"  converged={len(form_vals):,}  failed={n_failed:,}")

    print(f"\n  elapsed: {time.perf_counter()-t0:.1f}s")

    if not form_vals:
        print("  Nothing converged. Try smaller --dps or wider --b0-max.")
        return

    # ── Phase 3: Cluster by value ────────────────────────────────────────────
    _bar("PHASE 3  —  CLUSTER")
    key_digits = max(8, dps // 3)
    groups = _cluster(form_vals, key_digits)
    n_equiv = sum(1 for g in groups if len(g) > 1)

    print(f"  {len(groups):,} distinct values  ·  "
          f"{sum(len(g) for g in groups):,} converged forms  ·  "
          f"{n_equiv} equivalence cluster(s)")

    if n_equiv:
        print()
        for g in sorted((g for g in groups if len(g) > 1), key=lambda g: -len(g)):
            _, rep_val = g[0]
            print(f"  ≡  {mpmath.nstr(rep_val, 12)}  ({len(g)} forms):")
            for form, _ in g[:5]:
                print(f"       {_pcf_str(form)}")
            if len(g) > 5:
                print(f"       … and {len(g)-5} more")

    # ── Phase 4: PSLQ search in parallel ────────────────────────────────────
    # One representative (the first in each group) per distinct value.
    reps: list[tuple[PCFForm, mpmath.mpf]] = [g[0] for g in groups]

    if max_rational_denom > 0:
        reps_filtered = [
            (f, v) for f, v in reps
            if not _is_simple_rational(v, dps, max_rational_denom)
        ]
        n_rational = len(reps) - len(reps_filtered)
        if n_rational:
            print(f"\n  Skipping {n_rational} simple-rational value(s) "
                  f"(p/q with |q| ≤ {max_rational_denom:,}) from PSLQ pairs.")
        reps = reps_filtered

    n_pairs = len(reps) * (len(reps) - 1) // 2

    _bar("PHASE 4  —  PSLQ SEARCH")
    if n_pairs == 0:
        print("  Need ≥ 2 distinct values — increase bounds.")
        return

    print(f"  {len(reps):,} representatives  ·  {n_pairs:,} pairs"
          f"  ·  degree≤{max_degree}  ·  {workers} workers")
    t0 = time.perf_counter()

    found: list[tuple[PCFForm, mpmath.mpf, PCFForm, mpmath.mpf, dict]] = []
    tick = max(1, n_pairs // 40)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        pair_futs: dict = {}
        for (f1, v1), (f2, v2) in combinations(reps, 2):
            fut = pool.submit(_pslq_worker, v1, v2, dps, max_degree)
            pair_futs[fut] = (f1, v1, f2, v2)

        done = 0
        for fut in as_completed(pair_futs):
            f1, v1, f2, v2 = pair_futs[fut]
            done += 1
            try:
                result = fut.result()
            except Exception:
                result = None
            if result is not None:
                found.append((f1, v1, f2, v2, result))
            if done % tick == 0 or done == n_pairs:
                _progress(done, n_pairs, f"  relations found: {len(found)}")

    print(f"\n  elapsed: {time.perf_counter()-t0:.1f}s")

    # ── Results ──────────────────────────────────────────────────────────────
    nontrivial = [
        (f1, v1, f2, v2, res)
        for f1, v1, f2, v2, res in found
        if not _is_b0_additive(f1, f2, res)
    ]
    trivial_count = len(found) - len(nontrivial)
    display_list = found if show_trivial else nontrivial

    _bar(f"RESULTS  —  {len(nontrivial)} NON-TRIVIAL  +  {trivial_count} TRIVIAL (b0-additive)")

    if trivial_count and not show_trivial:
        print(f"  (hiding {trivial_count} b0-additive relation(s); use --show-trivial to see them)")

    if not display_list:
        if not found:
            print("  No algebraic relations in this search space.")
            print("  Suggestions: --max-coeff 3  --a-degree 2  --max-degree 2")
        else:
            print("  All relations were trivial b0-additive shifts.")
        return

    for i, (f1, v1, f2, v2, result) in enumerate(display_list, 1):
        vec = result["vector"]
        display = format_polynomial(result).replace("v1", "CF1").replace("v2", "CF2")
        mob = format_mobius(result, "CF1", "CF2") if len(vec) == 4 else None
        trivial_tag = "  [b0-additive]" if _is_b0_additive(f1, f2, result) else ""

        print(f"\n  [{i}]{trivial_tag}")
        print(f"      CF1: {_pcf_str(f1)}  ≈ {mpmath.nstr(v1, 12)}")
        print(f"      CF2: {_pcf_str(f2)}  ≈ {mpmath.nstr(v2, 12)}")
        print(f"      Relation:  {display}")
        if mob:
            print(f"      Möbius:    {mob}")
        print(f"      Residual:  {float(result['residual']):.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Parallel PCF relation search: enumerate → evaluate → PSLQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max-coeff", type=int, default=2,
                   help="Absolute bound on each polynomial coefficient")
    p.add_argument("--a-degree", type=int, default=1,
                   help="Polynomial degree of the CF numerator a(n)")
    p.add_argument("--b-degree", type=int, default=1,
                   help="Polynomial degree of the CF denominator b(n)")
    p.add_argument("--b0-min", type=int, default=0,
                   help="Minimum value of the leading constant b0")
    p.add_argument("--b0-max", type=int, default=3,
                   help="Maximum value of the leading constant b0")
    p.add_argument("--dps", type=int, default=50,
                   help="Decimal places of precision")
    p.add_argument("--max-degree", type=int, default=1,
                   help="Max total monomial degree in PSLQ basis (1=Möbius)")
    p.add_argument("--workers", type=int, default=None,
                   help="Worker processes (default: CPU count)")
    p.add_argument("--show-trivial", action="store_true", default=False,
                   help="Also print b0-additive (trivially shifted) relations")
    p.add_argument("--max-rational-denom", type=int, default=10_000_000,
                   help="Skip CFs whose value is p/q with |q| <= N; 0 to disable")
    args = p.parse_args()

    run_search(
        max_coeff=args.max_coeff,
        a_degree=args.a_degree,
        b_degree=args.b_degree,
        b0_min=args.b0_min,
        b0_max=args.b0_max,
        dps=args.dps,
        max_degree=args.max_degree,
        workers=args.workers,
        show_trivial=args.show_trivial,
        max_rational_denom=args.max_rational_denom,
    )


if __name__ == "__main__":
    main()
