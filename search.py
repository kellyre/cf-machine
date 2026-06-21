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

import oeis as oeis_mod
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


def enumerate_oeis_forms(
    seqs: dict[str, tuple[int, ...]],
    oeis_role: str,
    max_coeff: int,
    poly_degree: int,
    b0_min: int,
    b0_max: int,
) -> list[PCFForm]:
    """
    Generate PCFForms where one slot is an OEIS sequence and the other is a
    small polynomial.

    oeis_role='a' : OEIS drives a(k), polynomial drives b(k)
    oeis_role='b' : polynomial drives a(k), OEIS drives b(k)
                    (seqs must already be filtered for all terms > 0)
    """
    cr = range(-max_coeff, max_coeff + 1)
    forms: list[PCFForm] = []

    if oeis_role == 'a':
        poly_b_options = [
            b for b in product(cr, repeat=poly_degree + 1)
            if _b_stays_positive(b)
        ]
        for seq_id, terms in seqs.items():
            if terms[0] == 0:       # a(1) must be nonzero
                continue
            for b_coeffs in poly_b_options:
                for b0 in range(b0_min, b0_max + 1):
                    forms.append(PCFForm(
                        a_coeffs=(),
                        b_coeffs=b_coeffs,
                        b0=b0,
                        a_seq=terms,
                        a_seq_id=seq_id,
                    ))
    else:  # oeis_role == 'b'
        poly_a_options = [
            a for a in product(cr, repeat=poly_degree + 1)
            if sum(a) != 0 and not all(c == 0 for c in a)
        ]
        for seq_id, terms in seqs.items():
            for a_coeffs in poly_a_options:
                for b0 in range(b0_min, b0_max + 1):
                    forms.append(PCFForm(
                        a_coeffs=a_coeffs,
                        b_coeffs=(),
                        b0=b0,
                        b_seq=terms,
                        b_seq_id=seq_id,
                    ))
    return forms


def _is_oeis_form(form: PCFForm) -> bool:
    return form.a_seq is not None or form.b_seq is not None


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
    a_str = form.a_seq_id if form.a_seq is not None else str(list(form.a_coeffs))
    b_str = form.b_seq_id if form.b_seq is not None else str(list(form.b_coeffs))
    return f"a={a_str}  b={b_str}  b0={form.b0}"


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

_EVAL_BATCH: int = 8_000    # max live evaluation futures per batch
_PSLQ_BATCH: int = 50_000  # max live PSLQ futures per batch


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
    oeis_path: str | None = None,
    oeis_role: str = "a",
    oeis_min_length: int = 100,
    oeis_max_seqs: int | None = None,
    oeis_max_abs_term: int | None = None,
    oeis_dedup: bool = True,
    oeis_dedup_window: int = 10,
    oeis_dedup_min_diff: int = 2,
    oeis_dedup_max_shift: int = 2,
) -> None:
    """
    Run the full enumerate → evaluate → cluster → PSLQ pipeline.

    Parameters
    ----------
    max_coeff           : absolute coefficient bound for polynomial coefficients
    a_degree            : degree of the polynomial a(n)
    b_degree            : degree of the polynomial b(n)
    b0_min / b0_max     : range of the leading constant b0
    dps                 : decimal-place precision for evaluation and PSLQ
    max_degree          : max total monomial degree in PSLQ basis (1=Möbius)
    workers             : parallel worker processes (default: os.cpu_count())
    show_trivial        : include b0-additive relations in output
    max_rational_denom  : skip CFs whose value is p/q with |q| ≤ this; 0=off
    oeis_path           : path to stripped OEIS file; enables OEIS mode
    oeis_role           : 'a' → OEIS drives a(k), polynomial drives b(k)
                          'b' → polynomial drives a(k), OEIS drives b(k)
    oeis_min_length     : minimum sequence length to accept (default 100)
    oeis_max_seqs       : cap on OEIS sequences loaded (None = all qualifying)
    oeis_max_abs_term   : drop OEIS seqs with any |term| exceeding this

    OEIS mode
    ---------
    When oeis_path is given, two form pools are built:
      · Polynomial pool  — standard enumeration (max_coeff / a/b_degree)
      · OEIS pool        — each OEIS sequence paired with every valid
                           polynomial for the opposite slot
    Phase 3 flags any cluster that contains forms from BOTH pools (direct
    value match — the most exciting find).  Phase 4 runs PSLQ cross-pool
    only (OEIS rep × poly rep), keeping the pair count at
    N_oeis_reps × N_poly_reps rather than the full O(N²).
    """
    if workers is None:
        workers = os.cpu_count() or 4

    # ── Phase 1: Enumerate ──────────────────────────────────────────────────
    _bar("PHASE 1  —  ENUMERATE")
    t0 = time.perf_counter()

    poly_forms = enumerate_forms(max_coeff, a_degree, b_degree, b0_min, b0_max)
    print(
        f"  {len(poly_forms):,} polynomial forms  "
        f"(max_coeff={max_coeff}, a_deg={a_degree}, b_deg={b_degree}, "
        f"b0=[{b0_min},{b0_max}])"
    )

    oeis_forms: list[PCFForm] = []
    if oeis_path:
        print(f"  Loading OEIS sequences …", end="", flush=True)
        need_pos = (oeis_role == "b")
        seqs = oeis_mod.load(
            oeis_path,
            min_length=oeis_min_length,
            for_b=need_pos,
            for_a=(not need_pos),
            max_abs_term=oeis_max_abs_term,
            max_seqs=oeis_max_seqs,
            dedup=oeis_dedup,
            dedup_window=oeis_dedup_window,
            dedup_min_diff=oeis_dedup_min_diff,
            dedup_max_shift=oeis_dedup_max_shift,
        )
        print(f" {len(seqs):,} qualifying sequences"
              + (" (after dedup)" if oeis_dedup else ""))
        oeis_mod.stats(seqs)
        poly_deg = b_degree if oeis_role == "a" else a_degree
        oeis_forms = enumerate_oeis_forms(
            seqs, oeis_role, max_coeff, poly_deg, b0_min, b0_max
        )
        print(f"  {len(oeis_forms):,} OEIS-based forms  (role={oeis_role})")

    forms = poly_forms + oeis_forms
    print(f"  {len(forms):,} total  [{time.perf_counter()-t0:.2f}s]")

    if not forms:
        print("  No forms generated — try relaxing the filters.")
        return

    # ── Phase 2: Evaluate in parallel (batched) ─────────────────────────────
    _bar("PHASE 2  —  EVALUATE")
    print(f"  {len(forms):,} forms  ·  dps={dps}  ·  {workers} workers")
    t0 = time.perf_counter()

    form_vals: dict[PCFForm, mpmath.mpf] = {}
    n_failed = 0
    done = 0
    tick = max(1, len(forms) // 40)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch_start in range(0, len(forms), _EVAL_BATCH):
            batch = forms[batch_start : batch_start + _EVAL_BATCH]
            futs = {pool.submit(_eval_worker, f, dps): f for f in batch}
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
            has_oeis = any(_is_oeis_form(f) for f, _ in g)
            has_poly = any(not _is_oeis_form(f) for f, _ in g)
            cross_tag = "  *** OEIS↔poly match ***" if (has_oeis and has_poly) else ""
            print(f"  ≡  {mpmath.nstr(rep_val, 12)}  ({len(g)} forms){cross_tag}:")
            for form, _ in g[:5]:
                print(f"       {_pcf_str(form)}")
            if len(g) > 5:
                print(f"       … and {len(g)-5} more")

    # ── Phase 4: PSLQ search in parallel (batched) ──────────────────────────
    all_reps: list[tuple[PCFForm, mpmath.mpf]] = [g[0] for g in groups]

    if max_rational_denom > 0:
        before = len(all_reps)
        all_reps = [
            (f, v) for f, v in all_reps
            if not _is_simple_rational(v, dps, max_rational_denom)
        ]
        n_rat = before - len(all_reps)
        if n_rat:
            print(f"\n  Skipping {n_rat} simple-rational value(s) "
                  f"(p/q with |q| ≤ {max_rational_denom:,}) from PSLQ pairs.")

    # Cross-pool when OEIS is active; all-pairs otherwise.
    if oeis_path and oeis_forms:
        oeis_reps = [(f, v) for f, v in all_reps if _is_oeis_form(f)]
        poly_reps  = [(f, v) for f, v in all_reps if not _is_oeis_form(f)]
        n_pairs = len(oeis_reps) * len(poly_reps)
        def _pair_iter():
            for a in oeis_reps:
                for b in poly_reps:
                    yield a, b
        pool_label = (f"  ({len(oeis_reps):,} OEIS × {len(poly_reps):,} poly, cross-pool)")
    else:
        n_pairs = len(all_reps) * (len(all_reps) - 1) // 2
        def _pair_iter():
            yield from combinations(all_reps, 2)
        pool_label = ""

    _bar("PHASE 4  —  PSLQ SEARCH")
    if n_pairs == 0:
        print("  Need ≥ 2 distinct non-rational values — increase bounds.")
        return

    print(f"  {n_pairs:,} pairs  ·  degree≤{max_degree}  ·  {workers} workers{pool_label}")
    t0 = time.perf_counter()

    found: list[tuple[PCFForm, mpmath.mpf, PCFForm, mpmath.mpf, dict]] = []
    done = 0
    tick = max(1, n_pairs // 40)
    pair_gen = _pair_iter()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        while True:
            pair_futs: dict = {}
            for (f1, v1), (f2, v2) in pair_gen:
                fut = pool.submit(_pslq_worker, v1, v2, dps, max_degree)
                pair_futs[fut] = (f1, v1, f2, v2)
                if len(pair_futs) >= _PSLQ_BATCH:
                    break
            if not pair_futs:
                break
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
        poly_str = format_polynomial(result).replace("v1", "CF1").replace("v2", "CF2")
        mob = format_mobius(result, "CF1", "CF2") if len(vec) == 4 else None
        trivial_tag = "  [b0-additive]" if _is_b0_additive(f1, f2, result) else ""

        print(f"\n  [{i}]{trivial_tag}")
        print(f"      CF1: {_pcf_str(f1)}  ≈ {mpmath.nstr(v1, 12)}")
        print(f"      CF2: {_pcf_str(f2)}  ≈ {mpmath.nstr(v2, 12)}")
        print(f"      Relation:  {poly_str}")
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
    # OEIS options
    p.add_argument("--oeis", metavar="PATH", default=None,
                   help="Enable OEIS mode: path to stripped OEIS file")
    p.add_argument("--oeis-role", choices=["a", "b"], default="a",
                   help="'a': OEIS drives a(k), poly drives b(k) [default]; "
                        "'b': poly drives a(k), OEIS drives b(k)")
    p.add_argument("--oeis-min-length", type=int, default=100,
                   help="Minimum sequence length to accept from OEIS file")
    p.add_argument("--oeis-max-seqs", type=int, default=None,
                   help="Cap on OEIS sequences loaded (default: all qualifying)")
    p.add_argument("--oeis-max-abs-term", type=int, default=None,
                   help="Drop OEIS sequences with any |term| exceeding this")
    p.add_argument("--no-oeis-dedup", action="store_true", default=False,
                   help="Disable sequence deduplication (not recommended)")
    p.add_argument("--oeis-dedup-window", type=int, default=10,
                   help="Number of leading terms compared for deduplication")
    p.add_argument("--oeis-dedup-min-diff", type=int, default=2,
                   help="Min differing terms in window to keep both sequences")
    p.add_argument("--oeis-dedup-max-shift", type=int, default=2,
                   help="Term shifts 1..N treated as duplicates")
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
        oeis_path=args.oeis,
        oeis_role=args.oeis_role,
        oeis_min_length=args.oeis_min_length,
        oeis_max_seqs=args.oeis_max_seqs,
        oeis_max_abs_term=args.oeis_max_abs_term,
        oeis_dedup=not args.no_oeis_dedup,
        oeis_dedup_window=args.oeis_dedup_window,
        oeis_dedup_min_diff=args.oeis_dedup_min_diff,
        oeis_dedup_max_shift=args.oeis_dedup_max_shift,
    )


if __name__ == "__main__":
    main()
