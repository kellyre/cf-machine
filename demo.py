"""
Demo: evaluate the PCF library and search for algebraic relationships.

Run:
    python demo.py [--dps N]

Acceptance criteria:
  1. golden × sqrt5 → 2*phi - sqrt5 - 1 = 0  (PSLQ rediscovers the definition of φ)
  2. golden vs silver → no relation            (different quadratic fields)
  3. golden vs golden_equiv → v1 - v2 = 0     (equivalence transform identity)

Note on brouncker / π:
  Brouncker's formula converges O(1/n) — extracting 50 dps requires either
  Wynn epsilon acceleration with very high intermediate precision (~800 dps)
  or a different CF pairing. It is included in the catalog for reference but
  is not part of the acceptance tests here.
"""

from __future__ import annotations

import argparse
import sys
import time

import mpmath

# Detect gmpy2 / GMP backend before any heavy computation.
try:
    import gmpy2  # noqa: F401
    _backend = "gmpy2 (GMP/MPFR) — fast mode active"
except ImportError:
    _backend = "pure Python (mpmath default) — install gmpy2 for ~5-10× speedup"

from pcf import evaluate
from library import CATALOG
from relations import find_relation, print_relation


def _header(s: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {s}")
    print(f"{'='*60}")


def main(dps: int = 50) -> None:
    print(f"CF-Machine  —  backend: {_backend}")
    print(f"Working precision: {dps} decimal places\n")

    # ------------------------------------------------------------------ evaluate
    _header("EVALUATING LIBRARY")

    values: dict[str, mpmath.mpf] = {}
    for name, (desc, form) in CATALOG.items():
        t0 = time.perf_counter()
        val = evaluate(form, dps)
        elapsed = time.perf_counter() - t0
        values[name] = val
        print(f"  {name:20s}  {mpmath.nstr(val, 15)}   ({elapsed:.3f}s)")
        print(f"    {desc}")

    # ------------------------------------------------------------------ relation search
    _header("PAIRWISE RELATION SEARCH")

    # Pair 1: golden × sqrt5 — PSLQ rediscovers 2φ = 1 + √5
    print("Pair: golden  vs  sqrt5  (expect: 2*phi - sqrt5 - 1 = 0)")
    r = find_relation(
        [values["golden"], values["sqrt5"]],
        max_degree=1,
        dps=dps,
    )
    if r:
        print_relation(r, names=["v_φ", "v_√5"], label="golden/sqrt5")
        direct = 2 * values["golden"] - values["sqrt5"] - 1
        print(f"  Direct check:  2*v_φ - v_√5 - 1 = {mpmath.nstr(direct, 6)}")
    else:
        print("  [FAIL] No Möbius relation found — expected 2*phi - sqrt5 - 1 = 0\n")

    # Pair 2: golden vs silver — should find NO relation
    print("Pair: golden  vs  silver  (negative control — expect: no relation)")
    r2 = find_relation(
        [values["golden"], values["silver"]],
        max_degree=2,   # degree 2 to be thorough; cross-variable filter guards false hits
        dps=dps,
    )
    if r2 is None:
        print("  [PASS] No relation found (as expected — different quadratic fields)\n")
    else:
        print(f"  [UNEXPECTED] Relation found: {r2}\n")

    # Pair 3: golden vs golden_equiv — must find a relation (v1 = v2 or Möbius equivalent)
    print("Pair: golden  vs  golden_equiv  (equivalence transform — expect: v1 = v2)")
    r3 = find_relation(
        [values["golden"], values["golden_equiv"]],
        max_degree=1,
        dps=dps,
    )
    if r3:
        print_relation(r3, names=["v_gold", "v_gold_equiv"], label="equiv-transform")
    else:
        print("  [FAIL] No relation found — equivalence transform should give v1 = v2\n")

    # ------------------------------------------------------------------ summary
    _header("SUMMARY")
    all_pass = (
        r is not None
        and r2 is None
        and r3 is not None
    )
    if all_pass:
        print("  ALL ACCEPTANCE TESTS PASSED")
    else:
        print("  SOME TESTS FAILED — see output above")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CF-Machine demo")
    parser.add_argument("--dps", type=int, default=50,
                        help="Decimal places of working precision (default: 50)")
    args = parser.parse_args()
    main(dps=args.dps)
