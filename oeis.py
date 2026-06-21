"""
OEIS sequence loader and pre-filter.

Reads the stripped OEIS data file (stripped.gz format, decompressed) and
returns sequences that pass the requested filters.  One sequential pass
through the ~80 MB file is fast enough that no caching layer is needed.

Typical usage
-------------
    import oeis
    seqs_for_a = oeis.load(min_length=100, for_a=True)
    seqs_for_b = oeis.load(min_length=100, for_b=True)
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "oeis.txt"


def load(
    path: str | Path = DEFAULT_PATH,
    *,
    min_length: int = 100,
    for_b: bool = False,
    for_a: bool = False,
    max_abs_term: int | None = None,
    max_seqs: int | None = None,
) -> dict[str, tuple[int, ...]]:
    """
    Load and filter OEIS sequences from the stripped data file.

    Parameters
    ----------
    path          : path to the stripped OEIS file (one sequence per line)
    min_length    : minimum number of terms required (default 100).
                    The stripped file caps at ~348 terms per sequence;
                    only ~21 k sequences meet the 100-term threshold.
    for_b         : keep only sequences with ALL terms > 0.
                    Required when the sequence drives b(k): the CF denominator
                    recurrence must stay positive.
    for_a         : keep only sequences with ALL terms != 0.
                    Avoids a(k) = 0 collapsing the CF at that depth level.
    max_abs_term  : if set, drop sequences containing any |term| > this value.
                    Useful to exclude sequences that grow so fast they cause
                    convergence problems against slow-growing polynomial b(k).
    max_seqs      : if set, stop after collecting this many qualifying sequences
                    (sequences are processed in file order, i.e. A000001 first).

    Returns
    -------
    dict mapping OEIS id (e.g. 'A000045') to a tuple of integer terms.
    Terms are 0-indexed in the tuple but represent 1-indexed CF inputs:
    terms[k-1] is the value used as a(k) or b(k).
    """
    result: dict[str, tuple[int, ...]] = {}

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(',')
            seq_id = parts[0].strip()
            if not seq_id.startswith('A'):
                continue

            try:
                terms = tuple(int(t) for t in parts[1:] if t.strip())
            except ValueError:
                continue

            if len(terms) < min_length:
                continue
            if for_b and any(t <= 0 for t in terms):
                continue
            if for_a and any(t == 0 for t in terms):
                continue
            if max_abs_term is not None and any(abs(t) > max_abs_term for t in terms):
                continue

            result[seq_id] = terms

            if max_seqs is not None and len(result) >= max_seqs:
                break

    return result


def stats(seqs: dict[str, tuple[int, ...]]) -> None:
    """Print a brief summary of a loaded sequence dict."""
    if not seqs:
        print("  (empty)")
        return
    lengths = [len(v) for v in seqs.values()]
    lengths.sort()
    n = len(lengths)
    print(f"  {n:,} sequences  |  "
          f"min={lengths[0]}  median={lengths[n//2]}  max={lengths[-1]}")
