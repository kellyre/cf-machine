"""
OEIS sequence loader, pre-filter, and deduplicator.

Reads the stripped OEIS data file (stripped.gz format, decompressed) and
returns sequences that pass the requested filters.  One sequential pass
through the ~80 MB file is fast enough that no caching layer is needed.

Typical usage
-------------
    import oeis
    seqs = oeis.load(min_length=100, for_a=True, dedup=True)
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
    dedup: bool = True,
    dedup_window: int = 10,
    dedup_min_diff: int = 2,
    dedup_max_shift: int = 2,
) -> dict[str, tuple[int, ...]]:
    """
    Load and filter OEIS sequences from the stripped data file.

    Parameters
    ----------
    path            : path to the stripped OEIS file (one sequence per line)
    min_length      : minimum number of terms required (default 100).
                      The stripped file caps at ~348 terms per sequence;
                      only ~21 k sequences meet the 100-term threshold.
    for_b           : keep only sequences with ALL terms > 0.
                      Required when the sequence drives b(k).
    for_a           : keep only sequences with ALL terms != 0.
                      Avoids a(k) = 0 collapsing the CF at that depth level.
    max_abs_term    : drop sequences with any |term| exceeding this value.
    max_seqs        : stop after this many qualifying sequences (None = all).
    dedup           : deduplicate sequences that are too similar to each other
                      (default True).  See `deduplicate()` for details.
    dedup_window    : number of leading terms used for similarity checks.
    dedup_min_diff  : minimum number of differing terms required in the window.
    dedup_max_shift : shifts by 1..this are also treated as duplicates.

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

    if dedup and result:
        result = deduplicate(
            result,
            window=dedup_window,
            min_diff=dedup_min_diff,
            max_shift=dedup_max_shift,
        )

    return result


def deduplicate(
    seqs: dict[str, tuple[int, ...]],
    *,
    window: int = 10,
    min_diff: int = 2,
    max_shift: int = 2,
) -> dict[str, tuple[int, ...]]:
    """
    Remove sequences whose first `window` terms are too similar to an
    already-accepted sequence.  Processes in dict order; first one wins.

    Two sequences are considered "too similar" if ANY of the following hold:

    1. Fewer than `min_diff` of the first `window` terms differ.
       (min_diff=2 means: at least 2 terms must differ — so sequences that
       agree on 9 or more of the first 10 terms are treated as duplicates.)

    2. One is a term-shift of the other by 1..`max_shift` positions.
       Dropping or prepending a few terms produces a CF whose tail is
       identical, so the values converge to numbers that differ only by
       the tiny contribution of the first few levels — false matches.

    Algorithm
    ---------
    Uses three O(1)-lookup hash sets built incrementally:

    · prefix_set  — exact first-`window` prefixes (catches Hamming 0).
    · gapped_set  — 'window' gapped variants per accepted sequence, where
                    one position is replaced by a None sentinel.  A candidate
                    that shares any gapped variant with an accepted sequence
                    has Hamming distance ≤ 1, so it is rejected when
                    min_diff ≥ 2.  (For min_diff=1 only prefix_set is used.)
    · shifted_set — two entries per accepted sequence per shift k:
                    ('L', k, suffix) blocks candidates that are a k-left-shift
                    of an accepted sequence (candidate[:w-k] == accepted[k:w]).
                    ('R', k, prefix) blocks candidates that are a k-right-shift
                    of an accepted sequence (candidate[k:w] == accepted[:w-k]).

    Total work: O(N × window) — no pairwise comparisons needed.
    """
    accepted: dict[str, tuple[int, ...]] = {}
    prefix_set: set[tuple] = set()
    gapped_set: set[tuple] = set()
    shifted_set: set[tuple] = set()

    for seq_id, terms in seqs.items():
        prefix = terms[:window]
        w = len(prefix)

        # ── 1. Exact match (Hamming 0) ───────────────────────────────────────
        if prefix in prefix_set:
            continue

        # ── 2. Hamming distance < min_diff ──────────────────────────────────
        if min_diff >= 2:
            reject = False
            for i in range(w):
                gapped = prefix[:i] + (None,) + prefix[i + 1:]
                if gapped in gapped_set:
                    reject = True
                    break
            if reject:
                continue

        # ── 3. Shift by 1..max_shift ─────────────────────────────────────────
        if max_shift > 0:
            reject = False
            for k in range(1, min(max_shift + 1, w)):
                # candidate is k-left-shift of accepted: candidate[:w-k] == accepted[k:w]
                if ('L', k, prefix[:w - k]) in shifted_set:
                    reject = True
                    break
                # candidate is k-right-shift of accepted: candidate[k:w] == accepted[:w-k]
                if ('R', k, prefix[k:]) in shifted_set:
                    reject = True
                    break
            if reject:
                continue

        # ── Accept and update indices ────────────────────────────────────────
        accepted[seq_id] = terms
        prefix_set.add(prefix)

        if min_diff >= 2:
            for i in range(w):
                gapped_set.add(prefix[:i] + (None,) + prefix[i + 1:])

        if max_shift > 0:
            for k in range(1, min(max_shift + 1, w)):
                # blocks future: candidate[:w-k] == prefix[k:w]
                shifted_set.add(('L', k, prefix[k:]))
                # blocks future: candidate[k:w] == prefix[:w-k]
                shifted_set.add(('R', k, prefix[:w - k]))

    return accepted


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
