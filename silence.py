"""
Find gaps between words to cut, and remap timestamps from the original
timeline to the post-cut timeline (needed so captions line up after silence
removal shifts everything).
"""


def find_silence_gaps(words: list, seg_start: float, seg_end: float,
                       min_gap: float = 0.5) -> list:
    """
    Return list of (start, end) tuples to KEEP within [seg_start, seg_end],
    dropping any internal gap longer than min_gap seconds.
    """
    relevant = [w for w in words if w["end"] > seg_start and w["start"] < seg_end]
    if not relevant:
        return [(seg_start, seg_end)]

    relevant.sort(key=lambda w: w["start"])

    keep = []
    cur_start = seg_start
    prev_end = seg_start
    for w in relevant:
        w_start = max(w["start"], seg_start)
        w_end = min(w["end"], seg_end)
        gap = w_start - prev_end
        if gap > min_gap:
            keep.append((cur_start, prev_end))
            cur_start = w_start
        prev_end = max(prev_end, w_end)
    keep.append((cur_start, seg_end))

    return [(s, e) for s, e in keep if e - s > 0.05]


def remap_time(t: float, keep_intervals: list) -> float:
    """Map a timestamp on the original timeline to its position after concatenating keep_intervals."""
    new_t = 0.0
    for s, e in keep_intervals:
        if t <= s:
            return new_t
        if t <= e:
            return new_t + (t - s)
        new_t += (e - s)
    return new_t
