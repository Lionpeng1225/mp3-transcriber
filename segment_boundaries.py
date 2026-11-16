"""Assign overlap to one core interval, preferably at word boundaries."""
from types import SimpleNamespace


def owned_segment(segment, offset, core_start, core_end):
    words = getattr(segment, "words", None)
    if words:
        selected = [SimpleNamespace(start=offset + w.start, end=offset + w.end,
                                    word=w.word) for w in words
                    if core_start <= offset + (w.start + w.end) / 2 < core_end]
        if not selected:
            return None
        return SimpleNamespace(start=selected[0].start, end=selected[-1].end,
                               text="".join(w.word for w in selected), words=selected)
    start, end = offset + segment.start, offset + segment.end
    if not core_start <= (start + end) / 2 < core_end:
        return None
    return SimpleNamespace(start=max(core_start, start), end=min(core_end, end),
                           text=segment.text, words=[])
