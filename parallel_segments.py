"""Split on central acoustic silence and recognize two sides concurrently."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np

from audio_chunks import SAMPLE_RATE
from segment_boundaries import owned_segment


def central_silence(audio, sample_rate=SAMPLE_RATE, threshold_db=-35):
    """Return an internal >=0.5s quiet interval nearest the actual chunk midpoint.

    Quiet means every 20ms frame has RMS <= threshold_db dBFS. Only full
    frames count, so a partial trailing frame cannot make a short gap qualify.
    """
    frame_size = max(1, round(sample_rate * .02))
    count = len(audio) // frame_size
    if not count:
        return None
    frames = audio[:count * frame_size].reshape(count, frame_size)
    quiet = np.mean(np.square(frames, dtype=np.float32), axis=1) <= 10 ** (threshold_db / 10)
    edges = np.diff(np.concatenate(([False], quiet, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1) * frame_size
    ends = np.flatnonzero(edges == -1) * frame_size
    candidates = [(int(a), int(b)) for a, b in zip(starts, ends)
                  if b - a >= sample_rate * .5 and a > 0 and b < count * frame_size]
    return min(candidates, key=lambda gap: (abs(sum(gap) / 2 - len(audio) / 2), gap[0])) if candidates else None


def transcribe_pair(audio, gap, engine_factory, options, stop_check):
    """Wait for BOTH workers before yielding, so pausing never leaves one running.

    Each worker gets its own stateful pipeline, sharing only the underlying
    thread-safe Whisper model. Results keep original audio-relative timestamps.
    """
    abort = Event()
    cut = (gap[0] + gap[1]) // 2
    padding = SAMPLE_RATE
    def recognize(start, end, core_start, core_end):
        stop_check()
        segments, _ = engine_factory().transcribe(audio[start:end], **options)
        rows = []
        try:
            for segment in segments:
                stop_check()
                if abort.is_set():
                    return []
                owned = owned_segment(segment, start / SAMPLE_RATE,
                                      core_start / SAMPLE_RATE, core_end / SAMPLE_RATE)
                if owned is not None:
                    rows.append(owned)
            return rows
        except BaseException:
            abort.set()
            raise
        finally:
            segments.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Quiet audio is retained. Both sides get context, each word belongs to one core.
        futures = [pool.submit(recognize, 0, min(len(audio), cut + padding), 0, cut),
                   pool.submit(recognize, max(0, cut - padding), len(audio), cut, len(audio))]
        try:
            results = [future.result() for future in futures]
        finally:
            abort.set()
    stop_check()
    for rows in results:
        yield from rows
