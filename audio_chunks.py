import av
import numpy as np


SAMPLE_RATE = 16000


def audio_chunks(path, chunk_seconds=120):
    """Decode incrementally, yielding (start seconds, mono float32 samples)."""
    capacity = int(chunk_seconds * SAMPLE_RATE)
    if capacity < 1:
        raise ValueError("chunk_seconds must be positive")
    buffer = np.empty(capacity, dtype=np.float32)
    filled = 0
    emitted = 0
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)

        def frames():
            for frame in container.decode(audio=0):
                yield from resampler.resample(frame)
            yield from resampler.resample(None)

        for frame in frames():
            samples = frame.to_ndarray().reshape(-1)
            position = 0
            while position < samples.size:
                count = min(capacity - filled, samples.size - position)
                buffer[filled:filled + count] = samples[position:position + count]
                filled += count
                position += count
                if filled == capacity:
                    yield emitted / SAMPLE_RATE, buffer.copy()
                    emitted += filled
                    filled = 0
        if filled:
            yield emitted / SAMPLE_RATE, buffer[:filled].copy()
