"""One decode per source, disk-backed float32 audio shared by both engines."""
import json
import os
from pathlib import Path
from threading import Lock
import numpy as np
from audio_chunks import audio_chunks, SAMPLE_RATE

_guard = Lock()
_locks = {}


class AudioCache:
    def __init__(self, source):
        self.source = Path(source)
        self.path = self.source.parent / "audio.f32"
        self.meta = self.source.parent / "audio-cache.json"

    def ensure(self, checkpoint=lambda: None):
        with _guard:
            lock = _locks.setdefault(str(self.source.resolve()), Lock())
        while not lock.acquire(timeout=0.2):
            checkpoint()
        try:
            stat = self.source.stat()
            signature = [stat.st_size, stat.st_mtime_ns, SAMPLE_RATE, 1]
            try:
                meta = json.loads(self.meta.read_text(encoding="utf-8"))
                if meta["source"] == signature and self.path.stat().st_size == meta["samples"] * 4:
                    return meta["samples"] / SAMPLE_RATE
            except (OSError, ValueError, KeyError):
                pass
            samples = 0
            temporary = self.path.with_suffix(".tmp")
            with temporary.open("wb") as output:
                chunks = audio_chunks(self.source, chunk_seconds=10)
                try:
                    for _, audio in chunks:
                        checkpoint()
                        output.write(np.asarray(audio, dtype="<f4").tobytes())
                        samples += len(audio)
                finally:
                    chunks.close()
                output.flush()
                os.fsync(output.fileno())
            if not samples:
                raise ValueError("音軌沒有可解碼的樣本")
            temporary.replace(self.path)
            meta_tmp = self.meta.with_suffix(".tmp")
            meta_tmp.write_text(json.dumps({"source": signature, "samples": samples}), encoding="utf-8")
            meta_tmp.replace(self.meta)
            return samples / SAMPLE_RATE
        finally:
            lock.release()

    def read(self, start, end):
        lo = max(0, round(start * SAMPLE_RATE))
        hi = min(self.path.stat().st_size // 4, round(end * SAMPLE_RATE))
        if hi <= lo:
            return np.empty(0, dtype=np.float32)
        # Copy only the requested range; close the mapping immediately on Windows.
        mapping = np.memmap(self.path, dtype="<f4", mode="r", offset=lo * 4, shape=(hi - lo,))
        try:
            return np.array(mapping, dtype=np.float32)
        finally:
            mapping._mmap.close()

    def windows(self, duration, start=0, seconds=120, padding=1):
        core = start
        while core < duration:
            end = min(duration, core + seconds)
            left, right = max(0, core - padding), min(duration, end + padding)
            yield core, end, left, self.read(left, right)
            core = end
