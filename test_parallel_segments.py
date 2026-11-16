import unittest
from threading import Barrier, Lock
from types import SimpleNamespace

import numpy as np

from audio_chunks import SAMPLE_RATE as SR
from parallel_segments import central_silence, transcribe_pair


class ParallelTests(unittest.TestCase):
    def test_nearest_silence_and_minimum_length(self):
        audio = np.full(120 * SR, .2, dtype=np.float32)
        audio[20 * SR:22 * SR] = 0
        audio[59 * SR:60 * SR] = 0
        audio[80 * SR:83 * SR] = 0
        self.assertEqual(central_silence(audio), (59 * SR, 60 * SR))
        audio[59 * SR:60 * SR] = .2
        self.assertEqual(central_silence(audio), (80 * SR, 83 * SR))
        audio[:] = .2
        audio[59 * SR:59 * SR + int(.48 * SR)] = 0
        self.assertIsNone(central_silence(audio))
        audio[59 * SR:59 * SR + SR // 2] = 10 ** (-36 / 20)
        self.assertEqual(central_silence(audio), (59 * SR, 59 * SR + SR // 2))
        audio[59 * SR:59 * SR + SR // 2] = 10 ** (-34 / 20)
        self.assertIsNone(central_silence(audio))

    def test_silent_edges_and_short_final_chunk(self):
        self.assertIsNone(central_silence(np.zeros(120 * SR, dtype=np.float32)))
        audio = np.full(30 * SR, .2, dtype=np.float32)
        audio[:2 * SR] = 0
        audio[-2 * SR:] = 0
        self.assertIsNone(central_silence(audio))
        audio[14 * SR:16 * SR] = 0
        self.assertEqual(central_silence(audio), (14 * SR, 16 * SR))

    def test_concurrent_workers_and_original_timestamps(self):
        barrier = Barrier(2)
        engines = []
        lock = Lock()
        class Engine:
            def transcribe(self, audio, **options):
                def segments():
                    barrier.wait(timeout=3)  # fails if the halves run sequentially
                    yield SimpleNamespace(start=0, end=len(audio) / SR + 1, text=str(audio[0]))
                return segments(), None
        def factory():
            engine = Engine()
            with lock:
                engines.append(engine)
            return engine
        audio = np.ones(10 * SR, dtype=np.float32)
        audio[6 * SR:] = 2
        rows = list(transcribe_pair(audio, (4 * SR, 6 * SR), factory, {}, lambda: None))
        self.assertEqual([(r.start, r.end) for r in rows], [(0, 4), (6, 10)])
        self.assertEqual([r.text for r in rows], ['1.0', '2.0'])
        self.assertEqual(len(engines), 2)
        self.assertIsNot(engines[0], engines[1])

    def test_worker_error_propagates(self):
        class Engine:
            def transcribe(self, audio, **options):
                def segments():
                    raise RuntimeError("recognition failed")
                    yield
                return segments(), None
        with self.assertRaisesRegex(RuntimeError, "recognition failed"):
            list(transcribe_pair(np.ones(10 * SR, dtype=np.float32),
                                 (4 * SR, 6 * SR), Engine, {}, lambda: None))


if __name__ == "__main__":
    unittest.main()
