"""Media decoding and saved-work compatibility checks."""
from io import BytesIO
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from audio_chunks import audio_chunks, SAMPLE_RATE
from transcription_job import TranscriptionJob


class MediaInputTests(unittest.TestCase):
    def test_mp4_audio_decoding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.mp4"
            with av.open(str(path), "w") as output:
                stream = output.add_stream("aac", rate=SAMPLE_RATE)
                stream.layout = "mono"
                samples = np.sin(np.arange(SAMPLE_RATE) * 2 * np.pi * 440 / SAMPLE_RATE)
                frame = av.AudioFrame.from_ndarray(samples.astype("float32")[None, :],
                                                 format="fltp", layout="mono")
                frame.sample_rate = SAMPLE_RATE
                for packet in stream.encode(frame):
                    output.mux(packet)
                for packet in stream.encode(None):
                    output.mux(packet)
            chunks = list(audio_chunks(path, chunk_seconds=0.5))
            self.assertGreaterEqual(sum(len(c) for _, c in chunks), SAMPLE_RATE)
            self.assertTrue(all(c.dtype == np.float32 for _, c in chunks))
            self.assertEqual([start for start, _ in chunks],
                             [i * 0.5 for i in range(len(chunks))])

    def test_video_without_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "silent.mp4"
            with av.open(str(path), "w") as output:
                stream = output.add_stream("mpeg4", rate=1)
                stream.width = stream.height = 16
                stream.pix_fmt = "yuv420p"
                frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
                for packet in stream.encode(None):
                    output.mux(packet)
            with self.assertRaisesRegex(ValueError, "沒有音軌"):
                list(audio_chunks(path))

    def test_saved_mp4_and_legacy_mp3(self):
        with tempfile.TemporaryDirectory() as folder, patch("transcription_job.STORE_ROOT", Path(folder)):
            for extension in ("MP4", "mp3"):
                upload = BytesIO(b"test media")
                upload.name = "sample." + extension
                job = TranscriptionJob(upload, "small", "zh", "快速", "")
                if extension == "mp3":
                    state_path = job.directory / "state.json"
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    del state["source_filename"]
                    state_path.write_text(json.dumps(state), encoding="utf-8")
                restored = TranscriptionJob.restore(job.id)
                self.assertEqual(restored.source.suffix, "." + extension.lower())
                self.assertEqual(restored.source.read_bytes(), b"test media")


if __name__ == "__main__":
    unittest.main()
