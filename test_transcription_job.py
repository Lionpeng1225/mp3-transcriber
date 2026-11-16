import tempfile
from pathlib import Path
import time
import unittest
from io import BytesIO
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import transcription_job as module


class JobTests(unittest.TestCase):
    def setUp(self):
        self.entered = Event()
        self.release = Event()
        self.fail_inference = False
        self.job = None
        self.storage = tempfile.TemporaryDirectory(dir=".")
        self.patches = [
            patch.object(module, "STORE_ROOT", Path(self.storage.name)),
            patch.object(module, "load_model", return_value=self),
            patch.object(module, "BatchedInferencePipeline", return_value=self),
            patch.object(module, "audio_chunks", side_effect=lambda _: iter_chunks()),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        self.release.set()
        if self.job:
            self.job.stop()
            self.job.thread.join(3)
        for item in reversed(self.patches):
            item.stop()
        module.JOBS.clear()
        self.storage.cleanup()

    def transcribe(self, audio, **options):
        self.last_options = options
        def segments():
            self.entered.set()
            if not self.release.wait(3):
                raise RuntimeError("test gate timed out")
            yield SimpleNamespace(start=0, end=1, text="第一句")
            if self.fail_inference:
                raise RuntimeError("test failure")
            yield SimpleNamespace(start=1, end=2, text="第二句")
        return segments(), None

    def start_job(self, parallel_enabled=False):
        recording = BytesIO(b"test")
        recording.name = "test.mp3"
        self.job = module.TranscriptionJob(recording, "small", "zh", "快速", "", parallel_enabled=parallel_enabled)
        self.job.start()
        self.assertTrue(self.entered.wait(3))

    def wait_status(self, status):
        deadline = time.monotonic() + 3
        while self.job.snapshot()["status"] != status and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.job.snapshot()["status"], status)

    def test_pause_download_resume(self):
        self.start_job()
        self.job.pause()
        self.release.set()
        self.wait_status("paused")
        partial = self.job.snapshot()
        self.assertEqual(len(partial["rows"]), 1)
        self.assertEqual(module.export_rows(partial["rows"])["plain"], "第一句")
        time.sleep(0.05)
        self.assertEqual(self.job.snapshot()["rows"], partial["rows"])
        self.assertEqual(self.job.snapshot()["transcription_elapsed"], partial["transcription_elapsed"])
        self.assertEqual(self.job.snapshot()["elapsed"], partial["elapsed"])
        self.assertEqual(self.job.snapshot()["processed"], partial["processed"])
        self.job.resume()
        self.wait_status("completed")
        self.assertEqual(len(self.job.snapshot()["rows"]), 2)
        self.assertGreater(self.job.snapshot()["elapsed"], partial["elapsed"])
        self.assertEqual(len(partial["rows"]), 1)
        self.assertTrue(self.job.source.exists())

    def test_stop_while_paused(self):
        self.start_job()
        self.job.pause()
        self.release.set()
        self.wait_status("paused")
        paused_elapsed = self.job.snapshot()["elapsed"]
        time.sleep(0.05)
        self.job.stop()
        self.wait_status("stopped")
        self.assertEqual(self.job.snapshot()["elapsed"], paused_elapsed)
        self.assertEqual(len(self.job.snapshot()["rows"]), 1)
        self.job.resume()
        self.assertEqual(self.job.snapshot()["status"], "stopped")

    def test_stop_during_inference(self):
        self.start_job()
        self.job.stop()
        self.release.set()
        self.wait_status("stopped")
        self.assertEqual(len(self.job.snapshot()["rows"]), 1)

    def test_live_registry_reconnect(self):
        self.start_job()
        module.register_job(self.job)
        self.assertIs(module.get_job(self.job.id), self.job)
        self.release.set()
        self.wait_status("completed")

    def test_turbo_and_hotwords_persist(self):
        recording = BytesIO(b"test")
        recording.name = "test.mp3"
        self.job = module.TranscriptionJob(recording, "turbo", "zh", "快速", "dopamine 多巴胺")
        self.release.set()
        self.job.start()
        self.wait_status("completed")
        self.job.thread.join(3)
        self.assertEqual(self.last_options["hotwords"], "dopamine 多巴胺")
        self.assertEqual(self.last_options["beam_size"], 5)
        restored = module.TranscriptionJob.restore(self.job.id)
        self.assertEqual(restored.size, "turbo")
        self.assertEqual(restored.terms, "dopamine 多巴胺")

    def test_restore_and_resume_without_duplicates(self):
        self.fail_inference = True
        self.start_job()
        self.release.set()
        self.wait_status("failed")
        self.job.thread.join(3)
        old_id = self.job.id
        restored = module.get_job(old_id)
        self.assertEqual(len(restored.rows), 1)
        self.assertEqual(restored.resume_offset, 0)
        self.fail_inference = False
        self.job = restored
        restored.restart()
        self.wait_status("completed")
        restored.thread.join(3)
        self.assertEqual(len(restored.rows), 2)
        self.assertEqual(restored.resume_offset, 2)
        reloaded = module.TranscriptionJob.restore(old_id)
        self.assertEqual(reloaded.status, "completed")
        self.assertEqual(len(reloaded.rows), 2)

    def test_crashed_running_record_is_recoverable(self):
        recording = BytesIO(b"test")
        recording.name = "test.mp3"
        job = module.TranscriptionJob(recording, "small", "zh", "快速", "")
        restored = module.TranscriptionJob.restore(job.id)
        self.assertEqual(restored.status, "interrupted")
        self.assertTrue(restored.source.exists())

    def test_restart_skips_committed_chunks(self):
        recording = BytesIO(b"test")
        recording.name = "test.mp3"
        self.job = module.TranscriptionJob(recording, "small", "zh", "快速", "")
        self.job.rows = [(0, 1, "舊文字")]
        self.job.committed_rows = 1
        self.job.resume_offset = self.job.processed = 2
        self.job.status = "interrupted"
        self.release.set()
        def two_chunks(_):
            yield 0, np.zeros(32000, dtype=np.float32)
            yield 2, np.zeros(32000, dtype=np.float32)
        with patch.object(module, "audio_chunks", side_effect=two_chunks):
            self.job.restart()
            self.wait_status("completed")
            self.job.thread.join(3)
        self.assertEqual([row[0] for row in self.job.rows], [0, 2, 3])
        self.assertEqual(self.job.processed, 4)

    def test_error_preserves_partial(self):
        self.fail_inference = True
        self.start_job()
        self.release.set()
        self.wait_status("failed")
        self.assertEqual(len(self.job.snapshot()["rows"]), 1)
        self.assertIn("test failure", self.job.snapshot()["error"])
        self.assertTrue(self.job.source.exists())

    def test_parallel_chunk_commits_original_timeline(self):
        recording = BytesIO(b"test")
        recording.name = "test.mp3"
        self.job = module.TranscriptionJob(recording, "small", "zh", "快速", "")
        self.job.set_parallel_enabled(True)
        audio = np.ones(10 * 16000, dtype=np.float32)
        audio[4 * 16000:6 * 16000] = 0
        self.release.set()
        with patch.object(module, "audio_chunks", side_effect=lambda _: ((offset, samples) for offset, samples in [(120, audio)])):
            self.job.start()
            self.job.thread.join(3)
        self.assertEqual(self.job.snapshot()["status"], "completed")
        self.assertEqual([r[0] for r in self.job.rows], [120, 121, 126, 127])
        self.assertEqual(self.job.resume_offset, 130)
        self.assertEqual(self.job.committed_rows, 4)

    def test_parallel_pause_resume_and_stop(self):
        for stop in (False, True):
            with self.subTest(stop=stop):
                self.entered.clear()
                self.release.clear()
                audio = np.ones(10 * 16000, dtype=np.float32)
                audio[4 * 16000:6 * 16000] = 0
                def chunks(_):
                    yield 0, audio
                with patch.object(module, "audio_chunks", side_effect=chunks):
                    self.start_job(parallel_enabled=True)
                    self.job.pause()
                    self.release.set()
                    self.wait_status("paused")
                    frozen = self.job.snapshot()
                    time.sleep(.03)
                    self.assertEqual(self.job.snapshot()["elapsed"], frozen["elapsed"])
                    if stop:
                        self.job.stop()
                        self.wait_status("stopped")
                        self.assertEqual(self.job.resume_offset, 0)
                    else:
                        self.job.resume()
                        self.wait_status("completed")
                        self.assertEqual(self.job.resume_offset, 10)
                        self.assertEqual(len(self.job.rows), 4)
                    self.job.thread.join(3)

    def test_parallel_toggle_applies_next_chunk_and_persists(self):
        audio = np.ones(10 * 16000, dtype=np.float32)
        audio[4 * 16000:6 * 16000] = 0
        def chunks(_):
            yield 0, audio
            yield 10, audio
            self.job.set_parallel_enabled(False)
            yield 20, audio
        with patch.object(module, "audio_chunks", side_effect=chunks), patch.object(
                module, "central_silence", wraps=module.central_silence) as detect:
            self.start_job()
            self.assertFalse(self.job.snapshot()["parallel_enabled"])
            self.job.set_parallel_enabled(True)
            self.assertFalse(self.job.snapshot()["chunk_parallel_enabled"])
            restored = module.TranscriptionJob.restore(self.job.id)
            self.assertTrue(restored.parallel_enabled)
            self.release.set()
            self.job.thread.join(3)
            self.assertEqual(self.job.snapshot()["status"], "completed")
            self.assertEqual(detect.call_count, 1)
        self.assertEqual([r[0] for r in self.job.rows], [0, 1, 10, 11, 16, 17, 20, 21])
        self.assertFalse(module.TranscriptionJob.restore(self.job.id).parallel_enabled)

    def test_delete_completed_job_and_prevent_restart(self):
        self.start_job()
        module.register_job(self.job)
        self.release.set()
        self.job.thread.join(3)
        module.delete_job(self.job.id)
        self.assertFalse(self.job.directory.exists())
        self.assertNotIn(self.job.id, module.JOBS)
        self.assertEqual(module.saved_jobs(), [])
        with self.assertRaises(ValueError):
            self.job.restart()

    def test_delete_rejects_running_and_paused_jobs(self):
        self.start_job()
        module.register_job(self.job)
        with self.assertRaises(ValueError):
            module.delete_job(self.job.id)
        self.job.pause()
        self.release.set()
        self.wait_status("paused")
        with self.assertRaises(ValueError):
            module.delete_job(self.job.id)
        self.assertTrue(self.job.source.exists())

    def test_delete_rejects_invalid_paths_and_allows_unloaded_history(self):
        with self.assertRaises(ValueError):
            module.delete_job("../outside")
        recording = BytesIO(b"test")
        recording.name = "test.mp3"
        job = module.TranscriptionJob(recording, "small", "zh", "快速", "")
        module.delete_job(job.id)
        self.assertFalse(job.directory.exists())

    def test_transcription_clock_excludes_loading_and_freezes_at_completion(self):
        loading = Event()
        loaded = Event()
        def load(_):
            loading.set()
            if not loaded.wait(3):
                raise RuntimeError("loading gate timed out")
            return self
        recording = BytesIO(b"test")
        recording.name = "test.mp3"
        self.job = module.TranscriptionJob(recording, "small", "zh", "快速", "")
        with patch.object(module, "load_model", side_effect=load):
            self.job.start()
            try:
                self.assertTrue(loading.wait(3))
                self.assertEqual(self.job.snapshot()["transcription_elapsed"], 0)
            finally:
                loaded.set()
            self.assertTrue(self.entered.wait(3))
            self.assertIsNotNone(self.job.transcribing_since)
            self.release.set()
            self.job.thread.join(3)
        final = self.job.snapshot()
        self.assertEqual(final["status"], "completed")
        self.assertGreater(final["transcription_elapsed"], 0)
        self.assertEqual(final["transcription_elapsed"], self.job.snapshot()["transcription_elapsed"])
        restored = module.TranscriptionJob.restore(self.job.id)
        self.assertEqual(restored.snapshot()["transcription_elapsed"], final["transcription_elapsed"])

    def test_legacy_saved_job_defaults_transcription_clock_to_zero(self):
        import json
        recording = BytesIO(b"test")
        recording.name = "test.mp3"
        job = module.TranscriptionJob(recording, "small", "zh", "快速", "")
        state_path = job.directory / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.pop("transcription_elapsed")
        state_path.write_text(json.dumps(state), encoding="utf-8")
        restored = module.TranscriptionJob.restore(job.id)
        self.assertEqual(restored.snapshot()["transcription_elapsed"], 0)

    def test_pause_during_model_preparation_resume_before_or_after_ready(self):
        for resume_before_ready in (True, False):
            with self.subTest(resume_before_ready=resume_before_ready):
                loading, loaded = Event(), Event()
                self.entered.clear()
                self.release.clear()
                def load(_):
                    loading.set()
                    if not loaded.wait(3):
                        raise RuntimeError("loading gate timed out")
                    return self
                recording = BytesIO(b"test")
                recording.name = "test.mp3"
                self.job = module.TranscriptionJob(recording, "small", "zh", "快速", "")
                with patch.object(module, "load_model", side_effect=load):
                    self.job.start()
                    try:
                        self.assertTrue(loading.wait(3))
                        self.job.pause()
                        self.wait_status("paused")
                        frozen = self.job.snapshot()
                        time.sleep(0.03)
                        self.assertEqual(self.job.snapshot()["elapsed"], frozen["elapsed"])
                        self.assertEqual(frozen["transcription_elapsed"], 0)
                        if resume_before_ready:
                            self.job.resume()
                            self.assertEqual(self.job.snapshot()["status"], "running")
                            self.assertFalse(self.entered.is_set())
                        loaded.set()
                        if not resume_before_ready:
                            deadline = time.monotonic() + 3
                            while self.job.snapshot()["preparing_model"] and time.monotonic() < deadline:
                                time.sleep(0.01)
                            self.assertFalse(self.job.snapshot()["preparing_model"])
                            self.assertFalse(self.entered.is_set())
                            self.job.resume()
                        self.assertTrue(self.entered.wait(3))
                        self.release.set()
                        self.job.thread.join(3)
                        self.assertEqual(self.job.snapshot()["status"], "completed")
                    finally:
                        loaded.set()
                        self.release.set()
                        self.job.stop()
                        self.job.thread.join(3)


def iter_chunks():
    yield 0, np.zeros(32000, dtype=np.float32)


if __name__ == "__main__":
    unittest.main()
