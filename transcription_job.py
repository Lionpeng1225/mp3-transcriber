from functools import lru_cache
import gc
from pathlib import Path
import json
import os
from uuid import uuid4
from threading import Condition, Lock, Thread
from time import perf_counter

from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.utils import download_model
from opencc import OpenCC
from audio_chunks import SAMPLE_RATE, audio_chunks
from parallel_segments import central_silence, transcribe_pair

MODEL_LOCK = Lock()
TERMINAL = {"completed", "stopped", "failed", "interrupted"}
STORE_ROOT = Path(__file__).resolve().parent / "saved_jobs"
JOBS = {}
REGISTRY_LOCK = Lock()
PROGRESS_SAVE_INTERVAL = 5.0


def timestamp(seconds):
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


@lru_cache(maxsize=1)
def cached_model(size):
    models = Path(__file__).resolve().parent / "models"
    local = models / size
    if all((local / name).exists() for name in ("model.bin", "config.json", "tokenizer.json")):
        location = str(local)
    else:
        try:
            location = download_model(size, local_files_only=True)
            if not (Path(location) / "model.bin").exists():
                raise FileNotFoundError("模型快取尚未下載完整")
        except Exception:
            location = download_model(size, output_dir=str(models / size))
    return WhisperModel(location, device="cpu", compute_type="int8", cpu_threads=4, num_workers=2)


CURRENT_MODEL = None


def load_model(size):
    # Called under MODEL_LOCK. Release the previous model before allocating another.
    global CURRENT_MODEL
    if CURRENT_MODEL != size:
        cached_model.cache_clear()
        gc.collect()
        CURRENT_MODEL = size
    return cached_model(size)


class Stopped(Exception):
    pass


class TranscriptionJob:
    def __init__(self, recording, size, language, mode, terms, backend=None, parallel_enabled=False,
                 speaker_timing="manual", speaker_token=""):
        self.id = uuid4().hex
        self.name = Path(recording.name).stem
        self.size, self.language, self.mode, self.terms = size, language, mode, terms
        self.backend = backend or {"engine": "faster-whisper"}
        self.parallel_enabled = bool(parallel_enabled)
        self.chunk_parallel_enabled = None
        self.speaker_timing = speaker_timing
        self._speaker_token = speaker_token
        self.condition = Condition()
        self.pause_requested = False
        self.stop_requested = False
        self.status = "running"
        self.phase = "等待載入模型"
        self.rows = []
        self.processed = 0
        self.resume_offset = 0
        self.committed_rows = 0
        self.elapsed_before = 0
        self.elapsed_paused_at = None
        self.paused_seconds = 0
        self.transcription_elapsed_before = 0
        self.transcribing_since = None
        self.preparing_model = False
        self.started = perf_counter()
        self.ended = None
        self.error = ""
        self.last_saved_at = None
        self.speaker_labels = None
        self.speaker_status = "not_started"
        self.speaker_error = ""
        self.directory = STORE_ROOT / self.id
        self.directory.mkdir(parents=True)
        extension = Path(recording.name).suffix.lower()
        if extension not in {".mp3", ".mp4"}:
            extension = ".mp3"
        self.source = self.directory / ("recording" + extension)
        self.source.write_bytes(recording.getbuffer())
        self.save()

    def save(self, force=True):
        with self.condition:
            now = perf_counter()
            if (not force and self.last_saved_at is not None
                    and now - self.last_saved_at < PROGRESS_SAVE_INTERVAL):
                return False
            data = self.snapshot()
            data.update(id=self.id, source_filename=self.source.name,
                        size=self.size, language=self.language, backend=self.backend,
                        mode=self.mode, terms=self.terms,
                        resume_offset=self.resume_offset, committed_rows=self.committed_rows)
            data.update(speaker_labels=self.speaker_labels, speaker_status=self.speaker_status,
                        speaker_error=self.speaker_error, speaker_timing=self.speaker_timing)
            temporary = self.directory / "state.tmp"
            with temporary.open("w", encoding="utf-8") as output:
                json.dump(data, output, ensure_ascii=False)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(self.directory / "state.json")
            self.last_saved_at = perf_counter()
            return True

    @classmethod
    def restore(cls, job_id):
        directory = STORE_ROOT / job_id
        data = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        job = cls.__new__(cls)
        for key in ("name", "size", "language", "mode", "terms", "rows", "processed",
                    "resume_offset", "committed_rows", "error"):
            setattr(job, key, data[key])
        job.id = job_id
        job.backend = data.get("backend", {"engine": "faster-whisper"})
        job.parallel_enabled = data.get("parallel_enabled", False)
        job.chunk_parallel_enabled = None
        job.directory = directory
        source_filename = data.get("source_filename", "recording.mp3")
        if source_filename not in {"recording.mp3", "recording.mp4"}:
            raise ValueError("保存工作中的媒體檔名無效")
        job.source = directory / source_filename
        job.condition = Condition()
        job.pause_requested = job.stop_requested = False
        job.status = data["status"] if data["status"] in TERMINAL else "interrupted"
        job.phase = "已載入自動保存的內容"
        job.last_saved_at = perf_counter()
        job.speaker_labels = data.get("speaker_labels")
        job.speaker_status = data.get("speaker_status", "not_started")
        job.speaker_error = data.get("speaker_error", "")
        job.speaker_timing = data.get("speaker_timing", "manual")
        job._speaker_token = ""
        if job.speaker_status == "running":
            job.speaker_status = "failed"
            job.speaker_error = "程式曾在說話者辨識期間關閉，請重新啟動辨識。"
        job.elapsed_before = data["elapsed"]
        job.elapsed_paused_at = None
        job.paused_seconds = 0
        job.transcription_elapsed_before = data.get("transcription_elapsed", 0)
        job.transcribing_since = None
        job.preparing_model = False
        job.started = job.ended = perf_counter()
        return job

    def restart(self):
        with self.condition:
            if getattr(self, "deleted", False):
                raise ValueError("此工作已刪除")
            if self.status not in {"interrupted", "failed", "stopped"}:
                return
            if not self.source.exists():
                raise FileNotFoundError("保存的錄音檔不存在，無法接續")
            self.elapsed_before = self.snapshot()["elapsed"]
            self.elapsed_paused_at = None
            self.paused_seconds = 0
            self.rows = self.rows[:self.committed_rows]
            self.processed = self.resume_offset
            self.started = perf_counter()
            self.ended = None
            self.pause_requested = self.stop_requested = False
            self.status = "running"
            self.error = ""
            if not self._speaker_token:
                try:
                    from speaker_token_store import load_token
                    self._speaker_token = load_token(Path(__file__).resolve().parent / ".speaker-token.dpapi")
                except OSError:
                    self._speaker_token = ""
            self.phase = "準備從最後完成的分段接續"
            self.start()

    def start(self):
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def pause(self):
        with self.condition:
            if self.status not in TERMINAL and not self.stop_requested:
                self.pause_requested = True
                if self.preparing_model:
                    # Downloads cannot be suspended safely. Pause the transcription
                    # immediately while preparation finishes in the background.
                    if self.elapsed_paused_at is None:
                        self.elapsed_paused_at = perf_counter()
                    self.status = "paused"
                else:
                    self.status = "pausing"
                if self.speaker_timing == "pause":
                    self._launch_speaker_diarization(self._speaker_token or self._load_saved_speaker_token())

    def set_parallel_enabled(self, enabled):
        with self.condition:
            if getattr(self, "deleted", False):
                raise ValueError("此工作已刪除")
            self.parallel_enabled = bool(enabled)
            self.save()

    def set_speaker_timing(self, timing, token=""):
        if timing not in {"manual", "during", "pause", "after"}:
            raise ValueError("無效的說話者辨識時機")
        with self.condition:
            self.speaker_timing = timing
            if token:
                self._speaker_token = token
            self.save()
            if timing == "during" and (
                    self.status not in TERMINAL
                    or (self.status == "completed" and self.speaker_status != "completed")):
                self._launch_speaker_diarization(self._speaker_token or self._load_saved_speaker_token())
            elif timing == "pause" and self.status == "paused":
                self._launch_speaker_diarization(self._speaker_token or self._load_saved_speaker_token())
            elif timing == "after" and self.status == "completed":
                self._launch_speaker_diarization(self._speaker_token or self._load_saved_speaker_token())

    def start_speaker_diarization(self, token):
        with self.condition:
            if self.status != "completed":
                raise ValueError("請先完成轉錄，再辨識說話者")
            self._launch_speaker_diarization(token)

    def _load_saved_speaker_token(self):
        try:
            from speaker_token_store import load_token
            return load_token(Path(__file__).resolve().parent / ".speaker-token.dpapi")
        except OSError:
            return ""

    def _launch_speaker_diarization(self, token):
        with self.condition:
            if self.speaker_status == "running":
                return
            if not token:
                self.speaker_status = "failed"
                self.speaker_error = "尚未保存 Hugging Face 權杖；請在上方權杖設定中記住權杖後重試。"
                self.save()
                return
            self.speaker_status = "running"
            self.speaker_error = ""
            self.save()
            self.speaker_thread = Thread(target=self._run_speaker_diarization,
                                         args=(token,), daemon=True)
            self.speaker_thread.start()

    def _run_speaker_diarization(self, token):
        try:
            from speaker_diarization import diarize_transcript
            def progress(message):
                with self.condition:
                    self.speaker_phase = message
            def transcript_rows():
                with self.condition:
                    while self.status not in TERMINAL:
                        self.condition.wait(timeout=1)
                    return list(self.rows)
            labels = diarize_transcript(self.source, transcript_rows, token, progress)
            with self.condition:
                self.speaker_labels = labels
                self.speaker_status = "completed"
                self.speaker_phase = "說話者辨識完成"
                self.save()
        except Exception as error:
            with self.condition:
                self.speaker_status = "failed"
                self.speaker_error = str(error)
                self.speaker_phase = "說話者辨識失敗"
                self.save()

    def resume(self):
        with self.condition:
            if self.status not in TERMINAL and not self.stop_requested:
                self.pause_requested = False
                if self.elapsed_paused_at is not None:
                    self.paused_seconds += perf_counter() - self.elapsed_paused_at
                    self.elapsed_paused_at = None
                self.status = "running"
                self.condition.notify_all()

    def stop(self):
        with self.condition:
            if self.status not in TERMINAL:
                self.stop_requested = True
                self.status = "stopping"
                self.condition.notify_all()

    def checkpoint(self):
        with self.condition:
            was_transcribing = self.transcribing_since is not None
            while self.pause_requested and not self.stop_requested:
                self.stop_transcription_clock()
                if self.elapsed_paused_at is None:
                    self.elapsed_paused_at = perf_counter()
                self.status = "paused"
                self.save()
                self.condition.wait()
            if self.stop_requested:
                raise Stopped()
            if self.elapsed_paused_at is not None:
                self.paused_seconds += perf_counter() - self.elapsed_paused_at
                self.elapsed_paused_at = None
            if was_transcribing and self.transcribing_since is None:
                self.transcribing_since = perf_counter()

    def stop_transcription_clock(self):
        # Caller holds self.condition.
        if self.transcribing_since is not None:
            self.transcription_elapsed_before += perf_counter() - self.transcribing_since
            self.transcribing_since = None

    def check_stop(self):
        with self.condition:
            if self.stop_requested:
                raise Stopped()

    def snapshot(self):
        with self.condition:
            return dict(name=self.name, status=self.status, phase=self.phase,
                        rows=list(self.rows), processed=self.processed,
                        preparing_model=self.preparing_model,
                        parallel_enabled=self.parallel_enabled,
                        chunk_parallel_enabled=self.chunk_parallel_enabled,
                        speaker_labels=self.speaker_labels,
                        speaker_status=self.speaker_status,
                        speaker_timing=self.speaker_timing,
                        speaker_phase=getattr(self, "speaker_phase", ""),
                        speaker_error=self.speaker_error,
                        elapsed=self.elapsed_before + (
                            self.elapsed_paused_at if self.elapsed_paused_at is not None
                            else self.ended if self.ended is not None else perf_counter()
                        ) - self.started - self.paused_seconds,
                        transcription_elapsed=self.transcription_elapsed_before + (
                            perf_counter() - self.transcribing_since if self.transcribing_since is not None else 0),
                        error=self.error)

    def run(self):
        acquired = False
        outcome = "completed"
        try:
            self.checkpoint()
            while not acquired:
                acquired = MODEL_LOCK.acquire(timeout=0.2)
                self.checkpoint()
            with self.condition:
                self.preparing_model = True
                self.phase = "載入模型（首次使用會下載模型）"
            if self.backend["engine"] == "whisper.cpp":
                from whisper_cpp_engine import WhisperCppEngine
                cached_model.cache_clear()
                gc.collect()
                engine = WhisperCppEngine(self.backend, self.check_stop)
                engine_factory = lambda: WhisperCppEngine(self.backend, self.check_stop)
            else:
                model = load_model(self.size)
                engine = model if self.mode == "原始精細" else BatchedInferencePipeline(model)
                engine_factory = lambda: model if self.mode == "原始精細" else BatchedInferencePipeline(model)
            with self.condition:
                self.preparing_model = False
                self.phase = "模型準備完成，等待開始轉錄"
            self.checkpoint()
            if self.speaker_timing == "during":
                self._launch_speaker_diarization(self._speaker_token or self._load_saved_speaker_token())
            options = {
                "快速": {"beam_size": 5, "batch_size": 2},
                "極速": {"beam_size": 1, "batch_size": 2},
                "原始精細": {"beam_size": 5},
            }[self.mode]
            converter = OpenCC("s2twp")
            chunks = audio_chunks(self.source)
            try:
                for offset, audio in chunks:
                    self.checkpoint()
                    duration = len(audio) / SAMPLE_RATE
                    if offset + duration <= self.resume_offset:
                        continue
                    with self.condition:
                        self.chunk_parallel_enabled = self.parallel_enabled
                        parallel_enabled = self.chunk_parallel_enabled
                        self.phase = f"轉錄 {timestamp(offset)} 起的片段"
                        if self.transcribing_since is None:
                            self.transcribing_since = perf_counter()
                    gap = central_silence(audio) if parallel_enabled else None
                    transcribe_options = dict(language=self.language, vad_filter=True,
                                              hotwords=self.terms.strip() or None, **options)
                    if gap is not None:
                        with self.condition:
                            self.phase = (f"同時轉錄 {timestamp(offset)} 起的前後兩段；"
                                          f"略過靜音 {timestamp(offset + gap[0] / SAMPLE_RATE)}"
                                          f"–{timestamp(offset + gap[1] / SAMPLE_RATE)}")
                        segments = transcribe_pair(audio, gap, engine_factory, transcribe_options, self.check_stop)
                    else:
                        segments, _ = engine.transcribe(audio, **transcribe_options)
                    try:
                        for segment in segments:
                            text = converter.convert(segment.text.strip())
                            start = offset + min(max(segment.start, 0), duration)
                            end = offset + min(max(segment.end, 0), duration)
                            with self.condition:
                                if text and end > start:
                                    self.rows.append((start, end, text))
                                self.processed = max(self.processed, end)
                            self.save(force=False)
                            self.checkpoint()
                    finally:
                        segments.close()
                    with self.condition:
                        self.processed = offset + duration
                        self.resume_offset = self.processed
                        self.committed_rows = len(self.rows)
                    self.save(force=True)
                    del audio
                    self.checkpoint()
            finally:
                chunks.close()
        except Stopped:
            outcome = "stopped"
        except Exception as error:
            outcome = "failed"
            with self.condition:
                self.error = str(error)
        finally:
            if acquired:
                MODEL_LOCK.release()
            with self.condition:
                self.preparing_model = False
                self.stop_transcription_clock()
                self.status = "stopped" if self.stop_requested and outcome == "completed" else outcome
                self.ended = perf_counter()
                self.condition.notify_all()
                try:
                    self.save()
                except OSError as error:
                    self.error += f" 自動保存失敗：{error}"
                if self.status == "completed" and self.speaker_timing == "after":
                    self._launch_speaker_diarization(self._speaker_token or self._load_saved_speaker_token())


def register_job(job):
    with REGISTRY_LOCK:
        JOBS[job.id] = job


def get_job(job_id):
    if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise ValueError("無效的工作編號")
    with REGISTRY_LOCK:
        if job_id not in JOBS:
            JOBS[job_id] = TranscriptionJob.restore(job_id)
        return JOBS[job_id]


def saved_jobs():
    result = []
    for path in STORE_ROOT.glob("*/state.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            result.append((path.parent.name, data["name"], path.stat().st_mtime))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(result, key=lambda item: item[2], reverse=True)


def delete_job(job_id):
    """Delete only a stopped job's own directory, never a linked location."""
    import shutil
    if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise ValueError("無效的工作編號")
    root = STORE_ROOT.resolve()
    directory = STORE_ROOT / job_id
    if directory.resolve().parent != root or directory.is_symlink() or os.path.isjunction(directory):
        raise ValueError("工作資料夾位置不安全，無法刪除")
    with REGISTRY_LOCK:
        job = JOBS.get(job_id)
        lock = job.condition if job is not None else Condition()
        with lock:
            if job is not None and (
                job.status not in TERMINAL
                or (getattr(job, "thread", None) is not None and job.thread.is_alive())
            ):
                raise ValueError("請先停止轉錄，等待工作結束後再刪除；暫停中的工作不能刪除。")
            if directory.exists():
                for path in directory.rglob("*"):
                    if path.is_symlink() or os.path.isjunction(path) or not path.resolve().is_relative_to(root):
                        raise ValueError("工作資料夾包含連結，無法安全刪除")
                shutil.rmtree(directory)
            if job is not None:
                job.deleted = True
            JOBS.pop(job_id, None)



def export_rows(rows):
    return {
        "plain": "\n".join(text for _, _, text in rows),
        "timed": "\n".join(f"[{timestamp(start)}] {text}" for start, _, text in rows),
        "srt": "\n".join(
            f"{index}\n{timestamp(start)} --> {timestamp(end)}\n{text}\n"
            for index, (start, end, text) in enumerate(rows, 1)
        ),
    }
